import threading
from typing import List, Union, Optional, Dict
import time
import asyncio
import io
import logging
import sys
import os
import pathlib
import contextlib
import json


from fastapi import FastAPI, Query, Path, Depends, Request, WebSocket, APIRouter, HTTPException
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.openapi.docs import get_swagger_ui_html
import requests


from pydantic import BaseModel, BaseSettings
from starlette.responses import HTMLResponse


import vaex
import vaex.server
import vaex.settings
import vaex.server.websocket

global_lock = asyncio.Lock()
logger = logging.getLogger("vaex.server")
VAEX_FAVICON = 'https://vaex.io/img/logos/vaex_alt.png'
HERE = pathlib.Path(__file__).parent
use_graphql = vaex.utils.get_env_type(bool, 'VAEX_SERVER_GRAPHQL', False)


class ImageResponse(Response):
    media_type = "image/png"


class HistogramInput(BaseModel):
    dataset_id: str
    expression: str
    shape: int = 128
    min: Optional[Union[float, str]] = None
    max: Optional[Union[float, str]] = None
    filter: str = None
    virtual_columns: Dict[str, str] = None


class HistogramOutput(BaseModel):
    dataset_id: str
    # expression: str
    # what: str
    # bins: List[int]
    centers: List[float]
    # edges: List[float]
    values: Union[List[float], List[int]]  # counts


class HeatmapInput(BaseModel):
    dataset_id: str
    expression_x: str
    expression_y: str
    shape_x: int = 128
    shape_y: int = 128
    min_x: Optional[Union[float, int, str]] = None
    max_x: Optional[Union[float, int, str]] = None
    min_y: Optional[Union[float, int, str]] = None
    max_y: Optional[Union[float, int, str]] = None
    filter: str = None
    virtual_columns: Dict[str, str] = None


class HeatmapOutput(BaseModel):
    dataset_id: str
    expression_x: str
    expression_y: str
    # expression: str
    # what: str
    # bins: List[int]
    # centers: List[float]
    centers_x: List[float]
    centers_y: List[float]
    # edges: List[float]
    values: Union[List[List[float]], List[List[int]]]  # counts


datasets = {}

openapi_tags = [
    {
        "name": "quick",
        "description": "Quick API for common cases",
    }

]

router = APIRouter()
path_dataset = Path(..., title="The name of the dataset", description="The name of the dataset")


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    with (HERE / 'index.html').open() as f:
        content = f.read()
    data = []
    for name, ds in datasets.items():
        data.append({
            'name': name,
            'rows': ds.row_count,
            'column': list(ds),
            'schema': [{'name': k, 'type': str(vaex.dtype(v))} for k, v in ds.schema().items()]
        })
    content = content.replace('// DATA', 'app.$data.datasets = %s\n app.$data.graphql = %s' % (json.dumps(data), json.dumps(use_graphql)))
    return content


@router.get("/dataset", summary="Lists all dataset names")
async def dataset():
    return list(datasets.keys())


@router.get("/dataset/{dataset_id}", summary="Meta information about a dataset (schema etc)")
async def dataset(dataset_id: str = path_dataset):
    async with get_df(dataset_id) as df:
        schema = {k: str(v) for k, v in df.schema().items()}
        return {"id": dataset_id, "row_count": len(df), "schema": schema}

@contextlib.asynccontextmanager
async def get_df(name):
    if name not in datasets:
        raise HTTPException(status_code=404, detail=f"dataset {name!r} not found")
    # for now we only allow 1 request to execute at a time
    async with global_lock:
        yield vaex.from_dataset(datasets[name])


async def _compute_histogram(input: HistogramInput) -> HistogramOutput:
    async with get_df(input.dataset_id) as df:
        limits = [input.min, input.max]
        limits = df.limits(input.expression, limits, delay=True)
        await df.execute_async()
        limits = await limits

        counts = df.count(binby=input.expression, limits=limits, shape=input.shape, delay=True, selection=input.filter)
        await df.execute_async()
        counts = await counts
        return df, counts, limits


@router.get("/histogram/{dataset_id}/{expression}", response_model=HistogramOutput, tags=["quick"], summary="histogram data (1d)")
async def histogram(input: HistogramInput = Depends(HistogramInput)) -> HistogramOutput:
    df, counts, limits = await _compute_histogram(input)
    centers = df.bin_centers(input.expression, limits, input.shape)
    return HistogramOutput(dataset_id=input.dataset_id, values=counts.tolist(), centers=centers.tolist())


@router.post("/histogram", response_model=HistogramOutput, tags=["quick"], summary="histogram data (1d)")
async def histogram(input: HistogramInput) -> HistogramOutput:
    df, counts, limits = await _compute_histogram(input)
    centers = df.bin_centers(input.expression, limits, input.shape)
    return HistogramOutput(dataset_id=input.dataset_id,
                           expression=input.expression,
                           values=counts.tolist(),
                           centers=centers.tolist())


@router.get("/histogram.plot/{dataset_id}/{expression}", response_class=ImageResponse, tags=["quick"], summary="Quick histogram plot")
async def histogram_plot(input: HistogramInput = Depends(HistogramInput)) -> HistogramOutput:
    import matplotlib
    import matplotlib.pyplot as plt
    df, counts, limits = await _compute_histogram(input)
    matplotlib.use('agg', force=True)
    fig = plt.figure()
    df.viz.histogram(input.expression, limits=limits, shape=input.shape, grid=counts)
    with io.BytesIO() as f:
        fig.canvas.print_png(f)
        plt.close(fig)
        return ImageResponse(content=f.getvalue())


async def _compute_heatmap(input: HeatmapInput) -> HeatmapOutput:
    async with get_df(input.dataset_id) as df:
        limits_x = [input.min_x, input.max_x]
        limits_y = [input.min_y, input.max_y]
        limits_x = df.limits(input.expression_x, limits_x, delay=True)
        limits_y = df.limits(input.expression_y, limits_y, delay=True)
        await df.execute_async()
        limits_x = await limits_x
        limits_y = await limits_y

        limits = [limits_x, limits_y]
        state = {
            'virtual_columns': input.virtual_columns or {}
        }
        df.state_set(state)
        shape = [input.shape_x, input.shape_y]
        counts = df.count(binby=[input.expression_x, input.expression_y], limits=limits, shape=shape, delay=True, selection=input.filter)
        await df.execute_async()
        counts = await counts
        return df, counts, limits


@router.get("/heatmap/{dataset_id}/{expression_x}/{expression_y}", response_model=HeatmapOutput, tags=["quick"], summary="heatmap data (2d)")
async def heatmap(input: HeatmapInput = Depends(HeatmapInput)) -> HeatmapOutput:
    df, counts, limits = await _compute_heatmap(input)
    centers_x = df.bin_centers(input.expression_x, limits[0], input.shape_x)
    centers_y = df.bin_centers(input.expression_y, limits[1], input.shape_y)
    return HeatmapOutput(dataset_id=input.dataset_id,
                         expression_x=input.expression_x,
                         expression_y=input.expression_y,
                         values=counts.tolist(),
                         centers_x=centers_x.tolist(),
                         centers_y=centers_y.tolist())


@router.post("/heatmap", response_model=HeatmapOutput, tags=["quick"], summary="heatmap data (2d)")
async def heatmap(input: HeatmapInput) -> HeatmapOutput:
    df, counts, limits = await _compute_heatmap(input)
    centers_x = df.bin_centers(input.expression_x, limits[0], input.shape_x)
    centers_y = df.bin_centers(input.expression_y, limits[1], input.shape_y)
    return HeatmapOutput(dataset_id=input.dataset_id,
                         expression_x=input.expression_x,
                         expression_y=input.expression_y,
                         values=counts.tolist(),
                         centers_x=centers_x.tolist(),
                         centers_y=centers_y.tolist())


@router.get("/heatmap.plot/{dataset_id}/{expression_x}/{expression_y}", response_class=ImageResponse, tags=["quick"], summary="Quick heatmap plot")
async def heatmap_plot(input: HeatmapInput = Depends(HeatmapInput), f: str ="identity") -> HeatmapOutput:
    import matplotlib
    import matplotlib.pyplot as plt
    df, counts, limits = await _compute_heatmap(input)
    matplotlib.use('agg', force=True)
    fig = plt.figure()
    df.viz.heatmap(input.expression_x, input.expression_y, limits=limits, shape=[input.shape_x, input.shape_y], grid=counts, f=f)
    with io.BytesIO() as f:
        fig.canvas.print_png(f)
        plt.close(fig)
        return ImageResponse(content=f.getvalue())


@router.websocket("/websocket")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    handler = vaex.server.websocket.WebSocketHandler(websocket.send_bytes, service_threaded)
    while True:
        data = await websocket.receive()
        if data['type'] == 'websocket.disconnect':
            return
        asyncio.create_task(handler.handle_message(data['bytes']))


app = FastAPI(
    title="Vaex dataset/dataframe API",
    description="Vaex: Fast data aggregation",
    version=vaex.__version__["vaex-server"],
    openapi_tags=openapi_tags,
    docs_url=None,
)

app.include_router(router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        swagger_favicon_url=VAEX_FAVICON,
    )

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    executor = vaex.dataframe.get_main_executor()
    start_passes = executor.passes
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Data-Passes"] = str(executor.passes - start_passes)
    return response



class Settings(BaseSettings):
    vaex_config_file: str = "vaex-server.json"
    vaex_add_example: bool = True
    vaex_config: dict = None
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'



# used for testing
class Server(threading.Thread):
    def __init__(self, port, host='localhost', **kwargs):
        self.port = port
        self.host = host
        self.kwargs = kwargs
        self.started = threading.Event()
        self.stopped = threading.Event()
        super().__init__(name="fastapi-thread")
        self.setDaemon(True)

    def set_datasets(self, dfs):
        global datasets
        dfs = {df.name: df for df in dfs}
        update_service(dfs)

    def run(self):
        self.mainloop()

    def serve_threaded(self):
        logger.debug("start thread")
        self.start()
        logger.debug("wait for thread to run")
        self.started.wait()
        logger.debug("make tornado io loop the main thread's current")

    def wait_until_serving(self):
        for n in range(10):
            url = f'http://{self.host}:{self.port}/'
            try:
                response = requests.get(url)
            except requests.exceptions.ConnectionError:
                pass
            else:
                if response.status_code == 200:
                    return
            time.sleep(0.05)
        else:
            raise RuntimeError(f'Server at {url} does not seem to be running')

    def mainloop(self):
        logger.info("serving at http://%s:%d" % (self.host, self.port))

        from uvicorn.config import Config
        from uvicorn.server import Server

        # uvloop will trigger a: RuntimeError: There is no current event loop in thread 'fastapi-thread'
        config = Config(app, host=self.host, port=self.port, **self.kwargs, loop='asyncio')
        self.server = Server(config=config)
        self.started.set()
        try:
            self.server.run()
        except:
            logger.exception("Oops, server stopped unexpectedly")
        finally:
            self.stopped.set()

    def stop_serving(self):
        logger.debug("stopping server")
        self.server.should_exit = True
        if self.stopped.wait(1) is not None:
            logger.error('stopping server failed')
        logger.debug("stopped server")


for name, path in vaex.settings.webserver.get("datasets", {}).items():
    datasets[name] = vaex.open(path).dataset


def add_graphql():
    import vaex.graphql
    import graphene
    from starlette.graphql import GraphQLApp
    dfs = {name: vaex.from_dataset(ds) for name, ds in datasets.items()}
    Query = vaex.graphql.create_query(dfs)
    schema = graphene.Schema(query=Query)
    app.add_route("/graphql", GraphQLApp(schema=schema))


def ensure_example():
    if 'example' not in datasets:
        datasets['example'] = vaex.example().dataset

ensure_example()


def update_service(dfs=None):
    global service_threaded
    import vaex.server.service
    if dfs is None:
        dfs = {name: vaex.from_dataset(dataset) for name, dataset in datasets.items()}

    service_bare = vaex.server.service.Service(dfs)
    server_thread_count = 1
    threads_per_job = 32
    service_threaded = vaex.server.service.AsyncThreadedService(service_bare, server_thread_count, threads_per_job)


def main(argv=sys.argv):
    global use_graphql
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser(argv[0])
    parser.add_argument("filename", help="filename for dataset", nargs='*')
    parser.add_argument('--add-example', default=False, action='store_true', help="add the example dataset")
    parser.add_argument("--host", help="address to bind the server to (default: %(default)s)", default="0.0.0.0")
    parser.add_argument("--base-url", help="External base url (default is <host>:port)", default=None)
    parser.add_argument("--port", help="port to listen on (default: %(default)s)", type=int, default=8081)
    parser.add_argument('--verbose', '-v', action='count', help='show more info', default=2)
    parser.add_argument('--quiet', '-q', action='count', help="less info", default=0)
    parser.add_argument('--graphql', default=use_graphql, action='store_true', help="Add graphql endpoint")
    config = parser.parse_args(argv[1:])

    verbosity = ["ERROR", "WARNING", "INFO", "DEBUG"]
    logging.getLogger("vaex").setLevel(verbosity[config.verbose - config.quiet])

    if config.filename:
        datasets.clear()
        for path in config.filename:
            if "=" in path:
                name, path = path.split('=')
                df = vaex.open(path)
                datasets[name] = df.dataset
            else:
                df = vaex.open(path)
                name, _, _ = vaex.file.split_ext(os.path.basename(path))
                datasets[name] = df.dataset
    if not datasets:
        datasets['example'] = vaex.example().dataset
    if config.add_example:
        ensure_example()
    use_graphql = config.graphql
    if use_graphql:
        add_graphql()
    update_service()
    host = config.host
    port = config.port
    base_url = config.base_url
    if not base_url:
        base_url = host
        if port != 80:
            base_url += f":{port}"
    for name in datasets:
        line = f"{name}:  http://{base_url}/dataset/{name} for REST or ws://{base_url}/{name} for websocket"
        logger.info(line)

    uvicorn.run(app, port=port, host=host)


if __name__ == "__main__":
    main()
else:
    update_service()
    if use_graphql:
        add_graphql()
