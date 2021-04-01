from typing import List, Union, Optional, Dict
import time
import io
import logging

from fastapi import FastAPI, Query, Path, Depends, Request, WebSocket
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from pydantic import BaseModel, BaseSettings


import vaex
import vaex.server



logger = logging.getLogger("vaex.server")


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
    shape: List[int] = [128, 128]
    min_x: Optional[Union[float, str]] = None
    max_x: Optional[Union[float, str]] = None
    min_y: Optional[Union[float, str]] = None
    max_y: Optional[Union[float, str]] = None
    filter: str = None
    virtual_columns: Dict[str, str] = None


class HeatmapOutput(BaseModel):
    dataset_id: str
    # expression: str
    # what: str
    # bins: List[int]
    # centers: List[float]
    centers_x: List[float]
    centers_y: List[float]
    # edges: List[float]
    values: Union[List[List[float]], List[List[int]]]  # counts



class Settings(BaseSettings):
    vaex_config_file: str = "vaex-server.json"
    vaex_add_example: bool = True
    vaex_config: dict = None
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'


settings = Settings()
vaex_config_file = settings.vaex_config_file


if vaex.file.exists(vaex_config_file):
    config = vaex.utils.read_json_or_yaml(vaex_config_file)
    datasets = {name: vaex.open(path).dataset for name, path in config["datasets"].items()}
else:
    datasets = {}
if settings.vaex_add_example:
    datasets['example'] = vaex.example().dataset


openapi_tags = [
    {
        "name": "easy",
        "description": "Easy API for common cases",
    }

]

app = FastAPI(
    title="Vaex dataset/dataframe API",
    description="Vaex: Quick data aggregation",
    version=vaex.__version__["vaex-server"],
    openapi_tags=openapi_tags
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

path_dataset = Path(..., title="The name of the dataset", description="The name of the dataset")

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


@app.get("/")
async def root():
    return {"datasets": list(datasets)}


@app.get("/dataset")
async def dataset():
    return list(datasets.keys())

@app.get("/dataset/{dataset_id}")
async def dataset(dataset_id: str = path_dataset):
    return {"dataset_id": dataset_id}


async def _compute_histogram(input: HistogramInput) -> HistogramOutput:
    df = vaex.from_dataset(datasets[input.dataset_id])
    limits = [input.min, input.max]
    limits = df.limits(input.expression, limits, delay=True)
    await df.execute_async()
    limits = await limits

    counts = df.count(binby=input.expression, limits=limits, shape=input.shape, delay=True, selection=input.filter)
    await df.execute_async()
    counts = await counts
    return df, counts, limits


@app.get("/histogram/{dataset_id}/{expression}", response_model=HistogramOutput, tags=["easy"])
async def histogram(input: HistogramInput = Depends(HistogramInput)) -> HistogramOutput:
    df, counts, limits = await _compute_histogram(input)
    centers = df.bin_centers(input.expression, limits, input.shape)
    return HistogramOutput(dataset_id=input.dataset_id, values=counts.tolist(), centers=centers.tolist())


@app.post("/histogram/", response_model=HistogramOutput, tags=["easy"])
async def histogram(input: HistogramInput) -> HistogramOutput:
    df, counts, limits = await _compute_histogram(input)
    centers = df.bin_centers(input.expression, limits, input.shape)
    return HistogramOutput(dataset_id=input.dataset_id, values=counts.tolist(), centers=centers.tolist())


@app.get("/histogram.plot/{dataset_id}/{expression}", response_class=ImageResponse, tags=["easy"])
async def histogram(input: HistogramInput = Depends(HistogramInput)) -> HistogramOutput:
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
    df = vaex.from_dataset(datasets[input.dataset_id])
    limits_x = [input.min_x, input.max_x]
    limits_y = [input.min_y, input.max_y]
    print(limits_x, limits_y)
    limits_x = df.limits(input.expression_x, limits_x, delay=True)
    limits_y = df.limits(input.expression_y, limits_y, delay=True)
    await df.execute_async()
    limits_x = await limits_x
    limits_y = await limits_y
    print(limits_x, limits_y)

    limits = [limits_x, limits_y]
    state = {
        'virtual_columns': input.virtual_columns
    }
    df.state_set(state)
    counts = df.count(binby=[input.expression_x, input.expression_y], limits=limits, shape=input.shape, delay=True, selection=input.filter)
    await df.execute_async()
    counts = await counts
    return df, counts, limits


@app.get("/heatmap/{dataset_id}/{expression_x}/{expression_y}", response_model=HeatmapOutput, tags=["easy"], summary="2d aggragation data")
async def heatmap(input: HeatmapInput = Depends(HeatmapInput)) -> HeatmapOutput:
    df, counts, limits = await _compute_heatmap(input)
    centers_x = df.bin_centers(input.expression_x, limits[0], input.shape[0])
    centers_y = df.bin_centers(input.expression_y, limits[1], input.shape[1])
    return HeatmapOutput(dataset_id=input.dataset_id, values=counts.tolist(), centers_x=centers_x.tolist(), centers_y=centers_y.tolist())

@app.post("/heatmap", response_model=HeatmapOutput, tags=["easy"], summary="2d aggragation data")
async def heatmap(input: HeatmapInput) -> HeatmapOutput:
    df, counts, limits = await _compute_heatmap(input)
    centers_x = df.bin_centers(input.expression_x, limits[0], input.shape[0])
    centers_y = df.bin_centers(input.expression_y, limits[1], input.shape[1])
    return HeatmapOutput(dataset_id=input.dataset_id, values=counts.T.tolist(), centers_x=centers_x.tolist(), centers_y=centers_y.tolist())


@app.get("/heatmap.plot/{dataset_id}/{expression_x}/{expression_y}", response_class=ImageResponse, tags=["easy"], summary="plot of 2d aggragation data")
async def heatmap(input: HeatmapInput = Depends(HeatmapInput), f: str ="identity") -> HeatmapOutput:
    import matplotlib
    import matplotlib.pyplot as plt
    df, counts, limits = await _compute_heatmap(input)
    matplotlib.use('agg', force=True)
    fig = plt.figure()
    print(f)
    df.viz.heatmap(input.expression_x, input.expression_y, limits=limits, shape=input.shape, grid=counts, f=f)
    with io.BytesIO() as f:
        fig.canvas.print_png(f)
        plt.close(fig)
        return ImageResponse(content=f.getvalue())


from vaex.encoding import serialize, deserialize, Encoding
import asyncio
from .tornado_server import exception

TEST_LATENCY = 0


class WebSocketHandler:
    def __init__(self, send, service, token=None, token_trusted=None):
        self.send = send
        self.service = service
        self.token = token
        self.token_trusted = token_trusted
        self.trusted = False
        self._msg_id_to_tasks = {}

    # def check_origin(self, origin):
    #     return True

    # def open(self):
    #     logger.debug("WebSocket opened")

    # @tornado.gen.coroutine
    # def on_message(self, websocket_msg):
    #     # Tornado does not receive messages before the current is finished, this
    #     # avoids this limitation of tornado, so we can send progress/cancel information
    #     self.webserver.ioloop.add_callback(self._on_message, websocket_msg)

    # @tornado.gen.coroutine
    async def handle_message(self, websocket_msg):
        if TEST_LATENCY:
            await asyncio.sleep(TEST_LATENCY)
        msg_id = 'invalid'
        encoding = Encoding()
        try:
            websocket_msg = deserialize(websocket_msg, encoding)
            logger.debug("websocket message: %s", websocket_msg)
            msg_id, msg, auth = websocket_msg['msg_id'], websocket_msg['msg'], websocket_msg['auth']

            token = auth['token']  # are we even allowed to execute?
            token_trusted = auth['token-trusted']  # do we trust arbitrary code execution?
            trusted = token_trusted == self.token_trusted and token_trusted

            if not ((token == self.token) or
                    (self.token_trusted and token_trusted == self.token_trusted)):
                raise ValueError('No token provided, not authorized')

            last_progress = None
            ioloop = asyncio.get_event_loop()

            def progress(f):
                nonlocal last_progress

                async def send_progress():
                    vaex.asyncio.check_patch_tornado()  # during testing asyncio might be patched
                    nonlocal last_progress
                    logger.debug("progress: %r", f)
                    last_progress = f
                    # TODO: create task?
                    return await self.send({'msg_id': msg_id, 'msg': {'progress': f}})
                # emit when it's the first time (None), at least 0.05 sec lasted, or and the end
                # but never send old or same values
                if (last_progress is None or (f - last_progress) > 0.05 or f == 1.0) and (last_progress is None or f > last_progress):
                    ioloop.call_soon_threadsafe(send_progress)
                return True

            command = msg['command']
            if command == 'list':
                result = self.service.list()
                await self.write_json({'msg_id': msg_id, 'msg': {'result': result}})
            elif command == 'versions':
                result = {'vaex.core': vaex.core._version.__version_tuple__, 'vaex.server': vaex.server._version.__version_tuple__}
                await self.write_json({'msg_id': msg_id, 'msg': {'result': result}})
            elif command == 'execute':
                df = self.service[msg['df']].copy()
                df.state_set(msg['state'], use_active_range=True, trusted=trusted)
                tasks = encoding.decode_list('task', msg['tasks'], df=df)
                self._msg_id_to_tasks[msg_id] = tasks  # keep a reference for cancelling
                try:
                    results = await self.service.execute(df, tasks, progress=progress)
                finally:
                    del self._msg_id_to_tasks[msg_id]
                # make sure the final progress value is send, and also old values are not send
                last_progress = 1.0
                await self.write_json({'msg_id': msg_id, 'msg': {'progress': 1.0}})
                encoding = Encoding()
                results = encoding.encode_list('vaex-task-result', results)
                await self.write_json({'msg_id': msg_id, 'msg': {'result': results}}, encoding)
            elif command == 'cancel':
                try:
                    tasks = self._msg_id_to_tasks[msg['cancel_msg_id']]
                except KeyError:
                    pass  # already done, or cancelled
                else:
                    for task in tasks:
                        task.cancel()
            elif command == 'call-dataframe':
                df = self.service[msg['df']].copy()
                df.state_set(msg['state'], use_active_range=True, trusted=trusted)
                # TODO: yield
                if msg['method'] not in vaex.server.dataframe.allowed_method_names:
                    raise NotImplementedError("Method is not rmi invokable")
                results = self.service._rmi(df, msg['method'], msg['args'], msg['kwargs'])
                encoding = Encoding()
                if msg['method'] == "_evaluate_implementation":
                    results = encoding.encode('vaex-evaluate-result', results)
                else:
                    results = encoding.encode('vaex-rmi-result', results)
                await self.write_json({'msg_id': msg_id, 'msg': {'result': results}}, encoding)
            else:
                raise ValueError(f'Unknown command: {command}')

        except Exception as e:
            logger.exception("Exception while handling msg")
            msg = exception(e)
            await self.write_json({'msg_id': msg_id, 'msg': msg})

    async def write_json(self, msg, encoding=None):
        encoding = encoding or Encoding()
        logger.debug("writing json: %r", msg)
        try:
            return await self.send(serialize(msg, encoding))
        except:  # noqa
            logger.exception('Failed to write: %s', msg)

    def on_close(self):
        logger.debug("WebSocket closed")


import vaex.server.service
dfs = {name: vaex.from_dataset(dataset) for name, dataset in datasets.items()}
print(list(dfs))
service_bare = vaex.server.service.Service(dfs)
server_thread_count = 1
threads_per_job = 32
service_threaded = vaex.server.service.AsyncThreadedService(service_bare, server_thread_count, threads_per_job)


@app.websocket("/websocket")
async def websocket_endpoint(websocket: WebSocket):
    _test_latency = 0
    await websocket.accept()
    while True:
        handler = WebSocketHandler(websocket.send_bytes, service_threaded)
        data = await websocket.receive()
        if data['type'] == 'websocket.disconnect':
            return
        print("received", data)
        await handler.handle_message(data['bytes'])
        # print(repr(data)) 
