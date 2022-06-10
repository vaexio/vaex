from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import vaex.server.fastapi


@pytest.fixture(scope='session')
def request_client(webserver):
    client = TestClient(vaex.server.fastapi.app, raise_server_exceptions=True)
    return client


vaex.server.fastapi.ensure_example()


def test_list(request_client):
    response = request_client.get("/dataset")
    assert response.status_code == 200
    json = response.json()
    assert 'example' in json


def test_dataset(request_client):
    response = request_client.get("/dataset/example")
    assert response.status_code == 200
    json = response.json()
    assert 'row_count' in json


def test_histogram(request_client, df_example_original):
    df = df_example_original
    # GET
    min, max = 0, 10
    shape = 5
    response = request_client.get(f"/histogram/example/x?min={min}&max={max}&shape={shape}")
    assert response.status_code == 200
    json = response.json()
    values = df.count(binby='x', limits=[min, max], shape=shape)
    centers = df.bin_centers('x', [min, max], shape=shape)
    assert json['centers'] == centers.tolist()
    assert json['values'] == values.tolist()

    # POST
    response = request_client.post(f"/histogram", json=dict(dataset_id='example', expression='x', min=min, max=max, shape=shape))
    assert response.status_code == 200
    json = response.json()
    assert json['centers'] == centers.tolist()
    assert json['values'] == values.tolist()

    response = request_client.get(f"/histogram/doesnotexist/x?min={min}&max={max}&shape={shape}")
    assert response.status_code == 404


def test_histogram_plot(request_client, df_example_original):
    df = df_example_original
    # GET
    min, max = 0, 10
    shape = 5
    response = request_client.get(f"/histogram.plot/example/x?min={min}&max={max}&shape={shape}")
    assert response.status_code == 200


def test_heatmap(request_client, df_example_original):
    df = df_example_original
    # GET
    min_x, max_x = 0, 10
    shape_x = 5
    min_y, max_y = 0, 20
    shape_y = 10
    shape = shape_x, shape_y
    limits = (min_x, max_x), (min_y, max_y)
    response = request_client.get(f"/heatmap/example/x/y?min_x={min_x}&max_x={max_x}&min_y={min_y}&max_y={max_y}&shape_x={shape_x}&shape_y={shape_y}")
    assert response.status_code == 200
    json = response.json()
    values = df.count(binby=['x', 'y'], limits=limits, shape=shape)
    centers_x = df.bin_centers('x', limits[0], shape=shape[0])
    centers_y = df.bin_centers('y', limits[1], shape=shape[1])
    assert json['centers_x'] == centers_x.tolist()
    assert json['centers_y'] == centers_y.tolist()
    assert json['values'] == values.tolist()

    # POST
    response = request_client.post(f"/heatmap", json=dict(
        dataset_id='example',
        expression_x='x',
        expression_y='y',
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        shape_x=shape_x,
        shape_y=shape_y,
    ))
    assert response.status_code == 200, f"Unexpected response: {response.text}"
    json = response.json()
    assert json['centers_x'] == centers_x.tolist()
    assert json['centers_y'] == centers_y.tolist()
    assert json['values'] == values.tolist()




def test_heatmap_plot(request_client, df_example_original):
    df = df_example_original
    # GET
    min_x, max_x = 0, 10
    shape_x = 5
    min_y, max_y = 0, 20
    shape_y = 10
    shape = shape_x, shape_y
    limits = (min_x, max_x), (min_y, max_y)
    response = request_client.get(f"/heatmap.plot/example/x/y?min_x={min_x}&max_x={max_x}&min_y={min_y}&max_y={max_y}&shape_x={shape_x}&shape_y={shape_y}")
    assert response.status_code == 200


# TODO: we can't use this using threads, need to use asyncio
# def test_parallel():
#     tpe = ThreadPoolExecutor(4)
#     def request():
#         min, max = 0, 10
#         shape = 5
#         response = request_client.get(f"/histogram/example/x?min={min}&max={max}&shape={shape}")
#     N = 10
#     futures = []
#     for i in range(N):
#         futures.append(tpe.submit(request))
#     done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)
#     for future in done:
#         assert future.result() == 42


# unstable with JSON parser
# def test_docs():
#     docs = Path(__file__).parent.parent.parent / 'docs/source/server.md'
#     with open(docs) as f:
#         source = f.read()
#     tokens = to_tokens(source)
#     seen_code = False
#     for token in tokens:
#         if token.tag == "code" and token.info == "python":
#             code = token.content
#             for key, value in vaex.server.utils.get_overrides().items():
#                 code = code.replace(key, value)
#             c = compile(code, "<md>", mode="exec")
#             seen_code = True
#             exec(c)
#     assert seen_code
#         # print(token)
#         # print(token.type, token.content)
