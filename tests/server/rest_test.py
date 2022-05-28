import base64
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from starlette import responses

import pytest
from myst_parser.main import to_tokens
import numpy as np

import vaex
import vaex.server.download


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


def test_download_vaex(request_client, df_example_original, tmpdir):
    response = request_client.get(f"/download/example", params={"limit": 4, "offset": 1, "query": "x > 0"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/vnd.apache.arrow.stream"
    filename = tmpdir / "test.arrow"
    with open(filename, "wb") as f:
        f.write(response.content)
    df = vaex.open(filename)
    dfo = df_example_original
    dff = dfo[df.x > 0][1:5]
    assert df["x"].tolist() == dff["x"].tolist()


@pytest.mark.skipif(not vaex.settings.main.server.sql, reason="sql not enabled")
def test_query_digest(request_client):
    response1 = request_client.get(
        f"/sql/select",
        params={
            "query": "SELECT * from example LIMIT 10",
        },
        headers={"Want-Digest": "sha-256"},
    )
    assert response1.status_code == 200
    response2 = request_client.post(
        f"/sql/select",
        params={
            "query": "SELECT * from example LIMIT 10",
        },
        headers={"Want-Digest": "sha-256"},
    )
    assert response2.status_code == 200
    assert "Digest" not in response1.headers
    assert "Digest" in response2.headers
    import hashlib

    hash_expected = hashlib.sha256(response1.content).digest()
    algo, encoded = response2.headers["Digest"].split("=", 1)
    hash_result = base64.decodebytes(encoded.encode("ascii"))
    assert algo == "sha-256"
    assert hash_expected == hash_result


@pytest.mark.skipif(not vaex.settings.main.server.sql, reason="sql not enabled")
def test_query_sql_resume(request_client):
    response1 = request_client.get(
        f"/sql/select",
        params={
            "query": "SELECT * from example LIMIT 10",
        },
    )
    response2 = request_client.get(
        f"/sql/select",
        params={
            "query": "SELECT * from example LIMIT 10",
        },
        headers={"Range": "bytes=10-"},
    )
    assert len(response2.content) == len(response1.content) - 10
    assert response2.content == response1.content[10:]


@pytest.mark.skipif(not vaex.settings.main.server.sql, reason="sql not enabled")
def test_query_sql_query(request_client, df_example_original, tmpdir):
    # post
    response = request_client.post(
        f"/sql/select",
        params={
            "query": "SELECT * from example LIMIT 10",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/vnd.apache.arrow.stream"

    # get
    response = request_client.get(
        f"/sql/select",
        params={
            "query": "SELECT * from example LIMIT 10",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/vnd.apache.arrow.stream"
    filename = tmpdir / "test.arrow"
    with open(filename, "wb") as f:
        f.write(response.content)
    df = vaex.open(filename)
    assert df["x"].tolist() == df_example_original[:10]["x"].tolist()

    response = request_client.get(
        f"/sql/select",
        params={
            "query": "SELECT * from example LIMIT 10",
            "output": "parquet",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    filename = tmpdir / "test.parquet"
    with open(filename, "wb") as f:
        f.write(response.content)
    df = vaex.open(filename)
    assert df["x"].tolist() == df_example_original[:10]["x"].tolist()

    response = request_client.get(f"/sql/select", params={"query": "SELECT x, y from example LIMIT 10", "output": "json"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    filename = tmpdir / "test.json"
    with open(filename, "wb") as f:
        f.write(response.content)
    df = vaex.from_json(filename)
    np.testing.assert_array_almost_equal(df["x"].tolist(), df_example_original[:10]["x"].tolist(), decimal=3)


@pytest.mark.skipif(not vaex.settings.main.server.sql, reason="sql not enabled")
def test_download(request_client, tmpdir):
    path = vaex.server.download.download("/sql/select", sql_query="select * from example limit 10", directory=tmpdir, client=request_client)
    f = tmpdir / "test1.parquet"
    vaex.server.download.download("/sql/select", sql_query="select * from example limit 10", f=f, client=request_client)
    with open(tmpdir / "test2.parquet", "wb+") as f:
        vaex.server.download.download("/sql/select", sql_query="select * from example limit 10", f=f, client=request_client)


# TODO: we can't use this using threads, need to use asyncio
# def test_parallel():
#     tpe = ThreadPoolExecutor(4)    
#     def request():
#         min, max = 0, 10
#         shape = 5
#         response = request_client.get(f"/histogram/example/x?min={min}&max={max}&shape={shape}")
#         assert response.status_code == 200, f"Unexpected response: {response.text}"
#         return 42
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
