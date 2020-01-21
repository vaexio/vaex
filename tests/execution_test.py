import pytest
from unittest.mock import MagicMock

def test_signals(df):
    mock_begin = MagicMock()
    mock_progress = MagicMock()
    mock_end = MagicMock()
    len(df)  # ensure we have the filter precomputed
    df.executor.signal_begin.connect(mock_begin)
    df.executor.signal_progress.connect(mock_progress)
    df.executor.signal_end.connect(mock_end)
    df.sum(df.x, delay=True)
    df.sum(df.y, delay=True)
    df.execute()
    mock_begin.assert_called_once()
    mock_progress.assert_called_with(1.0)
    mock_end.assert_called_once()

# import vaex
# import vaex.dask
# import vaex.ray
# import numpy as np


# @pytest.fixture(params=['executor_dask', 'executor_ray'])
# def executor(request, executor_dask, executor_ray):
#     named = dict(executor_dask=executor_dask, executor_ray=executor_ray)
#     return named[request.param]


# @pytest.fixture(scope='session')
# def executor_ray():
#     return vaex.ray.Executor(chunk_size=2)


# @pytest.fixture(scope='session')
# def executor_dask():
#     return vaex.dask.Executor(chunk_size=2)


# @pytest.fixture
# def df():
#     x = np.arange(10)
#     y = x**2
#     df = vaex.from_arrays(x=x, y=y)
#     return df


# def test_task_sum(df, executor):
#     total = df.x.sum()
#     task = vaex.tasks.TaskSum(df, 'x')
#     # df.executor = None
#     # df._expressions = None
#     # executor = vaex.ray.ExecutorRay()
#     executor.schedule(task)
#     executor.execute()
#     assert task.result == total



# def test_sum(df, executor):
#     total = df.x.sum()
#     df.executor = executor
#     total2 = df.x.sum()
#     assert total == total2
