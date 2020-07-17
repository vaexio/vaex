from unittest.mock import MagicMock
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import platform

import pytest

from common import small_buffer
import vaex


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


def test_reentrant_catch(df_local):
    df = df_local

    # a 'worker' thread should not be allowed to trigger a new computation
    def progress(fraction):
        print('progress', fraction)
        df.count(df.x)  # enters the executor again

    with pytest.raises(RuntimeError) as exc:
        df.count(df.x, progress=progress)
    assert 'nested' in str(exc.value)


@pytest.mark.skipif(platform.system().lower() == 'windows', reason="hangs appveyor very often, bug?")
def test_thread_safe(df_local):
    df = df_local

    # but an executor should be thread save
    def do():
        return df_local.count(df.x)  # enters the executor from a thread

    count = df_local.count(df.x)
    tpe = ThreadPoolExecutor(4)
    futures = []

    passes = df.executor.passes
    N = 100
    with small_buffer(df):
        for i in range(N):
            futures.append(tpe.submit(do))

    done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)
    for future in done:
        assert count == future.result()
    assert df.executor.passes <= passes + N


def test_delayed(df):
    @vaex.delayed
    def add(a, b):
        return a + b
    total_promise = add(df.sum(df.x, delay=True), 1)
    df.execute()
    assert total_promise.get() == df.sum(df.x) + 1


def test_nested_task(df):
    @vaex.delayed
    def add(a, b):
        return a + b
    total_promise = add(df.sum(df.x, delay=True))

    @vaex.delayed
    def next(value):
        # during the handling of the sum task, we add a new task
        sumy_promise = df.sum(df.y, delay=True)
        if df.is_local():
            assert df.executor.local.executing
        # without callling the exector, since it should still be running its main loop
        return add(sumy_promise, value)
    total_promise = next(df.sum(df.x, delay=True))
    df.execute()
    assert total_promise.get() == df.sum(df.x) + df.sum(df.y)


# def test_add_and_cancel_tasks(df_executor):
#     df = df_executor

#     def add_task_and_cancel(fraction):
#         df.sum(df.x, delay=True)
#         return False

#     future = df.count(progress=add_task_and_cancel, delay=True)
#     df.execute()
#     with pytest.raises(vaex.execution.UserAbort):
#         future.get()
#     assert df.executor.tasks

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
