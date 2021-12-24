import asyncio
from unittest.mock import MagicMock
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import platform
import threading

import pytest
import numpy as np

from common import CallbackCounter, small_buffer
import vaex


def test_evaluate_expression_once():
    calls = 0
    def add(a, b):
        nonlocal calls
        if len(a) > 1:  # skip dtype calls
            calls += 1
        return a + b
    x = np.arange(5)
    y = x**2
    df = vaex.from_arrays(x=x, y=y)
    df.add_function('add', add)
    df['z'] = df.func.add(df.x, df.y)
    df.executor.passes = 0
    df.z.sum(delay=True)
    df._set('z', delay=True)
    calls = 0
    df.execute()
    assert df.executor.passes == 1
    assert calls == 1


def test_nested_use_of_executor():
    df = vaex.from_scalars(x=1, y=2)
    @vaex.delayed
    def next(x):
        # although the exector is still in its look, it's not using the threads anymore
        # so we should be able to use the executor again
        return x + df.y.sum()
    value = next(df.x.sum(delay=True))
    df.execute()
    assert value.get() == 1 + 2


def test_passes_two_datasets():
    df1 = vaex.from_scalars(x=1, y=2)
    df2 = vaex.from_scalars(x=1, y=3)
    executor = df1.executor
    executor.passes = 0
    df1.sum('x')
    assert executor.passes == 1
    df1.sum('x', delay=True)
    df2.sum('x', delay=True)
    df1.execute()
    assert executor.passes == 3


def test_passes_two_datasets_different_vars():
    x = np.array([2.])
    y = x**2
    dataset = vaex.dataset.DatasetArrays(x=x, y=y)
    df1 = vaex.from_dataset(dataset)
    df2 = vaex.from_dataset(dataset)
    df1.variables['a'] = 1
    df2.variables['a'] = 2
    df1['z'] = 'x + y * a'
    df2['z'] = 'x + y * a'
    executor = df1.executor
    executor.passes = 0
    s1 = df1.sum('z', delay=True)
    s2 = df2.sum('z', delay=True)
    df1.execute()
    assert executor.passes == 1
    assert s1.get() == 2 + 4 * 1
    assert s2.get() == 2 + 4 * 2


def test_passes_two_datasets_different_expressions():
    x = np.array([2.])
    y = x**2
    dataset = vaex.dataset.DatasetArrays(x=x, y=y)
    df1 = vaex.from_dataset(dataset)
    df2 = vaex.from_dataset(dataset)
    df1['a'] = 'x * y'
    df2['b'] = 'x + y'
    executor = df1.executor
    executor.passes = 0
    s1 = df1.sum('a', delay=True)
    s2 = df2.sum('b', delay=True)
    df1.execute()
    assert executor.passes == 1
    assert s1.get() == 2 * 4
    assert s2.get() == 2 + 4


def test_passes_filtering():
    x = np.arange(10)
    df = vaex.from_arrays(x=x, y=x**2)
    df1 = df[df.x < 4]
    df2 = df[df.x > 7]

    executor = df.executor
    executor.passes = 0
    result1 = df1.sum('x', delay=True)
    result2 = df2.sum('x', delay=True)
    df.execute()
    assert executor.passes == 1
    assert result1.get() == 1 + 2 + 3
    assert result2.get() == 8 + 9


def test_passes_mixed_filtering():
    x = np.arange(10)
    df = vaex.from_arrays(x=x, y=x**2)
    df1 = df[df.x < 4]
    df2 = df

    executor = df.executor
    executor.passes = 0
    result1 = df1.sum('x', delay=True)
    result2 = df2.sum('x', delay=True)
    df.execute()
    assert executor.passes == 1
    assert result1.get() == 1 + 2 + 3
    assert result2.get() == 45


def test_multiple_tasks_different_columns_names():
    df1 = vaex.from_scalars(x=1, y=2)
    df2 = vaex.from_scalars(x=1, y=2)
    x = df1.sum('x', delay=True)
    y = df2.sum('y', delay=True)
    df1.execute()
    assert x.get() == 1
    assert y.get() == 2


def test_merge_aggregation_tasks():
    df = vaex.from_arrays(x=[1, 2], y=[2, 3])
    binners = df._create_binners('x', [0.5, 2.5], 2)
    binners2 = df._create_binners('x', [0.5, 2.5], 2)
    assert len(binners) == 1
    vaex.agg.count().add_tasks(df, binners, progress=False)
    assert len(df.executor.tasks) == 1
    assert binners is not binners2
    assert binners[0] is not binners2[0]
    assert binners == binners2
    assert binners[0] == binners2[0]
    vaex.agg.sum('y').add_tasks(df, binners, progress=False)
    assert len(df.executor.tasks) == 2
    tasks = df.executor._pop_tasks()
    assert len(tasks) == 2
    tasks = vaex.execution._merge_tasks_for_df(tasks, df)
    assert len(tasks) == 1
    assert isinstance(tasks[0], vaex.tasks.TaskAggregations)


def test_merge_same_aggregation_tasks():
    df = vaex.from_arrays(x=[1, 2], y=[2, 3])
    binners = df._create_binners('x', [0.5, 2.5], 2)
    binners2 = df._create_binners('x', [0.5, 2.5], 2)
    assert len(binners) == 1
    # these two aggregations should be merged into 1 subtask
    [task1], result1 = vaex.agg.count().add_tasks(df, binners, progress=False)
    [task2], result2 = vaex.agg.count().add_tasks(df, binners, progress=False)
    assert len(df.executor.tasks) == 1
    df.execute()
    assert task1 is task2
    assert np.all(result1.get() == result2.get())


def test_stop_early():
    df = vaex.from_arrays(x=np.arange(100))
    counter = CallbackCounter(True)
    # with limit_raise=False, we can stop early
    df._hash_map_unique('x', delay=True, limit=1, limit_raise=False, progress=counter)
    assert len(df.executor.tasks) == 1
    task = df.executor.tasks[0]
    with small_buffer(df, 3):
        df.execute()
    assert task.stopped is True
    # thus progress should not be == 1 at the last call
    assert counter.last_args[0] < 1

    # if another task is added, we should continue
    df._hash_map_unique('x', delay=True, limit=1, limit_raise=False, progress=counter)
    task = df.executor.tasks[0]
    df.count('x', delay=True)
    assert len(df.executor.tasks) == 2
    with small_buffer(df, 3):
        df.execute()
    assert task.stopped is True
    # thus progress should be == 1 at the last call
    assert counter.last_args[0] == 1



def test_signals(df):
    x = np.arange(10)
    y = x**2
    sum_x_expected = x.sum()
    sum_y_expected = y.sum()
    with vaex.cache.off():
        mock_begin = MagicMock()
        mock_progress = MagicMock()
        mock_end = MagicMock()
        len(df)  # ensure we have the filter precomputed
        df.executor.signal_begin.connect(mock_begin)
        df.executor.signal_progress.connect(mock_progress)
        df.executor.signal_end.connect(mock_end)
        sum_x = df.sum(df.x, delay=True)
        sum_y = df.sum(df.y, delay=True)
        df.execute()
        assert sum_x.get() == sum_x_expected
        assert sum_y.get() == sum_y_expected
        mock_begin.assert_called_once()
        mock_progress.assert_called_with(1.0)
        mock_end.assert_called_once()


def test_reentrant_catch(df_local):
    with vaex.cache.off():
        df = df_local

        # a 'worker' thread should not be allowed to trigger a new computation
        def progress(fraction):
            print('progress', fraction)
            df.count(df.x)  # enters the executor again

        with pytest.raises(RuntimeError) as exc:
            df.count(df.x, progress=progress)
        assert 'nested' in str(exc.value)


@pytest.mark.asyncio
async def test_async_safe(df_local):
    df = df_local
    with vaex.cache.off():
        async def do():
            promise = df.x.count(delay=True)
            import random
            r = random.random() * 0.01
            await asyncio.sleep(r)
            await df.execute_async()
            return await promise
        awaitables = []
        passes = df.executor.passes = 0
        N = 1000
        with small_buffer(df):
            for i in range(N):
                awaitables.append(do())
        import asyncio
        values = await asyncio.gather(*awaitables)
        assert df.executor.passes < N



@pytest.mark.skipif(platform.system().lower() == 'windows', reason="hangs appveyor very often, bug?")
def test_thread_safe(df_local):
    with vaex.cache.off():
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
    with vaex.cache.off():
        @vaex.delayed
        def add(a, b):
            return a + b
        total_promise = add(df.sum(df.x, delay=True), 1)
        df.execute()
        assert total_promise.get() == df.sum(df.x) + 1


def test_nested_task(df):
    with vaex.cache.off():
        @vaex.delayed
        def add(a, b):
            return a + b
        total_promise = add(df.sum(df.x, delay=True))

        @vaex.delayed
        def next(value):
            # during the handling of the sum task, we add a new task
            sumy_promise = df.sum(df.y, delay=True)
            if df.is_local():
                assert not df.executor.local.executing
            # without callling the exector, since it should still be running its main loop
            return add(sumy_promise, value)
        total_promise = next(df.sum(df.x, delay=True))
        df.execute()
        assert total_promise.get() == df.sum(df.x) + df.sum(df.y)


def test_executor_from_other_thread():
    with vaex.cache.off():
        df = vaex.from_arrays(x=[1, 2])
        def execute():
            # but call execute from a different thread
            df.execute()
        # we add a tasks from the main thread, we use binby without limits to force
        # a double computation.
        c = df.count('x', binby='x', delay=True, edges=True)
        thread = threading.Thread(target=execute)
        thread.start()
        thread.join()
        assert sum(c.get()) == 2

def test_cancel_single_job():
    df = vaex.from_arrays(x=[1, 2, 3])
    res1 = df._set(df.x, unique_limit=1, delay=True)
    res2 = df._set(df.x, delay=True)
    df.execute()
    assert res1.isRejected
    assert res2.isFulfilled


def test_exception():
    df = vaex.from_arrays(x=[1, 2, 3])
    with pytest.raises(vaex.RowLimitException, match='.* >= 1 .*'):
        df._set(df.x, unique_limit=1)


def test_continue_next_task_after_cancel():
    df = vaex.from_arrays(x=[1, 2, 3])
    res1 = df._set(df.x, unique_limit=1, delay=True)
    def on_error(exception):
        return df._set(df.x, delay=True)
    result = res1.then(None, on_error)
    df.execute()
    assert res1.isRejected
    assert result.isFulfilled


@pytest.mark.asyncio
async def test_auto_execute():
    df = vaex.from_arrays(x=[2, 4])
    async def means():
        count, sum = await asyncio.gather(df.x.count(delay=True), df.x.sum(delay=True))
        mean = await df.x.mean(delay=True)
        return sum / count, mean
    async with df.executor.auto_execute():
        mean1, mean2 = await means()
        assert mean1 == 3
        assert mean2 == 3



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
