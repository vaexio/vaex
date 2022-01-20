from __future__ import division, print_function
import ast
import asyncio
from collections import defaultdict
import contextlib
import os
import time
import threading
import math
import multiprocessing
import logging
import queue
from typing import List

import dask.utils
import numpy as np

import vaex.asyncio
import vaex.cpu  # force registration of task-part-cpu
import vaex.encoding
import vaex.memory
import vaex.multithreading
import vaex.vaexfast
import vaex.events
import vaex.settings

try:
    # support py 36 for the moment
    import contextvars
    has_contextvars = True
except ModuleNotFoundError:
    has_contextvars = False


logger = logging.getLogger("vaex.execution")


class UserAbort(Exception):
    def __init__(self, reason):
        super(UserAbort, self).__init__(reason)


class Run:
    def __init__(self, tasks):
        self.tasks = tasks
        self.cancelled = False
        dataset = self.tasks[0].df.dataset
        if self.tasks[0].df._index_start != 0 or self.tasks[0].df._index_end != dataset.row_count:
            dataset = dataset.slice(self.tasks[0].df._index_start, self.tasks[0].df._index_end)
        for task in tasks:
            assert self.tasks[0].df._index_start == task.df._index_start
            assert self.tasks[0].df._index_end == task.df._index_end
        self.dataset = dataset
        self.pre_filter_per_df = {}
        self.expressions = list(set(expression for task in tasks for expression in task.expressions_all))
        self.tasks_per_df = dict()
        for task in tasks:
            if task.df not in self.tasks_per_df:
                self.tasks_per_df[task.df] = []
            self.tasks_per_df[task.df].append(task)
        for task in tasks:
            if task.df not in self.pre_filter_per_df:
                self.pre_filter_per_df[task.df] = task.pre_filter
            else:
                if self.pre_filter_per_df[task.df] != task.pre_filter:
                    raise ValueError(f"All tasks need to be pre_filter'ed or not pre_filter'ed, it cannot be mixed: {tasks}")


        # find the columns from the dataset we need
        dfs = set()
        for task in tasks:
            dfs.add(task.df)
        # shared between all datasets (TODO: can we detect the filter is cached already?)
        self.dataset_deps = set()
        # per dataframe:
        self.filter_deps = {}
        self.selection_deps = {}
        self.expression_deps = {}

        for df in dfs:
            self.filter_deps[df] = set()
            self.selection_deps[df] = set()
            self.expression_deps[df] = set()
            others = set(df.variables) | set(df.virtual_columns) | set(df.selection_histories)
            tasks_df = [task for task in tasks if task.df == df]
            expressions = list(set(expression for task in tasks_df for expression in task.expressions_all))
            selections = list(set(selection for task in tasks_df for selection in task.selections))
            variables = set()
            for expression in expressions:
                variables |= df._expr(expression).expand().variables(ourself=True)
            for var in variables:
                if var not in self.dataset:
                    if var not in others:
                        raise RuntimeError(f'Oops, requesting column {var} from dataset, but it does not exist')
                    else:
                        pass  # ok, not a column, just a var or virtual column
                else:
                    self.dataset_deps.add(var)
                    self.expression_deps[df].add(var)

            variables = set()
            if df.filtered:
                variables = df.get_selection(vaex.dataframe.FILTER_SELECTION_NAME).dependencies(df)
                for var in variables:
                    if var not in self.dataset:
                        if var not in others:
                            raise RuntimeError(f'Oops, requesting column {var} from dataset, but it does not exist')
                        else:
                            pass  # ok, not a column, just a var or virtual column
                    else:
                        self.dataset_deps.add(var)
                        self.filter_deps[df].add(var)
            variables = set()
            for selection in selections:
                if selection is not None:
                    variables |= df._selection_expression(selection).dependencies()
            for var in variables:
                if var not in self.dataset:
                    if var not in others:
                        raise RuntimeError(f'Oops, requesting column {var} from dataset, but it does not exist')
                    else:
                        pass  # ok, not a column, just a var or virtual column
                else:
                    self.dataset_deps.add(var)
                    self.selection_deps[df].add(var)

        logger.debug('Using columns %r from dataset', self.dataset_deps)


def _merge(tasks):
    dfs = set()
    for task in tasks:
        dfs.add(task.df)
    tasks_merged = []
    for df in dfs:
        tasks_df = [k for k in tasks if k.df == df]
        tasks_merged.extend(_merge_tasks_for_df(tasks_df, df))
    return tasks_merged


def _merge_tasks_for_df(tasks, df):
    # non-mergable:
    tasks_non_mergable = [task for task in tasks if not isinstance(task, vaex.tasks.TaskAggregation)]
    # mergable
    tasks_agg = [task for task in tasks if isinstance(task, vaex.tasks.TaskAggregation)]
    tasks_merged = []

    # Merge aggregation tasks to single aggregations tasks if possible
    tasks_agg_per_grid = defaultdict(list)
    for task in tasks_agg:
        tasks_agg_per_grid[task.binners].append(task)
    for binners, tasks in tasks_agg_per_grid.items():
        task_merged = vaex.tasks.TaskAggregations(df, binners)
        task_merged.original_tasks = tasks
        for i, subtask in enumerate(tasks):
            # chain progress
            def progress_wrapper(p, subtask=subtask):
                result = all(subtask.signal_progress.emit(p))
                return result
            task_merged.signal_progress.connect(progress_wrapper)
            task_merged.add_aggregation_operation(subtask.aggregation_description)
            @vaex.delayed
            def assign(value, i=i, subtask=subtask):
                subtask.fulfill(value[i])
            assign(task_merged)
            task_merged.done(None, subtask.reject)
            task_merged.signal_start.connect(subtask.signal_start.emit)
        tasks_merged.append(task_merged)
    return tasks_non_mergable + tasks_merged


class Executor:
    """An executor is responsible to executing tasks, they are not reentrant, but thread safe"""
    def __init__(self, async_method=None):
        self.tasks : List[Task] = []
        self.async_method = async_method
        self.signal_begin = vaex.events.Signal("begin")
        self.signal_progress = vaex.events.Signal("progress")
        self.signal_end = vaex.events.Signal("end")
        self.signal_cancel = vaex.events.Signal("cancel")
        self.local = threading.local()  # to make it non-reentrant
        if has_contextvars:
            # used for calling execute_async from different async callstacks
            self.isnested = contextvars.ContextVar('executor', default=False)
        self.lock = threading.Lock()
        self.event_loop = asyncio.new_event_loop()

    if hasattr(contextlib, 'asynccontextmanager'):
        @contextlib.asynccontextmanager
        async def auto_execute(self):
            '''This async executor will start executing tasks automatically when a task is awaited for.'''
            vaex.promise.auto_await_executor.set(self)
            try:
                yield
                await self.execute_async()
            finally:
                vaex.promise.auto_await_executor.set(None)
    else:
        async def auto_execute(self):
            raise NotImplemented('Only on Python >3.7')

    async def execute_async(self):
        raise NotImplementedError

    def execute(self):
        raise NotImplementedError

    def run(self, coro):
        async_method = self.async_method or vaex.settings.main.async_
        if async_method == "nest":
            return vaex.asyncio.just_run(coro)
        elif async_method == "awaitio":
            with vaex.asyncio.with_event_loop(self.event_loop):
                return self.event_loop.run_until_complete(coro)
        else:
            raise RuntimeError(f'No async method: {async_default}')

    def schedule(self, task):
        '''Schedules new task for execution, will return an existing tasks if the same task was already added'''
        with self.lock:
            for task_existing in self.tasks:
                if (task_existing.df == task.df) and task_existing.fingerprint() == task.fingerprint():
                    key = task.fingerprint()
                    logger.debug("Did not add already existing task with fingerprint: %r", key)
                    return task_existing

            if vaex.cache.is_on() and task.cacheable:
                key = task.fingerprint()                
                logger.debug("task fingerprint: %r (dataset fp=%r)", key, task.df.dataset.fingerprint)
                logger.debug("task repr: %r", task)
                result = vaex.cache.get(key, type="task")
                if result is not None:
                    logger.info("task not added, used cache key: %r", key)
                    task.fulfill(result)
                    return task
                else:
                    logger.info("task *will* be scheduled with cache key: %r", key)
            if task not in self.tasks:
                self.tasks.append(task)
                logger.info("task added, queue: %r", self.tasks)
                return task

    def _pop_tasks(self):
        # returns a list of tasks that can be executed in 1 pass over the data
        # (which currently means all tasks for 1 dataset) and drop them from the
        # list of tasks
        with self.lock:
            if len(self.tasks) == 0:
                return []
            else:
                dataset = self.tasks[0].df.dataset
                i1, i2 = self.tasks[0].df._index_start, self.tasks[0].df._index_end
                # if we have the same dataset *AND* we want to have the same slice out of it
                # we can share it in in single pass over the dataset
                def shares_dataset(df, i1=i1, i2=i2, dataset=dataset):
                    return dataset == df.dataset and i1 == df._index_start and i2 == df._index_end

                tasks = [task for task in self.tasks if shares_dataset(task.df)]
                logger.info("executing tasks in run: %r", tasks)
                for task in tasks:
                    self.tasks.remove(task)
                return tasks


class ExecutorLocal(Executor):
    def __init__(self, thread_pool=None, chunk_size=None, chunk_size_min=None, chunk_size_max=None, thread_mover=None, zigzag=True):
        super().__init__()
        self.thread_pool = thread_pool or vaex.multithreading.ThreadPoolIndex()
        self.chunk_size = chunk_size
        self.chunk_size_min = chunk_size_min
        self.chunk_size_max = chunk_size_max
        self.thread = None
        self.passes = 0  # how many times we passed over the data
        self.zig = True  # zig or zag
        self.zigzag = zigzag
        self.event_loop = asyncio.new_event_loop()

    def _cancel(self, run):
        logger.debug("cancelling")
        self.signal_cancel.emit()
        run.cancelled = True

    def chunk_size_for(self, row_count):
        chunk_size = self.chunk_size or vaex.settings.main.chunk.size
        chunk_size_min = self.chunk_size or vaex.settings.main.chunk.size_min
        chunk_size_max = self.chunk_size or vaex.settings.main.chunk.size_max
        if chunk_size is None:
            # we determine it automatically by defaulting to having each thread do 1 chunk
            chunk_size_1_pass = vaex.utils.div_ceil(row_count, self.thread_pool.nthreads)
            # brackated by a min and max chunk_size
            chunk_size = min(chunk_size_max, max(chunk_size_min, chunk_size_1_pass))
        return chunk_size

    async def execute_async(self):
        # consume awaitables from the generator, and await them here at the top
        # level, such that we don't need await in the downstream code
        # so we can reuse the same code in a sync way
        gen = self.execute_generator(use_async=True)
        value = None
        while True:
            try:
                value = gen.send(value)
                value = await value
            except StopIteration:
                break

    def execute(self):
        for _ in self.execute_generator():
            pass  # just eat all elements

    def execute_generator(self, use_async=False):
        logger.debug("starting with execute")

        with self.lock:  # setup thread local initial values
            if not hasattr(self.local, 'executing'):
                self.local.executing = False

        try:
            t0 = time.time()
            self.local.cancelled = False
            self.signal_begin.emit()
            # keep getting a list of tasks
            # we currently process tasks (grouped) per df
            # but also, tasks can add new tasks
            while True:
                tasks = self.local.tasks = self._pop_tasks()

                # wo don't allow any thread from our thread pool to enter (a computation should never produce a new task)
                # and we explicitly disallow reentry (this usually means a bug in vaex, or bad usage)
                chunk_executor_thread = threading.current_thread() in self.thread_pool._threads
                import traceback
                trace = ''.join(traceback.format_stack())
                if chunk_executor_thread or self.local.executing and (has_contextvars is False or self.isnested.get() is True):
                    logger.error("nested execute call")
                    raise RuntimeError("nested execute call: %r %r\nlast trace:\n%s\ncurrent trace:\n%s" % (chunk_executor_thread, self.local.executing, self.local.last_trace, trace))
                else:
                    self.local.last_trace = trace

                self.local.executing = True
                if has_contextvars:
                    self.isnested.set(True)

                if not tasks:
                    break
                tasks = _merge(tasks)
                run = Run(tasks)
                self.passes += 1
                dataset = run.dataset

                run.variables = {}
                for df in run.tasks_per_df.keys():
                    run.variables[df] = {key: df.evaluate_variable(key) for key in df.variables.keys()}

                # (re) thrown exceptions as soon as possible to avoid complicated stack traces
                for task in tasks:
                    if task.isRejected:
                        task.get()
                    if hasattr(task, "check"):
                        try:
                            task.check()
                        except Exception as e:
                            task.reject(e)
                            raise

                for task in run.tasks:
                    task.signal_start.emit(self)

                for task in run.tasks:
                    task._results = []
                    if not any(task.signal_progress.emit(0)):
                        logger.debug("task cancelled immediately")
                        task.cancelled = True
                row_count = dataset.row_count
                chunk_size = self.chunk_size_for(row_count)
                encoding = vaex.encoding.Encoding()
                run.nthreads = nthreads = self.thread_pool.nthreads
                task_checkers = vaex.tasks.create_checkers()
                memory_tracker = vaex.memory.create_tracker()
                vaex.memory.local.agg = memory_tracker
                # we track this for consistency
                memory_usage = 0
                for task in tasks:
                    for task_checker in task_checkers:
                        task_checker.add_task(task)
                    spec = encoding.encode('task', task)
                    spec['task-part-cpu-type'] = spec.pop('task-type')
                    def create_task_part():
                        nonlocal memory_usage
                        task_part = encoding.decode('task-part-cpu', spec, df=task.df, nthreads=nthreads)
                        memory_usage += task_part.memory_usage()
                        for task_checker in task_checkers:
                            task_checker.add_task(task)
                        if task.requires_fingerprint:
                            task_part.fingerprint = task.fingerprint()
                        return task_part
                    # We want at least 1 task part (otherwise we cannot do any work)
                    # then we ask for the task part how often we should split
                    # This means that we can have 100 threads, but only 2 task parts
                    # In this case, evaluation of expressions is still multithreaded,
                    # but aggregation is reduced to effectively 2 threads.
                    task_part_0 = create_task_part()
                    ideal_task_splits = task_part_0.ideal_splits(self.thread_pool.nthreads)
                    assert ideal_task_splits <= self.thread_pool.nthreads, f'Cannot have more splits {ideal_task_splits} then threads {self.thread_pool.nthreads}'
                    if ideal_task_splits == self.thread_pool.nthreads or task.see_all:
                        # in the simple case, we just use a list
                        task._parts = [task_part_0] + [create_task_part() for i in range(1, ideal_task_splits)]
                    else:
                        # otherwise a queue
                        task._parts = queue.Queue()
                        task._parts.put(task_part_0)
                        for i in range(1, ideal_task_splits):
                            task._parts.put(create_task_part())
                if memory_usage != memory_tracker.used:
                    raise RuntimeError(f"Reported memory usage by tasks was {memory_usage}, while tracker listed {memory_tracker.used}")
                vaex.memory.local.agg = None

                # TODO: in the future we might want to enable the zigzagging again, but this requires all datasets to implement it
                # if self.zigzag:
                #     self.zig = not self.zig
                def progress(p):
                    # no global cancel and at least 1 tasks wants to continue, then we continue
                    ok_tasks = any([task.progress(p) for task in tasks])
                    all_stopped = all([task.stopped for task in tasks])
                    ok_executor = all(self.signal_progress.emit(p))
                    if all_stopped:
                        logger.debug("Pass cancelled because all tasks are stopped: %r", tasks)
                    if not ok_tasks:
                        logger.debug("Pass cancelled because all tasks cancelled: %r", tasks)
                    if not ok_executor:
                        logger.debug("Pass cancelled because of the global progress event: %r", self.signal_progress.callbacks)
                    return ok_tasks and ok_executor and not all_stopped
                yield from self.thread_pool.map(self.process_part, dataset.chunk_iterator(run.dataset_deps, chunk_size),
                                                    dataset.row_count,
                                                    progress=progress,
                                                    cancel=lambda: self._cancel(run), unpack=True, run=run, use_async=use_async)
                duration_wallclock = time.time() - t0
                logger.debug("executing took %r seconds", duration_wallclock)
                self.local.executing = False
                if has_contextvars:
                    self.isnested.set(False)
                if True:  # kept to keep the diff small
                    for task in tasks:
                        if not task.cancelled:
                            logger.debug("fulfill task: %r", task)
                            parts = task._parts
                            if not isinstance(parts, list):
                                parts_queue = parts
                                parts = []
                                while not parts_queue.empty():
                                    parts.append(parts_queue.get())
                            parts[0].reduce(parts[1:])
                            logger.debug("wait for task: %r", task)
                            task._result = parts[0].get_result()
                            task.end()
                            task.fulfill(task._result)
                            logger.debug("got result for: %r", task)
                            if task._result is not None and task.cacheable:  # we don't want to store None
                                if vaex.cache.is_on():
                                    # we only want to store the original task results into the cache
                                    tasks_cachable = task.original_tasks if isinstance(task, vaex.tasks.TaskAggregations) else [task]
                                    for task_cachable in tasks_cachable:
                                        key = task_cachable.fingerprint()
                                        previous_result = vaex.cache.get(key, type='task')
                                        if (previous_result is not None):# and (previous_result != task_cachable.get()):
                                            try:
                                                if previous_result != task_cachable.get():
                                                    # this can happen with multithreading, where two threads enter the same tasks in parallel (IF using different executors)
                                                    logger.warning("calculated new result: %r, while cache had value: %r", previous_result, task_cachable.get())
                                            except ValueError:  # when comparing numpy results
                                                if not np.array_equal(previous_result, task_cachable.get(), equal_nan=True):
                                                    # this can happen with multithreading, where two threads enter the same tasks in parallel (IF using different executors)
                                                    logger.warning("calculated new result: %r, while cache had value: %r", previous_result, task_cachable.get())
                                        vaex.cache.set(key, task_cachable.get(), type='task', duration_wallclock=duration_wallclock)
                                        logger.info("added result: %r in cache under key: %r", task_cachable.get(), key)

                        else:
                            logger.debug("rejecting task: %r", task)
                            # we now reject, in the main thread
                            if task._toreject:
                                task.reject(task._toreject)
                            else:
                                task.reject(UserAbort("Task was cancelled"))
                            # remove references
                        task._result = None
                        task._results = None
                    self.signal_end.emit()
        except:  # noqa
            self.signal_cancel.emit()
            raise
        finally:
            self.local.executing = False
            if has_contextvars:
                self.isnested.set(False)

    def process_part(self, thread_index, i1, i2, chunks, run):
        if not run.cancelled:
            if thread_index >= run.nthreads:
                raise ValueError(f'thread_index={thread_index} while only having {run.nthreads} blocks')
            for df, tasks in run.tasks_per_df.items():
                self.process_tasks(thread_index, i1, i2, chunks, run, df, tasks)
        return i2 - i1

    def process_tasks(self, thread_index, i1, i2, chunks, run, df, tasks):
        if 1:  # avoid large diff
            if i1 == i2:
                raise RuntimeError(f'Oops, get an empty chunk, from {i1} to {i2}, that should not happen')
            N = i2 - i1
            for name, chunk in chunks.items():
                assert len(chunk) == N, f'Oops, got a chunk ({name}) of length {len(chunk)} while it is expected to be of length {N} (at {i1}-{i2}'

            assert df in run.variables
            from .scopes import _BlockScope

            expressions = list(set(expression for task in tasks for expression in task.expressions_all))
            pre_filter = tasks[0].pre_filter
            if pre_filter:
                filter_deps = run.filter_deps[df]
                filter_scope = _BlockScope(df, i1, i2, None, selection=True, values={**run.variables[df], **{k: chunks[k] for k in filter_deps if k in chunks}})
                filter_scope.filter_mask = None
                filter_mask = filter_scope.evaluate(vaex.dataframe.FILTER_SELECTION_NAME)
                deps = run.expression_deps[df] | run.selection_deps[df]
                chunks = {k:vaex.array_types.filter(v, filter_mask) for k, v, in chunks.items() if k in deps}
            else:
                filter_mask = None
            def sanity_check(name, chunk):
                if not vaex.column.is_column_like(chunk):
                    raise TypeError(f'Evaluated a chunk ({name}) that is not an array of column like object: {chunk!r} (type={type(chunk)}')
            for name, chunk in chunks.items():
                sanity_check(name, chunk)
            block_scope = _BlockScope(df, i1, i2, values={**run.variables[df], **chunks})
            block_scope.mask = filter_mask
            block_dict = {expression: block_scope.evaluate(expression) for expression in expressions}
            selection_scope = _BlockScope(df, i1, i2, None, selection=True, values={**block_scope.values})
            selection_scope.filter_mask = filter_mask
            if not pre_filter and df.filtered:
                filter_mask = selection_scope.evaluate(vaex.dataframe.FILTER_SELECTION_NAME)

            memory_tracker = vaex.memory.create_tracker()
            task_checkers = vaex.tasks.create_checkers()
            for task in tasks:
                if task.stopped:
                    continue
                assert df is task.df
                blocks = [block_dict[expression] for expression in task.expressions_all]
                def fix(ar):
                    if np.ma.isMaskedArray(ar):
                        ar = vaex.utils.unmask_selection_mask(ar)
                    return ar
                selections = [None if s is None else fix(selection_scope.evaluate(s)) for s in task.selections]
                if not run.cancelled:
                    if task.see_all:
                        assert isinstance(task._parts, list)
                        task_parts = task._parts
                    else:
                        if isinstance(task._parts, list):
                            task_parts = [task._parts[thread_index]]
                        else:
                            task_parts = [task._parts.get()]
                    some_parts_stopped = False
                    for task_index, task_part in enumerate(task_parts):
                        if not task_part.stopped:
                            try:
                                if task.see_all:
                                    task_part.process(thread_index, i1, i2, filter_mask, selections, blocks)
                                else:
                                    task_part.process(thread_index, i1, i2, filter_mask, selections, blocks)
                            except Exception as e:
                                # we cannot call .reject, since then we'll handle fallbacks in this thread
                                task._toreject = e
                                task.cancelled = True
                            finally:
                                if not isinstance(task._parts, list):
                                    task._parts.put(task_part)
                            # we could be done after processing
                            if task_part.stopped:
                                some_parts_stopped = True
                        else:
                            some_parts_stopped = True
                        if some_parts_stopped:
                            break
                    if some_parts_stopped:  # if 1 is done, the whole task is done
                        task.stopped = True
                if memory_tracker.track_live:
                    for part in task_parts:
                        memory_tracker.using(part.memory_usage())
                for task_checker in task_checkers:
                    if task_checker.track_live:
                        for part in task_parts:
                            task_checker.add_task_part(part)
