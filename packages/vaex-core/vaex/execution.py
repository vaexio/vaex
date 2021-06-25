from __future__ import division, print_function
import ast
from collections import defaultdict
import os
import time
import threading
import multiprocessing
import logging
import queue

import numpy as np

import vaex.cpu  # force registration of task-part-cpu
import vaex.encoding
import vaex.multithreading
import vaex.vaexfast
import vaex.events


chunk_size_min_default = int(os.environ.get('VAEX_CHUNK_SIZE_MIN', 1024))
chunk_size_max_default = int(os.environ.get('VAEX_CHUNK_SIZE_MAX', 1024*1024))
chunk_size_default = ast.literal_eval(os.environ.get('VAEX_CHUNK_SIZE', 'None'))
if chunk_size_default is not None:
    chunk_size_default = int(chunk_size_default)  # make sure it's an int

logger = logging.getLogger("vaex.execution")
thread_count_default = multiprocessing.cpu_count()


class UserAbort(Exception):
    def __init__(self, reason):
        super(UserAbort, self).__init__(reason)


class Run:
    def __init__(self, tasks):
        self.tasks = tasks
        self.cancelled = False
        self.df = self.tasks[0].df
        self.pre_filter = tasks[0].pre_filter
        if any(self.pre_filter != task.pre_filter for task in tasks[1:]):
            raise ValueError(f"All tasks need to be pre_filter'ed or not pre_filter'ed, it cannot be mixed: {tasks}")
        if self.pre_filter and not self.df.filtered:
            raise ValueError("Requested pre_filter for task while DataFrame is not filtered")
        self.expressions = list(set(expression for task in tasks for expression in task.expressions_all))

def _merge(tasks, df):
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
        tasks_merged.append(task_merged)
    return tasks_non_mergable + tasks_merged

class Executor:
    """An executor is responsible to executing tasks, they are not reentrant, but thread safe"""
    def __init__(self):
        self.tasks = []
        self.signal_begin = vaex.events.Signal("begin")
        self.signal_progress = vaex.events.Signal("progress")
        self.signal_end = vaex.events.Signal("end")
        self.signal_cancel = vaex.events.Signal("cancel")
        self.local = threading.local()  # to make it non-reentrant
        self.lock = threading.Lock()

    def schedule(self, task):
        '''Schedules new task for execution, will return an existing tasks if the same task was already added'''
        with self.lock:
            for task_existing in self.tasks:
                # WARNING: tasks fingerprint ignores the dataframe
                if (task_existing.df == task.df) and task_existing.fingerprint() == task.fingerprint():
                    key = task.fingerprint()
                    logger.debug("Did not add already existing task with fingerprint: %r", key)
                    return task_existing

            if vaex.cache.is_on() and task.cacheable:
                key_task = task.fingerprint()
                key_df = task.df.fingerprint()
                # WARNING tasks' fingerprints don't include the dataframe
                key = f'{key_task}-{key_df}'

                logger.debug("task fingerprint: %r", key)
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
        # (which currently means all tasks for 1 dataframe) and drop them from the
        # list of tasks
        with self.lock:
            dfs = list(set(task.df for task in self.tasks))
            if len(dfs) == 0:
                return []
            else:
                df = dfs[0]
                tasks = [task for task in self.tasks if task.df == df]
                logger.info("executing tasks in run: %r", tasks)
                for task in tasks:
                    self.tasks.remove(task)
                return tasks


class ExecutorLocal(Executor):
    def __init__(self, thread_pool=None, chunk_size=chunk_size_default, chunk_size_min=chunk_size_min_default, chunk_size_max=chunk_size_max_default, thread_mover=None, zigzag=True):
        super().__init__()
        self.thread_pool = thread_pool or vaex.multithreading.ThreadPoolIndex()
        self.chunk_size = chunk_size
        self.chunk_size_min = chunk_size_min_default
        self.chunk_size_max = chunk_size_max_default
        self.thread = None
        self.passes = 0  # how many times we passed over the data
        self.zig = True  # zig or zag
        self.zigzag = zigzag

    def _cancel(self, run):
        logger.debug("cancelling")
        self.signal_cancel.emit()
        run.cancelled = True

    def chunk_size_for(self, row_count):
        chunk_size = self.chunk_size
        if chunk_size is None:
            # we determine it automatically by defaulting to having each thread do 1 chunk
            chunk_size_1_pass = vaex.utils.div_ceil(row_count, self.thread_pool.nthreads)
            # brackated by a min and max chunk_size
            chunk_size = min(self.chunk_size_max, max(self.chunk_size_min, chunk_size_1_pass))
        return chunk_size

    async def execute_async(self):
        logger.debug("starting with execute")

        with self.lock:  # setup thread local initial values
            if not hasattr(self.local, 'executing'):
                self.local.executing = False

        # wo don't allow any thread from our thread pool to enter (a computation should never produce a new task)
        # and we explicitly disallow reentry (this usually means a bug in vaex, or bad usage)
        chunk_executor_thread = threading.current_thread() in self.thread_pool._threads
        import traceback
        trace = ''.join(traceback.format_stack())
        if chunk_executor_thread or self.local.executing:
            logger.error("nested execute call")
            raise RuntimeError("nested execute call: %r %r\nlast trace:\n%s\ncurrent trace:\n%s" % (chunk_executor_thread, self.local.executing, self.local.last_trace, trace))
        else:
            self.local.last_trace = trace

        self.local.executing = True
        try:
            t0 = time.time()
            self.local.cancelled = False
            self.signal_begin.emit()
            cancelled = False
            # keep getting a list of tasks
            # we currently process tasks (grouped) per df
            # but also, tasks can add new tasks
            while not cancelled:
                tasks = self.local.tasks = self._pop_tasks()
                if not tasks:
                    break
                tasks = _merge(tasks, tasks[0].df)
                run = Run(tasks)
                self.passes += 1

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
                    task._results = []
                    if not any(task.signal_progress.emit(0)):
                        logger.debug("task cancelled immediately")
                        task.cancelled = True
                row_count = run.df._index_end - run.df._index_start
                chunk_size = self.chunk_size_for(row_count)
                run.block_scopes = [run.df._block_scope(0, chunk_size) for i in range(self.thread_pool.nthreads)]
                encoding = vaex.encoding.Encoding()
                for task in tasks:
                    spec = encoding.encode('task', task)
                    spec['task-part-cpu-type'] = spec.pop('task-type')
                    def create_task_part():
                        return encoding.decode('task-part-cpu', spec, df=run.df)
                    # We want at least 1 task part (otherwise we cannot do any work)
                    # then we ask for the task part how often we should split
                    # This means that we can have 100 threads, but only 2 task parts
                    # In this case, evaluation of expressions is still multithreaded,
                    # but aggregation is reduced to effectively 2 threads.
                    task_part_0 = create_task_part()
                    ideal_task_splits = task_part_0.ideal_splits(self.thread_pool.nthreads)
                    assert ideal_task_splits <= self.thread_pool.nthreads, f'Cannot have more splits {ideal_task_splits} then threads {self.thread_pool.nthreads}'
                    if ideal_task_splits == self.thread_pool.nthreads:
                        # in the simple case, we just use a list
                        task._parts = [task_part_0] + [create_task_part() for i in range(1, self.thread_pool.nthreads)]
                    else:
                        # otherwise a queue
                        task._parts = queue.Queue()
                        task._parts.put(task_part_0)
                        for i in range(1, ideal_task_splits):
                            task._parts.put(create_task_part())
                length = run.df.active_length()
                if vaex.cache.is_on():
                    key_df = run.df.fingerprint()
                # TODO: in the future we might want to enable the zigzagging again, but this requires all datasets to implement it
                # if self.zigzag:
                #     self.zig = not self.zig
                dataset = run.df.dataset[run.df._index_start:run.df._index_end]
                # find the columns from the dataset we need
                variables = set()
                for expression in run.expressions:
                    variables |= run.df._expr(expression).expand().variables(ourself=True)
                columns = list(variables - set(run.df.variables) - set(run.df.virtual_columns))
                logger.debug('Using columns %r from dataset, chunk_size=%r', columns, chunk_size)
                for column in columns:
                    if column not in dataset:
                        raise RuntimeError(f'Oops, requesting column {column} from dataset, but it does not exist')
                async for element in self.thread_pool.map_async(self.process_part, dataset.chunk_iterator(columns, chunk_size),
                                                    dataset.row_count,
                                                    progress=lambda p: all(self.signal_progress.emit(p)) and
                                                    all([all(task.signal_progress.emit(p)) for task in tasks]) and
                                                    all([not task.cancelled for task in tasks]),
                                                    cancel=lambda: self._cancel(run), unpack=True, run=run):
                    pass  # just eat all element
                duration_wallclock = time.time() - t0
                logger.debug("executing took %r seconds", duration_wallclock)
                cancelled = self.local.cancelled or any(task.cancelled for task in tasks) or run.cancelled
                logger.debug("cancelled: %r", cancelled)
                if cancelled:
                    logger.debug("execution aborted")
                    for task in tasks:
                        task.reject(UserAbort("cancelled"))
                        # remove references
                        task._result = None
                        task._results = None
                        cancelled = True
                        if isinstance(task, vaex.tasks.TaskAggregations):
                            for subtask in task.original_tasks:
                                subtask.reject(UserAbort("cancelled"))
                else:
                    for task in tasks:
                        logger.debug("fulfill task: %r", task)
                        if not task.cancelled:
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
                                        key_task = task_cachable.fingerprint()
                                        # tasks' fingerprints don't include the dataframe
                                        key = f'{key_task}-{key_df}'
                                        previous_result = vaex.cache.get(key, type='task')
                                        if (previous_result is not None):# and (previous_result != task_cachable.get()):
                                            try:
                                                if previous_result != task_cachable.get():
                                                    # this can happen with multithreading, where two threads enter the same tasks in parallel (IF using different executors)
                                                    logger.warning("calculated new result: %r, while cache had value: %r", previous_result, task_cachable.get())
                                            except ValueError:  # when comparing numpy results
                                                if np.array_equal(previous_result, task_cachable.get(), equal_nan=True):
                                                    # this can happen with multithreading, where two threads enter the same tasks in parallel (IF using different executors)
                                                    logger.warning("calculated new result: %r, while cache had value: %r", previous_result, task_cachable.get())
                                        vaex.cache.set(key, task_cachable.get(), type='task', duration_wallclock=duration_wallclock)
                                        logger.info("added result: %r in cache under key: %r", task_cachable.get(), key)

                        else:
                            task.reject(UserAbort("Task was cancelled"))
                            # remove references
                            cancelled = True
                        task._result = None
                        task._results = None
                    self.signal_end.emit()
        except:  # noqa
            self.signal_cancel.emit()
            logger.exception("error in task, flush task queue")
            raise
        finally:
            self.local.executing = False

    def process_part(self, thread_index, i1, i2, chunks, run):
        if not run.cancelled:
            if thread_index >= len(run.block_scopes):
                raise ValueError(f'thread_index={thread_index} while only having {len(run.block_scopes)} blocks')
            block_scope = run.block_scopes[thread_index]
            block_scope.move(i1, i2)
            df = run.df
            if i1 == i2:
                raise RuntimeError(f'Oops, get an empty chunk, from {i1} to {i2}, that should not happen')
            N = i2 - i1
            for name, chunk in chunks.items():
                assert len(chunk) == N, f'Oops, got a chunk ({name}) of length {len(chunk)} while it is expected to be of length {N} (at {i1}-{i2}'
            if run.pre_filter:
                filter_mask = df.evaluate_selection_mask(None, i1=i1, i2=i2, cache=True)
                chunks = {k:vaex.array_types.filter(v, filter_mask) for k, v, in chunks.items()}
            else:
                filter_mask = None
            def sanity_check(name, chunk):
                if not vaex.column.is_column_like(chunk):
                    raise TypeError(f'Evaluated a chunk ({name}) that is not an array of column like object: {chunk!r} (type={type(chunk)}')
            for name, chunk in chunks.items():
                sanity_check(name, chunk)
            chunks = {name: df._auto_encode_data(name, ar) for name, ar in chunks.items()}
            chunks = {name: vaex.arrow.numpy_dispatch.wrap(ar) for name, ar in chunks.items()}
            block_scope.values.update(chunks)
            block_scope.mask = filter_mask
            block_dict = {expression: block_scope.evaluate(expression) for expression in run.expressions}
            for task in run.tasks:
                blocks = [block_dict[expression] for expression in task.expressions_all]
                if not run.cancelled:
                    # simple case, ntreads=nparts
                    if isinstance(task._parts, list):
                        task_part = task._parts[thread_index]
                    else:
                        task_part = task._parts.get()
                    try:
                        task_part.process(thread_index, i1, i2, filter_mask, *blocks)
                    except Exception as e:
                        task.reject(e)
                        run.cancelled = True
                        raise
                    finally:
                        if not isinstance(task._parts, list):
                            task._parts.put(task_part)

        return i2 - i1
