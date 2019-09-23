from __future__ import division, print_function
import functools
import vaex.vaexfast
import numpy as np
from vaex.utils import Timer
import vaex.events
from . import multithreading
import time
import math
# import vaex.ui.expressions as expr
from functools import reduce
import threading
import queue
import math
import multiprocessing
import sys
import collections
import vaex.multithreading
import logging

__author__ = 'breddels'

buffer_size_default = 1024 * 1024  # TODO: this should not be fixed, larger means faster but also large memory usage
# buffer_size_default = 1e4

lock = threading.Lock()

logger = logging.getLogger("vaex.execution")


thread_count_default = multiprocessing.cpu_count()


class UserAbort(Exception):
    def __init__(self, reason):
        super(UserAbort, self).__init__(reason)


class Column(collections.namedtuple('Column', ['df', 'expression'])):
    def needs_copy(self):
        return not \
            (self.expression in self.df.column_names and not
             isinstance(self.df.columns[self.expression], vaex.df._ColumnConcatenatedLazy) and
             self.df.columns[self.expression].dtype.type == np.float64 and
             self.df.columns[self.expression].strides[0] == 8 and
             self.expression not in self.df.virtual_columns)
        # and False:


class Job(object):
    def __init__(self, task, order):
        self.task = task
        self.order = order


# mutex for numexpr, it is not thread save
ne_lock = threading.Lock()


class Executor(object):
    def __init__(self, thread_pool=None, buffer_size=None, thread_mover=None, zigzag=True):
        self.thread_pool = thread_pool or vaex.multithreading.ThreadPoolIndex()
        self.task_queue = []
        self.buffer_size = buffer_size or buffer_size_default
        self.signal_begin = vaex.events.Signal("begin")
        self.signal_progress = vaex.events.Signal("progress")
        self.signal_end = vaex.events.Signal("end")
        self.signal_cancel = vaex.events.Signal("cancel")
        self.thread_mover = thread_mover or (lambda fn, *args, **kwargs: fn(*args, **kwargs))
        self._is_executing = False
        self.lock = threading.Lock()
        self.thread = None
        self.passes = 0  # how many times we passed over the data
        self.zig = True # zig or zag
        self.zigzag = zigzag

    def schedule(self, task):
        self.task_queue.append(task)
        logger.info("task added, queue: %r", self.task_queue)
        return task

    def run(self, task):
        # with self.lock:
        if 1:
            logger.debug("added task: %r", task)
            previous_queue = self.task_queue
            try:
                self.task_queue = [task]
                self.execute()
            finally:
                self.task_queue = previous_queue
            return task._value

    def execute_threaded(self):
        if self.thread is None:
            logger.info("starting thread for executor")
            self.thread = threading.Thread(target=self._execute_in_thread)
            self.thread.start()
            self.queue_semaphore = threading.Semaphore()
        logger.info("sending thread a msg that it can execute")
        self.queue_semaphore.release()

    def _execute_in_thread(self):
        while True:
            try:
                logger.info("waiting for jobs")
                self.queue_semaphore.acquire()
                logger.info("got jobs")
                if self.task_queue:
                    logger.info("executing tasks in thread: %r", self.task_queue)
                    self.execute()
                else:
                    logger.info("empty task queue")
            except:
                import traceback
                traceback.print_exc()
                logger.error("exception occured in thread")

    def execute(self):
        logger.debug("starting with execute")
        if self._is_executing:
            logger.debug("nested execute call")
            # this situation may happen since in this methods, via a callback (to update a progressbar) we enter
            # Qt's eventloop, which may execute code that will call execute again
            # as long as that code is using delay tasks (i.e. promises) we can simple return here, since after
            # the execute is almost finished, any new tasks added to the task_queue will get executing
            return
        # u 'column' is uniquely identified by a tuple of (df, expression)
        self._is_executing = True
        try:
            t0 = time.time()
            task_queue_all = list(self.task_queue)
            if not task_queue_all:
                logger.info("only had cancelled tasks")
            logger.info("clearing queue")
            # self.task_queue = [] # Ok, this was stupid.. in the meantime there may have been new tasks, instead, remove the ones we copied
            for task in task_queue_all:
                logger.info("remove from queue: %r", task)
                self.task_queue.remove(task)
            logger.info("left in queue: %r", self.task_queue)
            task_queue_all = [task for task in task_queue_all if not task.cancelled]
            # logger.debug("executing queue: %r" % (task_queue_all))

            # for task in self.task_queue:
            # print task, task.expressions_all
            dfs = set(task.df for task in task_queue_all)
            cancelled = [False]

            def cancel():
                logger.debug("cancelling")
                self.signal_cancel.emit()
                cancelled[0] = True
            try:
                # process tasks per df
                self.signal_begin.emit()
                for df in dfs:
                    self.passes += 1
                    task_queue = [task for task in task_queue_all if task.df == df]
                    expressions = list(set(expression for task in task_queue for expression in task.expressions_all))

                    for task in task_queue:
                        task._results = []
                        task.signal_progress.emit(0)
                    block_scopes = [df._block_scope(0, self.buffer_size) for i in range(self.thread_pool.nthreads)]

                    def process(thread_index, i1, i2):
                        if not cancelled[0]:
                            block_scope = block_scopes[thread_index]
                            block_scope.move(i1, i2)
                            # with ne_lock:
                            block_dict = {expression: block_scope.evaluate(expression) for expression in expressions}
                            for task in task_queue:
                                blocks = [block_dict[expression] for expression in task.expressions_all]
                                if not cancelled[0]:
                                    task._results.append(task.map(thread_index, i1, i2, *blocks))
                                # don't call directly, since ui's don't like being updated from a different thread
                                # self.thread_mover(task.signal_progress, float(i2)/length)
# time.sleep(0.1)

                    length = df.active_length()
                    parts = vaex.utils.subdivide(length, max_length=self.buffer_size)
                    if not self.zig:
                        parts = list(parts)[::-1]
                    if self.zigzag:
                        self.zig = not self.zig
                    for element in self.thread_pool.map(process, parts,
                                                        progress=lambda p: all(self.signal_progress.emit(p)) and
                                                        all([all(task.signal_progress.emit(p)) for task in task_queue]),
                                                        cancel=cancel, unpack=True):
                        pass  # just eat all element
                    self._is_executing = False
            except:
                # on any error we flush the task queue
                self.signal_cancel.emit()
                logger.exception("error in task, flush task queue")
                raise
            logger.debug("executing took %r seconds" % (time.time() - t0))
            # while processing the self.task_queue, new elements will be added to it, so copy it
            logger.debug("cancelled: %r", cancelled)
            if cancelled[0]:
                logger.debug("execution aborted")
                task_queue = task_queue_all
                for task in task_queue:
                    # task._result = task.reduce(task._results)
                    # task.reject(UserAbort("cancelled"))
                    # remove references
                    task._result = None
                    task._results = None
            else:
                task_queue = task_queue_all
                for task in task_queue:
                    logger.debug("fulfill task: %r", task)
                    if not task.cancelled:
                        task._result = task.reduce(task._results)
                        task.fulfill(task._result)
                        # remove references
                    task._result = None
                    task._results = None
                self.signal_end.emit()
                # if new tasks were added as a result of this, execute them immediately
                # TODO: we may want to include infinite recursion protection
                self._is_executing = False
                if len(self.task_queue) > 0:
                    logger.debug("task queue not empty.. start over!")
                    self.execute()
        finally:
            self._is_executing = False

# default_executor = None
