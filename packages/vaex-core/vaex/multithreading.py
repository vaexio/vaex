# -*- coding: utf-8 -*-
from __future__ import division, print_function
import asyncio
import threading
import multiprocessing
from warnings import warn
import vaex.utils
import logging
import concurrent.futures
import time
import vaex.settings

from .itertools import buffer

logger = logging.getLogger("vaex.multithreading")

main_pool = None
main_io_pool = None


thread_pools = {}


def get_thread_pool(nthreads):
    if nthreads not in thread_pools:
        thread_pools[nthreads] = ThreadPoolIndex(nthreads)
    return thread_pools[nthreads]


def get_main_pool():
    global main_pool
    if main_pool is None:
        main_pool = ThreadPoolIndex()
        thread_pools[main_pool.nthreads] = main_pool
    return main_pool


def get_main_io_pool():
    global main_io_pool
    if main_io_pool is None:
        main_io_pool = concurrent.futures.ThreadPoolExecutor(max_workers=vaex.settings.main.thread_count_io)
    return main_io_pool





class ThreadPoolIndex(concurrent.futures.ThreadPoolExecutor):
    """Thread pools that adds a thread index as first argument to the callback passed to map

    This is useful if you keep a piece of memory (like ndgrid) per array
    """

    def __init__(self, max_workers=None, *args, **kwargs):
        if max_workers is None:
            max_workers = vaex.settings.main.thread_count
        super(ThreadPoolIndex, self).__init__(max_workers, *args, **kwargs)
        self.lock = threading.Lock()
        self.thread_indices = iter(range(1000000))  # enough threads until 2100?
        self.local = threading.local()
        self.nthreads = self._max_workers
        self._debug_sleep = 0

    def map(self, callable, iterator, count, on_error=None, progress=None, cancel=None, unpack=False, use_async=False, **kwargs_extra):
        progress = progress or (lambda x: True)
        cancelled = False

        def wrapped(*args, **kwargs):
            if not cancelled:
                if self.nthreads == 1:
                    self.local.index = 0
                with self.lock:
                    if not hasattr(self.local, 'index'):
                        self.local.index = next(self.thread_indices)
                if unpack:
                    args = args[0]  # it's passed as a tuple.. not sure why
                if self._debug_sleep:
                    # print("SLEEP", self._debug_sleep)
                    time.sleep(self._debug_sleep)
                return callable(self.local.index, *args, **kwargs, **kwargs_extra)
        time_last = time.time() - 100
        min_delta_t = 1. / 10  # max 10 per second
        # we don't want to keep consuming the chunk iterator when we cancel
        chunk_iterator = iterator
        def cancellable_iter():
            for value in chunk_iterator:
                yield value
                if cancelled:
                    break
        if self.nthreads == 1:  # when using 1 thread, it makes debugging easier (better stacktrace)
            if use_async:
                iterator = self._map_async(wrapped, cancellable_iter())
            else:
                iterator = self._map(wrapped, cancellable_iter())
        else:
            if use_async:
                loop = asyncio.get_event_loop()
                iterator = (loop.run_in_executor(self, lambda value=value: wrapped(value)) for value in cancellable_iter())
            else:
                iterator = super(ThreadPoolIndex, self).map(wrapped, cancellable_iter())
        total = 0
        iterator = iter(buffer(iterator, self._max_workers + 3))
        try:
            for value in iterator:
                if use_async:
                    value = yield value
                else:
                    yield value
                if value != None:
                    total += value
                progress_value = (total) / count
                time_now = time.time()
                if progress_value == 1 or (time_now - time_last) > min_delta_t:
                    time_last = time_now
                    if progress(progress_value) is False:
                        cancelled = True
                        cancel()
                        break
        finally:
            if not cancelled:
                cancelled = True
            # consume the rest of the iterators and await them to avoid un-awaited exceptions, which trigger a
            # 'Future exception was never retrieved' printout
            # TODO: since we don't use async any more, I think we can get rid of this
            for value in iterator:
                try:
                    pass
                except:
                    pass

    def _map(self, callable, iterator):
        for i in iterator:
            yield callable(i)

    def _map_async(self, callable, iterator):
        for i in iterator:
            future = asyncio.Future()
            future.set_result(callable(i))
            yield future
