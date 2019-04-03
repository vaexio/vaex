# -*- coding: utf-8 -*-
from __future__ import division, print_function
import threading
import queue
import math
import multiprocessing
import sys
import vaex.utils
import logging
import traceback
import cProfile
import concurrent.futures
import time
import os

thread_count_default = os.environ.get('VAEX_NUM_THREADS', multiprocessing.cpu_count())  # * 2 + 1
thread_count_default = int(thread_count_default)
logger = logging.getLogger("vaex.multithreading")


class ThreadPoolIndex(concurrent.futures.ThreadPoolExecutor):
    """Thread pools that adds a thread index as first argument to the callback passed to map

    This is useful if you keep a piece of memory (like ndgrid) per array
    """

    def __init__(self, max_workers=None, *args, **kwargs):
        if max_workers is None:
            max_workers = thread_count_default
        super(ThreadPoolIndex, self).__init__(max_workers, *args, **kwargs)
        self.lock = threading.Lock()
        self.thread_indices = iter(range(1000000))  # enough threads until 2100?
        self.local = threading.local()
        self.nthreads = self._max_workers

    def map(self, callable, iterator, on_error=None, progress=None, cancel=None, unpack=False):
        progress = progress or (lambda x: True)
        cancelled = False

        def wrapped(*args, **kwargs):
            if not cancelled:
                with self.lock:
                    if not hasattr(self.local, 'index'):
                        self.local.index = next(self.thread_indices)
                if unpack:
                    args = args[0]  # it's passed as a tuple.. not sure why
                callable(self.local.index, *args, **kwargs)
        # convert to list so we can count
        values = list(iterator)
        N = len(values)
        time_last = time.time() - 100
        min_delta_t = 1. / 100  # max 100 per second
        for i, value in enumerate(super(ThreadPoolIndex, self).map(wrapped, values)):
            progress_value = (i + 1) / N
            time_now = time.time()
            if progress_value == 1 or (time_now - time_last) > min_delta_t:
                time_last = time_now
                if progress(progress_value) == False:
                    cancelled = True
            yield value


main_pool = None  # ThreadPoolIndex()


def get_main_pool():
    global main_pool
    if main_pool is None:
        main_pool = ThreadPoolIndex()
    return main_pool
