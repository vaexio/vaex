__author__ = 'maartenbreddels'
import threading
import time
import queue
import logging
import collections

logger = logging.getLogger("vaex.ext.common")

import time
import functools

def get_ioloop():
    import IPython, zmq
    ipython = IPython.get_ipython()
    if ipython and hasattr(ipython, 'kernel'):
        return zmq.eventloop.ioloop.IOLoop.instance()


def debounced(delay_seconds=0.5, method=False):
    def wrapped(f):
        counters = collections.defaultdict(int)

        @functools.wraps(f)
        def execute(*args, **kwargs):
            if method: # if it is a method, we want to have a counter per instance
                key = args[0]
            else:
                key = None
            counters[key] += 1
            def debounced_execute(counter=counters[key]):
                if counter == counters[key]: # only execute if the counter wasn't changed in the meantime
                    f(*args, **kwargs)
            ioloop = get_ioloop()

            def thread_safe():
                ioloop.add_timeout(time.time() + delay_seconds, debounced_execute)

            ioloop.add_callback(thread_safe)
        return execute
    return wrapped

class Job(object):
    def __init__(self, f, delay=0.0, **kwargs):
        self.cancelled = False
        self.f = f
        self.delay = delay
        self.kwargs = kwargs
        #self.thread = threading.Thread(target=self)
        self.result = None
        self.exception = False
        #self.finished = False

    def ____schedule(self):
        self.thread.start()

    def cancel(self):
        self.cancelled = True

    def __call__(self):
        time.sleep(self.delay)
        if not self.cancelled:
            try:
                self.result = self.f(self, **self.kwargs)
            except:
                self.exception = True
                raise



class SingleJobThread(threading.Thread):
    def __init__(self):
        self.job_queues = collections.defaultdict(queue.Queue)
        self.job_any = threading.Semaphore()
        self.start()

    def submit(self, job, queue_name):
        self.job_queues[queue_name].put(job)
        self.job_any.release()

    def run(self):
        try:
            logger.debug("vaex job thread is running")
            while True:
                self.job_any.acquire() # ok, there must be something to do...
                for key in self.job_queues.keys():
                    queue = self.job_queues[key]
                    if not queue.empty():
                        logging.debug("handling queue %s", key)
                        job = queue.get()
                        job()
        finally:
            logger.error("vaex job thread quits")

