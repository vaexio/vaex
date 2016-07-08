__author__ = 'maartenbreddels'
import threading
import time
class Job(object):
    def __init__(self, f, delay=0.0, **kwargs):
        self.cancelled = False
        self.f = f
        self.delay = delay
        self.kwargs = kwargs
        self.thread = threading.Thread(target=self)
        self.result = None
        self.exception = False

    def schedule(self):
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


