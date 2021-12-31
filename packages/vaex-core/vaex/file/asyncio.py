import io
import asyncio
import threading
import queue
import logging

from starlette import concurrency


logger = logging.getLogger('vaex.file.async')


class WriteStream(io.RawIOBase):
    '''File like object that has a sync write API, and a generator as consumer.
    
    This is useful for letting 1 thread write to this object, while another thread
    in consuming the byte chunks via an iterator, or async iterator.


    To let the writer thread stop, simply close the file.
    '''
    def __init__(self, queue_size=3):
        self.writes = []
        self.queue = queue.Queue(queue_size)
        self.closed = False
        self.pos = 0
        self.exception = None

    def __iter__(self):
        yield from self.chunks()

    async def __aiter__(self):    
        import starlette.concurrency
        async for item in starlette.concurrency.iterate_in_threadpool(self.chunks()):
            yield item

    def __enter__(self, *args):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        self.exception = value


    def read(self, *args):
        return b''

    def seek(self, *args):
        if self.closed:
            return ValueError('stream closed')
        return 0

    def readinto(self, *args):
        return b''

    def tell(self):
        if self.closed:
            return ValueError('stream closed')
        return self.pos

    def flush(self):
        if self.closed:
            return ValueError('stream closed')
        pass

    def writable(self):
        return True

    def readable(self):
        return False

    def seekable(self):
        return False
    
    def write(self, b):
        if self.closed:
            return ValueError('stream closed')
        logger.debug('write: %r', len(b))
        buffer = memoryview(b)
        # print("--", bytes(buffer))
        self.pos += len(buffer)
        # we need to copy it seems (otherwise if we use a buffered writer, it will reuse the memory)
        self.queue.put(bytes(buffer))
        # print(len(buffer))
        return len(buffer)

    def close(self, force=False):
        '''Note that close can block due to the queue, using force=True the queue will be cleared'''
        logger.debug('closing stream, putting None element in queue to stop chunk yielding')
        # make sure nobody will add new items to the queue, at max 1 (the current write)
        self.closed = True
        if force:
            self._force_put(None)
        else:
            self.queue.put(None)


    def closed(self):
        return self.closed

    def chunks(self):
        logger.debug('yielding chunks')
        while True:
            logger.debug('waiting for chunk')
            item = self.queue.get()
            if item is None:
                logger.debug('stop yielding, file closed')
                break
            # if isinstance(item, BaseException):
            #     logger.debug('stop yielding, exception occured')
            #     raise item
            yield item
        if self.exception:
            raise self.exception

    def getvalue(self):
        return b''.join(bytes(k) for k in self.chunks())

    # def stop(self, exception):
    #     '''Empty the queue, and put the exception on the queue, and close the file.

    #     If we don't empty the queue, this might block
    #     '''
    #     self._force_put(exception)

    #     self.closed = True

    def _force_put(self, item):
        # keep trying to put an exception on the queue
        # we dont want to do this blocking, since a producer
        # thread might fill this up.
        done = False
        while not done:
            logger.debug('clearing queue')
            while not self.queue.empty():
                self.queue.get(block=False)
            try:
                logger.debug('put %r on queue', item)
                self.queue.put(item, block=False)
                done = True
            except queue.Full:
                logger.debug('retry putting exception on queue, because it was filled')
