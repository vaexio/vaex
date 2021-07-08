from __future__ import absolute_import
__author__ = 'maartenbreddels'

import asyncio
import tornado.ioloop
import tornado.web
import tornado.httpserver
import tornado.websocket
import tornado.auth
import tornado.gen
import threading
import logging
import vaex as vx
import vaex.utils
import argparse
import vaex.events
import vaex.execution
import vaex.multithreading
import tornado.escape
from cachetools import LRUCache
import sys

from vaex.encoding import serialize, deserialize, Encoding
import vaex.server.service
import vaex.asyncio
import vaex.server.dataframe
import vaex.core._version
import vaex.server._version
import vaex.server.dataframe

from .utils import exception, error
import vaex.server.websocket


logger = logging.getLogger("vaex.webserver.tornado")


class WebSocketHandler(tornado.websocket.WebSocketHandler):
    def initialize(self, service, webserver, submit_threaded, cache, cache_selection, datasets=None):
        self.service = service
        self.webserver = webserver
        self.submit_threaded = submit_threaded
        self.handler = vaex.server.websocket.WebSocketHandler(self.send, self.service, token=self.webserver.token, token_trusted=self.webserver.token_trusted)

    async def send(self, value):
        await self.write_message(value, binary=True)

    async def on_message(self, websocket_msg):
        # Tornado does not receive messages before the current is finished, this
        # avoids this limitation of tornado, so we can send progress/cancel information
        logger.debug("get msg: %r", websocket_msg)
        asyncio.create_task(self._on_message(websocket_msg))

    async def _on_message(self, websocket_msg):
        logger.debug("handle msg: %r", websocket_msg)
        await self.handler.handle_message(websocket_msg)

    def on_close(self):
        logger.debug("WebSocket closed")


MB = 1024**2
GB = MB * 1024


class WebServer(threading.Thread):
    def __init__(self, address="127.0.0.1", port=9000, webserver_thread_count=2, cache_byte_size=500 * MB,
                 token=None, token_trusted=None, base_url=None,
                 cache_selection_byte_size=500 * MB, datasets=[], compress=True, development=False, threads_per_job=4):
        threading.Thread.__init__(self)
        self._test_latency = None  # for testing purposes
        self.setDaemon(True)
        self.address = address
        self.port = port
        self.started = threading.Event()
        self.service = None
        self.webserver_thread_count = webserver_thread_count
        self.threads_per_job = threads_per_job
        self.base_url = base_url
        if self.base_url is None:
            if self.port == 80:
                self.base_url = f'{self.address}'
            else:
                self.base_url = f'{self.address}:{self.port}'

        self.service_bare = vaex.server.service.Service({})
        self.service_threaded = vaex.server.service.AsyncThreadedService(self.service_bare, self.webserver_thread_count,
                                                                         self.threads_per_job)
        self.service = self.service_threaded
        self.set_datasets(datasets)
        self.token = token
        self.token_trusted = token_trusted

        self.cache = LRUCache(cache_byte_size, getsizeof=sys.getsizeof)
        self.cache_selection = LRUCache(cache_selection_byte_size, getsizeof=sys.getsizeof)

        self.options = dict(webserver=self, service=self.service, datasets=datasets, submit_threaded=self.submit_threaded, cache=self.cache,
                            cache_selection=self.cache_selection)

        # tornado.web.GZipContentEncoding.MIN_LENGTH = 1
        tornado.web.GZipContentEncoding.CONTENT_TYPES.add("application/octet-stream")
        self.application = tornado.web.Application([
            (r"/websocket", WebSocketHandler, self.options),
        ], compress_response=compress, debug=development)
        logger.debug("compression set to %r", compress)
        logger.debug("cache size set to %s", vaex.utils.filesize_format(cache_byte_size))
        logger.debug("thread count set to %r", self.webserver_thread_count)

    def set_datasets(self, datasets):
        self.datasets = list(datasets)
        self.datasets_map = dict([(ds.name, ds) for ds in self.datasets])
        self.service_bare.df_map = self.datasets_map

    def submit_threaded(self, callable, *args, **kwargs):
        def execute():
            value = callable(*args, **kwargs)
            return value
        future = self.thread_pool.submit(execute)
        return future

    def serve(self):
        self.mainloop()

    def serve_threaded(self):
        logger.debug("start thread")
        if tornado.version_info[0] >= 5:
            from tornado.platform.asyncio import AnyThreadEventLoopPolicy
            # see https://github.com/tornadoweb/tornado/issues/2308
            asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())
        self.start()
        logger.debug("wait for thread to run")
        self.started.wait()
        logger.debug("make tornado io loop the main thread's current")
        # this will make the main thread use this ioloop as current
        # self.ioloop.make_current()

    def run(self):
        self.mainloop()

    def mainloop(self):
        logger.info("serving at http://%s:%d" % (self.address, self.port))
        self.ioloop = tornado.ioloop.IOLoop.current()
        # listen doesn't return a server object, which we need to close
        # self.application.listen(self.port, address=self.address)
        from tornado.httpserver import HTTPServer
        self.server = HTTPServer(self.application)
        try:
            self.server.listen(self.port, self.address)
        except:  # noqa
            self.started.set()
            raise
        self.started.set()
        if tornado.version_info[0] >= 5:
            from tornado.platform.asyncio import AnyThreadEventLoopPolicy
            # see https://github.com/tornadoweb/tornado/issues/2308
            asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())

        try:
            self.ioloop.start()
        except RuntimeError:
            pass  # TODO: not sure why this happens in the unittest

    def stop_serving(self):
        logger.debug("stop server")
        self.server.stop()
        logger.debug("stop io loop")
        # self.ioloop.stop()
        self.service.stop()
        # for thread_pool in self.thread_pools:
        #     thread_pool.shutdown()


defaults_yaml = """
address: 0.0.0.0
port: 9000
filenames: []
verbose: 2
cache: 500000000
compress: true
filename: []
development: False
threads_per_job: 4
"""


def main(argv, WebServer=WebServer):

    parser = argparse.ArgumentParser(argv[0])
    parser.add_argument("filename", help="filename for dataset", nargs='*')
    parser.add_argument("--address", help="address to bind the server to (default: %(default)s)", default="0.0.0.0")
    parser.add_argument("--base-url", help="External base url (default is <address>:port)", default=None)
    parser.add_argument("--port", help="port to listen on (default: %(default)s)", type=int, default=9000)
    parser.add_argument('--verbose', '-v', action='count', default=2)
    parser.add_argument('--cache', help="cache size in bytes for requests, set to zero to disable (default: %(default)s)", type=int, default=500000000)
    parser.add_argument('--compress', help="compress larger replies (default: %(default)s)", default=True, action='store_true')
    parser.add_argument('--no-compress', dest="compress", action='store_false')
    parser.add_argument('--development', default=False, action='store_true', help="enable development features (auto reloading)")
    parser.add_argument('--add-example', default=False, action='store_true', help="add the example dataset")
    parser.add_argument('--token', default=None, help="optionally protect server access by a token")
    parser.add_argument('--token-trusted', default=None, help="when using this token, the server allows more deserialization (e.g. pickled function)")
    parser.add_argument('--threads-per-job', default=4, type=int, help="threads per job (default: %(default)s)")
    # config = layeredconfig.LayeredConfig(defaults, env, layeredconfig.Commandline(parser=parser, commandline=argv[1:]))
    config = parser.parse_args(argv[1:])

    verbosity = ["ERROR", "WARNING", "INFO", "DEBUG"]
    logging.getLogger("vaex").setLevel(verbosity[config.verbose])
    # import vaex
    # vaex.set_log_level_debug()

    filenames = []
    filenames = config.filename
    datasets = []
    for filename in filenames:
        df = vx.open(filename)
        if df is None:
            print("error opening file: %r" % filename)
        else:
            datasets.append(df)
    if config.add_example:
        df_example = vaex.example()
        df_example.name = "example"
        datasets.append(df_example)

    datasets = datasets or [vx.example()]

    # datasets = [ds for ds in datasets if ds is not None]
    logger.info("datasets:")
    for dataset in datasets:
        logger.info("\thttp://%s:%d/%s or ws://%s:%d/%s", config.address, config.port, dataset.name, config.address, config.port, dataset.name)
    server = WebServer(datasets=datasets, address=config.address, base_url=config.base_url, port=config.port, cache_byte_size=config.cache,
                       token=config.token, token_trusted=config.token_trusted,
                       compress=config.compress, development=config.development,
                       threads_per_job=config.threads_per_job)
    server.serve()


if __name__ == "__main__":
    main(sys.argv)
