import asyncio
import builtins
import logging
from urllib.parse import urlparse
import uuid

import tornado.websocket

import vaex
import vaex.asyncio
import vaex.server.utils
from vaex.server import client
from .executor import Executor

logger = logging.getLogger("vaex.server.tornado_client")


class Client(client.Client):
    def __init__(self, hostname, port=5000, base_path="/", background=False, thread_mover=None, websocket=True, token=None,
                 secure=False,
                 token_trusted=None):
        super().__init__(secure=secure)
        self.hostname = hostname
        self.port = port or (443 if secure else 80)
        self.base_path = base_path if base_path.endswith("/") else (base_path + "/")
        self.token = token
        self.token_trusted = token_trusted
        # jobs maps from uid to tasks
        self.jobs = {}
        self.msg_reply_futures = {}

        self.event_loop_main = asyncio.get_event_loop()
        if self.event_loop_main is None:
            raise RuntimeError('The client cannot work without a running event loop')

        self.executor = Executor(self)
        logger.debug("connect")
        self.connect()
        logger.debug("connected")
        self._check_version()
        self.update()


class ClientWebsocket(Client):
    def _send_and_forget(self, msg, msg_id=None):
        vaex.asyncio.check_patch_tornado()
        if msg_id is None:
            msg_id = str(uuid.uuid4())
        self.msg_reply_futures[msg_id] = asyncio.Future()
        auth = {'token': self.token, 'token-trusted': self.token_trusted}

        msg_encoding = vaex.encoding.Encoding()
        data = vaex.encoding.serialize({'msg_id': msg_id, 'msg': msg, 'auth': auth}, msg_encoding)
        assert self.event_loop_main is asyncio.get_event_loop()

        self.websocket.write_message(data, binary=True)
        return msg_id

    async def _send_async(self, msg, msg_id=None, wait_for_reply=True):
        msg_id = self._send_and_forget(msg, msg_id=msg_id)

        if wait_for_reply:
            reply_msg, reply_encoding = await self.msg_reply_futures[msg_id]
            return reply_msg['result'], reply_encoding

    def _send(self, msg, msg_id=None, wait_for_reply=True, add_promise=None):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._send_async(msg, msg_id))

    def close(self):
        self.websocket.close()

    def _progress(self, fraction, msg_id):
        cancel = False
        for task in self._msg_id_to_tasks.get(msg_id, ()):
            if any(result is False for result in task.signal_progress.emit(fraction)):
                cancel = True
                break
        if cancel:
            for task in self._msg_id_to_tasks.get(msg_id, ()):
                if not hasattr(task, '_server_side_cancel'):
                    task._server_side_cancel = False
                if not task._server_side_cancel:
                    cancel_msg = {'command': 'cancel', 'cancel_msg_id': msg_id}
                    self._send_and_forget(cancel_msg)
                    task.reject(vaex.execution.UserAbort("Progress returned false"))

    def _on_websocket_message(self, websocket_msg):
        if websocket_msg is None:
            return
        logger.debug("websocket msg: %r", websocket_msg)
        try:
            encoding = vaex.encoding.Encoding()
            websocket_msg = vaex.encoding.deserialize(websocket_msg, encoding)
            msg_id, msg = websocket_msg['msg_id'], websocket_msg['msg']
            if 'progress' in msg:
                fraction = msg['progress']

                self._progress(fraction, msg_id)
            elif 'error' in msg:
                exception = RuntimeError("error at server: %r" % msg)
                self.msg_reply_futures[msg_id].set_exception(exception)
            elif 'exception' in msg:
                class_name = msg["exception"]["class"]
                msg = msg["exception"]["msg"]
                if class_name == "UserAbort":
                    cls = vaex.execution.UserAbort
                else:
                    cls = getattr(builtins, class_name)
                exception = cls(msg)
                self.msg_reply_futures[msg_id].set_exception(exception)
            else:
                self.msg_reply_futures[msg_id].set_result((msg, encoding))
        except Exception as e:
            logger.exception("Exception interpreting msg reply: %r", websocket_msg)
            self.msg_reply_futures[msg_id].set_exception(e)

    async def connect_async(self):
        self.websocket = await tornado.websocket.websocket_connect(self._url, on_message_callback=self._on_websocket_message)

    def connect(self):
        vaex.asyncio.just_run(self.connect_async())


def connect(url, **kwargs):
    url = urlparse(url)
    if url.scheme in ["vaex+ws", "ws", "vaex+wss", "wss"]:
        websocket = True
    else:
        websocket = False
    assert url.scheme in ["ws", "http", "vaex+ws", "vaex+http", "vaex+wss", "wss"]
    port = url.port
    base_path = url.path
    hostname = url.hostname
    hostname = vaex.server.utils.hostname_override(hostname)
    if websocket:
        secure = "wss" in url.scheme
        return ClientWebsocket(hostname, base_path=base_path, port=port, secure=secure, **kwargs)
    elif url.scheme == "http":
        raise NotImplementedError("http not implemented")
        # return ClientHttp(hostname, base_path=base_path, port=port, **kwargs)
