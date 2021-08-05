import asyncio
import logging

import vaex
from vaex.encoding import serialize, deserialize, Encoding
from .utils import exception


TEST_LATENCY = 0
logger = logging.getLogger("vaex.webserver.websocket")


class WebSocketHandler:
    def __init__(self, send, service, token=None, token_trusted=None):
        self.send = send
        self.service = service
        self.token = token
        self.token_trusted = token_trusted
        self.trusted = False
        self._msg_id_to_tasks = {}
        self.tasks = []

    async def handle_message(self, websocket_msg):
        try:
            await self._handle_message(websocket_msg)
        except BaseException as e:
            encoding = Encoding()
            websocket_msg = deserialize(websocket_msg, encoding)
            msg_id = websocket_msg['msg_id']
            msg = exception(e)
            await self.write_json({'msg_id': msg_id, 'msg': msg})
            logger.exception("unhandled exception")

    async def _handle_message(self, websocket_msg):
        if TEST_LATENCY:
            await asyncio.sleep(TEST_LATENCY)
        msg_id = 'invalid'
        encoding = Encoding()
        try:
            websocket_msg = deserialize(websocket_msg, encoding)
            logger.debug("websocket message: %s", websocket_msg)
            msg_id, msg, auth = websocket_msg['msg_id'], websocket_msg['msg'], websocket_msg['auth']

            token = auth['token']  # are we even allowed to execute?
            token_trusted = auth['token-trusted']  # do we trust arbitrary code execution?
            trusted = token_trusted == self.token_trusted and token_trusted

            if not ((token == self.token) or
                    (self.token_trusted and token_trusted == self.token_trusted)):
                raise ValueError('No token provided, not authorized')

            last_progress = None
            ioloop = asyncio.get_event_loop()
            progress_futures = []

            def progress(f):
                nonlocal last_progress

                async def send_progress():
                    vaex.asyncio.check_patch_tornado()  # during testing asyncio might be patched
                    nonlocal last_progress
                    logger.debug("progress: %r", f)
                    last_progress = f
                    # TODO: create task?
                    return await self.write_json({'msg_id': msg_id, 'msg': {'progress': f}})
                # emit when it's the first time (None), at least 0.05 sec lasted, or and the end
                # but never send old or same values
                if (last_progress is None or (f - last_progress) > 0.05 or f == 1.0) and (last_progress is None or f > last_progress):
                    def wrapper():
                        progress_futures.append(asyncio.create_task(send_progress()))
                    ioloop.call_soon_threadsafe(wrapper)
                return True

            command = msg['command']
            if command == 'list':
                result = self.service.list()
                await self.write_json({'msg_id': msg_id, 'msg': {'result': result}})
            elif command == 'versions':
                result = {'vaex.core': vaex.core._version.__version_tuple__, 'vaex.server': vaex.server._version.__version_tuple__}
                await self.write_json({'msg_id': msg_id, 'msg': {'result': result}})
            elif command == 'execute':
                df = self.service[msg['df']].copy()
                df.state_set(msg['state'], use_active_range=True, trusted=trusted)
                tasks = encoding.decode_list('task', msg['tasks'], df=df)
                self._msg_id_to_tasks[msg_id] = tasks  # keep a reference for cancelling
                try:
                    results = await self.service.execute(df, tasks, progress=progress)
                finally:
                    del self._msg_id_to_tasks[msg_id]
                await asyncio.gather(*progress_futures)
                # make sure the final progress value is send, and also old values are not send
                last_progress = 1.0
                await self.write_json({'msg_id': msg_id, 'msg': {'progress': 1.0}})
                encoding = Encoding()
                results = encoding.encode_list('vaex-task-result', results)
                await self.write_json({'msg_id': msg_id, 'msg': {'result': results}}, encoding)
            elif command == 'cancel':
                try:
                    tasks = self._msg_id_to_tasks[msg['cancel_msg_id']]
                except KeyError:
                    pass  # already done, or cancelled
                else:
                    for task in tasks:
                        task.cancel()
            elif command == 'call-dataframe':
                df = self.service[msg['df']].copy()
                df.state_set(msg['state'], use_active_range=True, trusted=trusted)
                # TODO: yield
                if msg['method'] not in vaex.server.dataframe.allowed_method_names:
                    raise NotImplementedError("Method is not rmi invokable")
                results = self.service._rmi(df, msg['method'], msg['args'], msg['kwargs'])
                encoding = Encoding()
                if msg['method'] == "_evaluate_implementation":
                    results = encoding.encode('vaex-evaluate-result', results)
                else:
                    results = encoding.encode('vaex-rmi-result', results)
                await self.write_json({'msg_id': msg_id, 'msg': {'result': results}}, encoding)
            else:
                raise ValueError(f'Unknown command: {command}')

        except Exception as e:
            logger.exception("Exception while handling msg")
            msg = exception(e)
            await self.write_json({'msg_id': msg_id, 'msg': msg})

    async def write_json(self, msg, encoding=None):
        encoding = encoding or Encoding()
        logger.debug("writing json: %r", msg)
        try:
            return await self.send(serialize(msg, encoding))
        except:  # noqa
            logger.exception('Failed to write: %s', msg)

    def on_close(self):
        logger.debug("WebSocket closed")
