import asyncio
import base64
from enum import Enum
import hashlib
import io
import logging
import threading
from typing import Dict, Type
import uuid

from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.types import Receive, Scope, Send
from starlette.responses import JSONResponse, Response, StreamingResponse

import vaex
import vaex.cache
import vaex.file.asyncio


logger = logging.getLogger("vaex.server.responses")


def skip(input, bytes=bytes):
    skipped = 0
    for chunk in input:
        if skipped < bytes:
            skip_in_this_chunk = bytes - skipped
            if skip_in_this_chunk >= len(chunk):
                logger.debug("skip chunk")
                # we skip the whole chunk
                skipped += len(chunk)
                continue
            else:
                # the cut is in the chunk
                chunk = chunk[skip_in_this_chunk:]
                skipped += skip_in_this_chunk
        yield chunk


def hash(chunks, hasher=None, callback=lambda x: None):
    hasher = hasher or hashlib.sha256()
    nbytes = 0
    for chunk in chunks:
        nbytes += len(chunk)
        hasher.update(chunk)
        yield chunk
    digest = hasher.digest()
    logger.debug("hex digest: %r (bytesize=%i)", hasher.hexdigest(), nbytes)
    callback(digest)


class DataFrameResponseBase(StreamingResponse):
    parallel = False
    digest_cache = {}  # if vaex cache is off, we store it ourselves

    def __init__(self, df: vaex.dataframe.DataFrameLocal, fingerprint: str = None, method: str = None, byte_offset: int = 0, headers: dict = None, **kwargs):
        self.df = df
        self.send_header_only = method is not None and method.upper() == "HEAD"

        self.fingerprint = fingerprint
        # df can be None if we use it in combination with DelayedResponse
        if self.fingerprint is None and df is not None:
            self.fingerprint = df.fingerprint()
        if self.fingerprint is not None:
            self.fingerprint_digest = fp = f"digest-{self.fingerprint}"
        else:
            self.fingerprint_digest = None
        self.byte_offset = byte_offset
        self.write_stream = vaex.file.asyncio.WriteStream()
        self.write_file_buffered = io.BufferedWriter(self.write_stream, buffer_size=1024 ** 2 * 1)
        headers = headers or dict()
        if self.fingerprint:
            headers["ETag"] = self.fingerprint
        self.disconnected = False  # so we don't store partial digests
        self.has_digest = False
        if self.fingerprint_digest:
            digest = vaex.cache.get(self.fingerprint_digest)
            if not digest:
                digest = self.digest_cache.get(self.fingerprint_digest)
            if digest:
                self.has_digest = True
                logger.debug("Digest found for %r", fingerprint)
                v = base64.encodebytes(digest).decode("ascii").strip()
                headers["Digest"] = f"sha-256={v}"

        super().__init__(self.data_chunks(), headers=headers, **kwargs)

    def data_chunks(self):
        if self.send_header_only:
            return
        chunk_iter = self.raw_data_chunks()

        def cache_digest(digest):
            if self.disconnected:
                return
            logger.debug("digest for %r=%r", self.fingerprint, digest)
            if vaex.cache.is_on():
                vaex.cache.set(self.fingerprint_digest, digest)
            else:
                self.digest_cache[self.fingerprint_digest] = digest

        if not self.has_digest:
            chunk_iter = hash(chunk_iter, callback=cache_digest)
        chunk_iter = skip(chunk_iter, self.byte_offset)
        yield from chunk_iter

    async def listen_for_disconnect(self, receive: Receive) -> None:
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                self.disconnected = True
                self.write_stream.close(force=True)
                break

    def raw_data_chunks(self):
        if self.send_header_only:
            return
        if self.df is None:
            return

        def write():
            with self.write_stream, io.BufferedWriter(self.write_stream, buffer_size=1024 ** 2) as f:
                self.write_dataframe(f)

        writer_thread = threading.Thread(target=write, name="Vaex writer thread", daemon=True)
        writer_thread.start()
        logger.debug("yielding file chunks")
        try:
            for i, chunk in enumerate(self.write_stream):
                logger.debug("chunk: %s", i)
                yield chunk
            logger.debug("joining writer thread")
            writer_thread.join()
            logger.debug("writer thread stopped")
        except BaseException as e:
            if not self.disconnected:
                logger.exception("while yielding chunks")
                raise


class DataFrameResponseArrow(DataFrameResponseBase):
    media_type = "application/vnd.apache.arrow.stream"

    # we need status_code as default parameter, because otherwise the openapi docs don't render nicely
    def __init__(self, content, status_code=200, **kwargs):
        super().__init__(df=content, status_code=status_code, **kwargs)

    def write_dataframe(self, f):
        self.df.export_arrow(f, parallel=self.parallel)


class DataFrameResponseParquet(DataFrameResponseBase):
    # see https://issues.apache.org/jira/browse/PARQUET-1889
    # media_type = 'application/vnd.apache.parquet'
    media_type = "application/octet-stream"

    # we need status_code as default parameter, because otherwise the openapi docs don't render nicely
    def __init__(self, content, status_code=200, **kwargs):
        super().__init__(df=content, status_code=status_code, **kwargs)

    def write_dataframe(self, f):
        self.df.export_parquet(f, parallel=self.parallel)


class DataFrameResponseJson(DataFrameResponseBase):
    media_type = "application/json"

    # we need status_code as default parameter, because otherwise the openapi docs don't render nicely
    def __init__(self, content, status_code=200, **kwargs):
        super().__init__(df=content, status_code=status_code, **kwargs)

    def write_dataframe(self, f):
        self.df.export_json(f, parallel=self.parallel)


class OutputType(str, Enum):
    arrow = "arrow"
    parquet = "parquet"
    json = "json"
    # csv = "csv"


response_map: Dict[str, Type[DataFrameResponseBase]] = {
    OutputType.arrow: DataFrameResponseArrow,
    OutputType.parquet: DataFrameResponseParquet,
    OutputType.json: DataFrameResponseJson,
}


def dataframe_response(df: vaex.dataframe.DataFrame, request: Request, output: OutputType, fingerprint: str = None):
    content_range = request.headers.get("range")
    logging.info("dataframe request headers: %r", request.headers)
    print(request.headers)
    headers = {}
    headers["Accept-Ranges"] = "bytes"
    if content_range and content_range.endswith("-") and content_range.startswith("bytes="):
        skip = len("bytes=")
        content_start = content_range.split("-")[0][skip:]
        range_start = int(content_start)
        headers["Content-Range"] = f"bytes {range_start}-*/*"
    else:
        range_start = 0

    response_class: Type[DataFrameResponseBase] = response_map[output]
    return response_class(df, byte_offset=range_start, method=request.method, fingerprint=fingerprint, headers=headers)


class DelayedResponse(Response):
    jobs = {}

    def __init__(self, content, status_code: int = 200, headers: dict = None, media_type: str = None, background: BackgroundTask = None, response_class: Type[Response] = JSONResponse, **kwargs):
        self.response_class = response_class
        self.content = content
        self.status_code = status_code
        self._headers = headers or {}
        self.media_type = response_class.media_type if media_type is None else media_type
        self.background = background
        self.kwargs = kwargs

        self.job_id = str(uuid.uuid4())
        self.headers["X-Vaex-Job-Id"] = self.job_id
        self.jobs[self.job_id] = vaex.progress.tree()
        self.response = self.response_class(None, status_code=self.status_code, headers=self._headers, media_type=self.media_type, background=self.background, **self.kwargs)
        self.content_done = False

    async def listen_for_disconnect(self, receive: Receive) -> None:
        logger.debug("Listen for disconnect")
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                if not self.content_done:
                    logger.debug("disconnected, so cancel job")
                    self.jobs[self.job_id].cancel()
                break

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        asyncio.create_task(self.listen_for_disconnect(receive=receive))

        async def send_wrapper_star(msg):
            if msg["type"] == "http.response.start":
                return await send(msg)

        # remove content-length header, so we can send chunks
        def is_content_length_header(header):
            return header[0] == b"content-length"

        self.response.raw_headers = [k for k in self.response.raw_headers if not is_content_length_header(k)]
        self.response.body = None
        await self.response(scope, receive, send_wrapper_star)

        try:
            content = await self.content(self.jobs[self.job_id])
        finally:
            self.jobs[self.job_id].exit()
        self.content_done = True
        self.response = self.response_class(content, status_code=self.status_code, headers=self.headers, media_type=self.media_type, background=self.background, **self.kwargs)

        async def send_wrapper_body(msg):
            if msg["type"] == "http.response.body":
                return await send(msg)

        await self.response(scope, receive, send_wrapper_body)
