import asyncio
import contextlib


@contextlib.contextmanager
def with_event_loop(event_loop=None):
    loop_previous = asyncio.get_event_loop()
    if event_loop is None:
        loop_new = asyncio.new_event_loop()
    else:
        loop_new = event_loop
    asyncio.set_event_loop(loop_new)
    # private API, but present from CPython 3.3ish-3.10+
    asyncio.events._set_running_loop(None)
    try:
        yield
    finally:
        asyncio.set_event_loop(loop_previous)
