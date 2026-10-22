import asyncio
import contextlib
import sys


@contextlib.contextmanager
def with_event_loop(event_loop=None):
    try:
        loop_previous = asyncio.get_event_loop()
    except RuntimeError:  # cpython >= 3.14 does not create a loop when none is set
        loop_previous = None
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

def check_ipython():
    IPython = sys.modules.get('IPython')
    if IPython:
        IPython_version = tuple(map(int, IPython.__version__.split('.')[:3]))
        if IPython_version < (7, 0, 0):
            raise RuntimeError(f'You are using IPython {IPython.__version__} while we require 7.0.0, please update IPython')

def check_patch_tornado():
    '''If tornado is imported, add the patched asyncio.Future to its tuple of acceptable Futures'''
    if 'tornado' in sys.modules:
        import tornado.concurrent
        if asyncio.Future not in tornado.concurrent.FUTURES:
            tornado.concurrent.FUTURES = tornado.concurrent.FUTURES + (asyncio.Future, )            


def just_run(coro):
    '''Make the coroutine run, even if there is already an event loop running (using nest_asyncio)'''
    try:
        loop = asyncio.get_event_loop()
        had_loop = True
    except RuntimeError:
        had_loop = False
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    try:
        if had_loop:
            check_ipython()
            try:
                # prefer the maintained fork: nest_asyncio 1.6.0 is broken on cpython >= 3.14 (same API),
                # and it may still be installed alongside, so it cannot be tried first
                import nest_asyncio2 as nest_asyncio
            except ImportError:
                import nest_asyncio
            nest_asyncio.apply()
            check_patch_tornado()
        return loop.run_until_complete(coro)
    finally:
        # NOTE: even when we created the loop above (had_loop == False), we keep
        # the loop set as the thread's default loop rather than clearing it back
        # to None. Clearing would hand every subsequent just_run call a fresh
        # loop, while long-lived transport objects (e.g. the tornado websockets
        # of the vaex server client) hold onto the loop of the call that created
        # them; on cpython >= 3.14, which no longer auto-creates a sticky loop in
        # get_event_loop, that orphans the websocket's reader loop and incoming
        # replies are never processed, hanging requests forever. Keeping the loop
        # set restores cpython <= 3.12 behaviour.
        pass
