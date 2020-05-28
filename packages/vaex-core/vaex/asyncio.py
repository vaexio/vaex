import asyncio
import sys


def check_ipython():
    IPython = sys.modules.get('IPython')
    if IPython:
        IPython_version = tuple(map(int, IPython.__version__.split('.')))
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
            import nest_asyncio
            nest_asyncio.apply()
            check_patch_tornado()
        return loop.run_until_complete(coro)
    finally:
        if not had_loop:  # remove loop if we did not have one
            asyncio.set_event_loop(None)
