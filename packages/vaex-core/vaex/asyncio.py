import asyncio
import sys


def check_patch_tornado():
    '''If tornado is important, add the patched asyncio.Future to its tuple of acceptable Futures'''
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
        loop = None
        had_loop = False
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    try:
        if had_loop:
            import nest_asyncio
            nest_asyncio.apply()
            check_patch_tornado()
        return loop.run_until_complete(coro)
    finally:
        if not had_loop:  # remove loop if we did not have one
            asyncio.set_event_loop(None)
