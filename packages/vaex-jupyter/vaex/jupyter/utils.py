import collections
import time
import functools
import ipywidgets as widgets
from IPython.display import display, clear_output
import IPython
import asyncio


ipython = IPython.get_ipython()
_test_delay = None  # speed up testing
debounced_execute_queue = []
_debounced_futures = []
_debounced_futures_skip = []
debounce_enabled = True  # can be useful to turn off for debugging purposes
_is_gatherering = False
_call_soon_functions = []


def get_ioloop():
    return asyncio.get_event_loop()
    import zmq
    if ipython and hasattr(ipython, 'kernel'):
        return zmq.eventloop.ioloop.IOLoop.instance()


def flush(recursive_counts=-1, ignore_exceptions=False, all=False):
    """Run all non-executed debounced functions.

    If execution of debounced calls lead to scheduling of new calls, they will be recursively executed,
    with a limit or recursive_counts calls. recursive_counts=-1 means infinite.
    """
    ioloop = get_ioloop()
    if ioloop:
        ioloop.run_until_complete(gather(recursive_counts, ignore_exceptions=ignore_exceptions, all=all))
        return

    queue = debounced_execute_queue.copy()
    for f in queue:
        f()
        debounced_execute_queue.remove(f)
    if debounced_execute_queue and recursive_counts != 0:
        flush(recursive_counts-1, ignore_exceptions=ignore_exceptions, all=all)


_debounced_flush = flush  # old alias, TODO: remove


async def gather(recursive_counts=-1, ignore_exceptions=False, all=False):
    """Gather all debounced function result, useful for waiting till all schedules operations are executed
    """
    global _is_gatherering
    was_already_gatherering = _is_gatherering  # store old status
    _is_gatherering = True
    try:
        if all:
            await asyncio.gather(*_debounced_futures + _debounced_futures_skip)
        else:
            await asyncio.gather(*_debounced_futures)
    except:  # noqa
        if not ignore_exceptions:
            raise
    if _debounced_futures and recursive_counts != 0:
        await gather(recursive_counts-1, ignore_exceptions=ignore_exceptions, all=all)
    _is_gatherering = was_already_gatherering  # restore old status


def kernel_tick():
    """Execute a single command, to allow events from the frontend to get to the kernel during execution."""
    # For instance zoom events which should cancel vaex executions.
    # We should not execute more command during gathering, since that can execute the
    # next notebook cell. Maybe take a look at https://github.com/kafonek/ipython_blocking
    # for inspiration how to
    if ipython is not None and not _is_gatherering:
        ipython.kernel.do_one_iteration()
    # if _call_soon_functions:
    #     f = _call_soon_functions.pop(0)
    #     f()


def call_once(f):
    called = False

    def wrapper(*args, **kwargs):
        nonlocal called
        if not called:
            called = True
            f(*args, **kwargs)
    return wrapper


def call_soon(f):
    f_once = call_once(f)
    # whoever calls it first
    ioloop = get_ioloop()
    if ioloop:
        ioloop.call_soon_threadsafe(f_once)
    _call_soon_functions.append(f_once)


class _debounced_callable:
    def __init__(self, f, delay_seconds=0.5, skip_gather=False, on_error=None, obj=None, reentrant=False):
        self.f = f
        self.delay_seconds = delay_seconds
        self.skip_gather = skip_gather
        self.on_error = on_error
        self.counter = 0
        self.result_future = None  # we create this lazily, since the ioloop may not yet be present
        self.last_result_future = None
        self.previous_result_future = None
        self._pre_hook_future = None  # same
        self.name = None
        # for methods we also have the object
        self.obj = obj
        self.reentrant = reentrant
        self.method_cache = {}
        self.last_task = None  # for canceling

    def copy(self, obj):
        return _debounced_callable(self.f, self.delay_seconds, skip_gather=self.skip_gather, on_error=self.on_error, obj=obj, reentrant=self.reentrant)

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype):
        # for classes, this acts as a property, to get per instance debouncing
        if obj is None:
            return self
        else:
            # for methods, we create a copy (with and extra obj attr)
            # so that the debouncing happens per instance
            name = f"_{self.name}"
            if not hasattr(obj, name):
                setattr(obj, name, self.copy(obj))
            return getattr(obj, name)

    @property
    def pre_hook_future(self):
        if self._pre_hook_future is None:
            self._pre_hook_future = asyncio.Future()
        return self._pre_hook_future

    def __await__(self):
        return self._await_last_call().__await__()

    async def _await_last_call(self, previous=False):
        if previous:
            return await self.previous_result_future
        else:
            return await self.last_result_future

    def cancel(self):
        self.last_task.cancel()

    async def cancel_and_wait(self):
        self.last_task.cancel()
        try:
            await self.last_task
        except asyncio.CancelledError:
            pass

    def __call__(self, *args, **kwargs):
        if self.result_future is None:
            self.result_future = asyncio.Future()
            self.last_result_future = self.result_future
        # every call (even those that do not execute) has a future
        # and we keep track of that for testing/debugging
        future = asyncio.Future()
        if not self.skip_gather:
            _debounced_futures.append(future)
        else:
            _debounced_futures_skip.append(future)
        self.counter += 1

        @functools.wraps(self.f)
        def debounced_execute(counter=self.counter):
            async def run_async():
                # print("task runs", self.f)
                if not self.skip_gather:
                    _debounced_futures.remove(future)
                else:
                    _debounced_futures_skip.remove(future)
                if counter != self.counter:
                    # TODO: maybe we should set cancel instead?
                    future.set_result(None)
                try:
                    if counter == self.counter:
                        if self.previous_result_future and not self.reentrant:
                            try:
                                await self.previous_result_future
                            except:  # noqa
                                pass  # exception of previous run are already handled
                        # we should capture a reference to the current result_future
                        # and 'reset' it now, if we do this later, a next execution
                        # might set and reset it (since the function might take more time)
                        # than the debounce time
                        # CONSIDER: can in theory a later scheduled run_async execute sooner
                        # and should result_future be assigned above this coroutine?
                        result_future = self.result_future
                        pre_hook_future = self.pre_hook_future
                        self.previous_result_future = result_future
                        self.result_future = None  # reset this, but keep last_result_future
                        self._pre_hook_future = asyncio.Future()
                        pre_hook_future.set_result(None)
                        # this allows for pre_hook waiters to run first
                        await asyncio.sleep(1e-9)
                        if self.obj is not None:
                            result = self.f(self.obj, *args, **kwargs)
                        else:
                            result = self.f(*args, **kwargs)
                        if asyncio.iscoroutinefunction(self.f):
                            result = await result
                        future.set_result(result)
                        result_future.set_result(result)
                except Exception as e:
                    import vaex
                    vaex.utils.print_exception_trace(e)
                    try:
                        if self.on_error:
                            if self.obj:
                                self.on_error(self.obj, e)
                            else:
                                self.on_error(e)
                    except Exception as e2:
                        vaex.utils.print_exception_trace(e2)
                        # print("error in error handler", e, type(e))
                    future.set_exception(e)
                    result_future.set_exception(e)
                    # pre_hook_future.set_exception(e)
                finally:
                    debounced_execute_queue.remove(debounced_execute)
            # print("create task", self.f)
            ioloop.create_task(run_async())

        if debounce_enabled:
            ioloop = get_ioloop()

            def thread_safe():
                ioloop.call_later(_test_delay or self.delay_seconds, debounced_execute)
            debounced_execute_queue.append(debounced_execute)
            if ioloop is not None:  # not in IPython
                # print("schedule", self.f)
                ioloop.call_soon_threadsafe(thread_safe)
        else:
            debounced_execute()
        return self.result_future


def debounced(delay_seconds=0.5, method=False, skip_gather=False, on_error=None, reentrant=True):
    '''A decorator to debounce many method/function call into 1 call.

    Note: this only works in a IPython environment, such as a Jupyter notebook context. Outside
    of this context, calling :func:`flush` will execute pending calls.

    :param float delay_seconds: The amount of seconds that should pass without any call, before the (final) call will be executed.
    :param bool method: The decorator should know if the callable is a a method or not, otherwise the debounced is on a per-class basis.
    :param bool skip_gather: The decorated function will be be waited for when calling vaex.jupyter.gather()

    '''
    def wrapped(f):
        return functools.wraps(f)(_debounced_callable(f, delay_seconds, skip_gather, on_error, reentrant=reentrant))
        # the code below should do the same the the return object, but fails to run the tutorial_jupyter.ipynb
        # it looks as if the async part has a scoping issue/bug, CPython bug maybe?
        counters = collections.defaultdict(int)
        result_futures = collections.defaultdict(asyncio.Future)

        @functools.wraps(f)
        def execute(*args, **kwargs):
            if method:  # if it is a method, we want to have a counter per instance
                key = args[0]
            else:
                key = None
            counters[key] += 1
            # multiple calls to the same method share the same result_future
            result_future = result_futures[key]
            print("call", key, counters[key], id(result_future), list(result_futures.keys()))
            print("SCOPE CHECK1", key, counters[key], id(result_future), id(result_futures[key]))
            # but each call leads to a future, which is put in _debounced_futures for testing/debug purposes
            future = asyncio.Future()
            if not skip_gather:
                _debounced_futures.append(future)
            else:
                _debounced_futures_skip.append(future)

            print("asyncio.iscoroutine(f):", asyncio.iscoroutinefunction(f))
            @functools.wraps(execute)
            def debounced_execute(counter=counters[key]):
                print("SCOPE CHECK2", key, counters[key], id(result_future), id(result_futures[key]), counter)
                if asyncio.iscoroutinefunction(f):
                    # this path is a duplicate from the path below
                    # not sure how we can avoid that
                    async def run_async():
                        print("SCOPE CHECK3", key, counters[key], id(result_future), id(result_futures[key]), counter)
                        if counter != counters[key]:  # only execute if the counter wasn't changed in the meantime
                            future.set_result(None)
                            return
                        try:
                            # result = None
                            result = await f(*args, **kwargs)
                            # print("set", key, result, counter, result_future, id(result_future), result_futures)
                            print("SET", key, counters[key], id(result_future), list(result_futures.keys()))
                            del result_futures[key]  # reset result future
                            print("set done", list(result_futures.keys()))
                            result_future.set_result(result)
                        except Exception as e:
                            # if on_error:
                            #     if method:
                            #         on_error(args[0], e)
                            #     else:
                            #         on_error(e)
                            del result_futures[key]  # reset result future
                            future.set_exception(e)
                            result_future.set_exception(e)
                        else:
                            # TODO: maybe we should set cancel instead?
                            future.set_result(result)
                        finally:
                            if not skip_gather:
                                _debounced_futures.remove(future)
                            else:
                                _debounced_futures_skip.remove(future)

                    ioloop.create_task(run_async())
                else:
                    try:
                        result = None
                        if counter == counters[key]:  # only execute if the counter wasn't changed in the meantime
                            result = f(*args, **kwargs)
                            result_future.set_result(result)
                            del result_futures[key]  # reset result future
                    except Exception as e:
                        if on_error:
                            if method:
                                on_error(args[0])
                            else:
                                on_error()
                        future.set_exception(e)  # if nobody awaits this (like gather does, asyncio may complain on exit)
                        result_future.set_exception(e)
                        del result_futures[key]  # reset result future
                    else:
                        # TODO: maybe we should set cancel instead?
                        future.set_result(result)
                    finally:
                        if not skip_gather:
                            _debounced_futures.remove(future)
            if debounce_enabled:
                ioloop = get_ioloop()

                def thread_safe():
                    # ioloop.add_timeout(time.time() + delay_seconds, debounced_execute)
                    ioloop.call_later(delay_seconds, debounced_execute)
                if ioloop is None:  # not in IPython
                    debounced_execute_queue.append(debounced_execute)
                else:
                    ioloop.call_soon_threadsafe(thread_safe)
            else:
                debounced_execute()
            return result_future
        execute.original = f
        return execute
    return wrapped


_selection_hooks = []


def interactive_cleanup():
    for dataset, f in _selection_hooks:
        dataset.signal_selection_changed.disconnect(f)


def interactive_selection(df):
    global _selection_hooks

    def wrapped(f_interact):
        if not hasattr(f_interact, "widget"):
            output = widgets.Output()

            def _selection_changed(df, selection_name):
                with output:
                    clear_output(wait=True)
                    f_interact(df, selection_name)
            hook = df.signal_selection_changed.connect(_selection_changed)
            _selection_hooks.append((df, hook))
            _selection_changed(df, None)
            display(output)
            return functools.wraps(f_interact)
        else:
            def _selection_changed(df, selection_name):
                f_interact.widget.update(df, selection_name)
            hook = df.signal_selection_changed.connect(_selection_changed)
            _selection_hooks.append((df, hook))
            return functools.wraps(f_interact)
    return wrapped
