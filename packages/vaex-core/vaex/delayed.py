__author__ = 'maartenbreddels'
import aplus
import os

"""
Mini library to make working with promises/futures a bit more Python like.

Example:

@delayed
def f(grid):
    return grid**2

f(ds.count(delay=True))

See tests/delayed_test.py for more examples
"""


def promisify(value):
    # TODO, support futures etc..
    if isinstance(value, aplus.Promise):
        return value
    if isinstance(value, (list, tuple)):
        return aplus.listPromise(*list([promisify(k) for k in value]))
    if isinstance(value, dict):
        return aplus.dictPromise({k: promisify(v) for k, v in value.items()})
    else:
        return aplus.Promise.fulfilled(value)


def _log_error(name):
    def _wrapped(exc):
        if os.environ.get('VAEX_DEBUG', False):
            print(f"*** DEBUG: Error from {name}", exc)
        # import vaex
        # vaex.utils.print_stack_trace()
        raise exc
    return _wrapped


def delayed(f):
    '''Decorator to transparantly accept delayed computation.

    Example:

    >>> delayed_sum = ds.sum(ds.E, binby=ds.x, limits=limits,
    >>>                   shape=4, delay=True)
    >>> @vaex.delayed
    >>> def total_sum(sums):
    >>>     return sums.sum()
    >>> sum_of_sums = total_sum(delayed_sum)
    >>> ds.execute()
    >>> sum_of_sums.get()
    See the tutorial for a more complete example https://docs.vaex.io/en/latest/tutorial.html#Parallel-computations
    '''

    def wrapped(*args, **kwargs):
        # print "calling", f, "with", kwargs
        # key_values = kwargs.items()
        key_promise = list([(key, promisify(value)) for key, value in kwargs.items()])
        # key_promise = [(key, promisify(value)) for key, value in key_values]
        arg_promises = list([promisify(value) for value in args])
        kwarg_promises = list([promise for key, promise in key_promise])
        promises = arg_promises + kwarg_promises
        for promise in promises:
            def echo_error(exc, promise=promise):
                print("error with ", promise, "exception is", exc)
                raise exc

            def echo(value, promise=promise):
                print("done with ", repr(promise), "value is", value)
                return value
            # promise.then(echo, echo_error)

        # print promises
        allarguments = aplus.listPromise(*promises)

        def call(_):
            kwargs_real = {key: promise.get() for key, promise in key_promise}
            args_real = list([promise.get() for promise in arg_promises])
            return f(*args_real, **kwargs_real)

        return allarguments.then(call, _log_error("delayed decorator"))
    return wrapped


@delayed
def delayed_args(*args):
    return args


@delayed
def delayed_kwargs(**kwargs):
    return kwargs


@delayed
def delayed_list(l):
    return delayed_args(*l)


@delayed
def delayed_dict(d):
    return delayed_kwargs(**d)


@delayed
def delayed_apply(f, args, kwargs):
    @delayed
    def internal(f, args, kwargs):
        return f(*args, **kwargs)
    return internal(f, args, kwargs)
