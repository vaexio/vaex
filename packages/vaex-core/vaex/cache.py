'''(Currently experimental, use at own risk)
Vaex can cache task results, such as aggregations, or the internal hashmaps used for groupby to make recurring calculations much faster, at the cost of calculating cache keys and storing/retrieving the cached values.

Internally, Vaex calculates fingerprints (such as hashes of data, or file paths and mtimes) to create cache keys that are similar across processes, such that a restart of a process will most likely result in similar hash keys.

Caches can turned on globally, or used as a context manager:

>>> import vaex
>>> df = vaex.example()
>>> vaex.cache.memory_infinite()  # cache on globally
<cache restore context manager>
>>> vaex.cache.is_on()
True
>>> vaex.cache.off() # cache off globally
<cache restore context manager>
>>> vaex.cache.is_on()
False
>>> with vaex.cache.memory_infinite():
...     df.x.sum()  # calculated without cache
array(-20884.64307324)
>>> vaex.cache.is_on()
False


The functions :func:`vaex.cache.set` and  :func:`vaex.cache.get` simply look up the values in a global dict (``vaex.cache.cache``), but can be set for more complex behaviour.

A good library to use for in-memory caching is cachetools (https://pypi.org/project/cachetools/)

>>> import vaex
>>> import cachetools
>>> df = vaex.example()
>>> vaex.cache.cache = cachetools.LRUCache(1_000_000_000)  # 1gb cache

Configure using environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Especially when using the `vaex server <server.html>`_ it can be useful to turn on caching externally using enviroment variables.

    $ VAEX_CACHE=disk VAEX_CACHE_DISK_SIZE_LIMIT="10GB" python -m vaex.server

Will enable caching using :func:`vaex.cache.disk` and configure it to use at max 10 GB of disk space.


'''
import functools
from typing import MutableMapping
import logging
import shutil
import uuid
import pickle
import threading

import dask.base
import pyarrow as pa

import vaex.utils
diskcache = vaex.utils.optional_import('diskcache')
_redis = vaex.utils.optional_import('redis')

log = logging.getLogger('vaex.cache')
_cache_tasks_type = vaex.utils.get_env_type(str, 'VAEX_CACHE', None)  # disk/redis/memory_infinite
disk_size_limit = vaex.utils.get_env_type(str, 'VAEX_CACHE_DISK_SIZE_LIMIT', '1GB')

dask.base.normalize_token.register(pa.DataType, repr)

cache = None

class _cleanup:
    def __init__(self, gen, result):
        self._gen = gen
        self._result = result

    def __repr__(self):
        return '<cache restore context manager>'

    def __enter__(self):
        log.debug('entered')
        return self._result

    def __exit__(self, type=None, value=None, traceback=None):
        try:
            next(self._gen)
        except StopIteration:
            return False
        else:
            raise RuntimeError("generator didn't stop")

def _with_cleanup(f):
    '''Runs f directly, and the code after yield at exit'''
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        gen = f(*args, **kwargs)
        result = next(gen) # run up to the yield
        return _cleanup(gen, result)
    return wrapper


@_with_cleanup
def memory_infinite(clear=False):
    '''Sets a dict a cache, creating an infinite cache.
    
    Calling multiple times with clear=False will keep the current cache (useful in notebook usage)
    '''
    global cache
    log.debug("set cache to infinite")
    old_cache = cache
    if clear or (not isinstance(old_cache, dict)):
        cache = {}
    yield
    log.debug("restore old cache")
    cache = old_cache


@_with_cleanup
def disk(clear=False, size_limit=disk_size_limit, eviction_policy='least-recently-stored'):
    '''Stored cached values using the diskcache library.

    The path to store the cache is: ~/.vaex/cache/diskcache

    :param int or str size_limit: Max size of cache in bytes (or use a string like '128MB')
        See http://www.grantjenks.com/docs/diskcache/tutorial.html?highlight=eviction#tutorial-settings for more details.
    :param str eviction_policy: Eviction policy,
        See http://www.grantjenks.com/docs/diskcache/tutorial.html?highlight=eviction#tutorial-eviction-policies
    :param bool clear: Remove all disk space used for caching before turning on cache.
    '''
    from dask.utils import parse_bytes
    size_limit = parse_bytes(size_limit)
    global cache
    old_cache = cache
    path = vaex.utils.get_private_dir('cache/diskcache')
    if clear:
        try:
            log.debug(f"Clearing disk cache: {path}")
            shutil.rmtree(path)
        except OSError:  # Windows wonkiness
            log.exception(f"Error clearing disk cache: {path}")
    log.debug(f"Initializing disk cache: {path}")
    cache = diskcache.Cache(path, size_limit=size_limit, eviction_policy=eviction_policy)
    yield
    log.debug("Restored old cache")
    cache = old_cache


class _RedisPickle(MutableMapping):
    '''Wraps a client such that the values are pickled/unpickled'''
    def __init__(self, client):
        self.client = client

    def __getitem__(self, key):
        return pickle.loads(self.client[key])

    def __setitem__(self, key, value):
        self.client[key] = pickle.dumps(value)

    def __delitem__(self, key):
        del self.client[key]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.client.keys())

    def keys(self):
        return self.client.keys()


@_with_cleanup
def redis(client=None):
    '''Uses Redis for caching.

    :param client: Redis client, if None, will call redis.Redis()
    '''
    global cache
    log.debug("Set cache to Redis")
    old_cache = cache
    if (not isinstance(old_cache, _redis.Redis)):
        if client is None:
            client = _redis.Redis()
        cache = _RedisPickle(client)
    yield
    log.debug("restore old cache")
    cache = old_cache


@_with_cleanup
def on(type="memory_infinite", **kwargs):
    log.debug("Set cache to %r", type)
    if type == "memory_infinite":
        c = memory_infinite(**kwargs)
    elif type == "disk":
        c = disk(**kwargs)
    elif type == "redis":
        c = redis(**kwargs)
    else:
        raise ValueError(f'Unknown type of cache {type}')
    yield
    c.__exit__()


@_with_cleanup
def off():
    '''Turns off caching, or temporary when used as context manager

    >>> import vaex
    >>> df = vaex.example()
    >>> vaex.cache.memory_infinite()  # cache on
    <cache restore context manager>
    >>> with vaex.cache.off():
    ...     df.x.sum()  # calculated without cache
    array(-20884.64307324)
    >>> df.x.sum()  # calculated with cache
    array(-20884.64307324)
    >>> vaex.cache.off()  # cache off
    <cache restore context manager>
    >>> df.x.sum()  # calculated without cache
    array(-20884.64307324)
    '''

    global cache
    old_cache = cache
    cache = None
    yield
    cache = old_cache


def is_on():
    '''Returns True when caching is enabled'''
    return cache is not None


def set(key, value, type=None, duration_wallclock=None):
    '''Set a cache value

    Useful to more advanced strategies, where we want to have different behaviour based on the type
    and costs. Implementations can set this function override the default behaviour:

    >>> import vaex
    >>> vaex.cache.memory_infinite()  # doctest: +SKIP
    >>> def my_smart_cache_setter(key, value, type=None, duration_wallclock=None):
    ...     if duration_wallclock >= 0.1:  # skip fast calculations
    ...         vaex.cache.cache[key] = value
    ...
    >>> vaex.cache.set = my_smart_cache_setter

    :param str key: key for caching
    :param type: Currently unused.
    :param float duration_wallclock: Time spend on calculating the result (in wallclock time).
    :param value: Any value, typically needs to be pickleable (unless stored in memory)
    '''
    cache[key] = value


def get(key, default=None, type=None):
    '''Looks up the cache value for the key, or returns the default

    Will return None if the cache is turned off.

    :param str key: Cache key.
    :param default: Return when cache is on, but key not in cache
    :param type: Currently unused.
    '''
    if is_on():
        return cache.get(key, default)


class _ThreadLocalCallablePatch:
    def __init__(self, original, patch):
        self.local = threading.local()
        self.local.patch = patch
        self.original = original

    def __call__(self, *args, **kwargs):
        f = getattr(self.local, 'patch', self.original)
        return f(*args, **kwargs)


def _explain(*args):
    raise TypeError('You have passed in an object for which we cannot determine a fingerprint')


def fingerprint(*args, **kwargs):
    try:
        original = uuid.uuid4
        uuid.uuid4 = _ThreadLocalCallablePatch(original, _explain)
        return dask.base.tokenize(*args, **kwargs)
    finally:
        uuid.uuid4 = original


def output_file(callable=None, path_input=None, fs_options_input={}, fs_input=None, path_output=None, fs_options_output={}, fs_output=None):
    """Decorator to do cached conversion from path_input → path_output.

    Caching is active if with the input file, the *args, and **kwargs are unchanged, otherwise callable
    will be called again.


    """
    def wrapper1(callable=callable):
        def wrapper2(callable=callable):
            def call(callable=callable, *args, **kwargs):
                base, ext, fs_options_output_ = vaex.file.split_ext(path_output, fs_options_output)
                path_output_meta = base + '.yaml'
                # this fingerprint changes if input changes, or args, or kwargs
                fp = fingerprint(vaex.file.fingerprint(path_input, fs_options_input, fs=fs_input), args, kwargs)

                def write_fingerprint():
                    log.info('saving fingerprint %r for %s → %s conversion to %s', fp, path_input, path_output, path_output_meta)
                    with vaex.file.open(path_output_meta, 'w', fs_options=fs_options_output_, fs=fs_output) as f:
                        f.write(f"# this file exists so that we know when not to do the\n# {path_input} → {path_output} conversion\n")
                        vaex.utils.yaml_dump(f, {'fingerprint': fp})

                if not vaex.file.exists(path_output, fs_options=fs_options_output_, fs=fs_output):
                    log.info('file %s does not exist yet, running conversion %s → %s', path_output_meta, path_input, path_output)
                    value = callable(*args, **kwargs)
                    write_fingerprint()
                    return value

                if not vaex.file.exists(path_output_meta, fs_options=fs_options_output_, fs=fs_output):
                    log.info('file including fingerprint not found (%) or does not exist yet, running conversion %s → %s', path_output_meta, path_input, path_output)
                    value = callable(*args, **kwargs)
                    write_fingerprint()
                    return value

                # load fingerprint
                with vaex.file.open(path_output_meta, fs_options=fs_options_output_, fs=fs_output) as f:
                    output_meta = vaex.utils.yaml_load(f)

                if output_meta is None or output_meta['fingerprint'] != fp:
                    if output_meta is None:
                        log.error('fingerprint for %s (%s) was empty', path_input, path_output_meta)
                    else:
                        log.info('fingerprint for %s is out of date, rerunning conversion to %s', path_input, path_output)
                    value = callable(*args, **kwargs)
                    write_fingerprint()
                    return value
                else:
                    log.info('fingerprint for %s did not change, reusing converted file %s', path_input, path_output)
            return call
        return wrapper2
    return wrapper1()

if _cache_tasks_type:
    on(_cache_tasks_type)
