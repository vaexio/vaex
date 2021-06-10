import contextlib
import functools
import numpy
import dask.base
import logging
import uuid
import pyarrow as pa

import vaex.utils
import functools


log = logging.getLogger('vaex.cache')
_enable_cache_results = vaex.utils.get_env_type(bool, 'VAEX_CACHE_RESULTS', False)


dask.base.normalize_token.register(pa.DataType, repr)

cache = None

class _cleanup:
    def __init__(self, gen, result):
        self._gen = gen
        self._result = result

    def __enter__(self):
        log.debug('entered')
        return self._result

    def __exit__(self, type, value, traceback):
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
def infinite(reset=False):
    '''Sets the cache as a dictionary, creating an infinite cache'''
    global cache
    log.debug("set cache to infinite")
    old_cache = cache
    if reset or (not isinstance(old_cache, dict)):
        cache = {}
    yield
    log.debug("restore old cache")
    cache = old_cache


def is_on():
    return cache is not None


@_with_cleanup
def off():
    global cache
    old_cache = cache
    cache = None
    yield
    cache = old_cache


def get(key):
    if is_on():
        return cache.get(key)

def fingerprint(*args, **kwargs):
    try:
        _restore = uuid.uuid4
        def explain(*args):
            raise TypeError('You have passed in an object for which we cannot determine a fingerprint')
        uuid.uuid4 = explain
        return dask.base.tokenize(*args, **kwargs)
    finally:
        uuid.uuid4 = _restore


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

if _enable_cache_results:
    infinite()
