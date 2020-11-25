import pyarrow as pa

import vaex.arrow.convert
import vaex.multithreading
import vaex.utils


process_count_default = vaex.utils.get_env_type(int, 'VAEX_NUM_PROCESSES', vaex.multithreading.thread_count_default)


_pool = None
def _get_pool():
    global _pool
    global _mempool
    if _pool is None:
        from multiprocessing import Pool
        _pool = Pool(process_count_default)
    return _pool


def _trim(ar):
    if isinstance(ar, vaex.array_types.supported_arrow_array_types):
        ar = vaex.arrow.convert.trim_buffers_for_pickle(ar)
        pass
    return ar


def apply(f, args, kwargs, multiprocessing):
    if multiprocessing:
        args = [_trim(k) for k in args]
        kwargs = {k:_trim(v) for k, v in kwargs.items()}
        result = _get_pool().apply(f, args, kwargs)
        return result
    else:
        return f(*args, **kwargs)
