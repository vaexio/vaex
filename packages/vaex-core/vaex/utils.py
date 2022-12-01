# -*- coding: utf-8 -*-
from __future__ import absolute_import
import ast
import collections
import concurrent.futures
import contextlib
import functools
import json
import math
import os
import platform
import re
import sys
import threading
import time
from typing import MutableMapping
import warnings
import numbers
import keyword
from filelock import FileLock

import blake3
import dask.utils
import numpy as np
import pyarrow as pa
import six
import yaml

from .json import VaexJsonEncoder, VaexJsonDecoder
import vaex.file

try:
    from sys import version_info
    if version_info[:2] >= (3, 10):
        from importlib.metadata import entry_points
    else:
        from importlib_metadata import entry_points
except ImportError:
    import pkg_resources
    entry_points = pkg_resources.iter_entry_points

is_frozen = getattr(sys, 'frozen', False)
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

osname = dict(darwin="osx", linux="linux", windows="windows")[platform.system().lower()]
# $ export VAEX_DEV=1 to enabled dev mode (skips slow tests)
devmode = os.environ.get('VAEX_DEV', '0') == '1'


# so that vaex can be imported without compiling the extenstions (like in readthedocs)
has_c_extension = os.environ.get('VAEX_NO_C_EXTENSIONS', None) != "1"

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, key, value):
        self[key] = value


class RegistryCallable(MutableMapping):
    '''lazily load entries from entry_points'''
    def __init__(self, entry_points : str, typename : str):
        self.entry_points = entry_points
        self.typename = typename
        self.registry = {}
        self.lock = threading.Lock()

    def __getitem__(self, name):
        if name in self.registry:
            return self.registry[name]
        with self.lock:
            if name in self.registry:
                return self.registry[name]
            for entry in entry_points(group=self.entry_points):
                self.registry[entry.name] = entry.load()

        if name not in self.registry:
            raise NameError(f'No {self.typename} registered with name {name!r} under entry_point {self.entry_points!r}')
        return self.registry[name]

    def __delitem__(self, name):
        del self.registry[name]

    def __iter__(self):
        return self.registry.__iter__()

    def __len__(self):
        return self.registry.__len__()

    def __setitem__(self, name, value):
        self.registry.__setitem__(name, value)


def deprecated(reason):
    def wraps(f):
        if not f.__doc__:
            f.__doc__ = ""
        f.__doc__ = "Deprecated: {}\n\n{}".format(reason, f.__doc__)
        @functools.wraps(f)
        def wraps2(*args, **kwargs):
            warnings.warn("Call to deprecated function {}: {}".format(f.__name__, reason),
                          category=DeprecationWarning, stacklevel=2)
            return f(*args, **kwargs)

        return wraps2

    return wraps


def subdivide(length, parts=None, max_length=None):
    """Yields index tuple (i1, i2) that subdivide an array into parts of max_length"""
    if max_length is None:
        max_length = (length + parts - 1) / parts
    i1 = 0
    while i1 < length:
        i2 = min(i1 + max_length, length)
        yield i1, i2
        i1 = i2
    assert i1 == length


def subdivide_mask(mask, parts=None, max_length=None, logical_length=None):
    """Yields index tuple (l1, l2, i1, i2) that subdivide an array into parts such that it contains max_length True values in mask

    l1 an l2 refer to the logical indices, similar as :func:`subdivide`, while i1, and i2 refer to the 'raw' indices, such that
    `np.sum(mask[i1:i2]) == logical_length` (except for the last element).
    """
    if logical_length is None:
        logical_length = np.asarray(mask).sum()
    if logical_length == 0:
        return
    raw_length = len(np.asarray(mask))
    if max_length is None:
        max_length = (logical_length + parts - 1) / parts
    full_mask = mask
    logical_start = 0
    logical_end =  min(logical_start + max_length, logical_length)
    raw_index = full_mask.raw_offset(1)
    assert raw_index != -1

    while logical_start < logical_length:
        # slice the mask from our offset till end
        mask = full_mask.view(raw_index, raw_length)
        # count how many raw elements we need to skip to get a logical chunk_size
        raw_offset = mask.raw_offset(logical_end - logical_start)
        assert raw_offset != -1
        next_raw_index = raw_index + raw_offset + 1
        yield logical_start, logical_end, raw_index, next_raw_index
        raw_index = next_raw_index
        logical_start = min(logical_start + max_length, logical_length)
        logical_end = min(logical_start + max_length, logical_length)


def submit_subdivide(thread_count, f, length, max_length):
    futures = []
    thread_pool = concurrent.futures.ThreadPoolExecutor(thread_count)
    # thread_pool = concurrent.futures.ProcessPoolExecutor(thread_count)
    for i1, i2 in list(subdivide(length, max_length=max_length)):
        futures.append(thread_pool.submit(f, i1, i2))
    return futures


def linspace_centers(start, stop, N):
    return np.arange(N) / (N + 0.) * (stop - start) + float(stop - start) / N / 2 + start


def multisum(a, axes):
    correction = 0
    for axis in axes:
        a = np.nansum(a, axis=axis - correction)
        correction += 1
    return a


def disjoined(data):
    # create marginalized distributions and multiple them together
    data_disjoined = None
    dim = len(data.shape)
    for d in range(dim):
        axes = list(range(dim))
        axes.remove(d)
        data1d = multisum(data, axes)
        shape = [1 for k in range(dim)]
        shape[d] = len(data1d)
        data1d = data1d.reshape(tuple(shape))
        if d == 0:
            data_disjoined = data1d
        else:
            data_disjoined = data_disjoined * data1d
    return data_disjoined


def get_root_path():
    osname = platform.system().lower()
    # if (osname == "linux") and is_frozen: # we are using pyinstaller
    if is_frozen:  # we are using pyinstaller or py2app
        return os.path.dirname(sys.argv[0])
    else:
        return os.path.abspath(".")


def os_open(document):
    """Open document by the default handler of the OS, could be a url opened by a browser, a text file by an editor etc"""
    osname = platform.system().lower()
    if osname == "darwin":
        os.system("open \"" + document + "\"")
    if osname == "linux":
        cmd = "xdg-open \"" + document + "\"&"
        os.system(cmd)
    if osname == "windows":
        os.system("start \"" + document + "\"")


def filesize_format(value):
    for unit in ['bytes', 'KiB', 'MiB', 'GiB']:
        if value < 1024.0:
            return "%3.1f%s" % (value, unit)
        value /= 1024.0
    return "%3.1f%s" % (value, 'TiB')


log_timer = True


class Timer(object):
    def __init__(self, name=None, logger=None):
        self.name = name
        self.logger = logger

    def __enter__(self):
        if log_timer:
            if self.logger:
                self.logger.debug("%s starting" % self.name)
            else:
                print(('[%s starting]...' % self.name))
            self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if log_timer:
            msg = "%s done, %ss elapsed" % (self.name, time.time() - self.tstart)
            if self.logger:
                self.logger.debug(msg)
            else:
                print(msg)
            if type or value or traceback:
                print((type, value, traceback))
        return False


def get_vaex_home():
    '''Get vaex home directory, defaults to $HOME/.vaex.

    The $VAEX_HOME environment variable can be set to override this default.

    If both $VAEX_HOME and $HOME are not define, the current working directory is used.
    '''
    if 'VAEX_HOME' in os.environ:
        return os.environ['VAEX_HOME']
    if 'VAEX_PATH_HOME' in os.environ:  # for backwards compatibility
        return os.environ['VAEX_PATH_HOME']
    elif 'HOME' in os.environ:
        return os.path.join(os.environ['HOME'], ".vaex")
    else:
        return os.getcwd()


def get_private_dir(subdir=None, *extra):
    path = get_vaex_home()
    if subdir:
        path = os.path.join(path, subdir, *extra)
    os.makedirs(path, exist_ok=True)
    return path


def make_list(sequence):
    if isinstance(sequence, np.ndarray):
        return sequence.tolist()
    else:
        return list(sequence)


def confirm_on_console(topic, msg):
    done = False
    print(topic)
    while not done:
        output = raw_input(msg + ":[y/n]")
        if output.lower() == "y":
            return True
        if output.lower() == "n":
            return False


def yaml_dump(f, data):
    yaml.safe_dump(data, f, default_flow_style=False, encoding='utf-8', allow_unicode=True, sort_keys=False)


def yaml_load(f):
    return yaml.safe_load(f)


def write_json_or_yaml(file, data, fs_options={}, fs=None, old_style=True):
    file, path = vaex.file.file_and_path(file, mode='w', fs_options=fs_options, fs=fs)
    try:
        if path:
            base, ext = os.path.splitext(path)
        else:
            ext = '.json'  # default
        if ext == ".json":
            json.dump(data, file, indent=2, cls=VaexJsonEncoder if old_style else None)
        elif ext == ".yaml":
            yaml_dump(file, data)
        else:
            raise ValueError("file should end in .json or .yaml (not %s)" % ext)
    finally:
        file.close()


def read_json_or_yaml(file, fs_options={}, fs=None, old_style=True):
    file, path = vaex.file.file_and_path(file, fs_options=fs_options, fs=fs)
    try:
        if path:
            base, ext = os.path.splitext(path)
        else:
            ext = '.json'  # default
        if ext == ".json":
            return json.load(file, cls=VaexJsonDecoder if old_style else None) or {}
        elif ext == ".yaml":
            return yaml_load(file) or {}
        else:
            raise ValueError("file should end in .json or .yaml (not %s)" % ext)
    finally:
        file.close()


def check_memory_usage(bytes_needed, confirm):
    psutil = optional_import('psutil')
    if bytes_needed > psutil.virtual_memory().available:
        if bytes_needed < (psutil.virtual_memory().available + psutil.swap_memory().free):
            text = "Action requires %s, you have enough swap memory available but it will make your computer slower, do you want to continue?" % (
            filesize_format(bytes_needed),)
            return confirm("Memory usage issue", text)
        else:
            text = "Action requires %s, you do not have enough swap memory available, do you want try anyway?" % (
            filesize_format(bytes_needed),)
            return confirm("Memory usage issue", text)
    return True


def ensure_string(string_or_bytes, encoding="utf-8", cast=False):
    if cast:
        if six.PY2:
            string_or_bytes = unicode(string_or_bytes)
        else:
            string_or_bytes = str(string_or_bytes)
    if isinstance(string_or_bytes, six.string_types):
        return string_or_bytes
    else:
        return string_or_bytes.decode(encoding)


def filename_shorten(path, max_length=150):
    # parts = path.split(os.path.sep)
    parts = []
    done = False
    tail = path
    while not done:
        tail, head = os.path.split(tail)
        # print ">>", tail, head
        if not head:
            done = True
            parts.append(tail)
        else:
            parts.append(head)
    parts.reverse()
    # print "parts", parts, os.path.join(*parts)
    if len(parts) > 4:
        first, middle, last = os.path.join(parts[0], parts[1]), parts[2:-1], parts[-1]
        # print first, middle, last
        while (len(os.path.join(first, *(middle + [last]))) <= max_length) and middle:
            last = os.path.join(middle[-1], last)
            middle = middle[:-1]
        if middle:
            return os.path.join(first, "...", last)
        else:
            return os.path.join(first, last)
    else:
        return path


def listify(*args):
    import vaex.expression
    if isinstance(args[0], (six.string_types, vaex.expression.Expression)):
        return False, [[x] for x in args]
    try:
        _ = args[0][0]
        return True, args
    except:
        return False, [[x] for x in args]


def unlistify(waslist, *args):
    if waslist:
        if len(args) == 1:
            return args[0]
        return args
    else:
        values = [x[0] for x in args]
        if len(values) == 1:
            return values[0]


def valid_expression(names, name):
    if name in names and not valid_identifier(name):
        return f'df[%r]' % name
    else:
        return name


def valid_identifier(name):
    return name.isidentifier() and not keyword.iskeyword(name)


def find_valid_name(name, used=None):
    if used is None:
        used = []
    if isinstance(name, int):
        name = str(name)
    if name in used:
        nr = 1
        while name + ("_%d" % nr) in used:
            nr += 1
        name = name + ("_%d" % nr)
    return name


def _python_save_name(name, used=None):
    if used is None:
        used = []
    first, rest = name[0], name[1:]
    name = re.sub("[^a-zA-Z_]", "_", first) + re.sub("[^a-zA-Z_0-9]", "_", rest)
    if name in used:
        nr = 1
        while name + ("_%d" % nr) in used:
            nr += 1
        name = name + ("_%d" % nr)
    return name


@contextlib.contextmanager
def write_to(f, mode):
    """Flexible writing, where f can be a filename or f object, if filename, closed after writing"""
    if hasattr(f, 'write'):
        yield f
    else:
        f = open(f, mode)
        yield f
        f.close()


# these are used to add namespace to vaex' dataset
# like ds.viz.plot2d(ds.x, ds.y)

class BoundMethods(object):

    def __init__(self, obj, methods):
        for name, value in methods.items():
            setattr(self, name, value.__get__(obj))


class InnerNamespace(object):

    def __init__(self, methods, cls=None, prefix=''):
        self._methods = methods
        self.cls = cls
        self.prefix = prefix

    def _add(self, **kwargs):
        self._methods.update(kwargs)
        self.__dict__.update(kwargs)
        if self.cls:
            for name, value in kwargs.items():
                setattr(self.cls, self.prefix + name, value)

    # def __getattr__(self, name):
    # 	if name in self._methods:
    # 		return self._methods[name]
    # 	else:
    # 		return object.__getattr__(self, name)

    def __get__(self, obj, objtype):
        # print("get", obj, objtype)
        if obj is None:
            return self
        else:
            return BoundMethods(obj, self._methods)


def _parse_f(f):
    if f is None:
        return lambda x: x
    elif isinstance(f, six.string_types):
        if f == "identity":
            return lambda x: x
        else:
            if hasattr(np, f):
                return getattr(np, f)
            else:
                raise ValueError("do not understand f = %s, should be a function, string 'identity' or a function from numpy such as 'log', 'log1p'" % f)
    else:
        return f


def _normalize(a, axis=None):
    a = np.copy(a)  # we're gonna modify inplace, better copy iy
    mask = np.isfinite(a)
    a[~mask] = np.nan  # put inf to nan
    allaxis = list(range(len(a.shape)))
    if axis is not None:
        if type(axis) == int:
            axis = [axis]
        for ax in axis:
            allaxis.remove(ax)
        axis = tuple(allaxis)
    vmin = np.nanmin(a)
    vmax = np.nanmax(a)
    a = a - np.nanmin(a, axis=axis, keepdims=True)
    a /= np.nanmax(a, axis=axis, keepdims=True)
    return a, vmin, vmax


def _normalize_selection_name(name):
    if name is True:
        return "default"
    elif name is False:
        return None
    else:
        return name


def _normalize_selection(selection):
    if isinstance(selection, (list, tuple)):
        return type(selection)([_normalize_selection_name(k) for k in selection])
    else:
        return _normalize_selection_name(selection)


def _parse_n(n):
    if isinstance(n, six.string_types):
        if n == "normalize":
            return _normalize
            # return lambda x: x
        else:
            raise ValueError("do not understand n = %s, should be a function, or string 'normalize'" % n)
    else:
        return n


def _parse_reduction(name, colormap, colors):
    if name.startswith("stack.fade"):
        def _reduce_stack_fade(grid):
            return grid[..., -1]  # return last..
        return _reduce_stack_fade
    elif name.startswith("colormap"):
        import matplotlib.cm
        cmap = matplotlib.cm.get_cmap(colormap)

        def f(grid):
            masked_grid = np.ma.masked_invalid(grid)  # convert inf/nan to a mask so that mpl colors bad values correcty
            return cmap(masked_grid)
        return f
    elif name.startswith("stack.color"):
        def f(grid, colors=colors, colormap=colormap):
            import matplotlib.cm
            colormap = matplotlib.cm.get_cmap(colormap)
            if isinstance(colors, six.string_types):
                colors = matplotlib.cm.get_cmap(colors)
            if isinstance(colors, matplotlib.colors.Colormap):
                group_count = grid.shape[-1]
                colors = [colors(k / float(group_count - 1.)) for k in range(group_count)]
            else:
                colors = [matplotlib.colors.colorConverter.to_rgba(k) for k in colors]
            # print grid.shape
            total = np.nansum(grid, axis=0) / grid.shape[0]
            # grid /= total
            # mask = total > 0
            # alpha = total - total[mask].min()
            # alpha[~mask] = 0
            # alpha = total / alpha.max()
            # print np.nanmax(total), np.nanmax(grid)
            return colormap(total)
            rgba = grid.dot(colors)
            # def _norm(data):
            #   mask = np.isfinite(data)
            #   data = data - data[mask].min()
            #   data /= data[mask].max()
            #   return data
            # rgba[...,3] = (f(alpha))
            # rgba[...,3] = 1
            rgba[total == 0, 3] = 0.
            # mask = alpha > 0
            # if 1:
            #   for i in range(3):
            #       rgba[...,i] /= total
            #       #rgba[...,i] /= rgba[...,0:3].max()
            #       rgba[~mask,i] = background_color[i]
            # rgba = (np.swapaxes(rgba, 0, 1))
            return rgba
        return f

    else:
        raise ValueError("do not understand reduction = %s, should be a ..." % name)


def _is_string(x):
    return isinstance(x, six.string_types)


def _issequence(x):
    return isinstance(x, (tuple, list, np.ndarray))


def _isnumber(x):
    return isinstance(x, (numbers.Number, pa.Scalar))


def _is_limit(x):
    return isinstance(x, (tuple, list, np.ndarray)) and all([_isnumber(k) for k in x])


def _ensure_list(x):
    return [x] if not _issequence(x) else x


def _ensure_string_from_expression(expression):
    import vaex.expression
    if expression is None:
        return None
    elif isinstance(expression, bool):
        return expression
    elif isinstance(expression, six.string_types):
        return expression
    elif isinstance(expression, vaex.expression.Expression):
        return expression.expression
    else:
        raise ValueError('%r is not of string or Expression type, but %r' % (expression, type(expression)))


def _ensure_strings_from_expressions(expressions):
    if _issequence(expressions):
        return [_ensure_strings_from_expressions(k) for k in expressions]
    else:
        return _ensure_string_from_expression(expressions)


def _expand(x, dimension, type=tuple):
    if _issequence(x):
        assert len(x) == dimension, "wants to expand %r to dimension %d" % (x, dimension)
        return type(x)
    else:
        return type((x,) * dimension)


def _expand_shape(shape, dimension):
    if isinstance(shape, (tuple, list)):
        assert len(shape) == dimension, "wants to expand shape %r to dimension %d" % (shape, dimension)
        return tuple(shape)
    else:
        return (shape,) * dimension


def _expand_limits(limits, dimension):
    if isinstance(limits, (tuple, list, np.ndarray)) and \
            (isinstance(limits[0], (tuple, list, np.ndarray)) or isinstance(limits[0], str) or limits[0] is None):
        assert len(limits) == dimension, "wants to expand shape %r to dimension %d" % (limits, dimension)
        return tuple(limits)
    else:
        return (limits, ) * dimension


def as_flat_float(a):
    if a.dtype.type == np.float64 and a.strides[0] == 8:
        return a
    else:
        return a.astype(np.float64, copy=True)

def as_flat_array(a, dtype=np.float64):
    if a.dtype.type == dtype and a.strides[0] == 8:
        return a
    else:
        return a.astype(dtype, copy=True)


def is_contiguous(ar):
    return ar.flags['C_CONTIGUOUS']


def as_contiguous(ar):
    return ar if is_contiguous(ar) else ar.copy()


def _split_and_combine_mask(arrays):
    '''Combines all masks from a list of arrays, and logically ors them into a single mask'''
    masks = [np.ma.getmaskarray(block) for block in arrays if np.ma.isMaskedArray(block)]
    arrays = [block.data if np.ma.isMaskedArray(block) else block for block in arrays]
    mask = None
    if masks:
        mask = masks[0].copy()
        for other in masks[1:]:
            mask |= other
    return arrays, mask

def gen_to_list(fn=None, wrapper=list):
    '''A decorator which wraps a function's return value in ``list(...)``.

    Useful when an algorithm can be expressed more cleanly as a generator but
    the function should return an list.

    Example:

    >>> @gen_to_list
    ... def get_lengths(iterable):
    ...     for i in iterable:
    ...         yield len(i)
    >>> get_lengths(["spam", "eggs"])
    [4, 4]
    >>>
    >>> @gen_to_list(wrapper=tuple)
    ... def get_lengths_tuple(iterable):
    ...     for i in iterable:
    ...         yield len(i)
    >>> get_lengths_tuple(["foo", "bar"])
    (3, 3)

    :param fn: generator function to be converted list or touple
    :wrapper (str/tuple) wrapper:
    :return: list or tuple, depending on the wrapper input parameter
    :rtype: list or tuple
    '''
    def listify_return(fn):
        @functools.wraps(fn)
        def listify_helper(*args, **kw):
            return wrapper(fn(*args, **kw))
        return listify_helper
    if fn is None:
        return listify_return
    return listify_return(fn)


def find_type_from_dtype(namespace, prefix, *dtypes, transient=True, support_non_native=True):
    from .array_types import is_string_type
    non_native = False
    postfix = ""
    for i, dtype in enumerate(dtypes):
        if i != 0:
            postfix += "_"
        if dtype == 'string':
            if transient:
                postfix += 'string'
            else:
                postfix += 'string' # view not support atm
        else:
            dtype = dtype.numpy
            type_name = str(dtype)
            if type_name == '>f8':
                type_name = 'float64'
            if dtype.kind == "M":
                type_name = "int64"
            if dtype.kind == "m":
                type_name = "int64"
            postfix += type_name
            # for object there is no non-native version
            if support_non_native and dtype.kind != 'O' and dtype.byteorder not in ["<", "=", "|"]:
                if i != 0:
                    if not non_native:
                        raise TypeError('Mixed endianness')
                non_native = True
            else:
                if non_native:
                    raise TypeError('Mixed endianness')
    if non_native:
        postfix += "_non_native"
    name = prefix + postfix
    if hasattr(namespace, name):
        return getattr(namespace, name)
    else:
        raise ValueError('Could not find a class (%s), seems %s is not supported' % (name, dtype))


def to_native_dtype(dtype):
    if isinstance(dtype, np.dtype) and dtype.byteorder not in "<=|":
        return dtype.newbyteorder()
    else:
        return dtype


def to_native_array(ar):
    if ar.dtype.byteorder not in "<=|":
        return ar.astype(to_native_dtype(ar.dtype))
    else:
        return ar


def extract_central_part(ar):
    return ar[(slice(2,-1), ) * ar.ndim]

def unmask_selection_mask(selection_mask):
    if np.ma.isMaskedArray(selection_mask):
        # if we are doing a selection on a masked array
        selection_mask, mask = selection_mask.data, np.ma.getmaskarray(selection_mask)
        # exclude the masked values
        selection_mask = selection_mask & ~mask
    return selection_mask


def wrap_future_with_promise(future):
    from vaex.promise import Promise
    if isinstance(future, Promise):  # TODO: not so nice, sometimes we pass a promise
        return future
    promise = Promise()

    def callback(future):
        e = future.exception()
        if e:
            promise.reject(e)
        else:
            promise.fulfill(future.result())
    future.add_done_callback(callback)
    return promise


def required_dtype_for_max(N, signed=True):
    if signed:
        dtypes = [np.int8, np.int16, np.int32, np.int64]
    else:
        dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
    for dtype in dtypes:
        if N <= np.iinfo(dtype).max:
            return np.dtype(dtype)
    else:
        raise ValueError(f"Cannot store a max value of {N} inside an uint64/int64")


def required_dtype_for_range(vmin, vmax, signed=True):
    if signed:
        dtypes = [np.int8, np.int16, np.int32, np.int64]
    else:
        dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
    for dtype in dtypes:
        if (vmin >= np.iinfo(dtype).min) and (vmax <= np.iinfo(dtype).max):
            return np.dtype(dtype)
    else:
        raise ValueError(f"Cannot store the range {vmin}-{vmax} inside an uint64/int64")


def required_dtype_for_int(value, signed=True):
    if signed or value < 0:
        dtypes = [np.int8, np.int16, np.int32, np.int64]
    else:
        dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
    for dtype in dtypes:
        if value >= 0:
            if value <= np.iinfo(dtype).max:
                return np.dtype(dtype)
        else:
            if value >= np.iinfo(dtype).min:
                return np.dtype(dtype)
    else:
        raise ValueError(f"Cannot store a value of {value} inside an uint64/int64")


def print_stack_trace(*args, **kwargs):
    import traceback
    print("args: ", args, kwargs)
    traceback.print_stack()


def print_exception_trace(e):
    import traceback
    import sys
    print(''.join(traceback.format_exception(None, e, e.__traceback__)), file=sys.stdout, flush=True)


def format_exception_trace(e):
    import traceback
    return ''.join(traceback.format_exception(None, e, e.__traceback__))


class ProxyModule:
    def __init__(self, name, version, modules=None):
        self.name = name
        self.module = None
        self.version = version
        self.modules = modules if modules else [self.name]

    def _ensure_import(self):
        if self.module is None:
            import importlib
            try:
                for module_name in self.modules:
                    importlib.import_module(module_name)
                # the module object itself needs to be the top module
                top_name = self.name.split(".")[0]
                self.module = importlib.import_module(top_name)
            except Exception as e:
                raise ImportError(f'''Error importing module {self.name}: {e}

Vaex needs an optional dependency '{self.name}' for the feature you are using. To install, use:

$ pip install "{self.name}{self.version}"

Or when using conda:
$ conda install -c conda-forge "{self.name}{self.version}""

        ''') from e

    def __getattr__(self, name):
        self._ensure_import()
        return getattr(self.module, name)


def optional_import(name, version='', modules=None):
    return ProxyModule(name, version=version, modules=modules)


def div_ceil(n, d):
    """Integer divide that sounds up (to an int).

    See https://stackoverflow.com/a/54585138/5397207

    Examples
    >>> div_ceil(6, 2)
    3
    >>> div_ceil(7, 2)
    4
    >>> div_ceil(8, 2)
    4
    >>> div_ceil(9, 2)
    5
    """
    return (n + d - 1) // d


def get_env_memory(key, default=None):
    value = os.environ.get(key, default)
    if value is not None:
        try:
            value = ast.literal_eval(value)
        except:
            pass
        if isinstance(value, str):
            value = dask.utils.parse_bytes(value)
        if not isinstance(value, int):
            raise TypeError(f"Expected env var {key} to be of integer type")
    return value


def get_env_type(type, key, default=None):
    '''Get an env var named key, and cast to type

    >>> import os
    >>> get_env_type(int, 'VAEX_NUM_THREADS_TEST')
    >>> get_env_type(int, 'VAEX_NUM_THREADS_TEST', 10)
    10
    >>> get_env_type(int, 'VAEX_NUM_THREADS_TEST', '10')
    10
    >>> os.environ['VAEX_NUM_THREADS_TEST'] = '20'
    >>> get_env_type(int, 'VAEX_NUM_THREADS_TEST')
    20
    >>> get_env_type(int, 'VAEX_NUM_THREADS_TEST', '10')
    20
    >>> os.environ['VAEX_NUM_THREADS_TEST'] = ' '
    >>> get_env_type(int, 'VAEX_NUM_THREADS_TEST')
    >>> get_env_type(int, 'VAEX_NUM_THREADS_TEST', '11')
    11
    '''
    value = os.environ.get(key, default)
    if isinstance(value, str) and value.strip() == '' and type != str:
        # support empty strings
        value = default
    if value is not None:
        return type(ast.literal_eval(repr(value)))


def dropnan(sequence, expect=None):
    original_type = type(sequence)
    sequence = list(sequence)
    non_nan = [k for k in sequence if k == k]
    if expect is not None:
        assert len(sequence) - len(non_nan) == 1, "expected 1 nan value"
    return original_type(non_nan)


def dict_replace_key(d, key_old, key_new):
    '''Replace a key, without changing order'''
    d_new = {}
    for key, value in d.items():
        if key == key_old:
            key = key_new
        d_new[key] = value
    return d_new

# backwards compatibility
def progressbars(*args, **kwargs):
    from .progress import tree
    return tree(*args, **kwargs)


import contextlib
@contextlib.contextmanager
def file_lock(name):
    '''Context manager for creating a file lock in the file lock directory.

    :param name: A unique name for the context (e.g. a fingerprint) on which the filename is based.
    '''
    path = os.path.join(vaex.settings.main.path_lock, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with FileLock(path):
        yield


def create_hasher(initial_data=None, large_data=True):
    """Creates a blake3 hasher"""
    if not hasattr(blake3, "__version__"):
        version = (0, 2)
    else:
        version = tuple(map(int, blake3.__version__.split(".")[:2]))
    if version < (0, 3):
        return blake3.blake3(initial_data, multithreading=large_data)
    else:
        return blake3.blake3(initial_data, max_threads=blake3.blake3.AUTO if large_data else 1)
