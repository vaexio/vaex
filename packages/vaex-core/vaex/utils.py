# -*- coding: utf-8 -*-
from __future__ import absolute_import
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
import time
import warnings
import numbers
import keyword

import numpy as np
import progressbar
import psutil
import six
import yaml

from .column import str_type
from .json import VaexJsonEncoder, VaexJsonDecoder


is_frozen = getattr(sys, 'frozen', False)
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

osname = dict(darwin="osx", linux="linux", windows="windows")[platform.system().lower()]
# $ export VAEX_DEV=1 to enabled dev mode (skips slow tests)
devmode = os.environ.get('VAEX_DEV', '0') == '1'


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, key, value):
        self[key] = value


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


def get_private_dir(subdir=None, *extra):
    path = os.path.expanduser('~/.vaex')
    if subdir:
        path = os.path.join(path, subdir, *extra)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def make_list(sequence):
    if isinstance(sequence, np.ndarray):
        return sequence.tolist()
    else:
        return list(sequence)


# from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
# FileTransferSpeed, FormatLabel, Percentage, \
# ProgressBar, ReverseBar, RotatingMarker, \
# SimpleProgress, Timer, AdaptiveETA, AbsoluteETA, AdaptiveTransferSpeed
# from progressbar.widgets import TimeSensitiveWidgetBase, FormatWidgetMixin


class CpuUsage(progressbar.widgets.FormatWidgetMixin, progressbar.widgets.TimeSensitiveWidgetBase):
    def __init__(self, format='CPU Usage: %(cpu_usage)s%%', usage_format="% 5d"):
        super(CpuUsage, self).__init__(format=format)
        self.usage_format = usage_format
        self.utime_0 = None
        self.stime_0 = None
        self.walltime_0 = None

    def __call__(self, progress, data):
        utime, stime, child_utime, child_stime, walltime = os.times()
        if self.utime_0 is None:
            self.utime_0 = utime
        if self.stime_0 is None:
            self.stime_0 = stime
        if self.walltime_0 is None:
            self.walltime_0 = walltime
        data["utime_0"] = self.utime_0
        data["stime_0"] = self.stime_0
        data["walltime_0"] = self.walltime_0

        delta_time = utime - self.utime_0 + stime - self.stime_0
        delta_walltime = walltime - self.walltime_0
        # print delta_time, delta_walltime, utime, self.utime_0, stime, self.stime_0
        if delta_walltime == 0:
            data["cpu_usage"] = "---"
        else:
            cpu_usage = delta_time / (delta_walltime * 1.) * 100
            data["cpu_usage"] = self.usage_format % cpu_usage
        # utime0, stime0, child_utime0, child_stime0, walltime0 = os.times()
        return progressbar_mod.widgets.FormatWidgetMixin.__call__(self, progress, data)


progressbar_mod = progressbar


def _progressbar_progressbar2(type=None, name="processing", max_value=1):
    widgets = [
        name,
        ': ', progressbar_mod.widgets.Percentage(),
        ' ', progressbar_mod.widgets.Bar(),
        ' ', progressbar_mod.widgets.ETA(),
        # ' ', progressbar_mod.widgets.AdaptiveETA(),
        ' ', CpuUsage()
    ]
    bar = progressbar_mod.ProgressBar(widgets=widgets, max_value=max_value)
    bar.start()
    return bar
    # FormatLabel('Processed: %(value)d lines (in: %(elapsed)s)')


def _progressbar_vaex(type=None, name="processing", max_value=1):
    import vaex.misc.progressbar as pb
    return pb.ProgressBar(0, 1)

def _progressbar_widget(type=None, name="processing", max_value=1):
    import vaex.misc.progressbar as pb
    return pb.ProgressBarWidget(0, 1, name=name)


_progressbar_typemap = {}
_progressbar_typemap['progressbar2'] = _progressbar_progressbar2
_progressbar_typemap['vaex'] = _progressbar_vaex
_progressbar_typemap['widget'] = _progressbar_widget


def progressbar(type_name=None, title="processing", max_value=1):
    type_name = type_name or 'vaex'
    return _progressbar_typemap[type_name](name=title)


def progressbar_widget():
    pass


class _progressbar(object):
    pass


class _progressbar_wrapper(_progressbar):
    def __init__(self, bar):
        self.bar = bar

    def __call__(self, fraction):
        self.bar.update(fraction)
        if fraction == 1:
            self.bar.finish()
        return True

    def status(self, name):
        self.bar.bla = name


class _progressbar_wrapper_sum(_progressbar):
    def __init__(self, children=None, next=None, bar=None, parent=None, name=None):
        self.next = next
        self.children = children or list()
        self.finished = False
        self.last_fraction = None
        self.fraction = 0
        self.bar = bar
        self.parent = parent
        self.name = name
        self.cancelled = False
        self.oncancel = lambda: None

    def cancel(self):
        self.cancelled = True

    def __repr__(self):
        name = self.__class__.__module__ + "." + self.__class__.__name__
        return "<%s(name=%r)> instance at 0x%x" % (name, self.name, id(self))

    def add(self, name=None):
        pb = _progressbar_wrapper_sum(parent=self, name=name)
        self.children.append(pb)
        return pb

    def add_task(self, task, name=None):
        pb = self.add(name)
        pb.oncancel = task.cancel
        task.signal_progress.connect(pb)
        if self.bar and hasattr(self.bar, 'add_child'):
            self.bar.add_child(pb, task, name)

    def __call__(self, fraction):
        if self.cancelled:
            return False
        # ignore fraction
        result = True
        if len(self.children) == 0:
            self.fraction = fraction
        else:
            self.fraction = sum([c.fraction for c in self.children]) / len(self.children)
        fraction = self.fraction
        if fraction != self.last_fraction:  # avoid too many calls
            if fraction == 1 and not self.finished:  # make sure we call finish only once
                self.finished = True
                if self.bar:
                    self.bar.finish()
            elif fraction != 1:
                if self.bar:
                    self.bar.update(fraction)
            if self.next:
                result = self.next(fraction)
        if self.parent:
            assert self in self.parent.children
            result = self.parent(None) in [None, True] and result  # fraction is not used anyway..
            if result is False:
                self.oncancel()
        self.last_fraction = fraction
        return result

    def status(self, name):
        pass


def progressbars(f=True, next=None, name=None):
    if isinstance(f, _progressbar_wrapper_sum):
        return f
    if callable(f):
        next = f
        f = False
    if f in [None, False]:
        return _progressbar_wrapper_sum(next=next, name=name)
    else:
        if f is True:
            return _progressbar_wrapper_sum(bar=progressbar(), next=next, name=name)
        elif isinstance(f, six.string_types):
            return _progressbar_wrapper_sum(bar=progressbar(f), next=next, name=name)
        else:
            return _progressbar_wrapper_sum(next=next, name=name)


def progressbar_callable(title="processing", max_value=1):
    bar = progressbar(title=title, max_value=max_value)
    return _progressbar_wrapper(bar)


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
    yaml.safe_dump(data, f, default_flow_style=False, encoding='utf-8', allow_unicode=True)


def yaml_load(f):
    return yaml.safe_load(f)


def write_json_or_yaml(filename, data):
    base, ext = os.path.splitext(filename)
    if ext == ".json":
        with open(filename, "w") as f:
            json.dump(data, f, indent=2, cls=VaexJsonEncoder)
    elif ext == ".yaml":
        with open(filename, "w") as f:
            yaml_dump(f, data)
    else:
        raise ValueError("file should end in .json or .yaml (not %s)" % ext)


def read_json_or_yaml(filename):
    base, ext = os.path.splitext(filename)
    if ext == ".json":
        with open(filename, "r") as f:
            return json.load(f, cls=VaexJsonDecoder) or {}
    elif ext == ".yaml":
        with open(filename, "r") as f:
            return yaml_load(f) or {}
    else:
        raise ValueError("file should end in .json or .yaml (not %s)" % ext)


# from http://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts

_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG


def dict_representer(dumper, data):
    return dumper.represent_dict(data.iteritems() if hasattr(data, "iteritems") else data.items())


def dict_constructor(loader, node):
    return collections.OrderedDict(loader.construct_pairs(node))


yaml.add_representer(collections.OrderedDict, dict_representer, yaml.SafeDumper)
yaml.add_constructor(_mapping_tag, dict_constructor, yaml.SafeLoader)


def check_memory_usage(bytes_needed, confirm):
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


def find_valid_name(name, used=[]):
    first, rest = name[0], name[1:]
    if not first.isidentifier():
        if ('col_' + first).isidentifier():
            first = 'col_' + first
        else:
            first = 'col_'
    name = first
    for char in rest:
        # we test if it is an identifier with _ prefixed, since not every first character
        # and following character are treated the same
        # https://docs.python.org/3/reference/lexical_analysis.html#identifiers
        if not ('_' + char).isidentifier():
            name += '_'
        else:
            name += char
    if keyword.iskeyword(name):
        name += '_'
    if name in used:
        nr = 1
        while name + ("_%d" % nr) in used:
            nr += 1
        name = name + ("_%d" % nr)
    return name


_python_save_name = find_valid_name


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
    return isinstance(x, numbers.Number)


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
            (isinstance(limits[0], (tuple, list, np.ndarray)) or isinstance(limits[0], six.string_types)):
        assert len(limits) == dimension, "wants to expand shape %r to dimension %d" % (limits, dimension)
        return tuple(limits)
    else:
        return [limits, ] * dimension


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


def find_type_from_dtype(namespace, prefix, dtype, transient=True):
    if dtype == str_type:
        if transient:
            postfix = 'string'
        else:
            postfix = 'string' # view not support atm
    else:
        postfix = str(dtype)
        if postfix == '>f8':
            postfix = 'float64'
        if dtype.kind == "M":
            postfix = "uint64"
        if dtype.kind == "m":
            postfix = "int64"
        # for object there is no non-native version
        if dtype.kind != 'O' and dtype.byteorder not in ["<", "=", "|"]:
            postfix += "_non_native"
    name = prefix + postfix
    if hasattr(namespace, name):
        return getattr(namespace, name)
    else:
        raise ValueError('Could not find a class (%s), seems %s is not supported' % (name, dtype))


def to_native_dtype(dtype):
    if dtype.byteorder not in "<=|":
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


def upcast(dtype):
    if dtype.kind == "b":
        return np.dtype('int64')
    if dtype.kind == "i":
        return np.dtype('int64')
    if dtype.kind == "u":
        return np.dtype('uint64')
    if dtype.kind == "f":
        return np.dtype('float64')
    return dtype


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
            return dtype
    else:
        raise ValueError(f'Cannot store a max value on {N} inside an uint64/int64')


def print_stack_trace(*args, **kwargs):
    import traceback
    print("args: ", args, kwargs)
    traceback.print_stack()


def print_exception_trace(e):
    import traceback
    import sys
    print(''.join(traceback.format_exception(None, e, e.__traceback__)), file=sys.stdout, flush=True)
