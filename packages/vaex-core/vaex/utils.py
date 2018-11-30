# -*- coding: utf-8 -*-
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

import numpy as np
import progressbar
import psutil
import six
import yaml

is_frozen = getattr(sys, 'frozen', False)
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

osname = dict(darwin="osx", linux="linux", windows="windows")[platform.system().lower()]


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, key, value):
        self[key] = value


def deprecated(reason):
    def wraps(f):
        @functools.wraps(f)
        def wraps2(*args, **kwargs):
            warnings.warn("Call to deprecated function {}: {}".format(f.__name__, reason),
                          category=DeprecationWarning, stacklevel=2)
            return f(*args, **kwargs)

        return wraps2

    return wraps


def subdivide(length, parts=None, max_length=None):
    """Generates a list with start end stop indices of length parts, [(0, length/parts), ..., (.., length)]"""
    if max_length:
        i1 = 0
        done = False
        while not done:
            i2 = min(length, i1 + max_length)
            # print i1, i2
            yield i1, i2
            i1 = i2
            if i1 == length:
                done = True
    else:
        part_length = int(math.ceil(float(length) / parts))
        # subblock_count = math.ceil(total_length/subblock_size)
        # args_list = []
        for index in range(parts):
            i1, i2 = index * part_length, min(length, (index + 1) * part_length)
            yield i1, i2


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


def get_data_file(filename):
    try:  # this works for egg like stuff, but fails for py2app apps
        from pkg_resources import Requirement, resource_filename
        path = resource_filename(Requirement.parse("vaex"), filename)
        if os.path.exists(path):
            return path
    except:
        pass
    # this is where we expect data to be in normal installations
    for extra in ["", "data", "data/dist", "../data", "../data/dist"]:
        path = os.path.join(os.path.dirname(__file__), "..", "..", extra, filename)
        if os.path.exists(path):
            return path
        path = os.path.join(sys.prefix, extra, filename)
        if os.path.exists(path):
            return path
        # if all fails..
        path = os.path.join(get_root_path(), extra, filename)
        if os.path.exists(path):
            return path


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


def get_private_dir(subdir=None):
    path = os.path.expanduser('~/.vaex')
    if subdir:
        path = os.path.join(path, subdir)
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


class _ProgressBarWidget(object):
    def __init__(self, name=None):
        import ipywidgets as widgets
        from IPython.display import display
        self.value = 0
        self.widget = widgets.FloatProgress(min=0, max=1)
        self.widget.description = name or 'Progress:'
        display(self.widget)

    def __call__(self, value):
        self.value = value
        self.widget.value = value

    def update(self, value):
        self(value)

    def finish(self):
        self.widget.description = 'Finished:'
        self(1)

    # def add_child(self, progress, task, name):
    # 	#print('add', name)
    # 	pb = _ProgressBarWidget(name=name)
    # 	task.signal_progress.connect(pb)


_progressbar_typemap = {}
_progressbar_typemap['progressbar2'] = _progressbar_progressbar2
_progressbar_typemap['vaex'] = _progressbar_vaex
_progressbar_typemap['widget'] = _ProgressBarWidget


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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bytes_):
            return obj.decode('UTF-8')
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8');

        else:
            return super(NumpyEncoder, self).default(obj)


def write_json_or_yaml(filename, data):
    base, ext = os.path.splitext(filename)
    if ext == ".json":
        with open(filename, "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
    elif ext == ".yaml":
        with open(filename, "w") as f:
            yaml_dump(f, data)
    else:
        raise ValueError("file should end in .json or .yaml (not %s)" % ext)


def read_json_or_yaml(filename):
    base, ext = os.path.splitext(filename)
    if ext == ".json":
        with open(filename, "r") as f:
            return json.load(f) or {}
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
    if isinstance(args[0], six.string_types):
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
    name = re.sub("[^a-zA-Z_]", "_", first) + re.sub("[^a-zA-Z_0-9]", "_", rest)
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
