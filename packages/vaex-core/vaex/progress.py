from typing import List
import warnings

from .utils import get_env_type, RegistryCallable
import vaex
import vaex.settings

_progressbar_registry = RegistryCallable("vaex.progressbar", "progressbar")


# these are registered in the entry_points
def simple(type=None, title="processing", max_value=1):
    import vaex.misc.progressbar as pb
    return pb.ProgressBar(0, 1, title=title)

def widget(type=None, title="processing", max_value=1):
    import vaex.misc.progressbar as pb
    return pb.ProgressBarWidget(0, 1, title=title)

def rich(type=None, title="processing", max_value=1):
    import vaex.misc.progressbar as pb
    return pb.ProgressBarRich(0, 1, title=title)


_last_progress_tree = None
_last_traceback = None
_progress_tree_stack = []

class ProgressTree:
    def __init__(self, children=None, next=None, bar=None, parent=None, name=None, hide=False):
        global _last_progress_tree, _last_traceback
        self.next = next
        self.children : List[ProgressTree] = children or list()
        self.finished = False
        self.last_fraction = None
        self.fraction = 0
        self.root = self if parent is None else parent.root
        self.passes_start = None
        self.bar = bar
        self.parent = parent
        self.name = name
        self.hide = hide
        self.cancelled = False
        self.oncancel = lambda: None
        if self.bar:
            self.bar.start()
        if parent is None and not hide:
            if _last_progress_tree is None:
                _last_progress_tree = self
                import traceback
                _last_traceback = ''.join(traceback.format_stack(limit=8))
            else:
                if vaex.DEBUG_MODE:
                    warnings.warn("Already a root progress tree exists, you may want to pass down the progress argument", stacklevel=4)
                    import traceback
                    trace = ''.join(traceback.format_stack(limit=7))
                    print('progress tree triggerd from:\n', trace)
                    print("====\nPrevious traceback:\n", _last_traceback)

    def cancel(self):
        self.cancelled = True

    def __repr__(self):
        name = self.__class__.__module__ + "." + self.__class__.__name__
        return "<%s(name=%r)> instance at 0x%x" % (name, self.name, id(self))

    def __enter__(self):
        _progress_tree_stack.append(self)
        if self.bar:
            self.bar.start()
        return self

    def __exit__(self, *args):
        _progress_tree_stack.pop()
        self.exit()

    def exit(self):
        global _last_progress_tree
        if self.parent is None:
            _last_progress_tree = None
        self(1)
        if self.bar:
            self.bar.exit()

    def exit_on(self, promise):
        def ok(arg):
            self.exit()
            return arg
        def error(arg):
            self.exit()
            raise arg
        return promise.then(ok, error)
        # return promise


    def hidden(self):
        '''Will add a child node that is hidden and not participate in progress calculations'''
        pb = ProgressTree(hide=True)
        return pb

    def add(self, name=None):
        pb = ProgressTree(parent=self, name=name)
        if self.bar and hasattr(self.bar, 'add_child') and not self.hide:
            pb.bar = self.bar.add_child(pb, None, name)
        self.children.append(pb)
        self.finished = False
        self.fraction = sum([c.fraction for c in self.children]) / len(self.children)
        self(self.fraction)
        if self.root.bar:
            # could be that it was stopped
            self.root.bar.start()

        return pb

    def add_task(self, task, name=None, status_ready="from cache"):
        pb = self.add(name)
        pb.oncancel = task.cancel
        task.signal_progress.connect(pb)
        if task.isFulfilled:
            pb(status_ready)
            pb(1)
        def on_error(exc):
            if pb.bar:
                pb.bar.set_status(str(exc))
                pb(pb.last_fraction)
                pb.finished = True
                if pb.root is not pb:
                    pb.root(None)
        def on_start(executor):
            passes = executor.passes
            if self.root.passes_start is None:
                self.root.passes_start = passes
            if pb.bar:
                pb.bar.set_passes(passes - self.root.passes_start + 1)
        task.signal_start.connect(on_start)
        task.then(None, on_error).end()
        return pb

    def __call__(self, fraction_or_status):
        if isinstance(fraction_or_status, str):
            self.status = fraction_or_status
            if self.bar:
                self.bar.set_status(self.status)
            return True
        fraction = fraction_or_status
        if fraction != 1.0:
            self.finished = False
        if self.cancelled:
            return False
        # ignore fraction
        result = True
        if len(self.children) == 0:
            self.fraction = fraction
        else:
            self.fraction = sum([c.fraction if not c.finished else 1 for c in self.children]) / len(self.children)
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


def bar(type_name=None, title="processing", max_value=1):
    if type_name is None:
        type_name = vaex.settings.main.progress.force
        if type_name is None:
            type_name = vaex.settings.main.progress.type
    return _progressbar_registry[type_name](title=title)


def tree(f=None, next=None, name=None, title=None):
    if f in [False, None] and _progress_tree_stack:
        f = _progress_tree_stack[-1]
    if f in [False, None] and vaex.settings.main.progress.force:
        f = vaex.settings.main.progress.force
    if name is None:
        name = title
    if title is None:
        name = title
    if isinstance(f, ProgressTree):
        # collapse nameless or similarly named progressbars in 1
        if title is None or f.bar and f.bar.title == title:
            return f
        else:
            return f.add(title)
    if callable(f):
        next = f
        f = False
    if f in [None, False]:
        return ProgressTree(next=next, name=name)
    else:
        if f is True:
            return ProgressTree(bar=bar(title=title), next=next, name=name)
        elif isinstance(f, str):
            return ProgressTree(bar=bar(f, title=title), next=next, name=name)
        else:
            return ProgressTree(next=next, name=name)
