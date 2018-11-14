import collections
import time
import functools
import ipywidgets as widgets
from IPython.display import display, clear_output


def get_ioloop():
    import IPython
    import zmq
    ipython = IPython.get_ipython()
    if ipython and hasattr(ipython, 'kernel'):
        return zmq.eventloop.ioloop.IOLoop.instance()


def debounced(delay_seconds=0.5, method=False):
    def wrapped(f):
        counters = collections.defaultdict(int)

        @functools.wraps(f)
        def execute(*args, **kwargs):
            if method:  # if it is a method, we want to have a counter per instance
                key = args[0]
            else:
                key = None
            counters[key] += 1

            def debounced_execute(counter=counters[key]):
                if counter == counters[key]:  # only execute if the counter wasn't changed in the meantime
                    f(*args, **kwargs)
            ioloop = get_ioloop()

            def thread_safe():
                ioloop.add_timeout(time.time() + delay_seconds, debounced_execute)
            if ioloop is None:  # not in IPython
                debounced_execute()
            else:
                ioloop.add_callback(thread_safe)
        return execute
    return wrapped


_selection_hooks = []


def interactive_cleanup():
    for dataset, f in _selection_hooks:
        dataset.signal_selection_changed.disconnect(f)


def interactive_selection(dataset):
    global _selection_hooks

    def wrapped(f_interact):
        if not hasattr(f_interact, "widget"):
            output = widgets.Output()

            def _selection_changed(dataset):
                with output:
                    clear_output(wait=True)
                    f_interact()
            hook = dataset.signal_selection_changed.connect(_selection_changed)
            _selection_hooks.append((dataset, hook))
            _selection_changed(dataset)
            display(output)
            return functools.wraps(f_interact)
        else:
            def _selection_changed(dataset):
                f_interact.widget.update()
            hook = dataset.signal_selection_changed.connect(_selection_changed)
            _selection_hooks.append((dataset, hook))
            return functools.wraps(f_interact)
    return wrapped
