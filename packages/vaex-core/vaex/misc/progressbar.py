from __future__ import print_function
import os
import sys
import time


class ProgressBarBase(object):
    def __init__(self, min_value, max_value, title='vaex', format="%(percentage) 6.2f%% %(timeinfo)s cpu: %(cpu_usage)d%%"):
        if title is None:
            title = "vaex"
        self.min_value = min_value
        self.max_value = max_value
        self.format = format
        self.title = title
        self.value = self.min_value
        self.fraction = 0
        self.prevfraction = 0
        self.status = None

    def set_passes(self, passes):
        pass

    def start(self):
        pass

    def exit(self):
        pass

    def set_status(self, status):
        self.status = status

    def update_fraction(self):
        if (self.max_value-self.min_value) == 0:
            self.fraction = 1.
        else:
            self.fraction = (float(self.value)-self.min_value)/(self.max_value-self.min_value)

    def info(self):
        value = self.value
        if value == 0:
            self.prevtime = time.time()
            self.starttime = self.prevtime
            self.utime0, self.stime0 = os.times()[:2]
            self.walltime0 = time.time()
        currenttime = time.time()
        percentage = self.fraction * 100
        self.utime, self.stime = os.times()[:2]
        self.walltime  = time.time()
        cpu_time_delta = self.stime - self.stime0 + self.utime - self.utime0
        wall_time_delta = self.walltime - self.walltime0
        if wall_time_delta == 0:
            cpu_usage = 0  # it's formally infinity
        else:
            cpu_usage = cpu_time_delta / wall_time_delta * 100

        if (self.fraction > 0) and ((self.fraction-self.prevfraction) > 0):
            if self.fraction == 1:
                elapsedtime = (currenttime-self.starttime)
                seconds = elapsedtime
                minutes = seconds/60.
                hours = minutes/60.
                timeinfo = "elapsed time  : % 8.2fs = % 4.1fm = % 2.1fh" % (seconds, minutes, hours)
            else:
                #estimatedtime = (currenttime-self.starttime)/(self.fraction) * (1-self.fraction)
                #estimatedtime = (currenttime-self.prevtime)/(self.fraction-self.prevself.fraction) * (1-self.fraction)
                estimatedtime = (currenttime-self.prevtime) / (self.fraction) * (1-self.fraction)
                seconds = estimatedtime
                minutes = seconds/60.
                hours = minutes/60.
                timeinfo = "estimated time: % 8.2fs = % 4.1fm = % 2.1fh" % (seconds, minutes, hours)
        else:
            timeinfo = "estimated time: unknown                "
        return {"percentage":percentage, "timeinfo":timeinfo, "cpu_usage": cpu_usage, "title": self.title}

    def __repr__(self):
        output = ''
        self.update_fraction()
        return self.format % self.info()


class ProgressBar(ProgressBarBase):
    """
        Implementation of progress bar on a console

        Usage::

            progressbar = ProgressBar(0, 100)
            progressbar.update(0)
            ....
            progressbar.update(100)

        By default, the progress bar writes to stderr, so it doesn't clutter up log files when piping stdout
    """
    def __init__(self, min_value, max_value, title="vaex", format="%(percentage) 6.2f%% %(timeinfo)s", width=40, barchar="#", emptychar="-", output=sys.stdout):
        """		
            :param min_value: minimum value for update(..)
            :param format: format specifier for the output
            :param width: width of the progress bar's (excluding extra text)
            :param barchar: character used to print the bar
            :param output: where to write the output to
        """
        super(ProgressBar, self).__init__(min_value, max_value, format=format, title=title)
        self.width = width
        self.barchar = barchar
        self.emptychar = emptychar
        self.output = output

    def update(self, value):
        self.value = value
        print(repr(self), file=self.output, end=' ')
        self.output.flush()

    def finish(self):
        if self.value != self.max_value:
            self.value = self.max_value
            print(repr(self), file=self.output, end=' ')
        self.output.flush()

    def __repr__(self):
        output = ''
        if self.status:
            bar = f"[{self.status}".ljust(self.width-1, ' ') + ']'
        else:
            self.update_fraction()
            count = int(round(self.fraction * self.width))
            space = self.width - count
            bar = self.title + " [" + (self.barchar * count) + (self.emptychar * space) + "]"
        output = "\r" + bar + super(ProgressBar, self).__repr__()
        if self.fraction == 1:
            output += "\n" # last time print a newline char
        return output


class ProgressBarWidget(ProgressBarBase):
    def __init__(self, min_value, max_value, title=None):
        super(ProgressBarWidget, self).__init__(min_value, max_value)
        import ipywidgets as widgets
        from IPython.display import display
        self.bar = widgets.FloatProgress(min=self.min_value, max=self.max_value)
        self.text = widgets.Label(value='In progress...')
        self.widget = widgets.HBox([self.bar, self.text])
        # self.widget.description = repr(self)
        display(self.widget)

    def __call__(self, value):
        self.value = value
        self.bar.value = value
        self.text.value = repr(self)

    def update(self, value):
        self(value)

    def finish(self):
        self(self.max_value)

import rich.progress
from rich.text import Text


class TimeElapsedColumn(rich.progress.ProgressColumn):
    """Renders time elapsed."""

    def render(self, task: "Task") -> Text:
        """Show time remaining."""
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text("-:--:--", style="progress.elapsed")
        from datetime import timedelta
        delta = timedelta(seconds=(elapsed))
        time = str(delta)[:-4]
        if time.startswith('0:00:'):
            time = time[5:]
        time = time + 's'
        passes = task.fields.get('passes')
        if passes is not None:
            time += f'[{passes}]'
        else:
            time += '   '
        return Text(time, style="progress.elapsed")


class ProgressBarRich(ProgressBarBase):
    def __init__(self, min_value, max_value, title=None, progress=None, indent=0, parent=None):
        super(ProgressBarRich, self).__init__(min_value, max_value, title=title)
        import rich.progress
        import rich.table
        import rich.tree
        self.console = rich.console.Console(record=True)
        self.parent = parent
        if progress is None:
            self.progress = rich.progress.Progress(
                rich.progress.SpinnerColumn(),
                rich.progress.TextColumn("[progress.description]{task.description}"),
                rich.progress.BarColumn(),
                rich.progress.TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                # rich.progress.TimeRemainingColumn(),
                TimeElapsedColumn(),
                rich.progress.TextColumn("[red]{task.fields[status]}"),
                console=self.console,
                transient=False,
                expand=False,
            )
        else:
            self.progress = progress
        if parent is None:
            self.node = rich.tree.Tree(self.progress)
            from rich.live import Live
            self.live = Live(self.node, refresh_per_second=5, console=self.console)
        else:
            self.node = parent.add(self.progress)
        # we do 1000 discrete steps
        self.steps = 0
        self.indent = indent

        padding = max(0, 45- (self.indent * 4) - len(self.title))
        self.passes = None
        self.task = self.progress.add_task(f"[red]{self.title}" + (" " * padding), total=1000, start=False, status=self.status or '', passes=self.passes)
        self.started = False
        self.subtasks = []

    def add_child(self, parent, task, title):
        return ProgressBarRich(self.min_value, self.max_value, title, indent=self.indent+1, parent=self.node)

    def __call__(self, value):
        if not self.started:
            self.progress.start_task(self.task)
        if value > self.value:
            steps = int(value * 1000)
            delta = steps - self.steps
            if delta > 0:
                self.progress.update(self.task, advance=delta, refresh=False, passes=self.passes)
            else:
                start_time = self.progress.tasks[0].start_time
                self.progress.reset(self.task, completed=steps, refresh=False, status=self.status or '')
                self.progress.tasks[0].start_time = start_time
            self.steps = steps
        self.value = value

    def update(self, value):
        self(value)

    def finish(self):
        self(self.max_value)
        if self.parent is None:
            self.live.refresh()

    def start(self):
        if self.parent is None and not self.live.is_started:
            self.live.refresh()
            self.live.start()

    def exit(self):
        if self.parent is None:
            self.live.stop()

    def set_status(self, status):
        self.status = status
        self.progress.update(self.task, status=self.status)

    def set_passes(self, passes):
        self.passes = passes
