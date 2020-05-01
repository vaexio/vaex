from __future__ import print_function
import os
import sys
import time


class ProgressBarBase(object):
    def __init__(self, min_value, max_value, format="%(percentage) 6.2f%% %(timeinfo)s cpu: %(cpu_usage)d%%"):
        self.min_value = min_value
        self.max_value = max_value
        self.format = format
        self.value = self.min_value
        self.fraction = 0
        self.prevfraction = 0

    def info(self):
        value = self.value
        if value == 0:
            self.prevtime = time.time()
            self.starttime = self.prevtime
            self.utime0, self.stime0 = os.times()[:2]
            self.walltime0 = time.time()
        currenttime = time.time()
        if (self.max_value-self.min_value) == 0:
            self.fraction = 1.
        else:
            self.fraction = (float(value)-self.min_value)/(self.max_value-self.min_value)
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
        return {"percentage":percentage, "timeinfo":timeinfo, "cpu_usage": cpu_usage}

    def __repr__(self):
        output = ''
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
    def __init__(self, min_value, max_value, format="%(percentage) 6.2f%% %(timeinfo)s", width=40, barchar="#", emptychar="-", output=sys.stdout):
        """		
            :param min_value: minimum value for update(..)
            :param format: format specifier for the output
            :param width: width of the progress bar's (excluding extra text)
            :param barchar: character used to print the bar
            :param output: where to write the output to
        """
        super(ProgressBar, self).__init__(min_value, max_value, format=format)
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
        count = int(round(self.fraction * self.width))
        space = self.width - count
        bar = "[" + (self.barchar * count) + (self.emptychar * space) + "]"
        output = "\r" + bar + super(ProgressBar, self).__repr__()
        if self.fraction == 1:
            output += "\n" # last time print a newline char
        return output


class ProgressBarWidget(ProgressBarBase):
    def __init__(self, min_value, max_value, name=None):
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
