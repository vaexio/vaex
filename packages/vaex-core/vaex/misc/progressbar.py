from __future__ import print_function
import os
import sys
import threading
import time
import multiprocessing
import vaex.utils

cpu_count = multiprocessing.cpu_count()


class ProgressBarBase(object):
    def __init__(self, min_value, max_value, title='vaex', format="%(percentage) 6.2f%% %(timeinfo)s cpu: %(cpu_usage)d%%"):
        self.min_value = min_value
        self.max_value = max_value
        self.format = format
        self.title = title
        self.value = self.min_value
        self.fraction = 0
        self.prevfraction = 0

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
        return {"percentage":percentage, "timeinfo":timeinfo, "cpu_usage": cpu_usage, "title": self.title, "cpu_count": cpu_count}

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



class Psutil(threading.Thread):
    disk = ""
    memory = ""
    net = ""
    wants_to_stop = False

    def stop(self):
        self.wants_to_stop = True

    def run(self):
        import psutil
        t0 = time.time()
        disk_prev = psutil.disk_io_counters()
        net_prev = psutil.net_io_counters()
        while not self.wants_to_stop:
            available = psutil.virtual_memory().available
            total = psutil.virtual_memory().total
            used = total - available
            usedp = int(used / total * 100)
            availablep = int(available / total * 100)

            total = vaex.utils.filesize_format(total)
            available = vaex.utils.filesize_format(available)
            used = vaex.utils.filesize_format(used)
            self.memory = f'Memory:  total={total} avail={available} ({availablep}%) used={used} ({usedp}%)'

            time.sleep(0.5)
            disk_curr = psutil.disk_io_counters()
            net_curr = psutil.net_io_counters()
            t1 = time.time()

            dt = t1 - t0
            read = disk_curr.read_bytes - disk_prev.read_bytes
            readps = vaex.utils.filesize_format(read / dt)
            write = disk_curr.write_bytes - disk_prev.write_bytes
            writeps = vaex.utils.filesize_format(write / dt)
            self.disk = f'Disk:    read={readps}/s write={writeps}/s'

            dt = t1 - t0
            read = net_curr.bytes_recv -  net_prev.bytes_recv
            readps = vaex.utils.filesize_format(read / dt)
            write = net_curr.bytes_sent - net_prev.bytes_sent
            writeps = vaex.utils.filesize_format(write / dt)
            self.net = f'Network: read={readps}/s write={writeps}/s'

            disk_prev = disk_curr
            net_prev = net_curr
            t0 = t1

        t0 = time.time()

    def __rich_console__(self, console, options):
        yield self.memory
        yield self.disk
        yield self.net



# TODO: make this configutable with new settings system
show_extra = vaex.utils.get_env_type(bool, 'VAEX_PROGRESS_EXTRA', False)

class ProgressBarRich(ProgressBarBase):
    def __init__(self, min_value, max_value, title=None, progress=None, indent=0, parent=None):
        super(ProgressBarRich, self).__init__(min_value, max_value)
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
                rich.progress.TimeRemainingColumn(),
                rich.progress.TimeElapsedColumn(),
                console=self.console,
                transient=False,
                expand=False,
            )
        else:
            self.progress = progress
        if parent is None:
            self.node = rich.tree.Tree(self.progress)
            # TODO: make this configutable
            if show_extra:
                self.info = Psutil(daemon=True)
                self.info.start()
            from rich.live import Live
            self.live = Live(self, refresh_per_second=3, console=self.console)
            self.live.start()
        else:
            self.node = parent.add(self.progress)
        # we do 1000 discrete steps
        self.steps = 0
        self.indent = indent

        if len(title) > 60:
            title = title[:60-3] + "..."
        padding = max(0, 50 - (self.indent * 4) - len(title))
        self.task = self.progress.add_task(f"[red]{title}" + (" " * padding), total=1000, start=False)
        self.started = False
        self.subtasks = []

    def __rich_console__(self, console, options):
        if show_extra:
            yield self.info
        yield self.node

    def add_child(self, parent, task, title):
        return ProgressBarRich(self.min_value, self.max_value, title, indent=self.indent+1, parent=self.node)

    def __call__(self, value):
        if not self.started:
            self.progress.start_task(self.task)
        if value > self.value:
            steps = int(value * 1000)
            delta = steps - self.steps
            if delta > 0:
                self.progress.update(self.task, advance=delta, refresh=False)
            self.steps = steps
        self.value = value

    def update(self, value):
        self(value)

    def finish(self):
        self(self.max_value)
        if self.parent is None:
            if show_extra:
                self.info.stop()
            self.live.stop()


import ipyvuetify as v
import traitlets
class ProgressDashboard(v.VuetifyTemplate):
    items = traitlets.Any({}).tag(sync=True)
    extra = traitlets.Any({}).tag(sync=True)
    selected = traitlets.Unicode(default_value=None, allow_none=True).tag(sync=True)
    @traitlets.default('template')
    def _template(self):
        return '''
            <template>
                <div>
                    <pre>{{extra.memory}}
{{extra.disk}}
{{extra.net}}</pre>
                    <v-treeview open-all dense :items="items">
                    <template v-slot:prepend="{ item, open }">
                        <v-progress-circular v-if='!item.finished' rotate="-90" :indeterminate="!item.started" :key="item.percentage" :size="30" :width="5" :value="item.percentage" :color="item.percentage < 100 ? 'primary' : 'green'">{{ item.started ? item.percentage : '' }}</v-progress-circular>
                        <v-icon color="green" v-if='item.finished'>mdi-checkbox-marked-circle</v-icon>
                    </template>
                    <template v-slot:label="{ item, open }">
                        {{item.title}} | {{item.timeinfo}} |
                        <span :class="item.cpu_usage / item.cpu_count < 33 ? 'vaex-cpu-low' : (item.cpu_usage / item.cpu_count < 66 ? 'vaex-cpu-medium' : 'vaex-cpu-high')">{{Math.ceil(item.cpu_usage)}}% cpu</span>
                        <!--
                        <v-progress-linear v-model="item.cpu_usage / item.cpu_count" color="blue-grey" height="25" style="width: 100px">
                            <template v-slot:default="{ value }">
                                <strong>cpu: {{ Math.ceil(value) }}%</strong>
                            </template>
                        </v-progress-linear>                        
                        -->
                    </template>
                    </v-treeview>
                </div>
            </template>
            <style id="vaex-progress-dashboard-style">
                .vaex-cpu-low {
                    color: red;
                }
                .vaex-cpu-medium {
                    color: orange;
                }
                .vaex-cpu-high {
                    color: green;
                }

            </style>

        '''

class ProgressBarVuetify(ProgressBarBase):
    def __init__(self, min_value, max_value, title=None, progress=None, indent=0, parent=None):
        if len(title) > 60:
            title = title[:60-3] + "..."
        super(ProgressBarVuetify, self).__init__(min_value, max_value, title=title)
        self.parent = parent
        self.data = {'children': [], 'name': self.title, 'started': False, 'finished': False}
        self.data.update(self.info())
        self.parent = parent
        if parent is None:
            self.psutil = Psutil(daemon=True)
            self.psutil.start()
            self.data
            self.dashboard = ProgressDashboard(items=[self.data])
            from IPython.display import display
            display(self.dashboard)
        else:
            self.dashboard = None
        self.steps = 0
        self.indent = indent


    def add_child(self, parent, task, title):
        child = ProgressBarVuetify(self.min_value, self.max_value, title, indent=self.indent+1, parent=self)
        self.data['children'].append(child.data)
        return child

    def __call__(self, value):
        if self.value == 0:
            self.info()  # initialize
        self.value = value
        self.update_fraction()
        info = self.info()
        info['percentage'] = int(info['percentage'])
        info['started'] = True
        info['finished'] = self.value == self.max_value
        self.data.update(info)
        if self.parent is None:
            self.dashboard.extra = {
                'memory': self.psutil.memory,
                'disk': self.psutil.disk,
                'net': self.psutil.net,
            }
            self.dashboard.send_state('items')

    def update(self, value):
        self(value)

    def finish(self):
        self(self.max_value)
        if self.parent is None:
            self.psutil.stop()
