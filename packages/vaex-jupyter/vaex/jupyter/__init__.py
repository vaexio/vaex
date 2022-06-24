# -*- coding: utf-8 -*-
import os
import logging
import time
from .utils import debounced, flush, gather, kernel_tick, interactive_selection, interactive_cleanup  # noqa
import vaex
import IPython.display


base_path = os.path.dirname(__file__)
logger = logging.getLogger("vaex.jupyter")


def _add_toolbar(viz):
    from .widgets import ToolsToolbar, tools_items_default
    from traitlets import link
    interact_items = [k for k in tools_items_default if k['value'] in viz.TOOLS_SUPPORTED]
    toolbar = ToolsToolbar(supports_transforms=viz.supports_transforms,
                           supports_normalize=viz.supports_normalize,
                           interact_items=interact_items)
    viz.children = [toolbar, ] + viz.children
    link((viz, 'tool'), (toolbar, 'interact_value'))
    link((viz, 'transform'), (toolbar, 'transform_value'))
    link((viz, 'normalize'), (toolbar, 'normalize'))
    link((viz, 'selection_mode'), (toolbar, 'selection_mode'))
    return toolbar


class DataFrameAccessorWidget(object):
    def __init__(self, df):
        self.df = df
        import vaex.jupyter.grid
        self.grid = vaex.jupyter.model.GridCalculator(df, [])
        self._last_grid = None

    @debounced(delay_seconds=0.1, reentrant=False)
    async def execute_debounced(self):
        """Schedules an execution of dataframe tasks in the near future (debounced)."""
        try:
            logger.debug("Execute tasks... tasks=%r", self.df.executor.tasks)
            await self.df.execute_async()
            logger.debug("Execute tasks done")
        except vaex.execution.UserAbort:
            pass  # this is fine
        except Exception:
            logger.exception("Error while executing tasks")

    def clear(self):
        self.grid = vaex.jupyter.model.GridCalculator(self.df, [])

    def data_array(self, axes=[], selection=None, shared=False, display_function=IPython.display.display, **kwargs):
        '''Create a :func:`vaex.jupyter.model.DataArray` model and :func:`vaex.jupyter.view.DataArray` widget and links them.

        This is a convenience method to create the model and view, and hook them up.
        '''
        import vaex.jupyter.model
        import vaex.jupyter.view
        if selection is not None:
            selection = selection.copy()
        model = vaex.jupyter.model.DataArray(df=self.df, axes=axes, selection=selection, **kwargs)
        if shared:
            grid = self.grid
        else:
            grid = vaex.jupyter.model.GridCalculator(self.df, [])
        grid.model_add(model)
        view = vaex.jupyter.view.DataArray(model=model, display_function=display_function)
        return view

    def axis_model(self, expression, limits=None):
        return self._axes([expression], limits=[limits])[0]

    def _axes(self, expressions, limits):
        limits = self.df.limits(expressions, limits)
        axes = [vaex.jupyter.model.Axis(df=self.df, expression=expression, min=min, max=max) for expression, (min, max) in zip(expressions, limits)]
        return axes

    def histogram(self, x, limits=None, selection=None, selection_interact='default', toolbar=True, shared=False, **kwargs):
        import vaex.jupyter.model
        import vaex.jupyter.view
        if selection is not None:
            selection = selection.copy()
        x, = self._axes([x], limits)
        model = vaex.jupyter.model.Histogram(df=self.df, x=x, selection=selection, selection_interact=selection_interact, **kwargs)
        if shared:
            grid = self.grid
        else:
            grid = vaex.jupyter.model.GridCalculator(self.df, [])
        grid.model_add(model)
        viz = vaex.jupyter.view.Histogram(model=model)
        if toolbar:
            viz.toolbar = _add_toolbar(viz)
        return viz

    def pie(self, x, limits=None, shared=False, **kwargs):
        import vaex.jupyter.model
        import vaex.jupyter.view
        x, = self._axes([x], limits)
        model = vaex.jupyter.model.Histogram(df=self.df, x=x, **kwargs)
        if shared:
            grid = self.grid
        else:
            grid = vaex.jupyter.model.GridCalculator(self.df, [])
        grid.model_add(model)
        viz = vaex.jupyter.view.PieChart(model=model)
        return viz

    def heatmap(self, x, y, limits=None, selection=None, selection_interact='default', transform='log', toolbar=True, shape=256, shared=False, **kwargs):
        import vaex.jupyter.model
        import vaex.jupyter.view
        x, y = self._axes([x, y], limits)
        if selection is not None:
            selection = selection.copy()
        model = vaex.jupyter.model.Heatmap(df=self.df, x=x, y=y, selection=selection, shape=shape, **kwargs)
        if shared:
            grid = self.grid
        else:
            grid = vaex.jupyter.model.GridCalculator(self.df, [])
        self._last_grid = grid
        grid.model_add(model)
        viz = vaex.jupyter.view.Heatmap(model=model, transform=transform)
        if toolbar:
            viz.toolbar = _add_toolbar(viz)
        return viz

    def expression(self, value=None, label='Custom expression'):
        '''Create a widget to edit a vaex expression.

        If value is an :class:`vaex.jupyter.model.Axis` object, its expression will be (bi-directionally) linked to the widget.

        :param value: Valid expression (string or Expression object), or Axis
        '''
        from .widgets import ExpressionTextArea
        import vaex.jupyter.model
        if isinstance(value, vaex.jupyter.model.Axis):
            expression_value = str(value.expression)
        else:
            expression_value = str(value) if value is not None else None
        expression_widget = ExpressionTextArea(df=self.df, v_model=expression_value, label=label)
        if isinstance(value, vaex.jupyter.model.Axis):
            import traitlets
            traitlets.link((value, 'expression'), (expression_widget, 'value'))
        return expression_widget

    def column(self, value=None, label='Choose a column'):
        from .widgets import ColumnPicker
        if isinstance(value, vaex.jupyter.model.Axis):
            expression_value = str(value.expression)
        else:
            expression_value = str(value) if value is not None else None
        column_widget = ColumnPicker(df=self.df, value=expression_value, label=label)
        if isinstance(value, vaex.jupyter.model.Axis):
            import traitlets
            traitlets.link((value, 'expression'), (column_widget, 'value'))
        return column_widget

    def selection_expression(self, initial_value=None, name='default'):
        from .widgets import ExpressionSelectionTextArea
        if initial_value is None:
            if not self.df.has_selection(name):
                raise ValueError(f'No selection with name {name!r}')
            else:
                initial_value = self.df.get_selection(name).boolean_expression
        return ExpressionSelectionTextArea(df=self.df, selection_name=name, v_model=str(initial_value) if initial_value is not None else None)

    def progress_circular(self, width=10, size=70, color='#82B1FF', text='', auto_hide=False):
        from .widgets import ProgressCircularNoAnimation
        progress_circular = ProgressCircularNoAnimation(width=width, size=size, color=color, text=text, value=0)

        @self.df.executor.signal_begin.connect
        def progress_begin():
            if auto_hide:
                progress_circular.hidden = False

        @self.df.executor.signal_progress.connect
        def update_progress(value):
            progress_circular.value = value*100
            return True

        @self.df.executor.signal_end.connect
        def progress_update():
            if auto_hide:
                progress_circular.hidden = True
        return progress_circular

    def counter_processed(self, postfix="rows processed", update_interval=0.2):
        from .widgets import Counter
        counter_processed = Counter(value=0, postfix=postfix)
        last_time = 0

        @self.df.executor.signal_begin.connect
        def progress_begin():
            nonlocal last_time
            last_time = time.time()

        @self.df.executor.signal_progress.connect
        def update_progress(value):
            nonlocal last_time
            number = int(value * len(self.df))
            current_time = time.time()
            if (current_time - last_time) > update_interval or value in [0, 1]:
                counter_processed.value = number
                last_time = current_time
            return True

        return counter_processed

    def counter_selection(self, selection, postfix="rows selected", update_interval=0.2, lazy=False):
        from .widgets import Counter
        selected = self.df.count(selection=selection).item() if self.df.has_selection(name=selection) else 0
        counter_selected = Counter(value=selected, postfix=postfix)

        dirty = False
        @self.df.signal_selection_changed.connect
        def selection_changed(df, name):
            nonlocal dirty
            if name == selection:
                # we only need to run once
                if not dirty:
                    dirty = True

                    def update_value(value):
                        nonlocal dirty
                        dirty = False
                        try:
                            value = value.item()
                        except:  # noqa
                            pass
                        counter_selected.value = value
                    # if lazy is True, this will only schedule the calculation, not yet execute it
                    if lazy:
                        vaex.delayed(update_value)(self.df.count(selection=selection, delay=True))
                    else:
                        update_value(self.df.count(selection=selection))

        return counter_selected
    #     from .widgets import Tools
    #     from traitlets import link
    #     viz = [] if viz is None else viz
    #     viz = [viz] if not isinstance(viz, (tuple, list)) else viz
    #     tools = Tools(value=initial_value, children=[k.widget for k in viz])
    #     for v in viz:
    #         link((tools, 'value'), (v, 'tool'))
    #     return tools

    # def card(plot, title=None, subtitle=None, **kwargs):
    #     from .widget import Card
    #     return Card(main=plot, title=title, subtitle,


def add_namespace():
    pass
