import asyncio
import enum
import typing

import traitlets
import numpy as np
import xarray

import vaex
import vaex.jupyter
from .decorators import signature_has_traits
from .traitlets import Expression
import logging
from .vendor import contextlib


logger = logging.getLogger('vaex.jupyter.model')


class _HasState(traitlets.HasTraits):
    _debug = traitlets.Bool(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # if self._debug:
        self._allow_state_change = asyncio.Semaphore(0)
        self._current_status_wait_future = None

    # def _cancel_computation(self):
    #     pass

    def _debug_wait_for_status(self, status):
        self._current_status_wait_future = asyncio.Future()

        def on_status_change(change):
            if change['new'] == status:
                self._current_status_wait_future.set_result(None)
                self.unobserve(on_status_change, 'status')
            else:
                self._current_status_wait_future.set_exception(RuntimeError(f'Did not expect change to status: {change.new}'))
        self.observe(on_status_change, 'status')
        return self._current_status_wait_future

    def _error(self, e):
        if self._debug:
            if self._current_status_wait_future:
                self._current_status_wait_future.set_exception(e)
        if isinstance(e, asyncio.CancelledError):
            print("cancelled")
        else:
            print("Error: ", e, type(e))
            try:
                vaex.utils.print_exception_trace(e)
            except Exception as e:
                print(e)

    async def _allow_state_change_to(self, status):
        # print(f"allowing state change from {self.status} to {status}")
        self._allow_state_change.release()
        result = await self._debug_wait_for_status(status)
        return result

    async def _allow_state_change_cancel(self, previous=False):
        # print(f"allowing cancel")
        self._allow_state_change.release()
        await self.computation._await_last_call(previous)

    @contextlib.asynccontextmanager
    async def _state_change_to(self, new_status):
        current_status = self.status
        logger.debug(f'Current state is {self.status}')
        if self._debug:
            # print(f"waiting to allow a state change - from {current_status} to {new_status}")
            await self._allow_state_change.acquire()
        yield
        if current_status == self.status:
            self.status = new_status
        else:
            pass
            raise asyncio.CancelledError(f"Status expected to be {current_status}, but is {self.status}")
        logger.debug(f'State change {type(self)} from {self.status} to {new_status}')
        self.status = new_status


@signature_has_traits
class Axis(_HasState):
    class Status(enum.Enum):
        """
        State transitions
        NO_LIMITS -> STAGED_CALCULATING_LIMITS -> CALCULATING_LIMITS -> CALCULATED_LIMITS -> READY

        when expression changes:
            STAGED_CALCULATING_LIMITS: 
                calculation.cancel()
                ->NO_LIMITS
            CALCULATING_LIMITS: 
                calculation.cancel()
                ->NO_LIMITS

        when min/max changes:
            STAGED_CALCULATING_LIMITS: 
                calculation.cancel()
                ->NO_LIMITS
            CALCULATING_LIMITS: 
                calculation.cancel()
                ->NO_LIMITS
        """
        NO_LIMITS = 1
        STAGED_CALCULATING_LIMITS = 2
        CALCULATING_LIMITS = 3
        CALCULATED_LIMITS = 4
        READY = 5
        EXCEPTION = 6
        ABORTED = 7
    status = traitlets.UseEnum(Status, Status.NO_LIMITS)
    df = traitlets.Instance(vaex.dataframe.DataFrame)
    expression = Expression()
    slice = traitlets.CInt(None, allow_none=True)
    min = traitlets.CFloat(None, allow_none=True)
    max = traitlets.CFloat(None, allow_none=True)
    bin_centers = traitlets.Any()
    shape = traitlets.CInt(None, allow_none=True)
    shape_default = traitlets.CInt(64)
    _calculation = traitlets.Any(None, allow_none=True)
    exception = traitlets.Any(None, allow_none=True)
    _status_change_delay = traitlets.Float(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.min is not None and self.max is not None:
            self.status = Axis.Status.READY
            self._calculate_centers()
        else:
            self.computation()
        self.observe(self.on_change_expression, 'expression')
        self.observe(self.on_change_shape, 'shape')
        self.observe(self.on_change_shape_default, 'shape_default')

    def __repr__(self):
        def myrepr(value, key):
            if isinstance(value, vaex.expression.Expression):
                return str(value)
            return value
        args = ', '.join('{}={}'.format(key, myrepr(getattr(self, key), key)) for key in self.traits().keys() if key != 'df' and not key.startswith('_'))
        return '{}({})'.format(self.__class__.__name__, args)

    @property
    def has_missing_limit(self):
        # return not self.df.is_category(self.expression) and (self.min is None or self.max is None)
        return (self.min is None or self.max is None)

    def on_change_expression(self, change):
        self.min = None
        self.max = None
        self.status = Axis.Status.NO_LIMITS
        if self._calculation is not None:
            self._cancel_computation()
        self.computation()

    def on_change_shape(self, change):
        if self.min is not None and self.max is not None:
            self._calculate_centers()

    def on_change_shape_default(self, change):
        if self.min is not None and self.max is not None:
            self._calculate_centers()

    def _cancel_computation(self):
        self._continue_calculation = False

    @traitlets.observe('min', 'max')
    def on_change_limits(self, change):
        if self.min is not None and self.max is not None:
            self._calculate_centers()
        if self.status == Axis.Status.NO_LIMITS:
            if self.min is not None and self.max is not None:
                self.status = Axis.Status.READY
        elif self.status == Axis.Status.READY:
            if self.min is None or self.max is None:
                self.status = Axis.Status.NO_LIMITS
            else:
                # in this case, grids may want to be computed
                # this happens when a user change min/max
                pass
        else:
            if self._calculation is not None:
                self._cancel_computation()
                if self.min is not None and self.max is not None:
                    self.status = Axis.Status.READY
                else:
                    self.status = Axis.Status.NO_LIMITS
            else:
                # in this case we've set min/max after the calculation
                assert self.min is not None or self.max is not None

    @vaex.jupyter.debounced(delay_seconds=0.1, reentrant=False, on_error=_HasState._error)
    async def computation(self):
        categorical = self.df.is_category(self.expression)
        if categorical:
            N = self.df.category_count(self.expression)
            self.min, self.max = -0.5, N-0.5
            # centers = np.arange(N)
            # self.shape = N
            self._calculate_centers()
            self.status = Axis.Status.READY
        else:
            try:

                self._continue_calculation = True
                self._calculation = self.df.minmax(self.expression, delay=True, progress=self._progress)
                self.df.widget.execute_debounced()
                # keep a nearly reference to this, since awaits (which trigger the execution, AND reset of this future) may change it this
                execute_prehook_future = self.df.widget.execute_debounced.pre_hook_future
                async with self._state_change_to(Axis.Status.STAGED_CALCULATING_LIMITS):
                    pass
                async with self._state_change_to(Axis.Status.CALCULATING_LIMITS):
                    await execute_prehook_future
                async with self._state_change_to(Axis.Status.CALCULATED_LIMITS):
                    vmin, vmax = await self._calculation
                # indicate we are done with the calculation
                self._calculation = None
                if not self._continue_calculation:
                    assert self.status == Axis.Status.READY
                async with self._state_change_to(Axis.Status.READY):
                    self.min, self.max = vmin, vmax
                    self._calculate_centers()
            except vaex.execution.UserAbort:
                # probably means expression or min/max changed, we don't have to take action
                pass
            except asyncio.CancelledError:
                pass

    def _progress(self, f):
        # we use the progres callback to cancel as calculation
        return self._continue_calculation

    def _calculate_centers(self):
        categorical = self.df.is_category(self.expression)
        if categorical:
            N = self.df.category_count(self.expression)
            centers = np.arange(N)
            self.shape = N
        else:
            centers = self.df.bin_centers(self.expression, [self.min, self.max], shape=self.shape or self.shape_default)
        self.bin_centers = centers


@signature_has_traits
class DataArray(_HasState):
    class Status(enum.Enum):
        MISSING_LIMITS = 1
        STAGED_CALCULATING_LIMITS = 3
        CALCULATING_LIMITS = 4
        CALCULATED_LIMITS = 5
        NEEDS_CALCULATING_GRID = 6
        STAGED_CALCULATING_GRID = 7
        CALCULATING_GRID = 8
        CALCULATED_GRID = 9
        READY = 10
        EXCEPTION = 11
    status = traitlets.UseEnum(Status, Status.MISSING_LIMITS)
    status_text = traitlets.Unicode('Initializing')
    exception = traitlets.Any(None)
    df = traitlets.Instance(vaex.dataframe.DataFrame)
    axes = traitlets.List(traitlets.Instance(Axis), [])
    grid = traitlets.Instance(xarray.DataArray, allow_none=True)
    grid_sliced = traitlets.Instance(xarray.DataArray, allow_none=True)
    shape = traitlets.CInt(64)
    selection = traitlets.Any(None)

    def __init__(self, **kwargs):
        super(DataArray, self).__init__(**kwargs)
        self.signal_slice = vaex.events.Signal()
        self.signal_regrid = vaex.events.Signal()
        self.signal_grid_progress = vaex.events.Signal()
        self.observe(lambda change: self.signal_regrid.emit(), 'selection')
        self._on_axis_status_change()

        # keep a set of axis that need new limits
        self._dirty_axes = set()
        for axis in self.axes:
            assert axis.df is self.df, "axes should have the same dataframe"
            traitlets.link((self, 'shape'), (axis, 'shape_default'))
            axis.observe(self._on_axis_status_change, 'status')
            axis.observe(lambda _: self.signal_slice.emit(self), ['slice'])

            def on_change_min_max(change):
                if change.owner.status == Axis.Status.READY:
                    # this indicates a user changed the min/max
                    self.status = DataArray.Status.NEEDS_CALCULATING_GRID
            axis.observe(on_change_min_max, ['min', 'max'])

        self._on_axis_status_change()
        self.df.signal_selection_changed.connect(self._on_change_selection)

    def _on_change_selection(self, df, name):
        # TODO: check if the selection applies to us
        def _translate_selection(selection):
            if selection in [None, False]:
                return None
            if selection is True:
                return 'default'
            else:
                return selection
        if name == _translate_selection(self.selection) or (isinstance(self.selection, (list, tuple)) and name in [_translate_selection(k) for k in self.selection]):
            self.status = DataArray.Status.NEEDS_CALCULATING_GRID

    async def _allow_state_change_cancel(self):
        self._allow_state_change.release()

    def _on_axis_status_change(self, change=None):
        missing_limits = [axis for axis in self.axes if axis.status == Axis.Status.NO_LIMITS]
        staged_calculating_limits = [axis for axis in self.axes if axis.status == Axis.Status.STAGED_CALCULATING_LIMITS]
        calculating_limits = [axis for axis in self.axes if axis.status == Axis.Status.CALCULATING_LIMITS]
        calculated_limits = [axis for axis in self.axes if axis.status == Axis.Status.CALCULATED_LIMITS]

        def names(axes):
            return ", ".join([str(axis.expression) for axis in axes])

        if staged_calculating_limits:
            self.status = DataArray.Status.STAGED_CALCULATING_LIMITS
            self.status_text = 'Staged limit computation for {}'.format(names(staged_calculating_limits))
        elif missing_limits:
            self.status = DataArray.Status.MISSING_LIMITS
            self.status_text = 'Missing limits for {}'.format(names(missing_limits))
        elif calculating_limits:
            self.status = DataArray.Status.CALCULATING_LIMITS
            self.status_text = 'Computing limits for {}'.format(names(calculating_limits))
        elif calculated_limits:
            self.status = DataArray.Status.CALCULATED_LIMITS
            self.status_text = 'Computed limits for {}'.format(names(calculating_limits))
        else:
            assert all([axis.status == Axis.Status.READY for axis in self.axes])
            self.status = DataArray.Status.NEEDS_CALCULATING_GRID

    @traitlets.observe('status')
    def _on_change_status(self, change):
        if self.status == DataArray.Status.EXCEPTION:
            self.status_text = f'Exception: {self.exception}'
        elif self.status == DataArray.Status.NEEDS_CALCULATING_GRID:
            self.status_text = 'Grid needs to be calculated'
        elif self.status == DataArray.Status.STAGED_CALCULATING_GRID:
            self.status_text = 'Staged grid computation'
        elif self.status == DataArray.Status.CALCULATING_GRID:
            self.status_text = 'Calculating grid'
        elif self.status == DataArray.Status.CALCULATED_GRID:
            self.status_text = 'Calculated grid'
        elif self.status == DataArray.Status.READY:
            self.status_text = 'Ready'
        # GridCalculator can change the status
            # self._update_grid()
            # self.status_text = 'Computing limits for {}'.format(names(missing_limits))

    @property
    def has_missing_limits(self):
        return any([axis.has_missing_limit for axis in self.axes])

    def on_progress_grid(self, f):
        return all(self.signal_grid_progress.emit(f))


class Histogram(DataArray):
    x = traitlets.Instance(Axis)
    # type = traitlets.CaselessStrEnum(['count', 'min', 'max', 'mean'], default_value='count')
    # groupby = traitlets.Instance(Axis)
    # groupby_normalize = traitlets.Bool(False, allow_none=True)
    # grid = traitlets.Any()
    # grid_sliced = traitlets.Any()

    def __init__(self, **kwargs):
        kwargs['axes'] = [kwargs['x']]
        super().__init__(**kwargs)


class Heatmap(DataArray):
    x = traitlets.Instance(Axis)
    y = traitlets.Instance(Axis)

    def __init__(self, **kwargs):
        kwargs['axes'] = [kwargs['x'], kwargs['y']]
        super().__init__(**kwargs)


class GridCalculator(_HasState):
    '''A grid is responsible for scheduling the grid calculations and possible slicing'''
    class Status(enum.Enum):
        VOID = 1
        STAGED_CALCULATION = 3
        CALCULATING = 4
        READY = 9
    status = traitlets.UseEnum(Status, Status.VOID)
    df = traitlets.Instance(vaex.dataframe.DataFrame)
    models = traitlets.List(traitlets.Instance(DataArray))
    _calculation = traitlets.Any(None, allow_none=True)
    _debug = traitlets.Bool(False)

    def __init__(self, df, models):
        super().__init__(df=df, models=[])
        self._callbacks_regrid = []
        self._callbacks_slice = []
        for model in models:
            self.model_add(model)
        self._testing_exeception_regrid = False  # used for testing, to throw an exception
        self._testing_exeception_reslice = False  # used for testing, to throw an exception

    # def model_remove(self, model, regrid=True):
    #     index = self.models.index(model)
    #     del self.models[index]
    #     del self._callbacks_regrid[index]
    #     del self._callbacks_slice[index]

    def model_add(self, model):
        self.models = self.models + [model]
        if model.status == DataArray.Status.NEEDS_CALCULATING_GRID:
            if self._calculation is not None:
                self._cancel_computation()
            self.computation()

        def on_status_changed(change):
            if change.owner.status == DataArray.Status.NEEDS_CALCULATING_GRID:
                if self._calculation is not None:
                    self._cancel_computation()
                self.computation()
        model.observe(on_status_changed, 'status')
        # TODO: if we listen to the same axis twice it will trigger twice
        for axis in model.axes:
            axis.observe(lambda change: self.reslice(), 'slice')
        # self._callbacks_regrid.append(model.signal_regrid.connect(self.on_regrid))
        # self._callbacks_slice.append(model.signal_slice.connect(self.reslice))
        assert model.df == self.df

    # @vaex.jupyter.debounced(delay_seconds=0.05, reentrant=False)
    # def reslice_debounced(self):
    #     self.reslice()

    def reslice(self, source_model=None):
        if self._testing_exeception_reslice:
            raise RuntimeError("test:reslice")
        coords = []
        selection_was_list, [selections] = vaex.utils.listify(self.models[0].selection)
        selections = [k for k in selections if k is None or self.df.has_selection(k)]
        for model in self.models:
            subgrid = self.grid
            if not selection_was_list:
                subgrid = subgrid[0]
            subgrid_sliced = self.grid
            if not selection_was_list:
                subgrid_sliced = subgrid_sliced[0]
            axis_index = 1 if selection_was_list else 0
            has_slice = False
            dims = ["selection"] if selection_was_list else []
            coords = [selections.copy()] if selection_was_list else []
            mins = []
            maxs = []
            for other_model in self.models:
                if other_model == model:  # simply skip these axes
                    # for expression, shape, limit, slice_index in other_model.bin_parameters():
                    for axis in other_model.axes:
                        axis_index += 1
                        dims.append(str(axis.expression))
                        coords.append(axis.bin_centers)
                        mins.append(axis.min)
                        maxs.append(axis.max)
                else:
                    # for expression, shape, limit, slice_index in other_model.bin_parameters():
                    for axis in other_model.axes:
                        if axis.slice is not None:
                            subgrid_sliced = subgrid_sliced.__getitem__(tuple([slice(None)] * axis_index + [axis.slice])).copy()
                            subgrid = np.sum(subgrid, axis=axis_index)
                            has_slice = True
                        else:
                            subgrid_sliced = np.sum(subgrid_sliced, axis=axis_index)
                            subgrid = np.sum(subgrid, axis=axis_index)
            grid = xarray.DataArray(subgrid, dims=dims, coords=coords)
            # +1 to skip the selection axis
            dim_offset = 1 if selection_was_list else 0
            for i, (vmin, vmax) in enumerate(zip(mins, maxs)):
                grid.coords[dims[i+dim_offset]].attrs['min'] = vmin
                grid.coords[dims[i+dim_offset]].attrs['max'] = vmax
            model.grid = grid
            if has_slice:
                model.grid_sliced = xarray.DataArray(subgrid_sliced)
            else:
                model.grid_sliced = None

    def _regrid_error(self, e):
        try:
            self._error(e)
            for model in self.models:
                model._error(e)
            for model in self.models:
                model.exception = e
                model.status = vaex.jupyter.model.DataArray.Status.EXCEPTION
        except Exception as e2:
            print(e2)

    def on_regrid(self, ignore=None):
        self.regrid()

    @vaex.jupyter.debounced(delay_seconds=0.5, reentrant=False, on_error=_regrid_error)
    async def computation(self):
        try:
            logger.debug('Starting grid computation')
            # vaex.utils.print_stack_trace()
            if self._testing_exeception_regrid:
                raise RuntimeError("test:regrid")
            if not self.models:
                return
            binby = []
            shapes = []
            limits = []
            selection = self.models[0].selection
            selection_was_list, [selections] = vaex.utils.listify(self.models[0].selection)
            selections = [k for k in selections if k is None or self.df.has_selection(k)]

            for model in self.models:
                if model.selection != selection:
                    raise ValueError('Selections for all models should be the same')
                for axis in model.axes:
                    binby.append(axis.expression)
                    limits.append([axis.min, axis.max])
                    shapes.append(axis.shape or axis.shape_default)
            selections = [k for k in selections if k is None or self.df.has_selection(k)]

            self._continue_calculation = True
            logger.debug('Setting up grid computation...')
            self._calculation = self.df.count(binby=binby, shape=shapes, limits=limits, selection=selections, progress=self.progress, delay=True)

            logger.debug('Setting up grid computation done tasks=%r', self.df.executor.tasks)

            logger.debug('Schedule debounced execute')
            self.df.widget.execute_debounced()
            # keep a nearly reference to this, since awaits (which trigger the execution, AND reset of this future) may change it this
            execute_prehook_future = self.df.widget.execute_debounced.pre_hook_future

            async with contextlib.AsyncExitStack() as stack:
                for model in self.models:
                    await stack.enter_async_context(model._state_change_to(DataArray.Status.STAGED_CALCULATING_GRID))
            async with contextlib.AsyncExitStack() as stack:
                for model in self.models:
                    await stack.enter_async_context(model._state_change_to(DataArray.Status.CALCULATING_GRID))
                await execute_prehook_future
            async with contextlib.AsyncExitStack() as stack:
                for model in self.models:
                    await stack.enter_async_context(model._state_change_to(DataArray.Status.CALCULATED_GRID))
                # first assign to local
                grid = await self._calculation
                # indicate we are done with the calculation
                self._calculation = None
                # raise asyncio.CancelledError("User abort")
            async with contextlib.AsyncExitStack() as stack:
                for model in self.models:
                    await stack.enter_async_context(model._state_change_to(DataArray.Status.READY))
                self.grid = grid
                self.reslice()
        except vaex.execution.UserAbort:
            pass  # a user changed the limits or expressions
        except asyncio.CancelledError:
            pass  # cancelled...

    def _cancel_computation(self):
        logger.debug('Cancelling grid computation')
        self._continue_calculation = False

    def progress(self, f):
        return self._continue_calculation and all([model.on_progress_grid(f) for model in self.models])
