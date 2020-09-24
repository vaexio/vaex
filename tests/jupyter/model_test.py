import pytest
import vaex
import numpy as np
import vaex.jupyter.model
import vaex.jupyter.view
import vaex.jupyter.grid
from vaex.jupyter.utils import _debounced_flush as flush, gather


@pytest.fixture()
def df():
    x = np.array([0, 1, 2, 3, 4, 5])
    y = x ** 2
    g1 = np.array([0, 1, 1, 2, 2, 2])
    g2 = np.array([0, 0, 1, 1, 2, 3])
    ds = vaex.from_arrays(x=x, y=y, g1=g1, g2=g2)
    ds.categorize(ds.g1, inplace=True)
    ds.categorize(ds.g2, inplace=True)
    return ds


def test_axis_basics(df, flush_guard):
    x = vaex.jupyter.model.Axis(df=df, expression=df.x)
    flush()
    assert x.min == df.x.min()
    assert x.max == df.x.max()
    assert x.bin_centers.shape[0] == x.shape_default

    center_last = x.bin_centers[-1]
    x.max += 1
    flush()
    assert x.bin_centers[-1] > center_last

    x.shape = 5
    flush()
    assert x.bin_centers.shape[0] == 5

    x.shape = 6
    flush()
    assert x.bin_centers.shape[0] == 6

    x.shape_default = 7
    flush()
    assert x.bin_centers.shape[0] == 6

    x.shape = None
    flush(all=True)
    assert x.bin_centers.shape[0] == 7


def test_model_selection(df, flush_guard):
    x = vaex.jupyter.model.Axis(df=df, expression='g1')
    df.select(df.x > 0)
    model = vaex.jupyter.model.Histogram(df=df, x=x, selection=[None, 'default'])
    grid = vaex.jupyter.model.GridCalculator(df, [model])
    flush()
    assert model.x.min == -0.5
    assert model.x.max == 2.5
    assert model.x.shape == 3
    assert model.grid.data.tolist() == [[1, 2, 3], [0, 2, 3]]
    assert model.grid.dims == ('selection', 'g1')
    assert model.grid.coords['selection'].data.tolist() == [None, 'default']
    assert model.grid.coords['g1'].data.tolist() == [0, 1, 2]




def test_data_array_attrs(df, flush_guard):
    x = vaex.jupyter.model.Axis(df=df, expression='x', min=0, max=1)
    y = vaex.jupyter.model.Axis(df=df, expression='y')
    model = vaex.jupyter.model.Heatmap(df=df, x=x, y=y, shape=5)
    assert x.min == 0
    assert x.max == 1
    grid = vaex.jupyter.model.GridCalculator(df, [model])  # noqa
    assert x.min == 0
    assert x.max == 1
    flush(all=True)
    assert x.min == 0
    assert x.max == 1
    assert model.grid.coords['x'].attrs['min'] == 0
    assert model.grid.coords['x'].attrs['max'] == 1
    assert model.grid.coords['y'].attrs['min'] == y.min
    assert model.grid.coords['y'].attrs['max'] == y.max


@pytest.mark.asyncio
async def test_axis_status(df, flush_guard):
    x = vaex.jupyter.model.Axis(df=df, expression=df.x, _debug=True)
    assert x.status == x.Status.NO_LIMITS
    x.computation()
    await x._allow_state_change_to(x.Status.STAGED_CALCULATING_LIMITS)
    assert x.status == x.Status.STAGED_CALCULATING_LIMITS

    await x._allow_state_change_to(x.Status.CALCULATING_LIMITS)
    assert x.status == x.Status.CALCULATING_LIMITS

    await x._allow_state_change_to(x.Status.CALCULATED_LIMITS)
    assert x.status == x.Status.CALCULATED_LIMITS

    await x._allow_state_change_to(x.Status.READY)
    assert x.status == x.Status.READY

    x.max = 100
    assert x.status == x.Status.READY


def test_axis_minmax(df, flush_guard):
    x = vaex.jupyter.model.Axis(df=df, expression=df.x)
    model = vaex.jupyter.model.DataArray(df=df, axes=[x])
    grid = vaex.jupyter.model.GridCalculator(df, [model])  # noqa
    flush()
    assert x.min == df.x.min()
    assert x.max == df.x.max()

    x = vaex.jupyter.model.Axis(df=df, expression=df.x, min=1, max=2)
    model = vaex.jupyter.model.DataArray(df=df, axes=[x])
    grid = vaex.jupyter.model.GridCalculator(df, [model])  # noqa
    flush()
    assert x.min == 1
    assert x.max == 2

    # with 2 axes
    x = vaex.jupyter.model.Axis(df=df, expression=df.x, min=1, max=2)
    y = vaex.jupyter.model.Axis(df=df, expression=df.y, min=2, max=3)
    model = vaex.jupyter.model.DataArray(df=df, axes=[x, y])
    grid = vaex.jupyter.model.GridCalculator(df, [model])  # noqa
    flush()
    assert x.min == 1
    assert x.max == 2
    assert y.min == 2
    assert y.max == 3

    # with 2 axes, one with min/max
    x = vaex.jupyter.model.Axis(df=df, expression=df.x, min=1, max=2)
    y = vaex.jupyter.model.Axis(df=df, expression=df.y)
    model = vaex.jupyter.model.DataArray(df=df, axes=[x, y])
    grid = vaex.jupyter.model.GridCalculator(df, [model])  # noqa
    flush(all=True)
    assert x.min == 1
    assert x.max == 2
    assert y.min == df.y.min()
    assert y.max == df.y.max()


@pytest.mark.asyncio
async def test_model_status(df_executor, flush_guard, server_latency):
    df = df_executor
    x = vaex.jupyter.model.Axis(df=df, expression='x', min=0, max=1)
    assert x.status == x.Status.READY

    x = vaex.jupyter.model.Axis(df=df, expression='x', _status_change_delay=0.001, _debug=True)
    y = vaex.jupyter.model.Axis(df=df, expression='y', _status_change_delay=0.001, _debug=True)
    assert x.status == x.Status.NO_LIMITS
    assert y.status == y.Status.NO_LIMITS
    model = vaex.jupyter.model.Heatmap(df=df, x=x, y=y, shape=5, _debug=True, selection=[None, 'default'])
    grid = vaex.jupyter.model.GridCalculator(df, [model])  # noqa
    assert x.status == x.Status.NO_LIMITS
    assert y.status == y.Status.NO_LIMITS
    assert model.status == model.Status.MISSING_LIMITS
    assert model.status_text == 'Missing limits for x, y'

    # this should have scheduled the DataArray.computation
    # assert len(vaex.jupyter.utils._debounced_execute_queue) == 1

    await x._allow_state_change_to(x.Status.STAGED_CALCULATING_LIMITS)
    await y._allow_state_change_to(y.Status.STAGED_CALCULATING_LIMITS)
    assert x.status == y.Status.STAGED_CALCULATING_LIMITS
    assert y.status == y.Status.STAGED_CALCULATING_LIMITS
    assert model.status_text == 'Staged limit computation for x, y'
    assert model.status == model.Status.STAGED_CALCULATING_LIMITS

    # twice Axis.computation + 2x execute_debounced
    # but, due to latency, sometimes 1 of the executions is already scheduled
    assert len(vaex.jupyter.utils._debounced_execute_queue) in [3, 4]

    await x._allow_state_change_to(x.Status.CALCULATING_LIMITS)
    await y._allow_state_change_to(x.Status.CALCULATING_LIMITS)
    assert x.status == x.Status.CALCULATING_LIMITS
    assert y.status == x.Status.CALCULATING_LIMITS
    assert model.status == model.Status.CALCULATING_LIMITS
    assert model.status_text == 'Computing limits for x, y'

    # same
    assert len(vaex.jupyter.utils._debounced_execute_queue) in [2, 3, 4]

    await x._allow_state_change_to(x.Status.CALCULATED_LIMITS)
    await y._allow_state_change_to(x.Status.CALCULATED_LIMITS)
    await x._allow_state_change_to(x.Status.READY)
    await y._allow_state_change_to(x.Status.READY)
    assert x.status == x.Status.READY
    assert y.status == y.Status.READY
    await x.computation._await_last_call()
    await y.computation._await_last_call()
    # - twice Axis.computation, - all executions, + 1x GridCalculator.computation (for min and max)
    assert len(vaex.jupyter.utils._debounced_execute_queue) == 1

    assert model.status == model.Status.NEEDS_CALCULATING_GRID
    assert model.status_text == 'Grid needs to be calculated'
    await model._allow_state_change_to(model.Status.STAGED_CALCULATING_GRID)
    assert model.status_text == 'Staged grid computation'

    if df.is_local():
        assert len(vaex.jupyter.utils._debounced_execute_queue) == 2 # + DataFrameAccessorWidget.execute_debounced

    assert model.grid is None
    await model._allow_state_change_to(model.Status.CALCULATING_GRID)
    assert model.status == model.Status.CALCULATING_GRID
    assert model.status_text == 'Calculating grid'
    await model._allow_state_change_to(model.Status.CALCULATED_GRID)
    assert model.status == model.Status.CALCULATED_GRID
    assert model.status_text == 'Calculated grid'
    assert model.grid is None

    if df.is_local():
        assert len(vaex.jupyter.utils._debounced_execute_queue) == 1 # - DataFrameAccessorWidget.execute_debounced
    await model._allow_state_change_to(model.Status.READY)
    assert model.status == model.Status.READY
    assert model.status_text == 'Ready'
    assert model.grid is not None
    model._debug = False  # allow state changes freely

    x.expression = df.x * 2
    # TODO: do we want a state 'DIRTY_LIMITS' ?
    assert x.min is None
    assert x.max is None

    await x._allow_state_change_to(x.Status.STAGED_CALCULATING_LIMITS)
    await x._allow_state_change_to(x.Status.CALCULATING_LIMITS)
    assert x.status == x.Status.CALCULATING_LIMITS
    assert y.status == y.Status.READY
    assert model.status == model.Status.CALCULATING_LIMITS
    assert model.status_text == 'Computing limits for (x * 2)'

    await x._allow_state_change_to(x.Status.CALCULATED_LIMITS)
    await x._allow_state_change_to(x.Status.READY)
    assert x.status == x.Status.READY
    assert y.status == y.Status.READY
    # this will trigger 1x DataArrayModel._update_grid (for x max, since 2*0=0)
    await grid.computation._await_last_call()
    assert model.status == model.Status.READY
    assert model.status_text == 'Ready'
    assert model.grid is not None
    assert len(vaex.jupyter.utils._debounced_execute_queue) == 0

    # this should trigger a recomputation
    df.select(df.x > 0)
    assert model.status == model.Status.NEEDS_CALCULATING_GRID
    await grid.computation._await_last_call()
    assert model.status == model.Status.READY

    # now we change the x axis but will interrupt by setting the min/max during calculation

    x.expression = df.x * 3
    assert x.min is None
    assert x.max is None

    await x._allow_state_change_to(x.Status.STAGED_CALCULATING_LIMITS)
    calculation_task = x._calculation
    # df.executor.thread_pool._debug_sleep = 0.05  # make it slow, so we can cancel the task
    assert hasattr(calculation_task, 'then')  # make sure we have the promise
    await x._allow_state_change_to(x.Status.CALCULATING_LIMITS)
    assert x.status == x.Status.CALCULATING_LIMITS
    assert y.status == y.Status.READY
    assert model.status == model.Status.CALCULATING_LIMITS
    assert model.status_text == 'Computing limits for (x * 3)'

    model._debug = True
    x.min = -1
    x.max = 10
    assert x.status == x.Status.READY
    assert y.status == y.Status.READY
    # this will trigger 1x DataArrayModel._update_grid (for x max, since 2*0=0)
    assert model.status == model.Status.NEEDS_CALCULATING_GRID

    await x._allow_state_change_cancel()
    # await x.computation._await_last_call()  # this should not be needed
    assert calculation_task.isRejected
    assert model.status == model.Status.NEEDS_CALCULATING_GRID
    await model._allow_state_change_to(model.Status.STAGED_CALCULATING_GRID)
    await model._allow_state_change_to(model.Status.CALCULATING_GRID)
    await model._allow_state_change_to(model.Status.CALCULATED_GRID)
    await model._allow_state_change_to(model.Status.READY)
    # await x.computation._await_last_call()
    # await x._allow_state_change_to(x.Status.READY)
    await grid.computation._await_last_call()
    assert model.status == model.Status.READY
    assert len(vaex.jupyter.utils._debounced_execute_queue) == 0
    model._debug = False

    # now we change the x axis but will interrupt by setting the min/max BEFORE calculation

    x.expression = df.x * 4
    assert x.min is None
    assert x.max is None

    await x._allow_state_change_to(x.Status.STAGED_CALCULATING_LIMITS)
    assert x.status == x.Status.STAGED_CALCULATING_LIMITS
    assert y.status == y.Status.READY
    assert model.status == model.Status.STAGED_CALCULATING_LIMITS

    x.min = -1
    x.max = 10

    await x._allow_state_change_cancel()
    # await x.computation.cancel_and_wait()
    assert x.status == x.Status.READY
    assert y.status == y.Status.READY
    assert model.status == model.Status.NEEDS_CALCULATING_GRID
    await grid.computation._await_last_call()
    assert model.status == model.Status.READY

    # now we change the expression, during a during calculation

    x.expression = df.x * 5
    assert x.min is None
    assert x.max is None

    await x._allow_state_change_to(x.Status.STAGED_CALCULATING_LIMITS)
    # return
    await x._allow_state_change_to(x.Status.CALCULATING_LIMITS)
    assert x.status == x.Status.CALCULATING_LIMITS
    assert y.status == y.Status.READY
    assert model.status == model.Status.CALCULATING_LIMITS
    assert model.status_text == 'Computing limits for (x * 5)'

    x.expression = df.x * 6
    assert x.status == x.Status.NO_LIMITS
    assert model.status == model.Status.MISSING_LIMITS

    # x.computation() is called again, so we need to await the previous call
    await x._allow_state_change_cancel(previous=True)
    assert x.status == x.Status.NO_LIMITS
    await x._allow_state_change_to(x.Status.STAGED_CALCULATING_LIMITS)
    await x._allow_state_change_to(x.Status.CALCULATING_LIMITS)
    assert x.status == x.Status.CALCULATING_LIMITS
    assert y.status == y.Status.READY
    assert model.status == model.Status.CALCULATING_LIMITS
    assert model.status_text == 'Computing limits for (x * 6)'

    await x._allow_state_change_to(x.Status.CALCULATED_LIMITS)
    await x._allow_state_change_to(x.Status.READY)
    assert x.status == x.Status.READY
    assert y.status == y.Status.READY

    assert model.status == model.Status.NEEDS_CALCULATING_GRID
    await grid.computation._await_last_call()
    assert model.status == model.Status.READY

    # changes in min/max while in the ready state, should change the model state
    assert model.status == model.Status.READY
    x.min = -2
    x.max = 12
    assert model.status == model.Status.NEEDS_CALCULATING_GRID
    assert model.status_text == 'Grid needs to be calculated'
    await grid.computation._await_last_call()
    assert model.status == model.Status.READY

    # change in min/max during grid computation should cancel
    model._debug = True
    x.min = -3
    await model._allow_state_change_to(model.Status.STAGED_CALCULATING_GRID)
    calculation_task = grid._calculation
    assert calculation_task.isPending
    x.min = -4
    assert model.status == model.Status.NEEDS_CALCULATING_GRID
    await model._allow_state_change_cancel()
    assert model.status == model.Status.NEEDS_CALCULATING_GRID
    # we should already have scheduled a new call, so wait for the previous
    await grid.computation._await_last_call(previous=True)
    assert model.status == model.Status.NEEDS_CALCULATING_GRID
    # we still did not execute the tast
    assert calculation_task.isPending
    await df.widget.execute_debounced._await_last_call()
    assert calculation_task.isRejected
    await model._allow_state_change_to(model.Status.STAGED_CALCULATING_GRID)
    await model._allow_state_change_to(model.Status.CALCULATING_GRID)
    await model._allow_state_change_to(model.Status.CALCULATED_GRID)
    await model._allow_state_change_to(model.Status.READY)

    # same, but now at a later stage
    x.min = -5
    assert model.status == model.Status.NEEDS_CALCULATING_GRID
    await model._allow_state_change_to(model.Status.STAGED_CALCULATING_GRID)
    await model._allow_state_change_to(model.Status.CALCULATING_GRID)
    x.min = -6
    await model._allow_state_change_cancel()
    assert model.status == model.Status.NEEDS_CALCULATING_GRID
    # we should already have scheduled a new call, so wait for the previous
    await grid.computation._await_last_call(previous=True)
    assert model.status == model.Status.NEEDS_CALCULATING_GRID
    # in this case we DID execute the task
    assert calculation_task.isRejected
    await model._allow_state_change_to(model.Status.STAGED_CALCULATING_GRID)
    await model._allow_state_change_to(model.Status.CALCULATING_GRID)
    await model._allow_state_change_to(model.Status.CALCULATED_GRID)
    await model._allow_state_change_to(model.Status.READY)


def test_histogram_model_passes(df, flush_guard):
    passes = df.executor.passes
    x = vaex.jupyter.model.Axis(df=df, expression='x')
    model = vaex.jupyter.model.Histogram(df=df, x=x)
    assert df.executor.passes == passes
    flush()
    # this will do the minmax
    assert df.executor.passes == passes + 1

    # now will will manually do the grid
    grid = vaex.jupyter.model.GridCalculator(df, [model])
    grid.computation()
    flush()
    assert df.executor.passes == passes + 2

    # a minmax and a new grid
    model.x.expression = 'y'
    assert df.executor.passes == passes + 2
    flush(all=True)
    assert df.executor.passes == passes + 2 + 2


def test_two_model_passes(df, flush_guard):
    passes = df.executor.passes
    x1 = vaex.jupyter.model.Axis(df=df, expression='x')
    x2 = vaex.jupyter.model.Axis(df=df, expression='x')
    model1 = vaex.jupyter.model.Histogram(df=df, x=x1)
    model2 = vaex.jupyter.model.Histogram(df=df, x=x2)
    assert df.executor.passes == passes
    flush()
    # this will do the minmax for both in 1 pass
    assert df.executor.passes == passes + 1

    # now we will manually do the gridding, both in 1 pass
    grid1 = vaex.jupyter.model.GridCalculator(df, [model1])
    grid2 = vaex.jupyter.model.GridCalculator(df, [model2])
    grid1.computation()
    grid2.computation()
    assert model1.grid is None
    assert model2.grid is None
    flush(all=True)
    assert df.executor.passes == passes + 1 + 1


def test_heatmap_model_passes(df, flush_guard):
    passes = df.executor.passes
    x = vaex.jupyter.model.Axis(df=df, expression='x')
    y = vaex.jupyter.model.Axis(df=df, expression='y')
    model = vaex.jupyter.model.Heatmap(df=df, x=x, y=y, shape=5)
    assert df.executor.passes == passes
    flush()
    # this will do two minmaxes in 1 pass
    assert df.executor.passes == passes + 1

    # now will will manually do the grid
    grid = vaex.jupyter.model.GridCalculator(df, [model])
    grid.computation()
    flush()
    assert df.executor.passes == passes + 2

    # once a minmax and a new grid
    x.expression = 'y'
    assert df.executor.passes == passes + 2
    flush()
    assert df.executor.passes == passes + 2 + 2

    # twice a minmax in 1 pass, followed by a gridding
    x.expression = 'x*2'
    y.expression = 'y*2'
    assert df.executor.passes == passes + 2 + 2
    flush()
    assert df.executor.passes == passes + 2 + 2 + 2

    # once a minmax and a new grid
    x.expression = 'x*3'
    assert df.executor.passes == passes + 2 + 2 + 2
    flush(all=True)
    assert df.executor.passes == passes + 2 + 2 + 2 + 2


def test_histogram_model(df, flush_guard):
    x = vaex.jupyter.model.Axis(df=df, expression='g1')
    model = vaex.jupyter.model.Histogram(df=df, x=x)
    grid = vaex.jupyter.model.GridCalculator(df, [model])
    flush()
    assert model.x.min == -0.5
    assert model.x.max == 2.5
    assert model.x.shape == 3
    assert model.grid.data.tolist() == [1, 2, 3]
    assert model.grid.dims == ('g1', )
    assert model.grid.coords['g1'].data.tolist() == [0, 1, 2]

    viz = vaex.jupyter.view.Histogram(model=model, dimension_groups='slice')
    assert viz.plot.mark.y.tolist() == [[1, 2, 3]]
    assert viz.plot.x_axis.label == 'g1'
    assert viz.plot.y_axis.label == 'count'

    model.x.expression = 'g2'
    flush()
    assert model.x.min == -0.5
    assert model.x.max == 3.5
    assert model.x.shape == 4
    assert model.grid.data.tolist() == [2, 2, 1, 1]
    assert model.grid.dims == ('g2',)
    assert model.grid.coords['g2'].data.tolist() == [0, 1, 2, 3]
    assert viz.plot.x_axis.label == 'g2'

    x = vaex.jupyter.model.Axis(df=df, expression='x', min=-0.5, max=5.5)
    model = vaex.jupyter.model.Histogram(df=df, x=x, shape=6)
    flush()
    assert model.x.bin_centers.tolist() == [0, 1, 2, 3, 4, 5]
    assert model.x.min == -0.5
    assert model.x.max == 5.5
    grid = vaex.jupyter.model.GridCalculator(df, [model])  # noqa
    assert model.x.shape is None
    assert model.shape == 6
    flush(all=True)


def test_histogram_sliced(df, flush_guard):
    g1 = vaex.jupyter.model.Axis(df=df, expression='g1')
    g2 = vaex.jupyter.model.Axis(df=df, expression='g2')
    model1 = vaex.jupyter.model.Histogram(df=df, x=g1)
    model2 = vaex.jupyter.model.Histogram(df=df, x=g2)
    grid = vaex.jupyter.model.GridCalculator(df, [model1, model2])  # noqa
    flush(all=True)
    assert model1.grid.data is not None
    assert model1.x.bin_centers.tolist() == [0, 1, 2]

    assert model1.grid.data.tolist() == [1, 2, 3]
    assert model2.grid.data.tolist() == [2, 2, 1, 1]

    viz = vaex.jupyter.view.Histogram(model=model1, dimension_groups='slice')
    assert viz.plot.mark.y.tolist() == [[1, 2, 3]]
    assert model1.grid_sliced is None
    model2.x.slice = 0
    assert model1.grid.data.tolist() == [1, 2, 3]
    assert model1.grid_sliced.data.tolist() == [1, 1, 0]
    assert viz.plot.mark.y.tolist() == [[1, 2, 3], [1, 1, 0]]


def test_histogram_selections(df, flush_guard):
    g1 = vaex.jupyter.model.Axis(df=df, expression='g1')
    g2 = vaex.jupyter.model.Axis(df=df, expression='g2')
    df.select(df.g1 == 1)
    model1 = vaex.jupyter.model.Histogram(df=df, x=g1, selection=[None, True])
    model2 = vaex.jupyter.model.Histogram(df=df, x=g2, selection=[None, True])
    grid = vaex.jupyter.model.GridCalculator(df, [model1, model2])  # noqa
    flush(all=True)
    assert model1.grid.data.tolist() == [[1, 2, 3], [0, 2, 0]]
    assert model2.grid.data.tolist() == [[2, 2, 1, 1], [1, 1, 0, 0]]

    viz = vaex.jupyter.view.Histogram(model=model1, groups='selections')
    assert viz.plot.mark.y.tolist() == [[1, 2, 3], [0, 2, 0]]


def test_heatmap_model_basics(df, flush_guard):
    x = vaex.jupyter.model.Axis(df=df, expression='x', min=0, max=5)
    g1 = vaex.jupyter.model.Axis(df=df, expression='g1')
    model = vaex.jupyter.model.Heatmap(df=df, x=x, y=g1, shape=2)
    grid = vaex.jupyter.model.GridCalculator(df, [model])
    flush()
    assert model.x.min == 0
    assert model.x.max == 5
    assert model.y.min == -0.5
    assert model.y.max == 2.5
    assert model.shape == 2
    assert model.x.shape is None
    assert model.y.shape == 3
    assert model.grid.data.tolist() == [[1, 2, 0], [0, 0, 2]]

    viz = vaex.jupyter.view.Heatmap(model=model)
    flush()
    # TODO: if we use bqplot-image-gl we can test the data again
    # assert viz.heatmap.color.T.tolist() == [[1, 2, 0], [0, 0, 2]]
    assert viz.plot.x_label == 'x'
    assert viz.plot.y_label == 'g1'

    model.x.expression = 'g2'
    flush()
    assert model.x.min == -0.5
    assert model.x.max == 3.5
    assert model.x.shape == 4
    assert model.shape == 2
    grid = [[1, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 1]]
    assert model.grid.data.tolist() == grid
    flush(all=True)
    # TODO: if we use bqplot-image-gl we can test the data again
    # assert viz.heatmap.color.T.tolist() == grid


@pytest.mark.asyncio
async def test_grid_exception_handling(df, flush_guard):
    # since exception can occur doing debounced/async execution, we will not
    # see stack traces, so we need to carefully track those
    x = vaex.jupyter.model.Axis(df=df, expression='x')
    # y = vaex.jupyter.model.Axis(df=df, expression='y')
    df.select(df.g1 == 1)
    model1 = vaex.jupyter.model.Histogram(df=df, x=x)
    model2 = vaex.jupyter.model.Histogram(df=df, x=x)
    grid = vaex.jupyter.model.GridCalculator(df, [model1, model2])  # noqa
    assert model1.status == vaex.jupyter.model.DataArray.Status.MISSING_LIMITS
    assert model2.status == vaex.jupyter.model.DataArray.Status.MISSING_LIMITS
    await gather()
    assert model1.status == vaex.jupyter.model.DataArray.Status.READY
    assert model2.status == vaex.jupyter.model.DataArray.Status.READY
    grid._testing_exeception_regrid = True
    try:
        x.min = -1
        # flush(ignore_exceptions=True)
        # with pytest.raises(RuntimeError):
        await gather(ignore_exceptions=True)
        # await grid.computation._await_last_call()
        assert model1.status == vaex.jupyter.model.DataArray.Status.EXCEPTION
        assert model2.status == vaex.jupyter.model.DataArray.Status.EXCEPTION
        assert 'test:regrid' in str(model1.exception)
        assert 'test:regrid' in str(model2.exception)
        assert model1.exception is model2.exception
    finally:
        grid._testing_exeception_regrid = False

    old_exception = model1.exception
    model1.exception = None

    assert vaex.jupyter.utils._debounced_execute_queue == []
    grid._testing_exeception_reslice = True
    try:
        x.min = -2
        # with pytest.raises(RuntimeError):
        await gather(ignore_exceptions=True)
        assert model1.status == vaex.jupyter.model.DataArray.Status.EXCEPTION
        assert model2.status == vaex.jupyter.model.DataArray.Status.EXCEPTION
        assert model1.exception is not old_exception
        assert 'test:reslice' in str(model1.exception)
        assert 'test:reslice' in str(model2.exception)
        assert model1.exception is model2.exception
        # make asyncio happy with not having dangling exceptions in futures
        with pytest.raises(RuntimeError):
            await grid.computation._await_last_call()
    finally:
        grid._testing_exeception_reslice = True
    await gather(all=True)