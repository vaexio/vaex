import pytest
import vaex
import numpy as np
import vaex.jupyter.state
import vaex.jupyter.viz
import vaex.jupyter.create
import vaex.jupyter.grid

@pytest.fixture()
def ds():
    x  = np.array([0, 1, 2, 3, 4, 5])
    y  = x ** 2
    g1 = np.array([0, 1, 1, 2, 2, 2])
    g2 = np.array([0, 0, 1, 1, 2, 3])
    ds = vaex.from_arrays(x=x, y=y, g1=g1, g2=g2)
    ds.categorize(ds.g1)
    ds.categorize(ds.g2)
    return ds

def test_hist_state(ds):
    state = vaex.jupyter.state.VizHistogramState(ds, x_expression='g1')
    grid = vaex.jupyter.grid.Grid(ds, [state])
    assert state.x_min == -0.5
    assert state.x_max == 2.5
    assert state.x_shape == 3
    assert state.grid.tolist() == [1, 2, 3]
    
    viz = vaex.jupyter.viz.VizHistogramBqplot(state=state)
    assert viz.bar.y.tolist() == [1, 2, 3]
    assert viz.x_axis.label == 'g1'
    assert viz.y_axis.label == 'counts'

    state.x_expression = 'g2'
    assert state.x_min == -0.5
    assert state.x_max == 3.5
    assert state.x_shape == 4
    assert state.grid.tolist() == [2, 2, 1, 1]
    assert viz.x_axis.label == 'g2'

    state = vaex.jupyter.state.VizHistogramState(ds, x_expression='x', x_min=-0.5, x_max=5.5, shape=6)
    assert state.x_centers.tolist() == [0, 1, 2, 3, 4, 5]
    assert state.x_min == -0.5
    assert state.x_max == 5.5
    grid = vaex.jupyter.grid.Grid(ds, [state])
    assert state.x_shape == None
    assert state.shape == 6

def test_hist_sliced(ds):
    state1 = vaex.jupyter.state.VizHistogramState(ds, x_expression='g1')
    state2 = vaex.jupyter.state.VizHistogramState(ds, x_expression='g2')
    grid = vaex.jupyter.grid.Grid(ds, [state1, state2])
    assert state1.grid.tolist() == [1, 2, 3]
    assert state2.grid.tolist() == [2, 2, 1, 1]
    
    viz = vaex.jupyter.viz.VizHistogramBqplot(state=state1)
    assert viz.bar.y.tolist() == [1, 2, 3]
    assert state1.grid_sliced is None
    state2.x_slice = 0
    assert state1.grid.tolist() == [1, 2, 3]
    assert state1.grid_sliced.tolist() == [1, 1, 0]
    assert viz.bar.y.tolist() == [[1, 2, 3], [1, 1, 0]]

def test_hist_controls(ds):
    state1 = vaex.jupyter.state.VizHistogramState(ds, x_expression='g1')
    state2 = vaex.jupyter.state.VizHistogramState(ds, x_expression='g2')
    grid = vaex.jupyter.grid.Grid(ds, [state1, state2])

    viz = vaex.jupyter.viz.VizHistogramBqplot(state=state1)
    assert viz.normalize == False
    viz.control.normalize.v_model = True
    assert viz.normalize

    assert state1.x_expression == 'g1'
    viz.control.x.v_model = 'g2'
    assert state1.x_expression == 'g2'

# def test_geojson(ds):
#     geo_json = {
#         'features': [
#             {'geometry': {
#                 'type': 'MultiPolygon',
#                 'coordinates': []
#             },
#             'properties': {
#                 'objectid': 1
#             }
#             }
#         ]
#     }
#     state1 = vaex.jupyter.state.VizHistogramState(ds, x_expression='g1')
#     state2 = vaex.jupyter.state.VizHistogramState(ds, x_expression='g2')
#     grid = vaex.jupyter.grid.Grid(ds, [state1, state2])
#     viz = vaex.jupyter.viz.VizHistogramBqplot(state=state1)
#     vizgeo = vaex.jupyter.viz.VizMapGeoJSONLeaflet(geo_json, ['g2'], state=state2)
#     assert state1.grid.tolist() == [1, 2, 3]
#     assert state2.grid.tolist() == [2, 2, 1, 1]
    
#     assert viz.bar.y.tolist() == [1, 2, 3]
#     assert state1.grid_sliced is None
#     state2.x_slice = 0
#     assert state1.grid.tolist() == [1, 2, 3]
#     assert state1.grid_sliced.tolist() == [1, 1, 0]
#     assert viz.bar.y.tolist() == [[1, 2, 3], [1, 1, 0]]


@pytest.mark.skip(reason='unsure why it does not work')
def test_piechart(ds):
    state1 = vaex.jupyter.state.VizHistogramState(ds, x_expression='g1')
    state2 = vaex.jupyter.state.VizHistogramState(ds, x_expression='g2')
    grid = vaex.jupyter.grid.Grid(ds, [state1, state2])
    viz_pie = vaex.jupyter.viz.VizPieChartBqplot(state=state1)
    viz_bar = vaex.jupyter.viz.VizHistogramBqplot(state=state2)
    assert state1.grid.tolist() == [1, 2, 3]
    assert state2.grid.tolist() == [2, 2, 1, 1]

    state2.x_slice = None
    assert viz_pie.pie1.sizes.tolist() == [1, 2, 3]
    assert state2.grid_sliced is None
    state2.x_slice = 0
    assert state1.grid.tolist() == [1, 2, 3]
    assert state1.grid_sliced.tolist() == [1, 1, 0]
    assert viz_pie.pie2.sizes.tolist() == [1, 1, 0]


def test_heatmap_state(ds):
    state = vaex.jupyter.state.VizHeatmapState(ds, x_expression='x', y_expression='g1', shape=2, x_min=0, x_max=5)
    grid = vaex.jupyter.grid.Grid(ds, [state])
    assert state.x_min == 0
    assert state.x_max == 5
    assert state.y_min == -0.5
    assert state.y_max == 2.5
    assert state.shape == 2
    assert state.x_shape == None
    assert state.y_shape == 3
    assert state.grid.tolist() == [[1, 2, 0], [0, 0, 2]]
    
    viz = vaex.jupyter.viz.VizHeatmapBqplot(state=state)
    assert viz.heatmap.color.T.tolist() == [[1, 2, 0], [0, 0, 2]]
    assert viz.x_axis.label == 'x'
    assert viz.y_axis.label == 'g1'

    state.x_expression = 'g2'
    assert state.x_min == -0.5
    assert state.x_max == 3.5
    assert state.x_shape == 4
    assert state.shape == 2
    grid = [[1, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 1]]
    assert state.grid.tolist() == grid
    assert viz.heatmap.color.T.tolist() == grid

# def test_heatmap_sliced(ds):
#     state1 = vaex.jupyter.state.VizHeatmapState(ds, x_expression='g1')
#     state2 = vaex.jupyter.state.VizHeatmapState(ds, x_expression='g2')
#     grid = vaex.jupyter.grid.Grid(ds, [state1, state2])
#     assert state1.grid.tolist() == [1, 2, 3]
#     assert state2.grid.tolist() == [2, 2, 1, 1]
    
#     viz = vaex.jupyter.viz.VizHeatmapBqplot(state=state1)
#     assert viz.bar.y.tolist() == [1, 2, 3]
#     assert state1.grid_sliced is None
#     state2.x_slice = 0
#     assert state1.grid.tolist() == [1, 2, 3]
#     assert state1.grid_sliced.tolist() == [1, 1, 0]
#     assert viz.bar.y.tolist() == [[1, 2, 3], [1, 1, 0]]

# def test_heatmap_controls(ds):
#     state1 = vaex.jupyter.state.VizHeatmapState(ds, x_expression='x')
#     state2 = vaex.jupyter.state.VizHeatmapState(ds, x_expression='y')
#     grid = vaex.jupyter.grid.Grid(ds, [state1, state2])

#     viz = vaex.jupyter.viz.VizHistogramBqplot(state=state1)
#     assert viz.normalize == False
#     viz.control.normalize.value = True
#     assert viz.normalize

#     assert state1.x_expression == 'g1'
#     viz.control.x.value = 'g2'
#     assert state1.x_expression == 'g2'

@pytest.mark.skip(reason='requires icons PR in ipywidgets')
def test_create(ds):
    creator = vaex.jupyter.create.Creator(ds)
    creator.widget_button_new_histogram.click()
    assert len(creator.widget_container.children) == 1
    assert len(creator.viz) == 1
    assert creator.viz[0].state.ds is ds
    # assert creator.widget_container.selected_index == 0
    assert len(creator.widget_buttons_remove) == 1

    creator.widget_button_new_heatmap.click()
    assert len(creator.widget_container.children) == 2
    assert len(creator.viz) == 2
    assert creator.viz[1].state.ds is ds
    # assert creator.widget_container.selected_index == 1
    assert len(creator.widget_buttons_remove) == 2

    creator.widget_button_new_histogram.click()
    # assert creator.widget_container.selected_index == 2
    assert len(creator.widget_buttons_remove) == 3
    assert len(creator.widget_container.children) == 3

    creator.widget_container.selected_index = 1
    creator.widget_buttons_remove[1].click()
    # assert creator.widget_container.selected_index == 1
    assert len(creator.widget_container.children) == 2
    assert len(creator.widget_buttons_remove) == 2
    assert len(creator.grid.states) == 2

    creator.widget_buttons_remove[1].click()
    # assert creator.widget_container.selected_index == 0
    assert len(creator.widget_container.children) == 1
    assert len(creator.widget_buttons_remove) == 1
    assert len(creator.grid.states) == 1
