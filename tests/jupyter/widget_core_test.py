
#  this should be tested in plot_widget_test
# def test_hist_controls(ds):
#     state1 = vaex.jupyter.state.VizHistogramState(ds, x_expression='g1')
#     state2 = vaex.jupyter.state.VizHistogramState(ds, x_expression='g2')
#     grid = vaex.jupyter.grid.Grid(ds, [state1, state2])

#     viz = vaex.jupyter.view.Histogram(state=state1)
#     assert viz.normalize == False
#     viz.control.normalize.v_model = True
#     assert viz.normalize

#     assert state1.x_expression == 'g1'
#     viz.control.x.v_model = 'g2'
#     assert state1.x_expression == 'g2'

# TODO: later enable this again?
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
#     viz = vaex.jupyter.view.VizHistogramBqplot(state=state1, groups='slice')
#     vizgeo = vaex.jupyter.view.VizMapGeoJSONLeaflet(geo_json, ['g2'], state=state2)
#     assert state1.grid.tolist() == [[1, 2, 3]]
#     assert state2.grid.tolist() == [[2, 2, 1, 1]]
    
#     assert viz.bar.y.tolist() == [[1, 2, 3]]
#     assert state1.grid_sliced is None
#     state2.x_slice = 0
#     assert state1.grid.tolist() == [[1, 2, 3]]
#     assert state1.grid_sliced.tolist() == [[1, 1, 0]]
#     assert viz.bar.y.tolist() == [[1, 2, 3], [1, 1, 0]]


# @pytest.mark.skip(reason='unsure why it does not work')
# def test_piechart(ds, flush_guard):
#     state1 = vaex.jupyter.state.VizHistogramState(ds, x_expression='g1')
#     state2 = vaex.jupyter.state.VizHistogramState(ds, x_expression='g2')
#     grid = vaex.jupyter.grid.Grid(ds, [state1, state2])  # noqa
#     viz_pie = vaex.jupyter.view.VizPieChartBqplot(state=state1)
#     viz_bar = vaex.jupyter.view.VizHistogramBqplot(state=state2)  # noqa
#     assert state1.grid.tolist() == [1, 2, 3]
#     assert state2.grid.tolist() == [2, 2, 1, 1]

#     state2.x_slice = None
#     assert viz_pie.pie1.sizes.tolist() == [1, 2, 3]
#     assert state2.grid_sliced is None
#     state2.x_slice = 0
#     assert state1.grid.tolist() == [1, 2, 3]
#     assert state1.grid_sliced.tolist() == [1, 1, 0]
#     assert viz_pie.pie2.sizes.tolist() == [1, 1, 0]



# def test_heatmap_sliced(ds):
#     state1 = vaex.jupyter.state.VizHeatmapState(ds, x_expression='g1')
#     state2 = vaex.jupyter.state.VizHeatmapState(ds, x_expression='g2')
#     grid = vaex.jupyter.grid.Grid(ds, [state1, state2])
#     assert state1.grid.tolist() == [1, 2, 3]
#     assert state2.grid.tolist() == [2, 2, 1, 1]
    
#     viz = vaex.jupyter.view.VizHeatmapBqplot(state=state1)
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

#     viz = vaex.jupyter.view.VizHistogramBqplot(state=state1)
#     assert viz.normalize == False
#     viz.control.normalize.value = True
#     assert viz.normalize

#     assert state1.x_expression == 'g1'
#     viz.control.x.value = 'g2'
#     assert state1.x_expression == 'g2'


# @pytest.mark.skip(reason='requires icons PR in ipywidgets')
# def test_create(ds, flush_guard):
#     creator = vaex.jupyter.create.Creator(ds)
#     creator.widget_button_new_histogram.click()
#     assert len(creator.widget_container.children) == 1
#     assert len(creator.view) == 1
#     assert creator.view[0].state.ds is ds
#     # assert creator.widget_container.selected_index == 0
#     assert len(creator.widget_buttons_remove) == 1

#     creator.widget_button_new_heatmap.click()
#     assert len(creator.widget_container.children) == 2
#     assert len(creator.view) == 2
#     assert creator.view[1].state.ds is ds
#     # assert creator.widget_container.selected_index == 1
#     assert len(creator.widget_buttons_remove) == 2

#     creator.widget_button_new_histogram.click()
#     # assert creator.widget_container.selected_index == 2
#     assert len(creator.widget_buttons_remove) == 3
#     assert len(creator.widget_container.children) == 3

#     creator.widget_container.selected_index = 1
#     creator.widget_buttons_remove[1].click()
#     # assert creator.widget_container.selected_index == 1
#     assert len(creator.widget_container.children) == 2
#     assert len(creator.widget_buttons_remove) == 2
#     assert len(creator.grid.states) == 2

#     creator.widget_buttons_remove[1].click()
#     # assert creator.widget_container.selected_index == 0
#     assert len(creator.widget_container.children) == 1
#     assert len(creator.widget_buttons_remove) == 1
#     assert len(creator.grid.states) == 1
