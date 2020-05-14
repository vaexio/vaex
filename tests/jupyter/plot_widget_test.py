import vaex
import pytest
import bqplot
from vaex.jupyter.utils import _debounced_flush as flush
import vaex.jupyter.model


def test_plot_widget_bqplot(flush_guard):
    # basic coverage for now
    df = vaex.example()
    df.plot_widget(df.x, df.y)
    df.plot_widget(df.x.astype('float32'), df.y.astype('float32'))
    df.plot_widget(df.x.astype('float32'), df.y.astype('float32'), limits='minmax')
    flush(all=True)


def test_selection_event_calls(df, flush_guard):
    df.select(df.x > 3, name='bla')


def test_widget_selection(flush_guard):
    df = vaex.example()
    with pytest.raises(ValueError) as e:
        selection_widget_default = df.widget.selection_expression()
    assert "'default'" in str(e.value)

    counts = {'default': 0, 'pos': 0}
    @df.signal_selection_changed.connect
    def update(df, name):
        nonlocal counts
        counts[name] += 1
    count_pos = df.count(selection=df.x > 0)

    df.select(df.x > 0)
    selection_widget_default = df.widget.selection_expression()
    assert selection_widget_default.value.expression == '(x > 0)'

    selection_widget = df.widget.selection_expression(df.x > 0, name='pos')
    assert selection_widget_default.value.expression == '(x > 0)'
    # selection_widget = df.widget.selection(df.x > 0, name='pos')
    assert counts == {'default': 2, 'pos': 1}
    assert df.count(selection='pos') == count_pos
    selection_widget.v_model = 'x < 0'
    assert selection_widget.error_messages is None
    assert counts == {'default': 2, 'pos': 2}
    flush(all=True)


def test_data_array_view(flush_guard):
    df = vaex.example()

    x = vaex.jupyter.model.Axis(df=df, expression='x')
    y = vaex.jupyter.model.Axis(df=df, expression='y')
    view = df.widget.data_array(axes=[x, y])
    flush(all=True)
    assert view.model.grid is not None


def test_widget_histogram(flush_guard):
    df = vaex.example()
    assert df.widget is df.widget
    df.select_box(['x'], [[-10, 20]], name='check')

    check_range = df.count(selection='check')

    df.select(df.x > 0)
    check_positive = df.count(selection=True)
    histogram = df.widget.histogram('x', selection=[None, True], toolbar=True)

    flush()
    assert histogram.model.grid[1].sum() == check_positive  # for some reason, because 'x' it float32, we don't need -1
    toolbar = histogram.toolbar
    toolbar.interact_value = "pan-zoom"
    assert isinstance(histogram.plot.figure.interaction, bqplot.interacts.PanZoom)
    toolbar.interact_value = "select-x"
    assert isinstance(histogram.plot.figure.interaction, bqplot.interacts.BrushIntervalSelector)

    histogram.plot.figure.interaction.selected = [-10, 20]
    flush(all=True)
    assert histogram.model.grid.shape[0] == 2
    assert histogram.model.grid[1].sum() == check_range

    toolbar.interact_value = "doesnotexit"
    assert histogram.plot.figure.interaction is None
    # coverage
    histogram.plot.highlight(0)
    histogram.plot.highlight(None)

    vizdata = histogram.plot.mark.y.tolist()
    histogram.model.x_slice = 10
    assert histogram.plot.mark.y.tolist() == vizdata
    histogram.dimension_groups = 'slice'
    assert histogram.plot.mark.y.tolist() != vizdata


def test_widget_heatmap(flush_guard):
    df = vaex.example()
    df.select_rectangle('x', 'y', [[-10, 10], [-50, 50]], name='check')
    check_rectangle = df.count(selection='check')

    df.select(df.x > 0)
    check_positive = df.count(selection=True)
    heatmap = df.widget.heatmap('x', 'y', selection=[None, True])

    flush()
    assert heatmap.model.grid[1].sum().item() == check_positive-1
    toolbar = heatmap.toolbar
    toolbar.interact_value = "pan-zoom"
    assert isinstance(heatmap.plot.figure.interaction, bqplot.interacts.PanZoom)
    toolbar.interact_value = "select-rect"
    assert isinstance(heatmap.plot.figure.interaction, bqplot.interacts.BrushSelector)

    heatmap.plot.figure.interaction.selected_x = [-10, 10]
    heatmap.plot.figure.interaction.selected_y = [-50, 50]
    assert heatmap.model.grid.shape[0] == 2
    flush()
    assert heatmap.model.grid[1].sum().item() == check_rectangle

    toolbar.interact_value = "doesnotexit"
    assert heatmap.plot.figure.interaction is None

    # vizdata = heatmap.plot.mark.image.tolist()
    # heatmap.model.x_slice = 10
    # assert heatmap.plot.mark.image.tolist() != vizdata
    vizdata = heatmap.plot.mark.image.value
    heatmap.model.x.max = 10
    flush(all=True)
    assert heatmap.plot.mark.image.value != vizdata, "image should change"


# TODO
# def test_widget_pie(flush_guard):
#     df = vaex.example()
#     df['s'] = df.x < 0
#     df.categorize('s', labels=['neg', 'pos'], check=False, inplace=True)
#     hw = df.widget.pie('s')
#     flush()


def test_widget_process_circular(flush_guard):
    df = vaex.example()
    p = df.widget.progress_circular()
    df.sum(df.x)
    assert p.hidden is False
    assert p.value == 100


def test_widget_counter(flush_guard):
    df = vaex.example()
    c = df.widget.counter_processed()
    assert c.value == 0
    df.sum(df.x)
    assert c.value == len(df)


def test_widget_counter_selection(flush_guard):
    df = vaex.example()
    c = df.widget.counter_selection('test', lazy=True)
    assert c.value == 0
    # we don't want to *cause* an execution
    df.select(df.x > 0, name='test')
    assert c.value == 0
    # flush()
    # but will update when once happens
    df.sum(df.x)
    count_pos = df.count(selection='test')
    assert c.value == count_pos

    df.select(df.x < 0, name='test')
    assert c.value == count_pos
    df.sum(df.x)
    count_neg = df.count(selection='test')
    assert c.value == count_neg

    # existing selection, and non-lazy
    c = df.widget.counter_selection('test')
    assert c.value == count_neg
    df.select(df.x > 0, name='test')
    assert c.value == count_pos
    df.select(df.x < 0, name='test')
    assert c.value == count_neg
    flush(all=True)
