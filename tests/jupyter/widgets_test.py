import vaex.jupyter.traitlets as vt
import vaex
from vaex.jupyter.utils import _debounced_flush as flush
import vaex.jupyter.widgets


def test_selection_toggle_list(flush_guard):
    df = vaex.from_scalars(x=1)
    widget = vaex.jupyter.widgets.SelectionToggleList(df=df)
    assert widget.selection_names == []
    assert widget.value == []
    df.select('x > 0')
    assert widget.selection_names == ['default']
    assert widget.value == []
    widget.value = ['default']
    df.select('x < 0', name='neg')
    assert widget.selection_names == ['default', 'neg']
    assert widget.value == ['default']
    df.select_nothing('default')
    assert widget.selection_names == ['neg']
    assert widget.value == []
    df.select('x > 0')
    assert widget.selection_names == ['default', 'neg']
    assert widget.value == []

    widget.value = ['default', 'neg']
    df.select_nothing('default')
    assert widget.value == ['neg']
    df.select_nothing('neg')
    assert widget.value == []


def test_column_list_traitlets(flush_guard):
    df = vaex.from_scalars(x=1, y=2)
    df['z'] = df.x + df.y
    column_list = vt.ColumnsMixin(df=df)
    assert len(column_list.columns) == 3
    df['w'] = df.z * 2
    assert len(column_list.columns) == 4
    del df['w']
    assert len(column_list.columns) == 3


def test_expression(flush_guard):
    df = vaex.example()
    expression = df.widget.expression()
    assert expression.value is None
    expression.value = 'x'
    assert expression.value.expression == 'x'
    assert expression.valid
    assert expression.error_messages is None
    assert "good" in expression.success_messages
    flush(all=True)
    assert expression.error_messages is None
    assert expression.success_messages is None

    expression.v_model = 'x+'
    assert not expression.valid
    assert expression.error_messages is not None
    assert expression.success_messages is None
    flush()
    assert expression.error_messages is not None
    assert expression.success_messages is None

    expression = df.widget.expression(df.y)
    assert expression.value == 'y'

    axis = vaex.jupyter.model.Axis(df=df, expression=df.x + 2)
    expression = df.widget.expression(axis)
    assert str(expression.value) == '(x + 2)'
    axis.expression = df.x + 3
    assert str(expression.value) == '(x + 3)'
    flush(all=True)


def test_column(flush_guard):
    df = vaex.example()
    column = df.widget.column()
    assert column.value is None
    column = df.widget.column(df.y)
    assert column.value == 'y'

    axis = vaex.jupyter.model.Axis(df=df, expression=df.x)
    column_widget = df.widget.column(axis)
    assert str(column_widget.value) == 'x'
    axis.expression = df.y
    assert str(column_widget.value) == 'y'
    flush(all=True)
