import pytest
import ipywidgets as widgets
import traitlets
from vaex.jupyter.traitlets import Expression
from unittest.mock import MagicMock
import vaex


class SomeWidget(widgets.Widget):
    df = traitlets.Instance(vaex.dataframe.DataFrame)
    expression = Expression().tag(sync=True)

    def __init__(self, df, **kwargs):
        self.df = df
        super().__init__(**kwargs)


def test_validate_expression(flush_guard):
    df = vaex.example()
    w = SomeWidget(df=df, expression=df.x+1)
    w.expression = '(x + 2)'
    assert w.expression.expression == '(x + 2)'
    with pytest.raises(SyntaxError):
        w.expression = 'x + '
    with pytest.raises(NameError):
        w.expression = 'x2 + 1'
    assert w.expression.expression == '(x + 2)'


def test_observe_expression(flush_guard):
    call_counter = MagicMock()
    df = vaex.example()
    w = SomeWidget(df=df, expression=df.x+1)
    w.observe(call_counter, 'expression')
    call_counter.assert_not_called()
    w.expression = '(x + 2)'
    call_counter.assert_called_once()

def test_to_json(flush_guard):
    df = vaex.example()
    w = SomeWidget(df=df, expression=df.x+1)
    state = w.get_state()
    assert state['expression'] == '(x + 1)'
    state = state.copy()
    state['expression'] = '(x + 2)'
    w.set_state(state)
    w.expression.expression == '(x + 2)'