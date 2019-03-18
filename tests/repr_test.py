from common import *

def test_repr_default(ds_local):
    ds = ds_local
    code = ds._repr_mimebundle_()['text/plain']
    assert 'x' in code

def test_repr_html(ds_local):
    ds = ds_local
    code = ds._repr_html_()
    assert 'x' in code

def test_mask(ds_local):
    ds = ds_local
    code = ds._repr_html_()
    assert "'--'" not in code
    assert "--" in code

    code = ds._repr_mimebundle_()['text/plain']
    assert "'--'" not in code
    assert "--" in code

def test_repr_expression(ds_local):
    df = ds_local
    assert 'Error' not in repr(df.x)
