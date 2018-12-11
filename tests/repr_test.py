from common import *

def test_repr_default(ds_local):
    ds = ds_local
    code = repr(ds)
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

    code = repr(ds_local)
    assert "'--'" not in code
    assert "--" in code
