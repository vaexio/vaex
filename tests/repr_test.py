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

def test_repr_df_long_string():
    long_string = "Hi there" * 100
    df = vaex.from_arrays(s=[long_string] * 100)
    assert long_string not in repr(df)
    assert long_string not in str(df)
    assert long_string not in df._repr_html_()
    assert long_string not in df._as_html_table(0, 10)
