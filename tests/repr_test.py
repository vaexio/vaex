from common import *
import pandas as pd
import datetime


def test_repr_invalid_name(df):
    df['is'] = df.x * 1
    code = df._repr_mimebundle_()['text/plain']
    assert '_is' not in code, "the repr should show the aliased name"


def test_repr_default(df):
    code = df._repr_mimebundle_()['text/plain']
    assert 'x' in code


def test_repr_html(df):
    ds = df
    code = ds._repr_html_()
    assert 'x' in code


def test_repr_empty(df):
    df = df[df.x < 0]
    bundle = df._repr_mimebundle_()
    assert 'no rows' in bundle['text/plain'].lower()
    assert 'no rows' in bundle['text/html'].lower()


# TODO: it seems masked arrays + evaluate doesn't work well
# might have to do something with serializing it
def test_mask(df_local):
    ds = df_local
    code = ds._repr_html_()
    assert "'--'" not in code
    assert "--" in code

    code = ds._repr_mimebundle_()['text/plain']
    assert "'--'" not in code
    assert "--" in code


def test_repr_expression(df):
    df = df
    assert 'Error' not in repr(df.x)


def test_repr_df_long_string():
    long_string = "Hi there" * 100
    df = vaex.from_arrays(s=[long_string] * 100)
    assert long_string not in repr(df)
    assert long_string not in str(df)
    assert long_string not in df._repr_html_()
    assert long_string not in df._as_html_table(0, 10)

    # as objects
    df = vaex.from_arrays(o=[{"something": long_string}] * 100)
    assert long_string not in repr(df)
    assert long_string not in str(df)
    assert long_string not in df._repr_html_()
    assert long_string not in df._as_html_table(0, 10)


# TODO: because remote slicing of filtered datasets is not supported, we have a workaround
# we RMI the __repr__
def test_slice_filtered_remte(ds_remote):
    df = ds_remote
    dff = df[df.x > 0]
    assert "0.0bla" not in repr(dff[['x']])
