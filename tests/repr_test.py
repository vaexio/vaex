from common import *
import pandas as pd
import datetime

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

    # as objects
    df = vaex.from_arrays(o=[{"something": long_string}] * 100)
    assert long_string not in repr(df)
    assert long_string not in str(df)
    assert long_string not in df._repr_html_()
    assert long_string not in df._as_html_table(0, 10)


def test_repr_from_pandas():
    dd_dict = {
        'boolean': [True, True, False, None, True],
        'text': ['This', 'is', 'some', 'text', 'so...'],
        'numbers_1': [1, 30, -2, 1.5, 0.000],
        'numbers_2': [1, None, -2, 1.5, 0.000],
        'numbers_3': [1, np.nan, -2, 1.5, 0.000],
        'time_1': [pd.NaT, datetime.date(2019, 1, 1), datetime.date(2019, 11, 1), datetime.date(2019, 1, 11), datetime.date(2019, 11, 11)],
        'time_2': [pd.NaT, None, pd.NaT, pd.NaT, pd.NaT],
    }

    dd = pd.DataFrame(dd_dict)
    dd['time_3'] = pd.to_timedelta(dd['time_2'] - dd['time_1'])
    ds = vaex.from_pandas(dd)
    repr_value = repr(ds)
    str_value = str(ds)
    assert 'NaT' in repr_value
    assert 'NaT' in str_value
