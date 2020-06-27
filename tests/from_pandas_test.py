import pandas as pd
import numpy as np
import vaex
import datetime


def test_from_pandas():
    dd_dict = {
        'boolean': [True, True, False, None, True],
        'text': ['This', 'is', 'some', 'text', 'so...'],
        'text_missing': pd.Series(['Some', 'parts', None, 'missing', None], dtype='string'),
        'float': [1, 30, -2, 1.5, 0.000],
        'float_missing': [1, None, -2, 1.5, 0.000],
        'int_missing': pd.Series([1, None, 5, 1, 10], dtype='Int64'),
        'datetime_1': [pd.NaT, datetime.datetime(2019, 1, 1, 1, 1, 1), datetime.datetime(2019, 1, 1, 1, 1, 1), datetime.datetime(2019, 1, 1, 1, 1, 1), datetime.datetime(2019, 1, 1, 1, 1, 1)],
        'datetime_2': [pd.NaT, None, pd.NaT, pd.NaT, pd.NaT],
        'datetime_3': [pd.Timedelta('1M'), pd.Timedelta('1D'), pd.Timedelta('100M'), pd.Timedelta('2D'), pd.Timedelta('1H')],
        'datetime_4': [pd.Timestamp('2001-1-1 2:2:11'), pd.Timestamp('2001-12'), pd.Timestamp('2001-10-1'), pd.Timestamp('2001-03-1 2:2:11'), pd.Timestamp('2001-1-1 2:2:11')],
        'datetime_5': [datetime.date(2010, 1, 1), datetime.date(2010, 1, 1), datetime.date(2010, 1, 1), datetime.date(2010, 1, 1), datetime.date(2010, 1, 1)],
        'datetime_6': [datetime.time(21, 1, 1), datetime.time(21, 1, 1), datetime.time(21, 1, 1), datetime.time(21, 1, 1), datetime.time(21, 1, 1)],
    }

    # Get pandas dataframe
    pandas_df = pd.DataFrame(dd_dict)
    pandas_df['datetime_7'] = pd.to_timedelta(pandas_df['datetime_2'] - pandas_df['datetime_1'])
    vaex_df = vaex.from_pandas(pandas_df)
    repr_value = repr(vaex_df)
    str_value = str(vaex_df)

    assert 'NaT' in repr_value
    assert 'NaT' in str_value
    assert '--' in repr_value
    assert '--' in str_value

    # string columns are now arrows arrays
    # assert vaex_df.text_missing.is_masked == True
    assert vaex_df.int_missing.is_masked == True
    assert vaex_df.float_missing.is_masked == False
    assert vaex_df.int_missing.tolist() == [1, None, 5, 1, 10]
    assert vaex_df.text_missing.tolist() == ['Some', 'parts', None, 'missing', None]
    assert vaex_df.float_missing.values[[0, 2, 3, 4]].tolist() == [1.0, -2.0, 1.5, 0.0]
    assert np.isnan(vaex_df.float_missing.values[1])
