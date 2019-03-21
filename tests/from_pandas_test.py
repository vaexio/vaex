import vaex
import numpy as np
import pandas as pd
import datetime


def test_from_pandas():
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
    dd['time_3'] = pd.to_timedelta(dd['time_2'])
    ds = vaex.from_pandas(dd)
    print(ds)