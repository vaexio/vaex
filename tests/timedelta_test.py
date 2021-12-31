import vaex
import pandas as pd
import numpy as np


def test_timedelta_methods():
    delta = np.array([187201, 1449339, 11264958, -181614], dtype='timedelta64[s]')
    df = vaex.from_arrays(delta=delta)
    pdf = pd.DataFrame({'delta': pd.Series(delta, dtype=delta.dtype)})

    assert df.delta.td.days.tolist() == pdf.delta.dt.days.tolist()
    assert df.delta.td.seconds.tolist() == pdf.delta.dt.seconds.tolist()
    assert df.delta.td.microseconds.tolist() == pdf.delta.dt.microseconds.tolist()
    assert df.delta.td.nanoseconds.tolist() == pdf.delta.dt.nanoseconds.tolist()
    assert df.delta.td.total_seconds().tolist() == pdf.delta.dt.total_seconds().tolist()
