import vaex
import pandas as pd
import numpy as np


def test_timedelta_methods():
    delta = np.array([17658720110, 11047049384039, 40712636304958, -18161254954], dtype='timedelta64[s]')
    df = vaex.from_arrays(delta=delta)
    pdf = pd.DataFrame({'delta': pd.Series(delta, dtype=delta.dtype)})

    assert df.delta.td.days.tolist() == pdf.delta.dt.days.tolist()
    assert df.delta.td.seconds.tolist() == pdf.delta.dt.seconds.tolist()
    assert df.delta.td.microseconds.tolist() == pdf.delta.dt.microseconds.tolist()
    assert df.delta.td.nanoseconds.tolist() == pdf.delta.dt.nanoseconds.tolist()
    assert df.delta.td.total_seconds().tolist() == pdf.delta.dt.total_seconds().tolist()
