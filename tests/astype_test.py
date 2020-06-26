from common import *
import collections
import numpy as np

def test_astype(ds_local):
    ds = ds_local
    ds_original = ds.copy()
    #ds.columns['x'] = (ds.columns['x']*1).copy()  # convert non non-big endian for now
    ds['x'] = ds['x'].astype('f4')

    assert ds.x.evaluate().dtype == np.float32
    assert ds.x.tolist() == ds_original.x.evaluate().astype(np.float32).tolist()


def test_astype_str():
    df = vaex.from_arrays(x=['10,010', '-50,0', '11,111'])

    df['x'] = df['x'].str.replace(',', '').evaluate()
    df['x'] = (df['x'].astype('float')).astype('int64').evaluate()

    assert df.columns['x'].dtype == np.int64
    assert df.x.dtype == np.int64

def test_astype_dtype():
    df = vaex.from_arrays(x=[0, 1])
    assert df.x.astype(str).data_type() in [pa.string(), pa.large_string()]
    df = vaex.from_arrays(x=[np.nan, 1])
    # assert df.x.astype(str).dtype == vaex.column.str_type
    assert df.x.astype(str).data_type() in [pa.string(), pa.large_string()]
