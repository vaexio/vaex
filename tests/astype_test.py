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

    df.rename_column(df.column_names[0], 'value')
    df['value'] = df['value'].str.replace(',', '').evaluate()
    df['value'] = (df['value'].astype('float')).astype('int64').evaluate()

    assert df.columns['value'].dtype == np.int64
    assert df.value.dtype == np.int64
