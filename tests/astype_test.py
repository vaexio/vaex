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
