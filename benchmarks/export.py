import tempfile
import os

import numpy as np
import vaex


df = None

def setup_df(N, M):
    global df
    x = [np.arange(N, dtype=np.float64) for _ in range(M)] 
    df = vaex.from_dict({
        f'c{i}': x[i] for i in range(M)
    })

def time_export_plain(N, M):
    with tempfile.TemporaryDirectory() as tmpdir:
        df.export_hdf5(os.path.join(tmpdir, 'bench.hdf5'))
time_export_plain.setup = setup_df
time_export_plain.params = [[1024**2, 1024**2*16], [1, 4, 16]]
time_export_plain.param_names = ['N', 'M']


def time_export_correlated(N, M):
    names = df.get_column_names()
    new_names = [f't{i}' for i in range(M)]
    for i in range(M):
        df[f't{i}'] = sum(df[c] for c in names)
    dfc = df[new_names]
    with tempfile.TemporaryDirectory() as tmpdir:
        dfc.export_hdf5(os.path.join(tmpdir, 'bench.hdf5'))
time_export_correlated.setup = setup_df
time_export_correlated.params = [[1024**2, 1024**2*16], [1, 4, 16]]
time_export_correlated.param_names = ['N', 'M']

