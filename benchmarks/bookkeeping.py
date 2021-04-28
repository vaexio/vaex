import vaex
import numpy as np


# cache of dataframes
dfs = {}
dfs2 = {}

def time_create(n):
    arr = np.random.random(size=(n, 10))
    df = vaex.from_arrays(**{str(k):arr[k] for k in range(n)})
    return df
time_create.params = [10, 50, 100, 1000]


def setup_copy(n):
    dfs[n] = time_create(n)
def time_copy(n):
    arr = np.random.random(size=(n, 10))
    dfs[n].copy()
time_copy.setup = setup_copy
time_copy.params = time_create.params


def setup_concat(n):
    dfs[n] = time_create(n)
    dfs2[n] = time_create(n)
def time_concat(n):
    df1 = dfs[n]
    df2 = dfs2[n]
    df1.concat(df2)
time_concat.setup = setup_concat
time_concat.params = time_create.params
