from io import BytesIO
import pickle
from pathlib import Path

import numpy as np
import pytest

import vaex
import vaex.dataset as dataset

HERE = Path(__file__).parent


def rebuild(ds):
    # pick and unpickle
    f = BytesIO()
    picked = pickle.dump(ds, f)
    f.seek(0)
    return pickle.load(f)

def test_array_pickle():
    x = np.arange(10)
    y = x**2
    ds = dataset.DatasetArrays(x=x, y=y)
    assert ds == rebuild(ds)


def test_array_eq():
    x1 = np.arange(10)
    y1 = x1**2
    ds1 = dataset.DatasetArrays(x=x1, y=y1)
    assert ds1['x'] is x1
    assert ds1['y'] is y1

    x2 = np.arange(10)
    y2 = x2**2
    ds2 = dataset.DatasetArrays(x=x2, y=y2)
    assert ds2['x'] is x2
    assert ds2['y'] is y2

    # different data, but same ids/hashes
    assert ds1 == ds2
    assert ds1 == rebuild(ds2)


def test_array_rename():
    x = np.arange(10)
    y = x**2
    ds1 = dataset.DatasetArrays(x=x, y=y)
    ds2 = ds1.renamed({'x': 'z'})
    assert ds2['y'] is y
    assert ds2['z'] is x

    assert ds1 != ds2
    assert rebuild(ds1) != rebuild(ds2)

    ds3 = ds2.renamed({'z': 'x'})
    assert ds3['y'] is y
    assert ds3['x'] is x

    # different data, but same ids/hashes
    assert ds1 == ds3
    assert rebuild(ds1) == rebuild(ds3)


def test_merge():
    x = np.arange(10)
    y = x**2
    ds1 = dataset.DatasetArrays(x=x, y=y)
    dsx = dataset.DatasetArrays(x=x)
    dsy = dataset.DatasetArrays(y=y)
    ds2 = dsx.merged(dsy)

    assert ds1 == ds2
    assert rebuild(ds1) == rebuild(ds2)

    with pytest.raises(NameError):
        ds2.merged(dsx)


def test_slice_column():
    # slicing a colunm type should keep it column type
    x = np.arange(10)
    y = x**2
    ds1 = dataset.DatasetArrays(x=x, y=y)
    indices = np.array([1, 2, 5, 7, 9])
    ds2 = ds1.take(indices)
    ds3 = ds2[1:3]
    assert isinstance(ds3['x'], vaex.column.ColumnIndexed)


def test_slice():
    x = np.arange(10)
    y = x**2
    ds1 = dataset.DatasetArrays(x=x, y=y)
    ds2 = ds1[1:8]
    ds2b = ds1[1:8]
    ds2c = ds1[1:9]
    assert ds1 != ds2
    assert ds2 == ds2b
    assert ds2 != ds2c
    assert ds1.row_count == 10
    assert ds2.row_count == 7
    assert ds2b.row_count == 7
    assert ds2c.row_count == 8
    assert ds2['x'].tolist() == x[1:8].tolist()


    ds3 = dataset.DatasetArrays(x=x[1:8], y=y[1:8])

    assert ds2 != ds3
    assert rebuild(ds1) != rebuild(ds2)


def test_take():
    x = np.arange(10)
    y = x**2
    ds1 = dataset.DatasetArrays(x=x, y=y)
    indices = np.array([1, 2, 5])
    indices_other = np.array([1, 2, 6])
    ds2 = ds1.take(indices)
    ds2b = ds1.take(indices)
    ds2c = ds1.take(indices_other)
    assert ds1 != ds2
    assert ds2 == ds2b
    assert ds2 != ds2c
    assert ds1.row_count == 10
    assert ds2.row_count == len(indices)
    assert ds2b.row_count == len(indices)
    assert ds2c.row_count == len(indices_other)
    assert ds2['x'].tolist() == x[indices].tolist()

    ds3 = dataset.DatasetArrays(x=x[indices], y=y[indices])

    assert ds2 != ds3
    assert rebuild(ds1) != rebuild(ds2)


def test_project():
    x = np.arange(10)
    y = x**2
    ds1 = dataset.DatasetArrays(x=x, y=y)
    ds2 = ds1.project('x')
    ds3 = dataset.DatasetArrays(x=x)
    assert ds1 != ds2
    assert ds2 == ds3
    assert rebuild(ds2) == rebuild(ds3)


def test_drop():
    x = np.arange(10)
    y = x**2
    ds1 = dataset.DatasetArrays(x=x, y=y)
    ds2 = ds1.dropped('x')
    assert 'x' not in ds2
    ds3 = ds1.dropped('y')
    assert 'y' not in ds3
    assert ds1 == ds2.merged(ds3)
    assert rebuild(ds1) == rebuild(ds2.merged(ds3))


def test_concat():
    x = np.arange(10)
    y = x**2
    ds = dataset.DatasetArrays(x=x, y=y)
    mid = 4
    ds1 = dataset.DatasetArrays(x=x[:mid], y=y[:mid])
    ds2 = dataset.DatasetArrays(y=y[mid:], x=x[mid:])  # order should not matter
    dsc = ds1.concat(ds2)
    assert ds.row_count == dsc.row_count
    assert dsc.row_count == ds1.row_count + ds2.row_count


def test_example():
    df = vaex.example()
    assert isinstance(df, vaex.dataframe.DataFrame)
    assert isinstance(df.dataset, vaex.hdf5.dataset.Hdf5MemoryMapped)
    assert rebuild(df.dataset) == df.dataset


def test_hashable():
    # tests if we can use datasets as keys of dicts
    x = np.arange(10)
    y = x**2
    ds1 = dataset.DatasetArrays(x=x, y=y)
    df = vaex.example()
    some_dict = {ds1: '1', df.dataset: '2'}
    assert some_dict[ds1] == '1'
    assert some_dict[df.dataset] == '2'

    assert some_dict[rebuild(ds1)] == '1'
    assert some_dict[rebuild(df.dataset)] == '2'


def test_cache_hash():
    # TODO: what if the directory is not writable?
    # ds1 = dataset.DatasetArrays(x=x, y=y)
    path_data = HERE / 'data' / 'test.hdf5'
    if path_data.exists():
        path_data.unlink()
    path_hashes = HERE / 'data' / 'test.hdf5.d' / 'hashes.yaml'
    if path_hashes.exists():
        path_hashes.unlink()
    df = vaex.example()[:10]
    df.export(str(path_data))
    df2 = vaex.open(str(path_data))
    assert df2.dataset._hash_calculations == 0
    assert path_hashes.exists()

    # and pickling
    ds = df2.dataset
    ds2 = rebuild(ds)
    assert ds2._hash_calculations == 0
    assert ds == ds2
