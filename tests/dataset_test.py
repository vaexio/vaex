from io import BytesIO
import pickle
from pathlib import Path

import numpy as np
import pytest
import pyarrow.parquet

import vaex
import vaex.dataset as dataset

pytest.importorskip("blake3")

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
    ds = dataset.DatasetArrays(x=x, y=y).hashed()
    assert ds == rebuild(ds)


def test_no_hash():
    x1 = np.arange(10)
    y1 = x1**2
    ds1 = dataset.DatasetArrays(x=x1, y=y1)

    x2 = np.arange(10)
    y2 = x2**2
    ds2 = dataset.DatasetArrays(x=x2, y=y2)

    with pytest.raises(ValueError, match='.*hash.*'):
        ds1 == ds2
    with pytest.raises(ValueError, match='.*hash.*'):
        ds1 == ds2.hashed()
    with pytest.raises(ValueError, match='.*hash.*'):
        ds1.hashed() == ds2
    ds1.hashed() == ds2.hashed()


def test_array_eq():
    x1 = np.arange(10)
    y1 = x1**2
    ds1 = dataset.DatasetArrays(x=x1, y=y1).hashed()
    assert ds1['x'] is x1
    assert ds1['y'] is y1

    x2 = np.arange(10)
    y2 = x2**2
    ds2 = dataset.DatasetArrays(x=x2, y=y2).hashed()
    assert ds2['x'] is x2
    assert ds2['y'] is y2

    # different data, but same ids/hashes
    assert ds1 == ds2
    assert ds1 == rebuild(ds2)


def test_array_rename():
    x = np.arange(10)
    y = x**2
    ds1 = dataset.DatasetArrays(x=x, y=y).hashed()
    ds2 = ds1.renamed({'x': 'z'})
    assert ds2['y'] is y
    assert ds2['z'] is x

    assert 'z' in list(ds2.chunk_iterator(['z']))[0][-1]

    assert ds1 != ds2
    assert rebuild(ds1) != rebuild(ds2)

    ds3 = ds2.renamed({'z': 'x'})
    assert ds3.original is ds1, "no nested renaming"
    assert ds3['y'] is y
    assert ds3['x'] is x

    # different data, but same ids/hashes
    assert ds1 == ds3
    assert rebuild(ds1) == rebuild(ds3)

    # testing that
    # {'a': 'x', 'b': 'y'} and {'x': 'a', 'b': 'z', 'c', 'q'} -> {'b': 'z', 'c': 'q'}
    ds1 = dataset.DatasetArrays(a=x, b=y, c=x+y)
    ds2 = ds1.renamed({'a': 'x', 'b': 'y'})
    ds3 = ds2.renamed({'x': 'a', 'b': 'z', 'c': 'q'})
    assert ds3.original is ds1
    assert ds3.renaming == {'b': 'z', 'c': 'q'}


def test_merge():
    x = np.arange(10)
    y = x**2
    ds1 = dataset.DatasetArrays(x=x, y=y).hashed()
    dsx = dataset.DatasetArrays(x=x)
    dsy = dataset.DatasetArrays(y=y)
    ds2 = dsx.merged(dsy).hashed()

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
    assert ds1.hashed() != ds2.hashed()
    assert ds2.hashed() == ds2b.hashed()
    assert ds2.hashed() != ds2c.hashed()
    assert ds1.row_count == 10
    assert ds2.row_count == 7
    assert ds2b.row_count == 7
    assert ds2c.row_count == 8
    assert ds2['x'].tolist() == x[1:8].tolist()


    ds3 = dataset.DatasetArrays(x=x[1:8], y=y[1:8])

    assert ds2.hashed() != ds3.hashed()
    assert rebuild(ds1).hashed() != rebuild(ds2).hashed()



def test_filter():
    x = np.arange(10)
    y = x**2
    filter = (x % 2) == 1
    ds = dataset.DatasetArrays(x=x, y=y)
    ds1 = dataset.DatasetFiltered(ds, filter=filter)
    ds1c = dataset.DatasetFiltered(ds, filter=(x % 2) == 1)
    ds2 = dataset.DatasetFiltered(ds, filter=(x % 3) == 1)
    assert ds1.hashed() != ds2.hashed()
    assert ds1.hashed() == ds1c.hashed()
    assert ds1.row_count == 5
    assert ds2.row_count == 3
    assert ds1.slice(0, 1).row_count == 1
    assert ds1.slice(1, 3).row_count == 2
    iter = ds1.chunk_iterator(['x', 'y'], chunk_size=2)
    for i in range(3):
        i1, i2, chunks = next(iter)
        assert i1 == i*2
        assert i2 == min(5, (i + 1) * 2)
        chunks['x'].tolist() == x[filter][i1:i2].tolist()
        chunks['y'].tolist() == y[filter][i1:i2].tolist()


    assert ds1['x'].tolist() == x[filter].tolist()
    assert rebuild(ds1).hashed() != rebuild(ds2).hashed()



def test_take():
    x = np.arange(10)
    y = x**2
    ds1 = dataset.DatasetArrays(x=x, y=y)
    indices = np.array([1, 2, 5])
    indices_other = np.array([1, 2, 6])
    ds2 = ds1.take(indices)
    ds2b = ds1.take(indices)
    ds2c = ds1.take(indices_other)
    assert ds1.hashed() != ds2.hashed()
    assert ds2.hashed() == ds2b.hashed()
    assert ds2.hashed() != ds2c.hashed()
    assert ds1.row_count == 10
    assert ds2.row_count == len(indices)
    assert ds2b.row_count == len(indices)
    assert ds2c.row_count == len(indices_other)
    assert ds2['x'].tolist() == x[indices].tolist()

    ds3 = dataset.DatasetArrays(x=x[indices], y=y[indices])

    assert ds2.hashed() != ds3.hashed()
    assert rebuild(ds1).hashed() != rebuild(ds2).hashed()


def test_project():
    x = np.arange(10)
    y = x**2
    ds1 = dataset.DatasetArrays(x=x, y=y)
    ds2 = ds1.project('x')
    ds3 = dataset.DatasetArrays(x=x)
    assert ds1.hashed() != ds2.hashed()
    assert ds2.hashed() == ds3.hashed()
    assert rebuild(ds2).hashed() == rebuild(ds3).hashed()


def test_drop():
    x = np.arange(10)
    y = x**2
    ds1 = dataset.DatasetArrays(x=x, y=y)
    ds2 = ds1.dropped('x')
    assert 'x' not in ds2
    ds3 = ds1.dropped('y')
    assert 'y' not in ds3
    assert ds1.hashed() == ds2.merged(ds3).hashed()
    assert rebuild(ds1).hashed() == rebuild(ds2.merged(ds3)).hashed()


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

    # an empty list of columns follows a different codepath
    assert list(dsc.chunk_iterator([])) == [(0, 10, {})]
    assert list(dsc.chunk_iterator([], start=5, end=10)) == [(0, 5, {})]


def test_example():
    df = vaex.example().hashed()
    path_data = HERE / 'data' / 'test.hdf5'
    assert isinstance(df, vaex.dataframe.DataFrame)
    assert isinstance(df.dataset, vaex.hdf5.dataset.Hdf5MemoryMapped)
    assert rebuild(df.dataset) == df.dataset


def test_hashable():
    # tests if we can use datasets as keys of dicts
    x = np.arange(10)
    y = x**2
    ds1 = dataset.DatasetArrays(x=x, y=y).hashed()
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
    assert not path_hashes.exists()
    df2 = df2.hashed()
    assert df2.dataset._hash_calculations > 0
    assert path_hashes.exists()

    # and pickling
    ds = df2.dataset
    ds2 = rebuild(ds)
    assert ds2._hash_calculations == 0
    assert ds == ds2

    df3 = vaex.open(str(path_data))
    ds3 = df3.dataset
    assert ds3._hash_calculations == 0
    assert ds3 == ds2


def test_chunk_iterator():
    x = np.arange(10)
    y = x**2
    ds = dataset.DatasetArrays(x=x, y=y)
    chunk_it = ds.chunk_iterator(['y'], chunk_size=4)
    i1, i2, chunk0 = next(chunk_it)
    assert chunk0['y'].tolist() == y[0:4].tolist()
    assert i1 == 0
    assert i2 == 4

    i1, i2, chunk1 = next(chunk_it)
    assert chunk1['y'].tolist() == y[4:8].tolist()
    assert i1 == 4
    assert i2 == 8

    i1, i2, chunk2 = next(chunk_it)
    assert chunk2['y'].tolist() == y[8:].tolist()
    assert i1 == 8
    assert i2 == 10


@pytest.mark.parametrize("l1", list(range(1, 6)))
@pytest.mark.parametrize("l2", list(range(1, 6)))
def test_concat_chunk_iterator(l1, l2):
    i1 = 0
    i2 = i1 + l1
    i3 = i2 + l2
    x = np.arange(10)
    y = x**2
    g = x // 3
    ds = vaex.dataset.DatasetArrays(x=x, y=y, g=g)
    df_original = df = vaex.from_dataset(ds)
    df1 = df[i1:i2]
    df2 = df[i2:i3]
    df3 = df[i3:]
    df = vaex.concat([df1, df2, df3])
    ds_full = ds = df.dataset

    # very similar to the arrow/datase_test.py parquet test
    iter = ds.chunk_iterator(['x', 'y'], chunk_size=2)
    for i in range(5):
        i1, i2, chunks = next(iter)
        assert i1 == i*2
        assert i2 == (i + 1) * 2
        chunks['x'].tolist() == x[i1:i2].tolist()
        chunks['y'].tolist() == y[i1:i2].tolist()


    # no columns
    iter = ds.chunk_iterator([], chunk_size=2)
    for i in range(5):
        i1, i2, chunks = next(iter)
        assert i1 == i*2
        assert i2 == (i + 1) * 2

    ds = ds[1:10]
    assert 'x' in ds
    assert ds.row_count == 9
    iter = ds.chunk_iterator(['x', 'y'], chunk_size=2)
    for i in range(5):
        i1, i2, chunks = next(iter)
        if i == 4:
            assert i1 == 8
            assert i2 == 9
        else:
            assert i1 == i*2
            assert i2 == (i + 1) * 2
        # chunks = chunks
        chunks['x'].tolist() == x[i1:i2].tolist()
        chunks['y'].tolist() == y[i1:i2].tolist()

    ds = ds[1:9]
    assert ds.row_count == 8
    iter = ds.chunk_iterator(['x', 'y'], chunk_size=2)
    for i in range(4):
        i1, i2, chunks = next(iter)
        assert i1 == i*2
        assert i2 == (i + 1) * 2
        chunks['x'].tolist() == x[i1:i2].tolist()
        chunks['y'].tolist() == y[i1:i2].tolist()

    # no columns
    iter = ds.chunk_iterator([], chunk_size=2)
    for i in range(4):
        i1, i2, chunks = next(iter)
        assert i1 == i*2
        assert i2 == (i + 1) * 2

    # again, but here we skip of total of a chunk_size at the end
    ds = ds_full[:8]
    # import pdb; pdb.set_trace()
    assert ds.row_count == 8
    iter = ds.chunk_iterator(['x', 'y'], chunk_size=2)
    for i in range(4):
        i1, i2, chunks = next(iter)
        assert i1 == i*2
        assert i2 == (i + 1) * 2
        chunks['x'].tolist() == x[i1:i2].tolist()
        chunks['y'].tolist() == y[i1:i2].tolist()


    for i in range(9):
        for j in range(i+1, 10):
            ds = ds_full.slice(i, j)
            values = []
            for i1, i2, chunks in ds.chunk_iterator(['x']):
                values.extend(chunks['x'].tolist())
            assert x[i:j].tolist() == values

    assert df.x.tolist() == x.tolist()
    assert df.g.tolist() == g.tolist()

    ds_dropped = ds.dropped('x')
    assert 'x' not in ds_dropped

