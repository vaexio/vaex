from io import BytesIO
import pickle
from pathlib import Path

import numpy as np
import pytest
import pyarrow as pa

import vaex
import vaex.dataset as dataset
import vaex.encoding

from common import small_buffer

pytest.importorskip("blake3")

HERE = Path(__file__).parent

def rebuild_with_skip(ds, skip):
    repr(ds)  # for coverage
    # encoding and decode
    encoding = vaex.encoding.Encoding()
    encoding.set_object_spec(skip.id, None)  # this will cause it to skip serialization
    data = encoding.encode('dataset', ds)
    assert encoding._object_specs[skip.id] is None
    del encoding._object_specs[skip.id]
    blob = vaex.encoding.serialize(data, encoding)

    encoding = vaex.encoding.Encoding()
    encoding.set_object(skip.id, skip)
    data = vaex.encoding.deserialize(blob, encoding)
    return encoding.decode('dataset', data)


def test_array_rebuild_dataset(rebuild_dataset):
    x = np.arange(10)
    y = x**2
    ds = dataset.DatasetArrays(x=x, y=y).hashed()
    assert ds == rebuild_dataset(ds)


def test_no_hash():
    x1 = np.arange(10)
    y1 = x1**2
    ds1 = dataset.DatasetArrays(x=x1, y=y1, hashed=False)

    x2 = np.arange(10)
    y2 = x2**2
    ds2 = dataset.DatasetArrays(x=x2, y=y2, hashed=False)

    assert ds1 != ds2
    assert ds1 != ds2.hashed()
    assert ds1.hashed() != ds2
    assert ds1.hashed() == ds2.hashed()


def test_merge_array():
    # we prefer a single DatasetArrays instead of a tree with Merged
    x = np.arange(10)
    df = vaex.from_arrays(x=x)
    df['y'] = x**2
    assert isinstance(df.dataset, vaex.dataset.DatasetArrays)


def test_array_eq(rebuild_dataset):
    x1 = np.arange(10)
    y1 = x1**2
    ds1 = dataset.DatasetArrays(x=x1, y=y1).hashed()
    assert ds1['x'] is x1
    assert ds1['y'] is y1

    assert set(ds1.leafs()) == {ds1}

    x2 = np.arange(10)
    y2 = x2**2
    ds2 = dataset.DatasetArrays(x=x2, y=y2).hashed()
    assert ds2['x'] is x2
    assert ds2['y'] is y2

    # different data, but same ids/hashes
    assert ds1 == ds2
    assert ds1 == rebuild_dataset(ds2)


def test_array_rename(rebuild_dataset):
    x = np.arange(10)
    y = x**2
    ds1 = dataset.DatasetArrays(x=x, y=y).hashed()
    ds2 = ds1.renamed({'x': 'z'})
    assert set(ds2.leafs()) == {ds1}
    assert ds2['y'] is y
    assert ds2['z'] is x

    assert 'z' in list(ds2.chunk_iterator(['z']))[0][-1]

    assert ds1 != ds2
    assert rebuild_dataset(ds1) != rebuild_dataset(ds2)

    ds3 = ds2.renamed({'z': 'x'})
    assert ds3.original is ds1, "no nested renaming"
    assert ds3['y'] is y
    assert ds3['x'] is x

    # different data, but same ids/hashes
    assert ds1 == ds3
    assert rebuild_dataset(ds1) == rebuild_dataset(ds3)

    # testing that
    # {'a': 'x', 'b': 'y'} and {'x': 'a', 'b': 'z', 'c', 'q'} -> {'b': 'z', 'c': 'q'}
    ds1 = dataset.DatasetArrays(a=x, b=y, c=x+y)
    ds2 = ds1.renamed({'a': 'x', 'b': 'y'})
    ds3 = ds2.renamed({'x': 'a', 'b': 'z', 'c': 'q'})
    assert ds3.original is ds1
    assert ds3.renaming == {'b': 'z', 'c': 'q'}

    dsr = rebuild_with_skip(ds2, ds1)
    assert dsr == ds2


def test_merge(rebuild_dataset, array_factory):
    x = array_factory(np.arange(10))
    y = array_factory(np.arange(10)**2)
    ds1 = dataset.DatasetArrays(x=x, y=y).hashed()
    dsx = dataset.DatasetArrays(x=x)
    dsy = dataset.DatasetArrays(y=y)
    ds2 = dsx.merged(dsy).hashed()
    assert set(ds2.leafs()) == {dsx.hashed(), dsy.hashed()}

    assert ds1 == ds2
    assert rebuild_dataset(ds1) == rebuild_dataset(ds2)

    with pytest.raises(NameError):
        ds2.merged(dsx)

    assert rebuild_with_skip(ds2, dsx.hashed()) == ds1

def test_slice_column():
    # slicing a colunm type should keep it column type
    x = np.arange(10)
    y = x**2
    ds1 = dataset.DatasetArrays(x=x, y=y)
    indices = np.array([1, 2, 5, 7, 9])
    ds2 = ds1.take(indices)
    ds3 = ds2[1:3]
    assert isinstance(ds3['x'], vaex.column.ColumnIndexed)


def test_slice(rebuild_dataset):
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
    assert rebuild_dataset(ds1.hashed()) != rebuild_dataset(ds2.hashed())
    # TODO: support unhashed?
    assert rebuild_dataset(ds1).hashed() != rebuild_dataset(ds2).hashed()

    assert rebuild_with_skip(ds2.hashed(), ds1.hashed()) == ds2.hashed()



def test_filter(rebuild_dataset):
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
    # TODO unhashed
    # assert rebuild_dataset(ds1).hashed() != rebuild_dataset(ds2).hashed()
    assert rebuild_dataset(ds1.hashed()) != rebuild_dataset(ds2.hashed())
    assert rebuild_with_skip(ds2.hashed(), ds.hashed()) == ds2.hashed()

    # testing the encoding of the expression instead of the array
    df = vaex.from_arrays(x=x, y=y)
    df['z'] = df.x % 2
    dff = df[df.z == 1]
    dff._push_down_filter()
    # df = vaex.from_arrays(x=[11, 12, 13, 15], y=[33, 44, 55, 66])
    ds = dff.dataset.hashed()
    # assert ds.state is not None
    assert ds == rebuild_dataset(ds)


def test_take_masked(df_factory_numpy):
    df_factory = df_factory_numpy
    df = df_factory(x=[0, 1, None, 3], y=[2, 3, 4, 5])
    assert df.is_masked('x')
    assert not df.is_masked('y')

    dft = df.take([0, 1])
    assert dft.is_masked('x')
    assert not dft.is_masked('y')

    dft = df.copy()
    # -1 is used for missing values
    dft.dataset = dft.dataset.take(np.array([0, 1, -1]), masked=True)
    assert dft.is_masked('x')
    assert dft.is_masked('y')


def test_take(rebuild_dataset):
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
    assert rebuild_dataset(ds1).hashed() != rebuild_dataset(ds2).hashed()


def test_project(rebuild_dataset):
    x = np.arange(10)
    y = x**2
    ds1 = dataset.DatasetArrays(x=x, y=y)
    ds2 = ds1.project('x')
    ds3 = dataset.DatasetArrays(x=x)
    assert ds1.hashed() != ds2.hashed()
    assert ds2.hashed() == ds3.hashed()
    assert rebuild_dataset(ds2).hashed() == rebuild_dataset(ds3).hashed()


def test_drop(rebuild_dataset):
    x = np.arange(10)
    y = x**2
    ds1 = dataset.DatasetArrays(x=x, y=y)
    ds2 = ds1.dropped('x')
    assert 'x' not in ds2
    ds3 = ds1.dropped('y')
    assert 'y' not in ds3
    assert ds1.hashed() == ds2.merged(ds3).hashed()
    assert rebuild_dataset(ds1).hashed() == rebuild_dataset(ds2.merged(ds3)).hashed()

    ds1b = dataset.DatasetArrays(x=x, y=y)
    assert ds1.fingerprint == ds1b.fingerprint
    ds2b = ds1.dropped('x')
    assert ds2.fingerprint == ds2b.fingerprint


def test_concat(rebuild_dataset):
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

    assert rebuild_dataset(dsc).hashed() == dsc.hashed()



def test_example(rebuild_dataset):
    df = vaex.example().hashed()
    path_data = HERE / 'data' / 'test.hdf5'
    assert isinstance(df, vaex.dataframe.DataFrame)
    assert isinstance(df.dataset, vaex.hdf5.dataset.Hdf5MemoryMapped)
    assert rebuild_dataset(df.dataset) == df.dataset


def test_hashable(rebuild_dataset):
    # tests if we can use datasets as keys of dicts
    x = np.arange(10)
    y = x**2
    ds1 = dataset.DatasetArrays(x=x, y=y).hashed()
    df = vaex.example()
    some_dict = {ds1: '1', df.dataset: '2'}
    assert some_dict[ds1] == '1'
    assert some_dict[df.dataset] == '2'

    assert some_dict[rebuild_dataset(ds1)] == '1'
    assert some_dict[rebuild_dataset(df.dataset)] == '2'


def test_cache_hash(rebuild_dataset, tmpdir):
    # TODO: what if the directory is not writable?
    # ds1 = dataset.DatasetArrays(x=x, y=y)
    path_data = tmpdir / 'test.hdf5'
    if path_data.exists():
        path_data.unlink()
    path_hashes = tmpdir / 'test.hdf5.d' / 'hashes.yaml'
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
    ds2 = rebuild_dataset(ds)
    assert ds2._hash_calculations == 0
    assert ds == ds2

    df3 = vaex.open(str(path_data))
    ds3 = df3.dataset
    assert ds3._hash_calculations == 0
    assert ds3 == ds2
    df3.close()


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


def test_columns():
    df = vaex.from_arrays(x=vaex.vrange(0, 10))
    ar = pa.array(['foo', 'bar'])
    df = vaex.from_arrays(x=vaex.column.ColumnStringArrow.from_arrow(ar))


def test_cache(buffer_size):
    x = np.arange(10)
    y = x**2
    z = 2*x
    df = vaex.from_arrays(x=x, y=y, z=z)

    # first test the dataset interace
    cache = {}
    ds = vaex.dataset.DatasetCached(df.dataset, ['x', 'y'], cache)
    iter = ds.chunk_iterator(['x', 'y'], chunk_size=2)
    for i in range(5):
        i1, i2, chunks = next(iter)
        assert i1 == i*2
        assert i2 == (i + 1) * 2
        chunks['x'].tolist() == x[i1:i2].tolist()
        chunks['y'].tolist() == y[i1:i2].tolist()

    # more of an integration test
    df['q'] = df.x + df.y
    with small_buffer(df, 3):
        cache = {}
        df.dataset = vaex.dataset.DatasetCached(df.dataset, ['x', 'y'], cache)
        df.z.sum()
        assert len(cache) == 0
        df.x.sum()
        # it will also fill up due to dtype evaluation
        assert len(cache) == 2
        df.q.sum()
        assert len(cache) == 4
        df.y.sum()
        assert len(cache) == 4
