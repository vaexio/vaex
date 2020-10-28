import pytest
from pathlib import Path

import pyarrow.parquet
import numpy as np
import vaex
import vaex.arrow.dataset

HERE = Path(__file__).parent

@pytest.mark.parametrize("as_stream", [True, False])
def test_arrow(tmpdir, as_stream, rebuild_dataset):
    df = vaex.from_arrays(x=[1,2])
    path = str(tmpdir.join('test.arrow'))
    df.export_arrow(path, as_stream=as_stream)
    df = vaex.open(path)
    assert rebuild_dataset(df.dataset) == df.dataset


@pytest.mark.parametrize("lazy", [False, True])
def test_csv(tmpdir, df_filtered, rebuild_dataframe, lazy):
    path = tmpdir / "test.csv"
    df_filtered.drop('obj', inplace=True)
    df_filtered.drop('datetime', inplace=True)
    df_filtered.drop('timedelta', inplace=True)
    df_filtered.export_csv(path)
    df = vaex.from_csv_arrow(path, lazy=lazy)

    df2 = rebuild_dataframe(df)
    assert df2.dataset == df.dataset
    assert df2.dataset.fingerprint == df.dataset.fingerprint
    assert df2.x.tolist() == df.x.tolist()


@pytest.mark.parametrize("l1", list(range(1, 6)))
@pytest.mark.parametrize("l2", list(range(1, 6)))
def test_parquet(l1, l2, rebuild_dataset):
    i1 = 0
    i2 = i1 + l1
    i3 = i2 + l2

    x = np.arange(10)
    y = x**2
    g = x // 3
    ds = vaex.dataset.DatasetArrays(x=x, y=y, g=g)
    df = vaex.from_dataset(ds)
    path1 = HERE.parent / 'data' / 'parquet' / 'test1.parquet'
    path2 = HERE.parent / 'data' / 'parquet' / 'test2.parquet'
    path3 = HERE.parent / 'data' / 'parquet' / 'test3.parquet'
    path1.parent.mkdir(exist_ok=True)
    # df.export(str(path))
    pyarrow.parquet.write_table(df[i1:i2].to_arrow_table(), str(path1), row_group_size=2)
    pyarrow.parquet.write_table(df[i2:i3].to_arrow_table(), str(path2), row_group_size=2)
    pyarrow.parquet.write_table(df[i3:].to_arrow_table(), str(path3), row_group_size=2)
    ds = vaex.arrow.dataset.open_parquet([str(path1), str(path2), str(path3)])
    # TODO: future PR will require this:
    df = vaex.from_dataset(ds)
    ds_full = ds = df.dataset

    iter = ds.chunk_iterator(['x', 'y'], chunk_size=2)
    for i in range(5):
        i1, i2, chunks = next(iter)
        assert i1 == i*2
        assert i2 == (i + 1) * 2
        chunks['x'].to_pylist() == x[i1:i2].tolist()
        chunks['y'].to_pylist() == y[i1:i2].tolist()

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
        chunks['x'].to_pylist() == x[i1:i2].tolist()
        chunks['y'].to_pylist() == y[i1:i2].tolist()

    ds = ds[1:9]
    assert ds.row_count == 8
    iter = ds.chunk_iterator(['x', 'y'], chunk_size=2)
    for i in range(4):
        i1, i2, chunks = next(iter)
        assert i1 == i*2
        assert i2 == (i + 1) * 2
        chunks['x'].to_pylist() == x[i1:i2].tolist()
        chunks['y'].to_pylist() == y[i1:i2].tolist()

    # empty columns
    iter = ds.chunk_iterator([], chunk_size=2)
    for i in range(4):
        i1, i2, chunks = next(iter)
        assert i1 == i*2
        assert i2 == (i + 1) * 2

    for i in range(9):
        for j in range(i+1, 10):
            ds = ds_full.slice(i, j)
            values = []
            for i1, i2, chunks in ds.chunk_iterator(['x']):
                values.extend(chunks['x'].to_pylist())
            assert x[i:j].tolist() == values



    assert df.x.tolist() == x.tolist()
    assert df.g.tolist() == g.tolist()
    # ds.chunk_size = 4

    ds_dropped = ds.dropped('x')
    assert 'x' not in ds_dropped

    rebuild_dataset(ds).hashed() == ds.hashed()