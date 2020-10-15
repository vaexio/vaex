from pathlib import Path

import pyarrow.parquet
import numpy as np
import vaex


HERE = Path(__file__).parent


def test_parquet():
    x = np.arange(10)
    y = x**2
    g = x // 3
    ds = vaex.dataset.DatasetArrays(x=x, y=y, g=g)
    df = vaex.from_dataset(ds)
    path = HERE.parent / 'data' / 'parquet' / 'test.parquet'
    path.parent.mkdir(exist_ok=True)
    # df.export(str(path))
    pyarrow.parquet.write_table(df.to_arrow_table(), str(path), row_group_size=2)

    df = vaex.open(str(path))
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
