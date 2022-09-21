from pathlib import Path
import pytest

import pyarrow.csv
import numpy as np

from vaex.csv import _row_count
import vaex.csv

HERE = Path(__file__).parent


def test_row_count_basics():
    assert vaex.csv._row_count(b'a, 1\nb,3\n') == 2
    assert vaex.csv._row_count(b'a, 1\nb,3') == 2


@pytest.mark.skipif(vaex.utils.osname == 'windows',
                    reason="windows lazy not supported due to line ending differences")
def test_row_count_csv():
    import vaex.csv
    ds = vaex.csv.DatasetCsvLazy(HERE / 'data' / 'small3.csv', chunk_size=8, newline_readahead=4)
    assert ds.row_count == 3

    # chunks = list(ds.chunk_iterator(['x']))
    # assert chunks[0][2]['x'].tolist() == [5, 7, 9]


# def test_large_csv_count_rows():
#     csv_path = '/data/tmp/chicago-taxi.csv'
#     ds = vaex.csv.DatasetCsv(csv_path)
#     assert ds.row_count == 193938982


# def test_large_csv_count_array_lengths():
#     csv_path = '/data/tmp/chicago-taxi.csv'
#     ds = vaex.csv.DatasetCsv(csv_path, row_count=193938982)
#     total = 0
#     for i1, i2, chunks in ds.chunk_iterator(['Taxi ID']):
#         total += i2 - i1
#         assert len(chunks['Taxi ID']) == i2 - i1
#     assert total == ds.row_count


@pytest.mark.parametrize("l1", list(range(1, 3)))
@pytest.mark.parametrize("l2", list(range(1, 3)))
def test_chunk_iterator(l1, l2):
    i1 = 0
    i2 = i1 + l1
    i3 = i2 + l2
    x = np.arange(10)
    y = x**2
    g = x // 3
    ds = vaex.dataset.DatasetArrays(x=x, y=y, g=g)
    df_original = df = vaex.from_dataset(ds)
    path = HERE / 'data' / 'csv' / 'test.csv'
    path.parent.mkdir(exist_ok=True)
    df_original.export(str(path))

    df = vaex.from_csv_arrow(str(path), chunk_size=20, newline_readahead=10, lazy=True)
    ds_full = ds = df.dataset

    # very similar to the dataset_test::test_concat_chunk_iterator and arrow/datase_test.py parquet test
    iter = ds.chunk_iterator(['x', 'y'], chunk_size=2)
    for i in range(5):
        i1, i2, chunks = next(iter)
        assert i1 == i*2
        assert i2 == (i + 1) * 2
        vaex.array_types.tolist(chunks['x']) == x[i1:i2].tolist()
        vaex.array_types.tolist(chunks['y']) == y[i1:i2].tolist()


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
        vaex.array_types.tolist(chunks['x']) == x[i1:i2].tolist()
        vaex.array_types.tolist(chunks['y']) == y[i1:i2].tolist()

    ds = ds[1:9]
    assert ds.row_count == 8
    iter = ds.chunk_iterator(['x', 'y'], chunk_size=2)
    for i in range(4):
        i1, i2, chunks = next(iter)
        assert i1 == i*2
        assert i2 == (i + 1) * 2
        vaex.array_types.tolist(chunks['x']) == x[i1:i2].tolist()
        vaex.array_types.tolist(chunks['y']) == y[i1:i2].tolist()

    # no columns
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
                values.extend(vaex.array_types.tolist(chunks['x']))
            assert x[i:j].tolist() == values

    assert df.x.tolist() == x.tolist()
    assert df.g.tolist() == g.tolist()

    ds_dropped = ds.dropped('x')
    assert 'x' not in ds_dropped


def test_gz():
    import vaex

    df = vaex.from_csv_arrow(HERE / "data" / "small2.csv.gz")
    assert df.x.tolist() == [1, 3]

    df = vaex.from_csv(HERE / "data" / "small2.csv.gz")
    assert df.x.tolist() == [1, 3]


    with pytest.raises(NotImplementedError):
        df = vaex.from_csv_arrow(HERE / "data" / "small2.csv.gz", lazy=True)
        assert df.x.tolist() == [1, 3]
