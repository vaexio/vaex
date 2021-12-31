import pyarrow as pa
import vaex.groupby
import numpy as np



def test_combined_grouper():
    x = pa.array([1, 1, 2, 2])
    y = pa.array([1, 1, 3, 4])
    df = vaex.from_arrays(x=x, y=y)
    grouper_x = vaex.groupby.Grouper(df.x)
    grouper_y = vaex.groupby.Grouper(df.y)
    groupers = [grouper_x, grouper_y]
    df.execute()
    grouper_x._create_binner(df)
    grouper_y._create_binner(df)
    assert set(df[grouper_x.binby_expression].tolist()) == {0, 1}
    assert set(df[grouper_y.binby_expression].tolist()) == {0, 1, 2}
    grouper = vaex.groupby._combine(df, groupers, sort=True)
    df.execute()
    grouper = grouper.get()
    grouper._create_binner(df)
    assert set(df[grouper.binby_expression].tolist()) == {0, 1, 2}


def test_combined_grouper_over64bit():
    bits = [15, 16, 17] * 2
    assert sum(bits) > 64
    N = 2**max(bits)
    def unique_ints(offset, bit):
        # create 2**bits unique ints
        ar = np.full(N, offset, dtype='int32')
        n = 2**bit
        ar[:n] = np.arange(offset, offset + n)
        return ar
    arrays = {f'x_{i}': unique_ints(i, bit) for i, bit in enumerate(bits)}
    df = vaex.from_dict(arrays)
    groupers = [vaex.groupby.Grouper(df[name]) for name in arrays]
    df.execute()
    for grouper in groupers:
        grouper._create_binner(df)
    grouper = vaex.groupby._combine(df, groupers, sort=True)
    df.execute()
    grouper = grouper.get()
    grouper._create_binner(df)
    assert df[grouper.binby_expression].nunique() == N


def test_combined_grouper_cast():
    x = pa.array(range(127))
    y = pa.array(range(1, 128))
    df = vaex.from_arrays(x=x, y=y)
    grouper_x = vaex.groupby.Grouper(df.x)
    grouper_y = vaex.groupby.Grouper(df.y)
    groupers = [grouper_x, grouper_y]
    df.execute()
    grouper_x._create_binner(df)
    grouper_y._create_binner(df)
    assert df[grouper_x.binby_expression].dtype.numpy.name == 'int8'
    assert df[grouper_y.binby_expression].dtype.numpy.name == 'int8'
    grouper = vaex.groupby._combine(df, groupers, sort=True)
    df.execute()
    grouper = grouper.get()
    grouper._create_binner(df)
    assert set(df[grouper.binby_expression].tolist()) == set(range(127))
