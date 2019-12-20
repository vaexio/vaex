import vaex
import numpy as np

def test_chunking():
    x = np.arange(10)
    y = x**2
    df = vaex.from_arrays(x=x, y=y)
    def indices(df, chunk_size):
        logical_length = len(df)
        if df.filtered:
            full_mask = df._selection_masks[vaex.dataframe.FILTER_SELECTION_NAME]
            for l1, l2, i1, i2 in vaex.utils.subdivide_mask(full_mask, max_length=chunk_size, logical_length=logical_length):
                yield i1
            yield i2
        else:
            for i1, i2 in vaex.utils.subdivide(logical_length, max_length=chunk_size):
                yield i1
            yield i2

    assert list(indices(df, 2)) == [0, 2, 4, 6, 8, 10]
    assert list(indices(df, 3)) == [0, 3, 6, 9, 10]
    assert list(indices(df, 4)) == [0, 4, 8, 10]
    assert list(indices(df, 5)) == [0, 5, 10]
    assert list(indices(df, 6)) == [0, 6, 10]
    assert list(indices(df, 7)) == [0, 7, 10]
    assert list(indices(df, 12)) == [0, 10]
    dff = df[(df.x != 0) & (df.x != 5)  & (df.x != 8) ]
    assert list(indices(dff, 2)) == [1, 3, 5, 8, 10]
    assert list(indices(dff, 3)) == [1, 4, 8, 10]
    assert list(indices(dff, 4)) == [1, 5, 10]
    full_mask = np.array(dff._selection_masks[vaex.dataframe.FILTER_SELECTION_NAME], np.uint8)
    for n in [2, 3, 4]:
        total = 0
        for l1, l2, i1, i2 in dff._unfiltered_chunk_slices(n):
            part = full_mask[i1:i2]
            total += part.sum()
            assert part.sum() <= n
        assert total == 7