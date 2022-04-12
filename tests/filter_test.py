import numpy as np
import vaex


def test_set_active_range_and_trim(df_factory):
    df = df_factory(x=np.arange(8))
    df = df[(df.x % 2) == 0]  # even numbers
    assert len(df) == 4
    df.set_active_range(2, 6)  # these are unfiltered
    assert df._cached_filtered_length == 2
    assert df._selection_masks[vaex.dataframe.FILTER_SELECTION_NAME].count() == 4  # still the original mask (untrimmed)
    dft = df.trim()
    assert dft._cached_filtered_length == 2
    assert dft._selection_masks[vaex.dataframe.FILTER_SELECTION_NAME].count() == 2
    assert dft.x.tolist() == [2, 4]


def test_filter_cache():
    called = 0

    def odd(x):
        nonlocal called
        called += 1
        return (x % 2) == 1

    x = np.arange(10)
    df = vaex.from_arrays(x=x)
    df.add_function("odd", odd)
    dff = df[df.func.odd("x")]
    len(dff)
    assert called == 1
    df_sliced1 = dff[1:2]
    df_sliced2 = dff[2:4]
    assert called == 1
    repr(dff)
    assert called == 1

    len(df_sliced1)
    len(df_sliced2)
    assert called == 1
    df_sliced3 = df_sliced2[1:2]
    assert called == 1
    len(df_sliced3)
    assert called == 1


def test_filter_by_boolean_column():
    df = vaex.from_scalars(x=1, ok=True)
    dff = df[df.ok]
    assert dff[["x"]].x.tolist() == [1]


# def test_slice_no_compute(df_factory):
#     df = df_factory(x=np.arange(8))
#     df = df[(df.x % 2) == 0] # even numbers
#     # len(df)  # trigger cache
#     df.count()
#     # assert df.x.sum() == 0+2+4+6
#     dfs = df[1:3]
#     assert dfs._cached_filtered_length == 2
#     assert dfs.x.sum() == 2+4

#     dfs = df[1:2]
#     assert dfs._cached_filtered_length == 1
#     assert dfs.x.sum() == 2
