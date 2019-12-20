from common import *


@pytest.mark.parametrize("chunk_size", list(range(1, 13)))
@pytest.mark.parametrize("prefetch", [True, False])
@pytest.mark.parametrize("parallel", [True, False])
def test_evaluate_iterator(df_local, chunk_size, prefetch, parallel):
    df = df_local
    x = df.x.to_numpy()
    total = 0
    for i1, i2, chunk in df_local.evaluate_iterator('x', chunk_size=chunk_size, prefetch=prefetch, parallel=parallel):
        assert x[i1:i2].tolist() == chunk.tolist()
        total += chunk.sum()
    assert total == x.sum()


def test_evaluate_function_filtered_df():
    # Custom function to be applied to a filtered DataFrame
    def custom_func(x):
        assert 4 not in x; return x**2

    df = vaex.from_arrays(x=np.arange(10))
    df_filtered = df[df.x!=4]
    df_filtered.add_function('custom_function', custom_func)
    df_filtered['y'] = df_filtered.func.custom_function(df_filtered.x)
    assert df_filtered.y.tolist() == [0, 1, 4, 9, 25, 36, 49, 64, 81]

    # sliced exactly at the start of where we are going to filter
    # this used to trigger a bug in df.dtype, which would evaluate the first row
    df_sliced = df[4:]
    df_filtered = df_sliced[df_sliced.x!=4]
    df_filtered.add_function('custom_function', custom_func)
    df_filtered['y'] = df_filtered.func.custom_function(df_filtered.x)
    assert df_filtered.y.tolist() == [25, 36, 49, 64, 81]
