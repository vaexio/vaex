import vaex
import numpy as np

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


def test_evaluate_types():
    x = np.arange(10)
    y = pa.array(x**2)
    df = vaex.from_arrays(x=x, y=y)
    assert isinstance(df.columns['x'], np.ndarray)
    assert isinstance(df.columns['y'], pa.Array)

    assert df.evaluate("x", type=None) is x
    assert isinstance(df.evaluate("x", type="numpy"), np.ndarray)
    assert isinstance(df.evaluate("x", type="arrow"), pa.Array)

    assert df.evaluate("y", type=None) is y
    assert isinstance(df.evaluate("y", type=None), pa.Array) # WIP
    assert isinstance(df.evaluate("y", type="arrow"), pa.Array)
    assert isinstance(df.evaluate("y", type="numpy"), np.ndarray)
