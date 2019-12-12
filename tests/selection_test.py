from common import *


def test_selection_and_filter():
    x = np.arange(-10, 11, 1)
    y = np.arange(21)
    df = vaex.from_arrays(x=x, y=y)

    df.select(df.x < 0)
    selected_list = df.evaluate(df.x, selection=True).tolist()

    df_filtered = df[df.x < 0]
    filtered_list = df_filtered['x'].tolist()
    assert filtered_list == selected_list
    repr(df_filtered)

    # make sure we can slice, and repr
    df_sliced = df_filtered[:5]
    repr(df_sliced)


def test_filter(df):
    dff = df[df.x>4]
    assert dff.x.tolist() == list(range(5,10))

    # vaex can have filters 'grow'
    dff_bigger = dff.filter(dff.x < 3, mode="or")
    dff_bigger = dff_bigger.filter(dff_bigger.x >= 0, mode="and")  # restore old filter (df_filtered)
    assert dff_bigger.x.tolist() == list(range(3)) + list(range(5,10))


def test_filter_boolean_scalar_variable(df):
    df = df[df.x>4]
    assert df.x.tolist() == list(range(5,10))
    df.add_variable("production", True)
    df = df.filter("production", mode="or")
    df = df[df.x>=0] # restore old filter (df_filtered)
    df = df[df.x<10] # restore old filter (df_filtered)
    assert df.x.tolist() == list(range(10))


def test_selection_with_filtered_df_invalid_data():
    # Custom function to be applied to a filtered DataFrame
    def custom_func(x):
        assert 4 not in x; return x**2

    df = vaex.from_arrays(x=np.arange(10))
    df_filtered = df[df.x!=4]
    df_filtered.add_function('custom_function', custom_func)
    df_filtered['y'] = df_filtered.func.custom_function(df_filtered.x)
    # assert df_filtered.y.tolist() == [0, 1, 4, 9, 25, 36, 49, 64, 81]
    assert df_filtered.count(df_filtered.y, selection='y > 0') == 8
