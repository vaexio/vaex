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
