from common import *



def test_selection_basics(df):
    total = df["x"].sum()
    df.select("x > 5")
    df.select("x <= 5", name="inverse")

    counts = df.count("x", selection=["default", "inverse", "x > 5", "default | inverse"])
    np.testing.assert_array_almost_equal(counts, [4, 6, 4, 10])


    df.select("x <= 1", name="inverse", mode="subtract")
    counts = df.count("x", selection=["default", "inverse"])
    np.testing.assert_array_almost_equal(counts, [4, 4])

    total_subset = df["x"].sum(selection=True)
    assert total_subset < total
    for mode in vaex.selections._select_functions.keys():
        df.select("x > 5")
        df.select("x > 5", mode)
        df.select(None)
        df.select("x > 5", mode)


    df.select("x > 5")
    total_subset = df["x"].sum(selection=True)
    df.select_inverse()
    total_subset_inverse = df["x"].sum(selection=True)
    df.select("x <= 5")
    total_subset_inverse_compare = df["x"].sum(selection=True)
    assert total_subset_inverse == total_subset_inverse_compare
    assert total_subset_inverse + total_subset == total


    df.select("x > 5")
    df.select("x <= 5", name="inverse")
    df.select_inverse(name="inverse")
    counts = df.count("x", selection=["default", "inverse"])
    np.testing.assert_array_almost_equal(counts, [4, 4])



def test_selection_history(df):
    assert not df.has_selection()
    assert not df.selection_can_undo()
    assert not df.selection_can_redo()

    df.select_nothing()
    assert not df.has_selection()
    assert not df.selection_can_undo()
    assert not df.selection_can_redo()


    total = df["x"].sum()
    assert not df.has_selection()
    assert not df.selection_can_undo()
    assert not df.selection_can_redo()
    df.select("x > 5")
    assert df.has_selection()
    total_subset = df["x"].sum(selection=True)
    assert total_subset < total
    assert df.selection_can_undo()
    assert not df.selection_can_redo()

    df.select("x < 7", mode="and")
    total_subset2 = df["x"].sum(selection=True)
    assert total_subset2 < total_subset
    assert df.selection_can_undo()
    assert not df.selection_can_redo()

    df.selection_undo()
    total_subset_same = df["x"].sum(selection=True)
    total_subset == total_subset_same
    assert df.selection_can_undo()
    assert df.selection_can_redo()

    df.selection_redo()
    total_subset2_same = df["x"].sum(selection=True)
    total_subset2 ==  total_subset2_same
    assert df.selection_can_undo()
    assert not df.selection_can_redo()

    df.selection_undo()
    df.selection_undo()
    assert not df.has_selection()
    assert not df.selection_can_undo()
    assert df.selection_can_redo()

    df.selection_redo()
    assert df.has_selection()
    assert df.selection_can_undo()
    assert df.selection_can_redo()
    df.select("x < 7", mode="and")
    assert df.selection_can_undo()
    assert not df.selection_can_redo()

    df.select_nothing()
    assert not df.has_selection()
    assert df.selection_can_undo()
    assert not df.selection_can_redo()
    df.selection_undo()
    assert df.selection_can_undo()
    assert df.selection_can_redo()



def test_selection_serialize(df):
    selection_expression = vaex.selections.SelectionExpression("x > 5", None, "and")
    df.set_selection(selection_expression)
    total_subset = df["x"].sum(selection=True)

    df.select("x > 5")
    total_subset_same = df["x"].sum(selection=True)
    assert total_subset == total_subset_same

    values = selection_expression.to_dict()
    df.set_selection(vaex.selections.selection_from_dict(values))
    total_subset_same2 = df["x"].sum(selection=True)
    assert total_subset == total_subset_same2

    selection_expression = vaex.selections.SelectionExpression("x > 5", None, "and")
    selection_lasso = vaex.selections.SelectionLasso("x", "y", [0, 10, 10, 0], [-1, -1, 100, 100], selection_expression, "and")
    df.set_selection(selection_lasso)
    total_2 = df.sum("x", selection=True)
    assert total_2 == total_subset


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


def test_lasso(df):
    x = [-0.1, 5.1, 5.1, -0.1]
    y = [-0.1, -0.1, 4.1, 4.1]
    df.select_lasso("x", "y", x, y)
    sumx, sumy = df.sum(["x", "y"], selection=True)
    np.testing.assert_array_almost_equal(sumx, 0+1+2)
    np.testing.assert_array_almost_equal(sumy, 0+1+4)

    # now test with masked arrays, m ~= x
    x = [8-0.1, 9+0.1, 9+0.1, 8-0.1]
    y = [-0.1, -0.1, 1000, 1000]
    if df.is_local():
        df._invalidate_selection_cache()
    df.select_lasso("m", "y", x, y)
    sumx, sumy = df.sum(['m', 'y'], selection=True)
    np.testing.assert_array_almost_equal(sumx, 8)
    np.testing.assert_array_almost_equal(sumy, 8**2)


def test_selection_event_calls(df):
    counts = 0
    @df.signal_selection_changed.connect
    def update(df, name):
        nonlocal counts
        counts += 1
    df.select(df.x > 3, name='bla')
    assert counts == 1
