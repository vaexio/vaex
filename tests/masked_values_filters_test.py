import vaex
import numpy as np


x = np.ma.MaskedArray(data=[0, 1, 2, 3, 4], mask=[False, False, True, False, True])
y = np.ma.MaskedArray(data=[3, 5, 2, -1.5, 0], mask=[False, False, False, False, False])
w = np.ma.MaskedArray(data=['dog', 'dog', 'cat', 'cat', 'mouse'], mask=[False, False, True, False, True])
df = vaex.from_arrays(x=x, y=y, w=w)


def test_masked_values_selections():
    # Select on a column which has masked values
    assert df.y.count(selection='x < 3') == 2
    assert df.y.sum(selection='x < 3') == 8.
    assert df.y.mean(selection=df.x < 3) == 4.
    assert df.y.std(selection=df.x < 3) == 1.
    assert df.w.nunique(selection='x < 3') == 1.


def test_masked_values_numerical_filter():
    df_num_filter = df[df.x >= 1]
    assert len(df_num_filter) == 2
    assert df_num_filter.w.tolist() == ['dog', 'cat']
    assert df_num_filter.y.tolist() == [5., -1.5]
    assert df_num_filter.x.tolist() == [1., 3]


def test_masked_values_string_filter():
    df_str_filter = df[df.w == 'cat']
    assert len(df_str_filter) == 1
    assert df_str_filter.w.tolist() == ['cat']
    assert df_str_filter.y.tolist() == [-1.5]
    assert df_str_filter.x.tolist() == [3]


def test_masked_values_filter_and_selection():
    # assert df.evaluate_selection_mask(df.w == 'cat').tolist() == [False, False, False, True, False]
    df_filter = df[df.x < 4]
    # assert df_filter.evaluate_selection_mask(df_filter.w == 'cat').tolist() == [False, False, True]
    assert df_filter.y.count(selection="w == 'cat'") == df_filter.y.count(selection=df_filter.w == 'cat')
    assert df_filter.y.count(selection=df_filter.w == 'cat') == 1.
    assert df_filter.y.sum(selection=df_filter.w == 'cat') == -1.5
    assert df_filter.y.mean(selection=df_filter.w == 'cat') == -1.5
    assert df_filter.y.nunique(selection=df_filter.w == 'cat') == 1
