import pytest
import vaex
import numpy as np
import numpy.ma
from common import small_buffer

df_a = vaex.from_arrays(a=np.array(['A', 'B', 'C']),
                        x=np.array([0., 1., 2.]),
                        y=np.ma.array([0., 9., 2.], mask=[False, True, False]),
                        m=np.ma.array([1, 2, 3], mask=[False, True, False])
                        )

df_b = vaex.from_arrays(b=np.array(['A', 'B', 'D']),
                        x=np.array([2., 1., 0.]),
                        y=np.ma.array([9., 1., 2.], mask=[True, False, False]),
                        m=np.ma.array([3, 1, 2], mask=[True, False, False])
                        )

df_dup = vaex.from_arrays(b=np.array(['A', 'B', 'A']),
                          x=np.array([2., 1., 2.]),
                          y=np.ma.array([9., 1., 9.], mask=[True, False, False]),
                          m=np.ma.array([3, 1, 2], mask=[True, True, False])
                          )

df_c = vaex.from_arrays(c=np.array(['B', 'C']),
                        z1=np.array([-1., -2.]),
                        z2=np.array([True, False]),
                        )

df_d = vaex.from_arrays(a=np.array(['B', 'C', 'D']),
                        x1=np.array(['dog', 'cat', 'mouse']),
                        x2=np.array([3.1, 25, np.nan]),
                        )

df_e = vaex.from_arrays(a=np.array(['X', 'Y', 'Z']),
                        x1=np.array(['dog', 'cat', 'mouse']),
                        x2=np.array([3.1, 25, np.nan]),
                        )


def test_no_on():
    # just adds the columns
    df = df_a.join(df_b, rsuffix='_r')
    assert df.columns['b'] is df_b.columns['b']


def test_join_masked():
    df = df_a.join(other=df_b, left_on='m', right_on='m', rsuffix='_r')
    assert df.evaluate('m').tolist() == [1, None, 3]
    assert df.evaluate('m_r').tolist() == [1, None, None]
    assert df.columns['m_r'].indices.dtype == np.int8


def test_join_nomatch():
    df = df_a.join(df_e, on=df_a.a, rprefix='r_')
    assert df.x2.tolist() == [None, None, None]


def test_left_a_b():
    df = df_a.join(other=df_b, left_on='a', right_on='b', rsuffix='_r')
    assert df['a'].tolist() == ['A', 'B', 'C']
    assert df['b'].tolist() == ['A', 'B', None]
    assert df['x'].tolist() == [0, 1, 2]
    assert df['x_r'].tolist() == [2, 1, None]
    assert df['y'].tolist() == [0, None, 2]
    assert df['y_r'].tolist() == [None, 1, None]

def test_left_a_b_as_alias():
    df_ac = df_a.copy()
    df_bc = df_b.copy()
    df_ac['1'] = df_ac['a']
    df_bc['2'] = df_bc['b']
    df = df_ac.join(other=df_bc, left_on='1', right_on='2', rsuffix='_r')
    assert df.evaluate('a').tolist() == ['A', 'B', 'C']
    assert df.evaluate('b').tolist() == ['A', 'B', None]
    assert df.evaluate('x').tolist() == [0, 1, 2]
    assert df.evaluate('x_r').tolist() == [2, 1, None]
    assert df.evaluate('y').tolist() == [0, None, 2]
    assert df.evaluate('y_r').tolist() == [None, 1, None]


def test_join_indexed():
    df = df_a.join(other=df_b, left_on='a', right_on='b', rsuffix='_r')
    df_X = df_a.join(df, left_on='a', right_on='b', rsuffix='_r')
    assert df_X['b'].tolist() == ['A', 'B', None]


def test_left_a_b_filtered():
    df_af = df_a[df_a.x > 0]
    df = df_af.join(other=df_b, left_on='a', right_on='b', rsuffix='_r')
    assert df['a'].tolist() == ['B', 'C']
    assert df['b'].tolist() == ['B', None]
    assert df['x'].tolist() == [1, 2]
    assert df['x_r'].tolist() == [1, None]
    assert df['y'].tolist() == [None, 2]
    assert df['y_r'].tolist() == [1, None]

    # actually, even though the filter is applied, all rows will be matched
    # since the filter can change
    df.set_selection(None, vaex.dataset.FILTER_SELECTION_NAME)
    assert df['a'].tolist() == ['A', 'B', 'C']
    assert df['b'].tolist() == ['A', 'B', None]
    assert df['x'].tolist() == [0, 1, 2]
    assert df['x_r'].tolist() == [2, 1, None]
    assert df['y'].tolist() == [0, None, 2]
    assert df['y_r'].tolist() == [None, 1, None]

    # if we extract, that shouldn't be the case
    df_af = df_a[df_a.x > 0].extract()
    df = df_af.join(other=df_b, left_on='a', right_on='b', rsuffix='_r')
    df.set_selection(None, vaex.dataset.FILTER_SELECTION_NAME)
    assert df['a'].tolist() == ['B', 'C']
    assert df['b'].tolist() == ['B', None]
    assert df['x'].tolist() == [1, 2]
    assert df['x_r'].tolist() == [1, None]
    assert df['y'].tolist() == [None, 2]
    assert df['y_r'].tolist() == [1, None]


def test_inner_a_b_filtered():
    df_a_filtered = df_a[df_a.x > 0]
    df = df_a_filtered.join(other=df_b, left_on='a', right_on='b', rsuffix='_r', how='inner')
    assert df['a'].tolist() == ['B']
    assert df['b'].tolist() == ['B']
    assert df['x'].tolist() == [1]
    assert df['x_r'].tolist() == [1]
    assert df['y'].tolist() == [None]
    assert df['y_r'].tolist() == [1]


def test_left_a_b_filtered_right():
    # similar to test_left_a_b_filtered, but now the df we join is filtered
    # take b without the last tow
    df_bf = df_b[df_b.b.str.contains('A|B')]
    df = df_a.join(df_bf, how='left', on='x', rsuffix='_r')
    # columns of the left df
    assert df.x.tolist() == [0, 1, 2]
    assert df.a.tolist() == ['A', 'B', 'C']
    assert df.y.tolist() == [0, None, 2]
    assert df.m.tolist() == [1, None, 3]
    # columns of the right df
    assert df.b.tolist() == [None, 'B', 'A']
    assert df.x_r.tolist() == [None, 1, 2]
    assert df.y_r.tolist() == [None, 1, None]
    assert df.m_r.tolist() == [None, 1, None]


def test_right_x_x():
    df = df_a.join(other=df_b, on='x', rsuffix='_r', how='right')
    assert df['a'].tolist() == ['C', 'B', 'A']
    assert df['b'].tolist() == ['A', 'B', 'D']
    assert df['x'].tolist() == [2, 1, 0]
    assert df['x_r'].tolist() == [2, 1, 0]
    assert df['y'].tolist() == [2, None, 0]
    assert df['y_r'].tolist() == [None, 1, 2]
    assert 'y_r' not in df_b


def test_left_dup():
    df = df_a.join(df_dup, left_on='a', right_on='b', rsuffix='_r', allow_duplication=True)
    assert len(df) == 4
    # df = df_a.join(df_dup, on='x', rsuffix='_r')
    # df = df_a.join(df_dup, on='m', rsuffix='_r')


def test_left_a_c():
    df = df_a.join(df_c, left_on='a', right_on='c', how='left')
    assert df.a.tolist() == ['A', 'B', 'C']
    assert df.x.tolist() == [0, 1, 2]
    assert df.y.tolist() == [0., None, 2.]
    assert df.m.tolist() == [1, None, 3]
    assert df.c.tolist() == [None, 'B', 'C']
    assert df.z1.tolist() == [None, -1., -2.]
    assert df.z2.tolist() == [None, True, False]


def test_join_a_a_suffix_check():
    df = df_a.join(df_a, on='a', lsuffix='_left', rsuffix='_right')
    assert set(df.column_names) == {'a_left', 'x_left', 'y_left', 'm_left', 'a_right', 'x_right', 'y_right', 'm_right'}


def test_join_a_a_prefix_check():
    df = df_a.join(df_a, on='a', lprefix='left_', rprefix='right_')
    assert set(df.column_names) == {'left_a', 'left_x', 'left_y', 'left_m', 'right_a', 'right_x', 'right_y', 'right_m'}


def test_inner_a_d():
    df = df_a.join(df_d, on='a', right_on='a', how='inner', rsuffix='_r')
    assert df.a.tolist() == ['B', 'C']
    assert df.x.tolist() == [1., 2.]
    assert df.y.tolist() == [None, 2.]
    assert df.m.tolist() == [None, 3.]
    assert df.x1.tolist() == ['dog', 'cat']
    assert df.x2.tolist() == [3.1, 25.]


@pytest.mark.skip(reason='full join not supported yet')
def test_full_a_d():
    df = df_a.join(df_d, on='a', right_on='a', how='full')
    assert df.a.tolist() == ['A', 'B', 'C', 'D']
    assert df.x.tolist() == [0., 1., 2., None]
    assert df.y.tolist() == [0., None, 2., None]
    assert df.m.tolist() == [1, None, 3, None]
    assert df.x1.tolist() == [None, 'dog', 'cat', 'mouse']
    assert df.x2.tolist() == [None, 3.1, 25., np.nan]
    np.testing.assert_array_equal(np.array(df_d.x2.values), np.array([3.1, 25., np.nan]))


def test_left_virtual_filter():
    df = df_a.join(df_d, on='a', how='left', rsuffix='_b')
    df['r'] = df.x + df.x2
    df = df[df.r > 10]
    assert set(df[0]) == {'C', 2.0, 2.0, 3, 'C', 'cat', 25.0, 27.0}


def test_left_on_virtual_col():
    mapper = {0: 'A', 1: 'B', 2: 'C'}
    df_a['aa'] = df_a.x.map(mapper=mapper)
    df = df_a.join(df_d, left_on='aa', right_on='a', rsuffix='_right')
    assert df.a.tolist() == ['A', 'B', 'C']
    assert df.aa.tolist() == ['A', 'B', 'C']
    assert df.x.tolist() == [0, 1, 2]
    assert df.y.tolist() == [0., None, 2.]
    assert df.m.tolist() == [1, None, 3]
    assert df.x1.tolist() == [None, 'dog', 'cat']
    assert df.x2.tolist() == [None, 3.1, 25.]
    assert df.a_right.tolist() == [None, 'B', 'C']


def test_join_filtered_inner():
    df_a_filtered = df_a[df_a.y > 0]
    df_joined = df_a_filtered.join(other=df_b, on='x', how='inner', rsuffix='_', allow_duplication=True)
    assert len(df_joined) == len(df_a_filtered)

    x = np.arange(20)
    df = vaex.from_arrays(x=x, y=x**2)
    df = df[df.x > 5]
    dfj = df.join(df, on='x', rsuffix='right_', how='inner')
    repr(dfj)  # trigger issue with selection cache


def test_join_duplicate_column():
    df_left = vaex.from_arrays(index=[1, 2, 3], x=[10, 20, 30])
    df_right = vaex.from_arrays(index=[1, 2, 3], y=[0.1, 0.2, 0.3])

    df = df_left.join(df_right, on='index')
    assert df.column_count() == 3
    assert set(df.column_names) == {'index', 'x', 'y'}
    assert df['index'].tolist() == [1, 2, 3]
    assert df.x.tolist() == [10, 20, 30]
    assert df.y.tolist() == [0.1, 0.2, 0.3]


# we join row based and on a column
@pytest.mark.parametrize("on", [None, 'j'])
def test_join_virtual_columns(on):
    df1 = vaex.from_scalars(j=444, x=1, y=2)
    df1['z'] = df1.x + df1.y
    df1['__h'] = df1.z * 2
    df2 = vaex.from_scalars(j=444, x=2, yy=3)
    df2['z'] = df2.x + df2.yy
    df2['__h'] = df2.z * 3
    df = df1.join(df2, rprefix='r_', rsuffix='_rhs', on=on)
    assert df.x.values[0] == 1
    assert df.y.values[0] == 2
    assert df.z.values[0] == 3
    assert df.__h.values[0] == 6
    assert df.r_x_rhs.values[0] == 2
    assert df.yy.values[0] == 3
    assert df.r_z_rhs.values[0] == 5
    assert df.__r_h_rhs.values[0] == 15


def test_join_variables():
    df1 = vaex.from_scalars(j=444, x=1, y=2)
    df1.add_variable('a', 2)
    df1.add_variable('b', 3)
    df1['z'] = df1.x * df1['a'] + df1.y * df1['b']

    df2 = vaex.from_scalars(j=444, x=2, yy=3)
    df2.add_variable('a', 3)
    df2.add_variable('b', 4)
    df2['z'] = df2.x * df2['a'] + df2.yy * df2['b']
    df = df1.join(df2, rprefix='r_', rsuffix='_rhs')
    assert df.x.values[0] == 1
    assert df.y.values[0] == 2
    assert df.z.values[0] == 2 + 2*3
    # assert df.__h.values[0] == 6
    assert df.r_x_rhs.values[0] == 2
    assert df.yy.values[0] == 3
    assert df.r_z_rhs.values[0] == 2*3 + 3*4


def test_with_masked_no_short_circuit():
    # this test that the full table is joined, in some rare condition
    # it can happen that the left table has a value not present in the right
    # which causes it to not evaluate the other lookups, due to Python's short circuit
    # behaviour. E.g. True or func() will not call func
    N = 1000
    df = vaex.from_arrays(i=np.arange(100) % 10)
    df_right = vaex.from_arrays(i=np.arange(9), j=np.arange(9))
    with small_buffer(df, size=1):
        dfj = df.join(other=df_right, on='i')
    assert dfj.columns['j'].masked
    assert dfj[:10].columns['j'].masked
    assert dfj['j'][:10].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, None]
    dfj['j'].tolist()  # make sure we can evaluate the whole column
