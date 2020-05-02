import numpy as np

def tolist(x):
    return np.array(x).tolist()


def test_limits(df):
    xmin, xmax = df.limits('x', 'minmax')
    xmin_half, xmax_half = df.limits('x', 'minmax', selection=df.x < 5)
    assert xmin == xmin_half
    assert xmax > xmax_half

    assert df.limits('x', 'minmax').tolist() == df.minmax('x').tolist()
    assert df.limits('x', '99%').tolist() == df.limits_percentage('x', 99).tolist()
    assert tolist(df.limits(['x', 'y'], 'minmax')) == tolist(df.minmax(['x', 'y']))
    assert tolist(df.limits(['x', 'y'], ['minmax', 'minmax'])) == tolist(df.minmax(['x', 'y']))

    assert df.limits('x', [0, 10]) == [0, 10]

    assert tolist(df.limits('x', '90%')) == tolist(df.limits_percentage('x', 90.))
    assert tolist(df.limits([['x', 'y'], ['x', 'z']], 'minmax')) ==\
        tolist([df.minmax(['x', 'y']), df.minmax(['x', 'z'])])
    assert tolist(df.limits([['x', 'y'], ['x', 'z'], ['y', 'z']], 'minmax', shape=(10, 10))[0]) ==\
        tolist([df.minmax(['x', 'y']), df.minmax(['x', 'z']), df.minmax(['y', 'z'])])
    assert tolist(df.limits([['x', 'y'], ['x', 'z']], [[[0, 10], [0, 20]], 'minmax'])) ==\
        tolist([[[0, 10], [0, 20]], df.minmax(['x', 'z'])])

    assert df.limits(['x', 'y'], 'minmax', shape=10)[1] == [10, 10]
    assert df.limits(['x', 'y'], 'minmax', shape=(10, 12))[1] == [10, 12]


def test_limits_with_selection(df):
    limits_selection_perc = df.limits('x', value='90%', selection='x > 5')

    df_sliced = df[df.x > 5]
    limits_sliced = df_sliced.limits('x', value='90%')

    assert limits_sliced.tolist() == limits_selection_perc.tolist()
