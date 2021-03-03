import numpy as np

import vaex


def test_correlation():
    df = vaex.example()

    # A single column pair
    xy = yx = df.corr('x', 'y')
    xy_expected = np.corrcoef(df.x.values, df.y.values)[0,1]
    np.testing.assert_array_almost_equal(xy, xy_expected, decimal=5)

    np.testing.assert_array_almost_equal(df.corr('x', 'y'), df.corr('y', 'x'))

    xx = df.corr('x', 'x')
    yy = df.corr('y', 'y')
    zz = df.corr('z', 'z')
    
    zx = xz = df.corr('x', 'z')
    zy = yz = df.corr('y', 'z')


    # A list of columns
    result = df.corr(x=['x', 'y', 'z'])
    expected = np.array(([xx, xy, xz],
                         [yx, yy, yz],
                         [zx, zy, zz]))
    np.testing.assert_array_almost_equal(result, expected)

    # A list of columns and a single target
    desired = df.corr(x=['x', 'y', 'z'], y='z')
    expected = np.array([xz, yz, zz])
    np.testing.assert_array_almost_equal(desired, expected)

    result = df.corr(x=['x', 'y', 'z'], y=['y', 'z'])
    assert result.shape == (3, 2)
    expected = np.array(([xy, xz],
                         [yy, yz],
                         [zy, zz]
                         ))
    np.testing.assert_array_almost_equal(result, expected)


    result = df.corr(x=['x', 'y', 'z'], y=['y', 'z'])

    result = df.corr(['x', 'y'], binby='x', shape=4, limits=[-2, 2])
    result0 = df.corr(['x', 'y'], selection=(df.x >= -2) & (df.x < -1))
    np.testing.assert_array_almost_equal(result[0], result0)
