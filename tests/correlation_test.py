import numpy as np

import vaex


def test_correlation():
    df = vaex.example()

    # A single column pair
    xy = yx = df.correlation('x', 'y')
    xy_expected = np.corrcoef(df.x.values, df.y.values)[0,1]
    np.testing.assert_array_almost_equal(xy, xy_expected, decimal=5)

    np.testing.assert_array_almost_equal(df.correlation('x', 'y'), df.correlation('y', 'x'))

    xx = df.correlation('x', 'x')
    yy = df.correlation('y', 'y')
    zz = df.correlation('z', 'z')
    
    zx = xz = df.correlation('x', 'z')
    zy = yz = df.correlation('y', 'z')


    # A list of columns
    result = df.correlation(x=['x', 'y', 'z'])
    expected3 = expected = np.array(([xx, xy, xz],
                         [yx, yy, yz],
                         [zx, zy, zz]))
    np.testing.assert_array_almost_equal(result, expected)

    # A list of columns and a single target
    desired = df.correlation(x=['x', 'y', 'z'], y='z')
    expected = np.array([xz, yz, zz])
    np.testing.assert_array_almost_equal(desired, expected)

    result = df.correlation(x=['x', 'y', 'z'], y=['y', 'z'])
    assert result.shape == (3, 2)
    expected = np.array(([xy, xz],
                         [yy, yz],
                         [zy, zz]
                         ))
    np.testing.assert_array_almost_equal(result, expected)


    result = df.correlation(x=['x', 'y', 'z'], y=['y', 'z'])

    result = df.correlation(['x', 'y'], binby='x', shape=4, limits=[-2, 2])
    result0 = df.correlation(['x', 'y'], selection=(df.x >= -2) & (df.x < -1))
    np.testing.assert_array_almost_equal(result[0], result0)


    xar = df.correlation(['x', 'y', 'z'], array_type='xarray')
    np.testing.assert_array_almost_equal(xar.data, expected3)
    assert xar.dims == ("x", "y")
    assert xar.coords['x'].data.tolist() == ['x', 'y', 'z']
    assert xar.coords['y'].data.tolist() == ['x', 'y', 'z']

    dfc = df.correlation([('x', 'y'), ('x', 'z'), ('y', 'z')])
    assert len(dfc) == 3
    assert dfc['x'].tolist() == ['x', 'x', 'y']
    assert dfc['y'].tolist() == ['y', 'z', 'z']
    np.testing.assert_array_almost_equal(dfc['correlation'].tolist(), [xy, xz, yz])
