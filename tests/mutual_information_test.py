import numpy as np

import vaex


def test_mutual_information():
    df = vaex.example()

    # A single pair
    xy = yx = df.mutual_information('x', 'y')
    expected = np.array(0.068934)
    np.testing.assert_array_almost_equal(xy, expected)

    np.testing.assert_array_almost_equal(df.mutual_information('y', 'x'), df.mutual_information('x', 'y'))

    xx = df.mutual_information('x', 'x')
    yy = df.mutual_information('y', 'y')
    zz = df.mutual_information('z', 'z')

    zx = xz = df.mutual_information('x', 'z')
    zy = yz = df.mutual_information('y', 'z')

    # A list of columns
    result = df.mutual_information(x=['x', 'y', 'z'])
    expected = np.array(([xx, xy, xz],
                         [yx, yy, yz],
                         [zx, zy, zz]))
    np.testing.assert_array_almost_equal(result, expected)

    # A list of columns and a single target
    result = df.mutual_information(x=['x', 'y', 'z'], y='z')
    expected = np.array([xz, yz, zz])
    np.testing.assert_array_almost_equal(result, expected)

    # A list of columns and targets
    result = df.mutual_information(x=['x', 'y', 'z'], y=['y', 'z'])
    assert result.shape == (3, 2)
    expected = np.array(([xy, xz],
                         [yy, yz],
                         [zy, zz]
                         ))
    np.testing.assert_array_almost_equal(result, expected)

    # a list of custom pairs
    result = df.mutual_information(x=[['x', 'y'], ['x', 'z'], ['y', 'z']])
    assert result.shape == (3,)
    expected = np.array([xy, xz, yz])
    np.testing.assert_array_almost_equal(result, expected)


    result = df.mutual_information(x=['x', 'y'], dimension=3, mi_shape=4)
    assert result.shape == (2, 2, 2)
