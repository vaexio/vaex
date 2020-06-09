import numpy as np

import vaex


def test_mutual_information():
    df = vaex.example()

    # A single pair
    desired = df.mutual_information('x', 'y')
    expected = np.array(0.15118145)
    np.testing.assert_array_almost_equal(desired, expected)

    # A list of columns
    desired = df.mutual_information(x=['x', 'y', 'z'])
    expected = np.array(([4.80278827, 0.15118145, 0.18439181],
                         [0.15118145, 4.87766263, 0.21418761],
                         [0.18439181, 0.21418761, 4.61453743]))
    np.testing.assert_array_almost_equal(desired, expected)

    # A list of columns and a single target
    desired = df.mutual_information(x=['vx', 'vy', 'vz'], y='vx')
    expected = np.array([0.21501621, 0.2370898, 0.24951082])
    np.testing.assert_array_almost_equal(desired, expected)

    # A list of columns and targets
    desired = df.mutual_information(x=['vx', 'vy', 'vz'], y=['Lz', 'FeH'])
    expected = np.array(([0.21501621, 0.23708980, 0.24951082],
                         [0.12492044, 0.13276138, 0.14383603]))
    np.testing.assert_array_almost_equal(desired, expected)

    # No arguments (entire DataFrame)
    desired = df[['x', 'y', 'z']].mutual_information(x=None, y=None)
    expected = np.array(([4.80278827, 0.15118145, 0.18439181],
                         [0.15118145, 4.87766263, 0.21418761],
                         [0.18439181, 0.21418761, 4.61453743]))
