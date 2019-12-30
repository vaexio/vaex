import numpy as np

import vaex


def correlation_test():
    df = vaex.example()

    # A single column pair
    desired = df.correlation('x', 'y')
    expected = np.array(-0.066913)
    np.testing.assert_array_almost_equal(desired, expected)

    # A list of columns
    desired = df.correlation(x=['x', 'y', 'z'])
    expected = np.array([[ 1.00000000, -0.06691309, -0.02656313],
                         [-0.06691309,  1.00000000,  0.03083857],
                         [-0.02656313,  0.03083857,  1.00000000]])
    np.testing.assert_array_almost_equal(desired, expected)

    # A list of columns and a single target
    desired = df.correlation(x=['x', 'y', 'z'], y='vx')
    expected = np.array([-0.00779179,  0.01804911, -0.02175331])
    np.testing.assert_array_almost_equal(desired, expected)

    # A list of columns and a list of targets
    desired = df.correlation(x=['x', 'y', 'z'], y=['vx', 'vy'])
    expected = np.array(([-0.00779179,  0.01804911, -0.02175331],
                         [0.00014019, -0.00411498, 0.02988355]))

    # No arguments (entire DataFrame)
    desired = df[['x', 'y', 'z']].correlation(x=None, y=None)
    expected = np.array([[ 1.00000000, -0.06691309, -0.02656313],
                         [-0.06691309,  1.00000000,  0.03083857],
                         [-0.02656313,  0.03083857,  1.00000000]])
