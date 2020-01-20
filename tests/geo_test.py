from common import *


def test_virtual_columns_spherical():
    df = vaex.from_scalars(alpha=0, delta=0, distance=1)
    df.add_virtual_columns_spherical_to_cartesian("alpha", "delta", "distance", "x", "y", "z", radians=False)

    x, y, z = df['x'].values[0], df['y'].values[0], df['z'].values[0]

    np.testing.assert_array_almost_equal(x, 1)
    np.testing.assert_array_almost_equal(y, 0)
    np.testing.assert_array_almost_equal(z, 0)

    for radians in [True, False]:
        def dfs(alpha, delta, distance, radians=radians):
            ds_1 = vaex.from_scalars(alpha=alpha, delta=delta, distance=distance, alpha_e=0.1, delta_e=0.2, distance_e=0.3)
            ds_1.add_virtual_columns_spherical_to_cartesian("alpha", "delta", "distance", propagate_uncertainties=True, radians=radians)
            N = 1000000
            # distance
            alpha =        np.random.normal(0, 0.1, N) + alpha
            delta =        np.random.normal(0, 0.2, N) + delta
            distance =     np.random.normal(0, 0.3, N) + distance
            ds_many = vaex.from_arrays(alpha=alpha, delta=delta, distance=distance)
            ds_many.add_virtual_columns_spherical_to_cartesian("alpha", "delta", "distance", radians=radians)
            return ds_1, ds_many

        ds_1, ds_many = dfs(0, 0, 1.)
        x_e = ds_1.evaluate("x_uncertainty")[0]
        y_e = ds_1.evaluate("y_uncertainty")[0]
        z_e = ds_1.evaluate("z_uncertainty")[0]
        np.testing.assert_array_almost_equal(x_e, ds_many.std("x").item(), decimal=2)

        np.testing.assert_array_almost_equal(y_e, ds_many.std("y").item(), decimal=2)
        np.testing.assert_array_almost_equal(z_e, ds_many.std("z").item(), decimal=2)
        np.testing.assert_array_almost_equal(x_e, 0.3)

    # TODO: from cartesian tot spherical errors


    df.add_virtual_columns_cartesian_to_spherical("x", "y", "z", "theta", "phi", "r", radians=False)
    theta, phi, r = df("theta", "phi", "r").row(0)
    np.testing.assert_array_almost_equal(theta, 0)
    np.testing.assert_array_almost_equal(phi, 0)
    np.testing.assert_array_almost_equal(r, 1)


    df.add_virtual_columns_celestial("alpha", "delta", "l", "b", _matrix='eq2gal')
    # TODO: properly test, with and without radians
    df.evaluate("l")
    df.evaluate("b")

    ds = vaex.from_scalars(x=1, y=0, z=0)
    ds.add_virtual_columns_cartesian_to_spherical()
    assert ds.evaluate('b')[0] == 0
