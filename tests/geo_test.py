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


def test_inside_polygon_single():
    df = vaex.from_arrays(x=[1, 2, 3], y=[2, 3, 4])
    px = np.array([1.5, 2.5, 2.5, 1.5])
    py = np.array([2.5, 2.5, 3.5, 3.5])
    df['inside'] = df.geo.inside_polygon(df.x, df.y, px, py)
    assert df.inside.values.tolist() == [False, True, False]


def test_inside_polygons():
    df = vaex.from_arrays(x=[1, 2, 3], y=[2, 3, 4])
    px = np.array([1.5, 2.5, 2.5, 1.5])
    py = np.array([2.5, 2.5, 3.5, 3.5])
    df['inside'] = df.geo.inside_polygons(df.x, df.y, [px, px + 1], [py, py + 1], any=True)
    assert df.inside.values.tolist() == [False, True, True]


def test_which_polygon_single():
    df = vaex.from_arrays(x=[1, 2, 3], y=[2, 3, 4])
    px = np.array([1.5, 2.5, 2.5, 1.5])
    py = np.array([2.5, 2.5, 3.5, 3.5])
    df['polygon_index'] = df.geo.inside_which_polygon(df.x, df.y, [px, px + 1], [py, py + 1])
    assert df.polygon_index.values.tolist() == [None, 0, 1]


def test_which_polygons():
    df = vaex.from_arrays(x=[1, 2, 3], y=[2, 3, 4])
    # polygon1a = np.array( [(1.5, 2.5, 2.5, 1.5), (2.5, 2.5, 3.5, 3.5)] )
    # polygon1b = (polygon1a.T + [1, 1]).T
    px = np.array([1.5, 2.5, 2.5, 1.5])
    py = np.array([2.5, 2.5, 3.5, 3.5])
    polygon1a = [px, py]  # matches #1
    polygon1b = [px + 1, py + 1]  # matches #2
    polygon_nothing = [px + 10, py + 10]  # matches nothing

    pxw = np.array([1.5, 3.5, 3.5, 1.5])
    pyw = np.array([2.5, 2.5, 4.5, 4.5])
    polygon1c = [pxw, pyw]  # matches #1 and 2

    pxs = [[polygon1a, polygon1b], [polygon1b, polygon1c], [polygon1c]]
    df['polygon_index'] = df.geo.inside_which_polygons(df.x, df.y, pxs, any=True)
    assert df.polygon_index.values.tolist() == [None, 0, 0]

    df['polygon_index'] = df.geo.inside_which_polygons(df.x, df.y, pxs, any=False)
    assert df.polygon_index.values.tolist() == [None, 2, 1]

    pxs = [[polygon_nothing, polygon1a, polygon_nothing]]
    df['polygon_index'] = df.geo.inside_which_polygons(df.x, df.y, pxs, any=True)
    assert df.polygon_index.values.tolist() == [None, 0, None]

    pxs = [[polygon1a, polygon_nothing, polygon1a]]
    df['polygon_index'] = df.geo.inside_which_polygons(df.x, df.y, pxs, any=False)
    assert df.polygon_index.values.tolist() == [None, None, None]
