import vaex
import numpy as np

def test_inside_polygon():
    ds = vaex.from_arrays(x=[1, 2, 3], y=[2, 3, 4])
    polygon = np.array( [(1.5, 2.5, 2.5, 1.5), (2.5, 2.5, 3.5, 3.5)] )
    px = polygon[0]
    py = polygon[1]
    ds['inside'] = ds.func.inside_polygon(ds.x, ds.y, px, py)
    assert ds.inside.values.tolist() == [False, True, False]

def test_inside_polygon():
    ds = vaex.from_arrays(x=[1, 2, 3], y=[2, 3, 4])
    polygon1 = np.array( [(1.5, 2.5, 2.5, 1.5), (2.5, 2.5, 3.5, 3.5)] )
    px1 = polygon1[0]
    py1 = polygon1[1]
    polygon2 = (polygon1.T + [1, 1]).T
    px2 = polygon2[0]
    py2 = polygon2[1]
    px = [px1, px2]
    py = [py1, py2]
    ds['inside'] = ds.func.inside_polygons(ds.x, ds.y, px, py, any=True)
    assert ds.inside.values.tolist() == [False, True, True]


def test_which_polygon():
    ds = vaex.from_arrays(x=[1, 2, 3], y=[2, 3, 4])
    polygon1 = np.array( [(1.5, 2.5, 2.5, 1.5), (2.5, 2.5, 3.5, 3.5)] )
    px1 = polygon1[0]
    py1 = polygon1[1]
    polygon2 = (polygon1.T + [1, 1]).T
    px2 = polygon2[0]
    py2 = polygon2[1]
    px = [px1, px2]
    py = [py1, py2]
    ds['polygon_index'] = ds.func.inside_which_polygon(ds.x, ds.y, px, py)
    assert ds.polygon_index.values.tolist() == [None, 0, 1]

def test_which_polygons():
    ds = vaex.from_arrays(x=[1, 2, 3], y=[2, 3, 4])
    polygon1a = np.array( [(1.5, 2.5, 2.5, 1.5), (2.5, 2.5, 3.5, 3.5)] )
    polygon1b = (polygon1a.T + [1, 1]).T

    polygon2a = polygon1b
    polygon2b = polygon2a
    polygon2c = polygon2a

    pxs = [[polygon1a, polygon1b], [polygon2a, polygon2b, polygon2c]]
    ds['polygon_index'] = ds.func.inside_which_polygons(ds.x, ds.y, pxs, any=True)
    assert ds.polygon_index.values.tolist() == [None, 0, 0]

    ds['polygon_index'] = ds.func.inside_which_polygons(ds.x, ds.y, pxs, any=False)
    assert ds.polygon_index.values.tolist() == [None, None, 1]
