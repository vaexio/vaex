from common import *
import collections

try:
    import cupy
except:
    cupy = None


# From http://pythonhosted.org/pythran/MANUAL.html
def arc_distance(theta_1, phi_1, theta_2, phi_2):
    """
    Calculates the pairwise arc distance
    between all points in vector a and b.
    """
    temp = (np.sin((theta_2-theta_1)/2)**2
           + np.cos(theta_1)*np.cos(theta_2) * np.sin((phi_2-phi_1)/2)**2)
    distance_matrix = 2 * np.arctan2(np.sqrt(temp), np.sqrt(1-temp))
    return distance_matrix

def test_numba(ds):
    ds_original = ds.copy()
    #ds.columns['x'] = (ds.columns['x']*1).copy()  # convert non non-big endian for now
    expr = arc_distance(ds.y*1, ds.y*1, ds.y**2*ds.y, ds.x+ds.y)
    ds['arc_distance'] = expr
    #assert ds.arc_distance.expression == expr.expression
    ds['arc_distance_jit'] = ds['arc_distance'].jit_numba()
    np.testing.assert_array_almost_equal(ds.arc_distance.tolist(), ds.arc_distance_jit.tolist())
    # TODO: make it such that they can be pickled
    ds_original.state_set(ds.state_get())
    ds = ds_original
    np.testing.assert_array_almost_equal(ds.arc_distance.tolist(), ds.arc_distance_jit.tolist())


@pytest.mark.skipif(sys.version_info < (3,6) and sys.version_info[0] != 2,
                    reason="no support for python3.5 (numba segfaults)")
def test_jit_overwrite(ds_local):
    ds = ds_local # TODO: remote overwriting of functions does not work
    ds_original = ds.copy()
    expr = arc_distance(ds.y*1, ds.y*1, ds.y**2*ds.y, ds.x+ds.y)
    ds['arc_distance'] = expr
    ds['arc_distance_jit'] = ds['arc_distance'].jit_numba()
    ds['arc_distance_jit'] = ds['arc_distance * 2'].jit_numba()
    np.testing.assert_array_almost_equal((ds.arc_distance*2).tolist(), ds.arc_distance_jit.tolist())


@pytest.mark.skipif(cupy is None,
                    reason="cuda support relies on cupy")
def test_cuda(ds_local):
    ds = ds_local
    ds_original = ds.copy()
    #ds.columns['x'] = (ds.columns['x']*1).copy()  # convert non non-big endian for now
    expr = arc_distance(ds.y*1, ds.y*1, ds.y**2*ds.y, ds.x+ds.y)
    ds['arc_distance'] = expr
    print(expr)
    #assert ds.arc_distance.expression == expr.expression
    ds['arc_distance_jit'] = ds['arc_distance'].jit_cuda()
    np.testing.assert_almost_equal(ds.arc_distance.values, ds.arc_distance_jit.values)
    # TODO: make it such that they can be pickled
    ds_original.state_set(ds.state_get())
    ds = ds_original
    np.testing.assert_almost_equal(ds.arc_distance.values, ds.arc_distance_jit.values)


def test_metal(df_local):
    pytest.importorskip("Metal")
    df = df_local
    df_original = df.copy()
    #df.columns['x'] = (df.columns['x']*1).copy()  # convert non non-big endian for now
    expr = arc_distance(df.y*1, df.y*1, df.y**2*df.y, df.x+df.y)
    # expr = df.x + df.y
    df['arc_distance'] = expr
    #assert df.arc_distance.expression == expr.expression
    df['arc_distance_jit'] = df['arc_distance'].jit_metal()
    # assert df.arc_distance.tolist() == df.arc_distance_jit.tolist()
    np.testing.assert_almost_equal(df.arc_distance.values, df.arc_distance_jit.values, decimal=1)
    # TODO: make it such that they can be pickled
    df_original.state_set(df.state_get())
    df = df_original
    np.testing.assert_almost_equal(df.arc_distance.values, df.arc_distance_jit.values, decimal=1)


@pytest.mark.parametrize("type_name", vaex.array_types._type_names)
def test_types_metal(type_name, df_factory_numpy):
    pytest.importorskip("Metal")
    df = df_factory_numpy(x=np.array([0, 1, 2], dtype=type_name), y=[2, 3, 4])
    # df = df_factory_numpy(x=np.array([0, 1, 2], dtype=type_name), y=np.array([2, 3, 4], dtype=type_name))
    # df['x'] = df['x'].astype(type_name)
    df['z'] = (df['x'] + df['y']).jit_metal()
    assert df['z'].tolist() == [2, 4, 6]
