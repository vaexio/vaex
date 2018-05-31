from common import *

def test_add_virtual_columns_polar_velocities_to_cartesian():
    ds = vaex.example()

    ds.add_virtual_columns_cartesian_velocities_to_polar()
    ds.add_virtual_columns_cartesian_to_polar()

    # With azimuth = None
    ds.add_virtual_columns_polar_velocities_to_cartesian(vx_out='vx_', vy_out='vy_')
    np.testing.assert_almost_equal(ds.evaluate('vx'), ds.evaluate('vx_'), err_msg='error with converting polar to Cartesian velocities')
    np.testing.assert_almost_equal(ds.evaluate('vy'), ds.evaluate('vy_'), err_msg='error with converting polar to Cartesian velocities')

    # With azimuth provided
    ds.add_virtual_columns_polar_velocities_to_cartesian(azimuth='phi_polar', vx_out='vx_', vy_out='vy_')
    np.testing.assert_almost_equal(ds.evaluate('vx'), ds.evaluate('vx_'), err_msg='error with converting polar to Cartesian velocities')
    np.testing.assert_almost_equal(ds.evaluate('vy'), ds.evaluate('vy_'), err_msg='error with converting polar to Cartesian velocities')

