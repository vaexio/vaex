from common import *

def test_add_virtual_columns_polar_velocities_to_cartesian():
    ds = vaex.example()

    ds.add_virtual_columns_cartesian_velocities_to_polar()
    ds.add_virtual_columns_cartesian_to_polar()

    # With azimuth = None
    ds.add_virtual_columns_polar_velocities_to_cartesian(vx_out='vx_', vy_out='vy_')
    np.testing.assert_almost_equal(ds.evaluate('vx'), ds.evaluate('vx_'), err_msg='error with converting polar to Cartesian velocities', decimal=3)
    np.testing.assert_almost_equal(ds.evaluate('vy'), ds.evaluate('vy_'), err_msg='error with converting polar to Cartesian velocities', decimal=3)

    # With azimuth provided
    ds.add_virtual_columns_polar_velocities_to_cartesian(azimuth='phi_polar', vx_out='vx_', vy_out='vy_')
    np.testing.assert_almost_equal(ds.evaluate('vx'), ds.evaluate('vx_'), err_msg='error with converting polar to Cartesian velocities', decimal=3)
    np.testing.assert_almost_equal(ds.evaluate('vy'), ds.evaluate('vy_'), err_msg='error with converting polar to Cartesian velocities', decimal=3)

    # this tests the angular momentum conversion
    ds.add_virtual_columns_cartesian_angular_momenta(Lx='Lx_', Ly='Ly_', Lz='Lz_')
    ds['L_'] = np.sqrt(ds.Lx_**2. + ds.Ly_**2. + ds.Lz_**2.)
    np.testing.assert_almost_equal(ds.Lz.values, ds.Lz_.values, err_msg='error when calculating Lz', decimal=3)
    np.testing.assert_almost_equal(ds.L.values, ds.L_.values, err_msg='error when calculating the Ltotal', decimal=3)
