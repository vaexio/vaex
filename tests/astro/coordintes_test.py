import numpy as np
import vaex


def test_cartesian_to_spherical():
    x = np.arange(10)
    df = vaex.from_arrays(x=x)
    assert df.astro is not None


def test_eq2gal():
    df = vaex.from_scalars(ra=1, dec=2)
    df = df.astro.eq2gal()
    assert df.l.tolist() != 1
    assert df.b.tolist() != 2


def test_add_virtual_columns_proper_motion_eq2gal():
    for radians in [True, False]:
        def dfs(alpha, delta, pm_a, pm_d, radians=radians):
            ds_1 = vaex.from_scalars(alpha=alpha, delta=delta, pm_a=pm_a, pm_d=pm_d, alpha_e=0.01, delta_e=0.02, pm_a_e=0.003, pm_d_e=0.004)
            ds_1 = ds_1.astro.pm_eq2gal("alpha", "delta", "pm_a", "pm_d", "pm_l", "pm_b", propagate_uncertainties=True, radians=radians)
            N = 100000
            # distance
            alpha =        np.random.normal(0, 0.01, N)  + alpha
            delta =        np.random.normal(0, 0.02, N)  + delta
            pm_a =         np.random.normal(0, 0.003, N)  + pm_a
            pm_d =         np.random.normal(0, 0.004, N)  + pm_d
            ds_many = vaex.from_arrays(alpha=alpha, delta=delta, pm_a=pm_a, pm_d=pm_d)
            ds_many.astro.pm_eq2gal("alpha", "delta", "pm_a", "pm_d", "pm_l", "pm_b", radians=radians, inplace=True)
            return ds_1, ds_many
        ds_1, ds_many = dfs(0, 0, 1, 2)

        if 0: # only for testing the test
            c1_e = ds_1.evaluate("c1_uncertainty")[0]
            c2_e = ds_1.evaluate("c2_uncertainty")[0]
            np.testing.assert_almost_equal(c1_e, ds_many.std("__proper_motion_eq2gal_C1").item(), decimal=3)
            np.testing.assert_almost_equal(c2_e, ds_many.std("__proper_motion_eq2gal_C2").item(), decimal=3)

        pm_l_e = ds_1.evaluate("pm_l_uncertainty")[0]
        pm_b_e = ds_1.evaluate("pm_b_uncertainty")[0]
        np.testing.assert_almost_equal(pm_l_e, ds_many.std("pm_l").item(), decimal=3)
        np.testing.assert_almost_equal(pm_b_e, ds_many.std("pm_b").item(), decimal=3)
