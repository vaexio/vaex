from vaex.dataframe import DataFrame
import numpy as np


def add_plugin():
    pass  # importing this module already does the job


def patch(f):
    '''Adds method f to the DataFrame class'''
    name = f.__name__
    DataFrame.__hidden__[name] = f
    return f


@patch
def add_virtual_columns_eq2ecl(self, long_in="ra", lat_in="dec", long_out="lambda_", lat_out="beta", name_prefix="__celestial_eq2ecl", radians=False):
    kwargs = dict(**locals())
    kwargs.pop('self')
    return self.astro.eq2ecl(inplace=True, **kwargs)


@patch
def add_virtual_columns_eq2gal(self, long_in="ra", lat_in="dec", long_out="l", lat_out="b", name_prefix="__celestial_eq2gal", radians=False):
    kwargs = dict(**locals())
    kwargs.pop('self')
    return self.astro.eq2gal(inplace=True, **kwargs)


@patch
def add_virtual_columns_gal2eq(self, long_in='l', lat_in='b', long_out='ra', lat_out='dec', name_prefix="__celestial_gal2eq", radians=False):
    kwargs = dict(**locals())
    kwargs.pop('self')
    return self.astro.gal2eq(inplace=True, **kwargs)


@patch
def add_virtual_columns_distance_from_parallax(self, parallax="parallax", distance_name="distance", parallax_uncertainty=None, uncertainty_postfix="_uncertainty"):
    kwargs = dict(**locals())
    kwargs.pop('self')
    return self.astro.parallax2distance(inplace=True, **kwargs)


@patch
def add_virtual_columns_cartesian_velocities_to_pmvr(self, x="x", y="y", z="z", vx="vx", vy="vy", vz="vz", vr="vr", pm_long="pm_long", pm_lat="pm_lat", distance=None):
    kwargs = dict(**locals())
    kwargs.pop('self')
    return self.astro.velocity_cartesian2pmvr(inplace=True, **kwargs)


@patch
def add_virtual_columns_proper_motion_eq2gal(self, long_in="ra", lat_in="dec", pm_long="pm_ra", pm_lat="pm_dec", pm_long_out="pm_l", pm_lat_out="pm_b",
                                            name_prefix="__proper_motion_eq2gal",
                                            right_ascension_galactic_pole=192.85,
                                            declination_galactic_pole=27.12,
                                            propagate_uncertainties=False,
                                            radians=False, inverse=False):
    kwargs = dict(**locals())
    kwargs.pop('self')
    return self.astro.pm_eq2gal(inplace=True, **kwargs)


@patch
def add_virtual_columns_proper_motion_gal2eq(self, long_in="ra", lat_in="dec", pm_long="pm_l", pm_lat="pm_b", pm_long_out="pm_ra", pm_lat_out="pm_dec",
                                            name_prefix="__proper_motion_gal2eq",
                                            right_ascension_galactic_pole=192.85,
                                            declination_galactic_pole=27.12,
                                            propagate_uncertainties=False,
                                            radians=False):
    kwargs = dict(**locals())
    kwargs.pop('self')
    return self.astro.pm_gal2eq(inplace=True, **kwargs)


@patch
def add_virtual_columns_lbrvr_proper_motion2vcartesian(self, long_in="l", lat_in="b", distance="distance", pm_long="pm_l", pm_lat="pm_b",
                                                       vr="vr", vx="vx", vy="vy", vz="vz",
                                                       center_v=(0, 0, 0),
                                                       propagate_uncertainties=False, radians=False):
    kwargs = dict(**locals())
    kwargs.pop('self')
    return self.astro.lbrvrpm2vcartesian(inplace=True, **kwargs)


@patch
def add_virtual_columns_equatorial_to_galactic_cartesian(self, alpha, delta, distance, xname, yname, zname, radians=True, alpha_gp=np.radians(192.85948), delta_gp=np.radians(27.12825), l_omega=np.radians(32.93192)):
    kwargs = dict(**locals())
    kwargs.pop('self')
    return self.astro.equatorial2galactic_cartesian(inplace=True, **kwargs)


@patch
def add_virtual_columns_celestial(self, long_in, lat_in, long_out, lat_out, name_prefix="__celestial", radians=False, _matrix=None):
    kwargs = dict(**locals())
    kwargs.pop('self')
    return self.astro.celestial(inplace=True, **kwargs)

@patch
def add_virtual_columns_proper_motion2vperpendicular(self, distance="distance", pm_long="pm_l", pm_lat="pm_b",
                                                     vl="vl", vb="vb",
                                                     propagate_uncertainties=False,
                                                     radians=False):
    kwargs = dict(**locals())
    kwargs.pop('self')
    return self.astro.proper_motion2vperpendicular(inplace=True, **kwargs)


@patch
def add_virtual_columns_cartesian_angular_momenta(self, x='x', y='y', z='z',
                                                  vx='vx', vy='vy', vz='vz',
                                                  Lx='Lx', Ly='Ly', Lz='Lz',
                                                  propagate_uncertainties=False):
    kwargs = dict(**locals())
    kwargs.pop('self')
    return self.astro.cartesian_angular_momenta(inplace=True, **kwargs)
