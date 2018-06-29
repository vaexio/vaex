from vaex.dataset import Dataset
import numpy as np
import math

def patch(f):
    '''Adds method f to the Dataset class'''
    name = f.__name__
    setattr(Dataset, name, f)
    return f


@patch
def add_virtual_columns_eq2ecl(self, long_in="ra", lat_in="dec", long_out="lambda_", lat_out="beta", input=None, output=None, name_prefix="__celestial_eq2ecl", radians=False):
    """Add ecliptic coordates (long_out, lat_out) from equatorial coordinates.

    :param long_in: Name/expression for right ascension
    :param lat_in: Name/expression for declination
    :param long_out:  Output name for lambda coordinate
    :param lat_out: Output name for beta coordinate
    :param input:
    :param output:
    :param name_prefix:
    :param radians: input and output in radians (True), or degrees (False)
    :return:
    """
    import kapteyn.celestial as c
    self.add_virtual_columns_celestial(long_in, lat_in, long_out, lat_out, input=input or c.equatorial, output=output or c.ecliptic, name_prefix=name_prefix, radians=radians)


@patch
def add_virtual_columns_eq2gal(self, long_in="ra", lat_in="dec", long_out="l", lat_out="b", input=None, output=None, name_prefix="__celestial_eq2gal", radians=False):
    """Add galactic coordates (long_out, lat_out) from equatorial coordinates.

    :param long_in: Name/expression for right ascension
    :param lat_in: Name/expression for declination
    :param long_out:  Output name for galactic longitude
    :param lat_out: Output name for galactic latitude
    :param input:
    :param output:
    :param name_prefix:
    :param radians: input and output in radians (True), or degrees (False)
    :return:
    """
    import kapteyn.celestial as c
    self.add_virtual_columns_celestial(long_in, lat_in, long_out, lat_out, input=input or c.equatorial, output=output or c.galactic, name_prefix=name_prefix, radians=radians)


@patch
def add_virtual_columns_gal2eq(self, long_in='l', lat_in='b', long_out='ra', lat_out='dec', input=None, output=None, name_prefix="__celestial_gal2eq", radians=False):
    """
    Convert from galactic (l,b) to equatorial (ra,dec) spherical coordinate system.
    :param long_in: longitudinal angle l
    :param lat_in: latitudinal angle b
    :param long_out: right ascension
    :param lat_out: declination
    :param input:
    :param output:
    :param name_prefix:
    :param radians: input and output in radians (True), or degrees (False)
    """

    import kapteyn.celestial as c
    self.add_virtual_columns_celestial(long_in, lat_in, long_out, lat_out, input=input or c.galactic, output=output or c.equatorial, name_prefix=name_prefix, radians=radians)


@patch
def add_virtual_columns_distance_from_parallax(self, parallax="parallax", distance_name="distance", parallax_uncertainty=None, uncertainty_postfix="_uncertainty"):
    """Convert parallax to distance (i.e. 1/parallax)

    :param parallax: expression for the parallax, e.g. "parallax"
    :param distance_name: name for the virtual column of the distance, e.g. "distance"
    :param parallax_uncertainty: expression for the uncertainty on the parallax, e.g. "parallax_error"
    :param uncertainty_postfix: distance_name + uncertainty_postfix is the name for the virtual column, e.g. "distance_uncertainty" by default
    :return:
    """
    """


    """
    import astropy.units
    unit = self.unit(parallax)
    # if unit:
    #   convert = unit.to(astropy.units.mas)
    #   distance_expression = "%f/(%s)" % (convert, parallax)
    # else:
    distance_expression = "1/%s" % (parallax)
    self.ucds[distance_name] = "pos.distance"
    self.descriptions[distance_name] = "Derived from parallax (%s)" % parallax
    if unit:
        if unit == astropy.units.milliarcsecond:
            self.units[distance_name] = astropy.units.kpc
        if unit == astropy.units.arcsecond:
            self.units[distance_name] = astropy.units.parsec
    self.add_virtual_column(distance_name, distance_expression)
    if parallax_uncertainty:
        """
        y = 1/x
        sigma_y**2 = (1/x**2)**2 sigma_x**2
        sigma_y = (1/x**2) sigma_x
        sigma_y = y**2 sigma_x
        sigma_y/y = (1/x) sigma_x
        """
        name = distance_name + uncertainty_postfix
        distance_uncertainty_expression = "{parallax_uncertainty}/({parallax})**2".format(**locals())
        self.add_virtual_column(name, distance_uncertainty_expression)
        self.descriptions[name] = "Uncertainty on parallax (%s)" % parallax
        self.ucds[name] = "stat.error;pos.distance"


@patch
def add_virtual_columns_cartesian_velocities_to_pmvr(self, x="x", y="y", z="z", vx="vx", vy="vy", vz="vz", vr="vr", pm_long="pm_long", pm_lat="pm_lat", distance=None):
    """Concert velocities from a cartesian system to proper motions and radial velocities

    TODO: errors

    :param x: name of x column (input)
    :param y:         y
    :param z:         z
    :param vx:       vx
    :param vy:       vy
    :param vz:       vz
    :param vr: name of the column for the radial velocity in the r direction (output)
    :param pm_long: name of the column for the proper motion component in the longitude direction  (output)
    :param pm_lat: name of the column for the proper motion component in the latitude direction, positive points to the north pole (output)
    :param distance: Expression for distance, if not given defaults to sqrt(x**2+y**2+z**2), but if this column already exists, passing this expression may lead to a better performance
    :return:
    """
    if distance is None:
        distance = "sqrt({x}**2+{y}**2+{z}**2)".format(**locals())
    k = 4.74057
    self.add_variable("k", k, overwrite=False)
    self.add_virtual_column(vr, "({x}*{vx}+{y}*{vy}+{z}*{vz})/{distance}".format(**locals()))
    self.add_virtual_column(pm_long, "-({vx}*{y}-{x}*{vy})/sqrt({x}**2+{y}**2)/{distance}/k".format(**locals()))
    self.add_virtual_column(pm_lat, "-({z}*({x}*{vx}+{y}*{vy}) - ({x}**2+{y}**2)*{vz})/( ({x}**2+{y}**2+{z}**2) * sqrt({x}**2+{y}**2) )/k".format(**locals()))


@patch
def add_virtual_columns_proper_motion_eq2gal(self, long_in="ra", lat_in="dec", pm_long="pm_ra", pm_lat="pm_dec", pm_long_out="pm_l", pm_lat_out="pm_b",
                                            name_prefix="__proper_motion_eq2gal",
                                            right_ascension_galactic_pole=192.85,
                                            declination_galactic_pole=27.12,
                                            propagate_uncertainties=False,
                                            radians=False, inverse=False):
    """Transform/rotate proper motions from equatorial to galactic coordinates

    Taken from http://arxiv.org/abs/1306.2945

    :param long_in: Name/expression for right ascension
    :param lat_in: Name/expression for declination
    :param pm_long: Proper motion for ra
    :param pm_lat: Proper motion for dec
    :param pm_long_out:  Output name for output proper motion on l direction
    :param pm_lat_out: Output name for output proper motion on b direction
    :param name_prefix:
    :param radians: input and output in radians (True), or degrees (False)
    :parap inverse: (For internal use) convert from galactic to equatorial instead
    :return:
    """
    # import kapteyn.celestial as c
    """mu_gb =  mu_dec*(cdec*sdp-sdec*cdp*COS(ras))/cgb $
      - mu_ra*cdp*SIN(ras)/cgb"""
    long_in_original = long_in = self._expr(long_in)
    lat_in_original = lat_in = self._expr(lat_in)
    pm_long = self._expr(pm_long)
    pm_lat = self._expr(pm_lat)
    if not radians:
        long_in = long_in * np.pi/180
        lat_in = lat_in * np.pi/180
    c1_name = name_prefix + "_C1"
    c2_name = name_prefix + "_C2"
    right_ascension_galactic_pole = math.radians(right_ascension_galactic_pole)
    declination_galactic_pole = math.radians(declination_galactic_pole)
    self[c1_name] = c1 = np.sin(declination_galactic_pole) * np.cos(lat_in) - np.cos(declination_galactic_pole)*np.sin(lat_in)*np.cos(long_in-right_ascension_galactic_pole)
    self[c2_name] = c2 = np.cos(declination_galactic_pole) * np.sin(long_in - right_ascension_galactic_pole)
    c1 = self[c1_name]
    c2 = self[c2_name]
    if inverse:
        self[pm_long_out] = ( c1 * pm_long + -c2 * pm_lat)/np.sqrt(c1**2+c2**2)
        self[pm_lat_out] =  ( c2 * pm_long +  c1 * pm_lat)/np.sqrt(c1**2+c2**2)
    else:
        self[pm_long_out] = ( c1 * pm_long + c2 * pm_lat)/np.sqrt(c1**2+c2**2)
        self[pm_lat_out] =  (-c2 * pm_long + c1 * pm_lat)/np.sqrt(c1**2+c2**2)
    if propagate_uncertainties:
        self.propagate_uncertainties([self[pm_long_out], self[pm_lat_out]])

@patch
def add_virtual_columns_proper_motion_gal2eq(self, long_in="ra", lat_in="dec", pm_long="pm_l", pm_lat="pm_b", pm_long_out="pm_ra", pm_lat_out="pm_dec",
                                            name_prefix="__proper_motion_gal2eq",
                                            right_ascension_galactic_pole=192.85,
                                            declination_galactic_pole=27.12,
                                            propagate_uncertainties=False,
                                            radians=False):
    """Transform/rotate proper motions from galactic to equatorial coordinates.

    Inverse of :py:`add_virtual_columns_proper_motion_eq2gal`
    """
    kwargs = dict(**locals())
    kwargs.pop('self')
    kwargs['inverse'] = True
    self.add_virtual_columns_proper_motion_eq2gal(**kwargs)


@patch
def add_virtual_columns_lbrvr_proper_motion2vcartesian(self, long_in="l", lat_in="b", distance="distance", pm_long="pm_l", pm_lat="pm_b",
                                                       vr="vr", vx="vx", vy="vy", vz="vz",
                                                       center_v=(0, 0, 0),
                                                       propagate_uncertainties=False, radians=False):
    """Convert radial velocity and galactic proper motions (and positions) to cartesian velocities wrt the center_v

    Based on http://adsabs.harvard.edu/abs/1987AJ.....93..864J


    :param long_in: Name/expression for galactic longitude
    :param lat_in: Name/expression for galactic latitude
    :param distance: Name/expression for heliocentric distance
    :param pm_long: Name/expression for the galactic proper motion in latitude direction (pm_l*, so cosine(b) term should be included)
    :param pm_lat: Name/expression for the galactic proper motion in longitude direction
    :param vr: Name/expression for the radial velocity
    :param vx: Output name for the cartesian velocity x-component
    :param vy: Output name for the cartesian velocity y-component
    :param vz: Output name for the cartesian velocity z-component
    :param center_v: Extra motion that should be added, for instance lsr + motion of the sun wrt the galactic restframe
    :param radians: input and output in radians (True), or degrees (False)
    :return:
    """
    k = 4.74057
    a, d, distance = self._expr(long_in, lat_in, distance)
    pm_long, pm_lat, vr = self._expr(pm_long, pm_lat, vr)
    if not radians:
        a = a * np.pi/180
        d = d * np.pi/180
    A = [[np.cos(a)*np.cos(d), -np.sin(a), -np.cos(a)*np.sin(d)],
         [np.sin(a)*np.cos(d),  np.cos(a), -np.sin(a)*np.sin(d)],
         [np.sin(d), d*0, np.cos(d)]]
    self.add_virtual_columns_matrix3d(vr, k * pm_long * distance, k * pm_lat * distance, vx, vy, vz, A, translation=center_v)
    if propagate_uncertainties:
        self.propagate_uncertainties([self[vx], self[vy], self[vz]])


@patch
def add_virtual_columns_equatorial_to_galactic_cartesian(self, alpha, delta, distance, xname, yname, zname, radians=True, alpha_gp=np.radians(192.85948), delta_gp=np.radians(27.12825), l_omega=np.radians(32.93192)):
    """From http://arxiv.org/pdf/1306.2945v2.pdf"""
    if not radians:
        alpha = "pi/180.*%s" % alpha
        delta = "pi/180.*%s" % delta
    self.virtual_columns[zname] = "{distance} * (cos({delta}) * cos({delta_gp}) * cos({alpha} - {alpha_gp}) + sin({delta}) * sin({delta_gp}))".format(**locals())
    self.virtual_columns[xname] = "{distance} * (cos({delta}) * sin({alpha} - {alpha_gp}))".format(**locals())
    self.virtual_columns[yname] = "{distance} * (sin({delta}) * cos({delta_gp}) - cos({delta}) * sin({delta_gp}) * cos({alpha} - {alpha_gp}))".format(**locals())
    # self.write_virtual_meta()


@patch
def add_virtual_columns_celestial(self, long_in, lat_in, long_out, lat_out, input=None, output=None, name_prefix="__celestial", radians=False):
    import kapteyn.celestial as c
    input = input if input is not None else c.equatorial
    output = output if output is not None else c.galactic
    matrix = c.skymatrix((input, 'j2000', c.fk5), output)[0].tolist()
    if not radians:
        long_in = "pi/180.*%s" % long_in
        lat_in = "pi/180.*%s" % lat_in
    x_in = name_prefix + "_in_x"
    y_in = name_prefix + "_in_y"
    z_in = name_prefix + "_in_z"
    x_out = name_prefix + "_out_x"
    y_out = name_prefix + "_out_y"
    z_out = name_prefix + "_out_z"
    self.add_virtual_column(x_in, "cos({long_in})*cos({lat_in})".format(**locals()))
    self.add_virtual_column(y_in, "sin({long_in})*cos({lat_in})".format(**locals()))
    self.add_virtual_column(z_in, "sin({lat_in})".format(**locals()))
    self.add_virtual_columns_matrix3d(x_in, y_in, z_in, x_out, y_out, z_out,
                                      matrix, name_prefix + "_matrix")
    transform = "" if radians else "*180./pi"
    x = x_out
    y = y_out
    z = z_out
    self.add_virtual_column(long_out, "arctan2({y}, {x}){transform}".format(**locals()))
    self.add_virtual_column(lat_out, "(-arccos({z}/sqrt({x}**2+{y}**2+{z}**2))+pi/2){transform}".format(**locals()))

@patch
def add_virtual_columns_proper_motion2vperpendicular(self, distance="distance", pm_long="pm_l", pm_lat="pm_b",
                                                     vl="vl", vb="vb",
                                                     propagate_uncertainties=False,
                                                     radians=False):
    """Convert proper motion to perpendicular velocities.

    :param distance:
    :param pm_long:
    :param pm_lat:
    :param vl:
    :param vb:
    :param cov_matrix_distance_pm_long_pm_lat:
    :param uncertainty_postfix:
    :param covariance_postfix:
    :param radians:
    :return:
    """
    k = 4.74057
    self.add_variable("k", k, overwrite=False)
    self.add_virtual_column(vl, "k*{pm_long}*{distance}".format(**locals()))
    self.add_virtual_column(vb, "k* {pm_lat}*{distance}".format(**locals()))
    if propagate_uncertainties:
        self.propagate_uncertainties([self[vl], self[vb]])


@patch
def add_virtual_columns_cartesian_angular_momenta(self, x='x', y='y', z='z',
                                                  vx='vx', vy='vy', vz='vz',
                                                  Lx='Lx', Ly='Ly', Lz='Lz',
                                                  propagate_uncertainties=False):
    """
    Calculate the angular momentum components provided Cartesian positions and velocities.
    Be mindful of the point of origin: ex. if considering Galactic dynamics, and positions and
    velocities should be as seen from the Galactic centre.

    :param x: x-position Cartesian component
    :param y: y-position Cartesian component
    :param z: z-position Cartesian component
    :param vx: x-velocity Cartesian component
    :param vy: y-velocity Cartesian component
    :param vz: z-velocity Cartesian component
    :param Lx: name of virtual column
    :param Ly: name of virtual column
    :param Lz: name of virtial column
    :propagate_uncertainties: (bool) whether to propagate the uncertainties of
    the positions and velocities to the angular momentum components
    """

    x, y, z, vx, vy, vz = self._expr(x, y, z, vx, vy, vz)
    self.add_virtual_column(Lx, y * vz - z * vy)
    self.add_virtual_column(Ly, z * vx - x * vz)
    self.add_virtual_column(Lz, x * vy - y * vx)
    if propagate_uncertainties:
        self.propagate_uncertainties([self[Lx], self[Ly], self[Lz]])


def add_plugin():
    pass  # importing this module already does the job
