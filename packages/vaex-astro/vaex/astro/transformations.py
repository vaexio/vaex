import vaex
import numpy as np
import math


# All assume equinox J2000
comat = {'eq2ecl': [[0.9999999999999928, 1.1102233723050031e-07, 4.411803426976324e-08],
                     [-1.1941015020086788e-07, 0.9174821814419274, 0.39777688059582816],
                     [3.684608657254395e-09, -0.39777688059583055, 0.9174821814419342]],
         'eq2gal': [[-0.05487553939574265, -0.8734371047275962, -0.48383499177002515],
                     [0.49410945362774394, -0.4448295942975751, 0.7469822486998918],
                     [-0.8676661356833737, -0.19807638961301982, 0.45598379452141985]],
         'gal2eq': [[-0.0548756577126198, 0.4941094371971076, -0.8676661375571625],
                     [-0.873437051955779, -0.44482972122205366, -0.19807633727507046],
                     [-0.48383507361641837, 0.7469821839845096, 0.45598381369115243]]}


class DataFrameAccessorAstro(object):
    """Astronomy specific helper methods

    Example:
    >>> df_lb = df.geo.eq2gal(df.ra, df.dec)

    """
    def __init__(self, df):
        self.df = df

    def _trans(self, long_in, lat_in, long_out, lat_out, name_prefix="__celestial", radians=False, _matrix=None, inplace=False):
        df = self.df if inplace else self.df.copy()
        df.add_variable('pi', np.pi)
        matrix = comat[_matrix]
        if not radians:
            long_in = "pi/180.*%s" % long_in
            lat_in = "pi/180.*%s" % lat_in
        x_in = name_prefix + "_in_x"
        y_in = name_prefix + "_in_y"
        z_in = name_prefix + "_in_z"
        x_out = name_prefix + "_out_x"
        y_out = name_prefix + "_out_y"
        z_out = name_prefix + "_out_z"
        df.add_virtual_column(x_in, "cos({long_in})*cos({lat_in})".format(**locals()))
        df.add_virtual_column(y_in, "sin({long_in})*cos({lat_in})".format(**locals()))
        df.add_virtual_column(z_in, "sin({lat_in})".format(**locals()))
        df.add_virtual_columns_matrix3d(x_in, y_in, z_in, x_out, y_out, z_out,
                                        matrix, name_prefix + "_matrix")
        transform = "" if radians else "*180./pi"
        x = x_out
        y = y_out
        z = z_out
        df.add_virtual_column(long_out, "arctan2({y}, {x}){transform}".format(**locals()))
        df.add_virtual_column(lat_out, "(-arccos({z}/sqrt({x}**2+{y}**2+{z}**2))+pi/2){transform}".format(**locals()))
        return df

    def eq2ecl(self, long_in="ra", lat_in="dec", long_out="lambda_", lat_out="beta", name_prefix="__celestial_eq2ecl", radians=False, inplace=False):
        """Add ecliptic coordates (long_out, lat_out) from equatorial coordinates.

        :param long_in: Name/expression for right ascension
        :param lat_in: Name/expression for declination
        :param long_out:  Output name for lambda coordinate
        :param lat_out: Output name for beta coordinate
        :param name_prefix:
        :param radians: input and output in radians (True), or degrees (False)
        :return:
        """
        return self._trans(long_in, lat_in, long_out, lat_out, name_prefix=name_prefix, radians=radians, _matrix='eq2ecl', inplace=inplace)

    def eq2gal(self, long_in="ra", lat_in="dec", long_out="l", lat_out="b", name_prefix="__celestial_eq2gal", radians=False, inplace=False):
        """Add galactic coordates (long_out, lat_out) from equatorial coordinates.

        :param long_in: Name/expression for right ascension
        :param lat_in: Name/expression for declination
        :param long_out:  Output name for galactic longitude
        :param lat_out: Output name for galactic latitude
        :param name_prefix:
        :param radians: input and output in radians (True), or degrees (False)
        :return:
        """
        return self._trans(long_in, lat_in, long_out, lat_out, name_prefix=name_prefix, radians=radians, _matrix='eq2gal', inplace=inplace)

    def gal2eq(self, long_in='l', lat_in='b', long_out='ra', lat_out='dec', name_prefix="__celestial_gal2eq", radians=False, inplace=False):
        """
        Convert from galactic (l,b) to equatorial (ra,dec) spherical coordinate system.
        :param long_in: longitudinal angle l
        :param lat_in: latitudinal angle b
        :param long_out: right ascension
        :param lat_out: declination
        :param name_prefix:
        :param radians: input and output in radians (True), or degrees (False)
        """
        return self._trans(long_in, lat_in, long_out, lat_out, name_prefix=name_prefix, radians=radians, _matrix='gal2eq', inplace=inplace)


    def pm_eq2gal(self, long_in="ra", lat_in="dec", pm_long="pm_ra", pm_lat="pm_dec", pm_long_out="pm_l", pm_lat_out="pm_b",
                                            name_prefix="__proper_motion_eq2gal",
                                            right_ascension_galactic_pole=192.85,
                                            declination_galactic_pole=27.12,
                                            propagate_uncertainties=False,
                                            radians=False, inverse=False,
                                            inplace=False):
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
        """mu_gb =  mu_dec*(cdec*sdp-sdec*cdp*COS(ras))/cgb $
        - mu_ra*cdp*SIN(ras)/cgb"""
        df = self.df if inplace else self.df.copy()
        long_in_original = long_in = df._expr(long_in)
        lat_in_original = lat_in = df._expr(lat_in)
        pm_long = df._expr(pm_long)
        pm_lat = df._expr(pm_lat)
        if not radians:
            long_in = long_in * np.pi/180
            lat_in = lat_in * np.pi/180
        c1_name = name_prefix + "_C1"
        c2_name = name_prefix + "_C2"
        right_ascension_galactic_pole = math.radians(right_ascension_galactic_pole)
        declination_galactic_pole = math.radians(declination_galactic_pole)
        df[c1_name] = c1 = np.sin(declination_galactic_pole) * np.cos(lat_in) - np.cos(declination_galactic_pole)*np.sin(lat_in)*np.cos(long_in-right_ascension_galactic_pole)
        df[c2_name] = c2 = np.cos(declination_galactic_pole) * np.sin(long_in - right_ascension_galactic_pole)
        c1 = df[c1_name]
        c2 = df[c2_name]
        if inverse:
            df[pm_long_out] = ( c1 * pm_long + -c2 * pm_lat)/np.sqrt(c1**2+c2**2)
            df[pm_lat_out] =  ( c2 * pm_long +  c1 * pm_lat)/np.sqrt(c1**2+c2**2)
        else:
            df[pm_long_out] = ( c1 * pm_long + c2 * pm_lat)/np.sqrt(c1**2+c2**2)
            df[pm_lat_out] =  (-c2 * pm_long + c1 * pm_lat)/np.sqrt(c1**2+c2**2)
        if propagate_uncertainties:
            df.propagate_uncertainties([df[pm_long_out], df[pm_lat_out]])
        return df

    def pm_gal2eq(self, long_in="ra", lat_in="dec", pm_long="pm_l", pm_lat="pm_b", pm_long_out="pm_ra", pm_lat_out="pm_dec",
                                            name_prefix="__proper_motion_gal2eq",
                                            right_ascension_galactic_pole=192.85,
                                            declination_galactic_pole=27.12,
                                            propagate_uncertainties=False,
                                            radians=False,
                                            inplace=False):
        """Transform/rotate proper motions from galactic to equatorial coordinates.

        Inverse of :py:`pm_eq2gal`
        """
        kwargs = dict(**locals())
        kwargs.pop('self')
        kwargs['inverse'] = True
        return self.pm_eq2gal(**kwargs)

    def cartesian_angular_momenta(self, x='x', y='y', z='z',
                                                    vx='vx', vy='vy', vz='vz',
                                                    Lx='Lx', Ly='Ly', Lz='Lz',
                                                    propagate_uncertainties=False,
                                                    inplace=False):
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
        df = self.df if inplace else self.df.copy()
        x, y, z, vx, vy, vz = df._expr(x, y, z, vx, vy, vz)
        df.add_virtual_column(Lx, y * vz - z * vy)
        df.add_virtual_column(Ly, z * vx - x * vz)
        df.add_virtual_column(Lz, x * vy - y * vx)
        if propagate_uncertainties:
            df.propagate_uncertainties([df[Lx], df[Ly], df[Lz]])
        return df

    def proper_motion2vperpendicular(self, distance="distance", pm_long="pm_l", pm_lat="pm_b",
                                                        vl="vl", vb="vb",
                                                        propagate_uncertainties=False,
                                                        radians=False,
                                                        inplace=True):
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
        df = self.df if inplace else self.df.copy()
        k = 4.74057
        df.add_variable("k", k, overwrite=False)
        df.add_virtual_column(vl, "k*{pm_long}*{distance}".format(**locals()))
        df.add_virtual_column(vb, "k* {pm_lat}*{distance}".format(**locals()))
        if propagate_uncertainties:
            df.propagate_uncertainties([df[vl], df[vb]])
        return df

    def celestial(self, long_in, lat_in, long_out, lat_out, name_prefix="__celestial", radians=False, _matrix=None, inplace=False):
        """Adds celestial/spherical coordinates"""
        #TODO: return wrapped angle, currently returning angles between [-2pi:2pi]. Need modulo arhitmetics on vaex add_virtual_column expressions.
        #      for example see http://docs.astropy.org/en/stable/_modules/astropy/coordinates/angles.html#Angle.wrap_at
        df = self.df if inplace else self.df.copy()
        matrix = comat[_matrix]
        if not radians:
            long_in = "pi/180.*%s" % long_in
            lat_in = "pi/180.*%s" % lat_in
        x_in = name_prefix + "_in_x"
        y_in = name_prefix + "_in_y"
        z_in = name_prefix + "_in_z"
        x_out = name_prefix + "_out_x"
        y_out = name_prefix + "_out_y"
        z_out = name_prefix + "_out_z"
        df.add_virtual_column(x_in, "cos({long_in})*cos({lat_in})".format(**locals()))
        df.add_virtual_column(y_in, "sin({long_in})*cos({lat_in})".format(**locals()))
        df.add_virtual_column(z_in, "sin({lat_in})".format(**locals()))
        df.add_virtual_columns_matrix3d(x_in, y_in, z_in, x_out, y_out, z_out,
                                        matrix, name_prefix + "_matrix")
        transform = "" if radians else "*180./pi"
        x = x_out
        y = y_out
        z = z_out
        df.add_virtual_column(long_out, "arctan2({y}, {x}){transform}".format(**locals()))
        df.add_virtual_column(lat_out, "(-arccos({z}/sqrt({x}**2+{y}**2+{z}**2))+pi/2){transform}".format(**locals()))
        return df

    def equatorial2galactic_cartesian(self, alpha, delta, distance, xname, yname, zname, radians=True, alpha_gp=np.radians(192.85948), delta_gp=np.radians(27.12825), l_omega=np.radians(32.93192), inplace=False):
        """From http://arxiv.org/pdf/1306.2945v2.pdf"""
        df = self.df if inplace else self.df.copy()
        if not radians:
            df.add_variable('pi', np.pi)
            alpha = "pi/180.*%s" % alpha
            delta = "pi/180.*%s" % delta
        df.add_virtual_column(zname, "{distance} * (cos({delta}) * cos({delta_gp}) * cos({alpha} - {alpha_gp}) + sin({delta}) * sin({delta_gp}))".format(**locals()))
        df.add_virtual_column(xname, "{distance} * (cos({delta}) * sin({alpha} - {alpha_gp}))".format(**locals()))
        df.add_virtual_column(yname, "{distance} * (sin({delta}) * cos({delta_gp}) - cos({delta}) * sin({delta_gp}) * cos({alpha} - {alpha_gp}))".format(**locals()))
        return df
        # self.write_virtual_meta()

    def lbrvrpm2vcartesian(self, long_in="l", lat_in="b", distance="distance", pm_long="pm_l", pm_lat="pm_b",
                                                        vr="vr", vx="vx", vy="vy", vz="vz",
                                                        center_v=(0, 0, 0),
                                                        propagate_uncertainties=False, radians=False,
                                                        inplace=False):
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
        df = self.df if inplace else self.df.copy()
        k = 4.74057
        a, d, distance = df._expr(long_in, lat_in, distance)
        pm_long, pm_lat, vr = df._expr(pm_long, pm_lat, vr)
        if not radians:
            a = a * np.pi/180
            d = d * np.pi/180
        A = [[np.cos(a)*np.cos(d), -np.sin(a), -np.cos(a)*np.sin(d)],
            [np.sin(a)*np.cos(d),  np.cos(a), -np.sin(a)*np.sin(d)],
            [np.sin(d), d*0, np.cos(d)]]
        df.add_virtual_columns_matrix3d(vr, k * pm_long * distance, k * pm_lat * distance, vx, vy, vz, A, translation=center_v)
        if propagate_uncertainties:
            df.propagate_uncertainties([df[vx], df[vy], df[vz]])
        return df

    def velocity_cartesian2pmvr(self, x="x", y="y", z="z", vx="vx", vy="vy", vz="vz", vr="vr", pm_long="pm_long", pm_lat="pm_lat", distance=None, inplace=False):
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
        df = self.df if inplace else self.df.copy()
        if distance is None:
            distance = "sqrt({x}**2+{y}**2+{z}**2)".format(**locals())
        k = 4.74057
        df.add_variable("k", k, overwrite=False)
        df.add_virtual_column(vr, "({x}*{vx}+{y}*{vy}+{z}*{vz})/{distance}".format(**locals()))
        df.add_virtual_column(pm_long, "-({vx}*{y}-{x}*{vy})/sqrt({x}**2+{y}**2)/{distance}/k".format(**locals()))
        df.add_virtual_column(pm_lat, "-({z}*({x}*{vx}+{y}*{vy}) - ({x}**2+{y}**2)*{vz})/( ({x}**2+{y}**2+{z}**2) * sqrt({x}**2+{y}**2) )/k".format(**locals()))
        return df

    def parallax2distance(self, parallax="parallax", distance_name="distance", parallax_uncertainty=None, uncertainty_postfix="_uncertainty", inplace=False):
        """Convert parallax to distance (i.e. 1/parallax)

        :param parallax: expression for the parallax, e.g. "parallax"
        :param distance_name: name for the virtual column of the distance, e.g. "distance"
        :param parallax_uncertainty: expression for the uncertainty on the parallax, e.g. "parallax_error"
        :param uncertainty_postfix: distance_name + uncertainty_postfix is the name for the virtual column, e.g. "distance_uncertainty" by default
        :return:
        """
        df = self.df if inplace else self.df.copy()
        import astropy.units
        unit = df.unit(parallax)
        # if unit:
        #   convert = unit.to(astropy.units.mas)
        #   distance_expression = "%f/(%s)" % (convert, parallax)
        # else:
        distance_expression = "1/%s" % (parallax)
        df.ucds[distance_name] = "pos.distance"
        df.descriptions[distance_name] = "Derived from parallax (%s)" % parallax
        if unit:
            if unit == astropy.units.milliarcsecond:
                df.units[distance_name] = astropy.units.kpc
            if unit == astropy.units.arcsecond:
                df.units[distance_name] = astropy.units.parsec
        df.add_virtual_column(distance_name, distance_expression)
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
            df.add_virtual_column(name, distance_uncertainty_expression)
            df.descriptions[name] = "Uncertainty on parallax (%s)" % parallax
            df.ucds[name] = "stat.error;pos.distance"
        return df



