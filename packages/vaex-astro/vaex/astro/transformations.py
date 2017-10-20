from vaex.dataset import Dataset
import numpy as np

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
    #if unit:
        #convert = unit.to(astropy.units.mas)
        #   distance_expression = "%f/(%s)" % (convert, parallax)
        #else:
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
                                             cov_matrix_alpha_delta_pma_pmd=None,
                                             covariance_postfix="_covariance",
                                             uncertainty_postfix="_uncertainty",
                                             name_prefix="__proper_motion_eq2gal", radians=False):
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
    :return:
    """
    import kapteyn.celestial as c
    """mu_gb =  mu_dec*(cdec*sdp-sdec*cdp*COS(ras))/cgb $
      - mu_ra*cdp*SIN(ras)/cgb"""
    long_in_original = long_in
    lat_in_original  =  lat_in
    if not radians:
        long_in = "pi/180.*%s" % long_in
        lat_in = "pi/180.*%s" % lat_in
        to_radians = "*pi/180" # used for the derivatives
    else:
        to_radians = ""
    c1 = name_prefix + "_C1"
    c2 = name_prefix + "_C2"
    self.add_variable("right_ascension_galactic_pole", np.radians(192.85).item(), overwrite=False)
    self.add_variable("declination_galactic_pole", np.radians(27.12).item(), overwrite=False)
    self.add_virtual_column(c1, "sin(declination_galactic_pole) * cos({lat_in}) - cos(declination_galactic_pole)*sin({lat_in})*cos({long_in}-right_ascension_galactic_pole)".format(**locals()))
    self.add_virtual_column(c2, "cos(declination_galactic_pole) * sin({long_in}-right_ascension_galactic_pole)".format(**locals()))
    self.add_virtual_column(pm_long_out, "({c1} * {pm_long} + {c2} * {pm_lat})/sqrt({c1}**2+{c2}**2)".format(**locals()))
    self.add_virtual_column(pm_lat_out, "(-{c2} * {pm_long} + {c1} * {pm_lat})/sqrt({c1}**2+{c2}**2)".format(**locals()))
    if cov_matrix_alpha_delta_pma_pmd:
        # function and it's jacobian
        # f(long, lat, pm_long, pm_lat) = [pm_long, pm_lat, c1, c2] = [pm_long, pm_lat, ..., ...]
        J = [ [None, None, "1", None],
              [None, None, None, "1"],
              [                                                    "cos(declination_galactic_pole)*sin({lat_in})*sin({long_in}-right_ascension_galactic_pole){to_radians}",
                 "-sin(declination_galactic_pole) * sin({lat_in}){to_radians} - cos(declination_galactic_pole)*cos({lat_in})*cos({long_in}-right_ascension_galactic_pole){to_radians}",
                 None, None],
              ["cos(declination_galactic_pole)*cos({long_in}-right_ascension_galactic_pole){to_radians}", None, None, None],
          ]

        if cov_matrix_alpha_delta_pma_pmd in ["full", "auto"]:
            names = [long_in_original, lat_in_original, pm_long, pm_lat]
            cov_matrix_alpha_delta_pma_pmd = self._covariance_matrix_guess(names, full=cov_matrix_alpha_delta_pma_pmd=="full")

        cov_matrix_pm_long_pm_lat_c1_c2 = [[""] * 4 for i in range(4)]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        sigma = cov_matrix_alpha_delta_pma_pmd[k][l]
                        if sigma and J[i][k] and J[j][l]:
                            cov_matrix_pm_long_pm_lat_c1_c2[i][j] += "+(%s)*(%s)*(%s)" % (J[i][k], sigma, J[j][l])

        cov_matrix_pml_pmb = [[""] * 2 for i in range(2)]

        # function and it's jacobian
        # f(pm_long, pm_lat, c1, c2) = [pm_l, pm_b] = [..., ...]
        J = [
            [" ({c1}                               )/sqrt({c1}**2+{c2}**2)",
             " (                    {c2}           )/sqrt({c1}**2+{c2}**2)",
             "( {c2} *  {pm_long} - {c1} * {pm_lat})/    ({c1}**2+{c2}**2)**(3./2)*{c2}",
             "(-{c2} *  {pm_long} + {c1} * {pm_lat})/    ({c1}**2+{c2}**2)**(3./2)*{c1}"],
            ["(-{c2}                               )/sqrt({c1}**2+{c2}**2)",
             " (                    {c1}           )/sqrt({c1}**2+{c2}**2)",
             "({c1} * {pm_long} + {c2} * {pm_lat})/      ({c1}**2+{c2}**2)**(3./2)*{c2}",
             "-({c1} * {pm_long} + {c2} * {pm_lat})/     ({c1}**2+{c2}**2)**(3./2)*{c1}"]
        ]
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    for l in range(4):
                        sigma = cov_matrix_pm_long_pm_lat_c1_c2[k][l]
                        if sigma and J[i][k] != "0" and J[j][l] != "0":
                            cov_matrix_pml_pmb[i][j] += "+(%s)*(%s)*(%s)" % (J[i][k], sigma, J[j][l])
        names = [pm_long_out, pm_lat_out]
        #cnames = ["c1", "c2"]
        for i in range(2):
            for j in range(i+1):
                sigma = cov_matrix_pml_pmb[i][j].format(**locals())
                if i != j:
                    self.add_virtual_column(names[i]+"_" + names[j]+covariance_postfix, sigma)
                else:
                    self.add_virtual_column(names[i]+uncertainty_postfix, "sqrt(%s)" % sigma)
                    #sigma = cov_matrix_pm_long_pm_lat_c1_c2[i+2][j+2].format(**locals())
                    #self.add_virtual_column(cnames[i]+uncertainty_postfix, "sqrt(%s)" % sigma)
                    #sigma = cov_matrix_vr_vl_vb[i][j].format(**locals())

@patch
def add_virtual_columns_lbrvr_proper_motion2vcartesian(self, long_in="l", lat_in="b", distance="distance", pm_long="pm_l", pm_lat="pm_b",
                                                       vr="vr", vx="vx", vy="vy", vz="vz",
                                                       cov_matrix_vr_distance_pm_long_pm_lat=None,
                                                       uncertainty_postfix="_uncertainty", covariance_postfix="_covariance",
                                                       name_prefix="__lbvr_proper_motion2vcartesian", center_v=(0,0,0), center_v_name="solar_motion", radians=False):
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
    :param name_prefix:
    :param center_v: Extra motion that should be added, for instance lsr + motion of the sun wrt the galactic restframe
    :param center_v_name:
    :param radians: input and output in radians (True), or degrees (False)
    :return:
    """
    k = 4.74057
    self.add_variable("k", k, overwrite=False)
    A = [["cos({a})*cos({d})", "-sin({a})", "-cos({a})*sin({d})"],
         ["sin({a})*cos({d})",  "cos({a})", "-sin({a})*sin({d})"],
         [         "sin({d})",         "0",           "cos({d})"]]
    a = long_in
    d = lat_in
    if not radians:
        a = "pi/180.*%s" % a
        d = "pi/180.*%s" % d
    for i in range(3):
        for j in range(3):
            A[i][j] = A[i][j].format(**locals())
    if 0: # used for testing
        self.add_virtual_column("vl", "k*{pm_long}*{distance}".format(**locals()))
        self.add_virtual_column("vb", "k* {pm_lat}*{distance}".format(**locals()))
    self.add_virtual_columns_matrix3d(vr, "k*{pm_long}*{distance}".format(**locals()), "k*{pm_lat}*{distance}".format(**locals()), name_prefix +vx, name_prefix +vy, name_prefix +vz, \
                                      A, name_prefix+"_matrix", matrix_is_expression=True)
    self.add_variable(center_v_name, center_v)
    self.add_virtual_column(vx, "%s + %s[0]" % (name_prefix +vx, center_v_name))
    self.add_virtual_column(vy, "%s + %s[1]" % (name_prefix +vy, center_v_name))
    self.add_virtual_column(vz, "%s + %s[2]" % (name_prefix +vz, center_v_name))

    if cov_matrix_vr_distance_pm_long_pm_lat:
        # function and it's jacobian
        # f_obs(vr, distance, pm_long, pm_lat) = [vr, v_long, v_lat] = (vr, k * pm_long * distance, k * pm_lat * distance)
        J = [ ["1", "", "", ""],
             ["", "k * {pm_long}",  "k * {distance}", ""],
             ["", "k * {pm_lat}",                 "", "k * {distance}"]]

        if cov_matrix_vr_distance_pm_long_pm_lat in ["full", "auto"]:
            names = [vr, distance, pm_long, pm_lat]
            cov_matrix_vr_distance_pm_long_pm_lat = self._covariance_matrix_guess(names, full=cov_matrix_vr_distance_pm_long_pm_lat=="full")

        cov_matrix_vr_vl_vb = [[""] * 3 for i in range(3)]
        for i in range(3):
            for j in range(3):
                for k in range(4):
                    for l in range(4):
                        sigma = cov_matrix_vr_distance_pm_long_pm_lat[k][l]
                        if sigma and J[i][k] and J[j][l]:
                            cov_matrix_vr_vl_vb[i][j] += "+(%s)*(%s)*(%s)" % (J[i][k], sigma, J[j][l])

        cov_matrix_vx_vy_vz = [[""] * 3 for i in range(3)]

        # here A is the Jacobian
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        sigma = cov_matrix_vr_vl_vb[k][l]
                        if sigma and A[i][k] != "0" and A[j][l] != "0":
                            cov_matrix_vx_vy_vz[i][j] += "+(%s)*(%s)*(%s)" % (A[i][k], sigma, A[j][l])
        vnames = [vx, vy, vz]
        vrlb_names = ["vr", "vl", "vb"]
        for i in range(3):
            for j in range(i+1):
                sigma = cov_matrix_vx_vy_vz[i][j].format(**locals())
                if i != j:
                    self.add_virtual_column(vnames[i]+"_" + vnames[j]+covariance_postfix, sigma)
                else:
                    self.add_virtual_column(vnames[i]+uncertainty_postfix, "sqrt(%s)" % sigma)
                    #sigma = cov_matrix_vr_vl_vb[i][j].format(**locals())
                    #self.add_virtual_column(vrlb_names[i]+uncertainty_postfix, "sqrt(%s)" % sigma)


        #self.add_virtual_column(vx, x)


@patch
def add_virtual_columns_equatorial_to_galactic_cartesian(self, alpha, delta, distance, xname, yname, zname, radians=True, alpha_gp=np.radians(192.85948), delta_gp=np.radians(27.12825), l_omega=np.radians(32.93192)):
    """From http://arxiv.org/pdf/1306.2945v2.pdf"""
    if not radians:
        alpha = "pi/180.*%s" % alpha
        delta = "pi/180.*%s" % delta
    self.virtual_columns[zname] = "{distance} * (cos({delta}) * cos({delta_gp}) * cos({alpha} - {alpha_gp}) + sin({delta}) * sin({delta_gp}))".format(**locals())
    self.virtual_columns[xname] = "{distance} * (cos({delta}) * sin({alpha} - {alpha_gp}))".format(**locals())
    self.virtual_columns[yname] = "{distance} * (sin({delta}) * cos({delta_gp}) - cos({delta}) * sin({delta_gp}) * cos({alpha} - {alpha_gp}))".format(**locals())
    #self.write_virtual_meta()


@patch
def add_virtual_columns_celestial(self, long_in, lat_in, long_out, lat_out, input=None, output=None, name_prefix="__celestial", radians=False):
    import kapteyn.celestial as c
    input = input if input is not None else c.equatorial
    output = output if output is not None else c.galactic
    matrix = c.skymatrix((input,'j2000',c.fk5), output)[0]
    if not radians:
        long_in = "pi/180.*%s" % long_in
        lat_in = "pi/180.*%s" % lat_in
    x_in = name_prefix+"_in_x"
    y_in = name_prefix+"_in_y"
    z_in = name_prefix+"_in_z"
    x_out = name_prefix+"_out_x"
    y_out = name_prefix+"_out_y"
    z_out = name_prefix+"_out_z"
    self.add_virtual_column(x_in, "cos({long_in})*cos({lat_in})".format(**locals()))
    self.add_virtual_column(y_in, "sin({long_in})*cos({lat_in})".format(**locals()))
    self.add_virtual_column(z_in, "sin({lat_in})".format(**locals()))
    #self.add_virtual_columns_spherical_to_cartesian(long_in, lat_in, None, x_in, y_in, z_in, cov_matrix_alpha_delta=)
    self.add_virtual_columns_matrix3d(x_in, y_in, z_in, x_out, y_out, z_out, \
                                      matrix, name_prefix+"_matrix")
    #long_out_expr = "arctan2({y_out},{x_out})".format(**locals())
    #lat_out_expr = "arctan2({z_out},sqrt({x_out}**2+{y_out}**2))".format(**locals())
    #if not radians:
    #   long_out_expr = "180./pi*%s" % long_out_expr
    #   lat_out_expr = "180./pi*%s" % lat_out_expr
    transform = "" if radians else "*180./pi"
    x = x_out
    y = y_out
    z = z_out
    #self.add_virtual_column(long_out, "((arctan2({y}, {x})+2*pi) % (2*pi)){transform}".format(**locals()))
    self.add_virtual_column(long_out, "arctan2({y}, {x}){transform}".format(**locals()))
    self.add_virtual_column(lat_out, "(-arccos({z}/sqrt({x}**2+{y}**2+{z}**2))+pi/2){transform}".format(**locals()))

#self.add_virtual_column(long_out, long_out_expr)
#self.add_virtual_column(lat_out, lat_out_expr)

@patch
def add_virtual_columns_proper_motion2vperpendicular(self, distance="distance", pm_long="pm_l", pm_lat="pm_b",
                                                       vl="vl", vb="vb",
                                                       cov_matrix_distance_pm_long_pm_lat=None,
                                                       uncertainty_postfix="_uncertainty", covariance_postfix="_covariance",
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
    if cov_matrix_distance_pm_long_pm_lat:
        # function and it's jacobian
        # f_obs(distance, pm_long, pm_lat) = [v_long, v_lat] = (k * pm_long * distance, k * pm_lat * distance)
        J = [["k * {pm_long}",  "k * {distance}", ""],
             ["k * {pm_lat}",                 "", "k * {distance}"]]
        if cov_matrix_distance_pm_long_pm_lat in ["full", "auto"]:
            names = [distance, pm_long, pm_lat]
            cov_matrix_distance_pm_long_pm_lat = self._covariance_matrix_guess(names, full=cov_matrix_distance_pm_long_pm_lat=="full")


        cov_matrix_vl_vb = [[""] * 2 for i in range(2)]
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    for l in range(3):
                        sigma = cov_matrix_distance_pm_long_pm_lat[k][l]
                        if sigma and J[i][k] and J[j][l]:
                            cov_matrix_vl_vb[i][j] += "+(%s)*(%s)*(%s)" % (J[i][k], sigma, J[j][l])

        names = [vl, vb]
        for i in range(2):
            for j in range(i+1):
                sigma = cov_matrix_vl_vb[i][j].format(**locals())
                if i != j:
                    self.add_virtual_column(names[i]+"_" + names[j]+covariance_postfix, sigma)
                else:
                    self.add_virtual_column(names[i]+uncertainty_postfix, "sqrt(%s)" % sigma)





def add_plugin():
    pass  # importing this module already does the job