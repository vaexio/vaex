import vaex
import numpy as np
from .utils import _ensure_strings_from_expressions, _ensure_string_from_expression, as_flat_array
from .dataframe import docsubst


class DataFrameAccessorGeo(object):
    """Geometry/geographic helper methods

    Example:

    >>> df_xyz = df.geo.spherical2cartesian(df.longitude, df.latitude, df.distance)
    >>> df_xyz.x.mean()

    """
    def __init__(self, df):
        self.df = df

    @docsubst
    def spherical2cartesian(self, alpha, delta, distance, xname="x", yname="y", zname="z",
                            propagate_uncertainties=False,
                            center=[0, 0, 0], radians=False, inplace=False):
        """Convert spherical to cartesian coordinates.

        :param alpha:
        :param delta: polar angle, ranging from the -90 (south pole) to 90 (north pole)
        :param distance: radial distance, determines the units of x, y and z
        :param xname:
        :param yname:
        :param zname:
        :param propagate_uncertainties: {propagate_uncertainties}
        :param center:
        :param radians:
        :return: New dataframe (in inplace is False) with new x,y,z columns
        """
        df = self.df if inplace else self.df.copy()
        alpha = df._expr(alpha)
        delta = df._expr(delta)
        distance = df._expr(distance)
        if not radians:
            df.add_variable('pi', np.pi)
            alpha = alpha * df._expr('pi')/180
            delta = delta * df._expr('pi')/180

        # TODO: use sth like .optimize by default to get rid of the +0 ?
        if center[0]:
            df[xname] = np.cos(alpha) * np.cos(delta) * distance + center[0]
        else:
            df[xname] = np.cos(alpha) * np.cos(delta) * distance
        if center[1]:
            df[yname] = np.sin(alpha) * np.cos(delta) * distance + center[1]
        else:
            df[yname] = np.sin(alpha) * np.cos(delta) * distance
        if center[2]:
            df[zname] = np.sin(delta) * distance + center[2]
        else:
            df[zname] = np.sin(delta) * distance
        if propagate_uncertainties:
            df.propagate_uncertainties([df[xname], df[yname], df[zname]])
        return df

    def cartesian2spherical(self, x="x", y="y", z="z", alpha="l", delta="b", distance="distance", radians=False, center=None, center_name="solar_position", inplace=False):
        """Convert cartesian to spherical coordinates.



        :param x:
        :param y:
        :param z:
        :param alpha:
        :param delta: name for polar angle, ranges from -90 to 90 (or -pi to pi when radians is True).
        :param distance:
        :param radians:
        :param center:
        :param center_name:
        :return:
        """
        df = self.df if inplace else self.df.copy()
        df.add_variable('pi', np.pi)
        transform = "" if radians else "*180./pi"

        if center is not None:
            df.add_variable(center_name, center)
        if center is not None and center[0] != 0:
            x = "({x} - {center_name}[0])".format(**locals())
        if center is not None and center[1] != 0:
            y = "({y} - {center_name}[1])".format(**locals())
        if center is not None and center[2] != 0:
            z = "({z} - {center_name}[2])".format(**locals())
        df.add_virtual_column(distance, "sqrt({x}**2 + {y}**2 + {z}**2)".format(**locals()))
        # df.add_virtual_column(alpha, "((arctan2({y}, {x}) + 2*pi) % (2*pi)){transform}".format(**locals()))
        df.add_virtual_column(alpha, "arctan2({y}, {x}){transform}".format(**locals()))
        df.add_virtual_column(delta, "(-arccos({z}/{distance})+pi/2){transform}".format(**locals()))
        return df
    
    def cartesian_to_polar(self, x="x", y="y", radius_out="r_polar", azimuth_out="phi_polar",
                                               propagate_uncertainties=False,
                                               radians=False, inplace=False):
        """Convert cartesian to polar coordinates

        :param x: expression for x
        :param y: expression for y
        :param radius_out: name for the virtual column for the radius
        :param azimuth_out: name for the virtual column for the azimuth angle
        :param propagate_uncertainties: {propagate_uncertainties}
        :param radians: if True, azimuth is in radians, defaults to degrees
        :return:
        """
        df = self.df if inplace else self.df.copy()
        x = df._expr(x)
        y = df._expr(y)
        if radians:
            to_degrees = ""
        else:
            df.add_variable('pi', np.pi)
            to_degrees = "*180/pi"
        r = np.sqrt(x**2 + y**2)
        df[radius_out] = r
        phi = np.arctan2(y, x)
        if not radians:
            phi = phi * 180/np.pi
        df[azimuth_out] = phi
        if propagate_uncertainties:
            df.propagate_uncertainties([df[radius_out], df[azimuth_out]])
        return df

    @docsubst
    def velocity_polar2cartesian(self, x='x', y='y', azimuth=None, vr='vr_polar', vazimuth='vphi_polar', vx_out='vx', vy_out='vy', propagate_uncertainties=False, inplace=False):
        """ Convert cylindrical polar velocities to Cartesian.

        :param x:
        :param y:
        :param azimuth: Optional expression for the azimuth in degrees , may lead to a better performance when given.
        :param vr:
        :param vazimuth:
        :param vx_out:
        :param vy_out:
        :param propagate_uncertainties: {propagate_uncertainties}
        """
        df = self.df if inplace else self.df.copy()
        x = df._expr(x)
        y = df._expr(y)
        vr = df._expr(vr)
        vazimuth = df._expr(vazimuth)
        if azimuth is not None:
            azimuth = df._expr(azimuth)
            azimuth = np.deg2rad(azimuth)
        else:
            azimuth = np.arctan2(y, x)
        azimuth = df._expr(azimuth)
        df[vx_out] = vr * np.cos(azimuth) - vazimuth * np.sin(azimuth)
        df[vy_out] = vr * np.sin(azimuth) + vazimuth * np.cos(azimuth)
        if propagate_uncertainties:
            df.propagate_uncertainties([df[vx_out], df[vy_out]])
        return df

    @docsubst
    def velocity_cartesian2polar(self, x="x", y="y", vx="vx", radius_polar=None, vy="vy", vr_out="vr_polar", vazimuth_out="vphi_polar",
                                                          propagate_uncertainties=False, inplace=False):
        """Convert cartesian to polar velocities.

        :param x:
        :param y:
        :param vx:
        :param radius_polar: Optional expression for the radius, may lead to a better performance when given.
        :param vy:
        :param vr_out:
        :param vazimuth_out:
        :param propagate_uncertainties: {propagate_uncertainties}
        :return:
        """
        df = self.df if inplace else self.df.copy()
        x = df._expr(x)
        y = df._expr(y)
        vx = df._expr(vx)
        vy = df._expr(vy)
        if radius_polar is None:
            radius_polar = np.sqrt(x**2 + y**2)
        radius_polar = df._expr(radius_polar)
        df[vr_out]       = (x*vx + y*vy) / radius_polar
        df[vazimuth_out] = (x*vy - y*vx) / radius_polar
        if propagate_uncertainties:
            df.propagate_uncertainties([df[vr_out], df[vazimuth_out]])
        return df

    def velocity_cartesian2spherical(self, x="x", y="y", z="z", vx="vx", vy="vy", vz="vz", vr="vr", vlong="vlong", vlat="vlat", distance=None, inplace=False):
        """Convert velocities from a cartesian to a spherical coordinate system

        TODO: uncertainty propagation

        :param x: name of x column (input)
        :param y:         y
        :param z:         z
        :param vx:       vx
        :param vy:       vy
        :param vz:       vz
        :param vr: name of the column for the radial velocity in the r direction (output)
        :param vlong: name of the column for the velocity component in the longitude direction  (output)
        :param vlat: name of the column for the velocity component in the latitude direction, positive points to the north pole (output)
        :param distance: Expression for distance, if not given defaults to sqrt(x**2+y**2+z**2), but if this column already exists, passing this expression may lead to a better performance
        :return:
        """
        # see http://www.astrosurf.com/jephem/library/li110spherCart_en.htm
        df = self.df if inplace else self.df.copy()
        if distance is None:
            distance = "sqrt({x}**2+{y}**2+{z}**2)".format(**locals())
        df.add_virtual_column(vr, "({x}*{vx}+{y}*{vy}+{z}*{vz})/{distance}".format(**locals()))
        df.add_virtual_column(vlong, "-({vx}*{y}-{x}*{vy})/sqrt({x}**2+{y}**2)".format(**locals()))
        df.add_virtual_column(vlat, "-({z}*({x}*{vx}+{y}*{vy}) - ({x}**2+{y}**2)*{vz})/( {distance}*sqrt({x}**2+{y}**2) )".format(**locals()))
        return df

    def project_aitoff(self, alpha, delta, x, y, radians=True, inplace=False):
        """Add aitoff (https://en.wikipedia.org/wiki/Aitoff_projection) projection

        :param alpha: azimuth angle
        :param delta: polar angle
        :param x: output name for x coordinate
        :param y: output name for y coordinate
        :param radians: input and output in radians (True), or degrees (False)
        :return:
        """
        df = self.df if inplace else self.df.copy()
        transform = "" if radians else "*pi/180."
        aitoff_alpha = "__aitoff_alpha_%s_%s" % (alpha, delta)
        # sanatize
        aitoff_alpha = re.sub("[^a-zA-Z_]", "_", aitoff_alpha)

        df.add_virtual_column(aitoff_alpha, "arccos(cos({delta}{transform})*cos({alpha}{transform}/2))".format(**locals()))
        df.add_virtual_column(x, "2*cos({delta}{transform})*sin({alpha}{transform}/2)/sinc({aitoff_alpha}/pi)/pi".format(**locals()))
        df.add_virtual_column(y, "sin({delta}{transform})/sinc({aitoff_alpha}/pi)/pi".format(**locals()))
        return df

    def project_gnomic(self, alpha, delta, alpha0=0, delta0=0, x="x", y="y", radians=False, postfix="", inplace=False):
        """Adds a gnomic projection to the DataFrame"""
        df = self.df if inplace else self.df.copy()
        df.add_variable('pi', np.pi)
        if not radians:
            alpha = "pi/180.*%s" % alpha
            delta = "pi/180.*%s" % delta
            alpha0 = alpha0 * np.pi / 180
            delta0 = delta0 * np.pi / 180
        transform = "" if radians else "*180./pi"
        # aliases
        ra = alpha
        dec = delta
        ra_center = alpha0
        dec_center = delta0
        gnomic_denominator = 'sin({dec_center}) * tan({dec}) + cos({dec_center}) * cos({ra} - {ra_center})'.format(**locals())
        denominator_name = 'gnomic_denominator' + postfix
        xi = 'sin({ra} - {ra_center})/{denominator_name}{transform}'.format(**locals())
        eta = '(cos({dec_center}) * tan({dec}) - sin({dec_center}) * cos({ra} - {ra_center}))/{denominator_name}{transform}'.format(**locals())
        df.add_virtual_column(denominator_name, gnomic_denominator)
        df.add_virtual_column(x, xi)
        df.add_virtual_column(y, eta)
        return df

    def rotation_2d(self, x, y, xnew, ynew, angle_degrees, propagate_uncertainties=False, inplace=False):
        """Rotation in 2d.

        :param str x: Name/expression of x column
        :param str y: idem for y
        :param str xnew: name of transformed x column
        :param str ynew:
        :param float angle_degrees: rotation in degrees, anti clockwise
        :return:
        """
        df = self.df if inplace else self.df.copy()
        x = _ensure_string_from_expression(x)
        y = _ensure_string_from_expression(y)
        theta = np.radians(angle_degrees)
        matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        m = matrix_name = x + "_" + y + "_rot"
        for i in range(2):
            for j in range(2):
                df.set_variable(matrix_name + "_%d%d" % (i, j), matrix[i, j].item())
        df[xnew] = df._expr("{m}_00 * {x} + {m}_01 * {y}".format(**locals()))
        df[ynew] = df._expr("{m}_10 * {x} + {m}_11 * {y}".format(**locals()))
        if propagate_uncertainties:
            df.propagate_uncertainties([df[xnew], df[ynew]])
        return df

    def bearing(self, lon1, lat1, lon2, lat2, bearing="bearing", inplace=False):
        """Calculates a bearing, based on http://www.movable-type.co.uk/scripts/latlong.html"""
        df = self.df if inplace else self.df.copy()
        df.add_variable('pi', np.pi)
        lon1 = "(pickup_longitude * pi / 180)"
        lon2 = "(dropoff_longitude * pi / 180)"
        lat1 = "(pickup_latitude * pi / 180)"
        lat2 = "(dropoff_latitude * pi / 180)"
        p1 = lat1
        p2 = lat2
        l1 = lon1
        l2 = lon2
        # from http://www.movable-type.co.uk/scripts/latlong.html
        expr = "arctan2(sin({l2}-{l1}) * cos({p2}), cos({p1})*sin({p2}) - sin({p1})*cos({p2})*cos({l2}-{l1}))" \
            .format(**locals())
        df.add_virtual_column(bearing, expr)
        return df

    @docsubst
    def inside_which_polygons(self, x, y, pxss, pyss=None, any=True):
        """Find in which set of polygons (0 based index) a point resides.

        If any=True, it will be the first matching polygon set index, if any=False, it will
        be the first index that matches all polygons in the set.

        >>> import vaex
        >>> import numpy as np
        >>> df = vaex.from_arrays(x=[1, 2, 3], y=[2, 3, 4])
        >>> px = np.array([1.5, 2.5, 2.5, 1.5])
        >>> py = np.array([2.5, 2.5, 3.5, 3.5])
        >>> polygonA = [px, py]
        >>> polygonB = [px + 1, py + 1]
        >>> pxs = [[polygonA, polygonB], [polygonA]]
        >>> df['polygon_index'] = df.geo.inside_which_polygons(df.x, df.y, pxs, any=True)
        >>> df
        #    x    y  polygon_index
        0    1    2  --
        1    2    3  0
        2    3    4  0
        >>> df['polygon_index'] = df.geo.inside_which_polygons(df.x, df.y, pxs, any=False)
        >>> df
        #    x    y  polygon_index
        0    1    2  --
        1    2    3  1
        2    3    4  --

        :param x: {expression_one}
        :param y: {expression_one}
        :param px: list of N ndarrays with x coordinates for the polygon, N is the number of polygons
        :param px: list of N ndarrays with y coordinates for the polygon, if None, the shape of the ndarrays
                of the last dimention of the x arrays should be 2 (i.e. have the x and y coordinates)
        :param any: test if point it in any polygon (logically or), or all polygons (logically and)
        :return: Expression, 0 based index to which polygon the point belongs (or missing/masked value)
        """
        if pyss is None:
            list_of_polygons = pxss
            # polygon is a list of vertices shape: (N, 2)
            list_of_polygons = [[as_flat_array(np.asarray(polygon), np.float64) for polygon in polygons]
                                    for polygons in list_of_polygons]
            pxss = [[as_flat_array(polygon[0]) for polygon in polygons] for polygons in list_of_polygons]
            pyss = [[as_flat_array(polygon[1]) for polygon in polygons] for polygons in list_of_polygons]
        else:
            pxss = [[as_flat_array(np.asarray(pxs), np.float64) for pxs in polygons_x]
                                    for polygons_x in pxss]
            pyss = [[as_flat_array(np.asarray(pys), np.float64) for pys in polygons_y]
                                    for polygons_y in pyss]
        N = len(pxss)

        # iterate over polygon sets
        meanxss = []
        meanyss = []
        radiii = []
        for region_index in range(N):
            pxs = pxss[region_index]
            pys = pyss[region_index]
            meanxs = []
            meanys = []
            radii = []

            for polygons_index in range(len(pxs)):
                meanx = pxs[polygons_index].mean()
                meany = pys[polygons_index].mean()
                radius = np.sqrt((meanx - pxs[polygons_index])**2 + (meany - pys[polygons_index])**2).max()
                meanxs.append(meanx)
                meanys.append(meany)
                radii.append(radius)
            meanxss.append(meanxs)
            meanyss.append(meanys)
            radiii.append(radii)
        # this method is simply a wrapper around the true function, which also wants to have the
        # radii, which are precalculated for performance reasons
        return self.df.func.geo_inside_which_polygons(x, y, pxss, pyss, meanxss, meanyss, radiii, any=any)


@vaex.register_function(df_accessor=DataFrameAccessorGeo, name='inside_polygon')
def geo_inside_polygon(x, y, px, py):
    """Test if points defined by x and y are inside the polygon px, py

    Example:

    >>> import vaex
    >>> import numpy as np
    >>> df = vaex.from_arrays(x=[1, 2, 3], y=[2, 3, 4])
    >>> px = np.array([1.5, 2.5, 2.5, 1.5])
    >>> py = np.array([2.5, 2.5, 3.5, 3.5])
    >>> df['inside'] = df.geo.inside_polygon(df.x, df.y, px, py)
    >>> df
    #    x    y  inside
    0    1    2  False
    1    2    3  True
    2    3    4  False

    :param x: {expression_one}
    :param y: {expression_one}
    :param px: list of x coordinates for the polygon
    :param px: list of y coordinates for the polygon
    :return: Expression, which is true if point is inside, else false.
    """
    x = as_flat_array(x, np.float64)
    y = as_flat_array(y, np.float64)
    px = as_flat_array(np.asarray(px), np.float64)
    py = as_flat_array(np.asarray(py), np.float64)
    mask = np.zeros(len(x), dtype=np.bool)
    meanx = px.mean()
    meany = py.mean()
    radius = np.sqrt((meanx - px)**2 + (meany - py)**2).max()
    vaex.vaexfast.pnpoly(px, py, x, y, mask, meanx, meany, radius)
    return mask


@docsubst
@vaex.register_function(df_accessor=DataFrameAccessorGeo, name='inside_polygons')
def geo_inside_polygons(x, y, pxs, pys, any=True):
    """Test if points defined by x and y are inside all or any of the the polygons px, py

    Example:

    >>> import vaex
    >>> import numpy as np
    >>> df = vaex.from_arrays(x=[1, 2, 3], y=[2, 3, 4])
    >>> px = np.array([1.5, 2.5, 2.5, 1.5])
    >>> py = np.array([2.5, 2.5, 3.5, 3.5])
    >>> df['inside'] = df.geo.inside_polygons(df.x, df.y, [px, px + 1], [py, py + 1], any=True)
    >>> df
    #    x    y  inside
    0    1    2  False
    1    2    3  True
    2    3    4  True

    :param x: {expression_one}
    :param y: {expression_one}
    :param pxs: list of N ndarrays with x coordinates for the polygon, N is the number of polygons
    :param pxs: list of N ndarrays with y coordinates for the polygon
    :param any: return true if in any polygon, or all polygons
    :return: Expression , which is true if point is inside, else false.
    """
    x = as_flat_array(x, np.float64)
    y = as_flat_array(y, np.float64)
    mask = np.zeros(len(x), dtype=np.bool)

    N = len(pxs)
    submask = np.zeros(len(x), dtype=np.bool)
    if N > 0:
        for i in range(0, N):
            px = as_flat_array(pxs[i], np.float64)
            py = as_flat_array(pys[i], np.float64)
            meanx = px.mean()
            meany = py.mean()
            radius = np.sqrt((meanx - px)**2 + (meany - py)**2).max()
            vaex.vaexfast.pnpoly(px, py, x, y, submask if i > 0 else mask, meanx, meany, radius)
            if i > 0:
                if any:
                    mask = mask | submask
                else:
                    mask = mask & submask
    return mask


@docsubst
@vaex.register_function(df_accessor=DataFrameAccessorGeo, name='inside_which_polygon')
def geo_inside_which_polygon(x, y, pxs, pys):
    """Find in which polygon (0 based index) a point resides

    Example:

    >>> import vaex
    >>> import numpy as np
    >>> df = vaex.from_arrays(x=[1, 2, 3], y=[2, 3, 4])
    >>> px = np.array([1.5, 2.5, 2.5, 1.5])
    >>> py = np.array([2.5, 2.5, 3.5, 3.5])
    >>> df['polygon_index'] = df.geo.inside_which_polygon(df.x, df.y, [px, px + 1], [py, py + 1])
    >>> df
    #    x    y  polygon_index
    0    1    2  --
    1    2    3  0
    2    3    4  1

    :param x: {expression_one}
    :param y: {expression_one}
    :param px: list of N ndarrays with x coordinates for the polygon, N is the number of polygons
    :param px: list of N ndarrays with y coordinates for the polygon
    :return: Expression, 0 based index to which polygon the point belongs (or missing/masked value)
    """
    x = as_flat_array(x, np.float64)
    y = as_flat_array(y, np.float64)
    pxs = as_flat_array(np.asarray(pxs), np.float64)
    pys = as_flat_array(np.asarray(pys), np.float64)
    polygon_indices = np.zeros(len(x), dtype=np.int32)
    polygon_mask = np.ones(len(x), dtype=np.bool)
    polygon_indices = np.ma.array(polygon_indices, mask=polygon_mask)
    indices = np.arange(len(x), dtype=np.uint32)
    inside_mask = np.zeros(len(x), dtype=np.bool)

    meanx = pxs.mean()
    meany = pys.mean()
    radius = np.sqrt((meanx - pxs)**2 + (meany - pys)**2).max()
    N = len(pxs)
    # mask = mask | submask
    for i in range(N):
        vaex.vaexfast.pnpoly(pxs[i], pys[i], x, y, inside_mask, meanx, meany, radius)
        # mark all points that are inside with the index of the polygon
        polygon_indices.data[indices[inside_mask]] = i
        polygon_indices.mask[indices[inside_mask]] = False
        # now remove all values that found a matching polygon
        x = x[~inside_mask]
        y = y[~inside_mask]
        # and keep track of where they point to in the polygon_indices array
        indices = indices[~inside_mask]
        # trim mask
        inside_mask = inside_mask[:len(x)]
        if len(x) == 0:
            break
    return polygon_indices


@docsubst
@vaex.register_function()
def geo_inside_which_polygons(x, y, pxss, pyss, meanxss, meanyss, radiii, any):
    # real implementation of geo.inside_which_polygon
    x = as_flat_array(x, np.float64)
    y = as_flat_array(y, np.float64)
    polygon_indices = np.zeros(len(x), dtype=np.int32)
    polygon_mask = np.ones(len(x), dtype=np.bool)
    polygon_indices = np.ma.array(polygon_indices, mask=polygon_mask)
    indices = np.arange(len(x), dtype=np.uint32)
    inside_mask = np.zeros(len(x), dtype=np.bool)
    N = len(pxss)
    for i in range(N):
        pxs = pxss[i]
        pys = pyss[i]
        meanxs = meanxss[i]
        meanys = meanyss[i]
        radii = radiii[i]
        M = len(pxs)
        if M > 0:
            vaex.vaexfast.pnpoly(pxs[0], pys[0], x, y, inside_mask, meanxs[0], meanys[0], radii[0])
        if M > 1:
            inside_sub_mask = np.zeros(len(x), dtype=np.bool)
            for j in range(1, M):
                vaex.vaexfast.pnpoly(pxs[j], pys[j], x, y, inside_sub_mask, meanxs[j], meanys[j], radii[j])
                if any:
                    inside_mask = inside_mask | inside_sub_mask
                else:
                    inside_mask = inside_mask & inside_sub_mask

        # mark all points that are inside with the index of the polygon
        polygon_indices.data[indices[inside_mask]] = i
        polygon_indices.mask[indices[inside_mask]] = False
        # no remove all values that found a matching polygon
        x = x[~inside_mask]
        y = y[~inside_mask]
        # and keep track of where they point to in the polygon_indices array
        indices = indices[~inside_mask]
        # trim mask
        inside_mask = inside_mask[:len(x)]
        if len(x) == 0:
            break
    return polygon_indices
