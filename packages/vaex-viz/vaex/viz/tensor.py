import vaex
import numpy as np


def plot2d_tensor(self, x, y, vx, vy, shape=16, limits=None, delay=None, show=False, normalize=False, selection=None,
                  facecolor='green', alpha=0.5, edgecolor='black', scale=1., min_count=0):
    import matplotlib.pyplot as plt
    shape = vaex.dataset._expand_shape(shape, 2)

    @vaex.delayed
    def on_cov(limits, count, cov):
        # cov[:,:,0,0] = 1
        # cov[:,:,1,1] = 2.1
        # cov[:,:,0,1] = cov[:,:,1,0] = (cov[:,:,1,1] * cov[:,:,0,0]) **0.5* -0.8
        if normalize:
            length = (cov[:, :, 0, 0] + cov[:, :, 1, 1])
            with np.errstate(divide='ignore', invalid='ignore'):
                cov = (cov.T / length.T).T
        x_centers = self.bin_centers(x, limits[0], shape=shape[0])
        y_centers = self.bin_centers(y, limits[1], shape=shape[1])
        X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')
        X = X.flatten()
        Y = Y.flatten()
        count = count.flatten()
        cov = cov.reshape((-1,) + cov.shape[-2:])
        axes = plt.gca()
        fig = plt.gcf()
        width, height = fig.canvas.get_width_height()
        max_size = min(width, height)
        max_length = (np.nanmax(cov[:, 0, 0] + cov[:, 1, 1]))**0.5
        scaling_x = 1 / max_length * width / shape[0]  # (width*shape[0])
        scaling_y = 1 / max_length * height / shape[1]  # (height*shape[1])
        scaling = min(scaling_x, scaling_y)

        for i in range(len(X)):
            if not np.all(np.isfinite(cov[i])):
                continue
            if count[i] < min_count:
                continue
            eigen_values, eigen_vectors = np.linalg.eig(cov[i])
            indices = np.argsort(eigen_values)[::-1]
            v1 = eigen_vectors[:, indices[0]]  # largest eigen vector
            scale_dispersion = 1.
            device_width = (np.sqrt(np.max(eigen_values))) * scaling * scale
            device_height = (np.sqrt(np.min(eigen_values))) * scaling * scale
            varx = cov[i, 0, 0]
            vary = cov[i, 1, 1]
            angle = np.arctan2(v1[1], v1[0])
            e = ellipse(xy=(X[i], Y[i]), width=device_width, height=device_height, angle=np.degrees(angle),
                        scale=scale_dispersion,
                        alpha=alpha, facecolor=facecolor, edgecolor=edgecolor)  # rand()*360

            axes.add_artist(e)

        if show:
            plt.show()
        return

    @vaex.delayed
    def on_limits(limits):
        # we add them to really count, i.e. if one of them is missing, it won't be counted
        count = self.count(vx + vy, binby=['x', 'y'], limits=limits, shape=shape, selection=selection, delay=True)
        cov = self.cov([vx, vy], binby=['x', 'y'], limits=limits, shape=shape, selection=selection, delay=True)
        return on_cov(limits, count, cov)

    task = on_limits(self.limits([x, y], limits, selection=selection, delay=True))
    return self._delay(self._use_delay(delay), task)


def ellipse(*args, **kwargs):
    # for import performance reasons we don't import it globally
    import matplotlib.artist as artist
    import matplotlib.transforms as transforms
    import matplotlib.patches as patches
    from matplotlib.path import Path

    class DispersionEllipse(patches.Patch):
        """
        This ellipse has it's center in user coordinates, and the width and height in device coordinates
        such that is is not deformed
        """

        def __str__(self):
            return "DispersionEllipse(%s,%s;%sx%s)" % (self.center[0], self.center[1],
                                                       self.width, self.height)

        # @docstring.dedent_interpd
        def __init__(self, xy, width, height, scale=1.0, angle=0.0, **kwargs):
            """
            *xy*
              center of ellipse

            *width*
              total length (diameter) of horizontal axis

            *height*
              total length (diameter) of vertical axis

            *angle*
              rotation in degrees (anti-clockwise)

            Valid kwargs are:
            %(Patch)s
            """
            patches.Patch.__init__(self, **kwargs)

            self.center = xy
            self.width, self.height = width, height
            self.scale = scale
            self.angle = angle
            self._path = Path.unit_circle()
            # Note: This cannot be calculated until this is added to an Axes
            self._patch_transform = transforms.IdentityTransform()

        def _recompute_transform(self):
            """NOTE: This cannot be called until after this has been added
                     to an Axes, otherwise unit conversion will fail. This
                     maxes it very important to call the accessor method and
                     not directly access the transformation member variable.
            """
            center = (self.convert_xunits(self.center[0]),
                      self.convert_yunits(self.center[1]))
            width = self.width  # self.convert_xunits(self.width)
            height = self.height  # self.convert_yunits(self.height)
            trans = artist.Artist.get_transform(self)
            self._patch_transform = transforms.Affine2D() \
                .scale(width * 0.5 * self.scale, height * 0.5 * self.scale) \
                .rotate_deg(self.angle) \
                .translate(*trans.transform(center))

        def get_path(self):
            """
            Return the vertices of the rectangle
            """
            return self._path

        def get_transform(self):
            """
            Return the :class:`~matplotlib.transforms.Transform` applied
            to the :class:`Patch`.
            """
            return self.get_patch_transform()

        def get_patch_transform(self):
            self._recompute_transform()
            return self._patch_transform

        def contains(self, ev):
            if ev.x is None or ev.y is None:
                return False, {}
            x, y = self.get_transform().inverted().transform_point((ev.x, ev.y))
            return (x * x + y * y) <= 1.0, {}
    return DispersionEllipse(*args, **kwargs)
