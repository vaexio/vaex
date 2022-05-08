import vaex
import numpy as np


def plot2d_vector(self, x, y, vx, vy, shape=16, limits=None, delay=None, show=False, normalize=False,
                  selection=None, min_count=0, **kwargs):
    import matplotlib.pyplot as plt
    shape = vaex.dataset._expand_shape(shape, 2)

    @vaex.delayed
    def on_means(limits, count, mean_vx, mean_vy):
        if normalize:
            length = np.sqrt(mean_vx**2 + mean_vy**2)
            mean_vx = mean_vx / length
            mean_vy = mean_vy / length
        x_centers = self.bin_centers(x, limits[0], shape=shape[0])
        y_centers = self.bin_centers(y, limits[1], shape=shape[1])
        Y, X = np.meshgrid(y_centers, x_centers)  # , indexing='ij')
        count = count.flatten()
        mask = count >= min_count
        kwargs['alpha'] = kwargs.get('alpha', 0.7)
        plt.quiver(X.flatten()[mask],
                   Y.flatten()[mask],
                   mean_vx.flatten()[mask],
                   mean_vy.flatten()[mask],
                   **kwargs)
        if show:
            plt.show()
        return

    @vaex.delayed
    def on_limits(limits):
        # we add them to really count, i.e. if one of them is missing, it won't be counted
        count = self.count(vx + vy, binby=[x, y], limits=limits, shape=shape, selection=selection, delay=True)
        mean_vx = self.mean(vx, binby=[x, y], limits=limits, shape=shape, selection=selection, delay=True)
        mean_vy = self.mean(vy, binby=[x, y], limits=limits, shape=shape, selection=selection, delay=True)
        return on_means(limits, count, mean_vx, mean_vy)

    task = on_limits(self.limits([x, y], limits, selection=selection, delay=True))
    return self._delay(self._use_delay(delay), task)
