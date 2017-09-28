import ipyleaflet as ll
import numpy as np
import vaex.image
from .plot import BackendBase
import copy
from .utils import debounced

class IpyleafletBackend(BackendBase):
    def __init__(self, map=None, center=[53.3082834, 6.388399], zoom=12):
        self.map = map
        self._center = center
        self._zoom = zoom
        self.last_image_layer = None

    def create_widget(self, output, plot, dataset, limits):
        self.plot = plot
        self.dataset = dataset
        self.output = output
        self.limits = np.array(limits)[:2].tolist()
        if self.map is None:
            (xmin, xmax), (ymin, ymax)  = limits[:2]
            center = xmin + (xmax - xmin) / 2, ymin + (ymax - ymin)/2
            center = center[1], center[0]
            self.map = ll.Map(center=center, zoom=self._zoom)

        self.map.observe(self._update_limits, "_north")
        self.map.observe(self._update_limits, "_east")
        self.map.observe(self._update_limits, "_south")
        self.map.observe(self._update_limits, "_west")
        #self.map.bounds = self.limits
        #self.limits = self.map.bounds[1], self.map.bounds[0] # np.array(limits).tolist()
        #print(self.map.bounds, self.map.west)
        #print(self.limits)
        self.widget = self.map

    def _update_limits(self, *args):
        with self.output:
            #self._progressbar.cancel()
            limits = copy.deepcopy(self.limits)
            limits[0] = (self.map._west, self.map._east)
            limits[1] = (self.map._north, self.map._south)
            self.limits = limits

    @debounced(0.1, method=True)
    def update_image(self, rgb_image):
        with self.output:
            if self.last_image_layer:
                self.map.remove_layer(self.last_image_layer)
            url = vaex.image.rgba_to_url(rgb_image[::-1,::].copy())
            image = ll.ImageOverlay(url=url, bounds=list(self.map.bounds))
            #print("add ", self.limits, self.map.bounds)
            self.map.add_layer(image)
            self.last_image_layer = image
