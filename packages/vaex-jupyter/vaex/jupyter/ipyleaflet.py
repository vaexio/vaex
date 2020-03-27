import ipyleaflet as ll
import traitlets
import ipywidgets as widgets

import vaex.image


class IpyleafletImage(traitlets.HasTraits):
    x_min = traitlets.CFloat()
    x_max = traitlets.CFloat()
    y_min = traitlets.CFloat(None, allow_none=True)
    y_max = traitlets.CFloat(None, allow_none=True)
    x_label = traitlets.Unicode()
    y_label = traitlets.Unicode()
    tool = traitlets.Unicode(None, allow_none=True)

    def __init__(self, output, presenter, map=None, zoom=12, **kwargs):
        super().__init__(**kwargs)
        self.output = output
        self.presenter = presenter
        self.map = map
        self._zoom = zoom
        self.last_image_layer = None

        center = self.x_min + (self.x_max - self.x_min) / 2, self.y_min + (self.y_max - self.y_min) / 2
        center = center[1], center[0]
        self.map = ll.Map(center=center, zoom=self._zoom)

        widgets.dlink((self.map, 'west'), (self, 'x_min'))
        widgets.dlink((self.map, 'east'), (self, 'x_max'))
        widgets.dlink((self.map, 'north'), (self, 'y_min'))
        widgets.dlink((self.map, 'south'), (self, 'y_max'))

        self.widget = self.map

    def set_rgb_image(self, rgb_image):
        with self.output:
            if self.last_image_layer:
                self.map.remove_layer(self.last_image_layer)
            url = vaex.image.rgba_to_url(rgb_image[::-1, ::].copy())
            image = ll.ImageOverlay(url=url, bounds=list(self.map.bounds))
            self.map.add_layer(image)
            self.last_image_layer = image
