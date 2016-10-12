# -*- coding: utf-8 -*-
"""
import traitlets
import ipywidgets as widgets

class VolumeRenderingWidget(widgets.DOMWidget):
    _view_name = Unicode('VolumeRenderingView', sync=True)
    level1 = CInt(0, sync=True)

widgets.IntSlider

from IPython.display import HTML
"""


from IPython.display import HTML, display_html, display_javascript
import os
base_path = os.path.dirname(__file__)
#print base_path

import uuid
from base64 import b64encode
import json
try:
	from cStringIO import StringIO
except ImportError:
	from io import StringIO
class volr(object):
	def __init__(self, subspace_gridded, **settings):
		self.subspace_gridded = subspace_gridded
		self.settings = settings

	def set(self, **kwargs):
		settings = json.dumps(kwargs)
		js_code = """var o = $('#%s').data('vr');
		var new_settings = JSON.parse('%s');
		console.log(new_settings);
		console.log($.extend(o.settings, new_settings));
		o.update_transfer_function_array()
		o.drawScene();
		""" % (self.id, settings)
		display_javascript(js_code, raw=True)

	def _ipython_display_(self):
		f = StringIO()
		#filename = os.path.join(base_path, "cube.png")
		self.subspace_gridded.cube_png(file=f)
		#cube64 = "'data:image/png;base64," + b64encode(file(filename).read()) + "'"
		cube64 = "'data:image/png;base64," + b64encode(f.getvalue()) + "'"
		#display_javascript("""
		#window.cube_src = 'data:image/png;base64,%s';
		#""" % (cube64, colormap64), raw=True)

		self.id = id = uuid.uuid1()
		display_html("<canvas id='{id}' width=512 height=512  style='display: inline;'/>".format(**locals()), raw=True )
		display_javascript(""" $('#%s').vr(
				$.extend({cube:%s, colormap:window.colormap_src}, %s)
				)
				""" % (id, cube64, json.dumps(self.settings)), raw=True)

class init(object):
	def __init__(self, ):
		pass

	def _ipython_display_(self):
		display_javascript(file(os.path.join(base_path, "glMatrix-0.9.5.min.js")).read(), raw=True)
		display_javascript(file(os.path.join(base_path, "volumerenderer.js")).read(), raw=True)
		#cube64 = b64encode(file(os.path.join(base_path, "cube.png")).read())
		colormap64 = b64encode(file(os.path.join(base_path, "colormap.png")).read())
		display_javascript("""
		window.colormap_src = 'data:image/png;base64,%s';
		""" % (colormap64,), raw=True)

		js_code = "window.shader_cache = [];\n"
		for name in ["cube", "texture", "volr"]:
			for type in ["fragment", "vertex"]:
				text = file(os.path.join(base_path, "shaders", name + "-" + type + ".shader")).read()
				text = text.replace("\n", "\\n").replace("'", "\\'")
				js_code += "window.shader_cache['{name}_{type}'] = '{text}';\n".format(**locals())
		display_javascript(js_code, raw=True)
		#print js_code



	def ____ipython_display_(self):
		#base64 = file(os.path.join(base_path, "data.png")).read().encode("base64").replace("\n", "")
		#base64_colormap  = file(os.path.join(base_path, "colormap.png")).read().encode("base64").replace("\n", "")
		#print base64[:10]
		#code = "base64 = '" + base64 + "'; base64_colormap = '" + base64_colormap + "';"
		display_javascript(file(os.path.join(base_path, "all.js")).read(), raw=True)
		display_javascript(file(os.path.join(base_path, "vaex_volumerendering.js")).read(), raw=True)
		display_html(file(os.path.join(base_path, "snippet.js")).read(), raw=True)
		html1 = file(os.path.join(base_path, "snippet.html")).read()
		display_html(html1, raw=True)
		#print "ok"
		display_html("""<div>BLAAT</div> """, raw=True)

		if 0:
			js1 = file(os.path.join(base_path, "snippet.js")).read()
			js2 = file(os.path.join(base_path, "vaex_volumerendering.js")).read()
			js_lib = file(os.path.join(base_path, "all.js")).read()
			html1 = file(os.path.join(base_path, "snippet.html")).read()
			HTML("<script>"+js_lib +"\n" + code +"</script>"+"<script>"+js2+"</script>"+html1+js1)
