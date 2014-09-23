# -*- coding: utf-8 -*-
from qt import *

try:
	from PyQt4 import QtGui, QtCore
	from PyQt4 import QtOpenGL
except ImportError:
	from PySide import QtGui, QtCore
	from PySide import QtOpenGL
import OpenGL
from OpenGL.GL import * # import GL
from OpenGL.GL.framebufferobjects import *
from OpenGL.GLU import *
from OpenGL.GL.ARB.depth_texture import *
from OpenGL.GL.ARB.shadow import *
from OpenGL.GL import shaders

import numpy as np

import gavi.dataset
import gavi.vaex.colormaps
#print GL_R32F
#dsa
GL_R32F = 33326
class VolumeRenderWidget(QtOpenGL.QGLWidget):
	def __init__(self, parent = None):
		super(VolumeRenderWidget, self).__init__(parent)
		self.mouse_button_down = False
		self.mouse_button_down_right = False
		self.mouse_x, self.mouse_y = 0, 0
		self.angle1 = 0
		self.angle2 = 0
		self.mod1 = 0
		self.mod2 = 0
		self.mod3 = 0
		self.mod4 = 0
		self.mod5 = 0
		self.mod6 = 0
		self.setMouseTracking(True)
		shortcut = QtGui.QShortcut(QtGui.QKeySequence("space"), self)
		shortcut.activated.connect(self.toggle)

		shortcut = QtGui.QShortcut(QtGui.QKeySequence("w"), self)
		shortcut.activated.connect(self.write)

		self.texture_index = 2
		self.texture_size = 512 #*8
		
	def toggle(self, ignore=None):
		print "toggle"
		self.texture_index += 1
		self.update()
		
	def create_shader(self):
		self.vertex_shader = shaders.compileShader("""
			varying vec4 vertex_color;
			void main() {
				gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
				//vertex_color = gl_Color;
				vertex_color =  gl_Vertex.x > 1.5 ? vec4(1,0,0,0) : vec4(0,1,0,0)  ;// vec4(gl_Color) + vec4(1, 0, 0, 0);
				vertex_color =  gl_Vertex /60. + vec4(0.5, 0.5, 0.5, 0.);
			}""",GL_VERTEX_SHADER)
		self.fragment_shader = shaders.compileShader("""
			varying vec4 vertex_color;
			uniform sampler1D texture_colormap; 
			uniform sampler2D texture; 
			uniform sampler3D cube; 
			uniform sampler3D gradient;
			uniform vec2 size; // size of screen/fbo, to convert between pixels and uniform
			uniform vec2 minmax2d;
			uniform vec2 minmax3d;
			uniform vec2 minmax3d_total;
			//uniform float maxvalue2d;
			//uniform float maxvalue3d;
			uniform float alpha_mod; // mod3
			uniform float mod4;  // mafnifier
			uniform float mod5; // blend color and line integral
			uniform float mod6; 
			void main() {
				//gl_FragColor = vertex_color;
				//gl_FragColor = texture2D(texture, gl_FragCoord.xy/2.);// * 0.8;
				//gl_FragColor = texture2D(texture, vec2(44, 44));
				//  0.8;
				//gl_FragColor = texture2D(texture, gl_FragCoord.xy/128.) * 0.8;
				vec3 ray_end = vec3(texture2D(texture, vec2(gl_FragCoord.x/size.x, gl_FragCoord.y/size.y)));
				vec3 ray_start = vertex_color.xyz;
				float length = 0.;
				vec3 ray_dir = ray_end - ray_start;
				vec3 ray_delta = ray_dir / 200.;
				float ray_length = sqrt(ray_dir.x*ray_dir.x + ray_dir.y*ray_dir.y + ray_dir.z*ray_dir.z);
				vec3 pos = ray_start;
				float value = 0.;
				//mat3 direction_matrix = inverse(mat3(transpose(inverse(gl_ModelViewProjectionMatrix))));
				//mat3 direction_matrix = transpose(mat3(gl_ModelViewProjectionMatrix));
				//vec3 light_pos = (direction_matrix * vec3(-100.,100., -100)).zyx;
				//vec3 light_pos = (direction_matrix * vec3(-5.,5., -100));
				//vec3 origin = (direction_matrix * vec3(0., 0., 0)).xyz;
				vec3 origin = (vec4(0., 0., 0., 0.)).xyz;
				vec3 light_pos = (vec4(-1000., 0., -1000, 1.)).xyz;
				//mat3 mod = inverse(mat3(gl_ModelViewProjectionMatrix));
				vec4 color;
				vec3 light_dir = light_pos - origin;
				//light_dir = vec3(-1,-1,1);
				light_dir = light_dir / sqrt(light_dir.x*light_dir.x + light_dir.y*light_dir.y + light_dir.z*light_dir.z);
				float alpha_total = 0.;
				//float normalize = log(maxvalue);
				float intensity_total;
				for (int n = 0; n < 200; n++)  {
					//float fraction = float(n) / float(1000);
					//float z_depth = fraction*ray_length;
					//float current_value = texture3D(gradient, pos).b;
					vec3 normal = texture3D(gradient, pos).zyx;
					normal = normal/ sqrt(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
					float cosangle = -dot(light_dir, normal);
					cosangle = 1.;
					//cosangle = clamp(cosangle, 0.0, 1.);;
					//float cosangle = 1.0;
					//float s = 0.0001;
					//value = value + current_value*exp(-(pow(pos.x - 0.5, 2)/s));//+pow(pos.y - 0.5, 2)/s+pow(pos.z - 0.5, 2)/s));
					//value = value + current_value;
					//*max(max(exp(-(pow(pos.x - 0.5, 2)/s)), exp(-(pow(pos.y - 0.5, 2)/s))), exp(-(pow(pos.z - 0.5, 2)/s)));
					//+pow(pos.y - 0.5, 2)/s+pow(pos.z - 0.5, 2)/s));
					
					float intensity = texture3D(cube, pos).r;
					float intensity_normalized = (log(intensity + 1.) - log(minmax3d.x)) / (log(minmax3d.y) - log(minmax3d.x));
					
					//intensity_normalized = clamp(cosangle, 0., 1.);
					vec4 color_sample = texture1D(texture_colormap, intensity_normalized);// * clamp(cosangle, 0.1, 1.);
					//color_sample = color_sample * clamp(cosangle, 0., 1.) * 15.;
					//color_sample = texture1D(texture_colormap, cosangle * 2. - 1.);
					float alpha_sample = 10./200. * alpha_mod  * intensity_normalized;// * clamp(cosangle+0.2, 0.0, 1.);;
					alpha_sample = clamp(alpha_sample, 0., 1.);
					
					
					intensity_total += intensity;
					
					
					color = color + (1.0 - alpha_total) * color_sample * alpha_sample;
					alpha_total = clamp(alpha_total + alpha_sample, 0., 1.);
					
					float border_level = log(minmax3d_total.x) + (log(minmax3d_total.y) - log(minmax3d_total.x)) * mod6 * 0.5;
					float alpha_sample_border = exp(-pow(border_level-log(intensity)/3.,2.)) * mod5;// * clamp(cosangle, 0.1, 1);

					float ambient = 0.5; //atan(log(mod4)) / 3.14159 + 0.5 ;
					vec4 color_border = vec4(1,1,1,1);// * (ambient + clamp(cosangle, 0, 1.-ambient));
					//vec4 color_border = vec4(normal.xyz, 1);// * clamp(cosangle, 0.1, 1);
					color = color + (1.0 - alpha_total) * color_border * alpha_sample_border;
					alpha_total = clamp(alpha_total + alpha_sample_border, 0., 1.);
					
					pos += ray_delta;
					
				}
				gl_FragColor = vec4(color) * mod4;// / pow(0.9*alpha_total + 0.1, 1.0); // / sqrt(color.r*color.r + color.b*color.b + color.g*color.g);
				//value *= 10;
				//gl_FragColor = vec4(ray_end, 1);
				//gl_FragColor = vec4(texture1D(texture_colormap, clamp(log(value*0.0001*ray_length+1)/log(10) * 1.2 - 0.1, 0.01, 0.99)).rgb, 1);
				//gl_FragColor = vec4(texture1D(texture_colormap, log(value*1.1+1.) ).rgb, 1);
				float scale = log(minmax2d.y)/log(10.) - log(minmax2d.x)/log(10.);
				float intensity_total_scaled = (log(intensity_total+1.)/log(10.)-log(minmax2d.x)/log(10.)) / scale;
				//scaled = value / 100.;
				vec4 line_color = vec4(texture1D(texture_colormap, intensity_total_scaled).rgb, 1);
				//gl_FragColor = line_color;
				//float blend = atan(log(mod5)) / 3.14159 + 0.5 ;
				//vec3 = gl_ModelViewProjectionMatrix
				//gl_FragColor = vec4(light_dir, 1.);
				//gl_FragColor = (blend * line_color + (1.-blend) * vec4(color)*mod6) * mod4;
				//gl_FragColor = vec4(value, 0, 0, 1);
				//gl_FragColor = texture3D(cube, vec3(gl_FragCoord.x/size.x, gl_FragCoord.y/size.y, 0.5) );
				//gl_FragColor = texture3D(gradient, vec3(gl_FragCoord.x/size.x, gl_FragCoord.y/size.y, 0.5) );
				//gl_FragColor = vec4(ray_start, 1);
			}""",GL_FRAGMENT_SHADER)
		return shaders.compileProgram(self.vertex_shader, self.fragment_shader)

	def paintGL(self):
		glMatrixMode(GL_MODELVIEW)

		glLoadIdentity()
		glTranslated(0.0, 0.0, -15.0)
		glRotated(self.angle1, 1.0, 0.0, 0.0)
		glRotated(self.angle2, 0.0, 1.0, 0.0)
		
		
		
		self.draw_backside()
		self.draw_frontside()
		self.draw_to_screen()

	def draw_backside(self):
		glViewport(0, 0, self.texture_size, self.texture_size)
		#glViewport(0, 0, 128*2, 128*2)
		glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texture_backside, 0);
		glClearColor(1.0, 1.0, 0.0, 1.0)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		
		glEnable(GL_CULL_FACE);
		glCullFace(GL_FRONT);

		glShadeModel(GL_SMOOTH);
		self.cube(size=60)
		#return

	def draw_frontside(self):
		glViewport(0, 0, self.texture_size, self.texture_size)
		glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
		#glBindFramebuffer(GL_FRAMEBUFFER, 0)
		
		
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texture_final, 0);
		
		
		glClearColor(0.0, 0.0, 1.0, 1.0)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		
		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);
		
		
		glShadeModel(GL_SMOOTH);

		glUseProgram(self.shader)
		loc = glGetUniformLocation(self.shader, "texture");
		glUniform1i(loc, 0); # texture unit 0
		glBindTexture(GL_TEXTURE_2D, self.texture_backside)
		glEnable(GL_TEXTURE_2D)
		glActiveTexture(GL_TEXTURE0);
		if 1:
			loc = glGetUniformLocation(self.shader, "cube");
			glUniform1i(loc, 1); # texture unit 1
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_3D, self.texture_cube)
			#glEnable(GL_TEXTURE_3D)

			loc = glGetUniformLocation(self.shader, "gradient");
			glUniform1i(loc, 3); # texture unit 1
			glActiveTexture(GL_TEXTURE3);
			glBindTexture(GL_TEXTURE_3D, self.texture_gradient)
			#glEnable(GL_TEXTURE_3D)
		
			loc = glGetUniformLocation(self.shader, "texture_colormap");
			glUniform1i(loc, 2); # texture unit 2
			glActiveTexture(GL_TEXTURE2);
			index = gavi.vaex.colormaps.colormaps.index("afmhot")
			glBindTexture(GL_TEXTURE_1D, self.textures_colormap[index])
			glEnable(GL_TEXTURE_1D)

			glActiveTexture(GL_TEXTURE0);
		
		size = glGetUniformLocation(self.shader,"size");
		glUniform2f(size, self.texture_size, self.texture_size);
		
		#maxvalue = glGetUniformLocation(self.shader,"maxvalue");
		#glUniform1f(maxvalue, self.data3d.max()*10**self.mod2);
		
		
		minmax = glGetUniformLocation(self.shader,"minmax2d");
		glUniform2f(minmax, 1*10**self.mod1, self.data2d.max()*10**self.mod2);

		minmax = glGetUniformLocation(self.shader,"minmax3d");
		glUniform2f(minmax, 1*10**self.mod1, self.data3d.max()*10**self.mod2);
		
		minmax3d_total = glGetUniformLocation(self.shader,"minmax3d_total");
		glUniform2f(minmax3d_total, 1, self.data3d.max());
		

		alpha_mod = glGetUniformLocation(self.shader,"alpha_mod");
		glUniform1f(alpha_mod , 10**self.mod3);
		
		for i in range(4,7):
			name = "mod" + str(i)
			mod = glGetUniformLocation(self.shader, name)
			glUniform1f(mod, 10**getattr(self, name));
		
		
		
		

		glShadeModel(GL_SMOOTH);
		self.cube(size=60)
		glUseProgram(0)

		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_1D, 0)
		glEnable(GL_TEXTURE_2D)

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, 0)
		glEnable(GL_TEXTURE_2D)

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, 0)
		glEnable(GL_TEXTURE_2D)
		self.cube(size=60, gl_type=GL_LINE_LOOP)

		#return
		
	def draw_to_screen(self):
		w = self.width()
		h = self.height()
		glViewport(0, 0, w, h)
		#glShadeModel(GL_FLAT);
		
		glBindFramebuffer(GL_FRAMEBUFFER, 0)
		glClearColor(1.0, 0.0, 0.0, 1.0)
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
		
		glCullFace(GL_BACK);


		glBindTexture(GL_TEXTURE_2D, self.textures[self.texture_index % len(self.textures)])
		#glBindTexture(GL_TEXTURE_3D, self.texture_cube)
		glEnable(GL_TEXTURE_2D)
		glLoadIdentity()
		glBegin(GL_QUADS)
		w = 49
		z = -1
		glTexCoord2f(0,0); 
		glVertex3f(-w, -w, z)
		glTexCoord2f(1,0); 
		glVertex3f( w, -w, z)
		glTexCoord2f(1,1); 
		glVertex3f( w,  w, z)
		glTexCoord2f(0,1); 
		glVertex3f(-w,  w, z)
		glEnd()
		glBindTexture(GL_TEXTURE_2D, 0)
		
		#glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		
		
	def cube(self, size, gl_type=GL_QUADS):
		w = size/2.
		
		def vertex(x, y, z):
			glColor3f(x+w, y+w, z+w)
			#glMultiTexCoord3f(GL_TEXTURE1, x, y, z);
			glVertex3f(x, y, z)
		
		# back
		if 1:
			glColor3f(1, 0, 0)
			glBegin(gl_type);
			vertex(-w, -w, -w)
			vertex(-w,  w, -w)
			vertex( w,  w, -w)
			vertex( w, -w, -w)
			glEnd()

		# front
		if 1:
			glBegin(gl_type);
			glColor3f(0, 1, 0)
			vertex(-w, -w, w)
			vertex( w, -w, w)
			vertex( w,  w, w)
			vertex(-w,  w, w)
			glEnd()
		
		# right
		if 1:
			glBegin(gl_type);
			glColor3f(0, 0, 1)
			vertex(w, -w,  w)
			vertex(w, -w, -w)
			vertex(w,  w, -w)
			vertex(w,  w,  w)
			glEnd()
		
		# left
		if 1:
			glBegin(gl_type);
			glColor3f(0, 0, 1)
			vertex(-w, -w, -w)
			vertex(-w, -w,  w)
			vertex(-w,  w,  w)
			vertex(-w,  w, -w)
			glEnd()
		
		# top
		if 1:
			glBegin(gl_type);
			glColor3f(0, 0, 1)
			vertex( w,  w, -w)
			vertex(-w,  w, -w)
			vertex(-w,  w,  w)
			vertex( w,  w,  w)
			glEnd()
		
		# bottom
		if 1:
			glBegin(gl_type);
			glColor3f(0, 0, 1)
			vertex(-w, -w, -w)
			vertex( w, -w, -w)
			vertex( w, -w,  w)
			vertex(-w, -w,  w)
			glEnd()

	def resizeGL(self, w, h):
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		#glOrtho(-50, 50, -50, 50, -50.0, 50.0)
		glOrtho(-50, 50, -50, 50, -150.0, 150.0)
		glViewport(0, 0, w, h)

	def initializeGL(self):
		
		colormaps = gavi.vaex.colormaps.colormaps
		Nx, Ny = 1024, 16
		self.colormap_data = np.zeros((len(colormaps), Nx, 3), dtype=np.uint8)
		
		import matplotlib.cm
		self.textures_colormap = glGenTextures(len(colormaps))
		for i, colormap_name in enumerate(colormaps):
			colormap = matplotlib.cm.get_cmap(colormap_name)
			mapping = matplotlib.cm.ScalarMappable(cmap=colormap)
			#pixmap = QtGui.QPixmap(32*2, 32)
			x = np.arange(Nx) / (Nx -1.)
			#x = np.vstack([x]*Ny)
			rgba = mapping.to_rgba(x,bytes=True).reshape(Nx, 4)
			rgb = rgba[:,0:3] * 1
			self.colormap_data[i] = rgb #(rgb*255.).astype(np.uint8)
			if i == 0:
				print rgb[0], rgb[-1], 
			
			
			texture = self.textures_colormap[i]
			glBindTexture(GL_TEXTURE_1D, texture)
			glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
			glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
			glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
			glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
			glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
			glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB8, Nx, 0, GL_RGB, GL_UNSIGNED_BYTE, self.colormap_data[i]);
			#glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.texture_size, self.texture_size, 0, GL_RGBA, GL_UNSIGNED_BYTE, None);
			glBindTexture(GL_TEXTURE_1D, 0)
			

		if 1:
			N = 1024 * 4
			self.surface_data =  np.zeros((N, 3), dtype=np.uint8)
			self.texture_surface = glGenTextures(1)
			glBindTexture(GL_TEXTURE_1D, self.texture_surface)
			glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
			glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
			glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
			glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
			glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
			glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB8, Nx, 0, GL_RGB, GL_UNSIGNED_BYTE, self.surface_data);
			#glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.texture_size, self.texture_size, 0, GL_RGBA, GL_UNSIGNED_BYTE, None);
			glBindTexture(GL_TEXTURE_1D, 0)
			



		if 0:
			
			f = glCreateShaderObject(GL_FRAGMENT_SHADER);
			fragment_source = "void main(){ gl_FragColor=gl_FragCoord/512.0; }";
			glShaderSource(f, 1, fs, None);
			glCompileShaderARB(f);
		
			self.program = glCreateProgramObjectARB();
			glAttachObjectARB(self.program, f);

		
		glClearColor(0.0, 0.0, 0.0, 1.0)
		glClear(GL_COLOR_BUFFER_BIT)

		print bool(glGenFramebuffers)
		self.fbo = glGenFramebuffers(1)
		print self.fbo
		glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
		
		self.textures = self.texture_backside, self.texture_final = glGenTextures(2)
		print "textures", self.textures
		for texture in [self.texture_backside, self.texture_final]:
			glBindTexture(GL_TEXTURE_2D, texture)
			glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
			#glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
			#glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
			#glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, self.texture_size, self.texture_size, 0, GL_RGBA, GL_FLOAT, None);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.texture_size, self.texture_size, 0, GL_RGBA, GL_UNSIGNED_BYTE, None);
			glBindTexture(GL_TEXTURE_2D, 0)
			
		self.size3d = 128 # * 4
		self.data3d = np.zeros((self.size3d, self.size3d, self.size3d)) #.astype(np.float32)
		self.data2d = np.zeros((self.size3d, self.size3d)) #.astype(np.float32)
		
		self.data2d
		
		
		dataset = gavi.dataset.load_file(sys.argv[1])
		x, y, z = [dataset.columns[name] for name in sys.argv[2:]]
		import gavifast
		mi, ma = 45., 55.
		#print "histo"
		gavifast.histogram3d(x, y, z, None, self.data3d, mi+7, ma+7, mi+3, ma+3, mi, ma)
		mi, ma = -30., 30.
		mi, ma = -0.5, 0.5
		#gavifast.histogram3d(x, y, z, None, self.data3d, mi, ma, mi, ma, mi, ma)
		#mi, ma = -0.6, 0.6
		#mi, ma = -30., 30.
		#gavifast.histogram3d(x, y, z, None, self.data3d, x.min(), x.max(), y.min(), y.max(), z.min(), z.max())
		#mi, ma = -30., 30.
		#gavifast.histogram3d(x, y, z, None, self.data3d, mi, ma, mi, ma, mi, ma)
		print "histo done"
		#gavifast.histogram2d(x, y, None, self.data2d, x.min(), x.max(), y.min(), y.max())
		gavifast.histogram2d(x, y, None, self.data2d, mi, ma, mi, ma)
		#x, y, z = np.mesgrid
		#print self.data3d
		self.data3d = self.data3d.astype(np.float32)
		self.data2d = self.data2d.astype(np.float32)


		import scipy.ndimage
		#self.data3d = 10**scipy.ndimage.gaussian_filter(np.log10(self.data3d+1), 1.5)-1
		self.data3d = 10**scipy.ndimage.gaussian_filter(np.log10(self.data3d+1), 0.5)-1
		data3ds = scipy.ndimage.gaussian_filter((self.data3d), 1.5)
		#data3ds = data3ds.sum(axis=0)
		self.grad3d = np.gradient(data3ds)
		length = np.sqrt(self.grad3d[0]**2 + self.grad3d[1]**2 + self.grad3d[2]**2)
		self.grad3d[0] = self.grad3d[0] / length
		self.grad3d[1] = self.grad3d[1] / length
		self.grad3d[2] = self.grad3d[2] / length
		if 0:
			import pylab
			pylab.subplot(221)
			pylab.imshow(data3ds)
			pylab.subplot(222)
			pylab.imshow(self.grad3d[0])
			pylab.subplot(223)
			pylab.imshow(self.grad3d[1])
			pylab.show()
		if 1:
			self.grad3ddata = np.zeros((self.size3d, self.size3d, self.size3d, 3), dtype=np.float32)
			self.grad3ddata[:,:,:,0] = self.grad3d[0]
			self.grad3ddata[:,:,:,1] = self.grad3d[1]
			self.grad3ddata[:,:,:,2] = self.grad3d[2]
			self.grad3d = self.grad3ddata
		
		del self.grad3ddata
		print self.grad3d.shape
		
		
		#self.data3d -= self.data3d.min()
		#self.data3d /= self.data3d.max()
		#self.data3d = np.log10(self.data3d+1)
		#self.data2d = np.log10(self.data2d+1)

		#self.data3df = (self.data3d * 1.).astype(np.float32)
		#self.data2df = self.data2d * 1.0
		
		#self.data3d -= self.data3d.min()
		#self.data3d /= self.data3d.max()
		#self.data3d = (self.data3d * 255).astype(np.uint8)

		#self.data2d -= self.data2d.min()
		#self.data2d /= self.data2d.max()
		#self.data2d = (self.data2d * 255).astype(np.uint8)
		#print self.data3d.max()
		
		self.texture_cube = glGenTextures(1)
		self.texture_gradient = glGenTextures(1)
		self.texture_square = glGenTextures(1)
		
		#glActiveTexture(GL_TEXTURE1);
		#glBindTexture(GL_TEXTURE_3D, self.texture_cube)
		#glBindTexture(GL_TEXTURE_3D, self.texture_cube)
		glBindTexture(GL_TEXTURE_2D, self.texture_square)
		#glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE8, self.size3d, self.size3d, self.size3d, 0,
         #               GL_RED, GL_FLOAT, self.data3d)
		self.rgb = np.zeros((self.size3d, self.size3d, 3), dtype=np.uint8)
		self.rgb[:,:,0] = self.data2d
		self.rgb[:,:,1] = self.data2d
		self.rgb[:,:,2] = self.data2d

		self.rgb3d = np.zeros((self.size3d, self.size3d, self.size3d, 3), dtype=np.uint8)
		self.rgb3d[:,:,:,0] = self.data3d #.sum(axis=0)
		self.rgb3d[:,:,:,1] = self.data3d #.sum(axis=0)
		self.rgb3d[:,:,:,2] = self.data3d#.sum(axis=0)

		print self.rgb.max()
		glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE8, self.size3d, self.size3d, 0,
                        GL_LUMINANCE, GL_FLOAT, self.data2d)
		
		glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		if 0:
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
		glBindTexture(GL_TEXTURE_2D, 0)
		#glActiveTexture(GL_TEXTURE0);
		self.textures = [self.texture_square,] + list(self.textures)
		print "textures", self.textures
		
		
		glBindTexture(GL_TEXTURE_3D, self.texture_cube)
		glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, self.size3d, self.size3d, self.size3d, 0,
                        GL_RED, GL_FLOAT, self.data3d)
		#glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB8, self.size3d, self.size3d, self.size3d, 0,
         #               GL_RGB, GL_UNSIGNED_BYTE, self.rgb3d)
		
		glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		if 1:
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
			

		# gradient
		glBindTexture(GL_TEXTURE_3D, self.texture_gradient)
		glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F, self.size3d, self.size3d, self.size3d, 0,
                        GL_RGB, GL_FLOAT, self.grad3d)
		#glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB8, self.size3d, self.size3d, self.size3d, 0,
         #               GL_RGB, GL_UNSIGNED_BYTE, self.rgb3d)
		
		glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		if 1:
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
		
			
		
		
		glFramebufferTexture2D(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texture_backside, 0);

		self.render_buffer = glGenRenderbuffers(1);
		glBindRenderbuffer(GL_RENDERBUFFER, self.render_buffer);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.texture_size, self.texture_size);

		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.render_buffer);
		#glBindFramebuffer(GL_FRAMEBUFFER, 0);
		#from matplotlib import pylab
		#pylab.imshow(np.log((self.data3d.astype(np.float32)).sum(axis=0)+1), cmap='PaulT_plusmin', origin="lower")
		#pylab.show()


		if 1:
			self.shader = self.create_shader()

			#glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE)

	#sys.exit(0)
		#self.back_buffer, self.final_buffer = glGenTextures(2)
		#bo = glGenFramebuffers(1)


	def mouseMoveEvent(self, event):
		x, y = event.x(), event.y()
		dx = x - self.mouse_x
		dy = y - self.mouse_y
		
		
		speed = 1.
		speed_mod = 0.1/5./5.
		if self.mouse_button_down:
			self.angle2 += dx * speed
			self.angle1 += dy * speed
			print self.angle1, self.angle2
		if self.mouse_button_down_right:
			if QtGui.QApplication.keyboardModifiers() == QtCore.Qt.NoModifier:
				self.mod1 += dx * speed_mod
				self.mod2 += -dy * speed_mod
				print "mod1/2", self.mod1, self.mod2
			if QtGui.QApplication.keyboardModifiers() == QtCore.Qt.AltModifier:
				self.mod3 += dx * speed_mod
				self.mod4 += -dy * speed_mod
				print "mod3/4", self.mod3, self.mod4
			if QtGui.QApplication.keyboardModifiers() == QtCore.Qt.ShiftModifier:
				self.mod5 += dx * speed_mod
				self.mod6 += -dy * speed_mod
				print "mod5/6", self.mod5, self.mod6
			
		
		self.mouse_x, self.mouse_y = x, y
		if self.mouse_button_down or self.mouse_button_down_right:
			self.update()
		
	def mousePressEvent(self, event):
		if event.button() == QtCore.Qt.LeftButton:
			self.mouse_button_down = True
		if event.button() == QtCore.Qt.RightButton:
			self.mouse_button_down_right = True

	def mouseReleaseEvent(self, event):
		if event.button() == QtCore.Qt.LeftButton:
			self.mouse_button_down = False
		if event.button() == QtCore.Qt.RightButton:
			self.mouse_button_down_right = False
			
	def write(self):
		colormap_name = "afmhot"
		import matplotlib.cm
		colormap = matplotlib.cm.get_cmap(colormap_name)
		mapping = matplotlib.cm.ScalarMappable(cmap=colormap)
		#pixmap = QtGui.QPixmap(32*2, 32)
		data = np.zeros((128*8, 128*16, 4), dtype=np.uint8)
		
		mi, ma = 1*10**self.mod1, self.data3d.max()*10**self.mod2
		intensity_normalized = (np.log(self.data3d + 1.) - np.log(mi)) / (np.log(ma) - np.log(mi));
		import PIL.Image
		for y2d in range(8):
			for x2d in range(16):
				zindex = x2d + y2d*16
				I = intensity_normalized[zindex]
				rgba = mapping.to_rgba(I,bytes=True) #.reshape(Nx, 4)
				print rgba.shape
				subdata = data[y2d*128:(y2d+1)*128, x2d*128:(x2d+1)*128]
				for i in range(3):
					subdata[:,:,i] = rgba[:,:,i]
				subdata[:,:,3] = (intensity_normalized[zindex]*255).astype(np.uint8)
				if 0:
					filename = "cube%03d.png" % zindex
					img = PIL.Image.frombuffer("RGB", (128, 128), subdata[:,:,0:3] * 1)
					print "saving to", filename
					img.save(filename)
		img = PIL.Image.frombuffer("RGBA", (128*16, 128*8), data)
		filename = "cube.png"
		print "saving to", filename
		img.save(filename)
		
		filename = "colormap.png"
		print "saving to", filename
		height, width = self.colormap_data.shape[:2]
		img = PIL.Image.frombuffer("RGB", (width, height), self.colormap_data)
		img.save(filename)
		
		

class TestWidget(QtGui.QMainWindow):
	def __init__(self, parent):
		super(TestWidget, self).__init__(parent)
		self.resize(700, 700)
		self.show()
		self.raise_()
		shortcut = QtGui.QShortcut(QtGui.QKeySequence("Cmd+Q"), self)
		shortcut.activated.connect(self.myclose)
		
		self.main = VolumeRenderWidget(self)
		self.setCentralWidget(self.main)
		
		#self.layout = QtGui.QVBoxLayout(self)
		#self.layout.addWidget(self.main)
		#self.setLayout(self.layout)
		
	def myclose(self, ignore=None):
		self.hide()
		
		
		
if __name__ == "__main__":
	import gavi.vaex.colormaps
	colormaps = gavi.vaex.colormaps.colormaps
	import json
	js = json.dumps(gavi.vaex.colormaps.colormaps)
	print js

	app = QtGui.QApplication(sys.argv)
	widget = TestWidget(None)
	sys.exit(app.exec_())