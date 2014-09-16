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
		self.setMouseTracking(True)
		shortcut = QtGui.QShortcut(QtGui.QKeySequence("space"), self)
		shortcut.activated.connect(self.toggle)
		self.texture_index = 2
		self.texture_size = 512*2 #*8
		
	def toggle(self, ignore=None):
		print "toggle"
		self.texture_index += 1
		self.update()
		
	def create_shader(self):
		self.vertex_shader = shaders.compileShader(
			"""
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
			uniform vec2 size; // size of screen/fbo, to convert between pixels and uniform
			uniform vec2 minmax;
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
				vec3 ray_delta = ray_dir / 1000.;
				float ray_length = sqrt(ray_dir.x*ray_dir.x + ray_dir.y*ray_dir.y + ray_dir.z*ray_dir.z);
				vec3 pos = ray_start;
				float value = 0.;
				for (int n = 0; n < 1000; n++)  {
					float fraction = float(n) / float(1000);
					float z_depth = fraction*ray_length;
					float current_value = texture3D(cube, pos).r;
					float s = 0.0001;
					//value = value + current_value*exp(-(pow(pos.x - 0.5, 2)/s));//+pow(pos.y - 0.5, 2)/s+pow(pos.z - 0.5, 2)/s));
					value = value + current_value;//*max(max(exp(-(pow(pos.x - 0.5, 2)/s)), exp(-(pow(pos.y - 0.5, 2)/s))), exp(-(pow(pos.z - 0.5, 2)/s)));
					;//+pow(pos.y - 0.5, 2)/s+pow(pos.z - 0.5, 2)/s));
					pos += ray_delta;
				}
				//value *= 10;
				//gl_FragColor = vec4(ray_end, 1);
				//gl_FragColor = vec4(texture1D(texture_colormap, clamp(log(value*0.0001*ray_length+1)/log(10) * 1.2 - 0.1, 0.01, 0.99)).rgb, 1);
				//gl_FragColor = vec4(texture1D(texture_colormap, log(value*1.1+1.) ).rgb, 1);
				float scale = log(minmax.y)/log(10.) - log(minmax.x)/log(10.);
				float scaled = (log(value/10.+1.)/log(10.)-log(minmax.x)/log(10.)) / scale;// * 1.1 - 0.05;
				gl_FragColor = vec4(texture1D(texture_colormap, scaled * 1.2 - 0.1).rgb, 1);
				//gl_FragColor = texture3D(cube, vec3(gl_FragCoord.x/size.x, gl_FragCoord.y/size.y, 0.5) );
				//gl_FragColor = texture2D(cube, vec2(gl_FragCoord.x/size.x, gl_FragCoord.y/size.y) );
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
		
			loc = glGetUniformLocation(self.shader, "texture_colormap");
			glUniform1i(loc, 2); # texture unit 2
			glActiveTexture(GL_TEXTURE2);
			index = gavi.vaex.colormaps.colormaps.index("afmhot")
			glBindTexture(GL_TEXTURE_1D, self.textures_colormap[index])
			glEnable(GL_TEXTURE_1D)

			glActiveTexture(GL_TEXTURE0);
		
		size = glGetUniformLocation(self.shader,"size");
		glUniform2f(size, self.texture_size, self.texture_size);
		
		minmax = glGetUniformLocation(self.shader,"minmax");
		glUniform2f(minmax, 1*10**self.mod1, self.data2d.max()*10**self.mod2);
		

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
			
		self.size3d = 128 * 4
		self.data3d = np.zeros((self.size3d, self.size3d, self.size3d)) #.astype(np.float32)
		self.data2d = np.zeros((self.size3d, self.size3d)) #.astype(np.float32)
		
		self.data2d
		
		
		dataset = gavi.dataset.load_file(sys.argv[1])
		x, y, z = [dataset.columns[name] for name in sys.argv[2:]]
		import gavifast
		mi, ma = 45., 55.
		print "histo"
		gavifast.histogram3d(x, y, z, None, self.data3d, mi+7, ma+7, mi+3, ma+3, mi, ma)
		#mi, ma = -30., 30.
		#gavifast.histogram3d(x, y, z, None, self.data3d, mi, ma, mi, ma, mi, ma)
		#mi, ma = -0.6, 0.6
		#gavifast.histogram3d(x, y, z, None, self.data3d, mi, ma, mi, ma, mi, ma)
		print "histo done"
		gavifast.histogram2d(x, y, None, self.data2d, mi, ma, mi, ma)
		#x, y, z = np.mesgrid
		#print self.data3d
		self.data3d = self.data3d.astype(np.float32)
		self.data2d = self.data2d.astype(np.float32)
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
		speed_mod = 0.1/5
		if self.mouse_button_down:
			self.angle2 += dx * speed
			self.angle1 += dy * speed
			print self.angle1, self.angle2
		if self.mouse_button_down_right:
			self.mod1 += dx * speed_mod
			self.mod2 += -dy * speed_mod
			print self.mod1, self.mod2
			
		
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
	app = QtGui.QApplication(sys.argv)
	widget = TestWidget(None)
	sys.exit(app.exec_())