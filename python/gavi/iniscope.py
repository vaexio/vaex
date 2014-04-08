# -*- coding: utf-8 -*-
from __future__ import absolute_import
import ConfigParser as configparser
import sys
import os
import numpy
from math import *
import gavi.logging as logging
import math
import collections
import gavi.ordereddict
import glob

logger = logging.getLogger("gd.utils.iniscope")

enable_logging = True
enable_imports = True

module_exts = [".py"]
imported = []

def recursive_import(basedirname, dirname):
	if not enable_imports:
		return
	if dirname in imported:
		return
	imported.append(dirname)
	filenames = glob.glob(os.path.join(dirname, "*"))
	for filename in filenames:
		if  os.path.isdir(filename):
			#print "DIR", filename
			recursive_import(basedirname, filename)
		else:
			name, ext = os.path.splitext(filename)
			if ext in module_exts:
				modulename = name[len(basedirname)+1:].replace("/", ".")
				#print "MOD", modulename
				#if enable_logging:
				#	logger.debug("basedir: %s; name: %s; modulename: %s" % (basedirname, name, modulename))
				exec "import %s" % modulename
				
				
class SubScope(object):
	def __init__(self, scope, globals=[], includes=[], configurations=[]):
		self.scope = scope
		self.globals = globals
		self.includes = includes
		self.configurations = configurations
		
	def load(self):
		self.subscope = self.scope.clone()
		self.subscope.configurations.extend(self.configurations)
		self.subscope.globals.extend(self.globals)
		self.subscope.readfiles(*self.includes)
		
		
		
class IniScope(object):
	def __init__(self, filename, aliases={}, globals=[], load=False, configurations=[], **defaults):
		self.filename = filename
		self.aliases = aliases
		self.globals = globals
		self.defaults = defaults
		self.allfiles = []
		self.configurations = configurations
		self.reset()
		if load:
			self.readfiles()
			self.init()
			
	def clone(self):
		clone = IniScope(self.filename, aliases=self.aliases, globals=self.globals)
		clone.dict = dict(self.dict)
		clone.allfiles = list(self.allfiles)
		clone.ini = self.ini
		clone.defaults = dict(self.defaults)
		clone.configurations = list(self.configurations)
		return clone
		
	def __setitem__(self, name, value):
		#print "set:", name
		self.dict[name] = value
	def __getitem__(self,  name):
		#print "get:", name
		if name not in self.dict:
			#print self.ini.has_option("globals", name)
			#print dict(self.ini.items("globals"))
			found = False
			# first look for section
			if name in self.ini.sections():
				self.dict[name] = createobj_fromini(self.ini, name, self, self.configurations)
				logger.debug("found %r in section" % name)
				found = True
			# then 'overriden' globals
			if not found:
				if name in dict(self.globals):
					v = dict(self.globals)[name]
					self.dict[name] = eval(v, globals(), self)
					found = True
					logger.debug("found %r in globals(override)" % name)
			# then globals...
			if not found:
				# in reverse order of 'configurations'
				for conf in ([""] + self.configurations)[::-1]:
					itemname = "globals" +":" +conf if conf else "globals"
					if itemname in self.ini.sections():
						items = dict(self.ini.items(itemname))
						if name in items:
							value = items[name]
							try:
								self[name] = eval(value, globals(), self)
								found = True
							except:
								logger.error("error evaluating global %s=%s" % (name, value))
								raise
							logger.debug("%s: %s=%r (expr: %s)" % (itemname, name, self[name], value))
					if found:
						break
			#logger.debug("setting alias: %s=%s " % (name, value))
			"""if "parameters" in ini.sections():
				for name, value in ini.items("parameters"):
					v = eval(value, globals(), self)
					logger.debug("parameter: %s=%r (expr: %s)" % (name, v, value))
					self[name] = v
				
			elif self.ini.has_option("globals", name):
				#print name, "in globals"
				self.dict[name] = eval(self.ini.get("globals", name), globals(), self)
			elif self.ini.has_option("globals", name):
				#print name, "in globals"
				self.dict[name] = eval(self.ini.get("globals", name), globals(), self)"""
			#else:
				#print >>sys.stderr, "count not resolve %r" % name
				#print >>sys.stderr, "sections", self.ini.sections()
				#print >>sys.stderr, "globals", [name for name, value in self.ini.items("globals")]
		return self.dict[name]
	
	def __contains__(self, name):
		return self.dict.__contains__(name)
	
	def flush(self):
		self.dict = dict()
	
	def __delitem__(self, name):
		del self.dict[name]
		
	def reset(self):
		self.dict = dict(self.defaults)
		def prt(*args):
			print "dsaaaaaaaaaaaaaaaa"
			print args
		self.dict["print"] = prt
		self.ini = configparser.ConfigParser(dict_type=gavi.ordereddict.OrderedDict)
		self.ini.optionxform = str
		
	def _followincludes(self, filename, allfilenames):
		logger.debug("follow include for %s" % filename)
		tempini = configparser.ConfigParser()
		tempini.optionxform = str
		filename = filename.strip() # strip spaces
		if not os.path.exists(filename):
			print "error: file %s missing" % filename
			sys.exit(-1) 
		if os.path.exists(filename):
			tempini.read(filename)
			if "include" in tempini.sections():
				values = dict(tempini.items("include"))
				if "before" in values:
					dirname = os.path.dirname(filename)
					before_filenames = eval(values["before"], globals(), self)
					for name in before_filenames:
						f = os.path.abspath(os.path.join(dirname, name)) 
						#print name, f 
						#allfilenames.append(f)
						logger.debug("including %s (before)" % f)
						self._followincludes(f, allfilenames)
 
			allfilenames.append(filename)
			if "include" in tempini.sections():
				if "after" in values:
					dirname = os.path.dirname(filename)
					before_filenames = eval(values["after"], globals(), self)
					for name in before_filenames:
						f = os.path.abspath(os.path.join(dirname, name))
						#print name, f 
						#allfilenames.append(f)
						logger.debug("including %s (after)" % f) 
						self._followincludes(f, allfilenames)
	def re_readfiles(self, *extrafilenames):
		self.reset()
		self._readfiles(*extrafilenames)
		
	def readfiles(self, *extrafilenames):
		#self.dict["__filename__"] #f = os.path.abspath(os.path.join(dirname, name))
		assert os.path.exists(self.filename), "missing ini file: %s" % `self.filename`
		allfilenames = [self.filename] + list(extrafilenames)
		newallfilenames = []
		#print "all filenames", allfilenames
		#tempscope = AutoloadScope(self.filenames, self.aliases, self.globals, **self.defaults)
		#tempscope.init()
		#tempscope.readfiles()
		#print tempscope.ini.items("include")
		#dsa
		#tempscope.ini["
		#for name, value in self.globals:
		#	self[name] = value
		#	logger.debug("(overriding) setting global: %s=%s " % (name, value))
		for filename in allfilenames:
			self.ini.read(self.filename)
			self._followincludes(filename, newallfilenames)
				#allfilenames.append(filename)
		 
		#print newallfilenames
		self.reset()
		self.allfiles = newallfilenames
		self._readfiles()
	
	def _readfiles(self, *extrafilenames):
		for filename in self.allfiles + list(extrafilenames):
			#print >>sys.stderr, filename
			#print >>sys.stdout, filename
			if not os.path.exists(filename):
				print "file % missing" % filename
				sys.exit(-1)
			self.ini.read(filename)
		#print self.ini.sections()
		ini = self.ini
		if "imports" in ini.sections():
			for name, value in ini.items("imports"):
				logger.debug("import: %s (%s)" % (name, value))
				exec "import %s" % name
				basename = name.split(".")[0]
				#print "   basename", basename
				basemodule = eval(basename)
				module = eval(name)
				self[basename] = basemodule
				if value == "recursive":
					dirname = os.path.dirname(module.__file__)
					#print module, dirname
					basedirname = os.path.split(os.path.dirname(basemodule.__file__))[0]
					recursive_import(basedirname, dirname)
					#print dirname
					#sys.exit(0)
		if "parameters" in ini.sections():
			for name, value in ini.items("parameters"):
				v = eval(value, globals(), self)
				logger.debug("parameter: %s=%r (expr: %s)" % (name, v, value))
				self[name] = v
		if "print" in ini.sections():
			for name, value in ini.items("print"):
				v = eval(value, globals(), self)
				logger.info("print: %s=%r (expr: %s)" % (name, v, value))
		for name, value in self.aliases:
			ini.set(name, "alias", value)
			logger.debug("setting alias: %s=%s " % (name, value))
		self["scope"] = self
		
		return
		for name, value in self.globals:
			self[name] = value
			logger.debug("(overriding) setting global: %s=%s " % (name, value))
		for conf in [""] + self.configurations:
			itemname = "globals" +":" +conf if conf else "globals"
			if itemname in ini.sections():
				self["configuration"] = self
				for name, value in ini.items(itemname):
					if name not in dict(self.globals): #dict(ini.items("globals")):
						try:
							v = eval(value, globals(), self)
						except:
							logger.error("error evaluating global %s=%s" % (name, value))
							raise
						logger.debug("%s: %s=%r (expr: %s)" % (itemname, name, v, value))
						self[name] = v
		for name, value in self.aliases:
			ini.set(name, "alias", value)
			logger.debug("setting alias: %s=%s " % (name, value))
		if "parameters" in ini.sections():
			for name, value in ini.items("parameters"):
				v = eval(value, globals(), self)
				logger.debug("parameter: %s=%r (expr: %s)" % (name, v, value))
				self[name] = v
				#print "param:", name, value
		#print self.aliases
			
		
		
	def init(self):
		return
		ini = self.ini
		aliases = self.aliases
		if "imports" in ini.sections():
			for name, value in ini.items("imports"):
				logger.debug("import: %s (%s)" % (name, value))
				exec "import %s" % name
				basename = name.split(".")[0]
				#print "   basename", basename
				basemodule = eval(basename)
				module = eval(name)
				self[basename] = basemodule
				if value == "recursive":
					dirname = os.path.dirname(module.__file__)
					#print module, dirname
					basedirname = os.path.split(os.path.dirname(basemodule.__file__))[0]
					recursive_import(basedirname, dirname)
					#print dirname
					#sys.exit(0)
		if "globals" in ini.sections():
			self["configuration"] = self
			for name, value in ini.items("globals"):
				try:
					v = eval(value, globals(), self)
				except:
					logger.error("error evaluating global %s=%s" % (name, value))
					raise
				logger.debug("global: %s=%r (expr: %s)" % (name, v, value))
				self[name] = v
		if "parameters" in ini.sections():
			for name, value in ini.items("parameters"):
				v = eval(value, globals(), self)
				logger.debug("parameter: %s=%r (expr: %s)" % (name, v, value))
				self[name] = v
				# "param:", name, value
		#print aliases
		for name, value in aliases:
			ini.set(name, "alias", value)
			logger.debug("setting alias: %s=%s " % (name, value))
		for name, value in self.globals:
			self[name] = value
			logger.debug("(overriding) setting global: %s=%s " % (name, value))


def createobj_fromini(ini, name, scope, configurations):
	#def get(name):
	#	print "$$$$$$$$$$$$", name
	#	return createobj_fromini(ini, name, scope)
	if enable_logging:
		logger.debug("create: %s" % name)
	items = dict(ini.items(name))
	for conf in configurations:
		keyname = name + ":" + conf
		if keyname in ini.sections():
			items.update(dict(ini.items(keyname)))
	kwargs = {}
	args = None
	classname = None
	constructor = None
	post = None
	#print len(items), "alias" in dict(items), dict(items)
	#scope["get"] = get
	if "alias" in dict(items):
		value = dict(items)["alias"]
		try:
			obj = eval(value, globals(), scope)
		except:
			if enable_logging:
				logger.error("error evaluating alias %s (value is %r)[1]" % (name, value))
			raise
		if enable_logging:
			logger.debug("alias : %s -> %r (expr: %s)" % (name, obj, value))
		#obj = createobj_fromini(ini, dict(items)["alias"], scope)
		#scope["get"] = get
	elif "alias_condition" in dict(items):
		if enable_logging:
			logger.debug("alias (true) %s -> %s" % (name, dict(items)["alias_true"]))
			logger.debug("alias (false) %s -> %s" % (name, dict(items)["alias_false"]))
		expr = dict(items)["alias_condition"]
		try:
			result = eval(expr, globals(), scope)
		except:
			if enable_logging:
				logger.error("error evaluating key %s in section %s (value is %r)[2]" % (arg, name, item))
			raise
		if enable_logging:
			logger.debug("condition: %s == %r" % (expr, result))
		if result:
			obj = createobj_fromini(ini, dict(items)["alias_true"], scope, configurations)
		else:
			obj = createobj_fromini(ini, dict(items)["alias_false"], scope, configurations)
		#scope["get"] = get
	else:
		if "clone" in items:
			cloneditems = dict(ini.items(items["clone"]))
			cloneditems.update(items)
			items = cloneditems
			del items["clone"]
		if "__post__" in items:
			post = items["__post__"]
			del items["__post__"]
		for arg, value in items.items():
			#print name, ":", arg
			if arg == "class":
				classname = value
			elif arg == "constructor":
				constructor = value
			elif arg == "args":
				args = []
				items = value.split(",")
				args = eval(value, globals(), scope)
				if 0:
					for item in items:
						if item in scope:
							item = scope[item]
						elif ini.has_section(item):
							item = createobj_fromini(ini, item, scope, configurations)
							#scope["get"] = get
						else:
							try:
								item = eval(item, globals(), scope)
							except:
								if enable_logging:
									logger.error("error evaluating key %s in section %s (value is %r)[3]" % (arg, name, item))
								raise
						args.append(item)
			else:
				if value in scope:
					value = scope[value]
				elif ini.has_section(value):
					#print "create obj"
					value = createobj_fromini(ini, value, scope, configurations)
					#scope["get"] = get
				elif value == "parameter":
					nestedname = "%s.%s" % (name, arg)
					if nestedname not in scope:
						raise Exception, "missing parameter: %s" % nestedname
					value = scope[nestedname]
				else:
					try:
						value = eval(value, globals(), scope)
					except:
						logger.error("error evaluating key %s in section %s (value is %r)" % (arg, name, value))
						raise
				kwargs[arg] = value
		assert classname is not None, "class attribute not given for section: %s" % name
		fullclassname = classname
		if 0:
			modulename, classname = fullclassname.rsplit(".", 1)
			mod = __import__(modulename)
			obj = mod
			for attrname in fullclassname.split(".")[1:]:
				obj = getattr(obj, attrname)
			#print obj
		try:
			obj = eval(fullclassname, globals(), scope)
		except:
			logger.error("error evaluating classname: %r" % fullclassname)
			raise
		try:
			if constructor:
				args = []
				for param in constructor.split(","):
					args.append(kwargs[param])
				obj = obj(*args)
			else:
				if args:
					obj = obj(*args, **kwargs)
				else:
					obj = obj(**kwargs)
		except:
			logger.error("error constructing object of class: %s (section %s)" % (fullclassname, name))
			raise
			#import pdb;
			#pdb.set_trace()
		#print name
	scope[name] = obj
	if post:
		print "EXECUTING post: %r" % post
		eval(post, globals(), scope)
	return obj

if 0:
	for section in ini.sections():
		print "[%s]" % section
		for name, value in ini.items(section):
			print "%s=%s" % (name, value)
			

#$scope = Scope()
def creategalaxy():
	parameters = {}
	for name, value in ini.items("parameters"):
		parameters[name] = eval(value)
	galaxy = createobj_fromini(ini, "galaxy", parameters, [])
	return galaxy
	
def loadini(modelpath, scope=None, objectname="galaxy", filenames=[], aliases={}, globals=[]):
	global enable_imports
	#assert scope is None
	if scope is None:
		scope = dict()
	#filename = sys.argv[1]
	scope["modelpath"] = modelpath
	#filenames = []
	
	filename = os.path.join(modelpath, "galaxymodel.ini")
	filenames.insert(0, filename)
	filename = os.path.join(modelpath, "galaxy.ini")
	filenames.insert(0, filename)
	
	allfilenames = []
	for filename in filenames:
		ini = configparser.ConfigParser()
		ini.optionxform = str
		if os.path.exists(filename):
			ini.read(filename)
			if "options" in ini.sections():
				values = dict(ini.items("options"))
				if "include" in values:
					dirname = os.path.dirname(filename)
					f = os.path.abspath(os.path.join(dirname, values["include"])) 
					allfilenames.append(f)
					#print values["include"], dirname
					assert os.path.exists(f), "filename %s doesn't exist" % f
			allfilenames.append(filename)
	
	logger.debug("used ini files: %r" % allfilenames)
	#assert os.path.exists(filename), "%s not found" % filename 
	if isinstance(scope, IniScope):
		scope = scope.dict 
	scope = IniScope(allfilenames, aliases, globals, **scope)
	scope.readfiles()
	ini = scope.ini
	#assert os.path.exists(filename), "%s not found" % filename
	#if os.path.exists(filename):
	#	ini.read(filename)
	#for filename in filenames:
	#	ini.read(filename)
	scope.init()
	#createobj_fromini(ini, objectname, scope)
	#print scope.keys()
	#enable_imports = False
	return ini, scope
	
#print creategalaxy()