
major = 1
minor = 0
patch = 0
pre_release = "beta.3"

#versiontring = '0.2.42-g54a6613'
versiontuple = (major, minor, patch)
versionstring = "%d.%d.%d" % versiontuple
if pre_release:
	versionstring += "-" + pre_release

import platform
# from vaex.utils import osname, setup.py doesn't want imports...
osname = dict(darwin="osx", linux="linux", windows="windows")[platform.system().lower()]

if __name__ == "__main__":
	import vaex
	import sys
	#print vaex.__version_tuple__
	if sys.argv[1] == "version":
		print("version:", vaex.__version__)
	elif sys.argv[1] == "fullname":
		print("full name:", vaex.__full_name__)
	elif sys.argv[1] == "buildname":
		print("build name:", vaex.__build_name__)
	elif sys.argv[1] == "tagcmd":
		print("git tag %s" % versionstring)
		print("git push --tags")
	else:
		print("use version, fullname or buildname as argument")
