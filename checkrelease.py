__author__ = 'maartenbreddels'


import os
import sys

def system(cmd):
	print "Executing: ", cmd
	ret = os.system(cmd)
	if ret != 0:
		print "error, return code is", ret
		sys.exit(ret)

system("version=`git describe --tags --long`; python/gavi/vaex/setversion.py ${version}")



import imp
import gavi.vaex


print "version is %r" % gavi.vaex.__version__
print "release is %r" % gavi.vaex.__clean_release__

path_conf = "docs/source/conf.py"
#path_conf = "conf.py"
conf = imp.load_source('conf', path_conf)
ok = True
if conf.version != gavi.vaex.__version__:
	print "version for documentation is %r, while current is %r" % (conf.version, gavi.vaex.__version__)
	ok = False
if conf.release != gavi.vaex.__clean_release__:
	print "release for documentation is %r, while current is %r" % (conf.release, gavi.vaex.__clean_release__)
	ok = False
if not ok:
	print "Replacing version"
	system("sed -i.bak 's/version = .*/version = \"%s\"/g' %s" % (gavi.vaex.__version__, path_conf))
	system("sed -i.bak 's/release = .*/release = \"%s\"/g' %s" % (gavi.vaex.__clean_release__, path_conf))


system("sed -i.bak 's/version = .*/version = \"%s\"/g' %s" % (gavi.vaex.__version__, "setup.py"))
