import os
import sys
import vaex

if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(__file__)

#print "application path", application_path

def iconfile(name):
	dirname_2 = os.path.join(os.path.dirname(sys.argv[0]))
	for dirname in [dirname_2, "./vaex", ".", os.path.dirname(vaex.__file__), os.path.join(application_path), os.path.join(application_path, "..")] :
		path = os.path.join(dirname, "icons", name+".png")
		if os.path.exists(path):
			#print "icon path:", path
			return path
		else:
			#print "path", path, "does not exist"
			pass
	return path
