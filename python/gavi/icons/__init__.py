import os
import sys
import gavi

if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(__file__)

print "application path", application_path
def iconfile(name):
	for dirname in ["./python/gavi", ".", os.path.dirname(gavi.__file__), os.path.join(application_path), os.path.join(application_path, "..")] :
		path = os.path.join(dirname, "icons", name+".png")
		if os.path.exists(path):
			print "icon path:", path
			return path
		else:
			print "path", path, "does not exist"
	return path
