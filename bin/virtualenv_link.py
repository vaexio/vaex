__author__ = 'maartenbreddels'
import os
import sys
venv = sys.argv[1]

for name in sys.argv[2:]:
	module = __import__(name)
	if os.path.split(module.__file__)[-1].startswith("__init__"):
		source = os.path.dirname(module.__file__)
	else:
		source = module.__file__
	name = os.path.split(source)[-1]
	script = os.path.join(os.path.dirname(__file__), "virtualenv_link2.py")
	script
	os.system("source virtualenv/{venv}/bin/activate; python {script} {source} {name}".format(**locals()))
#code = from distutils.sysconfig import get_python_lib; print(get_python_lib())
#cmd =
#target = sys.argv[1]
