from __future__ import print_function
__author__ = 'maartenbreddels'
import sys
import os
from distutils.sysconfig import get_python_lib;

source, name = sys.argv[1:]

target = os.path.join(get_python_lib(), name)
cmd = "ln -s {source} {target}".format(**locals())
print(cmd)
os.system(cmd)