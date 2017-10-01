from setuptools import setup
import sys, os, imp

import sys, os, imp
from setuptools import Extension

import vaex.version as version
# dirname = os.path.dirname(__file__)
# path_version = os.path.join(dirname, "vaex/version.py")
# version = imp.load_source('version', path_version)

name        = 'vaex'
author      = "Maarten A. Breddels",
author_email= "maartenbreddels@gmail.com",
license     = 'MIT'
version     = version.versionstring
url         = 'https://www.github.com/maartenbreddels/vaex'
install_requires_hdf5 = ["vaex-core>=0.1", "h5py"]

setup(name=name+'-viz',
    version=version,
    description='Visualization for vaex',
    url=url,
    author=author,
    author_email=author_email,
    install_requires=install_requires_viz,
    license=license,
    packages=['vaex.viz'],
    zip_safe=False,)