from setuptools import setup
import sys, os, imp

import sys, os, imp
from setuptools import Extension

dirname = os.path.dirname(__file__)
path_version = os.path.join(dirname, "vaex/ui/_version.py")
version = imp.load_source('version', path_version)

name        = 'vaex'
author      = "Maarten A. Breddels"
author_email= "maartenbreddels@gmail.com"
license     = 'MIT'
version     = version.__version__
url         = 'https://www.github.com/maartenbreddels/vaex'
install_requires_ui = ["vaex-core>=0.1", "PyOpenGL", "scipy", "matplotlib", "vaex-hdf5", "vaex-astro"]

setup(name=name+'-ui',
    version=version,
    description='Graphical user interface for vaex based on Qt',
    url=url,
    author=author,
    author_email=author_email,
    install_requires=install_requires_ui,
    license=license,
    packages=['vaex.ui'],
    zip_safe=False
    )
