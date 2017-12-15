from setuptools import setup
import sys, os, imp

import sys, os, imp
from setuptools import Extension

dirname = os.path.dirname(__file__)
path_version = os.path.join(dirname, 'vaex/meta/_version.py')
version = imp.load_source('version', path_version)


name        = 'vaex'
author      = 'Maarten A. Breddels'
author_email= 'maartenbreddels@gmail.com'
license     = 'MIT'
version     = version.__version__
url         = 'https://www.github.com/maartenbreddels/vaex'

install_requires = 'vaex-core vaex-viz vaex-server vaex-ui vaex-hdf5 vaex-astro'.split()

setup(name=name,
    version=version,
    description='Out-of-Core DataFrames to visualize and explore big tabular datasets',
    long_description=open('README.rst').read(),
    url=url,
    author=author,
    author_email=author_email,
    install_requires=install_requires,
    license=license,
    packages=['vaex.meta'],
    zip_safe=False,
    )
