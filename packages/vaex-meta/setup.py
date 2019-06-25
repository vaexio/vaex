import os
import imp
from setuptools import setup
from setuptools import Extension

dirname = os.path.dirname(__file__)
path_version = os.path.join(dirname, 'vaex/meta/_version.py')
version = imp.load_source('version', path_version)


name = 'vaex'
author = 'Maarten A. Breddels'
author_email = 'maartenbreddels@gmail.com'
license = 'MIT'
version = version.__version__
url = 'https://www.github.com/maartenbreddels/vaex'

install_requires = [
      'vaex-core==0.7.3',
      'vaex-viz==0.3',
      'vaex-server==0.2',
      'vaex-hdf5==0.4',
      'vaex-astro==0.4',
      'vaex-arrow==0.3',
      'vaex-ml==0.4'
]

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
