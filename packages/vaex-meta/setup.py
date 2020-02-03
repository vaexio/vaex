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
      'vaex-core>=2.0.0-dev,<3',
      'vaex-viz>=0.4.0-dev,<0.5',
      'vaex-server>=0.3.0-dev,<0.4',
      'vaex-hdf5>=0.6.0-dev,<0.7',
      'vaex-astro>=0.7.0-dev,<0.8',
      'vaex-arrow>=0.5.0-dev,<0.6',
      'vaex-jupyter>=0.5.0-dev,<0.6',
      'vaex-ml>=0.9.0-dev,<0.10',
      # vaex-graphql it not on conda-forge yet
]

setup(name=name,
      version=version,
      description='Out-of-Core DataFrames to visualize and explore big tabular datasets',
      long_description=open('README.rst').read(),
      long_description_content_type='text/plain',
      url=url,
      author=author,
      author_email=author_email,
      install_requires=install_requires,
      license=license,
      packages=['vaex.meta'],
      zip_safe=False,
      )
