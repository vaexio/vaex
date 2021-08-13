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
url = 'https://www.github.com/vaexio/vaex'

install_requires = [
      'vaex-core>=4.4.0,<5',
      'vaex-viz>=0.5.0,<0.6',
      'vaex-server>=0.6.0,<0.7',
      'vaex-hdf5>=0.9.0,<0.10',
      'vaex-astro>=0.8.3,<0.9',
      'vaex-jupyter>=0.6.0,<0.7',
      'vaex-ml>=0.13.0,<0.14',
      # vaex-graphql is not on conda-forge yet
]

setup(name=name,
      version=version,
      description='Out-of-Core DataFrames to visualize and explore big tabular datasets',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      url=url,
      author=author,
      author_email=author_email,
      install_requires=install_requires,
      license=license,
      packages=['vaex.meta'],
      zip_safe=False,
      )
