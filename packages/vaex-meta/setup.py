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
      'vaex-core>=2.0.0,<3',
      'vaex-viz>=0.4.0,<0.5',
      'vaex-server>=0.3.1,<0.4',
      'vaex-hdf5>=0.6.0,<0.7',
      'vaex-astro>=0.7.0,<0.8',
      'vaex-jupyter>=0.5.2,<0.6',
      'vaex-ml>=0.10.0,<0.11',
      # vaex-graphql it not on conda-forge yet
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
