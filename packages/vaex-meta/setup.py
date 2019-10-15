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
      'vaex-core>=1.2.0,<2',
      'vaex-viz>=0.3.8,<0.4',
      'vaex-server>=0.2.1,<0.3',
      'vaex-hdf5>=0.5.5,<0.6',
      'vaex-astro>=0.6.1,<0.7',
      'vaex-arrow>=0.4.1,<0.5',
      # vaex-jupyter # want want to depend on this soon
      # 'vaex-ml==0.6.1' # we cannot depend yet on this because it is not on conda-forge yet
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
