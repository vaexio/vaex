import os
import imp
from numpy.lib.function_base import extract
from setuptools import setup

dirname = os.path.dirname(__file__)
path_version = os.path.join(dirname, 'vaex/contrib/_version.py')
version = imp.load_source('version', path_version)

name        = 'vaex'
license     = 'MIT'
version     = version.__version__
url         = 'https://www.github.com/vaexio/vaex'
install_requires_contrib = ['vaex-core>=4.0.0,<5']

setup(name=name + '-contrib',
      version=version,
      description='Community contributed modules to vaex',
      url=url,
      install_requires=install_requires_contrib,
      extras_require={
            'gcp': ['google-cloud-bigquery', 'google-cloud-bigquery-storage'],
      },
      license=license,
      packages=['vaex.contrib', 'vaex.contrib.io'],
      include_package_data=True,
      zip_safe=False,
)
