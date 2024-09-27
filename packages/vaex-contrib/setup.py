import os
from importlib.machinery import SourceFileLoader
from setuptools import setup

dirname = os.path.dirname(__file__)
path_version = os.path.join(dirname, 'vaex/contrib/_version.py')
version = SourceFileLoader('version', path_version).load_module()

name        = 'vaex'
license     = 'MIT'
version     = version.__version__
url         = 'https://www.github.com/vaexio/vaex'
install_requires_contrib = ['vaex-core~=4.0']

setup(name=name + '-contrib',
      version=version,
      description='Community contributed modules to vaex',
      long_description="Community contributed modules to vaex",
      long_description_content_type="text/markdown",
      url=url,
      install_requires=install_requires_contrib,
      extras_require={
            'all': ['google-cloud-bigquery', 'google-cloud-bigquery-storage'],
      },
      license=license,
      packages=['vaex.contrib', 'vaex.contrib.io'],
      include_package_data=True,
      zip_safe=False,
)
