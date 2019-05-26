import os
import imp
from setuptools import setup

dirname = os.path.dirname(__file__)
path_version = os.path.join(dirname, 'vaex/ml/_version.py')
version = imp.load_source('version', path_version)

name        = 'vaex'
author      = 'Jovan Veljanoski'
author_email= 'jovan.veljanoski@gmail.com'
license     = 'MIT'
version     = version.__version__
url         = 'https://www.github.com/vaexio/vaex'
install_requires_ml = ['vaex-core>=0.6', 'numba', 'traitlets','jinja2']

setup(name=name + '-ml',
      version=version,
      description='Machine learning support for vaex',
      url=url,
      author=author,
      author_email=author_email,
      install_requires=install_requires_ml,
      license=license,
      packages=['vaex.ml', 'vaex.ml.incubator', 'vaex.ml.datasets'],
      include_package_data=True,
      zip_safe=False,
    )
