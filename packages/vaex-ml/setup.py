import os
from importlib.machinery import SourceFileLoader
from setuptools import setup

dirname = os.path.dirname(__file__)
path_version = os.path.join(dirname, 'vaex/ml/_version.py')
version = SourceFileLoader('version', path_version).load_module()

name        = 'vaex'
author      = 'Jovan Veljanoski'
author_email= 'jovan.veljanoski@gmail.com'
license     = 'MIT'
version     = version.__version__
url         = 'https://www.github.com/vaexio/vaex'
install_requires_ml = [
      'vaex-core~=4.8',
      'numba',
      'traitlets',
      'jinja2',
      'annoy',
      'scikit-learn',
      'xgboost',
      'lightgbm~=4.0',
      'catboost',
]
extras_require_ml = {'all': ['tensorflow~=2.1']}

setup(name=name + '-ml',
      version=version,
      description='Machine learning support for vaex',
      url=url,
      author=author,
      author_email=author_email,
      install_requires=install_requires_ml,
      extras_require=extras_require_ml,
      license=license,
      packages=['vaex.ml', 'vaex.ml.incubator'],
      include_package_data=True,
      zip_safe=False,
      entry_points={'vaex.dataframe.accessor': ['ml = vaex.ml:DataFrameAccessorML',
                                                'ml.tensorflow = vaex.ml.tensorflow:DataFrameAccessorTensorflow',
                                                'ml.metrics = vaex.ml.metrics:DataFrameAccessorMetrics']}
)
