import os
from setuptools import setup

dirname = os.path.dirname(__file__)
path_version = os.path.join(dirname, 'vaex/ml/_version.py')
version = {}
with open(path_version) as f:
    exec(f.read(), version)
version = version['__version__']

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
      'scikit-learn',
      'xgboost',
      'lightgbm~=4.0',
      'catboost>=1.2.8',
]

extras_require_ml = {
    'all': [
        # wheel facts as of TF 2.21.0 (recheck with new TF releases): cp313 wheels
        # start at 2.20, (2.22.0rc0 is the only release with cp314 wheels), so on
        # 3.14 no tensorflow entry on purpose: vaex-ml[all] leaves TF absent there
        # until a stable release ships cp314 wheels
        "tensorflow~=2.18;python_version<'3.13'",
        "tensorflow>=2.20,<3;python_version=='3.13'",
        # https://stackoverflow.com/questions/79744362/import-tensorflow-statement-crashes-or-hangs-on-macos/79744363#79744363
        "tensorflow<2.20.0;sys_platform=='darwin' and python_version<'3.13'",
        # annoy is unmaintained, ships only an sdist that fails to build on 3.13+;
        # optional (use vaex-ml[annoy] to force it)
        "annoy;python_version<'3.13'",
    ],
    'annoy': ['annoy'],
}

setup(name=name + '-ml',
      version=version,
      description='Machine learning support for vaex',
      long_description="Machine learning support for vaex",
      long_description_content_type="text/markdown",
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
