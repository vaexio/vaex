import os
import imp
from setuptools import setup

dirname = os.path.dirname(__file__)
path_version = os.path.join(dirname, 'vaex/plotly/_version.py')
version = imp.load_source('version', path_version)

name        = 'vaex'
author      = 'Jovan Veljanoski'
author_email= 'jovan.veljanoski@gmail.com'
license     = 'MIT'
version     = version.__version__
url         = 'https://www.github.com/vaexio/vaex'
install_requires_plotly = ['plotly>=4.1.1']

setup(name=name + '-plotly',
      version=version,
      description='Visualisation for vaex using plotly',
      url=url,
      author=author,
      author_email=author_email,
      install_requires=install_requires_plotly,
      license=license,
      packages=['vaex.plotly'],
      include_package_data=True,
      zip_safe=False,
      entry_points={'vaex.dataframe.accessor': ['plotly = vaex.plotly:DataFrameAccessorPlotly']}
)
