import os
import imp
from setuptools import setup
from setuptools import Extension

dirname = os.path.dirname(__file__)
path_version = os.path.join(dirname, 'vaex/jupyter/_version.py')
version = imp.load_source('version', path_version)


name = 'vaex'
author = 'Maarten A. Breddels'
author_email = 'maartenbreddels@gmail.com'
license = 'MIT'
version = version.__version__
url = 'https://www.github.com/maartenbreddels/vaex'
install_requires_jupyter = ['vaex-core>=0.6', 'vaex-viz', 'bqplot>=0.10.1', 'ipyvolume>=0.4', 'ipyleaflet', 'ipympl', 'ipyvuetify']

setup(name=name + '-jupyter',
      version=version,
      description='Jupyter notebook and Jupyter lab support for vaex',
      url=url,
      author=author,
      author_email=author_email,
      install_requires=install_requires_jupyter,
      license=license,
      packages=['vaex.jupyter'],
      zip_safe=False,
      entry_points={'vaex.namespace': ['widget = vaex.jupyter:add_namespace']},
      include_package_data=True,  # include files listed in manifest.in
      )
