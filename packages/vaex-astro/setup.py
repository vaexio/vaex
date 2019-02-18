import os
import imp
from setuptools import setup

dirname = os.path.dirname(__file__)
path_version = os.path.join(dirname, 'vaex/astro/_version.py')
version = imp.load_source('version', path_version)

name        = 'vaex'
author      = 'Maarten A. Breddels'
author_email= 'maartenbreddels@gmail.com'
license     = 'MIT'
version     = version.__version__
url         = 'https://www.github.com/maartenbreddels/vaex'
install_requires_astro = ['vaex-core>=0.1']

setup(name=name + '-astro',
      version=version,
      description='Astronomy related transformations and FITS file support',
      url=url,
      author=author,
      author_email=author_email,
      install_requires=install_requires_astro,
      license=license,
      packages=['vaex.astro'],
      zip_safe=False,
      entry_points={'vaex.plugin': ['astro = vaex.astro.transformations:add_plugin']}
      )
