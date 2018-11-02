import os
import imp
from setuptools import setup

dirname = os.path.dirname(__file__)
path_version = os.path.join(dirname, "vaex_arrow/_version.py")
version = imp.load_source('version', path_version)


name = 'vaex'
author = "Maarten A. Breddels"
author_email = "maartenbreddels@gmail.com"
license = 'MIT'
version = version.__version__
url = 'https://www.github.com/maartenbreddels/vaex'
install_requires = ["vaex-core>=0.4.2", "matplotlib>=1.3.1", "pillow", "pyarrow"]

setup(name=name + '-arrow',
      version=version,
      description='Arrow support for vaex',
      url=url,
      author=author,
      author_email=author_email,
      install_requires=install_requires,
      license=license,
      packages=['vaex_arrow'],
      zip_safe=False,
      entry_points={'vaex.plugin': ['reader = vaex_arrow.opener:register_opener']}
      )
