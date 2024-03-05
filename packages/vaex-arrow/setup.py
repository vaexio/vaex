import os
from setuptools import setup
from importlib.machinery import SourceFileLoader

dirname = os.path.dirname(__file__)
path_version = os.path.join(dirname, "vaex_arrow/_version.py")
version = SourceFileLoader('version', path_version).load_module()


name = 'vaex'
author = "Maarten A. Breddels"
author_email = "maartenbreddels@gmail.com"
license = 'MIT'
version = version.__version__
url = 'https://www.github.com/maartenbreddels/vaex'
install_requires = ["vaex-core>=2.0.0,<3", "astropy>=2", "matplotlib>=1.3.1", "pillow", "pyarrow>=0.15"]

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
