import os
from setuptools import setup

dirname = os.path.dirname(__file__)
path_version = os.path.join(dirname, "vaex_arrow/_version.py")
version = {}
with open(path_version) as f:
    exec(f.read(), version)
version = version['__version__']


name = 'vaex'
author = "Maarten A. Breddels"
author_email = "maartenbreddels@gmail.com"
license = 'MIT'
url = 'https://www.github.com/maartenbreddels/vaex'
install_requires = ["vaex-core>=2.0.0,<3", "astropy>=2", "matplotlib>=1.3.1", "pillow"]

setup(name=name + '-arrow',
      version=version,
      description='Arrow support for vaex',
      long_description='Arrow support for vaex',
      long_description_content_type="text/markdown",
      url=url,
      author=author,
      author_email=author_email,
      install_requires=install_requires,
      license=license,
      packages=['vaex_arrow'],
      zip_safe=False,
      entry_points={'vaex.plugin': ['reader = vaex_arrow.opener:register_opener']}
      )
