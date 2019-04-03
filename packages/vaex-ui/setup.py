import os
import imp
from setuptools import setup
from setuptools import Extension

dirname = os.path.dirname(__file__)
path_version = os.path.join(dirname, "vaex/ui/_version.py")
version = imp.load_source('version', path_version)

name = 'vaex'
author = "Maarten A. Breddels"
author_email = "maartenbreddels@gmail.com"
license = 'MIT'
version = version.__version__
url = 'https://www.github.com/maartenbreddels/vaex'
install_requires_ui = ["vaex-core>=0.6.1", "PyOpenGL", "scipy", "matplotlib", "vaex-hdf5", "vaex-astro"]

setup(name=name + '-ui',
      version=version,
      description='Graphical user interface for vaex based on Qt',
      url=url,
      author=author,
      author_email=author_email,
      install_requires=install_requires_ui,
      include_package_data=True,  # include files listed in manifest.in
      license=license,
      packages=['vaex.ui', 'vaex.ui.plugin', 'vaex.ui.icons'],
      zip_safe=False
      )
