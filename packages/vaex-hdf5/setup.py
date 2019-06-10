import os
import imp
from setuptools import setup
from setuptools import Extension

dirname = os.path.dirname(__file__)
path_version = os.path.join(dirname, "vaex/hdf5/_version.py")
version = imp.load_source('version', path_version)

name = 'vaex'
author = "Maarten A. Breddels"
author_email = "maartenbreddels@gmail.com"
license = 'MIT'
version = version.__version__
url = 'https://www.github.com/maartenbreddels/vaex'
install_requires_hdf5 = ["vaex-core>=0.8", "h5py>=2.9", "s3fs"]

setup(name=name + '-hdf5',
      version=version,
      description='hdf5 file support for vaex',
      url=url,
      author=author,
      author_email=author_email,
      install_requires=install_requires_hdf5,
      license=license,
      packages=['vaex.hdf5'],
      zip_safe=False,)
