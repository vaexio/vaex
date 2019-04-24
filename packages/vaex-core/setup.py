from setuptools import setup
import sys
import os
import imp
from setuptools import Extension
import platform

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

dirname = os.path.dirname(__file__)
path_version = os.path.join(dirname, "vaex/core/_version.py")
version = imp.load_source('version', path_version)

name = 'vaex'
author = "Maarten A. Breddels"
author_email = "maartenbreddels@gmail.com"
license = 'MIT'
version = version.__version__
url = 'https://www.github.com/maartenbreddels/vaex'
# TODO: can we do without requests and progressbar2?
# TODO: after python2 supports frops, future and futures can also be dropped
# TODO: would be nice to have astropy only as dep in vaex-astro
install_requires_core = ["numpy>=1.11", "astropy>=2", "aplus", "tabulate>=0.8.3",
                         "future>=0.15.2", "pyyaml", "progressbar2", "psutil>=1.2.1",
                         "requests", "six", "cloudpickle"]
if sys.version_info[0] == 2:
    install_requires_core.append("futures>=2.2.0")
install_requires_viz = ["matplotlib>=1.3.1", ]
install_requires_astro = ["kapteyn"]

if "MACOSX_DEPLOYMENT_TARGET" not in os.environ:
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = "10.9"


class get_numpy_include(object):
    """Helper class to determine the numpy include path
    The purpose of this class is to postpone importing numpy
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self):
        pass

    def __str__(self):
        import numpy as np
        return np.get_include()

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        # this trick does not work anymore it seems, we now just vendor it
        # import pybind11
        # return pybind11.get_include(self.user)
        return 'vendor/pybind11/include'

if platform.system().lower() == 'windows':
    extra_compile_args = ["/EHsc"]
else:
    # TODO: maybe enable these flags for non-wheel/conda builds? ["-mtune=native", "-march=native"]
    extra_compile_args = ["-std=c++11", "-mfpmath=sse", "-O3", "-funroll-loops"]
    extra_compile_args.append("-g")
if sys.platform == 'darwin':
    extra_compile_args.append("-mmacosx-version-min=10.9")

# on windows (Conda-forge builds), the dirname is an absolute path
extension_vaexfast = Extension("vaex.vaexfast", [os.path.relpath(os.path.join(dirname, "src/vaexfast.cpp"))],
                               include_dirs=[get_numpy_include()],
                               extra_compile_args=extra_compile_args)
extension_strings = Extension("vaex.superstrings", [os.path.relpath(os.path.join(dirname, "src/strings.cpp"))],
                               include_dirs=[
                                   get_numpy_include(),
                                   get_pybind_include(),
                                   get_pybind_include(user=True),
                                   'vendor/string-view-lite/include',
                                   'vendor/boost',
                                   os.path.join(sys.prefix, 'include'),
                                   os.path.join(sys.prefix, 'Library', 'include') # windows
                               ],
                               library_dirs=[
                                   os.path.join(sys.prefix, 'lib'),
                                   os.path.join(sys.prefix, 'Library', 'lib') # windows
                               ],
                               extra_compile_args=extra_compile_args,
                               libraries=['pcre', 'pcrecpp']
                               )
extension_superutils = Extension("vaex.superutils", [
        os.path.relpath(os.path.join(dirname, "src/hash_object.cpp")),
        os.path.relpath(os.path.join(dirname, "src/hash_primitives.cpp")),
        os.path.relpath(os.path.join(dirname, "src/superutils.cpp")),
        os.path.relpath(os.path.join(dirname, "src/hash_string.cpp")),
    ],
    include_dirs=[
        get_numpy_include(), get_pybind_include(),
        get_pybind_include(user=True),
        'vendor/flat_hash_map',
        'vendor/sparse-map/include',
        'vendor/hopscotch-map/include',
        'vendor/string-view-lite/include'
    ],
    extra_compile_args=extra_compile_args)

extension_superagg = Extension("vaex.superagg", [
        os.path.relpath(os.path.join(dirname, "src/superagg.cpp")),
    ],
    include_dirs=[
        get_numpy_include(), get_pybind_include(),
        get_pybind_include(user=True),
        'vendor/flat_hash_map',
        'vendor/sparse-map/include',
        'vendor/hopscotch-map/include',
        'vendor/string-view-lite/include'
    ],
    extra_compile_args=extra_compile_args)

setup(name=name + '-core',
      version=version,
      description='Core of vaex',
      url=url,
      author=author,
      author_email=author_email,
      setup_requires=['numpy'],
      install_requires=install_requires_core,
      license=license,
      package_data={'vaex': ['test/files/*.fits', 'test/files/*.vot', 'test/files/*.hdf5']},
      packages=['vaex', 'vaex.core', 'vaex.file', 'vaex.test', 'vaex.ext', 'vaex.misc'],
      ext_modules=[extension_vaexfast] if on_rtd else [extension_vaexfast, extension_strings, extension_superutils, extension_superagg],
      zip_safe=False,
      entry_points={
          'console_scripts': ['vaex = vaex.__main__:main'],
          'gui_scripts': ['vaexgui = vaex.__main__:main']  # sometimes in osx, you need to run with this
      }
      )
