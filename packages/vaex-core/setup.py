from setuptools import setup
import sys
import os
import imp
from setuptools import Extension
import platform

use_skbuild = len(os.environ.get('VAEX_BUILD_SKBUILD', '')) > 0

if use_skbuild:
    from skbuild import setup
    import skbuild.command.build_ext

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
install_requires_core = ["numpy>=1.16", "aplus", "tabulate>=0.8.3",
                         "future>=0.15.2", "pyyaml", "progressbar2",
                         "requests", "six", "cloudpickle", "pandas", "dask!=2022.4.0",
                         "nest-asyncio>=1.3.3", "pyarrow>=3.0", "frozendict!=2.2.0",
                         "blake3", "filelock", "pydantic>=1.8.0", "rich",
                        ]
if sys.version_info[0] == 2:
    install_requires_core.append("futures>=2.2.0")
install_requires_viz = ["matplotlib>=1.3.1", ]
install_requires_astro = ["kapteyn"]

if "MACOSX_DEPLOYMENT_TARGET" not in os.environ:
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = "10.9"

extra_dev_options = []
# MB: I like these options during development, the second if for ccache
# extra_dev_options = ['-fmax-errors=4', '-fdiagnostics-color', '-pedantic-errors']

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


USE_ABSL = False
USE_TSL = True

define_macros = []
if USE_ABSL:
    define_macros += [('VAEX_USE_ABSL', None)]
if USE_TSL:
    define_macros += [('VAEX_USE_TSL', None)]

dll_files = []
if platform.system().lower() == 'windows':
    extra_compile_args = ["/EHsc"]
    dll_files = ['pcre.dll', 'pcrecpp.dll', 'vcruntime140_1.dll']
else:
    # TODO: maybe enable these flags for non-wheel/conda builds? ["-mtune=native", "-march=native"]
    extra_compile_args = ["-std=c++11", "-O3", "-funroll-loops", "-Werror=return-type", "-Wno-unused-parameter"]
    extra_compile_args.append("-g")
    extra_compile_args += extra_dev_options
if sys.platform == 'darwin':
    extra_compile_args.append("-mmacosx-version-min=10.9")


# on windows (Conda-forge builds), the dirname is an absolute path
extension_vaexfast = Extension("vaex.vaexfast", [os.path.relpath(os.path.join(dirname, "src/vaexfast.cpp"))],
                               include_dirs=[get_numpy_include()],
                               extra_compile_args=extra_compile_args)
extension_strings = Extension("vaex.superstrings", [
    os.path.relpath(os.path.join(dirname, "src/strings.cpp")),
    os.path.relpath(os.path.join(dirname, "src/string_utils.cpp")),
    ],
    include_dirs=[
        get_numpy_include(),
        get_pybind_include(),
        get_pybind_include(user=True),
        'vendor/string-view-lite/include',
        'vendor/boost',
        os.path.join(sys.prefix, 'include'),
        os.path.join(sys.prefix, 'Library', 'include'), # windows
        os.path.join(dirname, 'vendor', 'pcre', 'Library', 'include') # windows pcre from conda-forge
    ],
    library_dirs=[
        os.path.join(sys.prefix, 'lib'),
        os.path.join(sys.prefix, 'Library', 'lib'), # windows
        os.path.join(dirname, 'vendor', 'pcre', 'Library', 'lib'), # windows pcre from conda-forge
    ],
    extra_compile_args=extra_compile_args,
    libraries=['pcre', 'pcrecpp']
)
extension_superutils = Extension("vaex.superutils", [
        os.path.relpath(os.path.join(dirname, "src/hash_string.cpp")),
        os.path.relpath(os.path.join(dirname, "src/hash_primitives_pot.cpp")),
        os.path.relpath(os.path.join(dirname, "src/hash_object.cpp")),
        os.path.relpath(os.path.join(dirname, "src/hash_primitives_prime.cpp")),
        os.path.relpath(os.path.join(dirname, "src/superutils.cpp")),
        os.path.relpath(os.path.join(dirname, "src/string_utils.cpp")),
    ] + ([os.path.relpath(os.path.join(dirname, "vendor/abseil-cpp/absl/container/internal/raw_hash_set.cc"))] if USE_ABSL else []),
    include_dirs=[
        get_numpy_include(), get_pybind_include(),
        get_pybind_include(user=True),
        'vendor/abseil-cpp',
        'vendor/flat_hash_map',
        'vendor/sparse-map/include',
        'vendor/hopscotch-map/include',
        'vendor/string-view-lite/include',
    ],
    extra_compile_args=extra_compile_args,
    define_macros=define_macros,
    )

extension_superagg = Extension("vaex.superagg", [
        os.path.relpath(os.path.join(dirname, "src/agg_nunique_string.cpp")),
        os.path.relpath(os.path.join(dirname, "src/agg_minmax.cpp")),
        os.path.relpath(os.path.join(dirname, "src/agg_nunique.cpp")),
        os.path.relpath(os.path.join(dirname, "src/agg_sum.cpp")),
        os.path.relpath(os.path.join(dirname, "src/agg_first.cpp")),
        os.path.relpath(os.path.join(dirname, "src/agg_list.cpp")),
        os.path.relpath(os.path.join(dirname, "src/agg_count.cpp")),
        os.path.relpath(os.path.join(dirname, "src/agg.cpp")),
        os.path.relpath(os.path.join(dirname, "src/binner_combined.cpp")),
        os.path.relpath(os.path.join(dirname, "src/binner_ordinal.cpp")),
        os.path.relpath(os.path.join(dirname, "src/binner_hash.cpp")),
        os.path.relpath(os.path.join(dirname, "src/binners.cpp")),
        os.path.relpath(os.path.join(dirname, "src/string_utils.cpp")),
    ],
    include_dirs=[
        get_numpy_include(), get_pybind_include(),
        get_pybind_include(user=True),
        'vendor/flat_hash_map',
        'vendor/sparse-map/include',
        'vendor/hopscotch-map/include',
        'vendor/string-view-lite/include'
    ],
    extra_compile_args=extra_compile_args,
    define_macros=define_macros,
    )

setup(name=name + '-core',
      version=version,
      description='Core of vaex',
      url=url,
      author=author,
      author_email=author_email,
      setup_requires=['numpy'],
      install_requires=install_requires_core,
      license=license,
      package_data={'vaex': dll_files + ['test/files/*.fits', 'test/files/*.vot', 'test/files/*.hdf5']},
      packages=['vaex', 'vaex.arrow', 'vaex.core', 'vaex.file', 'vaex.test', 'vaex.ext', 'vaex.misc', 'vaex.datasets'],
      include_package_data=True,
      ext_modules=([extension_vaexfast] if on_rtd else [extension_vaexfast, extension_strings, extension_superutils, extension_superagg]) if not use_skbuild else [],
      zip_safe=False,
      extras_require={
          'all': ["gcsfs>=0.6.2", "s3fs"]
      },
      entry_points={
          'console_scripts': ['vaex = vaex.__main__:main'],
          'gui_scripts': ['vaexgui = vaex.__main__:main'],  # sometimes in osx, you need to run with this
          'vaex.dataframe.accessor': [
              'geo = vaex.geo:DataFrameAccessorGeo',
              'struct = vaex.struct:DataFrameAccessorStruct',
          ],
          'vaex.dataset.opener': [
              'arrow = vaex.arrow.opener:ArrowOpener',
              'parquet = vaex.arrow.opener:ParquetOpener',
              'feather = vaex.arrow.opener:FeatherOpener',
          ],
          'vaex.memory.tracker': [
              'default = vaex.memory:MemoryTracker'
          ],
          'vaex.progressbar': [
              'vaex = vaex.progress:simple',
              'simple = vaex.progress:simple',
              'widget = vaex.progress:widget',
              'rich = vaex.progress:rich',
          ],
          'vaex.file.scheme': [
              's3 = vaex.file.s3',
              'fsspec+s3 = vaex.file.s3fs',
              'arrow+s3 = vaex.file.s3arrow',
              'gs = vaex.file.gcs',
              'fsspec+gs = vaex.file.gcs',
          ]
      }
      )
