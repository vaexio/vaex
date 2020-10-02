import os
import imp
from setuptools import setup

dirname = os.path.dirname(__file__)
path_version = os.path.join(dirname, "vaex/viz/_version.py")
version = imp.load_source('version', path_version)


name = 'vaex'
author = "Maarten A. Breddels"
author_email = "maartenbreddels@gmail.com"
license = 'MIT'
version = version.__version__
url = 'https://www.github.com/maartenbreddels/vaex'
install_requires_viz = ["vaex-core>=4.0.0-alpha.3,<5", "matplotlib>=1.3.1", "pillow"]

setup(name=name + '-viz',
      version=version,
      description='Visualization for vaex',
      url=url,
      author=author,
      author_email=author_email,
      install_requires=install_requires_viz,
      license=license,
      packages=['vaex.viz'],
      zip_safe=False,
      entry_points={'vaex.dataframe.accessor': ['viz = vaex.viz:DataFrameAccessorViz'],
                    'vaex.plugin': ['plot = vaex.viz.mpl:add_plugin']}
      )
