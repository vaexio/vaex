import os
from importlib.machinery import SourceFileLoader
from setuptools import setup

dirname = os.path.dirname(__file__)
path_version = os.path.join(dirname, "vaex/viz/_version.py")
version = SourceFileLoader('version', path_version).load_module()


name = 'vaex'
author = "Maarten A. Breddels"
author_email = "maartenbreddels@gmail.com"
license = 'MIT'
version = version.__version__
url = 'https://www.github.com/maartenbreddels/vaex'
install_requires_viz = ["vaex-core>=4.0.0,<5", "matplotlib>=1.3.1", "pillow"]

setup(name=name + '-viz',
      version=version,
      description='Visualization for vaex',
      long_description="Visualization for vaex",
      long_description_content_type="text/markdown",
      url=url,
      author=author,
      author_email=author_email,
      install_requires=install_requires_viz,
      license=license,
      packages=['vaex.viz'],
      zip_safe=False,
      entry_points={'vaex.dataframe.accessor': ['viz = vaex.viz:DataFrameAccessorViz'],
                    'vaex.expression.accessor': ['viz = vaex.viz:ExpressionAccessorViz'],
                    'vaex.plugin': ['plot = vaex.viz.mpl:add_plugin']}
      )
