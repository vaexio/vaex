# only for local development / install from source, e.g.
# pip install https://github.com/vaexio/vaex/archive/refs/heads/master.zip

from pathlib import Path

from setuptools import setup

packages = [
    "vaex-core",
    "vaex-viz",
    "vaex-hdf5",
    "vaex-server",
    "vaex-astro",
    "vaex-jupyter",
    "vaex-ml",
    "vaex-graphql",
    "vaex-contrib",
    "vaex",
]

setup(
    name="vaex-meta",
    version="0.1.0",
    description="Convenience setup.py for when installing from the git repo",
    classifiers=[
        "Private :: Do Not Upload to pypi server",
    ],
    packages=[],
    install_requires=[
        f"{package}[all] @ {(Path(__file__).parent / 'packages' / package).as_uri()}"
        for package in packages
    ],
    extras_require={
        "ci": [
            # readthedocs
            "sphinx",
            "sphinx_book_theme",
            "sphinx_sitemap",
            "sphinxcontrib_googleanalytics",
            "sphinxext_rediraffe",
            "sphinx_gallery",
            "nbsphinx",
            "jupyter_sphinx",
            "myst_parser",
            # tests
            "pytest",
            "pytest-asyncio",
            "pytest-mpl",
            "pytest-timeout",
            # ipynb tests
            "nbconvert",
            "jupyterlab",
            "plotly",
            # https://github.com/vaexio/vaex/pull/2356#issuecomment-2320707228
            "graphene-tornado @ https://github.com/ddelange/graphene-tornado/archive/refs/heads/2.6.1.unpin-werkzeug.zip",
        ]
    },
)
