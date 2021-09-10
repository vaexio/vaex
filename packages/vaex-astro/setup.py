import os
import imp
from setuptools import setup

dirname = os.path.dirname(__file__)
path_version = os.path.join(dirname, 'vaex/astro/_version.py')
version = imp.load_source('version', path_version)

name        = 'vaex'
author      = 'Maarten A. Breddels'
author_email= 'maartenbreddels@gmail.com'
license     = 'MIT'
version     = version.__version__
url         = 'https://www.github.com/maartenbreddels/vaex'
install_requires_astro = ['vaex-core>=4.5.0,<5', 'astropy']

setup(
    name=name + '-astro',
    version=version,
    description='Astronomy related transformations and FITS file support',
    url=url,
    author=author,
    author_email=author_email,
    install_requires=install_requires_astro,
    license=license,
    packages=['vaex.astro'],
    zip_safe=False,
    entry_points={
        'vaex.plugin': ['astro = vaex.astro.legacy:add_plugin'],
        'vaex.dataframe.accessor': ['astro = vaex.astro.transformations:DataFrameAccessorAstro'],
        'vaex.dataset.opener': [
                'fits = vaex.astro.fits:FitsBinTable',
                'gadget = vaex.astro.gadget:MemoryMappedGadget',
                'votable = vaex.astro.votable:VOTable',
                'tap = vaex.astro.tap:DatasetTap',
        ],
    },
)
