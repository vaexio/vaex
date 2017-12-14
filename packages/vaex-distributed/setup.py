from setuptools import setup
import sys, os, imp

import sys, os, imp
from setuptools import Extension

dirname = os.path.dirname(__file__)
path_version = os.path.join(dirname, 'vaex/distributed/_version.py')
version = imp.load_source('version', path_version)


name        = 'vaex'
author      = 'Maarten A. Breddels'
author_email= 'maartenbreddels@gmail.com'
license     = 'MIT'
version     = version.__version__
url         = 'https://www.github.com/maartenbreddels/vaex'
install_requires_distributed = ['vaex-core', 'vaex-server']

setup(name=name+'-distributed',
    version=version,
    description='Distributed dataset for vaex',
    url=url,
    author=author,
    author_email=author_email,
    install_requires=install_requires_distributed,
    license=license,
    packages=['vaex.distributed'],
    zip_safe=False
    )
