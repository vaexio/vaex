from setuptools import setup
import sys, os, imp

import sys, os, imp
from setuptools import Extension

dirname = os.path.dirname(__file__)
path_version = os.path.join(dirname, "vaex/server/_version.py")
version = imp.load_source('version', path_version)


name        = 'vaex'
author      = "Maarten A. Breddels"
author_email= "maartenbreddels@gmail.com"
license     = 'MIT'
version     = version.__version__
url         = 'https://www.github.com/maartenbreddels/vaex'
install_requires_server = ["tornado>4.1", "cachetools"]

setup(name=name+'-server',
    version=version,
    description='Webserver and client for vaex for a remove dataset',
    url=url,
    author=author,
    author_email=author_email,
    install_requires=install_requires_server,
    license=license,
    packages=['vaex.server'],
    zip_safe=False,)