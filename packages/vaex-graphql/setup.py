import os
from importlib.machinery import SourceFileLoader
from setuptools import setup

dirname = os.path.dirname(__file__)
path_version = os.path.join(dirname, 'vaex/graphql/_version.py')
version = SourceFileLoader('version', path_version).load_module()

name        = 'vaex'
author      = 'Maarten A. Breddels'
author_email= 'maartenbreddels@gmail.com'
license     = 'MIT'
version     = version.__version__
url         = 'https://www.github.com/vaexio/vaex'
# graphene-tornado<3 pins werkzeug 0.12, which creates a conflict with vaex-ml (tensorflow 2.16+ required on python 3.12)
# upgrading to graphene-tornado==3.0.0b2 caused CI to fail, so instead we pin to graphene-tornado==2.6.1 with unpinned werkzeug
install_requires_graphql = ['vaex-core~=4.0', 'graphene-tornado @ https://github.com/ddelange/graphene-tornado/archive/refs/heads/2.6.1.unpin-werkzeug.zip']

setup(
    name=name + '-graphql',
    version=version,
    description='GraphQL support for accessing vaex DataFrame',
    url=url,
    author=author,
    author_email=author_email,
    install_requires=install_requires_graphql,
    license=license,
    packages=['vaex.graphql'],
    zip_safe=False,
    entry_points={
        'vaex.dataframe.accessor': ['graphql = vaex.graphql:DataFrameAccessorGraphQL'],
    },
)
