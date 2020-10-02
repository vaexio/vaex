import os
import imp
from setuptools import setup

dirname = os.path.dirname(__file__)
path_version = os.path.join(dirname, 'vaex/graphql/_version.py')
version = imp.load_source('version', path_version)

name        = 'vaex'
author      = 'Maarten A. Breddels'
author_email= 'maartenbreddels@gmail.com'
license     = 'MIT'
version     = version.__version__
url         = 'https://www.github.com/vaexio/vaex'
install_requires_graphql = ['vaex-core>=4.0.0-alpha.3,<5', 'graphene>=2.1.8,<3', 'graphene>=2.1.8,<3', 'graphene-tornado>=2.5.1,<3']

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
