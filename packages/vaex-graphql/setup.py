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
# for now we leave graphene-tornado / graphene on v2
# ref https://github.com/vaexio/vaex/pull/2356#issuecomment-2320707228
# in top level vaex[ci] we added the following (additional) direct reference requirement which adds one commit (https://github.com/ddelange/graphene-tornado/commit/d75f01f) on top of the 2.6.1 release to unpin werkzeug
# "graphene-tornado @ https://github.com/ddelange/graphene-tornado/archive/refs/heads/2.6.1.unpin-werkzeug.zip",
install_requires_graphql = ['vaex-core~=4.0', 'graphene-tornado~=2.5']

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
