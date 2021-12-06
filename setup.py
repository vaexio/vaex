from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
import pip
import os
import sys
import contextlib


@contextlib.contextmanager
def cwd(path):
    curdir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(curdir)

# inspired by https://blog.shazam.com/python-microlibs-5be9461ad979


packages = ['vaex-core', 'vaex-viz', 'vaex-hdf5', 'vaex-server', 'vaex-astro', 'vaex-jupyter', 'vaex-ml', 'vaex-meta', 'vaex-graphql', 'vaex-contrib']
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'


class DevelopCmd(develop):
    def run(self):
        relative = os.path.abspath(os.path.join('packages', 'vaex-core', 'vaex'))
        for package in packages:
            with cwd(os.path.join('packages', package)):
                err = os.system('python -m pip install -e .')
                if err:
                    raise RuntimeError(f'Oops, failed to install {package}')
            # we need to make symbolic links from vaex-core/vaex/<name> to vaex-<name>/vaex/<name>
            # otherwise development install do not work
            if package not in ['vaex-core']:
                name = package.split('-')[1]

                source = os.path.abspath(os.path.join('packages', package, 'vaex', name))
                rel_source = os.path.relpath(source, relative)
                with cwd(relative):
                    print('symlinking', source, name, rel_source)
                    if os.path.exists(name) and os.readlink(name) == rel_source:
                        print('symlink ok')
                    else:
                        if os.path.exists(name):
                            print('old symlink',  os.readlink(name))
                            os.remove(name)
                        os.symlink(rel_source, name)


class InstallCmd(install):
    """ Add custom steps for the install command """
    def run(self):
        for package in packages:
            with cwd(os.path.join('packages', package)):
                os.system('python -m pip install --no-deps .')
        for package in packages:
            with cwd(os.path.join('packages', package)):
                os.system('python -m pip install --upgrade .')


setup(
    name='vaex-meta',
    version="0.1.0",
    description="Convenience setup.py for when installing from the git repo",
    classifiers=[
        'Private :: Do Not Upload to pypi server',
    ],
    install_requires=[
        'pip',
    ],
    extras_require={
        'dev': [
            'pytest',
            'gcsfs',
            's3fs',
            'graphviz',
            'myst_parser',
            # For generating the documentation
            'sphinx',
            'nbsphinx',
            'sphinx_gallery',
            'sphinx_sitemap',
            'sphinx_book_theme',
        ]
    },
    cmdclass={
        'install': InstallCmd,
        'develop': DevelopCmd,
    },
)
