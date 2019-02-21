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

packages = ['vaex-core', 'vaex-viz', 'vaex-hdf5', 'vaex-server', 'vaex-astro', 'vaex-ui', 'vaex-jupyter', 'vaex-distributed']
import os
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'


class DevelopCmd(develop):
    def run(self):
        for package in packages:
            with cwd(os.path.join('packages', package)):
                os.system('python -m pip install -e .')
            # we need to make symbolic links from vaex-core/vaex/<name> to vaex-<name>/vaex/<name
            # otherwise development install do not work
            if package != 'vaex-core':
                name = package.split('-')[1]
                source = os.path.abspath(os.path.join('packages', package, 'vaex', name))
                target = os.path.abspath(os.path.join('packages', packages[0], 'vaex', name))
                if os.path.exists(target):
                    os.remove(target)
                os.symlink(source, target)

class InstallCmd(install):
    """ Add custom steps for the install command """
    def run(self):
        for package in packages:
            with cwd(os.path.join('packages', package)):
                os.system('python -m pip install --no-deps .')
        for package in packages:
            with cwd(os.path.join('packages', package)):
                os.system('python -m pip install --upgrade .')
        if on_rtd:
            os.system('python -m pip install vaex-ml==0.3.2')

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
    cmdclass={
        'install': InstallCmd,
        'develop': DevelopCmd,
    },
)
