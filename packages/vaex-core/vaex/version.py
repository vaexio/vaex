import platform
from ._version import __version_tuple__, __version__
versiontuple = __version_tuple__
versionstring = __version__
# if pre_release:
# ersionstring += "-" + pre_release


# from vaex.utils import osname, setup.py doesn't want imports...
osname = dict(darwin="osx", linux="linux", windows="windows")[platform.system().lower()]

if __name__ == "__main__":
    import vaex
    import sys
    # print vaex.__version_tuple__
    if sys.argv[1] == "version":
        print("version:", vaex.__version__)
    elif sys.argv[1] == "fullname":
        print("full name:", vaex.__full_name__)
    elif sys.argv[1] == "buildname":
        print("build name:", vaex.__build_name__)
    elif sys.argv[1] == "tagcmd":
        print("git tag %s" % versionstring)
        print("git push --tags")
    else:
        print("use version, fullname or buildname as argument")
