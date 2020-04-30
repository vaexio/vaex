__version_tuple__ = (2, 0, 0, 'dev')
__version__ = '2.0.0-dev'
from vaex.version import patch_version
__version_tuple__, __version__ = patch_version(__version_tuple__)
