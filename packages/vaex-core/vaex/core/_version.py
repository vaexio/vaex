import os

__version_tuple__ = (2, 0, 0, 'dev')
__version__ = '2.0.0-dev'

nightly = bool(os.environ.get('VAEX_RELEASE_NIGHTLY', ''))

def patch_version(version_tuple):
    if nightly:
        import datetime
        dt = datetime.datetime.now()
        date = (dt.year * 100 + dt.month)*100 + dt.day
        version_tuple = (*version_tuple[:3], 'dev', date)
    version_string = '.'.join(map(str, version_tuple[:4]))
    if len(version_tuple) > 4:
        version_string += str(version_tuple[-1])
    return version_tuple, version_string
__version_tuple__, __version__ = patch_version(__version_tuple__)
