import os
import pkg_resources


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


packages = ['vaex', 'vaex-core', 'vaex-viz', 'vaex-hdf5', 'vaex-server', 'vaex-astro', 'vaex-ui', 'vaex-jupyter', 'vaex-ml', 'vaex-distributed', 'vaex-arrow', 'vaex-graphql']


def get_versions():
    def is_installed(p):
        try:
            pkg_resources.get_distribution(p)
            return True
        except pkg_resources.DistributionNotFound:
            return False
    return {p: pkg_resources.get_distribution(p).version for p in packages if is_installed(p)}
