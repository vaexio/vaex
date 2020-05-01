import pkg_resources


packages = ['vaex', 'vaex-core', 'vaex-viz', 'vaex-hdf5', 'vaex-server', 'vaex-astro',
            'vaex-ui', 'vaex-jupyter', 'vaex-ml', 'vaex-distributed', 'vaex-arrow', 'vaex-graphql']


def get_versions():
    def is_installed(p):
        try:
            pkg_resources.get_distribution(p)
            return True
        except pkg_resources.DistributionNotFound:
            return False
    return {p: pkg_resources.get_distribution(p).version for p in packages if is_installed(p)}
