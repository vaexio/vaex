packages = ['vaex', 'vaex-core', 'vaex-viz', 'vaex-hdf5', 'vaex-server', 'vaex-astro',
            'vaex-ui', 'vaex-jupyter', 'vaex-ml', 'vaex-arrow', 'vaex-graphql', 'vaex-enterprise']


def get_versions():
    def is_installed(p):
        #  Concept from https://github.com/geoalchemy/geoalchemy2/pull/392/files#diff-208c493ee84f787a0a966b235a31f86a5d382ff76e1e5533b5b1da712996b7e2
        try:
            import importlib.metadata
        except ImportError:
            try:
                from pkg_resources import DistributionNotFound
                from pkg_resources import get_distribution
            except ImportError:  # pragma: no cover
                pass
            else:
                try:
                    get_distribution(p).version
                    return True
                except DistributionNotFound:  # pragma: no cover
                    return False
        else:
            try:
                importlib.metadata.version(p)
                return True
            except importlib.metadata.PackageNotFoundError:  # pragma: no cover
                return False
    return {p: pkg_resources.get_distribution(p).version for p in packages if is_installed(p)}
