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
                    return get_distribution(p).version
                except DistributionNotFound:  # pragma: no cover
                    raise Exception(f"Package not found: {p}")
        else:
            try:
                return importlib.metadata.version(p)

            except importlib.metadata.PackageNotFoundError:  # pragma: no cover
                raise Exception(f"Package not found: {p}")

    package_versions = {}

    for p in packages:
        try:
            version = is_installed(p)
            package_versions[p] = version
        except Exception as e:
            # check exception text contains 'Package not found: '
            if 'Package not found: ' not in str(e):
                raise e

    return package_versions
