packages = [
    "vaex",
    "vaex-core",
    "vaex-viz",
    "vaex-hdf5",
    "vaex-server",
    "vaex-astro",
    "vaex-ui",
    "vaex-jupyter",
    "vaex-ml",
    "vaex-arrow",
    "vaex-graphql",
    "vaex-enterprise",
]


# Attempt to use importlib.metadata first because it's much faster
# though it's only available in Python 3.8+ so we'll need to fall
# back to pkg_resources for Python 3.7 support
# See https://github.com/jacob-indigo/geoalchemy2/blob/c7351b2086d5d711cbeb0c92f27002829e72bab3/geoalchemy2/__init__.py#L446
def get_version(package):
    try:
        import importlib.metadata
    except ImportError:
        from pkg_resources import DistributionNotFound
        from pkg_resources import get_distribution

        try:
            version = get_distribution(package).version
        except DistributionNotFound:  # pragma: no cover
            pass
    else:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover
            pass
    return version


def get_versions():
    installed_packages = {}
    for p in packages:
        try:
            installed_packages[p] = get_version(p)
        except Exception:
            pass
    return installed_packages
