from releash import *
# these objects only tag when they are exe
gitpush = ReleaseTargetGitPush()

# core package
core = add_package("packages/vaex-core", "vaex-core")
version_core = VersionSource(core, '{path}/vaex/core/_version.py')
gittag_core = ReleaseTargetGitTagVersion(version_source=version_core, prefix='core-v')

core.version_source = version_core
core.version_targets.append(VersionTarget(core, '{path}/vaex/core/_version.py'))
core.version_targets.append(VersionTargetReplace(core, [
    'packages/vaex-meta/setup.py',
]))


def add_version_replace(package):
    # for pre-releases we always bump all requirements that are exact matches
    if not package.version_source.semver['prerelease']:
        return
    if any(k in package.version_source.semver['prerelease'] for k in "dev alpha beta rc"):
        package.version_targets.append(VersionTargetReplace(package, [
            'packages/vaex-meta/setup.py',
            'packages/vaex-arrow/setup.py',
            'packages/vaex-graphql/setup.py',
            'packages/vaex-hdf5/setup.py',
            'packages/vaex-jupyter/setup.py',
            'packages/vaex-ml/setup.py',
            'packages/vaex-server/setup.py',
            'packages/vaex-viz/setup.py',
        ], pattern='{name}(?P<cmp>[^0-9]*)' + str(package.version_source), ))


add_version_replace(core)

core.tag_targets.append(gittag_core)
core.release_targets.append(ReleaseTargetSourceDist(core))
#core.release_targets.append(gitpush)
core.release_targets.append(ReleaseTargetCondaForge(core, '../feedstocks/vaex-core-feedstock'))

packages = ['vaex-core', 'vaex-meta', 'vaex-viz', 'vaex-hdf5', 'vaex-server', 'vaex-astro', 'vaex-ui', 'vaex-jupyter', 'vaex-ml', 'vaex-graphql']
names = [k[5:] for k in packages[1:]]

for name in names:
    if name == 'meta':
        package = add_package("packages/vaex-" + name, "vaex-" +name, 'vaex.' + name, distribution_name='vaex')
        version = VersionSource(package, '{path}/vaex/' +name +'/_version.py')
    else:
        package = add_package("packages/vaex-" + name, "vaex-" +name, 'vaex.' + name)
        version = VersionSource(package, '{path}/vaex/' +name +'/_version.py')
    gittag = ReleaseTargetGitTagVersion(version_source=version, prefix=name + '-v', msg='Release {version} of vaex-' +name)
    package.version_source = version
    package.version_targets.append(VersionTarget(package, '{path}/vaex/' + name + '/_version.py'))
    add_version_replace(package)
    # it is ok to add this twice, it will only tag once
    package.tag_targets.append(gittag)
    package.release_targets.append(ReleaseTargetSourceDist(package))
    # also ok to add twice, it will only execute for the last package
    package.release_targets.append(gitpush)
    #if name in ['hdf5', 'viz']:
    if name == 'meta':
        package.release_targets.append(ReleaseTargetCondaForge(package, '../feedstocks/vaex' + '-feedstock'))
    else:
        package.release_targets.append(ReleaseTargetCondaForge(package, '../feedstocks/vaex-' + name + '-feedstock'))

