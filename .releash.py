from releash import *
# these objects only tag when they are exe
gitpush = ReleaseTargetGitPush()

# core package
core = add_package("packages/vaex-core", "vaex-core")
version_core = VersionSource(core, '{path}/vaex/core/_version.py')
gittag_core = ReleaseTargetGitTagVersion(version_source=version_core, prefix='core-v')

core.version_source = version_core
core.version_targets.append(VersionTarget(core, '{path}/vaex/core/_version.py'))

core.release_targets.append(gittag_core)
core.release_targets.append(ReleaseTargetSourceDist(core))
#core.release_targets.append(gitpush)
core.release_targets.append(ReleaseTargetCondaForge(core, '../feedstocks/vaex-core-feedstock'))

packages = ['vaex-core', 'vaex-meta', 'vaex-viz', 'vaex-hdf5', 'vaex-server', 'vaex-astro', 'vaex-ui', 'vaex-jupyter', 'vaex-distributed', 'vaex-arrow', 'vaex-ml', 'vaex-graphql']
names = [k[5:] for k in packages[1:]]

for name in names:
    if name == 'meta':
        package = add_package("packages/vaex-" + name, "vaex-" +name, 'vaex.' + name, distribution_name='vaex')
        version = VersionSource(package, '{path}/vaex/' +name +'/_version.py')
    elif name == 'arrow':
        package = add_package("packages/vaex-" + name, "vaex-" +name, 'vaex_' + name)
        version = VersionSource(package, '{path}/vaex_' +name +'/_version.py')
    else:
        package = add_package("packages/vaex-" + name, "vaex-" +name, 'vaex.' + name)
        version = VersionSource(package, '{path}/vaex/' +name +'/_version.py')
    gittag = ReleaseTargetGitTagVersion(version_source=version, prefix=name + '-v', msg='Release {version} of vaex-' +name)
    package.version_source = version
    if name == 'arrow':
        package.version_targets.append(VersionTarget(package, '{path}/vaex_' + name + '/_version.py'))
    else:
        package.version_targets.append(VersionTarget(package, '{path}/vaex/' + name + '/_version.py'))
    # it is ok to add this twice, it will only tag once
    package.release_targets.append(gittag)
    package.release_targets.append(ReleaseTargetSourceDist(package))
    # also ok to add twice, it will only execute for the last package
    package.release_targets.append(gitpush)
    #if name in ['hdf5', 'viz']:
    if name == 'meta':
        package.release_targets.append(ReleaseTargetCondaForge(package, '../feedstocks/vaex' + '-feedstock'))
    else:
        package.release_targets.append(ReleaseTargetCondaForge(package, '../feedstocks/vaex-' + name + '-feedstock'))

