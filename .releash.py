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


for name in ['hdf5', 'viz', 'astro', 'server', 'ui']:
    # hdf5 package
    package = add_package("packages/vaex-" + name, "vaex-" +name, 'vaex.' + name)
    version = VersionSource(package, '{path}/vaex/' +name +'/_version.py')
    gittag = ReleaseTargetGitTagVersion(version_source=version, prefix=name + '-v')
    package.version_source = version
    package.version_targets.append(VersionTarget(package, '{path}/vaex/' + name + '/_version.py'))
    # it is ok to add this twice, it will only tag once
    package.release_targets.append(gittag)
    package.release_targets.append(ReleaseTargetSourceDist(package))
    # also ok to add twice, it will only execute for the last package
    package.release_targets.append(gitpush)
    #package.release_targets.append(ReleaseTargetCondaForge(core, '../feedstocks/vaex-' + name + '-feedstock'))

