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

# hdf5 package
hdf5 = add_package("packages/vaex-hdf5", "vaex-hdf5")
# we have locked versions, but separate tags
gittag_hdf5 = ReleaseTargetGitTagVersion(version_source=version_core, prefix='hdf5-v')
hdf5.version_source = version_core
hdf5.version_targets.append(VersionTarget(hdf5, '{path}/vaex/hdf5/_version.py'))
# it is ok to add this twice, it will only tag once
hdf5.release_targets.append(gittag_hdf5)
hdf5.release_targets.append(ReleaseTargetSourceDist(hdf5))
# also ok to add twice, it will only execute for the last package
hdf5.release_targets.append(gitpush)

#core.release_targets.append(ReleaseTargetCondaForge(core, 'releash-fake-feedstock'))


# viz package
viz = add_package("packages/vaex-viz", "vaex-viz")
# we have locked versions, but separate tags
gittag_viz = ReleaseTargetGitTagVersion(version_source=version_core, prefix='viz-v')
viz.version_source = version_core
viz.version_targets.append(VersionTarget(viz, '{path}/vaex/viz/_version.py'))
# it is ok to add this twice, it will only tag once
viz.release_targets.append(gittag_viz)
viz.release_targets.append(ReleaseTargetSourceDist(viz))
# also ok to add twice, it will only execute for the last package
viz.release_targets.append(gitpush)

#core.release_targets.append(ReleaseTargetCondaForge(core, 'releash-fake-feedstock'))
