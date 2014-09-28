# -*- coding: utf-8 -*-
try:
	import version
except:
	import sys
	print >>sys.stderr, "version file not found, please run git/hooks/post-commit or git/hooks/post-checkout and/or install them as hooks (see git/README)"
	raise

__release_name__ = "alpha"
__version_tuple__ = version.versiontuple
__program_name__ = "vaex"
__version__ = "%d.%d.%d" % __version_tuple__
__release__ = version.versiontring
__full_name__ = __program_name__ + "-" + __release__
