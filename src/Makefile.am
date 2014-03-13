bin_PROGRAMS =  kerneldensity maxtreeForSpaces
#kdForSpaces_usingV2scott
#kdForSpaces_usingV2scott_SOURCES = kdForSpaces_usingV2scott.c
#kdForSpaces_usingV2scott_LDADD = libsubfind.a -lstdc++

maxtreeForSpaces_SOURCES = maxtreeForSpaces.c
#kerneldensity_SOURCSE = avs_io.
kerneldensity_LDADD = -lm libsubfind.a

kerneldensity_CFLAGS = -std=c99

maxtreeForSpaces_LDADD = -lm 
#libsubfind.a

#kdForSpaces_usingV2scott_LINK = $(CXX)
noinst_HEADERS = ShortVolume.h SimpleShortVolume.h definitions.h avs_io.h MaxTree3d.h MTAuxData.h Queue.h kerneldensity.h
lib_LIBRARIES = libsubfind.a
libsubfind_a_SOURCES = ShortVolume.cpp SimpleShortVolume.cpp MTfor3D_forC.cpp avs_io.c MaxTree3d.cpp pixel_qsortS.c combi2.c maxtreeForSpaces.c MTfor3D.cpp pca.c


pyexec_LTLIBRARIES = subspacefind.la
subspacefind_la_SOURCES = python_binding.cpp kerneldensity.c avs_io.c 
subspacefind_la_CFLAGS = -std=c99
subspacefind_la_CPPFLAGS = $(PYTHON_CPPFLAGS) ${CPPFLAGS} -I${NUMPY_INCLUDEDIR}
subspacefind_la_LDFLAGS = -avoid-version -module  $(PYTHON_LDFLAGS) -l${BOOST_PYTHON_LIB}
#libsubfind.a
