bin_PROGRAMS =  kerneldensity maxtreeForSpaces maxtree3d
#kdForSpaces_usingV2scott
#kdForSpaces_usingV2scott_SOURCES = kdForSpaces_usingV2scott.c
#kdForSpaces_usingV2scott_LDADD = libsubfind.a -lstdc++


maxtree3d_SOURCES = MTfor3D_forC.cpp MaxTree3d.cpp ShortVolume.cpp SimpleShortVolume.cpp avs_io.c

maxtreeForSpaces_SOURCES = maxtreeForSpaces.c pgm.c
kerneldensity_SOURCES = kerneldensity_program.c kerneldensity.c avs_io.c pgm.c
kerneldensity_LDADD = -lm

kerneldensity_CFLAGS = -std=c99 

maxtreeForSpaces_LDADD = -lm 
#libsubfind.a

#kdForSpaces_usingV2scott_LINK = $(CXX)
#noinst_HEADERS = ShortVolume.h SimpleShortVolume.h definitions.h avs_io.h MaxTree3d.h MTAuxData.h Queue.h kerneldensity.h
#lib_LIBRARIES = libsubfind.a
#libsubfind_a_SOURCES = ShortVolume.cpp SimpleShortVolume.cpp MTfor3D_forC.cpp avs_io.c MaxTree3d.cpp pixel_qsortS.c combi2.c maxtreeForSpaces.c MTfor3D.cpp pca.c


pyexec_LTLIBRARIES = subspacefind.la
subspacefind_la_SOURCES = python_binding.cpp kerneldensity.c avs_io.c maxtreeForSpaces.c pgm.c MaxTree3d.cpp ShortVolume.cpp SimpleShortVolume.cpp 
subspacefind_la_CFLAGS = -std=c99  -fopenmp -g -O0 -Wall
subspacefind_la_CXXFLAGS = -fopenmp -g -O0 -Wall
subspacefind_la_CPPFLAGS = $(PYTHON_CPPFLAGS) ${CPPFLAGS} -I${NUMPY_INCLUDEDIR}
subspacefind_la_LDFLAGS = -lstdc++ -avoid-version -module  $(PYTHON_LDFLAGS) -l${BOOST_PYTHON_LIB} -lgomp
