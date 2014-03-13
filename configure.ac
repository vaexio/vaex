AC_INIT([subspacefinding], [0.1])
AM_INIT_AUTOMAKE
AC_CONFIG_MACRO_DIR([m4])
#AC_CONFIG_MACRO_DIRS([m4])


AM_PATH_PYTHON
LT_INIT

AC_CONFIG_FILES([Makefile src/Makefile])
AC_CONFIG_SRCDIR([src/kdForSpaces_usingV2scott.c])
#AC_CONFIG_HEADERS(al[config.h])


#AC_PROG_RANLIB
AC_PROG_CXX
AC_PROG_CC
AM_PROG_CC_C_O
#AC_PROG_INSTALL

#AC_LANG([C++])
AX_BOOST_PYTHON
AX_PYTHON_NUMPY([],[])
echo ${ac_cv_boost_python}

if test "x${ac_cv_boost_python}" = "xno";
then
	echo "====================
Python boost missing, consider adding include dirs, for instance on mac using:
./configure CPPFLAGS=-I/opt/local/include LDFLAGS=-L/opt/local/lib
===================="
	(exit 1); exit 1;
fi

if test "x${BOOST_PYTHON_LIB}" = "x";
then
	echo "====================
Python boost library missing, consider adding include dirs, for instance on mac using:
./configure CPPFLAGS=-I/opt/local/include LDFLAGS=-L/opt/local/lib
===================="
	(exit 1); exit 1;
fi


AX_PYTHON_DEVEL

#AC_PATH_X
#AC_CHECK_HEADERS([malloc.h stddef.h stdlib.h string.h sys/time.h unistd.h])

#AC_SEARCH_LIBS([pthread_created], [pthread])





AC_OUTPUT
echo \
"===================================

${PACKAGE_NAME} Version ${PACKAGE_VERSION}

Prefix: '${prefix}'
C   Compiler: '${CC} ${CFLAGS} ${CPPFLAGS}'
C++ Compiler: '${CXX} ${CFLAGS} ${CPPFLAGS}'

Python:
	executable: ${PYTHON} (version ${PYTHON_VERSION})
	prefix: ${PYTHON_PREFIX}
	pythondir: ${pythondir}
	pkgpythondir: ${pkgpythondir}
	pyexecdir: ${pyexecdir}
	pkgpyexecdir: ${pkgpyexecdir}

Package features:
	no features implemented yet


--------------------
To build/install as a sysadmin:
make all; sudo make install

--------------------
As a user, if you want to install in a specific directory (or use stow):
(You might want to run configure as well with the same --prefix
make all PREFIX=/home/myusername/local;
make install

==================================="
