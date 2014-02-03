AC_INIT([subspacefinding], [0.1])
AC_CONFIG_FILES([Makefile src/Makefile])
AC_CONFIG_SRCDIR([src/kdForSpaces_usingV2scott.c])
AC_CONFIG_HEADERS([config.h])

AC_PROG_CXX
AC_PROG_CC
#AC_PROG_INSTALL

AC_LANG([C++])

AC_PATH_X
AC_CHECK_HEADERS([malloc.h stddef.h stdlib.h string.h sys/time.h unistd.h])

AC_SEARCH_LIBS([pthread_created], [pthread])


AC_OUTPUT
echo \
"===================================
${PACKAGE_NAME} Version ${PACKAGE_VERSION}
Prefix: '${prefix}'
C   Compiler: '${CC} ${CFLAGS} ${CPPFLAGS}'
C++ Compiler: '${CXX} ${CFLAGS} ${CPPFLAGS}'

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
