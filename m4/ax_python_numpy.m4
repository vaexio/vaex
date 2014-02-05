# ===========================================================================
#
# ===========================================================================
#
# SYNOPSIS
#
#   AX_PYTHON_NUMPY([ACTION_IF_FOUND], [ACTION_IF_NOT_FOUND])
#
# COPYING
#
#   Copyright (c) 2009 David Grundberg
#
#   This program is free software; you can redistribute it and/or modify it
#   under the terms of the GNU General Public License as published by the
#   Free Software Foundation; either version 2 of the License, or (at your
#   option) any later version.
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
#   Public License for more details.
#
#   You should have received a copy of the GNU General Public License along
#   with this program. If not, see <http://www.gnu.org/licenses/>.
#
#   As a special exception, the respective Autoconf Macro's copyright owner
#   gives unlimited permission to copy, distribute and modify the configure
#   scripts that are the output of Autoconf when processing the Macro. You
#   need not follow the terms of the GNU General Public License when using
#   or distributing such scripts, even though portions of the text of the
#   Macro appear in them. The GNU General Public License (GPL) does govern
#   all other use of the material that constitutes the Autoconf Macro.
#
#   This special exception to the GPL applies to versions of the Autoconf
#   Macro released by the Autoconf Macro Archive. When you make and
#   distribute a modified version of the Autoconf Macro, you may extend this
#   special exception to the GPL to apply to your modified version as well.

AC_DEFUN([AX_PYTHON_NUMPY],[
	NUMPY_INCLUDEDIR=

	AC_MSG_CHECKING([for python])
	AS_IF([test -z "$PYTHON"], [
		AC_MSG_RESULT([unknown])
	],[
		AC_MSG_RESULT([$PYTHON])
	])

	AC_MSG_CHECKING([for numpy includedir])
	AS_IF([test -z "$PYTHON"], [
		AC_MSG_RESULT([no (python unknown)])
	],[
		NUMPY_INCLUDEDIR=`$PYTHON -c '
try:
	from numpy import get_include
	print get_include()
except:
	pass
'`
		AC_MSG_RESULT([$NUMPY_INCLUDEDIR])
	])

	AS_IF([test -n "$NUMPY_INCLUDEDIR"], [

		AC_CACHE_CHECK([whether linking to numpy library works], [ax_python_numpy_cv_check],
		[
			ax_python_numpy_cv_check=no

			AC_LANG_PUSH([C++])

			ax_python_numpy_cppflags="$CPPFLAGS"
			ax_python_numpy_ldflags="$LDFLAGS"
			CPPFLAGS="$CPPFLAGS -I$NUMPY_INCLUDEDIR $PYTHON_CPPFLAGS"
			LDFLAGS="$LDFLAGS $PYTHON_LDFLAGS"
			
			AC_LANG_ASSERT(C++)
			AC_LINK_IFELSE(
			AC_LANG_PROGRAM(
				[[
#define PY_ARRAY_UNIQUE_SYMBOL my_array_symbol
#include <Python.h>
#include <numpy/oldnumeric.h>
#include <numpy/old_defines.h>
]],
				[[ &PyArray_FromDims; ]]),
				[ax_python_numpy_cv_check=yes],
				[ax_python_numpy_cv_check=no])
			CPPFLAGS="$ax_python_numpy_cppflags"
			LDFLAGS="$ax_python_numpy_ldflags"
			
			AC_LANG_POP([C++])
		])
	])

	AS_IF([test "x$ax_python_numpy_cv_check" != "xyes"], [
		NUMPY_INCLUDEDIR=

		AC_MSG_WARN([[
========================================================================
Can not link with Numpy.

Make sure the Numpy development package is installed.
========================================================================]])
	])

	AC_SUBST([NUMPY_INCLUDEDIR])

	# Execute ACTION_IF_FOUND or ACTION_IF_NOT_FOUND
	if test "x$ax_python_numpy_cv_check" == "xyes" ; then
		m4_ifvaln([$1],[$1],[:])dnl
		m4_ifvaln([$2],[else $2])dnl
	fi

])
