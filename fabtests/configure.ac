dnl
dnl Copyright (c) 2016-2017 Cisco Systems, Inc.  All rights reserved.
dnl Copyright (c) 2018-2019 Intel Corporation, Inc.  All rights reserved.
dnl
dnl Process this file with autoconf to produce a configure script.

AC_PREREQ(2.57)
AC_INIT([fabtests], [1.9.0a1], [ofiwg@lists.openfabrics.org])
AC_CONFIG_AUX_DIR(config)
AC_CONFIG_MACRO_DIR(config)
AC_CONFIG_HEADERS(config.h)
AM_INIT_AUTOMAKE([1.11 dist-bzip2 foreign -Wall -Werror subdir-objects])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

AC_CANONICAL_HOST

macos=0
linux=0
freebsd=0

case $host_os in
*darwin*)
	macos=1
	;;
*linux*)
	linux=1
	;;
*freebsd*)
	freebsd=1
	;;
*)
	AC_MSG_ERROR([libfabric only builds on Linux & OS X])
	;;
esac

AM_CONDITIONAL([MACOS], [test $macos -eq 1])
AM_CONDITIONAL([LINUX], [test $linux -eq 1])
AM_CONDITIONAL([FREEBSD], [test $freebsd -eq 1])

base_c_warn_flags="-Wall -Wundef -Wpointer-arith"
debug_c_warn_flags="-Wextra -Wno-unused-parameter -Wno-sign-compare -Wno-missing-field-initializers"
debug_c_other_flags="-fstack-protector-strong"

AC_ARG_ENABLE([debug],
	[AS_HELP_STRING([--enable-debug],
			[Enable debugging - default NO])],
	[CFLAGS="-g -O0 ${base_c_warn_flags} ${debug_c_warn_flags} ${debug_c_other_flags} $CFLAGS"
	 dbg=1],
	[dbg=0])

AC_DEFINE_UNQUOTED([ENABLE_DEBUG], [$dbg],
	[defined to 1 if configured with --enable-debug])

dnl Fix autoconf's habit of adding -g -O2 by default
AS_IF([test -z "$CFLAGS"],
      [CFLAGS="-O2 -DNDEBUG ${base_c_warn_flags}"])

# AM PROG_AR did not exist pre AM 1.11.x (where x is somewhere >0 and
# <3), but it is necessary in AM 1.12.x.
m4_ifdef([AM_PROG_AR], [AM_PROG_AR])

AM_PROG_LIBTOOL

AC_ARG_WITH([valgrind],
    AC_HELP_STRING([--with-valgrind],
		   [Enable valgrind annotations - default NO]))

if test "$with_valgrind" != "" && test "$with_valgrind" != "no"; then
	AC_DEFINE([INCLUDE_VALGRIND], 1,
		  [Define to 1 to enable valgrind annotations])
	if test -d $with_valgrind; then
		CPPFLAGS="$CPPLFAGS -I$with_valgrind/include"
	fi
fi

dnl Checks for programs
AC_PROG_CC
AM_PROG_CC_C_O

LT_INIT

have_clock_gettime=0

AC_SEARCH_LIBS([clock_gettime],[rt],
	       [have_clock_gettime=1],
	       [])

AC_DEFINE_UNQUOTED(HAVE_CLOCK_GETTIME, [$have_clock_gettime],
		   [Define to 1 if clock_gettime is available.])
AM_CONDITIONAL(HAVE_CLOCK_GETTIME, [test $have_clock_gettime -eq 1])

AC_ARG_WITH([libfabric],
            AC_HELP_STRING([--with-libfabric], [Use non-default libfabric location - default NO]),
            [AS_IF([test -d $withval/lib64], [fab_libdir="lib64"], [fab_libdir="lib"])
             CPPFLAGS="-I $withval/include $CPPFLAGS"
             LDFLAGS="-L$withval/$fab_libdir $LDFLAGS"],
            [])

dnl Checks for libraries
AC_CHECK_LIB([fabric], fi_getinfo, [],
    AC_MSG_ERROR([fi_getinfo() not found.  fabtests requires libfabric.]))

dnl Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADER([rdma/fabric.h], [],
    [AC_MSG_ERROR([<rdma/fabric.h> not found.  fabtests requires libfabric.])])

AC_MSG_CHECKING([for fi_trywait support])
AC_LINK_IFELSE([AC_LANG_PROGRAM([[#include <rdma/fi_eq.h>]],
	       [[fi_trywait(NULL, NULL, 0);]])],
	       [AC_MSG_RESULT([yes])],
	       [AC_MSG_RESULT([no])
	        AC_MSG_ERROR([fabtests requires fi_trywait support. Cannot continue])])

if test "$with_valgrind" != "" && test "$with_valgrind" != "no"; then
AC_CHECK_HEADER(valgrind/memcheck.h, [],
    AC_MSG_ERROR([valgrind requested but <valgrind/memcheck.h> not found.]))
fi

AC_CHECK_FUNC([epoll_create1], [have_epoll=1], [have_epoll=0])
AC_DEFINE_UNQUOTED([HAVE_EPOLL], [$have_epoll],
		   [Defined to 1 if Linux epoll is available])

AC_CONFIG_FILES([Makefile fabtests.spec])
AC_OUTPUT
