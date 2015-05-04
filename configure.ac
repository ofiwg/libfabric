dnl Process this file with autoconf to produce a configure script.

AC_PREREQ(2.57)
AC_INIT([fabtests], [1.0.0], [linux-rdma@vger.kernel.org])
AC_CONFIG_AUX_DIR(config)
AC_CONFIG_MACRO_DIR(config)
AC_CONFIG_HEADERS(config.h)
AM_INIT_AUTOMAKE([1.11 dist-bzip2 foreign -Wall -Werror subdir-objects])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

AC_CANONICAL_HOST

macos=0
linux=0

case $host_os in
*darwin*)
	macos=1
	;;
*linux*)
	linux=1
	;;
*)
	AC_MSG_ERROR([libfabric only builds on Linux & OS X])
	;;
esac

AM_CONDITIONAL([MACOS], [test $macos -eq 1])
AM_CONDITIONAL([LINUX], [test $linux -eq 1])

dnl Fix autoconf's habit of adding -g -O2 by default
AS_IF([test -z "$CFLAGS"],
      [CFLAGS='-O2 -DNDEBUG -Wall'])

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

if test "$with_valgrind" != "" && test "$with_valgrind" != "no"; then
AC_CHECK_HEADER(valgrind/memcheck.h, [],
    AC_MSG_ERROR([valgrind requested but <valgrind/memcheck.h> not found.]))
fi

AC_CHECK_FUNC([epoll_create1], [HAVE_EPOLL=1], [HAVE_EPOLL=0])
AM_CONDITIONAL(HAVE_EPOLL, [test $HAVE_EPOLL -eq 1])

AC_CONFIG_FILES([Makefile fabtests.spec])
AC_OUTPUT
