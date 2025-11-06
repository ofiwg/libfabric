# SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
# Copyright 2018, 2020 Hewlett Packard Enterprise Development LP

# Guidelines for updating version are found here:
#    https://www.gnu.org/software/libtool/manual/html_node/Updating-version-info.html

# ABI Versioning - Libtool
m4_define([vers_current],[6])
m4_define([vers_revision],[0])
m4_define([vers_age],[5])

# API Versioning - Semantic
m4_define([api_major],[1])
m4_define([api_minor],[7])
m4_define([api_patch],[0])

AC_PREREQ(2.60)
AC_INIT([libcxi], [api_major.api_minor.api_patch])
AC_CONFIG_SRCDIR([src/libcxi.c])
AC_CONFIG_MACRO_DIRS([m4])
AC_REQUIRE_AUX_FILE([tap-driver.sh])
AM_INIT_AUTOMAKE([
	check-news
	foreign
	1.11
	-Wall
	-Wno-portability
	silent-rules
	tar-pax
	no-dist-gzip
	dist-xz
	subdir-objects
])
PKG_PROG_PKG_CONFIG
PKG_INSTALLDIR
AC_PROG_CC_STDC
AC_PROG_AWK
AM_SILENT_RULES([yes])
LT_INIT

PKG_CHECK_MODULES([LIBCONFIG], [libconfig >= 1.5])
PKG_CHECK_MODULES([LIBUV], [libuv >= 1.18])
PKG_CHECK_MODULES([LIBFUSE], [fuse >= 2.9.7])
PKG_CHECK_MODULES([LIBYAML], [yaml-0.1 >= 0.1.7])
PKG_CHECK_MODULES([LIBNL], [libnl-3.0])
PKG_CHECK_MODULES([LIBNUMA], [numa >= 2.0])

pkg_CFLAGS="\
-Wall \
-Wchar-subscripts \
-Wformat-security \
-Wnested-externs \
-Wpointer-arith \
-Wstrict-prototypes \
-Wtype-limits \
-Werror \
"
AC_SUBST([pkg_CFLAGS])

AC_SUBST([lt_current],[vers_current])
AC_SUBST([lt_revision],[vers_revision])
AC_SUBST([lt_age],[vers_age])

AC_CHECK_DECLS([RDMA_NLDEV_SYS_ATTR_COPY_ON_FORK],
               [], [], [[#include <rdma/rdma_netlink.h>]])

have_criterion=false
libcxi_test_CPPFLAGS=
libcxi_test_LDFLAGS=
libcxi_test_LIBS=

AC_ARG_WITH([criterion], [AS_HELP_STRING([--with-criterion],
	[Location for criterion unit testing framework.])])

AS_IF([test "$with_criterion" != ""],
	[libcxi_test_CPPFLAGS="-I$with_criterion/include"
	 libcxi_test_LDFLAGS="-L$with_criterion/lib64 -Wl,-rpath=$(realpath $with_criterion/lib64)"
	 libcxi_test_LIBS="-lcriterion"
	have_criterion=true])

AM_CONDITIONAL([HAVE_CRITERION], [test "x$have_criterion" = "xtrue"])

AC_ARG_WITH([systemd],
     [AS_HELP_STRING([--with-systemd],
     [enable support for Systemd daemon API])])

systemd_service_type="simple"
AS_IF([test "x$with_systemd" != "xno"],
     [PKG_CHECK_MODULES([LIBSYSTEMD], [libsystemd],
          [AC_DEFINE([HAVE_LIBSYSTEMD], [1], [use Systemd daemon API])]
          [systemd_service_type="notify"],
          [AS_IF([test "x$with_systemd" != "xyes"],
               AC_MSG_RESULT([Building without libsystemd]),
               AC_MSG_ERROR([Could not find libsystemd]))])])

AC_SUBST(systemd_service_type)

# Check for plantuml
AC_PATH_PROG(PLANTUML, plantuml, no)
AM_CONDITIONAL(HAVE_PLANTUML, test "x$PLANTUML" != "xno")

AC_SUBST(libcxi_test_CPPFLAGS)
AC_SUBST(libcxi_test_LDFLAGS)
AC_SUBST(libcxi_test_LIBS)

AC_CONFIG_HEADERS(config.h)
AC_CONFIG_FILES([Makefile libcxi.pc retry_handler/cxi_rh@.service])

AC_ARG_WITH([systemdsystemunitdir],
     [AS_HELP_STRING([--with-systemdsystemunitdir=DIR], [Directory for systemd service files])],,
     [with_systemdsystemunitdir=auto])
AS_IF([test "x$with_systemdsystemunitdir" = "xyes" -o "x$with_systemdsystemunitdir" = "xauto"], [
     def_systemdsystemunitdir=$($PKG_CONFIG --variable=systemdsystemunitdir systemd)

     AS_IF([test "x$def_systemdsystemunitdir" = "x"],
   [AS_IF([test "x$with_systemdsystemunitdir" = "xyes"],
    [AC_MSG_ERROR([systemd support requested but pkg-config unable to query systemd package])])
    with_systemdsystemunitdir=no],
   [with_systemdsystemunitdir="$def_systemdsystemunitdir"])])
AS_IF([test "x$with_systemdsystemunitdir" != "xno"],
      [AC_SUBST([systemdsystemunitdir], [$with_systemdsystemunitdir])])
AM_CONDITIONAL([HAVE_SYSTEMD], [test "x$with_systemdsystemunitdir" != "xno"])

AC_ARG_WITH([udevrulesdir],
     [AS_HELP_STRING([--with-udevrulesdir=DIR], [Directory for udev rules])],,
     [with_udevrulesdir=auto])
AS_IF([test "x$with_udevrulesdir" = "xyes" -o "x$with_udevrulesdir" = "xauto"], [
     def_udevrulesdir=$($PKG_CONFIG --variable=udevdir udev)/rules.d
     AS_IF([test "x$def_udevrulesdir" = "x"],
          [AS_IF([test "x$with_udevrulesdir" = "xyes"],
               [AC_MSG_ERROR([udev support requested but pkg-config unable to query udev package])])
          with_udevrulesdir=no],
          [with_udevrulesdir="$def_udevrulesdir"])])
AS_IF([test "x$with_udevrulesdir" != "xno"],
     [AC_SUBST([udevrulesdir], [$with_udevrulesdir])])
AM_CONDITIONAL([HAVE_UDEV], [test "x$with_udevrulesdir" != "xno"])

dnl search for the zero library
AC_ARG_WITH([ze], [AS_HELP_STRING([--with-ze=DIR],
                                   [Alternate location for ze dir])],
                                   [], [withval="/usr"])
AS_IF([test -d $withval && test -d $withval/include/level_zero],
    [AC_MSG_RESULT([Have ZE $withval])
     CFLAGS="$CFLAGS -D HAVE_ZE_SUPPORT"
     CFLAGS="$CFLAGS -I $withval/include/"
     CFLAGS="$CFLAGS -Wno-strict-prototypes"]
     [AC_CHECK_HEADER(level_zero/ze_api.h, [],
                      [AC_MSG_ERROR([level_zero/ze_api.h not found])])]
     LDFLAGS="$LDFLAGS -ldl",
    [AC_MSG_RESULT([No ZE])])

dnl search for the ROCm HIP library
AC_ARG_WITH([rocm], [AS_HELP_STRING([--with-rocm=DIR],
                                   [Alternate location for rocm dir])],
                                   [], [withval="/opt/rocm"])
AS_IF([test -d $withval],
    [AC_MSG_RESULT([Have ROCm $withval])
     CFLAGS="$CFLAGS -D HAVE_HIP_SUPPORT -D __HIP_PLATFORM_HCC__ -D __HIP_PLATFORM_AMD__"
     CFLAGS="$CFLAGS -I $withval/include/"
     CFLAGS="$CFLAGS -Wno-strict-prototypes"]
     [AC_CHECK_HEADER(hip/hip_runtime_api.h, [],
                      [AC_MSG_ERROR([hip/hip_runtime_api.h not found])])]
     LDFLAGS="$LDFLAGS -ldl",
    [AC_MSG_RESULT([No HIP])])

dnl search for the cuda library
AC_ARG_WITH([cuda],
            [AS_HELP_STRING([--with-cuda=DIR],
                            [Alternate location for CUDA dir])],
                            [], [withval="/usr/local/cuda"])
AS_IF([test -d $withval],
    [AC_MSG_RESULT([Have CUDA $withval])
     CFLAGS="$CFLAGS -D HAVE_CUDA_SUPPORT"
     CFLAGS="$CFLAGS -I $withval/include"]
     [AC_CHECK_HEADER(cuda_runtime_api.h, [],
                      [AC_MSG_ERROR([cuda_runtime_api.h not found])])]
     AC_CHECK_DECL(HAVE_HIP_SUPPORT, [], [LDFLAGS="$LDFLAGS -ldl"], []),
    [AC_MSG_RESULT([No CUDA])])

AC_ARG_ENABLE([asan],
    [AS_HELP_STRING([--enable-asan],
	            [Enable address sanitizer @<:@default=no@:>@])
    ], [], [enable_asan=no])

AS_IF([test x"$enable_asan" != x"no"],
      [CFLAGS="-fsanitize=address $CFLAGS"])

AC_CHECK_HEADER(sensors/sensors.h, [],
[AC_MSG_ERROR([sensors/sensors.h not found])])

AC_CHECK_PROG([PANDOC],[pandoc],[yes])
AS_IF([test "x${PANDOC}" = x"yes"],
      [AC_MSG_RESULT([Found pandoc: $PANDOC])],
      [AC_MSG_WARN([Required executable pandoc not found, man pages will not be rebuilt])])
AM_CONDITIONAL([HAVE_PANDOC],[test "x${PANDOC}" = "xyes"])

AC_OUTPUT
AC_MSG_RESULT([
        $PACKAGE $VERSION
        =====

        prefix:                 ${prefix}
        sysconfdir:             ${sysconfdir}
        libdir:                 ${libdir}
        includedir:             ${includedir}
        systemdunitdir:         ${systemdsystemunitdir}
        udevrulesdir:           ${udevrulesdir}

        compiler:               ${CC}
        cflags:                 ${CFLAGS}
        ldflags:                ${LDFLAGS}
])
