dnl SPDX-License-Identifier: GPL-2.0
dnl
dnl Copyright 2018 Cray Inc. All rights reserved.

dnl CXI provider specific configuration

dnl Called to configure this provider
dnl
dnl Arguments:
dnl
dnl $1: action if configured successfully
dnl $2: action if not configured successfully
dnl
AC_DEFUN([FI_CXI_CONFIGURE],[
	# Determine if we can support the CXI provider
	have_cxi=false
	cxi_CPPFLAGS=
	cxi_LDFLAGS=
	have_criterion=false
	cxitest_CPPFLAGS=
	cxitest_LDFLAGS=
	cxitest_LIBS=

	AS_IF([test x"$enable_cxi" != x"no"],
		[FI_PKG_CHECK_MODULES([CXI], [libcxi],
			[cxi_CPPFLAGS=$CXI_CFLAGS
			cxi_LDFLAGS=$CXI_LIBS
			have_cxi=true])])

	AC_ARG_WITH([criterion], [AS_HELP_STRING([--with-criterion],
		[Location for criterion unit testing framework.])])

	AS_IF([test "$with_criterion" != ""],
		[cxitest_CPPFLAGS="-I$with_criterion/include"
		cxitest_LDFLAGS="-L$with_criterion/lib -Wl,-rpath=$(realpath $with_criterion/lib)"
		cxitest_LIBS="-lcriterion"
		have_criterion=true])

	AM_CONDITIONAL([HAVE_CRITERION], [test "x$have_criterion" = "xtrue"])

	AC_SUBST(cxi_CPPFLAGS)
	AC_SUBST(cxi_LDFLAGS)
	AC_SUBST(cxitest_CPPFLAGS)
	AC_SUBST(cxitest_LDFLAGS)
	AC_SUBST(cxitest_LIBS)

	AS_IF([test "x$have_cxi" = "xtrue" ], [$1], [$2])
])
