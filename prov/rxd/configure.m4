dnl Configury specific to the libfabric rxd provider

dnl Called to configure this provider
dnl
dnl Arguments:
dnl
dnl $1: action if configured successfully
dnl $2: action if not configured successfully
dnl
AC_DEFUN([FI_RXD_CONFIGURE],[
	# Determine if we can support the rxd provider
	rxd_h_happy=0
	AS_IF([test x"$enable_rxd" != x"no"], [rxd_h_happy=1])

	rxd_cmocka_rpath=""
	AC_ARG_ENABLE([rxd-unit-test],
		[
			AS_HELP_STRING([--enable-rxd-unit-test=CMOCKA_INSTALL_DIR],
				[Provide a path to the CMocka installation directory
				in order to enable RXD Unit Tests.])
		],
		[rxd_cmocka_dir=$enableval],
		[enable_rxd_unit_test=no])

	AS_IF([test x"$enable_rxd_unit_test" != xno ],
	[
		rxd_unit_test=1
		FI_CHECK_PACKAGE(rxd_cmocka,
			[cmocka.h],
			[cmocka],
			[_expect_any],
			[],
			[$rxd_cmocka_dir],
			[],
			[],
			[AC_MSG_ERROR([Cannot compile RXD unit tests without a valid Cmocka installation directory.])],
			[
				#include <stdarg.h>
				#include <stddef.h>
				#include <stdint.h>
				#include <setjmp.h>
			])
		AS_IF([test x"$rxd_cmocka_LDFLAGS" != x""],
			[rxd_cmocka_rpath=" -R${rxd_cmocka_LDFLAGS:3} "])
	],
	[
		rxd_unit_test=0
	])

	AC_SUBST(rxd_cmocka_rpath)
	AC_DEFINE_UNQUOTED([RXD_UNIT_TEST], [$rxd_unit_test], [RXD unit testing])
	AM_CONDITIONAL([ENABLE_RXD_UNIT_TEST], [ test x"$enable_rxd_unit_test" != xno])

        AS_IF([test $rxd_h_happy -eq 1], [$1], [$2])
])
