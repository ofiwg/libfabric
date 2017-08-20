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
        have_gtest=false
        gtest_tests_present=true
	rxdtest_CPPFLAGS=
	rxdtest_LDFLAGS=
        rxdtest_LIBS=

	AS_IF([test x"$enable_rxd" != x"no"], [rxd_h_happy=1])
        AS_IF([test $rxd_h_happy -eq 1], [$1], [$2])

	AS_IF([test -d $srcdir/prov/rxd/test],
                     [AC_ARG_WITH([criterion], [AS_HELP_STRING([--with-gtest],
                     [Location for criterion unit testing framework])])],
                     [gtest_present=false])

	if test "$with_gtest" != "" && test "$with_gtest" != "no"; then
		AS_IF([test "$gtest_tests_present" = "true"],
		      [AC_MSG_CHECKING([gtest path])
		       if test -d "$with_gtest"; then
                                AC_MSG_RESULT([yes])
				rxdtest_CPPFLAGS="-I$with_gtest/include $rxdtest_CPPFLAGS"
				rxdtest_LIBS="-lgtest $rxdtest_LIBS"
				rxdtest_LDFLAGS="-L$with_gtest -Wl,-rpath=$with_gtest $rxdtest_LDFLAGS"
				have_gtest=true
		       else
				AC_MSG_RESULT([no])
                                AC_MSG_ERROR([gtest requested but invalid path given])
		       fi],
		      [AC_MSG_ERROR([gtest requested tests not available])])
	fi

	AM_CONDITIONAL([HAVE_GTEST], [test "x$have_gtest" = "xtrue"])
])
