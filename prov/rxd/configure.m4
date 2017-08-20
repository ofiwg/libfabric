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
	have_gmock=false
	gmock_tests_present=true
	have_cmock=false
	cmock_tests_present=true
	rxdtest_CPPFLAGS=
	rxdtest_LDFLAGS=
        rxdtest_LIBS=

	AS_IF([test x"$enable_rxd" != x"no"], [rxd_h_happy=1])
        AS_IF([test $rxd_h_happy -eq 1], [$1], [$2])

	AS_IF([test -d $srcdir/prov/rxd/test],
                     [AC_ARG_WITH([gtest], [AS_HELP_STRING([--with-gtest],
                     [Location for gtest unit testing framework])])],
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
				if tets -d "$prefix"; then
					rxdtest_CPPFLAGS="-I$prefix/include $rxdtest_CPPFLAGS"
					rxdtest_LIBS="-lfabric $rxdtest_LIBS"
					rxdtest_LDFLAGS="-L$prefix/lib -Wl,-rpath=$prefix $rxdtest_LDFLAGS"
				fi
		       else
				AC_MSG_RESULT([no])
                                AC_MSG_ERROR([gtest requested but invalid path given])
		       fi],
		      [AC_MSG_ERROR([gtest requested tests not available])])
	fi

	AS_IF([test -d $srcdir/prov/rxd/test],
                     [AC_ARG_WITH([gmock], [AS_HELP_STRING([--with-gmock],
                     [Location for gmock mocking framework])])],
                     [gmock_present=false])

	if test "$with_gmock" != "" && test "$with_gmock" != "no"; then
		AS_IF([test "$gmock_tests_present" = "true"],
		      [AC_MSG_CHECKING([gmock path])
		       if test -d "$with_gmock"; then
                                AC_MSG_RESULT([yes])
				rxdtest_CPPFLAGS="-I$with_gmock/include $rxdtest_CPPFLAGS"
				rxdtest_LIBS="-lgmock $rxdtest_LIBS"
				rxdtest_LDFLAGS="-L$with_gmock -Wl,-rpath=$with_gmock $rxdtest_LDFLAGS"
				have_gmock=true
		       else
				AC_MSG_RESULT([no])
                                AC_MSG_ERROR([gmock requested but invalid path given])
		       fi],
		      [AC_MSG_ERROR([gtest requested tests not available])])
	fi

	AS_IF([test -d $srcdir/prov/rxd/test],
                     [AC_ARG_WITH([cmock], [AS_HELP_STRING([--with-cmock],
                     [Location for cmock helper framework])])],
                     [cmock_present=false])

	if test "$with_cmock" != "" && test "$with_cmock" != "no"; then
		AS_IF([test "$cmock_tests_present" = "true"],
		      [AC_MSG_CHECKING([cmock path])
		       if test -d "$with_cmock"; then
                                AC_MSG_RESULT([yes])
				rxdtest_CPPFLAGS="-I$with_cmock/include $rxdtest_CPPFLAGS"
				have_cmock=true
		       else
				AC_MSG_RESULT([no])
                                AC_MSG_ERROR([cmock requested but invalid path given])
		       fi],
		      [AC_MSG_ERROR([gtest requested tests not available])])
	fi

	AC_SUBST(rxdtest_CPPFLAGS)
	AC_SUBST(rxdtest_LDFLAGS)
	AC_SUBST(rxdtest_LIBS)

	AM_CONDITIONAL([HAVE_GTEST], [test "x$have_gtest" = "xtrue" &&
				      test "x$have_cmock" = "xtrue"])
])
