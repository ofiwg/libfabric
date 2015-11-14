dnl Configury specific to the libfabrics GNI provider

dnl Called to configure this provider

m4_include([config/fi_pkg.m4])

AC_DEFUN([FI_GNI_CONFIGURE],[
	# Determine if we can support the gni provider
        # have to pull in pkg.m4 manually
	ugni_lib_happy=0
	gni_header_happy=0
	AS_IF([test x"$enable_gni" != x"no"],
	      [FI_PKG_CHECK_MODULES([CRAY_UGNI], [cray-ugni],
                                 [ugni_lib_happy=1
                                  CPPFLAGS="$CRAY_UGNI_CFLAGS $CPPFLAGS"
                                  LDFLAGS="$CRAY_UGNI_LIBS $LDFLAGS"
                                 ],
                                 [ugni_lib_happy=0])
               FI_PKG_CHECK_MODULES([CRAY_GNI_HEADERS], [cray-gni-headers],
                                 [gni_header_happy=1
                                  CPPFLAGS="$CRAY_GNI_HEADERS_CFLAGS $CPPFLAGS"
                                  LDFLAGS="$CRAY_GNI_HEADER_LIBS $LDFLAGS"
                                 ],
                                 [gni_header_happy=0])
               FI_PKG_CHECK_MODULES_STATIC([CRAY_ALPS_LLI], [cray-alpslli],
                                 [alps_lli_happy=1
                                  CPPFLAGS="$CRAY_ALPS_LLI_CFLAGS $CPPFLAGS"
                                  LDFLAGS="$CRAY_ALPS_LLI_LIBS $LDFLAGS"
                                 ],
                                 [alps_lli_happy=0])
               FI_PKG_CHECK_MODULES([CRAY_ALPS_UTIL], [cray-alpsutil],
                                 [alps_util_happy=1
                                  CPPFLAGS="$CRAY_ALPS_UTIL_CFLAGS $CPPFLAGS"
                                  LDFLAGS="$CRAY_ALPS_UTIL_LIBS $LDFLAGS"
                                 ],
                                 [alps_util_happy=0])
	       ])

	have_criterion=false

	AC_ARG_WITH([criterion],
		[AS_HELP_STRING([--with-criterion],
			       [Location for criterion unit testing framework])])

	if test "$with_criterion" != "" && test "$with_criterion" != "no"; then
		AC_MSG_CHECKING([criterion path])
		if test -d "$with_criterion"; then
			AC_MSG_RESULT([yes])
			CPPFLAGS="-I$with_criterion/include $CPPFLAGS"
			if test -d "$with_criterion/lib"; then
				LDFLAGS="$CRAY_ALPS_LLI_STATIC_LIBS -L$with_criterion/lib $LDFLAGS"
				have_criterion=true
			elif test -d "$with_criterion/lib64"; then
				LDFLAGS="$CRAY_ALPS_LLI_STATIC_LIBS -L$with_criterion/lib64 $LDFLAGS"
				have_criterion=true
			else
				have_criterion=false
			fi
			FI_PKG_CHECK_MODULES([CRAY_PMI], [cray-pmi],
					   [],
					   [have_criterion=false])
		else
			AC_MSG_RESULT([no])
			AC_MSG_ERROR([criterion requested but invalid path given])
		fi
	fi

	AM_CONDITIONAL([HAVE_CRITERION], [test "x$have_criterion" = "xtrue"])


	AS_IF([test $gni_header_happy -eq 1 -a $ugni_lib_happy -eq 1 \
               -a $alps_lli_happy -eq 1 -a $alps_util_happy -eq 1], [$1], [$2])
])

