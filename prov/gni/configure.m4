dnl Configury specific to the libfabrics GNI provider

dnl Called to configure this provider
dnl
dnl Arguments:
dnl
dnl $1: action if configured successfully
dnl $2: action if not configured successfully
dnl

# _PKG_CONFIG2([VARIABLE], [COMMAND1], [COMMAND2], [MODULES])
# ---------------------------------------------
m4_define([_PKG_CONFIG2],
[if test -n "$$1"; then
	pkg_cv_[]$1="$$1"
elif test -n "$PKG_CONFIG"; then
	PKG_CHECK_EXISTS([$4],
		[pkg_cv_[]$1=`$PKG_CONFIG --[]$2 --[]$3 "$4" 2>/dev/null`
		test "x$?" != "x0" && pkg_failed=yes ],
		[pkg_failed=yes])
else
	pkg_failed=untried
fi[]dnl
])# _PKG_CONFIG2

# PKG_CHECK_MODULES_STATIC(VARIABLE-PREFIX, MODULES, [ACTION-IF-FOUND],
# [ACTION-IF-NOT-FOUND])
#
#
# Variation of PGK_CHECK_MODULES which also defines $1_LIBS_STATIC
# by using the pkg-config --libs --static pkg-config option.
#
# --------------------------------------------------------------
AC_DEFUN([PKG_CHECK_MODULES_STATIC],
[AC_REQUIRE([PKG_PROG_PKG_CONFIG])dnl
AC_ARG_VAR([$1][_CFLAGS], [C compiler flags for $1, overriding pkg-config])dnl
AC_ARG_VAR([$1][_LIBS], [linker flags for $1, overriding pkg-config])dnl
AC_ARG_VAR([$1][_STATIC_LIBS], [static linker flags for $1, overriding pkg-config])dnl

pkg_failed=no
AC_MSG_CHECKING([for $1])

_PKG_CONFIG([$1][_CFLAGS], [cflags], [$2])
_PKG_CONFIG([$1][_LIBS], [libs], [$2])
_PKG_CONFIG2([$1][_STATIC_LIBS], [libs], [static], [$2])

m4_define([_PKG_TEXT], [Alternatively, you may set the environment variables $1[]_CFLAGS
and $1[]_LIBS to avoid the need to call pkg-config.
See the pkg-config man page for more details.])

if test $pkg_failed = yes; then
	AC_MSG_RESULT([no])
	_PKG_SHORT_ERRORS_SUPPORTED
	if test $_pkg_short_errors_supported = yes; then
		$1[]_PKG_ERRORS=`$PKG_CONFIG --short-errors --print-errors --cflags --libs "$2" 2>&1`
	else
		$1[]_PKG_ERRORS=`$PKG_CONFIG --print-errors --cflags --libs "$2" 2>&1`
	fi
	# Put the nasty error message in config.log where it belongs
	echo "$$1[]_PKG_ERRORS" >&AS_MESSAGE_LOG_FD

	m4_default([$4], [AC_MSG_ERROR(
[Package requirements ($2) were not met:

$$1_PKG_ERRORS

Consider adjusting the PKG_CONFIG_PATH environment variable if you
installed software in a non-standard prefix.

_PKG_TEXT])[]dnl
])
elif test $pkg_failed = untried; then
	AC_MSG_RESULT([no])
	m4_default([$4], [AC_MSG_FAILURE(
[The pkg-config script could not be found or is too old.  Make sure it
is in your PATH or set the PKG_CONFIG environment variable to the full
path to pkg-config.

_PKG_TEXT

To get pkg-config, see <http://pkg-config.freedesktop.org/>.])[]dnl
])
else
	$1[]_CFLAGS=$pkg_cv_[]$1[]_CFLAGS
	$1[]_LIBS=$pkg_cv_[]$1[]_LIBS
	$1[]_STATIC_LIBS=$pkg_cv_[]$1[]_STATIC_LIBS
        AC_MSG_RESULT([yes])
	$3
fi[]dnl
])# PKG_CHECK_MODULES_STATIC

AC_DEFUN([FI_GNI_CONFIGURE],[
	# Determine if we can support the gni provider
        # have to pull in pkg.m4 manually
m4_include([config/pkg.m4])
	ugni_lib_happy=0
	gni_header_happy=0
	AS_IF([test x"$enable_gni" != x"no"],
	      [PKG_CHECK_MODULES([CRAY_UGNI], [cray-ugni],
                                 [ugni_lib_happy=1
                                  CPPFLAGS="$CRAY_UGNI_CFLAGS $CPPFLAGS"
                                  LDFLAGS="$CRAY_UGNI_LIBS $LDFLAGS"
                                 ],
                                 [ugni_lib_happy=0])
               PKG_CHECK_MODULES([CRAY_GNI_HEADERS], [cray-gni-headers],
                                 [gni_header_happy=1
                                  CPPFLAGS="$CRAY_GNI_HEADERS_CFLAGS $CPPFLAGS"
                                  LDFLAGS="$CRAY_GNI_HEADER_LIBS $LDFLAGS"
                                 ],
                                 [gni_header_happy=0])
               PKG_CHECK_MODULES_STATIC([CRAY_ALPS_LLI], [cray-alpslli],
                                 [alps_lli_happy=1
                                  CPPFLAGS="$CRAY_ALPS_LLI_CFLAGS $CPPFLAGS"
                                  LDFLAGS="$CRAY_ALPS_LLI_LIBS $LDFLAGS"
                                 ],
                                 [alps_lli_happy=0])
               PKG_CHECK_MODULES([CRAY_ALPS_UTIL], [cray-alpsutil],
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
			PKG_CHECK_MODULES([CRAY_PMI], [cray-pmi],
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

