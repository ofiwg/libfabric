dnl Configury specific to the libfabrics mxm provider

dnl Called to configure this provider
dnl
dnl Arguments:
dnl
dnl $1: action if configured successfully
dnl $2: action if not configured successfully
dnl
AC_DEFUN([FI_MXM_CONFIGURE],[
	# Determine if we can support the mxm provider
        mxm_happy=0
        AC_ARG_ENABLE([mxm],
		[AS_HELP_STRING([--enable-mxm],
                [Enable MXM provider @<:@default=no@:>@])
		],
		[],
                [enable_mxm=no])
        AC_CHECK_SIZEOF([void *])

        AS_IF([test x"$enable_mxm" = x"yes"],
              [AS_IF([test x"$ac_cv_sizeof_void_p" = x"8"],
                     [],
                     [
                         enable_mxm=no
                         AC_MSG_WARN([MXM OFI provider does not support 32 bit target platform])
                     ])],
              [])

        AC_ARG_WITH([mxm],
                [AS_HELP_STRING([--with-mxm=@<:@MXM installation path@:>@],
                        [Provide path to MXM installation])
		],
		[AS_CASE([$with_mxm],
                        [yes|no], [AC_DEFINE([HAVE_MXM], [1], [Define if MXM is enabled])],
                        [CPPFLAGS="-I$with_mxm/include $CPPFLAGS"
                        LDFLAGS="-L$with_mxm/lib $LDFLAGS"
			LIBS="-lmxm $LIBS"
                        AC_DEFINE([HAVE_MXM], [1], [Define if MXM is enabled])])
                ])

	AC_ARG_WITH([mxm-include],
		[AS_HELP_STRING([--with-mxm-include=@<:@MXM include path@:>@],
			[Provide path to MXM include files])
		],
		[AS_CASE([$with_mxm_include],
			[yes|no], [AC_DEFINE([HAVE_MXM], [1], [Define if MXM is enabled])],
			[CPPFLAGS="-I$with_mxm_include $CPPFLAGS"
			AC_DEFINE([HAVE_MXM], [1], [Define if MXM is enabled])
			])
                ])

	AC_ARG_WITH([mxm-lib],
		[AS_HELP_STRING([--with-mxm-lib=@<:@MXM library path@:>@],
			[Provide path to MXM library files])
		],
		[AS_CASE([$with_mxm_lib],
			[yes|no], [],
                        [LDFLAGS="-L$with_mxm_lib $LDFLAGS"
			LIBS="-lmxm $LIBS"
			AC_DEFINE([HAVE_MXM], [1], [Define if MXM is enabled])
			])
                ])

	AS_IF([test x"$enable_mxm" = x"yes"],
                [AC_CHECK_LIB(mxm, mxm_init,
                [AC_CHECK_HEADER([mxm/api/mxm_api.h], [],
                [AC_MSG_ERROR([mxm_api.h not found. Provide the correct path to MXM with --with-mxm-include (or --with-mxm)])]
		)],
                AC_MSG_ERROR([mxm_init() not found. Provide the correct path to MXM --with-mxm-lib]))
                        mxm_happy=1],
      		[AC_MSG_NOTICE(MXM not enabled)])

	AM_CONDITIONAL([HAVE_MXM], [test x"$enable_mxm" = x"yes"])



        AS_IF([test $mxm_happy -eq 1], [$1], [$2])
])

dnl A separate macro for AM CONDITIONALS, since they cannot be invoked
dnl conditionally
AC_DEFUN([FI_MXM_CONDITIONALS],[
	AM_CONDITIONAL([HAVE_MXM], [test $mxm_happy -eq 1])
        AM_CONDITIONAL([HAVE_MXM_DL], [test $mxm_dl -eq 1])
])
