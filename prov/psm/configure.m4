dnl Configury specific to the libfabrics PSM provider

dnl Called to configure this provider
AC_DEFUN([FI_PSM_CONFIGURE],[
	AC_MSG_NOTICE([*** Configuring PSM provider])
	AC_ARG_ENABLE([psm],
	      [AS_HELP_STRING([--enable-psm],
			      [Enable PSM provider @<:@default=auto@:>@])
	      ],
	      [],
	      [enable_psm=auto])

	psm_dl=0
	AS_CASE([$enable_psm],
	[yes|no], [],
	[dl],     [enable_psm=yes psm_dl=1],
	[auto],   [],
	[AS_IF([test ! -d "$enable_psm"],
	       [AC_MSG_WARN([supplied directory "$enable_psm" does not exist])
	        AC_MSG_ERROR([Cannot continue])
	       ])
	 AS_IF([test -d "$enable_psm/include"],
	       [CPPFLAGS="-I$enable_psm/include"],
	       [AC_MSG_WARN([could not find "include" subdirectory in supplied "$enable_psm" directory"])
	        AC_MSG_ERROR([Cannot continue])
	       ])
	 AS_IF([test -d "$enable_psm/lib"],
	       [LDFLAGS="-L$enable_psm/lib"],
	       [AS_IF([test -d "$enable_psm/lib64"],
		      [LDFLAGS="-L$enable_psm/lib64"],
		      [AC_MSG_WARN([could not find "lib" or "lib64" subdirectories in supplied "$enable_psm" directory"])
		       AC_MSG_ERROR([Cannot continue])
		      ])
	       ])
	])

	# First, determine if we can support the psm provider
	psm_happy=0
	AS_IF([test "x$enable_psm" != "xno"],
	      [psm_happy=1
	       AC_CHECK_HEADER([psm.h], [], [psm_happy=0])
	       AC_CHECK_LIB([psm_infinipath], [psm_init], [], [psm_happy=0])
	       ])

	# If psm was specifically requested but we can't build it,
	# error.
	AS_IF([test "$enable_psm $psm_happy" = "yes 0"],
	      [AC_MSG_WARN([psm provider was requested, but cannot be compiled])
	       AC_MSG_ERROR([Cannot continue])
	      ])

	AS_IF([test $psm_happy -eq 1],
	      [AS_IF([test $psm_dl -eq 1],
		     [AC_MSG_NOTICE([psm provider to be built as a DSO])],
		     [AC_MSG_NOTICE([psm provider to be built statically])])
	      ],
	      [AC_MSG_NOTICE([psm provider disabled])])

	AC_DEFINE_UNQUOTED([HAVE_PSM_DL], [$psm_dl],
		[Whether psm should be built as as DSO])
])

dnl A separate macro for AM CONDITIONALS, since they cannot be invoked
dnl conditionally
AC_DEFUN([FI_PSM_CONDITIONALS],[
	AM_CONDITIONAL([HAVE_PSM], [test x"$enable_psm" = x"yes"])
	AM_CONDITIONAL([HAVE_PSM_DL], [test x"$psm_dl" = x"yes"])
])
