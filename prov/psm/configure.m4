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

	AC_ARG_WITH([psm],
	    [AS_HELP_STRING([--with-psm=@<:@PSM installation path@:>@],
			    [Provide path to PSM installation])
	    ],
	    [AS_CASE([$with_psm],
		     [yes|no], [],
		     [CPPFLAGS="-I$with_psm/include $CPPFLAGS"
		      LDFLAGS="-L$with_psm/lib64 -Wl,-rpath=$with_psm/lib64 $LDFLAGS"])
	     enable_psm=yes
	    ])

	AC_ARG_WITH([psm-include],
            [AS_HELP_STRING([--with-psm-include=@<:@PSM include path@:>@],
                            [Provide path to PSM include files])
            ],
            [AS_CASE([$with_psm_include],
                     [yes|no], [],
                     [CPPFLAGS="-I$with_psm_include $CPPFLAGS"])
	     enable_psm=yes
            ])

	AC_ARG_WITH([psm-lib],
            [AS_HELP_STRING([--with-psm-lib=@<:@PSM library path@:>@],
                            [Provide path to PSM library files])
            ],
            [AS_CASE([$with_psm_lib],
                     [yes|no], [],
                     [LDFLAGS="-L$with_psm_lib -Wl,-rpath=$with_psm_lib $LDFLAGS"])
	     enable_psm=yes
            ])

	AS_CASE([$enable_psm],
	[auto], [AC_CHECK_LIB(psm_infinipath, psm_init,
			[AC_CHECK_HEADER([psm.h], [enable_psm=yes], [enable_psm=no])],
			[enable_psm=no])
		],
	[yes],	[AC_CHECK_LIB(psm_infinipath, psm_init,
			[AC_CHECK_HEADER([psm.h], [],
				[AC_MSG_ERROR([psm.h not found. Provide the correct path to PSM with --with-psm-include (or --with-psm)])])
			],
			[AC_MSG_ERROR([psm_init() not found. Provide the correct path to PSM --with-psm-lib])])
		],
	[dl],	[enable_psm=yes; psm_dl=yes],
	[no],	[],
	[])

	AS_IF([test x"$psm_dl" = x"yes"],
		[AC_DEFINE([HAVE_PSM_DL], [1],
			[Define if PSM will be built as module])])

	AS_IF([test x"$enable_psm" = x"yes"],
		[AC_DEFINE([HAVE_PSM], [1], [Define if PSM is enabled])
		 LIBS="-lpsm_infinipath $LIBS"
		],
		[AC_MSG_NOTICE(PSM not enabled)])
])

dnl A separate macro for AM CONDITIONALS, since they cannot be invoked
dnl conditionally
AC_DEFUN([FI_PSM_CONDITIONALS],[
	AM_CONDITIONAL([HAVE_PSM], [test x"$enable_psm" = x"yes"])
	AM_CONDITIONAL([HAVE_PSM_DL], [test x"$psm_dl" = x"yes"])
])
