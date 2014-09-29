dnl Configury specific to the libfabrics sockets provider

dnl Called to configure this provider
AC_DEFUN([FI_SOCKETS_CONFIGURE],[
	AC_MSG_NOTICE([*** Configuring sockets provider])
	AC_ARG_ENABLE([sockets],
	      [AS_HELP_STRING([--enable-sockets],
			      [Enable sockets provider @<:@default=yes@:>@])
	      ],
	      [],
	      [enable_sockets=yes])

	AS_IF([test x"$enable_sockets" = x"dl"],
		[sockets_dl=yes; enable_sockets=yes])

	AS_IF([test x"$enable_sockets" = x"yes"],
		[AC_DEFINE([HAVE_SOCKETS], [1],
			[Define if sockets provider is enabled])
		],
		[AC_MSG_NOTICE(Sockets provider not enabled)])

	AS_IF([test x"$sockets_dl" = x"yes"],
	      [AC_DEFINE([HAVE_SOCKETS_DL], [1],
		[Build sockets provider as module])])
])

dnl A separate macro for AM CONDITIONALS, since they cannot be invoked
dnl conditionally
AC_DEFUN([FI_SOCKETS_CONDITIONALS],[
	AM_CONDITIONAL([HAVE_SOCKETS], [test x"$enable_sockets" = x"yes"])
	AM_CONDITIONAL([HAVE_SOCKETS_DL], [test x"$sockets_dl" = x"yes"])
])
