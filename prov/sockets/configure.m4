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

	sockets_dl=0
	AS_IF([test x"$enable_sockets" = x"dl"],
	      [sockets_dl=1
	       enable_sockets=yes])

	# First, determine if we can support the sockets provider
	sockets_happy=0
	AS_IF([test x"$enable_sockets" != x"no"],
	      [sockets_happy=1
	       AC_CHECK_HEADER([sys/socket.h], [], [sockets_happy=0])
	      ])

	# If sockets was specifically requested but we can't build it,
	# error.
	AS_IF([test "$enable_sockets $sockets_happy" = "yes 0"],
	      [AC_MSG_WARN([sockets provider was requested, but cannot be compiled])
	       AC_MSG_ERROR([Cannot continue])
	      ])

	AS_IF([test $sockets_happy -eq 1],
		[AS_IF([test $sockets_dl -eq 1],
			[AC_MSG_NOTICE([sockets provider to be built as a DSO])],
			[AC_MSG_NOTICE([sockets provider to be built statically])])
		],
		[AC_MSG_NOTICE(sockets provider disabled)])

	AC_DEFINE_UNQUOTED([HAVE_SOCKETS_DL], [$sockets_dl],
		[Whether sockets should be built as as DSO])
])

dnl A separate macro for AM CONDITIONALS, since they cannot be invoked
dnl conditionally
AC_DEFUN([FI_SOCKETS_CONDITIONALS],[
	AM_CONDITIONAL([HAVE_SOCKETS], [test x"$enable_sockets" = x"yes"])
	AM_CONDITIONAL([HAVE_SOCKETS_DL], [test x"$sockets_dl" = x"yes"])
])
