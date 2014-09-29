dnl Configury specific to the libfabrics verbs provider

dnl Called to configure this provider
AC_DEFUN([FI_VERBS_CONFIGURE],[
	AC_MSG_NOTICE([*** Configuring verbs provider])
	AC_ARG_ENABLE([verbs],
		      [AS_HELP_STRING([--enable-verbs],
				      [Enable verbs provider @<:@default=auto@:>@])
		      ],
		      [],
		      [enable_verbs=auto])

	AS_CASE([$enable_verbs],
		[auto], [verbs_fail=no],
		[yes],  [verbs_fail=yes],
		[dl],   [verbs_dl=yes; enable_verbs=yes],
		[no],   [],
		[])

	AS_IF([test x"$enable_verbs" != x"no"],
		[AC_CHECK_LIB(ibverbs, ibv_open_device,
				[AC_CHECK_LIB(rdmacm, rsocket,
					[enable_verbs=yes],
					[enable_verbs=no])
				],
				[enable_verbs=no])
		],
		[AC_MSG_NOTICE(Verbs provider not enabled)])

	AS_IF([test x"$enable_verbs $verbs_fail" = x"no yes"],
		[AC_MSG_ERROR(libfabric requires libibverbs, librdmacm 1.0.16 or greater)])

	AS_IF([test x"$enable_verbs" = x"yes"],
		[AC_DEFINE([HAVE_VERBS], [1], [Define if verbs provider is enabled])
	         LIBS=" -libverbs -lrdmacm $LIBS"
		])

	AS_IF([test x"$verbs_dl" = x"yes"],
		[AC_DEFINE([HAVE_VERBS_DL], [1], [Define if verbs should be built as module])])

# JMS This should have a test seeing if MLX4 direct is *available* or
# not.  But I don't know what headers/libraries to test for...  (I
# might also be mis-understanding what this --enable-direct=mlx4
# switch is for...?)
	AS_CASE([$enable_direct],
		[mlx4], [AC_DEFINE([HAVE_MLX4_DIRECT], [1],
		[Define if mlx4 direct provider is enabled])],
	[])
])

dnl A separate macro for AM CONDITIONALS, since they cannot be invoked
dnl conditionally
AC_DEFUN([FI_VERBS_CONDITIONALS],[
	AM_CONDITIONAL([HAVE_VERBS], [test x"$enable_verbs" = x"yes"])
	AM_CONDITIONAL([HAVE_VERBS_DL], [test x"$verbs_dl" = x"yes"])
	AM_CONDITIONAL([HAVE_MLX4_DIRECT], [test x"$enable_direct" = x"mlx4"])
])
