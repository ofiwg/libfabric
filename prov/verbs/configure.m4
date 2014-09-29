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

	verbs_dl=0
	AS_IF([test "x$enable_verbs" = "xdl"],
	      [verbs_dl=1
	       enable_verbs=yes])

	# First, determine if we can support the verbs provider
	verbs_happy=0
	AS_IF([test "x$enable_verbs" != "xno"],
	      [verbs_happy=1
	       AC_CHECK_HEADER([infiniband/verbs.h], [], [verbs_happy=0])
	       AC_CHECK_HEADER([rdma/rsocket.h], [], [verbs_happy=0])
	       AC_CHECK_LIB([ibverbs], [ibv_open_device], [], [verbs_happy=0])
	       AC_CHECK_LIB([rdmacm], [rsocket], [], [verbs_happy=0])
	      ])

	# If verbs was specifically requested but we can't build it,
	# error.
	AS_IF([test "$enable_verbs $verbs_happy" = "yes 0"],
	      [AC_MSG_WARN([verbs provider was requested, but cannot be compiled])
	       AC_MSG_ERROR([Cannot continue])
	      ])

	AS_IF([test $verbs_happy -eq 1],
	      [AS_IF([test $verbs_dl -eq 1],
		     [AC_MSG_NOTICE([verbs provider to be built as a DSO])],
		     [AC_MSG_NOTICE([verbs provider to be built statically])])
	      ],
	      [AC_MSG_NOTICE([verbs provider disabled])])

	AC_DEFINE_UNQUOTED([HAVE_VERBS_DL], [$verbs_dl],
		[Whether verbs should be built as as DSO])

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
