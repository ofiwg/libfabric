dnl Configury specific to the libfabrics verbs provider

dnl Called to configure this provider
dnl
dnl Arguments:
dnl
dnl $1: action if configured successfully
dnl $2: action if not configured successfully
dnl
AC_DEFUN([FI_VERBS_CONFIGURE],[
	# Determine if we can support the verbs provider
	verbs_happy=0
	AS_IF([test x"$enable_verbs" != x"no"],
	      [verbs_happy=1
	       AC_CHECK_HEADER([infiniband/verbs.h], [], [verbs_happy=0])
	       AC_CHECK_HEADER([rdma/rsocket.h], [], [verbs_happy=0])
	       AC_CHECK_LIB([ibverbs], [ibv_open_device], [], [verbs_happy=0])
	       AC_CHECK_LIB([rdmacm], [rsocket], [], [verbs_happy=0])
	      ])

	AS_IF([test $verbs_happy -eq 1], [$1], [$2])
])

dnl A separate macro for AM CONDITIONALS, since they cannot be invoked
dnl conditionally
AC_DEFUN([FI_VERBS_CONDITIONALS],[
	AM_CONDITIONAL([HAVE_VERBS], [test x"$enable_verbs" = x"yes"])
	AM_CONDITIONAL([HAVE_VERBS_DL], [test x"$verbs_dl" = x"yes"])
])
