dnl Configury specific to the libfabric Amazon provider

dnl Called to configure this provider
dnl
dnl Arguments:
dnl
dnl $1: action if configured successfully
dnl $2: action if not configured successfully
dnl
AC_DEFUN([FI_EFA_CONFIGURE],[
	# Determine if we can support the efa provider
	efa_happy=0
	efa_h_enable_poisoning=0
	AS_IF([test x"$enable_efa" != x"no"],
	      [efa_happy=1])

	AC_ARG_ENABLE([efa-mem-poisoning],
		[AS_HELP_STRING([--enable-efa-mem-poisoning],
			[Enable EFA memory poisoning support for debugging @<:@default=no@:>@])
		],
		[efa_h_enable_poisoning=$enableval],
		[efa_h_enable_poisoning=no])
	AS_IF([test x"$efa_h_enable_poisoning" == x"yes"],
		[AC_DEFINE([ENABLE_EFA_POISONING], [1],
			[EFA memory poisoning support for debugging])],
		[])

	AC_MSG_CHECKING([for GCC])
	AS_IF([test x"$GCC" = x"yes"],
	      [AC_MSG_RESULT([yes])],
	      [AC_MSG_RESULT([no])
	       efa_happy=0])

	# verbs definitions file depends on linux/types.h
	AC_CHECK_HEADER([linux/types.h], [], [efa_happy=0])


	AS_IF([test $efa_happy -eq 1 ], [$1], [$2])
])
