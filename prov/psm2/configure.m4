dnl Configury specific to the libfabric PSM2 provider

dnl Called to configure this provider
dnl
dnl Arguments:
dnl
dnl $1: action if configured successfully
dnl $2: action if not configured successfully
dnl
AC_DEFUN([FI_PSM2_CONFIGURE],[
	# Determine if we can support the psm2 provider
	psm2_happy=0
	AS_IF([test x"$enable_psm2" != x"no"],
	      [FI_CHECK_PACKAGE([psm2],
				[psm2.h],
				[psm2],
				[psm_init],
				[],
				[],
				[],
				[psm2_happy=1],
				[psm2_happy=0])
	      ])

	AS_IF([test $psm2_happy -eq 1], [$1], [$2])

	psm2_CPPFLAGS="$CPPFLAGS $psm2_CPPFLAGS"
	psm2_LDFLAGS="$LDFLAGS $psm2_LDFLAGS"
	psm2_LIBS="$LIBS $psm2_LIBS"
	CPPFLAGS=
	LDFLAGS=
	LIBS=
])

