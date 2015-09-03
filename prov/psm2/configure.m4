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
	psm2_orig_LIBS=$psm2_LIBS
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

	AC_MSG_CHECKING([if use dlopen to load PSM2 library])
	AC_RUN_IFELSE([AC_LANG_SOURCE(
				      [[
				      int main()
				      {
					#ifndef PSMX_DL
					#define PSMX_DL 1
					#endif
					return PSMX_DL ? 0 : 1;
				      }
				      ]]
				     )],
			[AC_MSG_RESULT([yes]); psm2_LIBS=$psm2_orig_LIBS],
			[AC_MSG_RESULT([no]); psm2_LDFLAGS="$LDFLAGS $psm2_LDFLAGS"])

	psm2_CPPFLAGS="$CPPFLAGS $psm2_CPPFLAGS"
	CPPFLAGS="$psm2_orig_CPPFLAGS"
	LDFLAGS="$psm2_orig_LDFLAGS"
])

