dnl Configury specific to the libfabric PSM provider

dnl Called to configure this provider
dnl
dnl Arguments:
dnl
dnl $1: action if configured successfully
dnl $2: action if not configured successfully
dnl
AC_DEFUN([FI_PSM_CONFIGURE],[
	# Determine if we can support the psm provider
	psm_happy=0
	AS_IF([test x"$enable_psm" != x"no"],
	      [FI_CHECK_PACKAGE([psm],
				[psm.h],
				[psm_infinipath],
				[psm_init],
				[],
				[],
				[],
				[psm_happy=1],
				[psm_happy=0])
	       AS_IF([test $psm_happy -eq 1],
		     [AC_CHECK_LIB([psm_infinipath],
				   [psm_am_get_source],
				   [AC_MSG_RESULT([PSM version is not 1.x]); psm_happy=0],
				   [AC_MSG_RESULT([PSM version is 1.x])])
		     ])
	       AS_IF([test $psm_happy -eq 1],
		     [AC_CHECK_TYPE([psm_epconn_t],
		                    [],
				    [psm_happy=0],
				    [[#include <psm.h>]])])
	      ])

	AS_IF([test $psm_happy -eq 1], [$1], [$2])
])
