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
	psm_orig_LIBS=$psm_LIBS
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
		     [AC_MSG_CHECKING([if PSM version is 1.x])
		      AC_RUN_IFELSE([AC_LANG_SOURCE(
						    [[
						    #include <psm.h>
						    int main()
						    {
							return PSM_VERNO_MAJOR < 2 ? 0 : 1;
						    }
						    ]]
						   )],
				    [AC_MSG_RESULT([yes])],
				    [AC_MSG_RESULT([no]); psm_happy=0])
		     ])
	       AS_IF([test $psm_happy -eq 1],
		     [AC_CHECK_TYPE([psm_epconn_t],
		                    [],
				    [psm_happy=0],
				    [[#include <psm.h>]])])
	      ])

	AS_IF([test $psm_happy -eq 1], [$1], [$2])

	AC_MSG_CHECKING([if use dlopen to load PSM library])
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
			[AC_MSG_RESULT([yes]); psm_LIBS=$psm_orig_LIBS];
			[AC_MSG_RESULT([no]); psm_LDFLAGS="$LDFLAGS $psm_LDFLAGS"])

	psm_CPPFLAGS="$CPPFLAGS $psm_CPPFLAGS"
	CPPFLAGS="$psm_orig_CPPFLAGS"
	LDFLAGS="$psm_orig_LDFLAGS"
])
