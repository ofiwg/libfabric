dnl Configury specific to the libfabrics mxm provider

dnl Called to configure this provider
dnl
dnl Arguments:
dnl
dnl $1: action if configured successfully
dnl $2: action if not configured successfully
dnl
AC_DEFUN([FI_MXM_CONFIGURE],[
    # Determine if we can support the mxm provider
    mxm_happy=0
    AS_IF([test x"$enable_mxm" != x"no"],
              [FI_CHECK_PACKAGE([mxm],
                    [mxm/api/mxm_api.h],
                    [mxm],
                    [mxm_get_version],
                    [],
                    [$mxm_PREFIX],
                    [$mxm_LIBDIR],
                    [mxm_happy=1],
                    [mxm_happy=0])
         ])
    AS_IF([test $mxm_happy -eq 1], [$1], [$2])
])

