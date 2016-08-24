dnl Configury specific to the libfabrics mlx provider

dnl Called to configure this provider
dnl
dnl Arguments:
dnl
dnl $1: action if configured successfully
dnl $2: action if not configured successfully
dnl
AC_DEFUN([FI_MLX_CONFIGURE],[
    # Determine if we can support the mxm provider
    mlx_happy=0
    AS_IF([test x"$enable_mlx" != x"no"],
              [FI_CHECK_PACKAGE([mlx],
                    [ucp/api/ucp.h],
                    [ucp],
                    [ucp_get_version_string],
                    [],
                    [$mlx_PREFIX],
                    [$mlx_LIBDIR],
                    [mlx_happy=1],
                    [mlx_happy=0])
         ])
    AS_IF([test $mlx_happy -eq 1], [$1], [$2])
])

