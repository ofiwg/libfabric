dnl Configury specific to the libfabric linkx provider

dnl Called to configure this provider
dnl
dnl Arguments:
dnl
dnl $1: action if configured successfully
dnl $2: action if not configured successfully
dnl
AC_DEFUN([FI_LINKX_CONFIGURE],[
       # Determine if we can support the linkx provider
       linkx_happy=0
       AS_IF([test x"$enable_linkx" != x"no"], [linkx_happy=1])
       AS_IF([test $linkx_happy -eq 1], [$1], [$2])
])
