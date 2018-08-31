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
	 psm2_ARCH=`uname -m | sed -e 's,\(i[456]86\|athlon$$\),i386,'`
	 AM_CONDITIONAL([HAVE_PSM2_X86_64], [test x$psm2_ARCH = xx86_64])
	 AC_SUBST([HAVE_PSM2_X86_64])
	 AC_SUBST([psm2_ARCH])
	 AS_IF([test x$with_psm2_src = x], [have_psm2_src=0], [have_psm2_src=1])
	 AM_CONDITIONAL([HAVE_PSM2_SRC], [test x$have_psm2_src = x1])
	 AC_DEFINE_UNQUOTED([HAVE_PSM2_SRC], $have_psm2_src, [PSM2 source is built-in])
	 psm2_happy=0
	 have_psm2_am_register_handlers_2=1
	 AS_IF([test x"$enable_psm2" != x"no"],
	       [AS_IF([test x$have_psm2_src = x0],
		      [
			dnl build with stand-alone PSM2 library
			FI_CHECK_PACKAGE([psm2],
					 [psm2.h],
					 [psm2],
					 [psm2_am_register_handlers_2],
					 [],
					 [$psm2_PREFIX],
					 [$psm2_LIBDIR],
					 [psm2_happy=1],
					 [psm2_happy=0])
			AS_IF([test x$psm2_happy = x0],
			      [
				$as_echo "$as_me: recheck psm2 with reduced feature set."
				have_psm2_am_register_handlers_2=0
				FI_CHECK_PACKAGE([psm2],
						 [psm2.h],
						 [psm2],
						 [psm2_ep_epid_lookup2],
						 [],
						 [$psm2_PREFIX],
						 [$psm2_LIBDIR],
						 [psm2_happy=1],
						 [psm2_happy=0])
			      ])
		      ],
		      [
			dnl build with PSM2 source code included
			psm2_CPPFLAGS="-msse4.2"
			psm2_happy=1
			AC_COMPILE_IFELSE([AC_LANG_PROGRAM(
						[[#include <rdma/hfi/hfi1_user.h>]],
						[[
							#if HFI1_USER_SWMAJOR != 6
							#error "incorrect version of hfi1_user.h"
							#endif
						]])],
				          [],
				          [
						$as_echo "$as_me: hfi1_user.h version 6.x not found"
						psm2_happy=0
					  ])
			AS_IF([test x$psm2_happy = x1],
			      [FI_CHECK_PACKAGE([psm2],
						[numa.h],
						[numa],
						[numa_node_of_cpu],
						[],
						[$with_numa],
						[],
						[],
						[
							$as_echo "$as_me: numactl-devel or libnuma-devel not found"
							psm2_happy=0
						])])
			AS_IF([test x$psm2_happy = x1],
			      AS_IF([test -f $with_psm2_src/libpsm2.spec.in],
				    [AS_IF([grep -q psm2_am_register_handlers_2 $with_psm2_src/psm2_am.h],
					   [
						$as_echo "$as_me: creating links for PSM2 source code."
						mkdir -p $srcdir/prov/psm2/src/psm2
						cp -srf $with_psm2_src/* $srcdir/prov/psm2/src/psm2/
						ln -sf ../include/rbtree.h $srcdir/prov/psm2/src/psm2/ptl_ips/
						ln -sf ../include/rbtree.h $srcdir/prov/psm2/src/psm2/ptl_am/
					   ],
					   [
						$as_echo "$as_me: PSM2 source under <$with_psm2_src> is too old."
						psm2_happy=0
					   ])
				   ],
				   [
					$as_echo "$as_me: no PSM2 source under <$with_psm2_src>."
					psm2_happy=0
				   ]))
		      ])
	       ])
	 AS_IF([test $psm2_happy -eq 1], [$1], [$2])
	 AC_DEFINE_UNQUOTED([HAVE_PSM2_AM_REGISTER_HANDLERS_2],
			    $have_psm2_am_register_handlers_2,
			    [psm2_am_register_handlers_2 function is present])
])

AC_ARG_WITH([psm2-src],
	    AC_HELP_STRING([--with-psm2-src=DIR],
                           [Provide path to the source code of PSM2 library
			    to be compiled into the provider]))

AC_ARG_WITH([numa],
	    AC_HELP_STRING([--with-numa=DIR],
                           [Provide path to where the numactl-devel or libnuma-devel
			    package is installed]))

