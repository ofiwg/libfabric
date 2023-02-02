dnl Configury specific to the libfabric shm provider

dnl Called to configure this provider
dnl
dnl Arguments:
dnl
dnl $1: action if configured successfully
dnl $2: action if not configured successfully
dnl
AC_DEFUN([FI_SM2_CONFIGURE],[
	# Determine if we can support the shm provider
	sm2_happy=0
	cma_happy=0
	dsa_happy=0
	AS_IF([test x"$enable_shm" != x"no"],
	      [
	       # check if CMA support are present
	       AS_IF([test x$linux = x1 && test x$host_cpu = xx86_64],
		     [cma_happy=1],
		     [AC_CHECK_FUNC([process_vm_readv],
				    [cma_happy=1],
				    [cma_happy=0])]
	       )

	       # check if SHM support are present
	       AC_CHECK_FUNC([shm_open],
			     [sm2_happy=1],
			     [sm2_happy=0])

	       # look for shm_open in librt if not already present
	       AS_IF([test $sm2_happy -eq 0],
		     [FI_CHECK_PACKAGE([rt],
				[sys/mman.h],
				[rt],
				[shm_open],
				[],
				[],
				[],
				[sm2_happy=1],
				[sm2_happy=0])])
	       sm2_LIBS="$rt_LIBS"

	       AC_ARG_WITH([dsa],
			   [AS_HELP_STRING([--with-dsa=DIR],
					   [Enable DSA build and fail if not found.
					    Optional=<Path to where the DSA libraries
					    and headers are installed.>])])

	       AS_IF([test "x$with_dsa" != "xno"],
		     [FI_CHECK_PACKAGE([dsa],
				       [accel-config/libaccel_config.h],
				       [accel-config],
				       [accfg_new],
				       [],
				       [$with_dsa],
				       [],
				       [dsa_happy=1])])

	       AS_IF([test $dsa_happy -eq 1],
		     [FI_CHECK_PACKAGE([numa],
				       [numa.h],
		                       [numa],
		                       [numa_node_of_cpu],
		                       [],
		                       [],
		                       [],
		                       [],
		                       [dsa_happy=0])])

	      AS_IF([test $dsa_happy -eq 1],
		    [AC_CHECK_HEADER(linux/idxd.h, [], [dsa_happy=0])])

	      AS_IF([test "x$with_dsa" != "xno" && test -n "$with_dsa" && test $dsa_happy -eq 0 ],
		    [AC_MSG_ERROR([shm DSA support requested but DSA not available.])])

	      AS_IF([test $dsa_happy -eq 1 && test "x$with_dsa" != "xyes"],
	            [sm2_CPPFLAGS="$sm2_CPPFLAGS $dsa_CPPFLAGS $numa_LIBS"
		     sm2_LDFLAGS="$sm2_LDFLAGS $dsa_LDFLAGS $numa_LIBS"])
	      sm2_LIBS="$sm2_LIBS $dsa_LIBS $numa_LIBS"

	      AC_DEFINE_UNQUOTED([SM2_HAVE_DSA],[$dsa_happy],
				 [Whether DSA support is available])

	      AC_SUBST(sm2_CPPFLAGS)
	      AC_SUBST(sm2_LDFLAGS)
	      AC_SUBST(sm2_LIBS)
	      ])

	AS_IF([test $sm2_happy -eq 1 && \
	       test $cma_happy -eq 1], [$1], [$2])
])
