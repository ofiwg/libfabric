dnl Configury specific to the libfabrics BGQ provider

dnl Called to configure this provider
dnl
dnl Arguments:
dnl
dnl $1: action if configured successfully
dnl $2: action if not configured successfully
dnl
AC_DEFUN([FI_BGQ_CONFIGURE],[
	# Determine if we can support the bgq provider
	bgq_happy=0
	bgq_direct=0

	bgq_driver=/bgsys/drivers/ppcfloor
	AC_SUBST(bgq_driver)
	AC_ARG_WITH([bgq-driver],
		[AS_HELP_STRING([--with-bgq-driver=@<:@BGQ driver installation path@:>@],
			[Provide path to where BGQ system headers are installed])
		],
		[bgq_driver=$with_bgq_driver])

	bgq_driver_CPPFLAGS="-I$bgq_driver -I$bgq_driver/spi/include/kernel/cnk"
	CPPFLAGS="$bgq_driver_CPPFLAGS $CPPFLAGS"

	AS_IF([test x"$enable_bgq" != x"no"],
		[AC_CHECK_HEADER(hwi/include/bqc/MU_Descriptor.h,
			[bgq_happy=1],
			[bgq_happy=0])])

	bgq_external_source=auto
	AC_SUBST(bgq_external_source)
	AC_ARG_WITH([bgq-src],
		[AS_HELP_STRING([--with-bgq-src(=DIR)],
			[bgq opensource distribution @<:@default=auto@:>@])
		],
		[bgq_external_source=$with_bgq_src])

	AS_IF([test x"$bgq_external_source" == x"auto"],
		for bgq_dir in `ls -r /bgsys/source`; do
			AC_MSG_CHECKING([for bgq opensource distribution])
			AS_IF([test -f /bgsys/source/$bgq_dir/spi/src/kernel/cnk/memory_impl.c],
				bgq_external_source="/bgsys/source/$bgq_dir"
				AC_MSG_RESULT([$bgq_external_source])
				break)
		done
		AS_IF([test x"$bgq_external_source" == x"auto"],
			AC_MSG_RESULT([no]))
	)

	AS_IF([test ! -f $bgq_external_source/spi/src/kernel/cnk/memory_impl.c],
		AC_MSG_ERROR([unable to locate the bgq opensource distribution]))

	AS_IF([test x"$enable_direct" == x"bgq"],
		[AS_IF([test $bgq_happy -eq 1], [bgq_direct=1], 
			[AC_MSG_ERROR([cannot enable bgq direct])])])

	AC_CHECK_FUNC([shm_open],
		[],
		[FI_CHECK_PACKAGE([bgq],
			[sys/mman.h],
			[rt],
			[shm_open],
			[],
			[],
			[],
			[bgq_happy=1],
			[bgq_happy=0])])

	AC_ARG_WITH([bgq-progress],
		[AS_HELP_STRING([--with-bgq-progress(=auto|manual|runtime)],
			[Specify the FABRIC_DIRECT bgq progess mode  @<:@default=manual@:>@])
		])

	AS_CASE([$with_bgq_progress],
		[auto], [BGQ_FABRIC_DIRECT_PROGRESS=FI_PROGRESS_AUTO],
		[manual], [BGQ_FABRIC_DIRECT_PROGRESS=FI_PROGRESS_MANUAL],
		[runtime], [BGQ_FABRIC_DIRECT_PROGRESS=FI_PROGRESS_UNSPEC],
		[BGQ_FABRIC_DIRECT_PROGRESS=FI_PROGRESS_MANUAL])

	AC_SUBST(bgq_fabric_direct_progress, [$BGQ_FABRIC_DIRECT_PROGRESS])

	AC_CONFIG_FILES([prov/bgq/include/rdma/fi_direct.h])

	AS_IF([test $bgq_happy -eq 1], [$1], [$2])
])

dnl A separate macro for AM CONDITIONALS, since they cannot be invoked
dnl conditionally
AC_DEFUN([FI_BGQ_CONDITIONALS],[
])
