dnl Configury specific to the libfabric verbs provider

dnl Called to configure this provider
dnl
dnl Arguments:
dnl
dnl $1: action if configured successfully
dnl $2: action if not configured successfully
dnl
AC_DEFUN([FI_VERBS_CONFIGURE],[
	# Determine if we can support the verbs provider
	verbs_ibverbs_happy=0
	verbs_rdmacm_happy=0
	verbs_ibverbs_exp_happy=0
	AS_IF([test x"$enable_verbs" != x"no"],
	      [FI_CHECK_PACKAGE([verbs_ibverbs],
				[infiniband/verbs.h],
				[ibverbs],
				[ibv_open_device],
				[],
				[$verbs_PREFIX],
				[$verbs_LIBDIR],
				[FI_VERBS_DOUBLE_CHECK_LIBIBVERBS],
				[verbs_ibverbs_happy=0])

	      FI_CHECK_PACKAGE([verbs_ibverbs],
				[infiniband/verbs_exp.h],
				[ibverbs],
				[ibv_open_device],
				[],
				[$verbs_PREFIX],
				[$verbs_LIBDIR],
				[verbs_ibverbs_exp_happy=1],
				[verbs_ibverbs_exp_happy=0])

	       FI_CHECK_PACKAGE([verbs_rdmacm],
				[rdma/rdma_cma.h],
				[rdmacm],
				[rdma_create_qp],
				[],
				[$verbs_PREFIX],
				[$verbs_LIBDIR],
				[verbs_rdmacm_happy=1],
				[verbs_rdmacm_happy=0])
	      ])

	AC_COMPILE_IFELSE([AC_LANG_PROGRAM(
			   [
			    #include <infiniband/verbs_exp.h>
			   ],
			   [
			    return (IBV_EXP_DEVICE_ATTR_ODP | IBV_EXP_DEVICE_ODP);
			   ])
			  ],
			  [verbs_ibverbs_exp_happy=1],
			  [verbs_ibverbs_exp_happy=0])

	AS_IF([test $verbs_ibverbs_happy -eq 1 && \
	       test $verbs_rdmacm_happy -eq 1], [$1], [$2])

	AS_IF([test $verbs_ibverbs_happy -eq 1 && \
	       test $verbs_rdmacm_happy -eq 1 && \
	       test $verbs_ibverbs_exp_happy -eq 1],
		[AC_DEFINE([HAVE_VERBS_EXP_H], [1],
			   [Experimental verbs features support])],
		[])

	# Technically, verbs_ibverbs_CPPFLAGS and
	# verbs_rdmacm_CPPFLAGS could be different, but it is highly
	# unlikely that they ever will be.  So only list
	# verbs_ibverbs_CPPFLAGS here.  Same with verbs_*_LDFLAGS,
	# below.
	verbs_CPPFLAGS=$verbs_ibverbs_CPPFLAGS
	verbs_LDFLAGS=$verbs_ibverbs_LDFLAGS
	verbs_LIBS="$verbs_rdmacm_LIBS $verbs_ibverbs_LIBS"
	AC_SUBST(verbs_CPPFLAGS)
	AC_SUBST(verbs_LDFLAGS)
	AC_SUBST(verbs_LIBS)
])

dnl
dnl Per https://github.com/ofiwg/libfabric/issues/2070, it is possible
dnl that the AC_CHECK_LIB test for libibverbs is not sufficient --
dnl i.e., AC_CHECK_LIB may succeed, but then linking with libtool may
dnl fail.  This test therefore double checks that we can successfully
dnl use libtool to link against libibverbs.  NOTE: this test is
dnl contingent upon LT_OUTPUT having already been invoked (i.e., so that
dnl the libtool script exists).
dnl
AC_DEFUN([FI_VERBS_DOUBLE_CHECK_LIBIBVERBS],[
	AC_MSG_CHECKING(if libibverbs is linkable by libtool)
	file=conftemp.$$.c
	rm -f $file conftemp
	cat > $file <<-EOF
char ibv_open_device ();
int main ()
{ return ibv_open_device (); }
EOF

	cmd="./libtool --mode=link --tag=CC $CC $CPPFLAGS $CFLAGS $file -o conftemp $LDFLAGS -libverbs"
	echo "configure:$LINENO: $cmd" >> config.log 2>&1
	eval $cmd >> config.log 2>&1
	status=$?
	AS_IF([test $status -eq 0 && test -x conftemp],
		[AC_MSG_RESULT(yes)
		verbs_ibverbs_happy=1],
		[AC_MSG_RESULT(no)
		echo "configure: failed program was" >> config.log
		cat $file >> config.log
		verbs_ibverbs_happy=0])

	rm -f $file conftemp
])
