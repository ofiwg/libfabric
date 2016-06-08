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
	AS_IF([test x"$enable_verbs" != x"no"],
	      [FI_CHECK_PACKAGE([verbs_ibverbs],
				[infiniband/verbs.h],
				[ibverbs],
				[ibv_open_device],
				[],
				[$verbs_PREFIX],
				[$verbs_LIBDIR],
				[
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
					verbs_ibverbs_happy=0])
				rm -f $file conftemp],
				[verbs_ibverbs_happy=0])

	       FI_CHECK_PACKAGE([verbs_rdmacm],
				[rdma/rsocket.h],
				[rdmacm],
				[rdma_create_qp],
				[],
				[$verbs_PREFIX],
				[$verbs_LIBDIR],
				[verbs_rdmacm_happy=1],
				[verbs_rdmacm_happy=0])
	      ])

	AS_IF([test $verbs_ibverbs_happy -eq 1 && \
	       test $verbs_rdmacm_happy -eq 1], [$1], [$2])

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
