dnl
dnl SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
dnl SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved.
dnl
dnl Configure specific to the fabtests Amazon EFA provider


dnl Checks for presence of efadv verbs. Needed for building tests that calls efadv verbs.
have_efadv=0
AC_CHECK_HEADER([infiniband/efadv.h],
		[AC_CHECK_LIB(efa, efadv_query_device,
                          [have_efadv=1])])
AM_CONDITIONAL([HAVE_EFA_DV], [test $have_efadv -eq 1])

efa_rdma_checker_happy=0
AS_IF([test x"$have_efadv" = x"1"], [
        efa_rdma_checker_happy=1
        AC_CHECK_MEMBER(struct efadv_device_attr.max_rdma_size,
            [],
            [efa_rdma_checker_happy=0],
            [[#include <infiniband/efadv.h>]])

        AC_CHECK_MEMBER(struct efadv_device_attr.device_caps,
            [],
            [efa_rdma_checker_happy=0],
            [[#include <infiniband/efadv.h>]])

        AC_CHECK_DECL(EFADV_DEVICE_ATTR_CAPS_RDMA_WRITE,
            [],
            [efa_rdma_checker_happy=0],
            [[#include <infiniband/efadv.h>]])

        AC_CHECK_DECL(EFADV_DEVICE_ATTR_CAPS_UNSOLICITED_WRITE_RECV,
            [],
            [efa_rdma_checker_happy=0],
            [[#include <infiniband/efadv.h>]])
])
AM_CONDITIONAL([BUILD_EFA_RDMA_CHECKER], [test $efa_rdma_checker_happy -eq 1])

have_efadv_create_comp_cntr=0
AS_IF([test x"$have_efadv" = x"1"], [
	AC_CHECK_DECL([efadv_create_comp_cntr],
		[have_efadv_create_comp_cntr=1],
		[have_efadv_create_comp_cntr=0],
		[[#include <infiniband/efadv.h>]])
])
AM_CONDITIONAL([HAVE_EFADV_CREATE_COMP_CNTR], [test $have_efadv_create_comp_cntr -eq 1])

dnl Check for FI_EFA_WR_HIGH_PPS in fi_ext_efa.h (needed for efa_rma_bw).
have_fi_efa_wr_high_pps=0
AC_CHECK_DECL([FI_EFA_WR_HIGH_PPS],
	[have_fi_efa_wr_high_pps=1],
	[have_fi_efa_wr_high_pps=0],
	[[#include <rdma/fi_ext_efa.h>]])
AM_CONDITIONAL([HAVE_FI_EFA_WR_HIGH_PPS], [test $have_fi_efa_wr_high_pps -eq 1])

dnl EFA GDA support
AC_ARG_ENABLE([efagda],
	[AS_HELP_STRING([--enable-efagda],
		[Enable EFA GDA testing])],
	[],
	[enable_efagda=no])

AC_ARG_WITH([efa-dp-direct],
	[AS_HELP_STRING([--with-efa-dp-direct=DIR],
		[Path to efa-dp-direct library source])],
	[efa_dp_direct_path=$withval],
	[efa_dp_direct_path=""])

AS_IF([test "x$enable_efagda" = xyes], [
	AS_IF([test -z "$efa_dp_direct_path"], [
		AC_MSG_ERROR([--with-efa-dp-direct=DIR is required when --enable-efagda is used])
	])
	AS_IF([test -n "$with_cuda" && test "$with_cuda" != "yes" && test "$with_cuda" != "no"], [
		CPPFLAGS="-I$with_cuda/include $CPPFLAGS"
	])
	CPPFLAGS="-I${srcdir}/prov/efa/src/efagda -I${efa_dp_direct_path}/CUDA/src $CPPFLAGS"
	LDFLAGS="-L${efa_dp_direct_path}/CUDA/build -Wl,-rpath=${efa_dp_direct_path}/CUDA/build $LDFLAGS"
	AC_CHECK_HEADERS([efa_cuda_dp.h], [],
		[AC_MSG_ERROR([Cannot find efa_cuda_dp.h. Check --with-efa-dp-direct path.])])
	AC_CHECK_LIB([efacudadp], [efa_cuda_create_cq], [],
		[AC_MSG_ERROR([Cannot find libefacudadp. Check --with-efa-dp-direct path.])])
	AC_DEFINE([HAVE_EFAGDA], [1], [Enable EFA GDA testing])
	AC_CHECK_DECL(EFADV_DEVICE_ATTR_CAPS_CQ_WITH_EXT_MEM_DMABUF, [], [],
		[[#include <infiniband/efadv.h>]])
	AC_CHECK_DECL(efadv_query_qp_wqs, [], [],
		[[#include <infiniband/efadv.h>]])
	AC_CHECK_DECL(efadv_query_cq, [], [],
		[[#include <infiniband/efadv.h>]])
	])

AM_CONDITIONAL([EFAGDA], [test x$enable_efagda = xyes])
AC_SUBST([EFA_DP_DIRECT], [$efa_dp_direct_path])
