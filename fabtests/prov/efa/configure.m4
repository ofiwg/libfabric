dnl
dnl SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
dnl SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved.
dnl
dnl Configure specific to the fabtests Amazon EFA provider

dnl The efa fabtests are only usable, and only buildable, when libfabric was
dnl built with the efa provider: the efa extension header is installed with it.
AS_IF([test x"$enable_efa" = x"yes"],
      [AC_CHECK_HEADER([rdma/fi_ext_efa.h], [],
              [enable_efa=no
               AC_MSG_NOTICE([EFA fabtests are disabled: <rdma/fi_ext_efa.h> dnl
not found, libfabric was built without the efa provider])])])

AM_CONDITIONAL([ENABLE_EFA], [test x"$enable_efa" = x"yes"])


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

dnl Oldest efa-dp-direct the GDA tests build against. CI reads this line to
dnl pick the efa-dp-direct version to build, so keep it a plain assignment.
efa_dp_direct_min_version=1.1
efa_dp_direct_min_major=${efa_dp_direct_min_version%%.*}
efa_dp_direct_min_minor=${efa_dp_direct_min_version#*.}

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
	dnl The tests use the efa-dp-direct 1.x API from 1.1: the host, common
	dnl and device directories, the version-dispatched host API and the
	dnl sq_caps attribute. Older trees and other major versions are not
	dnl supported.
	AS_IF([test -f "${efa_dp_direct_path}/CUDA/src/efa_cuda_dp.h"], [
		AC_MSG_ERROR([efa-dp-direct at ${efa_dp_direct_path} predates 1.0 (it has CUDA/src). EFA GDA tests need efa-dp-direct ${efa_dp_direct_min_major}.x, ${efa_dp_direct_min_version} or later.])
	])
	CPPFLAGS="-I${srcdir}/prov/efa/src/efagda -I${efa_dp_direct_path}/CUDA/host -I${efa_dp_direct_path}/CUDA/common $CPPFLAGS"
	LDFLAGS="-L${efa_dp_direct_path}/CUDA/build -Wl,-rpath=${efa_dp_direct_path}/CUDA/build $LDFLAGS"
	AC_CHECK_HEADERS([efa_cuda_dp.h], [],
		[AC_MSG_ERROR([Cannot find efa_cuda_dp.h. Check --with-efa-dp-direct path.])])
	AC_MSG_CHECKING([for efa-dp-direct ${efa_dp_direct_min_major}.x, ${efa_dp_direct_min_version} or later])
	AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <efa_cuda_dp_version.h>]],
		[[#if EFA_CUDA_DP_VERSION_MAJOR != $efa_dp_direct_min_major || EFA_CUDA_DP_VERSION_MINOR < $efa_dp_direct_min_minor
		  #error unsupported efa-dp-direct version
		  #endif]])],
		[AC_MSG_RESULT([yes])],
		[AC_MSG_RESULT([no])
		 AC_MSG_ERROR([EFA GDA tests need efa-dp-direct ${efa_dp_direct_min_major}.x, ${efa_dp_direct_min_version} or later. Check --with-efa-dp-direct path.])])
	AC_CHECK_LIB([efacudadp], [efa_cuda_host_context_create], [],
		[AC_MSG_ERROR([Cannot find libefacudadp. Build it with "make -C <efa-dp-direct>/CUDA" and check --with-efa-dp-direct path.])])
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
