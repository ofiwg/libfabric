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

dnl EFA GDA support
AC_ARG_ENABLE([efagda],
	[AS_HELP_STRING([--enable-efagda],
		[Enable EFA GDA testing])],
	[],
	[enable_efagda=no])

AS_IF([test "x$enable_efagda" = xyes], [
	CPPFLAGS="-I${srcdir}/prov/efa/src/efagda $CPPFLAGS"
	LDFLAGS="-L${srcdir}/prov/efa/src/efagda -Wl,-rpath=${srcdir}/prov/efa/src/efagda $LDFLAGS"
	AC_DEFINE([HAVE_EFAGDA], [1], [Enable EFA GDA testing])
	AC_CHECK_DECL(EFADV_DEVICE_ATTR_CAPS_CQ_WITH_EXT_MEM_DMABUF, [], [],
		[[#include <infiniband/efadv.h>]])
	AC_CHECK_DECL(efadv_query_qp_wqs, [], [],
		[[#include <infiniband/efadv.h>]])
	AC_CHECK_DECL(efadv_query_cq, [], [],
		[[#include <infiniband/efadv.h>]])
	])

AM_CONDITIONAL([EFAGDA], [test x$enable_efagda = xyes])
