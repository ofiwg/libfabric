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
])
AM_CONDITIONAL([BUILD_EFA_RDMA_CHECKER], [test $efa_rdma_checker_happy -eq 1])
