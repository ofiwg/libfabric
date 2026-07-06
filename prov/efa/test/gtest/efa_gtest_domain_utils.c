/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_gtest_domain_utils.h"

uint32_t efa_test_get_qp_qkey(struct fid_ep *ep)
{
	struct efa_base_ep *base_ep =
		container_of(ep, struct efa_base_ep, util_ep.ep_fid);

	return base_ep->qp->qkey;
}

int efa_test_getname_qkey(struct fid_ep *ep, uint32_t *qkey)
{
	struct efa_ep_addr addr = {0};
	size_t addrlen = sizeof(addr);
	int ret;

	ret = fi_getname(&ep->fid, &addr, &addrlen);
	if (!ret)
		*qkey = addr.qkey;

	return ret;
}
