/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_gtest_domain_utils.h"

#if HAVE_EFADV_QUERY_QP_WQS
int efa_test_mock_efadv_query_qp_wqs(struct ibv_qp *ibvqp,
				     struct efadv_wq_attr *sq_attr,
				     struct efadv_wq_attr *rq_attr,
				     uint32_t inlen)
{
	sq_attr->buffer = (uint8_t *) 0x12345678;
	sq_attr->doorbell = (uint32_t *) 0x87654321;
	sq_attr->entry_size = 64;
	sq_attr->num_entries = 128;
	sq_attr->max_batch = 16;
#if HAVE_EFADV_WQ_ATTR_CAPS
	sq_attr->caps |= EFADV_WQ_CAPS_64_BIT_REQ_ID;
#endif

	rq_attr->buffer = (uint8_t *) 0x12345678;
	rq_attr->doorbell = (uint32_t *) 0x87654321;
	rq_attr->entry_size = 64;
	rq_attr->num_entries = 128;
	rq_attr->max_batch = 16;

	return 0;
}
#endif /* HAVE_EFADV_QUERY_QP_WQS */

uint16_t efa_test_mock_efadv_sq_caps(void)
{
#if HAVE_EFADV_WQ_ATTR_CAPS
	return FI_EFA_WQ_CAPS_64_BIT_REQ_ID;
#else
	return 0;
#endif
}

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
