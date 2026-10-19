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
#if HAVE_EFADV_COMP_ACTION
	sq_attr->wqe_blocks |= EFADV_WQE_BLOCK_COMP_ACTION_WITH_DATA;
	sq_attr->comp_action_with_data_block_offset =
		EFA_TEST_MOCK_ACTION_BLOCK_OFFSET;
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

uint16_t efa_test_mock_efadv_sq_wqe_blocks(void)
{
#if HAVE_EFADV_COMP_ACTION
	return FI_EFA_WQE_BLOCK_COMP_ACTION_WITH_DATA;
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

bool efa_test_comp_action_wr_matches(const struct efa_comp_action_wr *sig,
				     uint64_t feature_bits, uint32_t local_id,
				     uint64_t local_data, uint32_t remote_id,
				     uint64_t remote_data)
{
	return sig && sig->feature_bits == feature_bits &&
	       sig->local_action_id == local_id &&
	       sig->local_action_data == local_data &&
	       sig->remote_action_id == remote_id &&
	       sig->remote_action_data == remote_data;
}

bool efa_test_get_comp_action_enabled(struct fid_ep *ep)
{
	struct efa_base_ep *base_ep =
		container_of(ep, struct efa_base_ep, util_ep.ep_fid);

	return base_ep->comp_action_enabled;
}

bool efa_test_device_supports_comp_action(void)
{
	return efa_device_support_comp_action();
}

uint32_t efa_test_set_comp_action_supp_ops(struct fid_domain *domain_fid,
					   bool write, bool remote_write,
					   bool remote_write_imm)
{
#if HAVE_EFADV_COMP_ACTION
	struct efa_domain *domain = container_of(domain_fid, struct efa_domain,
						 util_domain.domain_fid);
	uint32_t prev = domain->device->efa_attr.comp_action_supp_ops;
	uint32_t ops = 0;

	if (write)
		ops |= EFADV_DEVICE_ATTR_COMP_ACTION_SUPPORTED_OP_RDMA_WRITE;
	if (remote_write)
		ops |= EFADV_DEVICE_ATTR_COMP_ACTION_SUPPORTED_OP_REMOTE_RDMA_WRITE;
	if (remote_write_imm)
		ops |= EFADV_DEVICE_ATTR_COMP_ACTION_SUPPORTED_OP_REMOTE_RDMA_WRITE_IMM;
	domain->device->efa_attr.comp_action_supp_ops = ops;
	return prev;
#else
	return 0;
#endif
}

void efa_test_restore_comp_action_supp_ops(struct fid_domain *domain_fid,
					   uint32_t supp_ops)
{
#if HAVE_EFADV_COMP_ACTION
	struct efa_domain *domain = container_of(domain_fid, struct efa_domain,
						 util_domain.domain_fid);

	domain->device->efa_attr.comp_action_supp_ops = supp_ops;
#endif
}

bool efa_test_build_has_comp_action(void)
{
#if HAVE_EFADV_COMP_ACTION
	return true;
#else
	return false;
#endif
}
