/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa.h"
#include "efa_av.h"
#include "efa_mr.h"
#include "efa_device.h"
#include "rdm/efa_rdm_ep.h"

void efa_test_fabricate_addr(struct fid_ep *ep, struct efa_ep_addr *addr)
{
	size_t addr_len = sizeof(*addr);
	static uint8_t gid_suffix = 1;

	memset(addr, 0, addr_len);
	// Get ep's actual address
	fi_getname(&ep->fid, addr, &addr_len);
	// Flip the first bit
	addr->raw[0] ^= 0xFF;
	addr->raw[15] = gid_suffix++;
	addr->qpn = 1;
	addr->qkey = 0x5678;
}

int efa_test_explicit_av_insert(struct fid_ep *ep, struct fid_av *av,
				fi_addr_t *addr)
{
	struct efa_ep_addr raw_addr;

	efa_test_fabricate_addr(ep, &raw_addr);
	return fi_av_insert(av, &raw_addr, 1, addr, 0, NULL);
}

fi_addr_t efa_test_insert_peer_new_gid(struct fid_ep *ep, struct fid_av *av)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_av *efa_av;
	struct efa_ep_addr raw_addr;
	fi_addr_t fi_addr = FI_ADDR_NOTAVAIL;
	int err;

	efa_rdm_ep =
		container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	efa_av = container_of(av, struct efa_av, util_av.av_fid);

	efa_test_fabricate_addr(ep, &raw_addr);

	ofi_genlock_lock(&efa_rdm_ep_rdm_domain(efa_rdm_ep)->srx_lock);
	err = efa_av_insert_one(efa_av, &raw_addr, &fi_addr, 0, NULL, true,
				true);
	ofi_genlock_unlock(&efa_rdm_ep_rdm_domain(efa_rdm_ep)->srx_lock);

	if (err)
		return FI_ADDR_NOTAVAIL;

	return fi_addr;
}

void efa_test_set_inject_rma_size(struct fid_ep *ep, size_t size)
{
	struct efa_base_ep *base_ep =
		container_of(ep, struct efa_base_ep, util_ep.ep_fid);

	base_ep->inject_rma_size = size;
}

int efa_test_set_track_mr(int value)
{
	int prev = efa_env.track_mr;

	efa_env.track_mr = value;
	return prev;
}

void efa_test_mr_set_iface_cuda(struct fid_mr *mr)
{
	struct efa_mr *efa_mr = container_of(mr, struct efa_mr, mr_fid);

	efa_mr->iface = FI_HMEM_CUDA;
}

fi_addr_t efa_test_rma_insert_peer(struct fid_ep *ep, struct fid_av *av)
{
	struct efa_ep_addr raw_addr;
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t fi_addr = FI_ADDR_NOTAVAIL;
	int ret;

	/* Reuse the EP's own GID; fi_av_insert's real ibv_create_ah rejects a
	 * fabricated one. qpn/qkey are not validated, so any value works. */
	ret = fi_getname(&ep->fid, &raw_addr, &raw_addr_len);
	if (ret)
		return FI_ADDR_NOTAVAIL;
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;

	ret = fi_av_insert(av, &raw_addr, 1, &fi_addr, 0, NULL);
	if (ret != 1)
		return FI_ADDR_NOTAVAIL;

	return fi_addr;
}

bool efa_test_device_supports_rma(void)
{
	if (g_efa_selected_device_cnt <= 0)
		return false;

	return efa_device_support_rdma_read() &&
	       efa_device_support_rdma_write();
}

void efa_test_get_zero_byte_bounce_buf(struct fid_ep *ep, uint64_t *addr,
				       uint32_t *lkey)
{
	struct efa_base_ep *base_ep =
		container_of(ep, struct efa_base_ep, util_ep.ep_fid);
	struct efa_domain *domain = base_ep->domain;

	*addr = (uint64_t) domain->zero_byte_bounce_buf;
	*lkey = domain->zero_byte_bounce_buf_mr->lkey;
}

struct ibv_ah *efa_test_implicit_addr_to_ibv_ah(struct fid_av *av,
						fi_addr_t fi_addr)
{
	struct efa_av *efa_av =
		container_of(av, struct efa_av, util_av.av_fid);
	struct efa_conn *conn = efa_av_addr_to_conn_implicit(efa_av, fi_addr);

	return conn ? conn->ah->ibv_ah : NULL;
}
