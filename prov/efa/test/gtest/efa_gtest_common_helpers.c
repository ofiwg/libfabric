/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_av.h"
#include "efa_cq.h"
#include "rdm/efa_rdm_ep.h"
#include "efa_gtest_common_helpers.h"

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

int efa_test_insert_self_peer(struct fid_ep *ep, struct fid_av *av,
			      fi_addr_t *addr)
{
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	int ret;

	ret = fi_getname(&ep->fid, &raw_addr, &raw_addr_len);
	if (ret)
		return ret;
	raw_addr.qpn = 0;
	raw_addr.qkey = 0x1234;

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

fi_addr_t efa_test_insert_self_gid_peer(struct fid_ep *ep, struct fid_av *av)
{
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	fi_addr_t peer_addr = FI_ADDR_NOTAVAIL;

	if (fi_getname(&ep->fid, &raw_addr, &raw_addr_len))
		return FI_ADDR_NOTAVAIL;
	raw_addr.qpn = 0;
	raw_addr.qkey = 0x1234;
	if (fi_av_insert(av, &raw_addr, 1, &peer_addr, 0, NULL) != 1)
		return FI_ADDR_NOTAVAIL;
	return peer_addr;
}

struct efa_context *efa_test_alloc_context(uint64_t completion_flags,
					   fi_addr_t addr)
{
	struct efa_context *ctx = calloc(1, sizeof(*ctx));

	if (!ctx)
		return NULL;
	ctx->completion_flags = completion_flags;
	ctx->addr = addr;
	return ctx;
}

struct efa_ibv_cq *efa_test_get_ibv_cq(struct fid_cq *cq_fid)
{
	struct efa_cq *efa_cq;

	efa_cq = container_of(cq_fid, struct efa_cq, util_cq.cq_fid);
	return &efa_cq->ibv_cq;
}

char *efa_test_get_cq_err_buf(struct fid_cq *cq_fid)
{
	struct efa_cq *efa_cq;

	efa_cq = container_of(cq_fid, struct efa_cq, util_cq.cq_fid);
	return efa_cq->err_buf;
}

const size_t efa_test_cq_err_buf_len = EFA_ERROR_MSG_BUFFER_LENGTH;

uint32_t efa_test_get_qp_num(struct fid_ep *ep)
{
	struct efa_base_ep *base_ep;

	base_ep = container_of(ep, struct efa_base_ep, util_ep.ep_fid);
	return base_ep->qp->qp_num;
}

void efa_test_set_ibv_cq_ex(struct efa_ibv_cq *ibv_cq, int status,
			    uint64_t wr_id)
{
	ibv_cq->ibv_cq_ex->status = status;
	ibv_cq->ibv_cq_ex->wr_id = wr_id;
}

int efa_test_set_track_mr(int value)
{
	int prev = efa_env.track_mr;

	efa_env.track_mr = value;
	return prev;
}

int efa_test_device_supports_rma(void)
{
	if (g_efa_selected_device_cnt <= 0)
		return 0;

	return efa_device_support_rdma_read() &&
	       efa_device_support_rdma_write();
}

size_t efa_test_ope_list_count(struct fid_ep *ep)
{
	struct efa_base_ep *base_ep =
		container_of(ep, struct efa_base_ep, util_ep.ep_fid);
	struct dlist_entry *item;
	size_t count = 0;

	dlist_foreach(&base_ep->ope_list, item)
		count++;

	return count;
}

struct ibv_ah *efa_test_implicit_addr_to_ibv_ah(struct fid_av *av,
						fi_addr_t fi_addr)
{
	struct efa_av *efa_av =
		container_of(av, struct efa_av, util_av.av_fid);
	struct efa_conn *conn = efa_av_addr_to_conn_implicit(efa_av, fi_addr);

	return conn ? conn->ah->ibv_ah : NULL;
}

static struct efa_rdm_domain *efa_test_rdm_domain(struct fid_domain *domain)
{
	struct efa_domain *efa_domain = container_of(
		domain, struct efa_domain, util_domain.domain_fid);

	return container_of(efa_domain, struct efa_rdm_domain, efa_domain);
}

struct fid_domain *efa_test_get_shm_domain(struct fid_domain *domain)
{
	return efa_test_rdm_domain(domain)->shm_domain;
}

void efa_test_set_shm_domain(struct fid_domain *domain,
			     struct fid_domain *shm_domain)
{
	efa_test_rdm_domain(domain)->shm_domain = shm_domain;
}

int efa_test_get_util_domain_ref(struct fid_domain *domain)
{
	struct efa_domain *efa_domain = container_of(
		domain, struct efa_domain, util_domain.domain_fid);

	return ofi_atomic_get32(&efa_domain->util_domain.ref);
}
