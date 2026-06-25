/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_av.h"
#include "efa_cq.h"
#include "efa_direct_ope.h"
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

struct efa_ibv_cq *efa_test_get_ibv_cq(struct fid_cq *cq_fid)
{
	struct efa_cq *efa_cq;

	efa_cq = container_of(cq_fid, struct efa_cq, util_cq.cq_fid);
	return &efa_cq->ibv_cq;
}

uint32_t efa_test_get_qp_num(struct fid_ep *ep)
{
	struct efa_rdm_ep *efa_rdm_ep;

	efa_rdm_ep =
		container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	return efa_rdm_ep->base_ep.qp->qp_num;
}

void efa_test_alloc_err_buf(struct efa_ibv_cq *ibv_cq)
{
	struct efa_cq *efa_cq;

	efa_cq = container_of(ibv_cq, struct efa_cq, ibv_cq);
	if (!efa_cq->err_buf)
		efa_cq->err_buf = malloc(EFA_ERROR_MSG_BUFFER_LENGTH);
}

void efa_test_free_err_buf(struct efa_ibv_cq *ibv_cq)
{
	struct efa_cq *efa_cq;

	efa_cq = container_of(ibv_cq, struct efa_cq, ibv_cq);
	free(efa_cq->err_buf);
	efa_cq->err_buf = NULL;
}

void efa_test_set_ibv_cq_ex(struct efa_ibv_cq *ibv_cq, int status,
			    uint64_t wr_id)
{
	ibv_cq->ibv_cq_ex->status = status;
	ibv_cq->ibv_cq_ex->wr_id = wr_id;
}

void *efa_test_alloc_direct_ope(struct efa_context *ctx)
{
	struct efa_direct_ope *direct_ope;

	direct_ope = calloc(1, sizeof(*direct_ope));
	if (!direct_ope)
		return NULL;
	direct_ope->context = ctx;
	return direct_ope;
}

void efa_test_free_direct_ope(void *direct_ope)
{
	free(direct_ope);
}

int efa_test_get_track_mr(void)
{
	return efa_env.track_mr;
}

void efa_test_set_track_mr(int val)
{
	efa_env.track_mr = val;
}

ssize_t efa_test_cq_read_staged_data_entry(struct fid_cq *cq_fid,
					   struct fi_cq_data_entry *entry)
{
	struct efa_cq *efa_cq =
		container_of(cq_fid, struct efa_cq, util_cq.cq_fid);

	/* ofi_cq_read_entries drains staged completions without calling
	 * cq->progress, so it does not re-enter the mocked poll path. */
	return ofi_cq_read_entries(&efa_cq->util_cq, entry, 1, NULL);
}

struct ibv_ah *efa_test_implicit_addr_to_ibv_ah(struct fid_av *av,
						fi_addr_t fi_addr)
{
	struct efa_av *efa_av =
		container_of(av, struct efa_av, util_av.av_fid);
	struct efa_conn *conn = efa_av_addr_to_conn_implicit(efa_av, fi_addr);

	return conn ? conn->ah->ibv_ah : NULL;
}
