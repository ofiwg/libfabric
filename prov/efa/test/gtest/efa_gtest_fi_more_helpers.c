/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_fi_more_helpers.h"
#include "efa.h"
#include "efa_ah.h"
#include "efa_av.h"
#include "efa_mr.h"
#include "efa_base_ep.h"
#include "efa_cq.h"
#include "efa_data_path_ops.h"
#include "efa_io_defs.h"
#include "rdm/efa_rdm_ep.h"
#include "rdm/efa_rdm_peer.h"
#include "rdm/efa_rdm_protocol.h"

/* Seeded into the redirected doorbell, so "not rung" is distinguishable. */
#define EFA_TEST_SQ_DB_SENTINEL 0xDEADBEEFu

static struct efa_base_ep *efa_test_base_ep_from_ep(struct fid_ep *ep)
{
	return container_of(ep, struct efa_base_ep, util_ep.ep_fid);
}

int efa_test_ep_is_wr_started(struct fid_ep *ep)
{
	return efa_test_base_ep_from_ep(ep)->is_wr_started;
}

/* ---------------------------------------------------------------------------
 * EFA_TEST_DP_RDMA_CORE: the work request vtable
 *
 * struct ibv_qp_ex carries no user cookie, so the mocked slots have no way to
 * reach per-probe state. gtest runs one test at a time in one thread, and
 * install() resets these, so one set of file-static counters is enough.
 * ------------------------------------------------------------------------ */

static unsigned g_wr_complete_cnt;
static int g_wr_complete_err;

static void efa_test_mock_wr_start(struct ibv_qp_ex *qpx)
{
}

static int efa_test_mock_wr_complete(struct ibv_qp_ex *qpx)
{
	g_wr_complete_cnt++;
	return g_wr_complete_err;
}

static void efa_test_mock_wr_send(struct ibv_qp_ex *qpx)
{
}

static void efa_test_mock_wr_send_imm(struct ibv_qp_ex *qpx, __be32 imm_data)
{
}

static void efa_test_mock_wr_rdma_read(struct ibv_qp_ex *qpx, uint32_t rkey,
				       uint64_t remote_addr)
{
}

static void efa_test_mock_wr_rdma_write(struct ibv_qp_ex *qpx, uint32_t rkey,
					uint64_t remote_addr)
{
}

static void efa_test_mock_wr_rdma_write_imm(struct ibv_qp_ex *qpx,
					    uint32_t rkey,
					    uint64_t remote_addr,
					    __be32 imm_data)
{
}

static void efa_test_mock_wr_set_sge_list(struct ibv_qp_ex *qpx, size_t num_sge,
					  const struct ibv_sge *sg_list)
{
}

static void efa_test_mock_wr_set_inline_data_list(
	struct ibv_qp_ex *qpx, size_t num_buf,
	const struct ibv_data_buf *buf_list)
{
}

static void efa_test_mock_wr_set_ud_addr(struct ibv_qp_ex *qpx,
					 struct ibv_ah *ah,
					 uint32_t remote_qpn,
					 uint32_t remote_qkey)
{
}

#if HAVE_EFADV_WR_PROCESSING_HINTS
static void efa_test_mock_wr_set_processing_hints(struct efadv_qp *efadv_qp,
						  uint32_t hints)
{
}
#endif

static int efa_test_wr_probe_install(struct efa_qp *qp,
				     struct efa_test_dp_probe *p)
{
	struct ibv_qp_ex *qpx;

	if (!qp->ibv_qp_ex)
		return -FI_EOPNOTSUPP;

	qpx = qp->ibv_qp_ex;
	p->saved_qpx = aligned_alloc(_Alignof(struct ibv_qp_ex),
				     sizeof(struct ibv_qp_ex));
	if (!p->saved_qpx)
		return -FI_ENOMEM;
	memcpy(p->saved_qpx, qpx, sizeof(*qpx));
	p->qpx = qpx;

#if HAVE_EFA_DATA_PATH_DIRECT
	p->saved_direct_enabled = qp->data_path_direct_enabled;
	/* Select the rdma-core arm of the efa_qp_post_* dispatch. */
	qp->data_path_direct_enabled = false;
#endif

	g_wr_complete_cnt = 0;
	g_wr_complete_err = 0;

	qpx->wr_start = &efa_test_mock_wr_start;
	qpx->wr_complete = &efa_test_mock_wr_complete;
	qpx->wr_send = &efa_test_mock_wr_send;
	qpx->wr_send_imm = &efa_test_mock_wr_send_imm;
	qpx->wr_rdma_read = &efa_test_mock_wr_rdma_read;
	qpx->wr_rdma_write = &efa_test_mock_wr_rdma_write;
	qpx->wr_rdma_write_imm = &efa_test_mock_wr_rdma_write_imm;
	qpx->wr_set_sge_list = &efa_test_mock_wr_set_sge_list;
	qpx->wr_set_inline_data_list = &efa_test_mock_wr_set_inline_data_list;
	qpx->wr_set_ud_addr = &efa_test_mock_wr_set_ud_addr;

#if HAVE_EFADV_WR_PROCESSING_HINTS
	{
		struct efadv_qp *efadv_qp = efadv_qp_from_ibv_qp_ex(qpx);

		p->saved_set_hints = efadv_qp->wr_set_processing_hints;
		efadv_qp->wr_set_processing_hints =
			&efa_test_mock_wr_set_processing_hints;
	}
#endif
	return 0;
}

static void efa_test_wr_probe_restore(struct efa_test_dp_probe *p)
{
	if (!p->qpx)
		return;

#if HAVE_EFADV_WR_PROCESSING_HINTS
	efadv_qp_from_ibv_qp_ex(p->qpx)->wr_set_processing_hints =
		p->saved_set_hints;
#endif
	memcpy(p->qpx, p->saved_qpx, sizeof(struct ibv_qp_ex));
	free(p->saved_qpx);
	p->saved_qpx = NULL;
	p->qpx = NULL;

#if HAVE_EFA_DATA_PATH_DIRECT
	((struct efa_qp *) p->qp)->data_path_direct_enabled =
		p->saved_direct_enabled;
#endif
}

/* ---------------------------------------------------------------------------
 * EFA_TEST_DP_DIRECT: the send queue descriptor buffer and doorbell
 * ------------------------------------------------------------------------ */

#if HAVE_EFA_DATA_PATH_DIRECT

static int efa_test_sq_probe_install(struct efa_qp *qp,
				     struct efa_test_dp_probe *p)
{
	struct efa_data_path_direct_sq *sq;
	size_t bytes;

	if (!qp->data_path_direct_enabled)
		return -FI_EOPNOTSUPP;

	sq = &qp->data_path_direct_qp.sq;
	bytes = (size_t) sq->wq.wqe_cnt * sq->wq.wqe_size;

	/*
	 * aligned_alloc, not malloc: __wrap_malloc counts ordinals for
	 * efa_test_fail_mallocs, and the probe must not shift that count.
	 * 64-byte alignment satisfies mmio_memcpy_x64, which is vst4q_u64 on
	 * aarch64.
	 */
	p->scratch_desc = aligned_alloc(64, bytes);
	if (!p->scratch_desc)
		return -FI_ENOMEM;
	memset(p->scratch_desc, 0, bytes);

	p->sq = sq;
	p->saved_desc = sq->desc;
	p->saved_db = sq->wq.db;
	p->scratch_db = EFA_TEST_SQ_DB_SENTINEL;

	sq->desc = p->scratch_desc;
	sq->wq.db = &p->scratch_db;
	return 0;
}

static void efa_test_sq_probe_restore(struct efa_test_dp_probe *p)
{
	struct efa_data_path_direct_sq *sq = p->sq;

	if (!sq)
		return;

	sq->desc = p->saved_desc;
	sq->wq.db = p->saved_db;
	/*
	 * A FI_MORE test deliberately leaves entries staged. Clear the count, or
	 * a later real post takes the pending branch and rings the real doorbell
	 * for a producer counter the device never saw.
	 */
	sq->num_wqe_pending = 0;

	free(p->scratch_desc);
	p->scratch_desc = NULL;
	p->sq = NULL;
}

static bool efa_test_sq_probe_pending(const struct efa_test_dp_probe *p)
{
	return ((struct efa_data_path_direct_sq *) p->sq)->num_wqe_pending != 0;
}

#else /* !HAVE_EFA_DATA_PATH_DIRECT */

static int efa_test_sq_probe_install(struct efa_qp *qp,
				     struct efa_test_dp_probe *p)
{
	return -FI_EOPNOTSUPP;
}

static void efa_test_sq_probe_restore(struct efa_test_dp_probe *p)
{
}

static bool efa_test_sq_probe_pending(const struct efa_test_dp_probe *p)
{
	return false;
}

#endif /* HAVE_EFA_DATA_PATH_DIRECT */

/* ---------------------------------------------------------------------------
 * Backend-agnostic probe
 * ------------------------------------------------------------------------ */

int efa_test_dp_probe_install(struct fid_ep *ep, int backend,
			      struct efa_test_dp_probe *p)
{
	struct efa_qp *qp = efa_test_base_ep_from_ep(ep)->qp;
	int ret;

	memset(p, 0, sizeof(*p));
	p->backend = backend;

	if (!qp)
		return -FI_EOPNOTSUPP;
	p->qp = qp;

	ret = (backend == EFA_TEST_DP_DIRECT) ?
		      efa_test_sq_probe_install(qp, p) :
		      efa_test_wr_probe_install(qp, p);
	if (ret)
		p->qp = NULL;
	return ret;
}

void efa_test_dp_probe_restore(struct efa_test_dp_probe *p)
{
	struct efa_qp *qp = p->qp;

	if (!qp)
		return;

	if (p->backend == EFA_TEST_DP_DIRECT)
		efa_test_sq_probe_restore(p);
	else
		efa_test_wr_probe_restore(p);

	/*
	 * Closing the endpoint does not flush an open ibv_wr_start session, so a
	 * later real post would skip ibv_wr_start on a session that no longer
	 * exists.
	 */
	qp->base_ep->is_wr_started = false;
	p->qp = NULL;
}

bool efa_test_dp_probe_submitted(const struct efa_test_dp_probe *p)
{
	if (p->backend == EFA_TEST_DP_DIRECT)
		return p->scratch_db != EFA_TEST_SQ_DB_SENTINEL;
	return g_wr_complete_cnt > 0;
}

bool efa_test_dp_probe_pending(const struct efa_test_dp_probe *p)
{
	if (p->backend == EFA_TEST_DP_DIRECT)
		return efa_test_sq_probe_pending(p);
	return ((struct efa_qp *) p->qp)->base_ep->is_wr_started;
}

void efa_test_dp_probe_reset(struct efa_test_dp_probe *p)
{
	if (p->backend == EFA_TEST_DP_DIRECT)
		p->scratch_db = EFA_TEST_SQ_DB_SENTINEL;
	else
		g_wr_complete_cnt = 0;
}

void efa_test_dp_probe_set_submit_error(struct efa_test_dp_probe *p, int err)
{
	g_wr_complete_err = err;
}

/* ---------------------------------------------------------------------------
 * efa-rdm
 * ------------------------------------------------------------------------ */

int efa_test_rdm_setup_peer(struct fid_ep *ep_fid, struct fid_av *av_fid,
			    fi_addr_t *peer_addr)
{
	struct efa_rdm_ep *ep = container_of(ep_fid, struct efa_rdm_ep,
					     base_ep.util_ep.ep_fid);
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	struct efa_rdm_peer *peer;
	int ret;

	ret = fi_getname(&ep_fid->fid, &raw_addr, &raw_addr_len);
	if (ret)
		return ret;
	/* Own GID with a different QPN: the AH is created against the real
	 * device, but the peer is not self, so shm is not a candidate. */
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	if (fi_av_insert(av_fid, &raw_addr, 1, peer_addr, 0, NULL) != 1)
		return -FI_EINVAL;

	peer = efa_rdm_ep_get_peer_explicit(ep, *peer_addr);
	if (!peer)
		return -FI_EINVAL;

	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;
	peer->extra_info[0] |= EFA_RDM_EXTRA_FEATURE_RDMA_READ |
			       EFA_RDM_EXTRA_FEATURE_RDMA_WRITE;
	peer->p2p_supported = true;
	if (peer->conn)
		peer->conn->shm_fi_addr = FI_ADDR_NOTAVAIL;
	/* use_device_rdma defaults off on some platforms, which would route a
	 * read or write through the emulated protocols instead of the QP. */
	ep->use_device_rdma = true;

	return 0;
}
