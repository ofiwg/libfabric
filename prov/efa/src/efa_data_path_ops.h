/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

/*
 * EFA Data Path Operations
 *
 * This file contains wrapper functions for EFA device operations that are used
 * in the data transfer path. These operations provide a unified interface for
 * both regular IBV operations and direct CQ operations, allowing the EFA provider
 * to seamlessly switch between different hardware acceleration modes based on
 * device capabilities and configuration.
 *
 * The wrapper functions handle:
 * - Queue Pair (QP) operations: post_recv, work request operations (send, RDMA read/write)
 * - Completion Queue (CQ) operations: polling, reading completion data
 * - Automatic selection between IBV and direct CQ implementations
 */

#ifndef EFA_DATA_PATH_OPS_H
#define EFA_DATA_PATH_OPS_H

#include <infiniband/verbs.h>
#include <infiniband/efadv.h>

/* Forward declarations to avoid cyclic dependencies */
#include "efa_base_ep.h"
#include "efa_cq.h"

#if HAVE_EFA_DATA_PATH_DIRECT
#include "efa_data_path_direct_entry.h"
#endif

#if EFA_UNIT_TEST
/*
 * Give the data path wrappers external linkage so that `ld --wrap` can
 * intercept them. Their bodies are emitted once, in
 * prov/efa/test/efa_unit_test_data_path_ops.c, which includes
 * efa_data_path_ops_body.h with this macro expanding to nothing. An unmocked
 * call therefore reaches the real implementation and the real device; a test
 * that must not touch the device installs a mock that does nothing.
 */
#define EFA_PROD_STATIC_INLINE

/* For unit tests, declare functions that are defined in efa_unit_test_data_path_ops.c */
int efa_ibv_post_send(struct efa_qp *qp, const struct ibv_sge *sge_list,
		      const struct ibv_data_buf *inline_data_list,
		      size_t data_count, bool use_inline, uintptr_t wr_id,
		      uint64_t data, uint64_t flags, struct efa_ah *ah,
		      uint32_t qpn, uint32_t qkey);
int efa_ibv_post_read(struct efa_qp *qp, const struct ibv_sge *sge_list,
		      size_t sge_count, uint32_t remote_key,
		      uint64_t remote_addr, uintptr_t wr_id, uint64_t flags,
		      struct efa_ah *ah, uint32_t qpn, uint32_t qkey);
int efa_ibv_post_write(struct efa_qp *qp, const struct ibv_sge *sge_list,
		       size_t sge_count,
		       const struct ibv_data_buf *inline_data_list,
		       bool use_inline, uint32_t remote_key,
		       uint64_t remote_addr, uintptr_t wr_id, uint64_t data,
		       uint64_t flags, struct efa_ah *ah, uint32_t qpn,
		       uint32_t qkey);
int efa_qp_post_recv(struct efa_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad);
int efa_qp_post_send(struct efa_qp *qp, const struct ibv_sge *sge_list,
		      const struct ibv_data_buf *inline_data_list,
		      size_t iov_count, bool use_inline, uintptr_t wr_id,
		      uint64_t data, uint64_t flags, struct efa_ah *ah,
		      uint32_t qpn, uint32_t qkey);
int efa_qp_post_read(struct efa_qp *qp, const struct ibv_sge *sge_list,
		      size_t sge_count, uint32_t remote_key,
		      uint64_t remote_addr, uintptr_t wr_id, uint64_t flags,
		      struct efa_ah *ah, uint32_t qpn, uint32_t qkey);
int efa_qp_post_write(struct efa_qp *qp, const struct ibv_sge *sge_list, size_t sge_count,
		       const struct ibv_data_buf *inline_data_list, bool use_inline,
		       uint32_t remote_key, uint64_t remote_addr,
		       uintptr_t wr_id, uint64_t data,
		       uint64_t flags, struct efa_ah *ah, uint32_t qpn,
		       uint32_t qkey);
int efa_ibv_cq_start_poll(struct efa_ibv_cq *ibv_cq, struct ibv_poll_cq_attr *attr);
int efa_ibv_cq_next_poll(struct efa_ibv_cq *ibv_cq);
enum ibv_wc_opcode efa_ibv_cq_wc_read_opcode(struct efa_ibv_cq *ibv_cq);
void efa_ibv_cq_end_poll(struct efa_ibv_cq *ibv_cq);
uint32_t efa_ibv_cq_wc_read_qp_num(struct efa_ibv_cq *ibv_cq);
uint32_t efa_ibv_cq_wc_read_vendor_err(struct efa_ibv_cq *ibv_cq);
uint32_t efa_ibv_cq_wc_read_src_qp(struct efa_ibv_cq *ibv_cq);
uint32_t efa_ibv_cq_wc_read_slid(struct efa_ibv_cq *ibv_cq);
uint32_t efa_ibv_cq_wc_read_byte_len(struct efa_ibv_cq *ibv_cq);
unsigned int efa_ibv_cq_wc_read_wc_flags(struct efa_ibv_cq *ibv_cq);
__be32 efa_ibv_cq_wc_read_imm_data(struct efa_ibv_cq *ibv_cq);
bool efa_ibv_cq_wc_is_unsolicited(struct efa_ibv_cq *ibv_cq);

int efa_ibv_cq_wc_read_sgid(struct efa_ibv_cq *ibv_cq, union ibv_gid *sgid);

int efa_ibv_get_cq_event(struct efa_ibv_cq *ibv_cq, void **cq_context);
int efa_ibv_req_notify_cq(struct efa_ibv_cq *ibv_cq, int solicited_only);

#else
/* For production, define them all static inline. */
#define EFA_PROD_STATIC_INLINE static inline
#include "efa_data_path_ops_body.h"

#endif /* EFA_UNIT_TEST */

/**
 * @brief Check whether a completion consumes recv buffer
 *
 * @param ibv_cq efa ibv cq
 * @return true the wc consumes a recv buffer
 * @return false the wc doesn't consume a recv buffer
 */
static inline bool efa_cq_wc_is_unsolicited(struct efa_ibv_cq *ibv_cq)
{
	return ibv_cq->unsolicited_write_recv_enabled && efa_ibv_cq_wc_is_unsolicited(ibv_cq);
}

static inline bool efa_cq_wc_available(struct efa_ibv_cq *cq)
{
	return cq->poll_active && !cq->poll_err;
}

static inline void efa_cq_report_poll_err(struct efa_ibv_cq *cq)
{
	int err = cq->poll_err;

	if (err && err != ENOENT)
		EFA_INFO(FI_LOG_CQ, "Ignoring CQ entries from destroyed queue pair\n");
}

static inline void efa_cq_start_poll(struct efa_ibv_cq *cq)
{
	/**
	 * It is possible that the last efa_cq_readfrom
	 * is leaving the device cq in a poll active status
	 * when polling a failed cqe and leave it for the efa_cq_readfrom, efa_cq_readerr
	 * or efa_cq_poll_ibv_cq to consume it. And efa_cq_poll_ibv_cq
	 * will call this wrapper at the beginning.
	 * We shouldn't start poll in this stuation as it will make the
	 * cqe index shifted and the entry lost.
	 */
	if (cq->poll_active)
		return;

	/* Pass an empty ibv_poll_cq_attr struct (zero-initialized) for
	 * ibv_start_poll. EFA expects .comp_mask = 0, or otherwise returns EINVAL.
	 */
	cq->poll_err = efa_ibv_cq_start_poll(cq, &(struct ibv_poll_cq_attr){0});
	if (!cq->poll_err) {
		cq->poll_active = true;
		EFA_DBG(FI_LOG_CQ, "Polled CQE: wr_id 0x%lx\n", cq->ibv_cq_ex->wr_id);
	} else {
		efa_cq_report_poll_err(cq);
	}
}

static inline void efa_cq_next_poll(struct efa_ibv_cq *cq)
{
	assert(cq->poll_active);
	cq->poll_err = efa_ibv_cq_next_poll(cq);
	if (cq->poll_err) {
		efa_cq_report_poll_err(cq);
		return;
	}
	EFA_DBG(FI_LOG_CQ, "Polled CQE: wr_id 0x%lx\n", cq->ibv_cq_ex->wr_id);
}

static inline void efa_cq_end_poll(struct efa_ibv_cq *cq)
{
	if (cq->poll_active)
		efa_ibv_cq_end_poll(cq);
	cq->poll_active = false;
	cq->poll_err = 0;
}

static inline struct efa_base_ep *efa_ibv_cq_get_base_ep_from_cur_cqe(struct efa_ibv_cq *cq, struct efa_domain *efa_domain)
	/* No lock analysis: completion path is guarded by the CQ lock. */
	OFI_TSA_NO_ANALYSIS
{
	struct efa_qp *qp = efa_domain->device->qp_table[efa_ibv_cq_wc_read_qp_num(cq) & efa_domain->device->qp_table_sz_m1];

	return qp ? qp->base_ep : NULL;
}

#endif /* EFA_DATA_PATH_OPS_H */
