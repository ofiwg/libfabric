/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

/*
 * Definitions of the EFA data path operations: the rdma-core post helpers, and
 * the wrappers that dispatch each queue pair and completion queue operation
 * between them and the direct data path.
 *
 * These bodies live in their own header so that the same text can be compiled
 * with two different linkages, selected by EFA_PROD_STATIC_INLINE:
 *
 *   - Production build: efa_data_path_ops.h defines it to "static inline" and
 *     includes this file, so every translation unit gets its own inlinable copy.
 *
 *   - Unit test build (EFA_UNIT_TEST): efa_data_path_ops.h defines it to nothing
 *     and declares these functions extern instead.
 *     prov/efa/test/efa_unit_test_data_path_ops.c is then the single translation
 *     unit that includes this file, producing one external-linkage definition
 *     per function so that `ld --wrap` can intercept them.
 *
 * A test build therefore reaches the real implementation unless it installs a
 * mock. A test that must not touch the device installs one that does nothing.
 *
 * Do not include this file directly: include it only from efa_data_path_ops.h,
 * or from the unit test translation unit that emits the definitions, and there
 * only after efa_data_path_ops.h.
 */

#ifndef EFA_DATA_PATH_OPS_BODY_H
#define EFA_DATA_PATH_OPS_BODY_H

/**
 * @brief RDMA-core version of send operation using ibv_* APIs
 */
EFA_PROD_STATIC_INLINE int
efa_ibv_post_send(
		struct efa_qp *qp,
		const struct ibv_sge *sge_list,
		const struct ibv_data_buf *inline_data_list,
		size_t data_count,
		bool use_inline,
		uintptr_t wr_id,
		uint64_t data,
		uint64_t flags,
		struct efa_ah *ah,
		uint32_t qpn,
		uint32_t qkey)
{
	struct efa_base_ep *base_ep = qp->base_ep;
	int ret;

	if (!base_ep->is_wr_started) {
		ibv_wr_start(qp->ibv_qp_ex);
		base_ep->is_wr_started = true;
	}

	qp->ibv_qp_ex->wr_id = wr_id;

	if (flags & FI_REMOTE_CQ_DATA) {
		ibv_wr_send_imm(qp->ibv_qp_ex, data);
	} else {
		ibv_wr_send(qp->ibv_qp_ex);
	}

	if (use_inline) {
		ibv_wr_set_inline_data_list(qp->ibv_qp_ex, data_count, inline_data_list);
	} else {
		ibv_wr_set_sge_list(qp->ibv_qp_ex, data_count, sge_list);
	}

	ibv_wr_set_ud_addr(qp->ibv_qp_ex, ah->ibv_ah, qpn, qkey);

	if (!(flags & FI_MORE)) {
		ret = ibv_wr_complete(qp->ibv_qp_ex);
		base_ep->is_wr_started = false;
		return ret;
	}

	return 0;
}

/**
 * @brief RDMA-core version of RDMA read operation using ibv_* APIs
 */
EFA_PROD_STATIC_INLINE int
efa_ibv_post_read(
		struct efa_qp *qp,
		const struct ibv_sge *sge_list,
		size_t sge_count,
		uint32_t remote_key,
		uint64_t remote_addr,
		uintptr_t wr_id,
		uint64_t flags,
		struct efa_ah *ah,
		uint32_t qpn,
		uint32_t qkey)
{
	struct efa_base_ep *base_ep = qp->base_ep;
	int ret;

	if (!base_ep->is_wr_started) {
		ibv_wr_start(qp->ibv_qp_ex);
		base_ep->is_wr_started = true;
	}

	qp->ibv_qp_ex->wr_id = wr_id;
	ibv_wr_rdma_read(qp->ibv_qp_ex, remote_key, remote_addr);
	ibv_wr_set_sge_list(qp->ibv_qp_ex, sge_count, sge_list);
	ibv_wr_set_ud_addr(qp->ibv_qp_ex, ah->ibv_ah, qpn, qkey);

	if (!(flags & FI_MORE)) {
		ret = ibv_wr_complete(qp->ibv_qp_ex);
		base_ep->is_wr_started = false;
		return ret;
	}

	return 0;
}

/**
 * @brief RDMA-core version of RDMA write operation using ibv_* APIs
 */
EFA_PROD_STATIC_INLINE int
efa_ibv_post_write(
		struct efa_qp *qp,
		const struct ibv_sge *sge_list,
		size_t sge_count,
		const struct ibv_data_buf *inline_data_list,
		bool use_inline,
		uint32_t remote_key,
		uint64_t remote_addr,
		uintptr_t wr_id,
		uint64_t data,
		uint64_t flags,
		struct efa_ah *ah,
		uint32_t qpn,
		uint32_t qkey)
{
	struct efa_base_ep *base_ep = qp->base_ep;
	int ret;

	if (!base_ep->is_wr_started) {
		ibv_wr_start(qp->ibv_qp_ex);
		base_ep->is_wr_started = true;
	}

	qp->ibv_qp_ex->wr_id = wr_id;

	if (flags & FI_REMOTE_CQ_DATA) {
		ibv_wr_rdma_write_imm(qp->ibv_qp_ex, remote_key, remote_addr, data);
	} else {
		ibv_wr_rdma_write(qp->ibv_qp_ex, remote_key, remote_addr);
	}

	if (use_inline)
		ibv_wr_set_inline_data_list(qp->ibv_qp_ex, sge_count, inline_data_list);
	else
		ibv_wr_set_sge_list(qp->ibv_qp_ex, sge_count, sge_list);

	ibv_wr_set_ud_addr(qp->ibv_qp_ex, ah->ibv_ah, qpn, qkey);

#if HAVE_EFADV_WR_PROCESSING_HINTS
	if (flags & FI_EFA_WR_HIGH_PPS)
		efadv_wr_set_processing_hints(efadv_qp_from_ibv_qp_ex(qp->ibv_qp_ex),
					     EFADV_WR_PROCESSING_HINT_BURST_PPS_SENSITIVE);
#endif

	if (!(flags & FI_MORE)) {
		ret = ibv_wr_complete(qp->ibv_qp_ex);
		base_ep->is_wr_started = false;
		return ret;
	}

	return 0;
}

/* QP wrapper functions */
EFA_PROD_STATIC_INLINE int efa_qp_post_recv(struct efa_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	if (qp->data_path_direct_enabled)
		return efa_data_path_direct_post_recv(qp, wr, bad);
#endif
	return ibv_post_recv(qp->ibv_qp, wr, bad);
}

/**
 * @brief Wrapper for send operations - chooses between direct and IBV paths
 */
EFA_PROD_STATIC_INLINE int
efa_qp_post_send(struct efa_qp *qp,
                 const struct ibv_sge *sge_list,
                 const struct ibv_data_buf *inline_data_list,
                 size_t data_count,
                 bool use_inline,
                 uintptr_t wr_id,
                 uint64_t data,
                 uint64_t flags,
                 struct efa_ah *ah,
                 uint32_t qpn,
                 uint32_t qkey)
{
	EFA_DBG(FI_LOG_EP_DATA, "Posting WQE: qp=%p data_count=%ld use_inline=%d wr_id=0x%lx data=0x%lx flags=0x%lx qpn=%u qkey=0x%x\n",
		qp, data_count, use_inline, wr_id, data, flags, qpn, qkey);
#if HAVE_EFA_DATA_PATH_DIRECT
	if (qp->data_path_direct_enabled)
		return efa_data_path_direct_post_send(qp, sge_list, inline_data_list, data_count,
					   use_inline, wr_id, data, flags, ah, qpn, qkey);
#endif
	return efa_ibv_post_send(qp, sge_list, inline_data_list, data_count,
				use_inline, wr_id, data, flags, ah, qpn, qkey);
}

/**
 * @brief Wrapper for RDMA read operations - chooses between direct and IBV paths
 */
EFA_PROD_STATIC_INLINE int
efa_qp_post_read(struct efa_qp *qp,
                 const struct ibv_sge *sge_list,
                 size_t sge_count,
                 uint32_t remote_key,
                 uint64_t remote_addr,
                 uintptr_t wr_id,
                 uint64_t flags,
                 struct efa_ah *ah,
                 uint32_t qpn,
                 uint32_t qkey)
{
	EFA_DBG(FI_LOG_EP_DATA, "Posting WQE: qp=%p sge_count=%ld remote_key=%u remote_addr=0x%lx wr_id=0x%lx flags=0x%lx qpn=%u qkey=0x%x\n",
		qp, sge_count, remote_key, remote_addr, wr_id, flags, qpn, qkey);
#if HAVE_EFA_DATA_PATH_DIRECT
	if (qp->data_path_direct_enabled)
		return efa_data_path_direct_post_read(qp, sge_list, sge_count,
						remote_key, remote_addr, wr_id, flags, ah, qpn, qkey);
#endif
	return efa_ibv_post_read(qp, sge_list, sge_count,
				 remote_key, remote_addr, wr_id, flags, ah, qpn, qkey);
}

/**
 * @brief Wrapper for RDMA write operations - chooses between direct and IBV paths
 */
EFA_PROD_STATIC_INLINE int
efa_qp_post_write(struct efa_qp *qp,
                  const struct ibv_sge *sge_list,
                  size_t sge_count,
                  const struct ibv_data_buf *inline_data_list,
                  bool use_inline,
                  uint32_t remote_key,
                  uint64_t remote_addr,
                  uintptr_t wr_id,
                  uint64_t data,
                  uint64_t flags,
                  struct efa_ah *ah,
                  uint32_t qpn,
                  uint32_t qkey)
{
	EFA_DBG(FI_LOG_EP_DATA, "Posting WQE: qp=%p sge_count=%ld remote_key=%u remote_addr=0x%lx wr_id=0x%lx data=0x%lx flags=0x%lx qpn=%u qkey=0x%x\n",
		qp, sge_count, remote_key, remote_addr, wr_id, data, flags, qpn, qkey);
#if HAVE_EFA_DATA_PATH_DIRECT
	if (qp->data_path_direct_enabled)
		return efa_data_path_direct_post_write(qp, sge_list, sge_count,
					 inline_data_list, use_inline,
					 remote_key, remote_addr, wr_id, data, flags, ah, qpn, qkey);
#endif
	return efa_ibv_post_write(qp, sge_list, sge_count,
				  inline_data_list, use_inline,
				  remote_key, remote_addr, wr_id, data, flags, ah, qpn, qkey);
}

/* CQ wrapper functions */

EFA_PROD_STATIC_INLINE int efa_ibv_cq_start_poll(struct efa_ibv_cq *ibv_cq, struct ibv_poll_cq_attr *attr)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	if (ibv_cq->data_path_direct_enabled)
		return efa_data_path_direct_start_poll(ibv_cq, attr);
#endif
	return ibv_start_poll(ibv_cq->ibv_cq_ex, attr);
}

EFA_PROD_STATIC_INLINE int efa_ibv_cq_next_poll(struct efa_ibv_cq *ibv_cq)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	if (ibv_cq->data_path_direct_enabled)
		return efa_data_path_direct_next_poll(ibv_cq);
#endif
	return ibv_next_poll(ibv_cq->ibv_cq_ex);
}

EFA_PROD_STATIC_INLINE enum ibv_wc_opcode efa_ibv_cq_wc_read_opcode(struct efa_ibv_cq *ibv_cq)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	if (ibv_cq->data_path_direct_enabled)
		return efa_data_path_direct_wc_read_opcode(ibv_cq);
#endif
	return ibv_wc_read_opcode(ibv_cq->ibv_cq_ex);
}

EFA_PROD_STATIC_INLINE void efa_ibv_cq_end_poll(struct efa_ibv_cq *ibv_cq)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	if (ibv_cq->data_path_direct_enabled) {
		efa_data_path_direct_end_poll(ibv_cq);
		return;
	}
#endif
	ibv_end_poll(ibv_cq->ibv_cq_ex);
}

EFA_PROD_STATIC_INLINE uint32_t efa_ibv_cq_wc_read_qp_num(struct efa_ibv_cq *ibv_cq)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	if (ibv_cq->data_path_direct_enabled)
		return efa_data_path_direct_wc_read_qp_num(ibv_cq);
#endif
	return ibv_wc_read_qp_num(ibv_cq->ibv_cq_ex);
}

EFA_PROD_STATIC_INLINE uint32_t efa_ibv_cq_wc_read_vendor_err(struct efa_ibv_cq *ibv_cq)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	if (ibv_cq->data_path_direct_enabled)
		return efa_data_path_direct_wc_read_vendor_err(ibv_cq);
#endif
	return ibv_wc_read_vendor_err(ibv_cq->ibv_cq_ex);
}

EFA_PROD_STATIC_INLINE uint32_t efa_ibv_cq_wc_read_src_qp(struct efa_ibv_cq *ibv_cq)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	if (ibv_cq->data_path_direct_enabled)
		return efa_data_path_direct_wc_read_src_qp(ibv_cq);
#endif
	return ibv_wc_read_src_qp(ibv_cq->ibv_cq_ex);
}

EFA_PROD_STATIC_INLINE uint32_t efa_ibv_cq_wc_read_slid(struct efa_ibv_cq *ibv_cq)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	if (ibv_cq->data_path_direct_enabled)
		return efa_data_path_direct_wc_read_slid(ibv_cq);
#endif
	return ibv_wc_read_slid(ibv_cq->ibv_cq_ex);
}

EFA_PROD_STATIC_INLINE uint32_t efa_ibv_cq_wc_read_byte_len(struct efa_ibv_cq *ibv_cq)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	if (ibv_cq->data_path_direct_enabled)
		return efa_data_path_direct_wc_read_byte_len(ibv_cq);
#endif
	return ibv_wc_read_byte_len(ibv_cq->ibv_cq_ex);
}

EFA_PROD_STATIC_INLINE unsigned int efa_ibv_cq_wc_read_wc_flags(struct efa_ibv_cq *ibv_cq)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	if (ibv_cq->data_path_direct_enabled)
		return efa_data_path_direct_wc_read_wc_flags(ibv_cq);
#endif
	return ibv_wc_read_wc_flags(ibv_cq->ibv_cq_ex);
}

EFA_PROD_STATIC_INLINE __be32 efa_ibv_cq_wc_read_imm_data(struct efa_ibv_cq *ibv_cq)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	if (ibv_cq->data_path_direct_enabled)
		return efa_data_path_direct_wc_read_imm_data(ibv_cq);
#endif
	return ibv_wc_read_imm_data(ibv_cq->ibv_cq_ex);
}


EFA_PROD_STATIC_INLINE bool efa_ibv_cq_wc_is_unsolicited(struct efa_ibv_cq *ibv_cq)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	if (ibv_cq->data_path_direct_enabled)
		return efa_data_path_direct_wc_is_unsolicited(ibv_cq);
#endif
#if HAVE_CAPS_UNSOLICITED_WRITE_RECV
	return efadv_wc_is_unsolicited(efadv_cq_from_ibv_cq_ex(ibv_cq->ibv_cq_ex));
#else
	return false;
#endif
}

EFA_PROD_STATIC_INLINE int efa_ibv_cq_wc_read_sgid(struct efa_ibv_cq *ibv_cq, union ibv_gid *sgid)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	if (ibv_cq->data_path_direct_enabled)
		return efa_data_path_direct_wc_read_sgid(ibv_cq, sgid);
#endif

#if HAVE_EFADV_CQ_EX
	return efadv_wc_read_sgid(efadv_cq_from_ibv_cq_ex(ibv_cq->ibv_cq_ex), sgid);
#else
	return false;
#endif
}

EFA_PROD_STATIC_INLINE int efa_ibv_get_cq_event(struct efa_ibv_cq *ibv_cq, void **cq_context)
{
	struct ibv_cq *cq = ibv_cq_ex_to_cq(ibv_cq->ibv_cq_ex);
#if HAVE_EFA_DATA_PATH_DIRECT && HAVE_EFADV_CQ_ATTR_DB
	if (ibv_cq->data_path_direct_enabled)
		return efa_data_path_direct_get_cq_event(ibv_cq, &cq, cq_context);
#endif
#if HAVE_EFA_CQ_NOTIFICATION
	return ibv_get_cq_event(ibv_cq->channel, &cq, cq_context);
#else
	return -FI_ENOSYS;
#endif
}

EFA_PROD_STATIC_INLINE int efa_ibv_req_notify_cq(struct efa_ibv_cq *ibv_cq, int solicited_only)
{
#if HAVE_EFA_DATA_PATH_DIRECT && HAVE_EFADV_CQ_ATTR_DB
	if (ibv_cq->data_path_direct_enabled)
		return efa_data_path_direct_req_notify_cq(ibv_cq, solicited_only);
#endif
#if HAVE_EFA_CQ_NOTIFICATION
	return ibv_req_notify_cq(ibv_cq_ex_to_cq(ibv_cq->ibv_cq_ex), solicited_only);
#else
	return -FI_ENOSYS;
#endif
}

#endif /* EFA_DATA_PATH_OPS_BODY_H */
