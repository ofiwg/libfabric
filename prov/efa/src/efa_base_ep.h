/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_BASE_EP_H
#define EFA_BASE_EP_H

#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <infiniband/efadv.h>

#include "ofi.h"
#include "ofi_util.h"
#include "efa_av.h"
#include "efa_thread_annotations.h"
#include "rdm/efa_rdm_protocol.h"
#include "efa_data_path_direct_structs.h"

#define EFA_QP_DEFAULT_SERVICE_LEVEL 0
#define EFA_QP_LOW_LATENCY_SERVICE_LEVEL 8
#define EFA_ERROR_MSG_BUFFER_LENGTH 1024

/* 0x80000000 and up is the privileged Q Key range, which is not usable by
 * unprivileged endpoints. Both the provider generated QKEYs and the ones
 * supplied by the application must stay below it. */
#define EFA_QKEY_PRIVILEGED_MASK 0x80000000

/* Default rnr_retry for efa-rdm ep.
 * If first attempt to send a packet failed,
 * this value controls how many times firmware
 * retries the send before it report an RNR error
 * (via rdma-core error cq entry).
 * The valid number is from
 *      0 (no retry)
 * to
 *      EFA_RNR_INFINITY_RETRY (retry infinitely)
 */
#define EFA_RDM_DEFAULT_RNR_RETRY	(3)
/**
 * Infinite retry.
 * NOTICE: this is the default rnr_retry
 * mode for SRD qp. So modifying qp_attr.rnr_retry
 * to this value has the same behavior as
 * not modifying qp's rnr_retry attribute
 */
#define EFA_RNR_INFINITE_RETRY		(7)

#define efa_rx_flags(efa_base_ep) ((efa_base_ep)->util_ep.rx_op_flags)
#define efa_tx_flags(efa_base_ep) ((efa_base_ep)->util_ep.tx_op_flags)

struct efa_qp {
	struct ibv_qp *ibv_qp;
	struct ibv_qp_ex *ibv_qp_ex;
	struct efa_base_ep *base_ep;
	uint32_t qp_num;
	uint32_t qkey;
	bool data_path_direct_enabled;
#if HAVE_EFA_DATA_PATH_DIRECT
	struct efa_data_path_direct_qp data_path_direct_qp;
#endif
	bool unsolicited_write_recv_enabled;
};

/*
 * Per-work-request completion-signal descriptor, passed by the message-form
 * data path (fi_writemsg/fi_sendmsg with FI_EFA_EXTENDED_MSG) down to the WQE
 * builder. Built on the stack for the current WR; a NULL pointer or
 * feature_bits == 0 means no signals to attach. Not persistent state.
 */
struct efa_comp_signal_wr {
	uint64_t feature_bits;
	uint32_t local_signal_id;
	uint32_t remote_signal_id;
	uint32_t local_signal_data;
	uint32_t remote_signal_data;
};

/*
 * Query the device's maximum inline data for a QP configuration. qp_flags and
 * wr_flags are the exact EFADV_QP_FLAGS_* / EFADV_WR_EX_* the QP uses; both are
 * validated by the verb and describe the QP whose inline size is queried (the
 * completion-signal wr_flags shrink the inline region). Returns the max inline
 * size (>= 0) or a negative errno (-FI_ENOSYS on builds without efadv
 * completion-with-signal support). Shared by QP creation (to clamp the
 * requested inline size) and fi_getopt (to report the effective inline size).
 *
 * Defined in efa_base_ep.c so the efadv attr types stay confined to a TU
 * compiled with the efa provider's rdma-core include path.
 */
int efa_query_max_inline_data(struct ibv_context *ctx, uint32_t qp_flags,
			      uint32_t wr_flags);

#if HAVE_INLINE_BUF_SIZE_EX
/*
 * Query the device's maximum send-queue depth for a wide-WQE configuration.
 * Wide WQEs consume more send-queue memory per entry, lowering the max depth.
 * sq_depth_flags selects which wide-WQE feature(s) apply
 * (EFADV_SQ_DEPTH_ATTR_INLINE_WRITE for large inline data,
 * EFADV_SQ_DEPTH_ATTR_COMP_SIGNAL for signals); max_inline_data is the inline
 * size the QP will use. Returns the max SQ depth (>= 0) or a negative errno.
 * Shared by QP creation (to clamp max_send_wr) and fi_getopt (to report the
 * effective tx size).
 */
int efa_query_max_sq_depth(struct ibv_context *ctx, uint32_t sq_depth_flags,
			   uint32_t max_inline_data);
#endif /* HAVE_INLINE_BUF_SIZE_EX */

struct efa_av;

struct efa_recv_wr {
	/** @brief Work request struct used by rdma-core */
	struct ibv_recv_wr wr;

	/** @brief Scatter gather element array
	 *
	 * @details
	 * EFA device supports a maximum of 2 iov/SGE
	 */
	struct ibv_sge sge[2];
};

/*
 * FI_CONTEXT2 mode of an endpoint (and of the CQ it is bound to). efa-direct
 * can only use the caller-supplied context buffer when USE_CONTEXT2. Otherwise
 * the provider must not dereference the context buffer, so features that depend
 * on it (inject, selective completion, MR tracking) are disabled. UNASSIGNED is
 * used only by a CQ before its first endpoint is enabled; an endpoint is always
 * NO_CONTEXT or USE_CONTEXT2.
 */
enum context_mode {
	UNASSIGNED,
	NO_CONTEXT,
	USE_CONTEXT2,
};

struct efa_base_ep {
	struct util_ep util_ep;
	struct efa_domain *domain;
	struct efa_qp *qp;
	struct efa_av *av;
	struct fi_info *info;
	enum context_mode context_mode;
	size_t rnr_retry;
	struct efa_ep_addr src_addr;

	bool util_ep_initialized;
	bool efa_qp_enabled;
	bool is_wr_started;

	struct efa_recv_wr *efa_recv_wr_vec;
	size_t recv_wr_index;

	size_t max_msg_size;		/**< #FI_OPT_MAX_MSG_SIZE */
	size_t max_rma_size;		/**< #FI_OPT_MAX_RMA_SIZE */
	size_t inject_msg_size;		/**< #FI_OPT_INJECT_MSG_SIZE */
	size_t inject_rma_size;		/**< #FI_OPT_INJECT_RMA_SIZE */

	bool use_unsolicited_write_recv;

	/* Whether completion-with-signal support is enabled on this endpoint
	 * (via FI_OPT_EFA_COMP_SIGNAL). Must be set before the endpoint is
	 * enabled because it governs QP send-queue allocation (wide WQEs). */
	bool comp_signal_enabled;

	/* Pools and list for outstanding operation entries, one pool per
	 * direction. Each endpoint type fills them with its own entry
	 * type, efa_direct_ope for efa-direct and efa_rdm_ope for efa-rdm.
	 * Entries from both are linked via ope_list and used to warn on MR
	 * close while an operation that still references the MR is in flight. */
	struct ofi_bufpool *txe_pool;
	struct ofi_bufpool *rxe_pool;
	struct dlist_entry ope_list;

	/* entry for efa_domain->base_ep_list */
	struct dlist_entry base_ep_entry;
};

int efa_base_ep_bind_av(struct efa_base_ep *base_ep, struct efa_av *av);

int efa_base_ep_destruct(struct efa_base_ep *base_ep);

int efa_base_ep_enable(struct efa_base_ep *base_ep)
	OFI_TSA_REQUIRES(efa_qp_table_lock_sym);

int efa_base_ep_construct(struct efa_base_ep *base_ep,
			  struct fid_domain* domain_fid,
			  struct fi_info *info,
			  ofi_ep_progress_func progress,
			  void *context);

int efa_base_ep_getname(fid_t fid, void *addr, size_t *addrlen);

int efa_ep_open(struct fid_domain *domain_fid, struct fi_info *user_info,
		struct fid_ep **ep_fid, void *context);

int efa_qp_create(struct efa_qp **qp, struct ibv_qp_init_attr_ex *init_attr_ex,
		   uint32_t tclass, bool enable_unsolicited_write_recv);

void efa_qp_destruct(struct efa_qp *qp);

void efa_base_ep_close_util_ep(struct efa_base_ep *base_ep);

int efa_base_ep_destruct_qp(struct efa_base_ep *base_ep);

int efa_base_ep_destruct_qp_unsafe(struct efa_base_ep *base_ep)
	OFI_TSA_REQUIRES(efa_qp_table_lock_sym);

bool efa_qp_support_op_in_order_aligned_128_bytes(struct efa_qp *qp,
						       enum ibv_wr_opcode op);

void efa_base_ep_write_eq_error(struct efa_base_ep *ep,
				ssize_t err,
				ssize_t prov_errno);

const char *efa_base_ep_raw_addr_str(struct efa_base_ep *base_ep, char *buf,
				     size_t *buflen);

struct efa_ep_addr *efa_base_ep_get_peer_raw_addr(struct efa_base_ep *base_ep,
						  fi_addr_t addr);

const char *efa_base_ep_get_peer_raw_addr_str(struct efa_base_ep *base_ep,
					      fi_addr_t addr, char *buf,
					      size_t *buflen);

struct efa_cq *efa_base_ep_get_tx_cq(struct efa_base_ep *ep);

struct efa_cq *efa_base_ep_get_rx_cq(struct efa_base_ep *ep);

int efa_base_ep_check_qp_in_order_aligned_128_bytes(struct efa_base_ep *base_ep,
						   enum ibv_wr_opcode op_code);

int efa_base_ep_insert_cntr_ibv_cq_poll_list(struct efa_base_ep *ep);

void efa_base_ep_remove_cntr_ibv_cq_poll_list(struct efa_base_ep *ep);

int efa_base_ep_create_and_enable_qp(struct efa_base_ep *ep);

void efa_base_ep_construct_ibv_qp_init_attr_ex(struct efa_base_ep *ep,
						struct ibv_qp_init_attr_ex *attr_ex,
						struct ibv_cq_ex *tx_cq,
						struct ibv_cq_ex *rx_cq);

#if ENABLE_DEBUG
void efa_ep_addr_print(char *prefix, struct efa_ep_addr *addr);
#endif

static inline size_t efa_base_ep_get_rx_pool_size(struct efa_base_ep *base_ep)
{
	return MIN(base_ep->domain->device->rdm_info->rx_attr->size, base_ep->info->rx_attr->size);
}

static inline size_t efa_base_ep_get_tx_pool_size(struct efa_base_ep *base_ep)
{
	return MIN(base_ep->domain->device->rdm_info->tx_attr->size, base_ep->info->tx_attr->size);
}

#endif
