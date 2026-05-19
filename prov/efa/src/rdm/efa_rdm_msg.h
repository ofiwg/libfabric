/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

struct efa_rdm_proto;
static inline
void efa_rdm_msg_construct(struct fi_msg *msg, const struct iovec *iov, void **desc,
		       size_t count, fi_addr_t addr, void *context, uint64_t data)
{
	msg->msg_iov = iov;
	msg->desc = desc;
	msg->iov_count = count;
	msg->addr = addr;
	msg->context = context;
	msg->data = data;
}

/**
 * functions to allocate rxe for two sided operations
 */
struct efa_rdm_ope *
efa_rdm_msg_alloc_rxe_for_msgrtm(struct efa_rdm_ep *ep,
				 struct efa_rdm_pke **pkt_entry_ptr);

struct efa_rdm_ope *
efa_rdm_msg_alloc_rxe_for_tagrtm(struct efa_rdm_ep *ep,
				 struct efa_rdm_pke **pkt_entry_ptr);

struct efa_rdm_ope *efa_rdm_msg_split_rxe(struct efa_rdm_ep *ep,
					    struct efa_rdm_ope *posted_entry,
					    struct efa_rdm_ope *consumer_entry,
					    struct efa_rdm_pke *pkt_entry);
ssize_t efa_rdm_msg_post_rtm(struct efa_rdm_ep *ep, struct efa_rdm_ope *txe);

/**
 * @brief Compute the effective fi_flags for a TX operation.
 *
 * Merges per-operation flags with the endpoint's tx_op_flags, respecting
 * the tx_msg_flags setting for FI_COMPLETION.
 *
 * @param[in] ep        RDM endpoint
 * @param[in] fi_flags  Per-operation flags from the application
 * @return              Merged flags to use for the TX operation
 */
static inline
uint64_t efa_rdm_msg_get_tx_flags(struct efa_rdm_ep *ep, uint64_t fi_flags)
{
	uint64_t tx_op_flags;

	assert(ep->base_ep.util_ep.tx_msg_flags == 0 ||
	       ep->base_ep.util_ep.tx_msg_flags == FI_COMPLETION);
	tx_op_flags = ep->base_ep.util_ep.tx_op_flags;
	if (ep->base_ep.util_ep.tx_msg_flags == 0)
		tx_op_flags &= ~FI_COMPLETION;
	return fi_flags | tx_op_flags;
}

ssize_t efa_rdm_msg_post_rtm_proto(struct efa_rdm_ep *ep, struct efa_rdm_ope *txe,
				    struct efa_rdm_proto *proto);
/*
 * The following 2 OP structures are defined in efa_rdm_msg.c and is
 * used by #efa_rdm_ep_open()
 */
extern struct fi_ops_msg efa_rdm_msg_ops;

extern struct fi_ops_tagged efa_rdm_msg_tagged_ops;