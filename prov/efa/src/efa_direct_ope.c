/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

/*
 * This file implements the EFA direct operation entry (ope) pool management.
 * Direct operation entries are used to track outstanding data transfer operations
 * (send, receive, and RMA) and their associated memory descriptors when memory
 * region tracking is enabled (efa_env.track_mr). The pool provides efficient
 * allocation and release of these entries during data transfer operations.
 */

#include "efa.h"
#include "efa_direct_ope.h"

int efa_direct_ope_pool_create(struct efa_direct_ep *ep)
{
	int ret;

	if (!efa_env.track_mr) {
		ep->ope_pool = NULL;
		return 0;
	}

	dlist_init(&ep->ope_list);

	ret = ofi_bufpool_create(&ep->ope_pool,
				 sizeof(struct efa_direct_ope),
				 EFA_RDM_BUFPOOL_ALIGNMENT,
				 ep->base_ep.info->tx_attr->size + ep->base_ep.info->rx_attr->size,
				 ep->base_ep.info->tx_attr->size + ep->base_ep.info->rx_attr->size,
				 OFI_BUFPOOL_NO_TRACK);
	if (ret) {
		EFA_WARN(FI_LOG_EP_CTRL,
			 "Failed to create EFA direct op entry pool: %s\n",
			 fi_strerror(-ret));
		return ret;
	}

	ret = ofi_bufpool_grow(ep->ope_pool);
	if (ret) {
		ofi_bufpool_destroy(ep->ope_pool);
		ep->ope_pool = NULL;
		return ret;
	}

	EFA_INFO(FI_LOG_EP_CTRL, "ep %p: Created EFA direct op entry pool with size %zu\n",
		 ep, ep->base_ep.info->tx_attr->size + ep->base_ep.info->rx_attr->size);

	return 0;
}

void efa_direct_ope_pool_destroy(struct efa_direct_ep *ep)
{
	struct efa_direct_ope *direct_ope;
	struct dlist_entry *tmp;

	if (!ep->ope_pool)
		return;

	ofi_genlock_lock(&ep->base_ep.domain->util_domain.lock);
	if (!dlist_empty(&ep->ope_list)) {
		dlist_foreach_container_safe(&ep->ope_list,
					     struct efa_direct_ope,
					     direct_ope, entry, tmp) {
			dlist_remove(&direct_ope->entry);
			ofi_buf_free(direct_ope);
		}
	}
	ofi_genlock_unlock(&ep->base_ep.domain->util_domain.lock);

	EFA_INFO(FI_LOG_EP_CTRL, "ep %p: Destroying EFA direct op entry pool\n", ep);
	ofi_bufpool_destroy(ep->ope_pool);
	ep->ope_pool = NULL;
}

struct efa_direct_ope *efa_direct_ope_alloc(struct efa_base_ep *base_ep,
					    struct efa_context *context,
					    const struct fi_msg *msg,
					    const struct fi_msg_rma *msg_rma)
{
	struct efa_direct_ep *ep = container_of(base_ep, struct efa_direct_ep, base_ep);
	struct efa_direct_ope *direct_ope;
	const struct iovec *msg_iov = msg ? msg->msg_iov : msg_rma->msg_iov;
	void **desc = msg ? msg->desc : msg_rma->desc;
	size_t iov_count = msg ? msg->iov_count : msg_rma->iov_count;
	void *op_context = msg ? msg->context : msg_rma->context;
	uint64_t data = msg ? msg->data : msg_rma->data;
	size_t i;

	if (!ep->ope_pool)
		return NULL;

	direct_ope = ofi_buf_alloc(ep->ope_pool);
	if (OFI_UNLIKELY(!direct_ope)) {
		EFA_WARN(FI_LOG_EP_DATA,
			 "Failed to allocate EFA direct OPE\n");
		return NULL;
	}

	direct_ope->context = context;
	direct_ope->cq_entry.op_context = op_context;
	direct_ope->cq_entry.flags = context ? context->completion_flags : 0;
	direct_ope->cq_entry.len = ofi_total_iov_len(msg_iov, iov_count);
	direct_ope->cq_entry.buf = iov_count ? msg_iov[0].iov_base : NULL;
	direct_ope->cq_entry.data = data;
	direct_ope->cq_entry.tag = 0;
	direct_ope->iov_count = iov_count;
	if (desc) {
		for (i = 0; i < iov_count; i++)
			direct_ope->desc[i] = desc[i];
	}

	ofi_genlock_lock(&base_ep->domain->util_domain.lock);
	dlist_insert_tail(&direct_ope->entry, &ep->ope_list);
	ofi_genlock_unlock(&base_ep->domain->util_domain.lock);

	return direct_ope;
}

void efa_direct_ope_release(struct efa_base_ep *base_ep,
				  struct efa_direct_ope *direct_ope)
{
	struct efa_direct_ep *ep;

	if (!direct_ope || !base_ep)
		return;

	ep = container_of(base_ep, struct efa_direct_ep, base_ep);
	if (!ep->ope_pool)
		return;

	ofi_genlock_lock(&base_ep->domain->util_domain.lock);
	dlist_remove(&direct_ope->entry);
	ofi_genlock_unlock(&base_ep->domain->util_domain.lock);
	ofi_buf_free(direct_ope);
}
