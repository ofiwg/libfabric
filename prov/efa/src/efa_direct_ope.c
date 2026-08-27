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

static int efa_direct_ope_pool_create_one(struct ofi_bufpool **pool,
					  size_t size)
{
	int ret;

	ret = ofi_bufpool_create(pool, sizeof(struct efa_direct_ope),
				 EFA_RDM_BUFPOOL_ALIGNMENT, size, size,
				 OFI_BUFPOOL_NO_TRACK);
	if (ret) {
		EFA_WARN(FI_LOG_EP_CTRL,
			 "Failed to create EFA direct op entry pool: %s\n",
			 fi_strerror(-ret));
		return ret;
	}

	ret = ofi_bufpool_grow(*pool);
	if (ret) {
		ofi_bufpool_destroy(*pool);
		*pool = NULL;
	}

	return ret;
}

int efa_direct_ope_pool_create(struct efa_base_ep *base_ep)
{
	int ret;

	if (!efa_env.track_mr) {
		base_ep->txe_pool = NULL;
		base_ep->rxe_pool = NULL;
		return 0;
	}

	dlist_init(&base_ep->ope_list);

	/* Cap each pool with the same helper that sizes the QP for that
	 * direction, so a pool holds one entry per queue slot. */
	ret = efa_direct_ope_pool_create_one(
		&base_ep->txe_pool, efa_base_ep_get_tx_pool_size(base_ep));
	if (ret)
		return ret;

	ret = efa_direct_ope_pool_create_one(
		&base_ep->rxe_pool, efa_base_ep_get_rx_pool_size(base_ep));
	if (ret) {
		ofi_bufpool_destroy(base_ep->txe_pool);
		base_ep->txe_pool = NULL;
		return ret;
	}

	EFA_INFO(FI_LOG_EP_CTRL,
		 "ep %p: Created EFA direct op entry pools, tx cap %zu rx cap %zu\n",
		 base_ep, efa_base_ep_get_tx_pool_size(base_ep),
		 efa_base_ep_get_rx_pool_size(base_ep));

	return 0;
}

void efa_direct_ope_pool_destroy(struct efa_base_ep *base_ep)
{
	struct efa_direct_ope *direct_ope;
	struct dlist_entry *tmp;

	if (!base_ep->txe_pool)
		return;

	ofi_genlock_lock(&base_ep->domain->util_domain.lock);
	if (!dlist_empty(&base_ep->ope_list)) {
		dlist_foreach_container_safe(&base_ep->ope_list,
					     struct efa_direct_ope,
					     direct_ope, entry, tmp) {
			dlist_remove(&direct_ope->entry);
			ofi_buf_free(direct_ope);
		}
	}
	ofi_genlock_unlock(&base_ep->domain->util_domain.lock);

	EFA_INFO(FI_LOG_EP_CTRL, "ep %p: Destroying EFA direct op entry pool\n", base_ep);
	ofi_bufpool_destroy(base_ep->txe_pool);
	base_ep->txe_pool = NULL;
	ofi_bufpool_destroy(base_ep->rxe_pool);
	base_ep->rxe_pool = NULL;
}

static struct efa_direct_ope *
efa_direct_ope_alloc(struct efa_base_ep *base_ep, struct ofi_bufpool *pool,
		     struct efa_context *context, const struct iovec *msg_iov,
		     void **desc, size_t iov_count, void *op_context,
		     uint64_t data)
{
	struct efa_direct_ope *direct_ope;
	size_t i;

	if (!pool)
		return NULL;

	direct_ope = ofi_buf_alloc(pool);
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
	dlist_insert_tail(&direct_ope->entry, &base_ep->ope_list);
	ofi_genlock_unlock(&base_ep->domain->util_domain.lock);

	return direct_ope;
}

/**
 * @brief Allocate and record a tx operation entry
 *
 * Exactly one of @p msg and @p msg_rma describes the transfer, see the
 * declaration for the full parameter contract.
 */
struct efa_direct_ope *efa_direct_txe_alloc(struct efa_base_ep *base_ep,
					    struct efa_context *context,
					    const struct fi_msg *msg,
					    const struct fi_msg_rma *msg_rma)
{
	if (msg)
		return efa_direct_ope_alloc(base_ep, base_ep->txe_pool,
					    context, msg->msg_iov, msg->desc,
					    msg->iov_count, msg->context,
					    msg->data);

	assert(msg_rma);
	return efa_direct_ope_alloc(base_ep, base_ep->txe_pool, context,
				    msg_rma->msg_iov, msg_rma->desc,
				    msg_rma->iov_count, msg_rma->context,
				    msg_rma->data);
}

struct efa_direct_ope *efa_direct_rxe_alloc(struct efa_base_ep *base_ep,
					    struct efa_context *context,
					    const struct fi_msg *msg)
{
	return efa_direct_ope_alloc(base_ep, base_ep->rxe_pool, context,
				    msg->msg_iov, msg->desc, msg->iov_count,
				    msg->context, msg->data);
}

void efa_direct_ope_release(struct efa_base_ep *base_ep,
				  struct efa_direct_ope *direct_ope)
{
	if (!direct_ope || !base_ep)
		return;

	if (!base_ep->txe_pool)
		return;

	ofi_genlock_lock(&base_ep->domain->util_domain.lock);
	dlist_remove(&direct_ope->entry);
	ofi_genlock_unlock(&base_ep->domain->util_domain.lock);
	ofi_buf_free(direct_ope);
}
