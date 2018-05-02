
/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>

#include <ofi_list.h>
#include <ofi.h>

#include "cxi_prov.h"

#define CXI_LOG_DBG(...) _CXI_LOG_DBG(FI_LOG_CQ, __VA_ARGS__)
#define CXI_LOG_ERROR(...) _CXI_LOG_ERROR(FI_LOG_CQ, __VA_ARGS__)

void cxi_cq_add_tx_ctx(struct cxi_cq *cq, struct cxi_tx_ctx *tx_ctx)
{
	struct dlist_entry *entry;
	struct cxi_tx_ctx *curr_ctx;

	fastlock_acquire(&cq->list_lock);
	for (entry = cq->tx_list.next; entry != &cq->tx_list;
	     entry = entry->next) {
		curr_ctx = container_of(entry, struct cxi_tx_ctx, cq_entry);
		if (tx_ctx == curr_ctx)
			goto out;
	}
	dlist_insert_tail(&tx_ctx->cq_entry, &cq->tx_list);
	ofi_atomic_inc32(&cq->ref);
out:
	fastlock_release(&cq->list_lock);
}

void cxi_cq_remove_tx_ctx(struct cxi_cq *cq, struct cxi_tx_ctx *tx_ctx)
{
	fastlock_acquire(&cq->list_lock);
	dlist_remove(&tx_ctx->cq_entry);
	ofi_atomic_dec32(&cq->ref);
	fastlock_release(&cq->list_lock);
}

void cxi_cq_add_rx_ctx(struct cxi_cq *cq, struct cxi_rx_ctx *rx_ctx)
{
	struct dlist_entry *entry;
	struct cxi_rx_ctx *curr_ctx;

	fastlock_acquire(&cq->list_lock);

	for (entry = cq->rx_list.next; entry != &cq->rx_list;
	     entry = entry->next) {
		curr_ctx = container_of(entry, struct cxi_rx_ctx, cq_entry);
		if (rx_ctx == curr_ctx)
			goto out;
	}
	dlist_insert_tail(&rx_ctx->cq_entry, &cq->rx_list);
	ofi_atomic_inc32(&cq->ref);
out:
	fastlock_release(&cq->list_lock);
}

void cxi_cq_remove_rx_ctx(struct cxi_cq *cq, struct cxi_rx_ctx *rx_ctx)
{
	fastlock_acquire(&cq->list_lock);
	dlist_remove(&rx_ctx->cq_entry);
	ofi_atomic_dec32(&cq->ref);
	fastlock_release(&cq->list_lock);
}
