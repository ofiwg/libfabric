/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation. All rights reserved.
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include "config.h"

#include <stdlib.h>
#include <string.h>

#include "cxi_prov.h"

#define CXI_LOG_DBG(...) _CXI_LOG_DBG(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXI_LOG_ERROR(...) _CXI_LOG_ERROR(FI_LOG_EP_CTRL, __VA_ARGS__)

struct cxi_rx_ctx *cxi_rx_ctx_alloc(const struct fi_rx_attr *attr,
				    void *context, int use_shared)
{
	struct cxi_rx_ctx *rx_ctx;

	rx_ctx = calloc(1, sizeof(*rx_ctx));
	if (!rx_ctx)
		return NULL;

	dlist_init(&rx_ctx->cq_entry);
	dlist_init(&rx_ctx->ep_list);
	fastlock_init(&rx_ctx->lock);

	rx_ctx->ctx.fid.fclass = FI_CLASS_RX_CTX;
	rx_ctx->ctx.fid.context = context;
	rx_ctx->num_left = attr->size;
	rx_ctx->attr = *attr;
	rx_ctx->use_shared = use_shared;

	return rx_ctx;
}

void cxi_rx_ctx_free(struct cxi_rx_ctx *rx_ctx)
{
	fastlock_destroy(&rx_ctx->lock);
	free(rx_ctx);
}

static struct cxi_tx_ctx *cxi_tx_context_alloc(const struct fi_tx_attr *attr,
					       void *context, int use_shared,
					       size_t fclass)
{
	struct cxi_tx_ctx *tx_ctx;

	tx_ctx = calloc(sizeof(*tx_ctx), 1);
	if (!tx_ctx)
		return NULL;

	dlist_init(&tx_ctx->cq_entry);
	dlist_init(&tx_ctx->ep_list);
	fastlock_init(&tx_ctx->lock);

	switch (fclass) {
	case FI_CLASS_TX_CTX:
		tx_ctx->fid.ctx.fid.fclass = FI_CLASS_TX_CTX;
		tx_ctx->fid.ctx.fid.context = context;
		tx_ctx->fclass = FI_CLASS_TX_CTX;
		tx_ctx->use_shared = use_shared;
		break;
	case FI_CLASS_STX_CTX:
		tx_ctx->fid.stx.fid.fclass = FI_CLASS_STX_CTX;
		tx_ctx->fid.stx.fid.context = context;
		tx_ctx->fclass = FI_CLASS_STX_CTX;
		break;
	default:
		goto err;
	}
	tx_ctx->attr = *attr;
	tx_ctx->attr.op_flags |= FI_TRANSMIT_COMPLETE;

	return tx_ctx;

err:
	fastlock_destroy(&tx_ctx->lock);
	free(tx_ctx);
	return NULL;
}


struct cxi_tx_ctx *cxi_tx_ctx_alloc(const struct fi_tx_attr *attr,
				    void *context, int use_shared)
{
	return cxi_tx_context_alloc(attr, context, use_shared, FI_CLASS_TX_CTX);
}

struct cxi_tx_ctx *cxi_stx_ctx_alloc(const struct fi_tx_attr *attr,
					void *context)
{
	return cxi_tx_context_alloc(attr, context, 0, FI_CLASS_STX_CTX);
}

void cxi_tx_ctx_free(struct cxi_tx_ctx *tx_ctx)
{
	fastlock_destroy(&tx_ctx->lock);
	free(tx_ctx);
}
