/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include "cxi_prov.h"

#include <ofi_util.h>

#define CXI_LOG_DBG(...) _CXI_LOG_DBG(FI_LOG_EP_DATA, __VA_ARGS__)
#define CXI_LOG_ERROR(...) _CXI_LOG_ERROR(FI_LOG_EP_DATA, __VA_ARGS__)

void cxi_cntr_add_tx_ctx(struct cxi_cntr *cntr, struct cxi_tx_ctx *tx_ctx)
{
	int ret;
	struct fid *fid = &tx_ctx->fid.ctx.fid;

	ret = fid_list_insert(&cntr->tx_list, &cntr->list_lock, fid);
	if (ret)
		CXI_LOG_ERROR("Error in adding ctx to progress list\n");
	else
		ofi_atomic_inc32(&cntr->ref);
}

void cxi_cntr_remove_tx_ctx(struct cxi_cntr *cntr, struct cxi_tx_ctx *tx_ctx)
{
	struct fid *fid = &tx_ctx->fid.ctx.fid;

	fid_list_remove(&cntr->tx_list, &cntr->list_lock, fid);
	ofi_atomic_dec32(&cntr->ref);
}

void cxi_cntr_add_rx_ctx(struct cxi_cntr *cntr, struct cxi_rx_ctx *rx_ctx)
{
	int ret;
	struct fid *fid = &rx_ctx->ctx.fid;

	ret = fid_list_insert(&cntr->rx_list, &cntr->list_lock, fid);
	if (ret)
		CXI_LOG_ERROR("Error in adding ctx to progress list\n");
	else
		ofi_atomic_inc32(&cntr->ref);
}

void cxi_cntr_remove_rx_ctx(struct cxi_cntr *cntr, struct cxi_rx_ctx *rx_ctx)
{
	struct fid *fid = &rx_ctx->ctx.fid;

	fid_list_remove(&cntr->rx_list, &cntr->list_lock, fid);
	ofi_atomic_dec32(&cntr->ref);
}

