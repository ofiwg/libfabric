/*
 * Copyright (c) 2013-2017 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "ofi_util.h"
#include "rxd.h"

#define RXD_FLAG(flag, mask) (((flag) & (mask)) == (mask))

int rxd_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		  struct fid_cntr **cntr_fid, void *context)
{
	int ret;
	struct util_cntr *cntr;

	cntr = calloc(1, sizeof(*cntr));
	if (!cntr)
		return -FI_ENOMEM;

	ret = ofi_cntr_init(&rxd_prov, domain, attr, cntr,
			    &ofi_cntr_progress, context);
	if (ret)
		goto free;

	*cntr_fid = &cntr->cntr_fid;
	return FI_SUCCESS;

free:
	free(cntr);
	return ret;
}

void rxd_cntr_report_tx_comp(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry)
{
        struct util_cntr *cntr;

	switch (tx_entry->op_type) {
	case RXD_TX_MSG:
	case RXD_TX_TAG:
		cntr = ep->util_ep.tx_cntr;
		break;
	case RXD_TX_WRITE:
		cntr = ep->util_ep.wr_cntr;
		break;
	case RXD_TX_READ_REQ:
		cntr = ep->util_ep.rem_rd_cntr;
		break;
	case RXD_TX_READ_RSP:
		return;
	default:
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "invalid op type\n");
		return;
	}

	if (cntr)
		cntr->cntr_fid.ops->add(&cntr->cntr_fid, 1);
}

void rxd_cntr_report_error(struct rxd_ep *ep, struct fi_cq_err_entry *err)
{
        struct util_cntr *cntr;

	cntr = RXD_FLAG(err->flags, (FI_WRITE)) ? ep->util_ep.wr_cntr :
	       RXD_FLAG(err->flags, (FI_ATOMIC)) ? ep->util_ep.wr_cntr :
	       RXD_FLAG(err->flags, (FI_READ)) ? ep->util_ep.rd_cntr :
	       RXD_FLAG(err->flags, (FI_SEND)) ? ep->util_ep.tx_cntr :
	       RXD_FLAG(err->flags, (FI_RECV)) ? ep->util_ep.rx_cntr :
	       NULL;

	if (cntr)
		cntr->cntr_fid.ops->adderr(&cntr->cntr_fid, 1);
}



