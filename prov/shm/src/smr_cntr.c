/*
 * Copyright (c) 2019 Intel Corporation. All rights reserved.
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

#include "smr.h"

int smr_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		  struct fid_cntr **cntr_fid, void *context)
{
	int ret;
	struct util_cntr *cntr;

	if (attr->wait_obj != FI_WAIT_NONE) {
		FI_INFO(&smr_prov, FI_LOG_CNTR, "cntr wait not yet supported\n");
		return -FI_ENOSYS;
	}

	cntr = calloc(1, sizeof(*cntr));
	if (!cntr)
		return -FI_ENOMEM;

	ret = ofi_cntr_init(&smr_prov, domain, attr, cntr,
			    &ofi_cntr_progress, context);
	if (ret)
		goto free;

	*cntr_fid = &cntr->cntr_fid;
	return FI_SUCCESS;

free:
	free(cntr);
	return ret;
}

void smr_cntr_report_tx_comp(struct smr_ep *ep, uint32_t op)
{
	uint64_t flags = ofi_tx_cq_flags(op) &
			 (FI_SEND | FI_WRITE | FI_READ);

	assert(ofi_lsb(flags) == ofi_msb(flags));
	ofi_ep_cntr_inc_funcs[flags](&ep->util_ep);
}

void smr_cntr_report_rx_comp(struct smr_ep *ep, uint32_t op)
{
	uint64_t flags = ofi_rx_cq_flags(op) &
			(FI_RECV | FI_REMOTE_WRITE | FI_REMOTE_READ);

	assert(ofi_lsb(flags) == ofi_msb(flags));
	ofi_ep_cntr_inc_funcs[flags](&ep->util_ep);
}
