/*
 * Copyright (c) 2013-2018 Intel Corporation. All rights reserved
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#include <stdlib.h>
#include <string.h>
#include <sys/uio.h>

#include "ofi_iov.h"
#include "sm2.h"

static int sm2_tx_comp(struct sm2_ep *ep, void *context, uint32_t op)
{
	struct sm2_cq *sm2_cq;

	sm2_cq = container_of(ep->util_ep.tx_cq, struct sm2_cq, util_cq);

	return sm2_cq->peer_cq->owner_ops->write(sm2_cq->peer_cq, context,
					     ofi_tx_cq_flags(op),
					     0, NULL, 0, 0, FI_ADDR_NOTAVAIL);
}

int sm2_complete_tx(struct sm2_ep *ep, void *context, uint32_t op,
		    uint64_t flags)
{
	ofi_ep_tx_cntr_inc_func(&ep->util_ep, op);

	if (!(flags & FI_COMPLETION))
		return 0;

	return sm2_tx_comp(ep, context, op);
}

int sm2_write_err_comp(struct util_cq *cq, void *context,
		       uint64_t flags, uint64_t tag, uint64_t err)
{
	struct fi_cq_err_entry err_entry;
	struct sm2_cq *sm2_cq;

	sm2_cq = container_of(cq, struct sm2_cq, util_cq);

	memset(&err_entry, 0, sizeof err_entry);
	err_entry.op_context = context;
	err_entry.flags = flags;
	err_entry.tag = tag;
	err_entry.err = err;
	err_entry.prov_errno = -err;
	return sm2_cq->peer_cq->owner_ops->writeerr(sm2_cq->peer_cq, &err_entry);
}

int sm2_tx_comp_signal(struct sm2_ep *ep, void *context, uint32_t op)
{
	int ret;

	ret = sm2_tx_comp(ep, context, op);
	if (ret)
		return ret;

	ep->util_ep.tx_cq->wait->signal(ep->util_ep.tx_cq->wait);
	return 0;
}

int sm2_complete_rx(struct sm2_ep *ep, void *context, uint32_t op,
		    uint64_t flags, size_t len, void *buf, int64_t id,
		    uint64_t tag, uint64_t data)
{
	struct sm2_cq *cq;

	ofi_ep_rx_cntr_inc_func(&ep->util_ep, op);

	if (!(flags & (FI_REMOTE_CQ_DATA | FI_COMPLETION)))
		return 0;

	flags &= ~FI_COMPLETION;

	cq = container_of(ep->util_ep.rx_cq, struct sm2_cq, util_cq);
	return ep->rx_comp(cq, context, flags, len, buf,
			   id, tag, data);
}

int sm2_rx_comp(struct sm2_cq *cq, void *context, uint64_t flags, size_t len,
		void *buf, fi_addr_t fi_addr, uint64_t tag, uint64_t data)
{
	return cq->peer_cq->owner_ops->write(cq->peer_cq, context,
				flags, len, buf, data,
				tag, FI_ADDR_NOTAVAIL);
}

int sm2_rx_src_comp(struct sm2_cq *cq, void *context, uint64_t flags, size_t len,
		    void *buf, fi_addr_t fi_addr, uint64_t tag, uint64_t data)
{
	return cq->peer_cq->owner_ops->write(cq->peer_cq, context,
				flags, len, buf, data, tag, fi_addr);
}