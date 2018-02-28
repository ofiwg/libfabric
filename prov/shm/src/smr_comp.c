/*
 * Copyright (c) 2013-2018 Intel Corporation. All rights reserved
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
#include "smr.h"


int smr_tx_comp(struct smr_ep *ep, void *context, uint64_t flags, uint64_t err)
{
	struct fi_cq_tagged_entry *comp;
	struct util_cq_err_entry *entry;

	comp = ofi_cirque_tail(ep->util_ep.tx_cq->cirq);
	if (err) {
		if (!(entry = calloc(1, sizeof(*entry))))
			return -FI_ENOMEM;
		entry->err_entry.op_context = context;
		entry->err_entry.flags = flags;
		entry->err_entry.err = err;
		entry->err_entry.prov_errno = -err;
		slist_insert_tail(&entry->list_entry,
				  &ep->util_ep.tx_cq->err_list);
		comp->flags = UTIL_FLAG_ERROR;
	} else {
		comp->op_context = context;
		comp->flags = flags;
		comp->len = 0;
		comp->buf = NULL;
		comp->data = 0;
	}
	ofi_cirque_commit(ep->util_ep.tx_cq->cirq);
	return 0;
}

int smr_tx_comp_signal(struct smr_ep *ep, void *context, uint64_t flags,
		       uint64_t err)
{
	int ret;

	ret = smr_tx_comp(ep, context, flags, err);
	if (ret)
		return ret;
	ep->util_ep.tx_cq->wait->signal(ep->util_ep.tx_cq->wait);
	return 0;
}

int smr_rx_comp(struct smr_ep *ep, void *context, uint64_t flags, size_t len,
		void *buf, void *addr, uint64_t tag, uint64_t data,
		uint64_t err)
{
	struct fi_cq_tagged_entry *comp;
	struct util_cq_err_entry *entry;

	comp = ofi_cirque_tail(ep->util_ep.rx_cq->cirq);
	if (err) {
		if (!(entry = calloc(1, sizeof(*entry))))
			return -FI_ENOMEM;
		entry->err_entry.op_context = context;
		entry->err_entry.flags = flags;
		entry->err_entry.tag = tag;
		entry->err_entry.err = err;
		entry->err_entry.prov_errno = -err;
		slist_insert_tail(&entry->list_entry,
				  &ep->util_ep.rx_cq->err_list);
		comp->flags = UTIL_FLAG_ERROR;
	} else {
		comp->op_context = context;
		comp->flags = flags;
		comp->len = len;
		comp->buf = buf;
		comp->data = data;
		comp->tag = tag;
	}
	ofi_cirque_commit(ep->util_ep.rx_cq->cirq);
	return 0;
}

int smr_rx_src_comp(struct smr_ep *ep, void *context, uint64_t flags,
		    size_t len, void *buf, void *addr, uint64_t tag,
		    uint64_t data, uint64_t err)
{
	ep->util_ep.rx_cq->src[ofi_cirque_windex(ep->util_ep.rx_cq->cirq)] =
		(uint32_t) (uintptr_t) addr;
	return smr_rx_comp(ep, context, flags, len, buf, addr, tag, data, err);
}

int smr_rx_comp_signal(struct smr_ep *ep, void *context, uint64_t flags,
		       size_t len, void *buf, void *addr, uint64_t tag,
		       uint64_t data, uint64_t err)
{
	int ret;

	ret = smr_rx_comp(ep, context, flags, len, buf, addr, tag, data, err);
	if (ret)
		return ret;
	ep->util_ep.rx_cq->wait->signal(ep->util_ep.rx_cq->wait);
	return 0;
}

int smr_rx_src_comp_signal(struct smr_ep *ep, void *context, uint64_t flags,
			   size_t len, void *buf, void *addr, uint64_t tag,
			   uint64_t data, uint64_t err)
{
	int ret;

	ret = smr_rx_src_comp(ep, context, flags, len, buf, addr, tag, data, err);
	if (ret)
		return ret;
	ep->util_ep.rx_cq->wait->signal(ep->util_ep.rx_cq->wait);
	return 0;

}

static const uint64_t smr_tx_flags[] = {
	[ofi_op_msg] = FI_SEND,
	[ofi_op_tagged] = FI_SEND | FI_TAGGED,
	[ofi_op_read_req] = FI_RMA | FI_READ,
	[ofi_op_write] = FI_RMA | FI_WRITE,
	[ofi_op_atomic] = FI_ATOMIC | FI_WRITE,
	[ofi_op_atomic_fetch] = FI_ATOMIC | FI_WRITE | FI_READ,
	[ofi_op_atomic_compare] = FI_ATOMIC | FI_WRITE | FI_READ,
};

uint64_t smr_tx_comp_flags(uint32_t op)
{
	return smr_tx_flags[op];
}

static const uint64_t smr_rx_flags[] = {
	[ofi_op_msg] = FI_RECV,
	[ofi_op_tagged] = FI_RECV | FI_TAGGED,
	[ofi_op_read_req] = FI_RMA | FI_REMOTE_READ,
	[ofi_op_write] = FI_RMA | FI_REMOTE_WRITE,
	[ofi_op_atomic] = FI_ATOMIC | FI_REMOTE_WRITE,
	[ofi_op_atomic_fetch] = FI_ATOMIC | FI_REMOTE_WRITE | FI_REMOTE_READ,
	[ofi_op_atomic_compare] = FI_ATOMIC | FI_REMOTE_WRITE | FI_REMOTE_READ,
};

uint64_t smr_rx_comp_flags(uint32_t op, uint16_t op_flags)
{
	uint64_t flags;

	flags = smr_rx_flags[op];

	if (op_flags & SMR_REMOTE_CQ_DATA)
		flags |= FI_REMOTE_CQ_DATA;

	return flags;
}

static const uint64_t smr_mr_flags[] = {
	[ofi_op_msg] = FI_RECV,
	[ofi_op_tagged] = FI_RECV,
	[ofi_op_read_req] = FI_REMOTE_READ,
	[ofi_op_write] = FI_REMOTE_WRITE,
	[ofi_op_atomic] = FI_REMOTE_WRITE,
	[ofi_op_atomic_fetch] =  FI_REMOTE_WRITE | FI_REMOTE_READ,
	[ofi_op_atomic_compare] = FI_REMOTE_WRITE | FI_REMOTE_READ,
};

uint64_t smr_mr_reg_flags(uint32_t op, uint16_t atomic_op)
{
	if (atomic_op == FI_ATOMIC_READ)
		return FI_REMOTE_READ;

	return smr_mr_flags[op];
}
