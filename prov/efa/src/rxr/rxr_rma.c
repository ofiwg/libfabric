/*
 * Copyright (c) 2019 Amazon.com, Inc. or its affiliates.
 * All rights reserved.
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
#include <ofi_mem.h>
#include <ofi_iov.h>
#include "rxr.h"
#include "rxr_rma.h"

int rxr_rma_verified_copy_iov(struct rxr_ep *ep, struct fi_rma_iov *rma,
			      size_t count, uint32_t flags, struct iovec *iov)
{
	struct util_domain *util_domain;
	int i, ret;

	util_domain = &rxr_ep_domain(ep)->util_domain;

	for (i = 0; i < count; i++) {
		ret = ofi_mr_verify(&util_domain->mr_map,
				    rma[i].len,
				    (uintptr_t *)(&rma[i].addr),
				    rma[i].key,
				    flags);
		if (ret) {
			FI_WARN(&rxr_prov, FI_LOG_EP_CTRL,
				"MR verification failed (%s)\n",
				fi_strerror(-ret));
			return -FI_EACCES;
		}

		iov[i].iov_base = (void *)rma[i].addr;
		iov[i].iov_len = rma[i].len;
	}
	return 0;
}

/* Upon receiving a read request, Remote EP call this function to create
 * a tx entry for sending data back.
 */
struct rxr_tx_entry *rxr_readrsp_tx_entry_init(struct rxr_ep *rxr_ep,
					       struct rxr_rx_entry *rx_entry)
{
	struct rxr_tx_entry *tx_entry;

	tx_entry = ofi_buf_alloc(rxr_ep->readrsp_tx_entry_pool);
	if (OFI_UNLIKELY(!tx_entry)) {
		FI_WARN(&rxr_prov, FI_LOG_EP_CTRL, "Read Response TX entries exhausted.\n");
		return NULL;
	}

	assert(tx_entry);
#if ENABLE_DEBUG
	dlist_insert_tail(&tx_entry->tx_entry_entry, &rxr_ep->tx_entry_list);
#endif

	/*
	 * this tx_entry works similar to a send tx_entry thus its op was
	 * set to ofi_op_msg. Note this tx_entry will not write a completion
	 */
	rxr_generic_tx_entry_init(tx_entry, rx_entry->iov, rx_entry->iov_count,
				  NULL, 0, rx_entry->addr, 0, 0, NULL,
				  ofi_op_msg, 0);

	tx_entry->cq_entry.flags |= FI_READ;
	/* rma_loc_rx_id is for later retrieve of rx_entry
	 * to write rx_completion
	 */
	tx_entry->rma_loc_rx_id = rx_entry->rx_id;

	/* the following is essentially handle CTS */
	tx_entry->rx_id = rx_entry->rma_initiator_rx_id;
	tx_entry->window = rx_entry->window;

	/* this tx_entry does not send rts
	 * therefore should not increase msg_id
	 */
	tx_entry->msg_id = 0;
	return tx_entry;
}

ssize_t rxr_generic_rma(struct fid_ep *ep,
			const struct iovec *iov, size_t iov_count,
			const struct fi_rma_iov *rma_iov, size_t rma_iov_count,
			fi_addr_t addr, uint64_t data, void *context, uint32_t op,
			uint64_t flags)
{
	assert(iov_count <= RXR_IOV_LIMIT && rma_iov_count <= RXR_IOV_LIMIT);
	int tag = 0; // RMA is not tagged

	return rxr_tx(ep, iov, iov_count, rma_iov, rma_iov_count, addr,
		      tag, data, context, op, flags);
}

ssize_t rxr_read(struct fid_ep *ep, void *buf, size_t len, void *desc,
		 fi_addr_t src_addr, uint64_t addr, uint64_t key,
		 void *context)
{
	struct iovec iov;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;
	return rxr_readv(ep, &iov, &desc, 1, src_addr, addr, key, context);
}

ssize_t rxr_readv(struct fid_ep *ep, const struct iovec *iov, void **desc,
		  size_t iov_count, fi_addr_t src_addr, uint64_t addr,
		  uint64_t key, void *context)
{
	struct fi_rma_iov rma_iov;
	struct fi_msg_rma msg;

	rma_iov.addr = addr;
	rma_iov.len = ofi_total_iov_len(iov, iov_count);
	rma_iov.key = key;

	memset(&msg, 0, sizeof(msg));
	msg.msg_iov = iov;
	msg.desc = desc;
	msg.iov_count = iov_count;
	msg.addr = src_addr;
	msg.context = context;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;

	return rxr_readmsg(ep, &msg, 0);
}

ssize_t rxr_readmsg(struct fid_ep *ep, const struct fi_msg_rma *msg, uint64_t flags)
{
	return rxr_generic_rma(ep, msg->msg_iov, msg->iov_count,
			       msg->rma_iov, msg->rma_iov_count,
			       msg->addr, msg->data, msg->context,
			       ofi_op_read_req, flags);
}

ssize_t rxr_write(struct fid_ep *ep, const void *buf, size_t len, void *desc,
		  fi_addr_t dest_addr, uint64_t addr, uint64_t key,
		  void *context)
{
	struct iovec iov;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;
	return rxr_writev(ep, &iov, &desc, 1, dest_addr, addr, key, context);
}

ssize_t rxr_writev(struct fid_ep *ep, const struct iovec *iov, void **desc,
		   size_t iov_count, fi_addr_t dest_addr, uint64_t addr,
		   uint64_t key, void *context)
{
	struct fi_rma_iov rma_iov;
	struct fi_msg_rma msg;

	rma_iov.addr = addr;
	rma_iov.len = ofi_total_iov_len(iov, iov_count);
	rma_iov.key = key;

	memset(&msg, 0, sizeof(msg));
	msg.msg_iov = iov;
	msg.desc = desc;
	msg.iov_count = iov_count;
	msg.addr = dest_addr;
	msg.context = context;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;

	return rxr_writemsg(ep, &msg, 0);
}

ssize_t rxr_writemsg(struct fid_ep *ep, const struct fi_msg_rma *msg,
		     uint64_t flags)
{
	ssize_t ret = 0;

	if (msg->data == 0) {
		ret = rxr_generic_rma(ep, msg->msg_iov, msg->iov_count,
				      msg->rma_iov, msg->rma_iov_count,
				      msg->addr, 0, NULL, ofi_op_write, 0);
	} else {
		ret = rxr_generic_rma(ep, msg->msg_iov, msg->iov_count,
				      msg->rma_iov, msg->rma_iov_count,
				      msg->addr, msg->data,
				      msg->context, ofi_op_write,
				      FI_REMOTE_CQ_DATA);
	}

	return ret;
}

ssize_t rxr_writedata(struct fid_ep *ep, const void *buf, size_t len,
		      void *desc, uint64_t data, fi_addr_t dest_addr,
		      uint64_t addr, uint64_t key, void *context)
{
	struct iovec iov;
	struct fi_rma_iov rma_iov;
	struct fi_msg_rma msg;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len  = len;
	rma_iov.key = key;

	memset(&msg, 0, sizeof(msg));
	msg.msg_iov = &iov;
	msg.desc = desc;
	msg.iov_count = 1;
	msg.addr = dest_addr;
	msg.context = context;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.data = data;

	return rxr_writemsg(ep, &msg, 0);
}

ssize_t rxr_inject(struct fid_ep *ep, const void *buf, size_t len,
		   fi_addr_t dest_addr, uint64_t addr, uint64_t key)
{
	struct iovec iov;
	struct fi_rma_iov rma_iov;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len  = len;
	rma_iov.key = key;
	return rxr_generic_rma(ep, &iov, 1, &rma_iov, 1, dest_addr,
			       0, NULL, ofi_op_write, FI_INJECT |
			       RXR_NO_COMPLETION);
}

ssize_t rxr_inject_data(struct fid_ep *ep, const void *buf, size_t len,
			uint64_t data, fi_addr_t dest_addr, uint64_t addr,
			uint64_t key)
{
	struct iovec iov;
	struct fi_rma_iov rma_iov;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len  = len;
	rma_iov.key = key;
	return rxr_generic_rma(ep, &iov, 1, &rma_iov, 1, dest_addr,
			       data, NULL, ofi_op_write, FI_INJECT |
			       RXR_NO_COMPLETION | FI_REMOTE_CQ_DATA);
}

struct fi_ops_rma rxr_ops_rma = {
	.size = sizeof(struct fi_ops_rma),
	.read = rxr_read,
	.readv = rxr_readv,
	.readmsg = rxr_readmsg,
	.write = rxr_write,
	.writev = rxr_writev,
	.writemsg = rxr_writemsg,
	.inject = rxr_inject,
	.writedata = rxr_writedata,
	.injectdata = rxr_inject_data,
};
