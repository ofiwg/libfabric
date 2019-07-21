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

char *rxr_rma_read_hdr(struct rxr_ep *ep,
		       struct rxr_rx_entry *rx_entry,
		       struct rxr_pkt_entry *pkt_entry,
		       char *rma_hdr)
{
	uint32_t rma_access;
	struct fi_rma_iov *rma_iov = NULL;
	struct rxr_rts_hdr *rts_hdr;
	int ret;

	rma_iov = (struct fi_rma_iov *)rma_hdr;
	rts_hdr = rxr_get_rts_hdr(pkt_entry->pkt);
	if (rts_hdr->flags & RXR_READ_REQ) {
		rma_access = FI_SEND;
		rx_entry->cq_entry.flags |= (FI_RMA | FI_READ);
	} else {
		assert(rts_hdr->flags | RXR_WRITE);
		rma_access = FI_RECV;
		rx_entry->cq_entry.flags |= (FI_RMA | FI_WRITE);
	}

	assert(rx_entry->iov_count == 0);

	rx_entry->iov_count = rts_hdr->rma_iov_count;
	ret = rxr_rma_verified_copy_iov(ep, rma_iov, rts_hdr->rma_iov_count,
					rma_access, rx_entry->iov);
	if (ret) {
		FI_WARN(&rxr_prov, FI_LOG_CQ, "RMA address verify failed!\n");
		rxr_cq_handle_cq_error(ep, -FI_EIO);
	}

	rx_entry->cq_entry.len = ofi_total_iov_len(&rx_entry->iov[0],
						   rx_entry->iov_count);
	rx_entry->cq_entry.buf = rx_entry->iov[0].iov_base;
	return rma_hdr + rts_hdr->rma_iov_count * sizeof(struct fi_rma_iov);
}

int rxr_rma_process_write_rts(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_rx_entry *rx_entry;
	struct rxr_rts_hdr *rts_hdr;
	uint64_t tag = ~0;
	char *rma_hdr;
	char *data;
	size_t data_size;

	/*
	 * rma is one sided operation, match is not expected
	 * we need to create a rx entry upon receiving a rts
	 */
	rx_entry = rxr_ep_get_rx_entry(ep, NULL, 0, tag, 0, NULL, pkt_entry->addr, ofi_op_write, 0);
	if (OFI_UNLIKELY(!rx_entry)) {
		FI_WARN(&rxr_prov, FI_LOG_CQ,
			"RX entries exhausted.\n");
		rxr_eq_write_error(ep, FI_ENOBUFS, -FI_ENOBUFS);
		return -FI_ENOBUFS;
	}

	rx_entry->bytes_done = 0;

	rts_hdr = rxr_get_rts_hdr(pkt_entry->pkt);
	rma_hdr = rxr_cq_read_rts_hdr(ep, rx_entry, pkt_entry);
	data = rxr_rma_read_hdr(ep, rx_entry, pkt_entry, rma_hdr);
	data_size = rxr_get_rts_data_size(ep, rts_hdr);
	return rxr_cq_handle_rts_with_data(ep, rx_entry,
					   pkt_entry, data,
					   data_size);
}

int rxr_rma_process_read_rts(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_rx_entry *rx_entry;
	struct rxr_tx_entry *tx_entry;
	uint64_t tag = ~0;
	int ret = 0;
	char *rma_hdr;
	struct rxr_rma_read_hdr *rma_read_hdr;
	/*
	 * rma is one sided operation, match is not expected
	 * we need to create a rx entry upon receiving a rts
	 */
	rx_entry = rxr_ep_get_rx_entry(ep, NULL, 0, tag, 0, NULL, pkt_entry->addr, ofi_op_read_rsp, 0);
	if (OFI_UNLIKELY(!rx_entry)) {
		FI_WARN(&rxr_prov, FI_LOG_CQ,
			"RX entries exhausted.\n");
		rxr_eq_write_error(ep, FI_ENOBUFS, -FI_ENOBUFS);
		return -FI_ENOBUFS;
	}

	rx_entry->bytes_done = 0;

	rma_hdr = (char *)rxr_cq_read_rts_hdr(ep, rx_entry, pkt_entry);
	rma_read_hdr = (struct rxr_rma_read_hdr *)rxr_rma_read_hdr(ep, rx_entry, pkt_entry, rma_hdr);

	rx_entry->rma_initiator_rx_id = rma_read_hdr->rma_initiator_rx_id;
	rx_entry->window = rma_read_hdr->window;
	assert(rx_entry->window > 0);

	tx_entry = rxr_readrsp_tx_entry_init(ep, rx_entry);
	assert(tx_entry);
	/* the only difference between a read response packet and
	 * a data packet is that read response packet has remote EP tx_id
	 * which initiator EP rx_entry need to send CTS back
	 */

	ret = rxr_ep_post_readrsp(ep, tx_entry);
	if (!ret) {
		tx_entry->state = RXR_TX_SENT_READRSP;
		if (tx_entry->bytes_sent < tx_entry->total_len) {
			/* as long as read response packet has been sent,
			 * data packets are ready to be sent. it is OK that
			 * data packets arrive before read response packet,
			 * because tx_id is needed by the initator EP in order
			 * to send CTS, which will not occur until
			 * all data packets in current window are received, which
			 * include the data in the read response packet.
			 */
			dlist_insert_tail(&tx_entry->entry, &ep->tx_pending_list);
			tx_entry->state = RXR_TX_SEND;
		}
	} else if (ret == -FI_EAGAIN) {
		dlist_insert_tail(&tx_entry->queued_entry, &ep->tx_entry_queued_list);
		tx_entry->state = RXR_TX_QUEUED_READRSP;
		ret = 0;
	} else {
		if (rxr_cq_handle_tx_error(ep, tx_entry, ret))
			assert(0 && "failed to write err cq entry");
	}

	rx_entry->state = RXR_RX_WAIT_READ_FINISH;
	rxr_release_rx_pkt_entry(ep, pkt_entry);
	return ret;
}

/* Upon receiving a read request, Remote EP call this function to create
 * a tx entry for sending data back.
 */
struct rxr_tx_entry *rxr_readrsp_tx_entry_init(struct rxr_ep *rxr_ep,
					       struct rxr_rx_entry *rx_entry)
{
	struct rxr_tx_entry *tx_entry;
	struct fi_msg msg;

	tx_entry = ofi_buf_alloc(rxr_ep->readrsp_tx_entry_pool);
	if (OFI_UNLIKELY(!tx_entry)) {
		FI_WARN(&rxr_prov, FI_LOG_EP_CTRL, "Read Response TX entries exhausted.\n");
		return NULL;
	}

	assert(tx_entry);
#if ENABLE_DEBUG
	dlist_insert_tail(&tx_entry->tx_entry_entry, &rxr_ep->tx_entry_list);
#endif

	msg.msg_iov = rx_entry->iov;
	msg.iov_count = rx_entry->iov_count;
	msg.addr = rx_entry->addr;
	msg.desc = NULL;
	/*
	 * this tx_entry works similar to a send tx_entry thus its op was
	 * set to ofi_op_msg. Note this tx_entry will not write a completion
	 */
	rxr_generic_tx_entry_init(rxr_ep, tx_entry, &msg, 0, NULL, 0,
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

ssize_t rxr_generic_rma(struct fid_ep *ep, const struct fi_msg_rma *rma_msg,
			uint32_t op, uint64_t flags)
{
	struct fi_msg msg;

	msg.msg_iov = rma_msg->msg_iov;
	msg.desc = rma_msg->desc;
	msg.iov_count = rma_msg->iov_count;
	msg.addr = rma_msg->addr;
	msg.context = rma_msg->context;
	msg.data = rma_msg->data;

	assert(rma_msg->iov_count <= RXR_IOV_LIMIT &&
	       rma_msg->rma_iov_count <= RXR_IOV_LIMIT);

	return rxr_tx(ep, &msg, 0, rma_msg->rma_iov, rma_msg->rma_iov_count,
		      op, flags);
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
	return rxr_generic_rma(ep, msg, ofi_op_read_req, flags);
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

	return rxr_generic_rma(ep, msg, ofi_op_write,
			      (msg->data == 0) ? 0 : FI_REMOTE_CQ_DATA);
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
	struct fi_msg_rma msg;
	struct iovec iov;
	struct fi_rma_iov rma_iov;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len  = len;
	rma_iov.key = key;

	memset(&msg, 0, sizeof(msg));
	msg.msg_iov = &iov;
	msg.iov_count = 1;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.addr = dest_addr;

	return rxr_generic_rma(ep, &msg, ofi_op_write, FI_INJECT |
			       RXR_NO_COMPLETION);
}

ssize_t rxr_inject_data(struct fid_ep *ep, const void *buf, size_t len,
			uint64_t data, fi_addr_t dest_addr, uint64_t addr,
			uint64_t key)
{
	struct fi_msg_rma msg;
	struct iovec iov;
	struct fi_rma_iov rma_iov;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len  = len;
	rma_iov.key = key;

	memset(&msg, 0, sizeof(msg));
	msg.msg_iov = &iov;
	msg.iov_count = 1;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.addr = dest_addr;
	msg.data = data;

	return rxr_generic_rma(ep, &msg, ofi_op_write, FI_INJECT |
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
