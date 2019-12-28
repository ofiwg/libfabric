/*
 * Copyright (c) 2019-2020 Amazon.com, Inc. or its affiliates.
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

#include "efa.h"
#include "rxr.h"
#include "rxr_rma.h"
#include "rxr_cntr.h"
#include "rxr_rdma.h"

int rxr_locate_iov_pos(struct iovec *iov, int iov_count, size_t offset,
		       int *iov_idx, size_t *iov_offset)
{
	int i;
	size_t curoffset;

	curoffset = 0;
	for (i = 0; i < iov_count; ++i) {
		if (offset >= curoffset &&
		    offset < curoffset + iov[i].iov_len) {
			*iov_idx = i;
			*iov_offset = offset - curoffset;
			return 0;
		}

		curoffset += iov[i].iov_len;
	}

	return -1;
}

int rxr_locate_rma_iov_pos(struct fi_rma_iov *rma_iov, int rma_iov_count, size_t offset,
			   int *rma_iov_idx, size_t *rma_iov_offset)
{
	int i;
	size_t curoffset;

	curoffset = 0;
	for (i = 0; i < rma_iov_count; ++i) {
		if (offset >= curoffset &&
		    offset < curoffset + rma_iov[i].len) {
			*rma_iov_idx = i;
			*rma_iov_offset = offset - curoffset;
			return 0;
		}

		curoffset += rma_iov[i].len;
	}

	return -1;
}

size_t rxr_total_rma_iov_len(struct fi_rma_iov *rma_iov, int rma_iov_count)
{
	int i;
	size_t total_len;

	total_len = 0;
	for (i = 0; i < rma_iov_count; ++i)
		total_len += rma_iov[i].len;

	return total_len;
}

struct rxr_rdma_entry *rxr_rdma_alloc_entry(struct rxr_ep *ep, int entry_type, void *x_entry,
					    enum rxr_lower_ep_type lower_ep_type)
{
	struct rxr_tx_entry *tx_entry = NULL;
	struct rxr_rx_entry *rx_entry = NULL;
	struct rxr_rdma_entry *rdma_entry;
	int i, err;
	size_t total_iov_len, total_rma_iov_len;
	void **mr_desc;

	rdma_entry = ofi_buf_alloc(ep->rdma_entry_pool);
	if (OFI_UNLIKELY(!rdma_entry)) {
		FI_WARN(&rxr_prov, FI_LOG_EP_CTRL, "RDMA entries exhausted\n");
		return NULL;
	}

	rdma_entry->rdma_id = ofi_buf_index(rdma_entry);
	rdma_entry->state = RXR_RDMA_ENTRY_CREATED;
	rdma_entry->src_type = entry_type;

	if (entry_type == RXR_TX_ENTRY) {
		tx_entry = (struct rxr_tx_entry *)x_entry;
		assert(tx_entry->op == ofi_op_read_req);
		rdma_entry->src_id = tx_entry->tx_id;
		rdma_entry->addr = tx_entry->addr;

		rdma_entry->iov_count = tx_entry->iov_count;
		memcpy(rdma_entry->iov, tx_entry->iov, tx_entry->iov_count * sizeof(struct iovec));

		rdma_entry->rma_iov_count = tx_entry->rma_iov_count;
		memcpy(rdma_entry->rma_iov, tx_entry->rma_iov,
		       tx_entry->rma_iov_count * sizeof(struct fi_rma_iov));

		total_iov_len = ofi_total_iov_len(tx_entry->iov, tx_entry->iov_count);
		total_rma_iov_len = rxr_total_rma_iov_len(tx_entry->rma_iov, tx_entry->rma_iov_count);
		rdma_entry->total_len = MIN(total_iov_len, total_rma_iov_len);
		mr_desc = tx_entry->desc;
	} else {
		rx_entry = (struct rxr_rx_entry *)x_entry;
		assert(rx_entry->op == ofi_op_write || rx_entry->op == ofi_op_msg ||
		       rx_entry->op == ofi_op_tagged);

		rdma_entry->src_id = rx_entry->rx_id;
		rdma_entry->addr = rx_entry->addr;

		rdma_entry->iov_count = rx_entry->iov_count;
		memcpy(rdma_entry->iov, rx_entry->iov, rx_entry->iov_count * sizeof(struct iovec));

		rdma_entry->rma_iov_count = rx_entry->rma_iov_count;
		memcpy(rdma_entry->rma_iov, rx_entry->rma_iov,
		       rx_entry->rma_iov_count * sizeof(struct fi_rma_iov));

		mr_desc = NULL;
		total_iov_len = ofi_total_iov_len(rx_entry->iov, rx_entry->iov_count);
		total_rma_iov_len = rxr_total_rma_iov_len(rx_entry->rma_iov, rx_entry->rma_iov_count);
		rdma_entry->total_len = MIN(total_iov_len, total_rma_iov_len);
	}

	if (lower_ep_type == EFA_EP) {
		/* EFA provider need local buffer registration */
		for (i = 0; i < rdma_entry->iov_count; ++i) {
			if (mr_desc && mr_desc[i]) {
				rdma_entry->mr[i] = NULL;
				rdma_entry->mr_desc[i] = mr_desc[i];
			} else {
				err = fi_mr_reg(rxr_ep_domain(ep)->rdm_domain,
						rdma_entry->iov[i].iov_base, rdma_entry->iov[i].iov_len,
						FI_RECV, 0, 0, 0, &rdma_entry->mr[i], NULL);
				if (err) {
					FI_WARN(&rxr_prov, FI_LOG_MR, "Unable to register MR buf\n");
					return NULL;
				}

				rdma_entry->mr_desc[i] = fi_mr_desc(rdma_entry->mr[i]);
			}
		}
	} else {
		assert(lower_ep_type == SHM_EP);
		memset(rdma_entry->mr, 0, rdma_entry->iov_count * sizeof(struct fid_mr *));
		memset(rdma_entry->mr_desc, 0, rdma_entry->iov_count * sizeof(void *));
	}

	rdma_entry->lower_ep_type = lower_ep_type;
	rdma_entry->bytes_submitted = 0;
	rdma_entry->bytes_finished = 0;
	return rdma_entry;
}

void rxr_rdma_release_entry(struct rxr_ep *ep, struct rxr_rdma_entry *rdma_entry)
{
	int i, err;

	for (i = 0; i < rdma_entry->iov_count; ++i) {
		if (rdma_entry->mr[i]) {
			err = fi_close((struct fid *)rdma_entry->mr[i]);
			if (err) {
				FI_WARN(&rxr_prov, FI_LOG_MR, "Unable to close mr\n");
				rxr_rdma_handle_error(ep, rdma_entry, err);
			}
		}
	}

#ifdef ENABLE_EFA_POISONING
	rxr_poison_mem_region((uint32_t *)rdma_entry, sizeof(struct rxr_rdma_entry));
#endif
	rdma_entry->state = RXR_RDMA_ENTRY_FREE;
	ofi_buf_free(rdma_entry);
}

int rxr_rdma_post_read_or_queue(struct rxr_ep *ep, struct rxr_rdma_entry *rdma_entry)
{
	int ret;

	ret = rxr_rdma_post_read(ep, rdma_entry);
	if (ret == -FI_EAGAIN) {
		dlist_insert_tail(&rdma_entry->pending_entry, &ep->rdma_pending_list);
		rdma_entry->state = RXR_RDMA_ENTRY_PENDING;
		ret = 0;
	}

	return ret;
}

int rxr_rdma_post_read(struct rxr_ep *ep, struct rxr_rdma_entry *rdma_entry)
{
	int ret;
	int iov_idx = 0, rma_iov_idx = 0;
	void *iov_ptr, *rma_iov_ptr;
	struct rxr_pkt_entry *pkt_entry;
	size_t iov_offset = 0, rma_iov_offset = 0;
	size_t total_iov_len, total_rma_iov_len;
	size_t segsize, max_iov_segsize, max_rma_iov_segsize, max_rdma_size;
	struct fid_ep *lower_ep;
	fi_addr_t lower_ep_addr;

	assert(rdma_entry->iov_count > 0);
	assert(rdma_entry->rma_iov_count > 0);
	assert(rdma_entry->bytes_submitted < rdma_entry->total_len);

	if (rdma_entry->lower_ep_type == EFA_EP) {
		max_rdma_size = efa_max_rdma_size(ep->rdm_ep);
		lower_ep = ep->rdm_ep;
		lower_ep_addr = rdma_entry->addr;
	} else {
		max_rdma_size = SIZE_MAX;
		lower_ep = ep->shm_ep;
		lower_ep_addr = rxr_ep_get_peer(ep, rdma_entry->addr)->shm_fiaddr;
	}
	assert(max_rdma_size > 0);

	ret = rxr_locate_iov_pos(rdma_entry->iov, rdma_entry->iov_count,
				 rdma_entry->bytes_submitted,
				 &iov_idx, &iov_offset);
	assert(ret == 0);

	ret = rxr_locate_rma_iov_pos(rdma_entry->rma_iov, rdma_entry->rma_iov_count,
				     rdma_entry->bytes_submitted,
				     &rma_iov_idx, &rma_iov_offset);
	assert(ret == 0);

	total_iov_len = ofi_total_iov_len(rdma_entry->iov, rdma_entry->iov_count);
	total_rma_iov_len = rxr_total_rma_iov_len(rdma_entry->rma_iov, rdma_entry->rma_iov_count);
	assert(rdma_entry->total_len == MIN(total_iov_len, total_rma_iov_len));

	while (rdma_entry->bytes_submitted < rdma_entry->total_len) {
		assert(iov_idx < rdma_entry->iov_count);
		assert(iov_offset < rdma_entry->iov[iov_idx].iov_len);
		assert(rma_iov_idx < rdma_entry->rma_iov_count);
		assert(rma_iov_offset < rdma_entry->rma_iov[rma_iov_idx].len);

		iov_ptr = (char *)rdma_entry->iov[iov_idx].iov_base + iov_offset;
		rma_iov_ptr = (char *)rdma_entry->rma_iov[rma_iov_idx].addr + rma_iov_offset;

		max_iov_segsize = rdma_entry->iov[iov_idx].iov_len - iov_offset;
		max_rma_iov_segsize = rdma_entry->rma_iov[rma_iov_idx].len - rma_iov_offset;
		segsize = MIN(max_iov_segsize, max_rma_iov_segsize);
		if (rdma_entry->lower_ep_type == EFA_EP)
			segsize = MIN(segsize, rxr_env.efa_rdma_read_segment_size);
		segsize = MIN(segsize, max_rdma_size);

		/* because fi_send uses a pkt_entry as context
		 * we had to use a pkt_entry as context too
		 */
		if (rdma_entry->lower_ep_type == SHM_EP)
			pkt_entry = rxr_pkt_entry_alloc(ep, ep->tx_pkt_shm_pool);
		else
			pkt_entry = rxr_pkt_entry_alloc(ep, ep->tx_pkt_efa_pool);

		if (OFI_UNLIKELY(!pkt_entry))
			return -FI_EAGAIN;

		rxr_pkt_init_read_context(ep, rdma_entry, segsize, pkt_entry);

		ret = fi_read(lower_ep,
			      iov_ptr, segsize, rdma_entry->mr_desc[iov_idx],
			      lower_ep_addr,
			      (uint64_t)rma_iov_ptr, rdma_entry->rma_iov[rma_iov_idx].key,
			      pkt_entry);

		if (OFI_UNLIKELY(ret)) {
			rxr_pkt_entry_release_tx(ep, pkt_entry);
			return ret;
		}

		rdma_entry->bytes_submitted += segsize;

		iov_offset += segsize;
		assert(iov_offset <= rdma_entry->iov[iov_idx].iov_len);
		if (iov_offset == rdma_entry->iov[iov_idx].iov_len) {
			iov_idx += 1;
			iov_offset = 0;
		}

		rma_iov_offset += segsize;
		assert(rma_iov_offset <= rdma_entry->rma_iov[rma_iov_idx].len);
		if (rma_iov_offset == rdma_entry->rma_iov[rma_iov_idx].len) {
			rma_iov_idx += 1;
			rma_iov_offset = 0;
		}
	}

	if (rdma_entry->total_len == total_iov_len) {
		assert(iov_idx == rdma_entry->iov_count);
		assert(iov_offset == 0);
	}

	if (rdma_entry->total_len == total_rma_iov_len) {
		assert(rma_iov_idx == rdma_entry->rma_iov_count);
		assert(rma_iov_offset == 0);
	}

	return 0;
}

int rxr_rdma_handle_error(struct rxr_ep *ep, struct rxr_rdma_entry *rdma_entry, int ret)
{
	struct rxr_tx_entry *tx_entry;
	struct rxr_rx_entry *rx_entry;

	if (rdma_entry->src_type == RXR_TX_ENTRY) {
		tx_entry = ofi_bufpool_get_ibuf(ep->tx_entry_pool, rdma_entry->src_id);
		ret = rxr_cq_handle_tx_error(ep, tx_entry, ret);
	} else {
		assert(rdma_entry->src_type == RXR_RX_ENTRY);
		rx_entry = ofi_bufpool_get_ibuf(ep->rx_entry_pool, rdma_entry->src_id);
		ret = rxr_cq_handle_rx_error(ep, rx_entry, ret);
	}

	dlist_remove(&rdma_entry->pending_entry);
	return ret;
}

