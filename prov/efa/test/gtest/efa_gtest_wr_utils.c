/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_av.h"
#include "efa_base_ep.h"
#include "efa_io_defs.h"
#include "efa_mr.h"
#include "efa_wr.h"
#include "efa_gtest_wr_utils.h"

static struct efa_base_ep *efa_test_wr_base_ep(struct fid_ep *ep_fid)
{
	return container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);
}

size_t efa_test_ep_recv_wr_index(struct fid_ep *ep_fid)
{
	return efa_test_wr_base_ep(ep_fid)->recv_wr_index;
}

int efa_test_ep_tx_wr_pending(struct fid_ep *ep_fid)
{
	struct efa_base_ep *base_ep = efa_test_wr_base_ep(ep_fid);

#if HAVE_EFA_DATA_PATH_DIRECT
	if (base_ep->qp->data_path_direct_enabled)
		return base_ep->qp->data_path_direct_qp.sq.num_wqe_pending != 0;
#endif
	return base_ep->is_wr_started;
}

size_t efa_test_wr_tx_size(void)
{
	return efa_wr_tx_size();
}

size_t efa_test_wr_rx_size(size_t num_sge)
{
	return efa_wr_rx_size(num_sge);
}

int efa_test_av_ud_addr(struct fid_ep *ep_fid, fi_addr_t addr, uint16_t *ahn,
			uint16_t *qpn, uint32_t *qkey)
{
	struct efa_base_ep *base_ep = efa_test_wr_base_ep(ep_fid);
	struct efa_av_entry *entry;

	entry = efa_av_addr_to_entry(base_ep->av, addr);
	if (!entry)
		return -1;

	if (ahn)
		*ahn = entry->ah->ahn;
	if (qpn)
		*qpn = efa_av_entry_ep_addr(entry)->qpn;
	if (qkey)
		*qkey = efa_av_entry_ep_addr(entry)->qkey;

	return 0;
}

uint32_t efa_test_mr_desc_lkey(void *desc)
{
	return ((struct efa_mr *) desc)->lkey;
}

#if HAVE_EFA_DATA_PATH_DIRECT

int efa_test_ep_data_path_direct_enabled(struct fid_ep *ep_fid)
{
	return efa_test_wr_base_ep(ep_fid)->qp->data_path_direct_enabled;
}

unsigned int efa_test_ep_tx_num_wqe_pending(struct fid_ep *ep_fid)
{
	return efa_test_wr_base_ep(ep_fid)->qp->data_path_direct_qp.sq.num_wqe_pending;
}

unsigned int efa_test_ep_rx_wqe_posted(struct fid_ep *ep_fid)
{
	return efa_test_wr_base_ep(ep_fid)->qp->data_path_direct_qp.rq.wq.wqe_posted;
}

int efa_test_wr_tx_op_type(const void *wr)
{
	const struct efa_io_tx_wqe_128 *wqe = wr;

	return EFA_GET(&wqe->meta.ctrl1, EFA_IO_TX_META_DESC_OP_TYPE);
}

int efa_test_wr_op_type_send(void)
{
	return EFA_IO_SEND;
}

int efa_test_wr_op_type_rdma_read(void)
{
	return EFA_IO_RDMA_READ;
}

int efa_test_wr_op_type_rdma_write(void)
{
	return EFA_IO_RDMA_WRITE;
}

int efa_test_wr_rx_desc_is_last(const void *wr, size_t index)
{
	const struct efa_io_rx_desc *rx_buf = wr;

	return EFA_GET(&rx_buf[index].lkey_ctrl, EFA_IO_RX_DESC_LAST);
}

static bool efa_test_wr_op_is_rdma(enum fi_op_type op_type)
{
	return op_type == FI_OP_READ || op_type == FI_OP_WRITE;
}

void efa_test_wr_tx_ud_addr(const void *wr, uint16_t *ahn, uint16_t *qpn,
			    uint32_t *qkey)
{
	const struct efa_io_tx_meta_desc *meta =
		&((const struct efa_io_tx_wqe_128 *) wr)->meta;

	if (ahn)
		*ahn = meta->ah;
	if (qpn)
		*qpn = meta->dest_qp_num;
	if (qkey)
		*qkey = meta->qkey;
}

int efa_test_wr_tx_is_inline(const void *wr)
{
	const struct efa_io_tx_wqe_128 *wqe = wr;

	return EFA_GET(&wqe->meta.ctrl1, EFA_IO_TX_META_DESC_INLINE_MSG);
}

uint16_t efa_test_wr_tx_length(const void *wr)
{
	const struct efa_io_tx_wqe_128 *wqe = wr;

	return wqe->meta.length;
}

int efa_test_wr_tx_has_imm(const void *wr)
{
	const struct efa_io_tx_wqe_128 *wqe = wr;

	return EFA_GET(&wqe->meta.ctrl1, EFA_IO_TX_META_DESC_HAS_IMM);
}

uint32_t efa_test_wr_tx_imm_data(const void *wr)
{
	const struct efa_io_tx_wqe_128 *wqe = wr;

	return htobe32(wqe->meta.immediate_data);
}

void efa_test_wr_tx_comp_actions(struct fid_ep *ep_fid, const void *wr,
				 int *local_req, uint32_t *local_id,
				 uint64_t *local_data, int *remote_req,
				 uint32_t *remote_id, uint64_t *remote_data)
{
	const struct efa_io_tx_wqe_128 *wqe = wr;
	const struct efa_io_action_block *block;
	struct efa_base_ep *base_ep = efa_test_wr_base_ep(ep_fid);

	*local_req = EFA_GET(&wqe->meta.ctrl2,
			     EFA_IO_TX_META_DESC_LOCAL_ACTION_REQ);
	*remote_req = EFA_GET(&wqe->meta.ctrl2,
			      EFA_IO_TX_META_DESC_REMOTE_ACTION_REQ);
	*local_id = *remote_id = 0;
	*local_data = *remote_data = 0;
	if (!*local_req && !*remote_req)
		return;

	block = (const struct efa_io_action_block *) (
		(const uint8_t *) wqe +
		base_ep->qp->data_path_direct_qp.sq.action_block_offset);
	if (*local_req) {
		*local_id = block->local_action_handle;
		*local_data = block->local_action_data;
	}
	if (*remote_req) {
		*remote_id = block->remote_action_handle;
		*remote_data = block->remote_action_data;
	}
}

int efa_test_wr_tx_processing_hints(const void *wr)
{
	const struct efa_io_tx_wqe_128 *wqe = wr;

	return EFA_GET(&wqe->meta.ctrl3, EFA_IO_TX_META_DESC_PROCESSING_HINTS);
}

int efa_test_wr_processing_hint_high_pps(void)
{
	return EFA_IO_PROCESSING_HINT_BURST_PPS_SENSITIVE;
}

void efa_test_wr_tx_sge(const void *wr, enum fi_op_type op_type, size_t index,
			uint64_t *addr, uint32_t *len, uint32_t *lkey)
{
	const struct efa_io_tx_wqe_128 *wqe = wr;
	const struct efa_io_tx_buf_desc *tx_buf;

	tx_buf = efa_test_wr_op_is_rdma(op_type) ?
			 &wqe->data.rdma_req.local_mem[index] :
			 &wqe->data.sgl[index];

	if (addr)
		*addr = ((uint64_t) tx_buf->buf_addr_hi << 32) |
			tx_buf->buf_addr_lo;
	if (len)
		*len = tx_buf->length;
	if (lkey)
		*lkey = EFA_GET(&tx_buf->lkey, EFA_IO_TX_BUF_DESC_LKEY);
}

const void *efa_test_wr_tx_inline_data(const void *wr, enum fi_op_type op_type)
{
	const struct efa_io_tx_wqe_128 *wqe = wr;

	return efa_test_wr_op_is_rdma(op_type) ? wqe->data.rdma_req.inline_data :
						 wqe->data.inline_data;
}

void efa_test_wr_tx_rdma_addr(const void *wr, uint64_t *addr, uint32_t *rkey,
			      uint32_t *len)
{
	const struct efa_io_remote_mem_addr *remote_mem =
		&((const struct efa_io_tx_wqe_128 *) wr)->data.rdma_req.remote_mem;

	if (addr)
		*addr = ((uint64_t) remote_mem->buf_addr_hi << 32) |
			remote_mem->buf_addr_lo;
	if (rkey)
		*rkey = remote_mem->rkey;
	if (len)
		*len = remote_mem->length;
}

void efa_test_wr_rx_desc(const void *wr, size_t index, uint64_t *addr,
			 uint16_t *len, uint32_t *lkey)
{
	const struct efa_io_rx_desc *rx_buf =
		&((const struct efa_io_rx_desc *) wr)[index];

	if (addr)
		*addr = ((uint64_t) rx_buf->buf_addr_hi << 32) |
			rx_buf->buf_addr_lo;
	if (len)
		*len = rx_buf->length;
	if (lkey)
		*lkey = EFA_GET(&rx_buf->lkey_ctrl, EFA_IO_RX_DESC_LKEY);
}

#endif /* HAVE_EFA_DATA_PATH_DIRECT */
