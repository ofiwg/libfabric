/*
 * Copyright (c) 2026, Amazon.com, Inc.  All rights reserved.
 *
 * This software is available to you under the BSD license
 * below:
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

#include <stdbool.h>
#include <stdlib.h>

#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_tagged.h>

#include "shared.h"

/*
 * Work request API posting.
 *
 * With --use-wr, ft_post_tx_buf() and ft_post_rx_buf() format the operation with
 * fi_wr_prepare and then submit it with fi_wr_queue_tx / fi_wr_queue_recv /
 * fi_wr_queue_trecv plus the matching flush, instead of calling fi_send /
 * fi_recv / fi_tsend / fi_trecv.  Every test built on those two helpers
 * therefore also covers the work request interface, including the ping-pong and
 * bandwidth benchmarks.
 *
 */
static fi_wr ft_tx_wr, ft_rx_wr;
static size_t ft_tx_wr_size, ft_rx_wr_size;

enum ft_wr_queue_type {
	FT_WR_QUEUE_TX,
	FT_WR_QUEUE_RECV,
	FT_WR_QUEUE_TRECV,
};

struct ft_wr_key {
	bool valid;
	void *buf;
	void *desc;
	size_t len;
	fi_addr_t addr;
	uint64_t data;
	uint64_t tag;
	uint64_t flags;
	enum fi_op_type op_type;
	/* RMA (FI_OP_WRITE / FI_OP_READ) remote target. */
	uint64_t rma_addr;
	uint64_t rma_key;
};

static struct ft_wr_key ft_tx_wr_key, ft_rx_wr_key;

/*
 * When set, the ft_wr_post_* helpers queue the operation without flushing, so a
 * benchmark can batch a window of posts behind a single doorbell. The caller
 * flushes with fi_tx_flush() / fi_recv_flush() / fi_trecv_flush() at the window
 * boundary; ft_get_tx_comp() / ft_get_rx_comp() flush whatever is still queued,
 * so blocking on a completion can never wait on un-initiated work.
 */
static bool ft_wr_batch;

/* Queue type of the work queued since the last flush, -1 when nothing is. */
static int ft_tx_wr_queued = -1, ft_rx_wr_queued = -1;

void ft_wr_set_batch(bool enable)
{
	ft_wr_batch = enable;
}

void ft_free_wrs(void)
{
	free(ft_tx_wr);
	free(ft_rx_wr);
	ft_tx_wr = NULL;
	ft_rx_wr = NULL;
	ft_tx_wr_size = 0;
	ft_rx_wr_size = 0;
	ft_tx_wr_key.valid = false;
	ft_rx_wr_key.valid = false;
	ft_wr_batch = false;
	ft_tx_wr_queued = -1;
	ft_rx_wr_queued = -1;
}

static int ft_alloc_wrs(void)
{
	if (ft_tx_wr && ft_rx_wr)
		return 0;

	ft_tx_wr_size = fi->ep_attr->max_tx_wr_size;
	ft_rx_wr_size = fi->ep_attr->max_rx_wr_size;

	if (!ft_tx_wr_size || !ft_rx_wr_size) {
		FT_ERR("--use-wr given but %s reports no work request support "
		       "(max_tx_wr_size %zu, max_rx_wr_size %zu)",
		       fi->fabric_attr->prov_name, ft_tx_wr_size,
		       ft_rx_wr_size);
		return -FI_ENOSYS;
	}

	ft_tx_wr = calloc(1, ft_tx_wr_size);
	ft_rx_wr = calloc(1, ft_rx_wr_size);
	if (!ft_tx_wr || !ft_rx_wr) {
		ft_free_wrs();
		return -FI_ENOMEM;
	}

	return 0;
}

static bool ft_wr_key_matches(const struct ft_wr_key *key,
			      enum fi_op_type op_type, void *op_buf,
			      void *op_mr_desc, size_t len, fi_addr_t addr,
			      uint64_t data, uint64_t tag, uint64_t flags,
			      uint64_t rma_addr, uint64_t rma_key)
{
	return key->valid && key->op_type == op_type && key->buf == op_buf &&
	       key->desc == op_mr_desc && key->len == len &&
	       key->addr == addr && key->data == data && key->tag == tag &&
	       key->flags == flags && key->rma_addr == rma_addr &&
	       key->rma_key == rma_key;
}

static bool ft_wr_op_is_tagged(enum fi_op_type op_type)
{
	return op_type == FI_OP_TSEND || op_type == FI_OP_TRECV;
}

static bool ft_wr_op_is_rma(enum fi_op_type op_type)
{
	return op_type == FI_OP_WRITE || op_type == FI_OP_READ;
}

static bool ft_wr_op_is_tx(enum fi_op_type op_type)
{
	return op_type == FI_OP_SEND || op_type == FI_OP_TSEND ||
	       op_type == FI_OP_WRITE || op_type == FI_OP_READ;
}

/*
 * Update a prepared work request in place when only modifiable fields changed.
 */
static bool ft_wr_try_modify(struct fid_ep *ep_ptr, enum fi_op_type op_type,
			     struct ft_wr_key *key, fi_wr wr, void *op_buf,
			     void *op_mr_desc, size_t len, fi_addr_t addr,
			     uint64_t data, uint64_t tag, uint64_t flags,
			     uint64_t rma_addr, uint64_t rma_key)
{
	int ret;

	if (!key->valid || key->op_type != op_type || key->flags != flags ||
	    (key->data != NO_CQ_DATA) != (data != NO_CQ_DATA))
		return false;

	if (key->addr != addr) {
		ret = fi_wr_modify_addr(ep_ptr, wr, op_type, addr);
		if (ret) {
			FT_PRINTERR("fi_wr_modify_addr", ret);
			return false;
		}
		key->addr = addr;
	}

	if (key->buf != op_buf || key->desc != op_mr_desc || key->len != len ||
	    ft_wr_op_is_tx(op_type)) {
		struct iovec iov = {
			.iov_base = op_buf,
			.iov_len = len,
		};
		void *desc = op_mr_desc;

		ret = fi_wr_modify_iov(ep_ptr, wr, op_type, &iov, &desc, 1);
		if (ret) {
			FT_PRINTERR("fi_wr_modify_iov", ret);
			return false;
		}
		key->buf = op_buf;
		key->desc = op_mr_desc;
		key->len = len;
	}

	if (ft_wr_op_is_rma(op_type) &&
	    (key->rma_addr != rma_addr || key->rma_key != rma_key)) {
		struct fi_rma_iov rma_iov = {
			.addr = rma_addr,
			.len = len,
			.key = rma_key,
		};

		ret = fi_wr_modify_rma_iov(ep_ptr, wr, op_type, &rma_iov, 1);
		if (ret) {
			FT_PRINTERR("fi_wr_modify_rma_iov", ret);
			return false;
		}
		key->rma_addr = rma_addr;
		key->rma_key = rma_key;
	}

	if (ft_wr_op_is_tagged(op_type) && key->tag != tag) {
		ret = fi_wr_modify_tag(ep_ptr, wr, op_type, tag, 0);
		if (ret) {
			FT_PRINTERR("fi_wr_modify_tag", ret);
			return false;
		}
		key->tag = tag;
	}

	if (data != NO_CQ_DATA && key->data != data) {
		ret = fi_wr_modify_data(ep_ptr, wr, op_type, data);
		if (ret) {
			FT_PRINTERR("fi_wr_modify_data", ret);
			return false;
		}
		key->data = data;
	}

	return true;
}

static int ft_wr_prepare_op(struct fid_ep *ep_ptr, enum fi_op_type op_type,
			    struct ft_wr_key *key, fi_wr wr, size_t wr_size,
			    void *op_buf, void *op_mr_desc, size_t len,
			    fi_addr_t addr, uint64_t data, uint64_t tag,
			    uint64_t flags, uint64_t rma_addr, uint64_t rma_key)
{
	struct iovec iov = {
		.iov_base = op_buf,
		.iov_len = len,
	};
	struct fi_rma_iov rma_iov = {
		.addr = rma_addr,
		.len = len,
		.key = rma_key,
	};
	struct fi_msg msg = {
		.msg_iov = &iov,
		.desc = &op_mr_desc,
		.iov_count = 1,
		.addr = addr,
		.context = NULL,
		.data = (data == NO_CQ_DATA) ? 0 : data,
	};
	struct fi_msg_tagged tagged_msg = {
		.msg_iov = &iov,
		.desc = &op_mr_desc,
		.iov_count = 1,
		.addr = addr,
		.tag = tag,
		.ignore = 0,
		.context = NULL,
		.data = (data == NO_CQ_DATA) ? 0 : data,
	};
	struct fi_msg_rma rma_msg = {
		.msg_iov = &iov,
		.desc = &op_mr_desc,
		.iov_count = 1,
		.addr = addr,
		.rma_iov = &rma_iov,
		.rma_iov_count = 1,
		.context = NULL,
		.data = (data == NO_CQ_DATA) ? 0 : data,
	};
	struct fi_op_msg op_msg = {
		.ep = ep_ptr,
		.msg = msg,
		.flags = flags,
	};
	struct fi_op_tagged op_tagged = {
		.ep = ep_ptr,
		.msg = tagged_msg,
		.flags = flags,
	};
	struct fi_op_rma op_rma = {
		.ep = ep_ptr,
		.msg = rma_msg,
		.flags = flags,
	};
	struct fi_wr_attr attr = {
		.op_type = op_type,
	};
	size_t wr_len = wr_size;
	int ret;

	switch (op_type) {
	case FI_OP_TSEND:
	case FI_OP_TRECV:
		attr.op.tagged = &op_tagged;
		break;
	case FI_OP_WRITE:
	case FI_OP_READ:
		attr.op.rma = &op_rma;
		break;
	default:
		attr.op.msg = &op_msg;
		break;
	}

	ret = fi_wr_prepare(ep_ptr, &attr, wr, &wr_len);
	if (ret) {
		FT_PRINTERR("fi_wr_prepare", ret);
		return ret;
	}

	if (wr_len > wr_size) {
		FT_ERR("fi_wr_prepare reported length %zu over the %zu byte "
		       "work request buffer", wr_len, wr_size);
		return -FI_EOTHER;
	}

	*key = (struct ft_wr_key) {
		.valid = true,
		.buf = op_buf,
		.desc = op_mr_desc,
		.len = len,
		.addr = addr,
		.data = data,
		.tag = tag,
		.flags = flags,
		.op_type = op_type,
		.rma_addr = rma_addr,
		.rma_key = rma_key,
	};

	return 0;
}

static ssize_t ft_wr_queue(struct fid_ep *ep_ptr, const fi_wr wr, void *ctx,
			   enum ft_wr_queue_type type)
{
	switch (type) {
	case FT_WR_QUEUE_TX:
		return fi_wr_queue_tx(ep_ptr, wr, ctx);
	case FT_WR_QUEUE_RECV:
		return fi_wr_queue_recv(ep_ptr, wr, ctx);
	default:
		return fi_wr_queue_trecv(ep_ptr, wr, ctx);
	}
}

static const char *ft_wr_flush_str(enum ft_wr_queue_type type)
{
	switch (type) {
	case FT_WR_QUEUE_TX:
		return "fi_tx_flush";
	case FT_WR_QUEUE_RECV:
		return "fi_recv_flush";
	default:
		return "fi_trecv_flush";
	}
}

static ssize_t ft_wr_flush(struct fid_ep *ep_ptr, enum ft_wr_queue_type type)
{
	int *queued = (type == FT_WR_QUEUE_TX) ? &ft_tx_wr_queued :
						 &ft_rx_wr_queued;
	ssize_t ret;

	switch (type) {
	case FT_WR_QUEUE_TX:
		ret = fi_tx_flush(ep_ptr, 0);
		break;
	case FT_WR_QUEUE_RECV:
		ret = fi_recv_flush(ep_ptr, 0);
		break;
	default:
		ret = fi_trecv_flush(ep_ptr, 0);
		break;
	}

	if (ret) {
		FT_ERR("%s(): ret=%d (%s)", ft_wr_flush_str(type), (int) ret,
		       fi_strerror((int) -ret));
		return ret;
	}

	*queued = -1;
	return 0;
}

ssize_t ft_wr_flush_tx_queued(struct fid_ep *ep_ptr)
{
	if (ft_tx_wr_queued < 0)
		return 0;

	return ft_wr_flush(ep_ptr, ft_tx_wr_queued);
}

ssize_t ft_wr_flush_rx_queued(struct fid_ep *ep_ptr)
{
	if (ft_rx_wr_queued < 0)
		return 0;

	return ft_wr_flush(ep_ptr, ft_rx_wr_queued);
}

static ssize_t ft_wr_queue_only(struct fid_ep *ep_ptr, const fi_wr wr,
				void *ctx, enum ft_wr_queue_type type)
{
	bool tx = (type == FT_WR_QUEUE_TX);
	struct fid_cq *cq = tx ? txcq : rxcq;
	uint64_t *seq = tx ? &tx_seq : &rx_seq;
	uint64_t *cq_cntr = tx ? &tx_cq_cntr : &rx_cq_cntr;

	FT_POST(ft_wr_queue, ft_progress, cq, (*seq), cq_cntr, "fi_wr_queue",
		ep_ptr, wr, ctx, type);

	if (tx)
		ft_tx_wr_queued = type;
	else
		ft_rx_wr_queued = type;

	return 0;
}

static ssize_t ft_wr_queue_and_flush(struct fid_ep *ep_ptr, const fi_wr wr,
				     void *ctx, enum ft_wr_queue_type type)
{
	ssize_t ret;

	ret = ft_wr_queue_only(ep_ptr, wr, ctx, type);
	if (ret)
		return ret;

	return ft_wr_flush(ep_ptr, type);
}

static ssize_t ft_wr_submit(struct fid_ep *ep_ptr, const fi_wr wr, void *ctx,
			    enum ft_wr_queue_type type)
{
	if (ft_wr_batch)
		return ft_wr_queue_only(ep_ptr, wr, ctx, type);

	return ft_wr_queue_and_flush(ep_ptr, wr, ctx, type);
}

ssize_t ft_wr_post_tx_buf(struct fid_ep *ep_ptr, fi_addr_t fi_addr,
			  size_t size, uint64_t data, void *ctx,
			  void *op_buf, void *op_mr_desc, uint64_t op_tag)
{
	uint64_t flags = (data == NO_CQ_DATA) ? 0 : FI_REMOTE_CQ_DATA;
	enum fi_op_type op_type;
	int ret;

	ret = ft_alloc_wrs();
	if (ret)
		return ret;

	if (hints->caps & FI_TAGGED) {
		op_type = FI_OP_TSEND;
		op_tag = op_tag ? op_tag : tx_seq;
	} else {
		op_type = FI_OP_SEND;
		op_tag = 0;
	}

	if (!ft_wr_try_modify(ep_ptr, op_type, &ft_tx_wr_key, ft_tx_wr, op_buf,
			      op_mr_desc, size, fi_addr, data, op_tag, flags, 0,
			      0)) {
		ret = ft_wr_prepare_op(ep_ptr, op_type, &ft_tx_wr_key, ft_tx_wr,
				       ft_tx_wr_size, op_buf, op_mr_desc, size,
				       fi_addr, data, op_tag, flags, 0, 0);
		if (ret)
			return ret;
	}

	return ft_wr_submit(ep_ptr, ft_tx_wr, ctx, FT_WR_QUEUE_TX);
}

ssize_t ft_wr_post_rx_buf(struct fid_ep *ep_ptr, fi_addr_t fi_addr,
			  size_t size, void *ctx, void *op_buf,
			  void *op_mr_desc, uint64_t op_tag)
{
	enum ft_wr_queue_type type;
	enum fi_op_type op_type;
	int ret;

	ret = ft_alloc_wrs();
	if (ret)
		return ret;

	if (hints->caps & FI_TAGGED) {
		op_type = FI_OP_TRECV;
		type = FT_WR_QUEUE_TRECV;
		op_tag = op_tag ? op_tag : rx_seq;
	} else {
		op_type = FI_OP_RECV;
		type = FT_WR_QUEUE_RECV;
		op_tag = 0;
	}

	if (!ft_wr_key_matches(&ft_rx_wr_key, op_type, op_buf, op_mr_desc, size,
			       fi_addr, NO_CQ_DATA, op_tag, 0, 0, 0) &&
	    !ft_wr_try_modify(ep_ptr, op_type, &ft_rx_wr_key, ft_rx_wr, op_buf,
			      op_mr_desc, size, fi_addr, NO_CQ_DATA, op_tag, 0,
			      0, 0)) {
		ret = ft_wr_prepare_op(ep_ptr, op_type, &ft_rx_wr_key, ft_rx_wr,
				       ft_rx_wr_size, op_buf, op_mr_desc, size,
				       fi_addr, NO_CQ_DATA, op_tag, 0, 0, 0);
		if (ret)
			return ret;
	}

	return ft_wr_submit(ep_ptr, ft_rx_wr, ctx, type);
}

ssize_t ft_wr_post_rma(enum fi_op_type op_type, struct fid_ep *ep_ptr,
		       void *op_buf, void *op_mr_desc, size_t size,
		       fi_addr_t fi_addr, uint64_t rma_addr, uint64_t rma_key,
		       uint64_t data, uint64_t flags, void *ctx)
{
	int ret;

	ret = ft_alloc_wrs();
	if (ret)
		return ret;

	if (!ft_wr_try_modify(ep_ptr, op_type, &ft_tx_wr_key, ft_tx_wr, op_buf,
			      op_mr_desc, size, fi_addr, data, 0, flags,
			      rma_addr, rma_key)) {
		ret = ft_wr_prepare_op(ep_ptr, op_type, &ft_tx_wr_key, ft_tx_wr,
				       ft_tx_wr_size, op_buf, op_mr_desc, size,
				       fi_addr, data, 0, flags, rma_addr,
				       rma_key);
		if (ret)
			return ret;
	}

	return ft_wr_submit(ep_ptr, ft_tx_wr, ctx, FT_WR_QUEUE_TX);
}
