/*
 * Copyright (c) 2017 Intel Corporation, Inc.  All rights reserved.
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

#ifndef _OFI_VERBS_DGRAM_H_
#define _OFI_VERBS_DGRAM_H_

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include "../fi_verbs.h"
#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* Verbs-DGRAM Pool functionality */
struct fi_ibv_dgram_buf_pool;

typedef void(*fi_ibv_dgram_pool_entry_cancel_hndlr) (struct fi_ibv_dgram_buf_pool *);

struct fi_ibv_dgram_buf_pool {
	struct util_buf_pool	*pool;
	struct dlist_entry	buf_list;

	fi_ibv_dgram_pool_entry_cancel_hndlr cancel_hndlr;
};

typedef int(*handle_wr_cb)(struct util_cq *util_cq,
			   struct util_cntr *util_cntr,
			   struct ibv_wc *wc);

struct fi_ibv_dgram_wr_entry_hdr {
	struct dlist_entry		entry;
	void				*desc;
	struct fi_ibv_dgram_ep		*ep;
	int32_t				comp_unsignaled_cnt;
	void				*context;
	uint64_t			flags;
	handle_wr_cb			suc_cb;
	handle_wr_cb			err_cb;
};

struct fi_ibv_dgram_pool_attr {
	size_t	count;
	size_t	size;
	void	*pool_ctx;

	fi_ibv_dgram_pool_entry_cancel_hndlr cancel_hndlr;

	util_buf_region_alloc_hndlr	alloc_hndlr;
	util_buf_region_free_hndlr	free_hndlr;
};

/*
 * The Global Routing Header (GRH) of the incoming message
 * will be placed in the first 40 bytes of the buffer(s)
 * in the scatter list.
 * @note If no GRH is present in the incoming message,
 *       then the first bytes will be undefined.
 * This means that in all cases, the actual data of the
 * incoming message will start at an offset of 40 bytes
 * into the buffer(s) in the scatter list.
 */

struct fi_ibv_dgram_wr_entry {
	struct fi_ibv_dgram_wr_entry_hdr	hdr;
	char					grh_buf[];
};

#define VERBS_DGRAM_GRH_LENGTH	40
#define VERBS_DGRAM_WR_ENTRY_SIZE					\
	(sizeof(struct fi_ibv_dgram_wr_entry) + VERBS_DGRAM_GRH_LENGTH)

static inline struct fi_ibv_dgram_wr_entry_hdr*
fi_ibv_dgram_wr_entry_get(struct fi_ibv_dgram_buf_pool *pool)
{
	struct fi_ibv_dgram_wr_entry_hdr *buf;
	void *mr = NULL;

	buf = util_buf_alloc_ex(pool->pool, &mr);
	if (OFI_UNLIKELY(!buf))
		return NULL;
	buf->desc = fi_mr_desc((struct fid_mr *)mr);
	dlist_insert_tail(&buf->entry, &pool->buf_list);

	return buf;
}

static inline void
fi_ibv_dgram_wr_entry_release(struct fi_ibv_dgram_buf_pool *pool,
			      struct fi_ibv_dgram_wr_entry_hdr *buf)
{
	dlist_remove(&buf->entry);
	util_buf_release(pool->pool, buf);
}

static inline void
fi_ibv_dgram_mr_buf_close(void *pool_ctx, void *context)
{
	/* We would get a (fid_mr *) in context, but
	 * it is safe to cast it into (fid *) */
	fi_close((struct fid *)context);
}

static inline int
fi_ibv_dgram_mr_buf_reg(void *pool_ctx, void *addr,
			size_t len, void **context)
{
	int ret;
	struct fid_mr *mr;
	struct fid_domain *domain = (struct fid_domain *)pool_ctx;

	ret = fi_mr_reg(domain, addr, len, FI_SEND | FI_RECV,
			0, 0, 0, &mr, NULL);
	*context = mr;
	return ret;
}

static inline void
fi_ibv_dgram_pool_wr_entry_cancel(struct fi_ibv_dgram_buf_pool *pool)
{
	struct dlist_entry *entry;
	struct fi_ibv_dgram_wr_entry_hdr *buf;
	entry = pool->buf_list.next;
	buf = container_of(entry, struct fi_ibv_dgram_wr_entry_hdr,
			   entry);
	fi_ibv_dgram_wr_entry_release(pool, buf);
}

static inline void
fi_ibv_dgram_pool_destroy(struct fi_ibv_dgram_buf_pool *pool)
{
	if (pool->cancel_hndlr) {
		while (!dlist_empty(&pool->buf_list))
			pool->cancel_hndlr(pool);
	}

	util_buf_pool_destroy(pool->pool);
}

static inline int
fi_ibv_dgram_pool_create(struct fi_ibv_dgram_pool_attr *attr,
			 struct fi_ibv_dgram_buf_pool *pool)
{
	int ret = util_buf_pool_create_ex(&pool->pool, attr->size,
					  16, 0, attr->count,
					  attr->alloc_hndlr,
					  attr->free_hndlr,
					  attr->pool_ctx);
	if (ret) {
		VERBS_WARN(FI_LOG_EP_DATA,
			   "Unable to create buf pool\n");
		return ret;
	}
	pool->cancel_hndlr = attr->cancel_hndlr;
	dlist_init(&pool->buf_list);

	return FI_SUCCESS;
}
/* ~Verbs UD Pool functionality */

struct fi_ibv_dgram_cq {
	struct util_cq	util_cq;
	struct ibv_cq	*ibv_cq;
};

struct fi_ibv_dgram_av {
	struct util_av	util_av;
};

struct fi_ibv_dgram_eq {
	struct util_eq	util_eq;
};

struct fi_ibv_dgram_cntr {
	struct util_cntr	util_cntr;
};

struct fi_ibv_dgram_av_entry {
	struct ofi_ib_ud_ep_name	*addr;
	struct ibv_ah			*ah;
};

struct fi_ibv_dgram_ep {
	struct util_ep			util_ep;
	struct ibv_qp			*ibv_qp;
	struct fi_info			*info;
	struct fi_ibv_dgram_av		*av;
	struct fi_ibv_domain		*domain;
	struct fi_ibv_dgram_buf_pool	grh_pool;
	struct ofi_ib_ud_ep_name	ep_name;
	int				service;
	int				ep_flags;
	int32_t				max_unsignaled_send_cnt;
	ofi_atomic32_t			unsignaled_send_cnt;
};

extern struct fi_ops_msg fi_ibv_dgram_msg_ops;

static inline struct fi_ibv_dgram_av_entry*
fi_ibv_dgram_av_lookup_av_entry(struct fi_ibv_dgram_av *av, int index)
{
	assert((index >= 0) && ((size_t)index < av->util_av.count));
	return ofi_av_get_addr(&av->util_av, index);
}

static inline
int fi_ibv_dgram_is_completion(uint64_t cq_flags, uint64_t op_flags)
{
	if ((op_flags & FI_COMPLETION) ||
	    (op_flags & (FI_INJECT_COMPLETE	|
			 FI_TRANSMIT_COMPLETE	|
			 FI_DELIVERY_COMPLETE)))
		return 1;
	else if (op_flags & FI_INJECT)
		return 0;
	else if (!(cq_flags & FI_SELECTIVE_COMPLETION))
		return 1;
	return 0;
}

int fi_ibv_dgram_rx_cq_comp(struct util_cq *util_cq,
			    struct util_cntr *util_cntr,
			    struct ibv_wc *wc);
int fi_ibv_dgram_tx_cq_comp(struct util_cq *util_cq,
			    struct util_cntr *util_cntr,
			    struct ibv_wc *wc);
int fi_ibv_dgram_tx_cq_report_error(struct util_cq *util_cq,
				    struct util_cntr *util_cntr,
				    struct ibv_wc *wc);
int fi_ibv_dgram_rx_cq_report_error(struct util_cq *util_cq,
				    struct util_cntr *util_cntr,
				    struct ibv_wc *wc);
int fi_ibv_dgram_tx_cq_no_action(struct util_cq *util_cq,
				 struct util_cntr *util_cntr,
				 struct ibv_wc *wc);
int fi_ibv_dgram_rx_cq_no_action(struct util_cq *util_cq,
				 struct util_cntr *util_cntr,
				 struct ibv_wc *wc);

void fi_ibv_dgram_recv_cq_progress(struct util_ep *util_ep);
void fi_ibv_dgram_send_cq_progress(struct util_ep *util_ep);
void fi_ibv_dgram_send_recv_cq_progress(struct util_ep *util_ep);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _OFI_VERBS_UD_H_ */
