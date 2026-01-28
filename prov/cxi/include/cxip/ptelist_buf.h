/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_PTELIST_BUF_H_
#define _CXIP_PTELIST_BUF_H_

#include <ofi_atom.h>
#include <ofi_list.h>
#include <stdbool.h>
#include <stddef.h>

/* Forward declarations */
struct cxip_md;
struct cxip_req;
struct cxip_rxc_hpc;
struct cxip_ux_send;

/* Type definitions */
struct cxip_ptelist_bufpool_attr {
	enum c_ptl_list list_type;

	/* Callback to handle PtlTE link error/unlink events */
	int (*ptelist_cb)(struct cxip_req *req, const union c_event *event);
	size_t buf_size;
	size_t min_space_avail;
	size_t min_posted;
	size_t max_posted;
	size_t max_cached;
};

struct cxip_ptelist_bufpool {
	struct cxip_ptelist_bufpool_attr attr;
	struct cxip_rxc_hpc *rxc;
	size_t buf_alignment;

	/* Ordered list of buffers emitted to hardware */
	struct dlist_entry active_bufs;

	/* List of consumed buffers which cannot be reposted yet
	 * since unexpected entries have not been matched.
	 */
	struct dlist_entry consumed_bufs;

	/* List of available buffers that may be appended to the list.
	 * These could be from a previous append failure or be cached
	 * from previous message processing to avoid map/unmap of
	 * list buffer.
	 */
	struct dlist_entry free_bufs;

	ofi_atomic32_t bufs_linked;
	ofi_atomic32_t bufs_allocated;
	ofi_atomic32_t bufs_free;
};

struct cxip_ptelist_req {
	/* Pending list of unexpected header entries which could not be placed
	 * on the RX context unexpected header list due to put events being
	 * received out-of-order.
	 */
	struct dlist_entry pending_ux_list;
};

struct cxip_ptelist_buf {
	struct cxip_ptelist_bufpool *pool;

	/* RX context the request buffer is posted on. */
	struct cxip_rxc_hpc *rxc;
	enum cxip_le_type le_type;
	struct dlist_entry buf_entry;
	struct cxip_req *req;

	/* Memory mapping of req_buf field. */
	struct cxip_md *md;

	/* The number of bytes consume by hardware when the request buffer was
	 * unlinked.
	 */
	size_t unlink_length;

	/* Current offset into the buffer where packets/data are landing. When
	 * the cur_offset is equal to unlink_length, software has completed
	 * event processing for the buffer.
	 */
	size_t cur_offset;

	/* Request list specific control information */
	struct cxip_ptelist_req request;

	/* The number of unexpected headers posted placed on the RX context
	 * unexpected header list which have not been matched.
	 */
	ofi_atomic32_t refcount;

	/* Buffer used to land packets. */
	char *data;
};

/* Function declarations */
int cxip_ptelist_bufpool_init(struct cxip_rxc_hpc *rxc,
			      struct cxip_ptelist_bufpool **pool,
			      struct cxip_ptelist_bufpool_attr *attr);

void cxip_ptelist_bufpool_fini(struct cxip_ptelist_bufpool *pool);

int cxip_ptelist_buf_replenish(struct cxip_ptelist_bufpool *pool,
			       bool seq_restart);

void cxip_ptelist_buf_link_err(struct cxip_ptelist_buf *buf, int rc_link_error);

void cxip_ptelist_buf_unlink(struct cxip_ptelist_buf *buf);

void cxip_ptelist_buf_put(struct cxip_ptelist_buf *buf, bool repost);

void cxip_ptelist_buf_get(struct cxip_ptelist_buf *buf);

void cxip_ptelist_buf_consumed(struct cxip_ptelist_buf *buf);

void _cxip_req_buf_ux_free(struct cxip_ux_send *ux, bool repost);

#endif /* _CXIP_PTELIST_BUF_H_ */
