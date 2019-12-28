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

#if HAVE_CONFIG_H
#include <config.h>
#endif /* HAVE_CONFIG_H */

#ifndef _RXR_RDMA_H_
#define _RXR_RDMA_H_

enum rxr_rdma_entry_state {
	RXR_RDMA_ENTRY_FREE = 0,
	RXR_RDMA_ENTRY_CREATED,
	RXR_RDMA_ENTRY_PENDING
};

/* rxr_rdma_entry was arranged as a packet
 * and was put in a rxr_pkt_entry. Because rxr_pkt_entry is used
 * as context.
 */
struct rxr_rdma_entry {
	int rdma_id;
	enum rxr_lower_ep_type lower_ep_type;

	int src_type;
	int src_id;
	int state;

	fi_addr_t addr;

	struct iovec iov[RXR_IOV_LIMIT];
	size_t iov_count;
	struct fid_mr *mr[RXR_IOV_LIMIT];
	void *mr_desc[RXR_IOV_LIMIT];

	struct fi_rma_iov rma_iov[RXR_IOV_LIMIT];
	size_t rma_iov_count;

	size_t total_len;
	size_t bytes_submitted; /* bytes fi_read() succeeded */
	size_t bytes_finished; /* bytes received completion */

	struct dlist_entry pending_entry;
};

struct rxr_rdma_entry *rxr_rdma_alloc_entry(struct rxr_ep *ep, int entry_type, void *x_entry,
					    enum rxr_lower_ep_type lower_ep_type);

void rxr_rdma_release_entry(struct rxr_ep *ep, struct rxr_rdma_entry *rdma_entry);

int rxr_rdma_post_read_or_queue(struct rxr_ep *ep, struct rxr_rdma_entry *pkt_entry);

int rxr_rdma_post_read(struct rxr_ep *ep, struct rxr_rdma_entry *rdma_entry);

void rxr_rdma_handle_read_completion(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry);

int rxr_rdma_handle_error(struct rxr_ep *ep, struct rxr_rdma_entry *rdma_entry, int ret);

#endif

