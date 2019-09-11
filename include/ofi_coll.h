/*
 * Copyright (c) 2019 Intel Corporation, Inc.  All rights reserved.
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

#ifndef _OFI_COLL_H_
#define _OFI_COLL_H_

#include <rdma/fi_collective.h>

#define OFI_WORLD_CONTEXT_ID 0
#define OFI_CONTEXT_ID_SIZE 4

enum barrier_type {
	NO_BARRIER,
	BARRIER,
};

enum util_coll_op_type {
	UTIL_COLL_JOIN_OP,
	UTIL_COLL_BARRIER_OP,
	UTIL_COLL_ALLREDUCE_OP,
	UTIL_COLL_BROADCAST_OP,
};

struct util_av_set {
	struct fid_av_set	av_set_fid;
	struct util_av		*av;
	fi_addr_t		*fi_addr_array;
	size_t			fi_addr_count;
	uint64_t		flags;
	ofi_atomic32_t		ref;
	fastlock_t		lock;
};

enum coll_work_type {
	UTIL_COLL_SEND,
	UTIL_COLL_RECV,
	UTIL_COLL_REDUCE,
	UTIL_COLL_COPY,
	UTIL_COLL_COMP,
};

struct util_coll_hdr {
	struct slist_entry	entry;
	enum coll_work_type	type;
	/* only valid for xfer_item*/
	uint64_t		tag;
	int 			is_barrier;
};

struct util_coll_xfer_item {
	struct util_coll_hdr	hdr;
	void 			*buf;
	int			count;
	union {
		int		src_rank;
		int		dest_rank;
	};
	enum fi_datatype	datatype;
};

struct util_coll_copy_item {
	struct util_coll_hdr	hdr;
	void 			*in_buf;
	void			*out_buf;
	int			count;
	enum fi_datatype	datatype;
};

struct util_coll_reduce_item {
	struct util_coll_hdr	hdr;
	void 			*in_buf;
	void 			*inout_buf;
	int			count;
	enum fi_datatype	datatype;
	enum fi_op		op;
};

struct util_coll_comp_item;

typedef void (*util_coll_comp_t)(struct util_coll_mc *coll_mc,
				 struct util_coll_comp_item *comp);

struct util_coll_join_comp_data {
	uint64_t		cid_buf[OFI_CONTEXT_ID_SIZE];
	uint64_t		tmp_cid_buf[OFI_CONTEXT_ID_SIZE];
};

struct util_coll_comp_item {
	struct util_coll_hdr	hdr;
	enum util_coll_op_type	op_type;
	void			*data;
	util_coll_comp_t	comp_fn;
};

struct util_coll_mc {
	struct fid_mc		mc_fid;
	struct fid_ep		*ep;
	struct util_av_set 	*av_set;
	struct slist		barrier_list;
	struct slist		deferred_list;
	int 			my_rank;
	uint16_t		cid;
	uint16_t		tag_seq;
	ofi_atomic32_t		ref;
};

int ofi_join_collective(struct fid_ep *ep, fi_addr_t coll_addr,
			const struct fid_av_set *set, uint64_t flags,
			struct fid_mc **mc, void *context);

int ofi_av_set(struct fid_av *av, struct fi_av_set_attr *attr,
	       struct fid_av_set **av_set_fid, void * context);

ssize_t ofi_ep_barrier(struct fid_ep *ep, fi_addr_t coll_addr, void *context);

ssize_t ofi_ep_writeread(struct fid_ep *ep, const void *buf, size_t count,
		     void *desc, void *result, void *result_desc,
		     fi_addr_t coll_addr, enum fi_datatype datatype,
		     enum fi_op op, uint64_t flags, void *context);

ssize_t ofi_ep_writereadmsg(struct fid_ep *ep, const struct fi_msg_collective *msg,
			struct fi_ioc *resultv, void **result_desc,
			size_t result_count, uint64_t flags);


#endif // _OFI_COLL_H_
