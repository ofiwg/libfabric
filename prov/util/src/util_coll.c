/*
 * Copyright (c) 2019 Intel Corporation. All rights reserved.
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

#include "config.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netdb.h>
#include <netinet/in.h>
#include <inttypes.h>

#if HAVE_GETIFADDRS
#include <net/if.h>
#include <ifaddrs.h>
#endif

#include <ofi_util.h>

#include <rdma/fi_collective.h>
#include <rdma/fi_cm.h>
#include <ofi_list.h>
#include <ofi_atomic.h>
#include <ofi_coll.h>
#include <ofi_osd.h>

static uint64_t util_coll_cid[OFI_CONTEXT_ID_SIZE];
/* TODO: if collective support is requested, initialize up front
 * when opening the domain or EP
 */
static bool util_coll_cid_initialized = 0;


int ofi_av_set_union(struct fid_av_set *dst, const struct fid_av_set *src)
{
	struct util_av_set *src_av_set;
	struct util_av_set *dst_av_set;
	size_t temp_count;
	int i,j;

	src_av_set = container_of(src, struct util_av_set, av_set_fid);
	dst_av_set = container_of(dst, struct util_av_set, av_set_fid);

	assert(src_av_set->av == dst_av_set->av);
	temp_count = dst_av_set->fi_addr_count;

	for (i = 0; i < src_av_set->fi_addr_count; i++) {
		for (j = 0; j < dst_av_set->fi_addr_count; j++) {
			if (dst_av_set->fi_addr_array[j] ==
			    src_av_set->fi_addr_array[i])
				break;
		}
		if (j == dst_av_set->fi_addr_count) {
			dst_av_set->fi_addr_array[temp_count++] =
				src_av_set->fi_addr_array[i];
		}
	}

	dst_av_set->fi_addr_count = temp_count;
	return FI_SUCCESS;
}

int ofi_av_set_intersect(struct fid_av_set *dst, const struct fid_av_set *src)
{
	struct util_av_set *src_av_set;
	struct util_av_set *dst_av_set;
	int i,j, temp;

	src_av_set = container_of(src, struct util_av_set, av_set_fid);
	dst_av_set = container_of(dst, struct util_av_set, av_set_fid);

	assert(src_av_set->av == dst_av_set->av);

	temp = 0;
	for (i = 0; i < src_av_set->fi_addr_count; i++) {
		for (j = temp; j < dst_av_set->fi_addr_count; j++) {
			if (dst_av_set->fi_addr_array[j] ==
			    src_av_set->fi_addr_array[i]) {
				dst_av_set->fi_addr_array[temp++] =
					dst_av_set->fi_addr_array[j];
				break;
			}
		}
	}
	dst_av_set->fi_addr_count = temp;
	return FI_SUCCESS;
}

int ofi_av_set_diff(struct fid_av_set *dst, const struct fid_av_set *src)
{

	struct util_av_set *src_av_set;
	struct util_av_set *dst_av_set;
	int i,j, temp;

	src_av_set = container_of(src, struct util_av_set, av_set_fid);
	dst_av_set = container_of(dst, struct util_av_set, av_set_fid);

	assert(src_av_set->av == dst_av_set->av);

	temp = dst_av_set->fi_addr_count;
	for (i = 0; i < src_av_set->fi_addr_count; i++) {
		for (j = 0; j < temp; j++) {
			if (dst_av_set->fi_addr_array[j] ==
			    src_av_set->fi_addr_array[i]) {
				dst_av_set->fi_addr_array[--temp] =
					dst_av_set->fi_addr_array[j];
				break;
			}
		}
	}
	dst_av_set->fi_addr_count = temp;
	return FI_SUCCESS;
}

int ofi_av_set_insert(struct fid_av_set *set, fi_addr_t addr)
{
	struct util_av_set *av_set;
	int i;

	av_set = container_of(set, struct util_av_set, av_set_fid);

	for (i = 0; i < av_set->fi_addr_count; i++) {
		if (av_set->fi_addr_array[i] == addr)
			return -FI_EINVAL;
	}
	av_set->fi_addr_array[av_set->fi_addr_count++] = addr;
	return FI_SUCCESS;
}

int ofi_av_set_remove(struct fid_av_set *set, fi_addr_t addr)

{
	struct util_av_set *av_set;
	int i;

	av_set = container_of(set, struct util_av_set, av_set_fid);

	for (i = 0; i < av_set->fi_addr_count; i++) {
		if (av_set->fi_addr_array[i] == addr) {
			av_set->fi_addr_array[i] =
				av_set->fi_addr_array[--av_set->fi_addr_count];
			return FI_SUCCESS;
		}
	}
	return -FI_EINVAL;
}

static inline void util_coll_init_cid(uint64_t *cid)
{
	int i;

	for (i = 0; i < OFI_CONTEXT_ID_SIZE; i++) {
		cid[i] = -1;
	}

	/* reserving the first bit in context id to whole av set */
	cid[0] &= ~0x1;
}

static inline int util_coll_mc_alloc(struct util_coll_mc **coll_mc)
{

	*coll_mc = calloc(1, sizeof(**coll_mc));
	if (!*coll_mc)
		return -FI_ENOMEM;

	slist_init(&(*coll_mc)->barrier_list);
	slist_init(&(*coll_mc)->deferred_list);
	return FI_SUCCESS;
}

/* TODO: see ofi.h for power of 2 helper */
static inline int util_coll_pof2(int num)
{
	int pof2 = 1;

	while (pof2 <= num)
		pof2 <<= 1;

	return (pof2 >> 1);
}

static int util_coll_sched_send(struct util_coll_mc *coll_mc, int dest,
				void *buf, int count, enum fi_datatype datatype,
				uint64_t tag, int is_barrier)
{
	struct util_coll_xfer_item *xfer_item;

	xfer_item = calloc(1, sizeof(*xfer_item));
	if (!xfer_item)
		return -FI_ENOMEM;

	xfer_item->hdr.type = UTIL_COLL_SEND;
	xfer_item->hdr.is_barrier = is_barrier;
	xfer_item->hdr.tag = tag;
	xfer_item->buf = buf;
	xfer_item->count = count;
	xfer_item->datatype = datatype;
	xfer_item->dest_rank = dest;

	slist_insert_tail(&xfer_item->hdr.entry, &coll_mc->deferred_list);
	return FI_SUCCESS;
}

static int util_coll_sched_recv(struct util_coll_mc *coll_mc, int src,
				void *buf, int count, enum fi_datatype datatype,
				uint64_t tag, int is_barrier)
{
	struct util_coll_xfer_item *xfer_item;
	uint64_t recv_tag = src;

	xfer_item = calloc(1, sizeof(*xfer_item));
	if (!xfer_item)
		return -FI_ENOMEM;

	xfer_item->hdr.type = UTIL_COLL_RECV;
	xfer_item->hdr.is_barrier = is_barrier;

	xfer_item->hdr.tag = (recv_tag << 32) | (tag & 0xffffffff);
	xfer_item->buf = buf;
	xfer_item->count = count;
	xfer_item->datatype = datatype;
	xfer_item->src_rank = src;

	slist_insert_tail(&xfer_item->hdr.entry, &coll_mc->deferred_list);
	return FI_SUCCESS;
}

static int util_coll_sched_reduce(struct util_coll_mc *coll_mc, void *in_buf,
				  void *inout_buf, int count,
				  enum fi_datatype datatype, enum fi_op op,
				  int is_barrier)
{
	struct util_coll_reduce_item *reduce_item;

	reduce_item = calloc(1, sizeof(*reduce_item));
	if (!reduce_item)
		return -FI_ENOMEM;

	reduce_item->hdr.type = UTIL_COLL_REDUCE;
	reduce_item->hdr.is_barrier = is_barrier;
	reduce_item->in_buf = in_buf;
	reduce_item->inout_buf = inout_buf;
	reduce_item->count = count;
	reduce_item->datatype = datatype;
	reduce_item->op = op;

	slist_insert_tail(&reduce_item->hdr.entry, &coll_mc->deferred_list);
	return FI_SUCCESS;
}

static int util_coll_sched_copy(struct util_coll_mc *coll_mc, void *in_buf,
				void *out_buf, int count,
				enum fi_datatype datatype, int is_barrier)
{
	struct util_coll_copy_item *copy_item;

	copy_item = calloc(1, sizeof(*copy_item));
	if (!copy_item)
		return -FI_ENOMEM;

	copy_item->hdr.type = UTIL_COLL_COPY;
	copy_item->hdr.is_barrier = is_barrier;
	copy_item->in_buf = in_buf;
	copy_item->out_buf = out_buf;
	copy_item->count = count;
	copy_item->datatype = datatype;

	slist_insert_tail(&copy_item->hdr.entry, &coll_mc->deferred_list);
	return FI_SUCCESS;
}

static int util_coll_sched_comp(struct util_coll_mc *coll_mc,
				enum util_coll_op_type op_type,
				void *data, util_coll_comp_t comp_fn)
{
	struct util_coll_comp_item *comp_item;

	comp_item = calloc(1, sizeof(*comp_item));
	if (!comp_item)
		return -FI_ENOMEM;

	comp_item->hdr.type = UTIL_COLL_COMP;
	comp_item->hdr.is_barrier = 0;

	comp_item->op_type = op_type;
	comp_item->data = data;
	comp_item->comp_fn = comp_fn;

	slist_insert_tail(&comp_item->hdr.entry, &coll_mc->deferred_list);
	return FI_SUCCESS;
}

static inline uint64_t util_coll_get_next_tag(struct util_coll_mc *coll_mc)
{
	uint64_t tag = coll_mc->my_rank;

	tag = tag << 16 | coll_mc->cid;
	tag = tag << 16 | coll_mc->tag_seq++;
	return tag;
}

/* TODO: when this fails, clean up the already scheduled work in this function */
static int util_coll_allreduce(struct util_coll_mc *coll_mc, void *send_buf,
			void *recv_buf, int count, enum fi_datatype datatype,
			enum fi_op op)
{
	uint64_t tag;
	int rem, pof2, my_new_id;
	int dest, new_dest;
	int ret;
	int mask = 1;

	tag = util_coll_get_next_tag(coll_mc);
	pof2 = util_coll_pof2(coll_mc->av_set->fi_addr_count);
	rem = coll_mc->av_set->fi_addr_count - pof2;

	if (coll_mc->my_rank < 2 * rem) {
		if (coll_mc->my_rank % 2 == 0) {
			ret = util_coll_sched_send(coll_mc, coll_mc->my_rank + 1,
						   send_buf, count, datatype,
						   tag, BARRIER);
			if (ret)
				return ret;

			my_new_id = -1;
		} else {
			ret = util_coll_sched_recv(coll_mc, coll_mc->my_rank - 1,
						   recv_buf, count, datatype,
						   tag, BARRIER);
			if (ret)
				return ret;

			my_new_id = coll_mc->my_rank / 2;

			ret = util_coll_sched_reduce(coll_mc, recv_buf, send_buf,
						     count, datatype, op, BARRIER);
			if (ret)
				return ret;
		}
	} else {
		my_new_id = coll_mc->my_rank - rem;
	}

	if (my_new_id != -1) {
		while (mask < pof2) {
			new_dest = my_new_id ^ mask;
			dest = (new_dest < rem) ? new_dest * 2 + 1 :
				new_dest + rem;

			ret = util_coll_sched_recv(coll_mc, dest, recv_buf,
						   count, datatype, tag, NO_BARRIER);
			if (ret)
				return ret;
			ret = util_coll_sched_send(coll_mc, dest, send_buf,
						   count, datatype, tag, BARRIER);
			if (ret)
				return ret;

			if (dest < coll_mc->my_rank) {
				ret = util_coll_sched_reduce(coll_mc, recv_buf, send_buf,
							     count, datatype, op, BARRIER);
				if (ret)
					return ret;

			} else {
				ret = util_coll_sched_reduce(coll_mc, send_buf, recv_buf,
							     count, datatype, op, BARRIER);
				if (ret)
					return ret;

				ret = util_coll_sched_copy(coll_mc, recv_buf, send_buf,
							   count, datatype, BARRIER);
				if (ret)
					return ret;
			}
			mask <<= 1;
		}
	}

	if (coll_mc->my_rank < 2 * rem) {
		if (coll_mc->my_rank % 2) {
			ret = util_coll_sched_send(coll_mc, coll_mc->my_rank - 1,
						   send_buf, count, datatype,
						   tag, BARRIER);
			if (ret)
				return ret;
		} else {
			ret = util_coll_sched_recv(coll_mc, coll_mc->my_rank + 1,
						   send_buf, count, datatype,
						   tag, BARRIER);
			if (ret)
				return ret;
		}
	}
	return FI_SUCCESS;
}

static int util_coll_close(struct fid *fid)
{
	struct util_coll_mc *coll_mc;

	coll_mc = container_of(fid, struct util_coll_mc,
			       mc_fid.fid);
	free(coll_mc);
	return FI_SUCCESS;
}

static struct fi_ops util_coll_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = util_coll_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

/* TODO: Figure out requirements for using collectives.
 * e.g. require local address to be in AV?
 * Determine best way to handle first join request
 */
static int util_coll_find_my_rank(struct fid_ep *ep,
				  struct util_coll_mc *coll_mc)
{
	size_t addrlen;
	char *addr;
	int ret, mem;

	addrlen = sizeof(mem);
	addr = (char *) &mem;

	ret = fi_getname(&ep->fid, addr, &addrlen);
	if (ret != -FI_ETOOSMALL) {
		return ret;
	}

	addr = calloc(1, addrlen);
	if (!addr)
		return -FI_ENOMEM;

	ret = fi_getname(&ep->fid, addr, &addrlen);
	if (ret) {
		free(addr);
		return ret;
	}
	coll_mc->my_rank =
		ofi_av_lookup_fi_addr(coll_mc->av_set->av, addr);

	return FI_SUCCESS;
}

void util_coll_join_comp(struct util_coll_mc *coll_mc,
			 struct util_coll_comp_item *comp)
{
	struct fi_eq_err_entry entry;
	struct util_coll_join_comp_data *comp_data;
	struct util_ep *ep;
	ssize_t bytes;
	uint64_t tmp;
	int iter, lsb_set_pos = 0, pos;

	comp_data = (struct util_coll_join_comp_data *)comp->data;

	for (iter = 0; iter < OFI_CONTEXT_ID_SIZE; iter++) {

		if (comp_data->cid_buf[iter]) {
			tmp = comp_data->cid_buf[iter];
			pos = 0;
			while (!(tmp & 0x1)) {
				tmp >>= 1;
				pos++;
			}

			/* clear the bit from global cid space */
			util_coll_cid[iter] ^= (1 << pos);
			lsb_set_pos += pos;
		} else {
			lsb_set_pos += sizeof(comp_data->cid_buf[0]) * 8;
		}
	}
	assert(lsb_set_pos < OFI_CONTEXT_ID_SIZE * 8);
	coll_mc->cid = lsb_set_pos;
	coll_mc->tag_seq = 0;
	free(comp_data);

	/* write to the eq  */
	memset(&entry, 0, sizeof(entry));
	entry.fid = &coll_mc->mc_fid.fid;
	entry.context = coll_mc->mc_fid.fid.context;
	bytes = sizeof(struct fi_eq_entry);

	ep  = container_of(coll_mc->ep, struct util_ep, ep_fid);
	if (ofi_eq_write(&ep->eq->eq_fid, FI_JOIN_COMPLETE,
			 &entry, (size_t) bytes, FI_COLLECTIVE) < 0)
		FI_WARN(ep->domain->fabric->prov, FI_LOG_DOMAIN,
			"join collective - eq write failed\n");
}

void util_coll_barrier_comp(struct util_coll_mc *coll_mc,
			    struct util_coll_comp_item *comp)
{
/* TODO: We should write a completion to the CQ, not the EQ
	struct fi_eq_err_entry entry;
	struct util_ep *ep;
	ssize_t bytes;

	memset(&entry, 0, sizeof(entry));
	entry.fid = &coll_mc->mc_fid.fid;
	entry.context = coll_mc->mc_fid.fid.context;
	bytes = sizeof(struct fi_eq_entry);
	free(comp->data);

	ep  = container_of(coll_mc->ep, struct util_ep, ep_fid);
	if (ofi_eq_write(&ep->eq->eq_fid, FI_COLL,
			 &entry, (size_t) bytes, FI_COLLECTIVE) < 0)
		FI_WARN(ep->domain->fabric->prov, FI_LOG_DOMAIN,
			"barrier collective - eq write failed\n");
*/
}

/* TODO: We can probably use the atomic defines for these, rather than
 * duplicate the definitions here
 */
#define UTIL_COLL_REDUCE_FUNC(src,dst,cnt,OP,TYPE)	\
		do { \
			int i; \
			TYPE *d = (dst); \
			TYPE *s = (src); \
			for (i=0; i<(cnt); i++) {\
				OP(d[i],s[i]); \
			} \
		} while (0)

#define SWITCH_REDUCE(type,FUNC,src,dst,cnt,OP)		\
switch (type) {							 \
		case FI_INT8:	FUNC(src,dst,cnt,OP,int8_t); break; \
		case FI_UINT8:	FUNC(src,dst,cnt,OP,uint8_t); break; \
		case FI_INT16:	FUNC(src,dst,cnt,OP,int16_t); break; \
		case FI_UINT16: FUNC(src,dst,cnt,OP,uint16_t); break; \
		case FI_INT32:	FUNC(src,dst,cnt,OP,int32_t); break; \
		case FI_UINT32: FUNC(src,dst,cnt,OP,uint32_t); break; \
		case FI_INT64:	FUNC(src,dst,cnt,OP,int64_t); break; \
		case FI_UINT64: FUNC(src,dst,cnt,OP,uint64_t); break; \
		default: return -FI_EOPNOTSUPP; \
		}

#define UTIL_COLL_MIN(dst,src)	if ((dst) > (src)) (dst) = (src)
#define UTIL_COLL_MAX(dst,src)	if ((dst) < (src)) (dst) = (src)
#define UTIL_COLL_SUM(dst,src)	(dst) += (src)
#define UTIL_COLL_PROD(dst,src)	(dst) *= (src)
#define UTIL_COLL_LOR(dst,src)	(dst) = (dst) || (src)
#define UTIL_COLL_LAND(dst,src)	(dst) = (dst) && (src)
#define UTIL_COLL_BOR(dst,src)	(dst) |= (src)
#define UTIL_COLL_BAND(dst,src)	(dst) &= (src)
#define UTIL_COLL_LXOR(dst,src)	(dst) = ((dst) && !(src)) || (!(dst) && (src))
#define UTIL_COLL_BXOR(dst,src)	(dst) ^= (src)

static int util_coll_proc_reduce_item(struct util_coll_mc *coll_mc,
				      struct util_coll_reduce_item *reduce_item)
{
	switch (reduce_item->op) {
	case FI_MIN:
		SWITCH_REDUCE(reduce_item->datatype, UTIL_COLL_REDUCE_FUNC,
			      reduce_item->in_buf, reduce_item->inout_buf,
			      reduce_item->count, UTIL_COLL_MIN);
		break;
	case FI_MAX:
		SWITCH_REDUCE(reduce_item->datatype, UTIL_COLL_REDUCE_FUNC,
			      reduce_item->in_buf, reduce_item->inout_buf,
			      reduce_item->count, UTIL_COLL_MAX);
		break;
	case FI_SUM:
		SWITCH_REDUCE(reduce_item->datatype, UTIL_COLL_REDUCE_FUNC,
			      reduce_item->in_buf, reduce_item->inout_buf,
			      reduce_item->count, UTIL_COLL_SUM);
		break;
	case FI_PROD:
		SWITCH_REDUCE(reduce_item->datatype, UTIL_COLL_REDUCE_FUNC,
			      reduce_item->in_buf, reduce_item->inout_buf,
			      reduce_item->count, UTIL_COLL_PROD);
		break;
	case FI_LOR:
		SWITCH_REDUCE(reduce_item->datatype, UTIL_COLL_REDUCE_FUNC,
			      reduce_item->in_buf, reduce_item->inout_buf,
			      reduce_item->count, UTIL_COLL_LOR);
		break;
	case FI_LAND:
		SWITCH_REDUCE(reduce_item->datatype, UTIL_COLL_REDUCE_FUNC,
			      reduce_item->in_buf, reduce_item->inout_buf,
			      reduce_item->count, UTIL_COLL_LAND);
		break;
	case FI_BOR:
		SWITCH_REDUCE(reduce_item->datatype, UTIL_COLL_REDUCE_FUNC,
			      reduce_item->in_buf, reduce_item->inout_buf,
			      reduce_item->count, UTIL_COLL_BOR);
		break;
	case FI_BAND:
		SWITCH_REDUCE(reduce_item->datatype, UTIL_COLL_REDUCE_FUNC,
			      reduce_item->in_buf, reduce_item->inout_buf,
			      reduce_item->count, UTIL_COLL_BAND);
		break;
	case FI_LXOR:
		SWITCH_REDUCE(reduce_item->datatype, UTIL_COLL_REDUCE_FUNC,
			      reduce_item->in_buf, reduce_item->inout_buf,
			      reduce_item->count, UTIL_COLL_LXOR);
		break;
	case FI_BXOR:
		SWITCH_REDUCE(reduce_item->datatype, UTIL_COLL_REDUCE_FUNC,
			      reduce_item->in_buf, reduce_item->inout_buf,
			      reduce_item->count, UTIL_COLL_BXOR);
		break;
	case FI_ATOMIC_READ:
	case FI_ATOMIC_WRITE:
	case FI_CSWAP:
	case FI_CSWAP_NE:
	case FI_CSWAP_LE:
	case FI_CSWAP_LT:
	case FI_CSWAP_GE:
	case FI_CSWAP_GT:
	case FI_MSWAP:
	case FI_ATOMIC_OP_LAST:
	case FI_BARRIER:
	case FI_BROADCAST:
	case FI_ALLTOALL:
	case FI_ALLGATHER:
	default:
		return -FI_ENOSYS;
	};
	return FI_SUCCESS;
}

static int util_coll_process_work_items(struct util_coll_mc *coll_mc)
{
	struct util_coll_hdr *hdr;
	struct util_coll_xfer_item *xfer_item;
	struct util_coll_reduce_item *reduce_item;
	struct util_coll_copy_item *copy_item;
	struct util_coll_comp_item *comp_item;
	struct slist_entry *entry;
	struct fi_msg_tagged msg = {0};
	struct iovec iov;
	int ret;

	while (!slist_empty(&coll_mc->deferred_list)) {
		entry = slist_remove_head(&coll_mc->deferred_list);
		hdr = container_of(entry, struct util_coll_hdr, entry);
		switch (hdr->type) {
		case UTIL_COLL_SEND:
			xfer_item = (struct util_coll_xfer_item *) hdr;
			iov.iov_base = xfer_item->buf;
			iov.iov_len = (xfer_item->count *
				       ofi_datatype_size(xfer_item->datatype));
			msg.msg_iov = &iov;
			msg.iov_count = 1;
			msg.addr = coll_mc->av_set->fi_addr_array[xfer_item->dest_rank];
			msg.tag = hdr->tag;
			msg.context = (void *) coll_mc;
			do {
				ret = fi_tsendmsg(coll_mc->ep, &msg, FI_COLLECTIVE);
				if (ret && ret != -EAGAIN) {
					slist_insert_head(&xfer_item->hdr.entry,
							  &coll_mc->deferred_list);
					return ret;
				}
			} while (ret == -EAGAIN);

			slist_insert_tail(entry, &coll_mc->barrier_list);
			break;
		case UTIL_COLL_RECV:
			xfer_item = (struct util_coll_xfer_item *) hdr;
			iov.iov_base = xfer_item->buf;
			iov.iov_len = (xfer_item->count *
				       ofi_datatype_size(xfer_item->datatype));
			msg.msg_iov = &iov;
			msg.iov_count = 1;
			msg.addr = coll_mc->av_set->fi_addr_array[xfer_item->src_rank];
			msg.tag = hdr->tag;
			msg.context = (void *) coll_mc;

			ret = fi_trecvmsg(coll_mc->ep, &msg, FI_COLLECTIVE);
			if (ret) {
				return ret;
			}

			slist_insert_tail(entry, &coll_mc->barrier_list);
			break;
		case UTIL_COLL_REDUCE:
			reduce_item = (struct util_coll_reduce_item *) hdr;
			ret = util_coll_proc_reduce_item(coll_mc, reduce_item);
			free(reduce_item);
			if (ret)
				return ret;
			break;
		case UTIL_COLL_COPY:
			copy_item = (struct util_coll_copy_item *) hdr;
			memcpy(copy_item->out_buf, copy_item->in_buf,
			       copy_item->count *
			       ofi_datatype_size(copy_item->datatype));
			free(copy_item);
			break;
		case UTIL_COLL_COMP:
			comp_item = (struct util_coll_comp_item *) hdr;
			if (comp_item->comp_fn)
				comp_item->comp_fn(coll_mc, comp_item);
			free(comp_item);
			break;
		default:
			break;
		}

		if (hdr->is_barrier &&
		    !slist_empty(&coll_mc->barrier_list)) {
			break;
		}
	}
	return FI_SUCCESS;
}

static int util_coll_schedule(struct util_coll_mc *coll_mc)
{
	int ret;

	if (slist_empty(&coll_mc->barrier_list)) {
		ret = util_coll_process_work_items(coll_mc);
		if (ret)
			return ret;
	}
	return FI_SUCCESS;
}

int ofi_join_collective(struct fid_ep *ep, fi_addr_t coll_addr,
		       const struct fid_av_set *set,
		       uint64_t flags, struct fid_mc **mc, void *context)
{
	struct util_coll_mc *new_coll_mc;
	struct util_av_set *av_set;
	struct util_coll_mc *coll_mc;
	struct util_coll_join_comp_data *comp_data;
	int ret;

	av_set = container_of(set, struct util_av_set, av_set_fid);

	if (coll_addr == FI_ADDR_NOTAVAIL) {
		assert(av_set->av->coll_mc != NULL);
		coll_mc = av_set->av->coll_mc;
	} else {
		coll_mc = (struct util_coll_mc *) coll_addr;
	}

	ret = util_coll_mc_alloc(&new_coll_mc);
	if (ret)
		return ret;

	if (util_coll_cid_initialized == 0) {
		util_coll_init_cid(util_coll_cid);
		util_coll_cid_initialized = 1;
	}

	new_coll_mc->mc_fid.fid.fclass = FI_CLASS_MC;
	new_coll_mc->mc_fid.fid.context = context;
	new_coll_mc->mc_fid.fid.ops = &util_coll_fi_ops;
	new_coll_mc->mc_fid.fi_addr = (uintptr_t) new_coll_mc;

	new_coll_mc->ep = ep;
	coll_mc->ep = ep;
	new_coll_mc->av_set = av_set;

	/* get the rank */
	util_coll_find_my_rank(ep, new_coll_mc);
	util_coll_find_my_rank(ep, coll_mc);

	comp_data = calloc(1, sizeof(*comp_data));
	if (!comp_data) {
		ret =  -FI_ENOMEM;
		goto err1;
	}

	if (new_coll_mc->my_rank != FI_ADDR_NOTAVAIL) {
		memcpy(comp_data->cid_buf, util_coll_cid,
		       OFI_CONTEXT_ID_SIZE * sizeof(uint64_t));
	} else {
		util_coll_init_cid(comp_data->cid_buf);
	}

	ret = util_coll_allreduce(coll_mc, comp_data->cid_buf,
				  comp_data->tmp_cid_buf,
				  OFI_CONTEXT_ID_SIZE, FI_INT64,
				  FI_BAND);
	if (ret)
		goto err2;

	ret = util_coll_sched_comp(coll_mc, UTIL_COLL_JOIN_OP,
			     comp_data, util_coll_join_comp);
	if (ret)
		goto err2;

	*mc = &new_coll_mc->mc_fid;
	util_coll_schedule(coll_mc);
	return FI_SUCCESS;

err2:
	free(comp_data);
err1:
	free(new_coll_mc);
	return ret;
}

static struct fi_ops_av_set util_av_set_ops= {
	.set_union	= 	ofi_av_set_union,
	.intersect	=	ofi_av_set_intersect,
	.diff		=	ofi_av_set_diff,
	.insert		=	ofi_av_set_insert,
	.remove		=	ofi_av_set_remove,
};

static int util_coll_copy_from_av(struct util_av *av, void *addr,
			      fi_addr_t fi_addr, void *arg)
{
	struct util_av_set *av_set = (struct util_av_set *) arg;
	av_set->fi_addr_array[av_set->fi_addr_count++] = fi_addr;
	return FI_SUCCESS;
}

static int util_coll_av_init(struct util_av *av)
{

	struct util_coll_mc *coll_mc;
	int ret;

	assert(!av->coll_mc);

	if (util_coll_cid_initialized == 0) {
		util_coll_init_cid(util_coll_cid);
		util_coll_cid_initialized = 1;
	}

	ret = util_coll_mc_alloc(&coll_mc);
	if (ret)
		return ret;

	coll_mc->av_set = calloc(1, sizeof(*coll_mc->av_set));
	if (!coll_mc->av_set) {
		ret = -FI_ENOMEM;
		goto err1;
	}

	coll_mc->av_set->fi_addr_array =
		calloc(av->count, sizeof(*coll_mc->av_set->fi_addr_array));
	if (!coll_mc->av_set->fi_addr_array) {
		ret = -FI_ENOMEM;
		goto err2;
	}

	ret = fastlock_init(&coll_mc->av_set->lock);
	if (ret)
		goto err3;

	coll_mc->av_set->av = av;
	ret = ofi_av_elements_iter(av, util_coll_copy_from_av,
				   (void *)coll_mc->av_set);
	if (ret)
		goto err4;

	coll_mc->av_set->av_set_fid.fid.fclass = FI_CLASS_AV_SET;
	coll_mc->av_set->av_set_fid.ops = &util_av_set_ops;

	coll_mc->mc_fid.fi_addr = (uintptr_t) coll_mc;
	coll_mc->mc_fid.fid.fclass = FI_CLASS_MC;
	coll_mc->mc_fid.fid.context = NULL;
	coll_mc->mc_fid.fid.ops = &util_coll_fi_ops;
	av->coll_mc = coll_mc;
	return FI_SUCCESS;

err4:
	fastlock_destroy(&coll_mc->av_set->lock);
err3:
	free(coll_mc->av_set->fi_addr_array);
err2:
	free(coll_mc->av_set);
err1:
	free(coll_mc);
	return ret;
}

int ofi_av_set(struct fid_av *av, struct fi_av_set_attr *attr,
	       struct fid_av_set **av_set_fid, void * context)
{
	struct util_av *util_av = container_of(av, struct util_av, av_fid);
	struct util_av_set *av_set;
	int ret, iter;

	if (!util_av->coll_mc) {
		ret = util_coll_av_init(util_av);
		if (ret)
			return ret;
	}

	av_set = calloc(1,sizeof(*av_set));
	if (!av_set)
		return -FI_ENOMEM;

	ret = fastlock_init(&av_set->lock);
	if (ret)
		goto err1;

	av_set->fi_addr_array =
		calloc(util_av->count, sizeof(*av_set->fi_addr_array));
	if (!av_set->fi_addr_array)
		goto err2;

	for (iter = 0; iter < attr->count; iter++) {

		av_set->fi_addr_array[iter] =
			util_av->coll_mc->av_set->fi_addr_array[iter * attr->stride];
		av_set->fi_addr_count++;
	}

	av_set->av = util_av;
	av_set->av_set_fid.ops = &util_av_set_ops;
	av_set->av_set_fid.fid.fclass = FI_CLASS_AV_SET;
	av_set->av_set_fid.fid.context = context;
	(*av_set_fid) = &av_set->av_set_fid;
	return FI_SUCCESS;
err2:
	fastlock_destroy(&av_set->lock);
err1:
	free(av_set);
	return ret;
}

ssize_t ofi_ep_barrier(struct fid_ep *ep, fi_addr_t coll_addr, void *context)
{
	struct util_coll_mc *coll_mc = (struct util_coll_mc *) coll_addr;
	struct util_coll_comp_item *comp_item;
	int ret;

	comp_item = calloc(1, sizeof(*comp_item));
	if (!comp_item)
		return -FI_ENOMEM;

	comp_item->data = (void *) calloc(1, sizeof(uint32_t));
	if (!comp_item->data) {
		ret = -FI_ENOMEM;
		goto err1;
	}

	ret = util_coll_allreduce(coll_mc, comp_item->data,
				  comp_item->data, 1, FI_UINT32,
				  FI_BAND);
	if (ret)
		goto err2;

	util_coll_sched_comp(coll_mc, UTIL_COLL_BARRIER_OP,
			     NULL, util_coll_barrier_comp);

	util_coll_schedule(coll_mc);
	return FI_SUCCESS;
err2:
	free(comp_item->data);
err1:
	free(comp_item);
	return ret;
}

static int util_coll_match_tag(struct slist_entry *entry, const void *arg)
{
	struct util_coll_xfer_item *item;
	uint64_t *tag_ptr = (uint64_t *) arg;

	item = container_of(entry, struct util_coll_xfer_item, hdr.entry);
	if (item->hdr.tag == *tag_ptr)
		return 1;

	return 0;
}

void util_coll_handle_comp(uint64_t tag, void *ctx)
{
	struct util_coll_mc *coll_mc = (struct util_coll_mc *) ctx;
	struct util_coll_xfer_item *item;
	struct slist_entry *entry;
	uint64_t *tag_ptr = &tag;

	entry = slist_remove_first_match(&coll_mc->barrier_list,
					 util_coll_match_tag,
					 (void *) tag_ptr);
	if (!entry) {
		item = container_of(entry, struct util_coll_xfer_item,
				    hdr.entry);
		free(item);
	}

	util_coll_schedule(coll_mc);
}

ssize_t ofi_ep_writeread(struct fid_ep *ep, const void *buf, size_t count,
		     void *desc, void *result, void *result_desc,
		     fi_addr_t coll_addr, enum fi_datatype datatype,
		     enum fi_op op, uint64_t flags, void *context)
{
	return -FI_ENOSYS;
}


ssize_t ofi_ep_writereadmsg(struct fid_ep *ep, const struct fi_msg_collective *msg,
			struct fi_ioc *resultv, void **result_desc,
			size_t result_count, uint64_t flags)
{
	return -FI_ENOSYS;
}
