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

#ifndef FI_COLLECTIVE_H
#define FI_COLLECTIVE_H

#include <rdma/fi_atomic.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_cm.h>


#ifdef __cplusplus
extern "C" {
#endif

#ifdef FABRIC_DIRECT
#include <rdma/fi_direct_collective_def.h>
#endif /* FABRIC_DIRECT */


struct fi_ops_av_set {
	size_t	size;
	int	(*setUnion)(struct fid_av_set *dst,
			const struct fid_av_set *src);
	int	(*intersect)(struct fid_av_set *dst,
			const struct fid_av_set *src);
	int	(*diff)(struct fid_av_set *dst, const struct fid_av_set *src);
	int	(*insert)(struct fid_av_set *set, fi_addr_t addr);
	int	(*remove)(struct fid_av_set *set, fi_addr_t addr);
	int	(*addr)(struct fid_av_set *set, fi_addr_t *collAddr);
};

struct fid_av_set {
	struct fid		fid;
	struct fi_ops_av_set	*ops;
};

struct fi_collective_attr {
	enum fi_op 		op;
	enum fi_datatype 	datatype;
	struct fi_atomic_attr 	datatypeAttr;
	size_t 			maxMembers;
	uint64_t 		mode;
};

struct fi_collective_addr {
	const struct fid_av_set	*set;
	fi_addr_t		collAddr;
};

struct fi_msg_collective {
	const struct fi_ioc	*msgIov;
	void			**desc;
	size_t			iovCount;
	fi_addr_t		collAddr;
	fi_addr_t		rootAddr;
	enum fi_collective_op	coll;
	enum fi_datatype	datatype;
	enum fi_op		op;
	void			*context;
};

struct fi_ops_collective {
	size_t	size;

	ssize_t	(*barrier)(struct fid_ep *ep, fi_addr_t collAddr, void *context);
	ssize_t	(*broadcast)(struct fid_ep *ep,
			void *buf, size_t count, void *desc,
			fi_addr_t collAddr, fi_addr_t rootAddr,
			enum fi_datatype datatype, uint64_t flags, void *context);
	ssize_t	(*alltoall)(struct fid_ep *ep,
			const void *buf, size_t count, void *desc,
			void *result, void *resultDesc, fi_addr_t collAddr,
			enum fi_datatype datatype, uint64_t flags, void *context);
	ssize_t	(*allreduce)(struct fid_ep *ep,
			const void *buf, size_t count, void *desc,
			void *result, void *resultDesc, fi_addr_t collAddr,
			enum fi_datatype datatype, enum fi_op op,
			uint64_t flags, void *context);
	ssize_t	(*allgather)(struct fid_ep *ep,
			const void *buf, size_t count, void *desc,
			void *result, void *resultDesc, fi_addr_t collAddr,
			enum fi_datatype datatype, uint64_t flags, void *context);
	ssize_t	(*reduceScatter)(struct fid_ep *ep,
			const void *buf, size_t count, void *desc,
			void *result, void *resultDesc, fi_addr_t collAddr,
			enum fi_datatype datatype, enum fi_op op,
			uint64_t flags, void *context);
	ssize_t	(*reduce)(struct fid_ep *ep,
			const void *buf, size_t count, void *desc,
			void *result, void *resultDesc, fi_addr_t collAddr,
			fi_addr_t rootAddr, enum fi_datatype datatype, enum fi_op op,
			uint64_t flags, void *context);
	ssize_t	(*scatter)(struct fid_ep *ep,
			const void *buf, size_t count, void *desc,
			void *result, void *resultDesc,
			fi_addr_t collAddr, fi_addr_t rootAddr,
			enum fi_datatype datatype, uint64_t flags, void *context);
	ssize_t	(*gather)(struct fid_ep *ep,
			const void *buf, size_t count, void *desc,
			void *result, void *resultDesc,
			fi_addr_t collAddr, fi_addr_t rootAddr,
			enum fi_datatype datatype, uint64_t flags, void *context);
	ssize_t	(*msg)(struct fid_ep *ep,
			const struct fi_msg_collective *msg,
			struct fi_ioc *resultv, void **resultDesc,
			size_t resultCount, uint64_t flags);
	ssize_t	(*barrier2)(struct fid_ep *ep, fi_addr_t collAddr, uint64_t flags,
			void *context);
};


#ifdef FABRIC_DIRECT
#include <rdma/fi_direct_collective.h>
#endif /* FABRIC_DIRECT */

#ifndef FABRIC_DIRECT_COLLECTIVE

static inline int
fiAvSet(struct fid_av *av, struct fi_av_set_attr *attr,
	  struct fid_av_set **set, void * context)
{
	return FI_CHECK_OP(av->ops, struct fi_ops_av, avSet) ?
		av->ops->avSet(av, attr, set, context) : -FI_ENOSYS;
}

static inline int
fiAvSetUnion(struct fid_av_set *dst, const struct fid_av_set *src)
{
	return dst->ops->setUnion(dst, src);
}

static inline int
fiAvSetIntersect(struct fid_av_set *dst, const struct fid_av_set *src)
{
	return dst->ops->intersect(dst, src);
}

static inline int
fiAvSetDiff(struct fid_av_set *dst, const struct fid_av_set *src)
{
	return dst->ops->diff(dst, src);
}

static inline int
fiAvSetInsert(struct fid_av_set *set, fi_addr_t addr)
{
	return set->ops->insert(set, addr);
}

static inline int
fiAvSetRemove(struct fid_av_set *set, fi_addr_t addr)
{
	return set->ops->remove(set, addr);
}

static inline int
fiAvSetAddr(struct fid_av_set *set, fi_addr_t *collAddr)
{
	return set->ops->addr(set, collAddr);
}

static inline int
fiJoinCollective(struct fid_ep *ep, fi_addr_t collAddr,
		   const struct fid_av_set *set,
		   uint64_t flags, struct fid_mc **mc, void *context)
{
	struct fi_collective_addr addr;

	addr.set = set;
	addr.collAddr = collAddr;
	return fiJoin(ep, &addr, flags | FI_COLLECTIVE, mc, context);
}

static inline ssize_t
fiBarrier(struct fid_ep *ep, fi_addr_t collAddr, void *context)
{
	return ep->collective->barrier(ep, collAddr, context);
}

static inline ssize_t
fiBarrier2(struct fid_ep *ep, fi_addr_t collAddr, uint64_t flags, void *context)
{
	if (!flags)
		return fiBarrier(ep, collAddr, context);

	return FI_CHECK_OP(ep->collective, struct fi_ops_collective, barrier2) ?
		ep->collective->barrier2(ep, collAddr, flags, context) :
		-FI_ENOSYS;
}

static inline ssize_t
fiBroadcast(struct fid_ep *ep, void *buf, size_t count, void *desc,
	     fi_addr_t collAddr, fi_addr_t rootAddr,
	     enum fi_datatype datatype, uint64_t flags, void *context)
{
	return ep->collective->broadcast(ep, buf, count, desc,
		collAddr, rootAddr, datatype, flags, context);
}

static inline ssize_t
fiAlltoall(struct fid_ep *ep, const void *buf, size_t count, void *desc,
	    void *result, void *resultDesc,
	    fi_addr_t collAddr, enum fi_datatype datatype,
	    uint64_t flags, void *context)
{
	return ep->collective->alltoall(ep, buf, count, desc,
		result, resultDesc, collAddr, datatype, flags, context);
}

static inline ssize_t
fiAllreduce(struct fid_ep *ep, const void *buf, size_t count, void *desc,
	     void *result, void *resultDesc, fi_addr_t collAddr,
	     enum fi_datatype datatype, enum fi_op op,
	     uint64_t flags, void *context)
{
	return ep->collective->allreduce(ep, buf, count, desc,
		result, resultDesc, collAddr, datatype, op, flags, context);
}

static inline ssize_t
fiAllgather(struct fid_ep *ep, const void *buf, size_t count, void *desc,
	     void *result, void *resultDesc, fi_addr_t collAddr,
	     enum fi_datatype datatype, uint64_t flags, void *context)
{
	return ep->collective->allgather(ep, buf, count, desc,
		result, resultDesc, collAddr, datatype, flags, context);
}

static inline ssize_t
fiReduceScatter(struct fid_ep *ep, const void *buf, size_t count, void *desc,
		  void *result, void *resultDesc, fi_addr_t collAddr,
		  enum fi_datatype datatype, enum fi_op op,
		  uint64_t flags, void *context)
{
	return ep->collective->reduceScatter(ep, buf, count, desc,
		result, resultDesc, collAddr, datatype, op, flags, context);
}

static inline ssize_t
fiReduce(struct fid_ep *ep, const void *buf, size_t count, void *desc,
	  void *result, void *resultDesc, fi_addr_t collAddr,
	  fi_addr_t rootAddr, enum fi_datatype datatype, enum fi_op op,
	  uint64_t flags, void *context)
{
	return ep->collective->reduce(ep, buf, count, desc, result, resultDesc,
		collAddr, rootAddr, datatype, op, flags, context);
}


static inline ssize_t
fiScatter(struct fid_ep *ep, const void *buf, size_t count, void *desc,
	   void *result, void *resultDesc, fi_addr_t collAddr,
	   fi_addr_t rootAddr, enum fi_datatype datatype,
	   uint64_t flags, void *context)
{
	return ep->collective->scatter(ep, buf, count, desc, result, resultDesc,
		collAddr, rootAddr, datatype, flags, context);
}


static inline ssize_t
fiGather(struct fid_ep *ep, const void *buf, size_t count, void *desc,
	  void *result, void *resultDesc, fi_addr_t collAddr,
	  fi_addr_t rootAddr, enum fi_datatype datatype,
	  uint64_t flags, void *context)
{
	return ep->collective->gather(ep, buf, count, desc, result, resultDesc,
		collAddr, rootAddr, datatype, flags, context);
}

static inline
int fiQueryCollective(struct fid_domain *domain, enum fi_collective_op coll,
			struct fi_collective_attr *attr, uint64_t flags)
{
	return FI_CHECK_OP(domain->ops, struct fi_ops_domain, queryCollective) ?
		       domain->ops->queryCollective(domain, coll, attr, flags) :
		       -FI_ENOSYS;
}

#endif

#ifdef __cplusplus
}
#endif

#endif /* FI_COLLECTIVE_H */
