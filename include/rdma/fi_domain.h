/*
 * Copyright (c) 2013-2017 Intel Corporation. All rights reserved.
 * (C) Copyright 2020 Hewlett Packard Enterprise Development LP
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

#ifndef FI_DOMAIN_H
#define FI_DOMAIN_H

#include <string.h>
#include <rdma/fabric.h>
#include <rdma/fi_eq.h>


#ifdef __cplusplus
extern "C" {
#endif


/*
 * AV = Address Vector
 * Maps and stores transport/network addresses.
 */

#define FI_SYMMETRIC		(1ULL << 59)
#define FI_SYNC_ERR		(1ULL << 58)
#define FI_UNIVERSE		(1ULL << 57)
#define FI_BARRIER_SET		(1ULL << 40)
#define FI_BROADCAST_SET	(1ULL << 41)
#define FI_ALLTOALL_SET		(1ULL << 42)
#define FI_ALLREDUCE_SET	(1ULL << 43)
#define FI_ALLGATHER_SET	(1ULL << 44)
#define FI_REDUCE_SCATTER_SET	(1ULL << 45)
#define FI_REDUCE_SET		(1ULL << 46)
#define FI_SCATTER_SET		(1ULL << 47)
#define FI_GATHER_SET		(1ULL << 48)

struct fi_av_attr {
	enum fi_av_type		type;
	int			rxCtxBits;
	size_t			count;
	size_t			epPerNode;
	const char		*name;
	void			*mapAddr;
	uint64_t		flags;
};

struct fi_av_set_attr {
	size_t			count;
	fi_addr_t		startAddr;
	fi_addr_t		endAddr;
	uint64_t		stride;
	size_t			commKeySize;
	uint8_t			*commKey;
	uint64_t		flags;
};

struct fid_av_set;

struct fi_ops_av {
	size_t	size;
	int	(*insert)(struct fid_av *av, const void *addr, size_t count,
			fi_addr_t *fiAddr, uint64_t flags, void *context);
	int	(*insertsvc)(struct fid_av *av, const char *node,
			const char *service, fi_addr_t *fiAddr,
			uint64_t flags, void *context);
	int	(*insertsym)(struct fid_av *av, const char *node, size_t nodecnt,
			const char *service, size_t svccnt, fi_addr_t *fiAddr,
			uint64_t flags, void *context);
	int	(*remove)(struct fid_av *av, fi_addr_t *fiAddr, size_t count,
			uint64_t flags);
	int	(*lookup)(struct fid_av *av, fi_addr_t fiAddr, void *addr,
			size_t *addrlen);
	const char * (*straddr)(struct fid_av *av, const void *addr,
			char *buf, size_t *len);
	int	(*avSet)(struct fid_av *av, struct fi_av_set_attr *attr,
			struct fid_av_set **avSet, void *context);
};

struct fid_av {
	struct fid		fid;
	struct fi_ops_av	*ops;
};


/*
 * MR = Memory Region
 * Tracks registered memory regions, primarily for remote access,
 * but also for local access until we can remove that need.
 */
struct fid_mr {
	struct fid		fid;
	void			*memDesc;
	uint64_t		key;
};

enum fi_hmem_iface {
	FI_HMEM_SYSTEM	= 0,
	FI_HMEM_CUDA,
	FI_HMEM_ROCR,
	FI_HMEM_ZE,
	FI_HMEM_NEURON,
	FI_HMEM_SYNAPSEAI,
};

static inline int fiHmemZeDevice(int driverIndex, int deviceIndex)
{
	return driverIndex << 16 | deviceIndex;
}

struct fi_mr_attr {
	const struct iovec	*mrIov;
	size_t			iovCount;
	uint64_t		access;
	uint64_t		offset;
	uint64_t		requestedKey;
	void			*context;
	size_t			authKeySize;
	uint8_t			*authKey;
	enum fi_hmem_iface	iface;
	union {
		uint64_t	reserved;
		int		cuda;
		int		ze;
		int		neuron;
		int		synapseai;
	} device;
};

struct fi_mr_modify {
	uint64_t		flags;
	struct fi_mr_attr	attr;
};

#define FI_SET_OPS_HMEM_OVERRIDE "hmem_override_ops"

struct fi_hmem_override_ops {
	size_t	size;

	ssize_t	(*copyFromHmemIov)(void *dest, size_t size,
				      enum fi_hmem_iface iface, uint64_t device,
				      const struct iovec *hmem_iov,
				      size_t hmemIovCount,
				      uint64_t hmemIovOffset);

	ssize_t (*copyToHmemIov)(enum fi_hmem_iface iface, uint64_t device,
				    const struct iovec *hmemIov,
				    size_t hmemIovCount,
				    uint64_t hmemIovOffset, const void *src,
				    size_t size);
};

#ifdef FABRIC_DIRECT
#include <rdma/fi_direct_atomic_def.h>
#endif /* FABRIC_DIRECT */

#ifndef FABRIC_DIRECT_ATOMIC_DEF

#define FI_COLLECTIVE_OFFSET 256

enum fi_datatype {
	FI_INT8,
	FI_UINT8,
	FI_INT16,
	FI_UINT16,
	FI_INT32,
	FI_UINT32,
	FI_INT64,
	FI_UINT64,
	FI_FLOAT,
	FI_DOUBLE,
	FI_FLOAT_COMPLEX,
	FI_DOUBLE_COMPLEX,
	FI_LONG_DOUBLE,
	FI_LONG_DOUBLE_COMPLEX,
	/* End of point to point atomic datatypes */
	FI_DATATYPE_LAST,
	/*
	 * enums for 128-bit integer atomics, existing ordering and
	 * FI_DATATYPE_LAST preserved for compatabilty.
	 */
	FI_INT128 = FI_DATATYPE_LAST,
	FI_UINT128,

	/* Collective datatypes */
	FI_VOID = FI_COLLECTIVE_OFFSET,
};

enum fi_op {
	FI_MIN,
	FI_MAX,
	FI_SUM,
	FI_PROD,
	FI_LOR,
	FI_LAND,
	FI_BOR,
	FI_BAND,
	FI_LXOR,
	FI_BXOR,
	FI_ATOMIC_READ,
	FI_ATOMIC_WRITE,
	FI_CSWAP,
	FI_CSWAP_NE,
	FI_CSWAP_LE,
	FI_CSWAP_LT,
	FI_CSWAP_GE,
	FI_CSWAP_GT,
	FI_MSWAP,
	/* End of point to point atomic ops */
	FI_ATOMIC_OP_LAST,

	/* Collective datatypes */
	FI_NOOP = FI_COLLECTIVE_OFFSET,
};

#endif

#ifndef FABRIC_DIRECT_COLLECTIVE_DEF

enum fi_collective_op {
	FI_BARRIER,
	FI_BROADCAST,
	FI_ALLTOALL,
	FI_ALLREDUCE,
	FI_ALLGATHER,
	FI_REDUCE_SCATTER,
	FI_REDUCE,
	FI_SCATTER,
	FI_GATHER,
};

#endif


struct fi_atomic_attr;
struct fi_cq_attr;
struct fi_cntr_attr;
struct fi_collective_attr;

struct fi_ops_domain {
	size_t	size;
	int	(*avOpen)(struct fid_domain *domain, struct fi_av_attr *attr,
			struct fid_av **av, void *context);
	int	(*cqOpen)(struct fid_domain *domain, struct fi_cq_attr *attr,
			struct fid_cq **cq, void *context);
	int	(*endpoint)(struct fid_domain *domain, struct fi_info *info,
			struct fid_ep **ep, void *context);
	int	(*scalableEp)(struct fid_domain *domain, struct fi_info *info,
			struct fid_ep **sep, void *context);
	int	(*cntrOpen)(struct fid_domain *domain, struct fi_cntr_attr *attr,
			struct fid_cntr **cntr, void *context);
	int	(*pollOpen)(struct fid_domain *domain, struct fi_poll_attr *attr,
			struct fid_poll **pollset);
	int	(*stxCtx)(struct fid_domain *domain,
			struct fi_tx_attr *attr, struct fid_stx **stx,
			void *context);
	int	(*srxCtx)(struct fid_domain *domain,
			struct fi_rx_attr *attr, struct fid_ep **rxEp,
			void *context);
	int	(*queryAtomic)(struct fid_domain *domain,
			enum fi_datatype datatype, enum fi_op op,
			struct fi_atomic_attr *attr, uint64_t flags);
	int	(*queryCollective)(struct fid_domain *domain,
			enum fi_collective_op coll,
			struct fi_collective_attr *attr, uint64_t flags);
	int	(*endpoint2)(struct fid_domain *domain, struct fi_info *info,
			struct fid_ep **ep, uint64_t flags, void *context);
};

/* Memory registration flags */
/* #define FI_RMA_EVENT		(1ULL << 56) */

struct fi_ops_mr {
	size_t	size;
	int	(*reg)(struct fid *fid, const void *buf, size_t len,
			uint64_t access, uint64_t offset, uint64_t requestedKey,
			uint64_t flags, struct fid_mr **mr, void *context);
	int	(*regv)(struct fid *fid, const struct iovec *iov,
			size_t count, uint64_t access,
			uint64_t offset, uint64_t requestedKey,
			uint64_t flags, struct fid_mr **mr, void *context);
	int	(*regattr)(struct fid *fid, const struct fi_mr_attr *attr,
			uint64_t flags, struct fid_mr **mr);
};

/* Domain bind flags */
#define FI_REG_MR		(1ULL << 59)

struct fid_domain {
	struct fid		fid;
	struct fi_ops_domain	*ops;
	struct fi_ops_mr	*mr;
};


#ifdef FABRIC_DIRECT
#include <rdma/fi_direct_domain.h>
#endif	/* FABRIC_DIRECT */

#ifndef FABRIC_DIRECT_DOMAIN

static inline int
fiDomain(struct fid_fabric *fabric, struct fi_info *info,
	   struct fid_domain **domain, void *context)
{
	return fabric->ops->domain(fabric, info, domain, context);
}

static inline int
fiDomain2(struct fid_fabric *fabric, struct fi_info *info,
	   struct fid_domain **domain, uint64_t flags, void *context)
{
	if (!flags)
		return fiDomain(fabric, info, domain, context);

	return FI_CHECK_OP(fabric->ops, struct fi_ops_fabric, domain2) ?
		fabric->ops->domain2(fabric, info, domain, flags, context) :
		-FI_ENOSYS;
}

static inline int
fiDomainBind(struct fid_domain *domain, struct fid *fid, uint64_t flags)
{
	return domain->fid.ops->bind(&domain->fid, fid, flags);
}

static inline int
fiCqOpen(struct fid_domain *domain, struct fi_cq_attr *attr,
	   struct fid_cq **cq, void *context)
{
	return domain->ops->cqOpen(domain, attr, cq, context);
}

static inline int
fiCntrOpen(struct fid_domain *domain, struct fi_cntr_attr *attr,
	      struct fid_cntr **cntr, void *context)
{
	return domain->ops->cntrOpen(domain, attr, cntr, context);
}

static inline int
fiWaitOpen(struct fid_fabric *fabric, struct fi_wait_attr *attr,
	     struct fid_wait **waitset)
{
	return fabric->ops->waitOpen(fabric, attr, waitset);
}

static inline int
fiPollOpen(struct fid_domain *domain, struct fi_poll_attr *attr,
	     struct fid_poll **pollset)
{
	return domain->ops->pollOpen(domain, attr, pollset);
}

static inline int
fiMrReg(struct fid_domain *domain, const void *buf, size_t len,
	  uint64_t acs, uint64_t offset, uint64_t requestedKey,
	  uint64_t flags, struct fid_mr **mr, void *context)
{
	return domain->mr->reg(&domain->fid, buf, len, acs, offset,
			       requestedKey, flags, mr, context);
}

static inline int
fiMrRegv(struct fid_domain *domain, const struct iovec *iov,
			size_t count, uint64_t acs,
			uint64_t offset, uint64_t requestedKey,
			uint64_t flags, struct fid_mr **mr, void *context)
{
	return domain->mr->regv(&domain->fid, iov, count, acs,
			offset, requestedKey, flags, mr, context);
}

static inline int
fiMrRegattr(struct fid_domain *domain, const struct fi_mr_attr *attr,
			uint64_t flags, struct fid_mr **mr)
{
	return domain->mr->regattr(&domain->fid, attr, flags, mr);
}

static inline void *fiMrDesc(struct fid_mr *mr)
{
	return mr->memDesc;
}

static inline uint64_t fiMrKey(struct fid_mr *mr)
{
	return mr->key;
}

static inline int
fiMrRawAttr(struct fid_mr *mr, uint64_t *baseAddr,
	       uint8_t *rawKey, size_t *keySize, uint64_t flags)
{
	struct fi_mr_raw_attr attr;
	attr.flags = flags;
	attr.baseAddr = baseAddr;
	attr.rawKey = rawKey;
	attr.keySize = keySize;
	return mr->fid.ops->control(&mr->fid, FI_GET_RAW_MR, &attr);
}

static inline int
fiMrMapRaw(struct fid_domain *domain, uint64_t baseAddr,
	      uint8_t *rawKey, size_t keySize, uint64_t *key, uint64_t flags)
{
	struct fi_mr_map_raw map;
	map.flags = flags;
	map.baseAddr = baseAddr;
	map.rawKey = rawKey;
	map.keySize = keySize;
	map.key = key;
	return domain->fid.ops->control(&domain->fid, FI_MAP_RAW_MR, &map);
}

static inline int
fiMrUnmapKey(struct fid_domain *domain, uint64_t key)
{
	return domain->fid.ops->control(&domain->fid, FI_UNMAP_KEY, &key);
}

static inline int fiMrBind(struct fid_mr *mr, struct fid *bfid, uint64_t flags)
{
	return mr->fid.ops->bind(&mr->fid, bfid, flags);
}

static inline int
fiMrRefresh(struct fid_mr *mr, const struct iovec *iov, size_t count,
	      uint64_t flags)
{
	struct fi_mr_modify modify;
	memset(&modify, 0, sizeof(modify));
	modify.flags = flags;
	modify.attr.mrIov = iov;
	modify.attr.iovCount = count;
	return mr->fid.ops->control(&mr->fid, FI_REFRESH, &modify);
}

static inline int fiMrEnable(struct fid_mr *mr)
{
	return mr->fid.ops->control(&mr->fid, FI_ENABLE, NULL);
}

static inline int
fiAvOpen(struct fid_domain *domain, struct fi_av_attr *attr,
	   struct fid_av **av, void *context)
{
	return domain->ops->avOpen(domain, attr, av, context);
}

static inline int
fiAvBind(struct fid_av *av, struct fid *fid, uint64_t flags)
{
	return av->fid.ops->bind(&av->fid, fid, flags);
}

static inline int
fiAvInsert(struct fid_av *av, const void *addr, size_t count,
	     fi_addr_t *fiAddr, uint64_t flags, void *context)
{
	return av->ops->insert(av, addr, count, fiAddr, flags, context);
}

static inline int
fiAvInsertsvc(struct fid_av *av, const char *node, const char *service,
		fi_addr_t *fiAddr, uint64_t flags, void *context)
{
	return av->ops->insertsvc(av, node, service, fiAddr, flags, context);
}

static inline int
fiAvInsertsym(struct fid_av *av, const char *node, size_t nodecnt,
		const char *service, size_t svccnt,
		fi_addr_t *fiAddr, uint64_t flags, void *context)
{
	return av->ops->insertsym(av, node, nodecnt, service, svccnt,
			fiAddr, flags, context);
}

static inline int
fiAvRemove(struct fid_av *av, fi_addr_t *fiAddr, size_t count, uint64_t flags)
{
	return av->ops->remove(av, fiAddr, count, flags);
}

static inline int
fiAvLookup(struct fid_av *av, fi_addr_t fiAddr, void *addr, size_t *addrlen)
{
        return av->ops->lookup(av, fiAddr, addr, addrlen);
}

static inline const char *
fiAvStraddr(struct fid_av *av, const void *addr, char *buf, size_t *len)
{
	return av->ops->straddr(av, addr, buf, len);
}

static inline fi_addr_t
fiRxAddr(fi_addr_t fiAddr, int rxIndex, int rxCtxBits)
{
	return (fi_addr_t) (((uint64_t) rxIndex << (64 - rxCtxBits)) | fiAddr);
}

#endif

#ifdef __cplusplus
}
#endif

#endif /* FI_DOMAIN_H */
