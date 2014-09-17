/*
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenFabrics.org BSD license below:
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

#ifndef _FI_DOMAIN_H_
#define _FI_DOMAIN_H_

#include <rdma/fabric.h>
#include <rdma/fi_eq.h>


#ifdef __cplusplus
extern "C" {
#endif


/*
 * AV = Address Vector
 * Maps and stores transport/network addresses.
 */

#define FI_RANGE		(1ULL << 0)

enum fi_av_type {
	FI_AV_MAP,
	FI_AV_TABLE
};

struct fi_av_attr {
	enum fi_av_type		type;
	size_t			count;
	const char		*name;
	void			*map_addr;
	uint64_t		flags;
};

struct fi_ops_av {
	size_t	size;
	int	(*insert)(struct fid_av *av, const void *addr, size_t count,
			fi_addr_t *fi_addr, uint64_t flags);
	int	(*remove)(struct fid_av *av, fi_addr_t *fi_addr, size_t count,
			uint64_t flags);
	int	(*lookup)(struct fid_av *av, fi_addr_t fi_addr, void *addr,
			size_t *addrlen);
	const char * (*straddr)(struct fid_av *av, const void *addr,
			char *buf, size_t *len);
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
	void			*mem_desc;
	uint64_t		key;
};

struct fi_mr_attr {
	const struct iovec	*mr_iov;
	size_t			iov_count;
	uint64_t		access;
	uint64_t		offset;
	uint64_t		requested_key;
	void			*context;
};


struct fi_cq_attr;
struct fi_cntr_attr;


/* fi_info domain capabilities */
#define FI_WRITE_COHERENT	(1ULL << 0)
#define FI_CONTEXT		(1ULL << 1)
#define FI_LOCAL_MR		(1ULL << 2)
#define FI_USER_MR_KEY		(1ULL << 3)
#define FI_DYNAMIC_MR		(1ULL << 4)


struct fi_ops_domain {
	size_t	size;
	int	(*query)(struct fid_domain *domain, struct fi_domain_attr *attr);
	int	(*av_open)(struct fid_domain *domain, struct fi_av_attr *attr,
			struct fid_av **av, void *context);
	int	(*cq_open)(struct fid_domain *domain, struct fi_cq_attr *attr,
			struct fid_cq **cq, void *context);
	int	(*endpoint)(struct fid_domain *domain, struct fi_info *info,
			struct fid_ep **ep, void *context);
	int	(*cntr_open)(struct fid_domain *domain, struct fi_cntr_attr *attr,
			struct fid_cntr **cntr, void *context);
	int	(*wait_open)(struct fid_domain *domain, struct fi_wait_attr *attr,
			struct fid_wait **waitset);
	int	(*poll_open)(struct fid_domain *domain, struct fi_poll_attr *attr,
			struct fid_poll **pollset);
};


/* Memory registration flags */
#define FI_MR_OFFSET		(1ULL << 0)
#define FI_MR_KEY		(1ULL << 3)	/* FI_USER_MR_KEY */

struct fi_ops_mr {
	size_t	size;
	int	(*reg)(struct fid_domain *domain, const void *buf, size_t len,
			uint64_t access, uint64_t offset, uint64_t requested_key,
			uint64_t flags, struct fid_mr **mr, void *context);
	int	(*regv)(struct fid_domain *domain, const struct iovec *iov,
			size_t count, uint64_t access,
			uint64_t offset, uint64_t requested_key,
			uint64_t flags, struct fid_mr **mr, void *context);
	int	(*regattr)(struct fid_domain *domain, const struct fi_mr_attr *attr,
			uint64_t flags, struct fid_mr **mr);
};

/* Domain bind flags */
#define FI_REG_MR		(1ULL << 0)

struct fid_domain {
	struct fid		fid;
	struct fi_ops_domain	*ops;
	struct fi_ops_mr	*mr;
};


#ifndef FABRIC_DIRECT

static inline int
fi_fdomain(struct fid_fabric *fabric, struct fi_domain_attr *attr,
	   struct fid_domain **domain, void *context)
{
	return fabric->ops->domain(fabric, attr, domain, context);
}

static inline int
fi_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
	   struct fid_cq **cq, void *context)
{
	return domain->ops->cq_open(domain, attr, cq, context);
}

static inline int
fi_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
	      struct fid_cntr **cntr, void *context)
{
	return domain->ops->cntr_open(domain, attr, cntr, context);
}

static inline int
fi_mr_reg(struct fid_domain *domain, const void *buf, size_t len,
	  uint64_t access, uint64_t offset, uint64_t requested_key,
	  uint64_t flags, struct fid_mr **mr, void *context)
{
	return domain->mr->reg(domain, buf, len, access, offset,
			       requested_key, flags, mr, context);
}

static inline void *fi_mr_desc(struct fid_mr *mr)
{
	return mr->mem_desc;
}

static inline uint64_t fi_mr_key(struct fid_mr *mr)
{
	return mr->key;
}

static inline int
fi_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
	   struct fid_av **av, void *context)
{
	return domain->ops->av_open(domain, attr, av, context);
}

static inline int
fi_av_insert(struct fid_av *av, const void *addr, size_t count,
	     fi_addr_t *fi_addr, uint64_t flags)
{
	return av->ops->insert(av, addr, count, fi_addr, flags);
}
#define fi_av_map(av, addr, count, fi_addr, flags) \
	fi_av_insert(av, addr, count, fi_addr, flags)

static inline int
fi_av_remove(struct fid_av *av, fi_addr_t *fi_addr, size_t count, uint64_t flags)
{
	return av->ops->remove(av, fi_addr, count, flags);
}

static inline int
fi_av_lookup(struct fid_av *av, fi_addr_t fi_addr, void *addr, size_t *addrlen)
{
        return av->ops->lookup(av, fi_addr, addr, addrlen);
}

static inline int fi_av_sync(struct fid_av *av, uint64_t flags, void *context)
{
	return fi_sync(&av->fid, flags, context);
}


#else // FABRIC_DIRECT
#include <rdma/fi_direct_domain.h>
#endif

#ifdef __cplusplus
}
#endif

#endif /* _FI_DOMAIN_H_ */
