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


#ifdef __cplusplus
extern "C" {
#endif


/*
 * AV = Address Vector
 * Maps and stores transport/network addresses.
 */

#define FI_RANGE		(1ULL << 0)

struct fi_av_addr {
	struct fid_av		*av;
	uint64_t		av_index;
};

enum fi_av_type {
	FI_AV_MAP,
	FI_AV_TABLE
};

enum {
	FI_AV_ATTR_TYPE		= 1 << 0,
	FI_AV_ATTR_COUNT	= 1 << 1,
	FI_AV_ATTR_FLAGS	= 1 << 2,
	FI_AV_ATTR_MASK_V1	= (FI_AV_ATTR_FLAGS << 1) - 1
};

struct fi_av_attr {
	int			mask;
	enum fi_av_type		type;
	size_t			count;
	uint64_t		flags;
};

struct fi_ops_av {
	size_t	size;
	int	(*insert)(struct fid_av *av, const void *addr, size_t count,
			void **fi_addr, uint64_t flags);
	int	(*remove)(struct fid_av *av, void *fi_addr, size_t count,
			uint64_t flags);
	int	(*lookup)(struct fid_av *av, const void *fi_addr, void *addr,
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


/*
 * EQ = Event Queue
 * Used to report various events and the completion of asynchronous
 * operations.
 */
#define FI_AUTO_RESET		(1ULL << 1)

/* #define FI_TRUNC		(1ULL << 1) */
/* #define FI_CTRUNC		(1ULL << 2) */

enum fi_eq_domain {
	FI_EQ_DOMAIN_GENERAL,
	FI_EQ_DOMAIN_COMP,
	FI_EQ_DOMAIN_CM,
	FI_EQ_DOMAIN_AV
};

enum fi_eq_format {
	FI_EQ_FORMAT_UNSPEC,
	FI_EQ_FORMAT_CONTEXT,
	FI_EQ_FORMAT_COMP,
	FI_EQ_FORMAT_DATA,
	FI_EQ_FORMAT_TAGGED,
	FI_EQ_FORMAT_CM,
};

/* Use fi_control GETWAIT to get underlying wait object */
enum fi_eq_wait_obj {
	FI_EQ_WAIT_NONE,
	FI_EQ_WAIT_FD
};

enum fi_eq_wait_cond {
	FI_EQ_COND_NONE,
	FI_EQ_COND_THRESHOLD	/* size_t threshold */
};

enum {
	FI_EQ_ATTR_DOMAIN	= 1 << 0,
	FI_EQ_ATTR_FORMAT	= 1 << 1,
	FI_EQ_ATTR_WAIT_OBJ	= 1 << 2,
	FI_EQ_ATTR_WAIT_COND	= 1 << 3,
	FI_EQ_ATTR_SIZE		= 1 << 4,
	FI_EQ_ATTR_VECTOR	= 1 << 5,
	FI_EQ_ATTR_FLAGS	= 1 << 6,
	FI_EQ_ATTR_COND		= 1 << 7,
	FI_EQ_ATTR_MASK_V1	= (FI_EQ_ATTR_COND << 1) - 1
};

struct fi_eq_attr {
	int			mask;
	enum fi_eq_domain	domain;
	enum fi_eq_format	format;
	enum fi_eq_wait_obj	wait_obj;
	enum fi_eq_wait_cond	wait_cond;
	size_t			size;
	int			signaling_vector;
	uint64_t		flags;
	/* If AUTO_RESET is enabled, and wait_cond is not NONE */
	void			*cond;
};

struct fi_eq_entry {
	void			*op_context;
};

struct fi_eq_comp_entry {
	void			*op_context;
	uint64_t		flags;
	size_t			len;
};

struct fi_eq_data_entry {
	void			*op_context;
	void			*buf;
	uint64_t		flags;
	size_t			len;
	/* data depends on operation and/or flags - e.g. immediate data */
	uint64_t		data;
};

struct fi_eq_tagged_entry {
	void			*op_context;
	void			*buf;
	uint64_t		flags;
	size_t			len;
	uint64_t		data;
	uint64_t		tag;
};

struct fi_eq_err_entry {
	void			*op_context;
	union {
		void		*fid_context;
		void		*buf;
	};
	uint64_t		flags;
	size_t			len;
	size_t			olen;
	uint64_t		data;
	int			err;
	int			prov_errno;
	/* prov_data is available until the next time the EQ is read */
	void			*prov_data;
};

enum fi_cm_event {
	FI_CONNREQ,
	FI_CONNECTED,
	FI_SHUTDOWN
};

struct fi_eq_cm_entry {
	void			*fid_context;
	uint64_t		flags;
	enum fi_cm_event	event;
	/* user must call fi_freeinfo to release info */
	struct fi_info		*info;
	/* connection data placed here, up to space provided */
	uint8_t			data[0];
};

struct fi_ops_eq {
	size_t	size;
	ssize_t	(*read)(struct fid_eq *eq, void *buf, size_t len);
	ssize_t	(*readfrom)(struct fid_eq *eq, void *buf, size_t len,
			void *src_addr, size_t *addrlen);
	ssize_t	(*readerr)(struct fid_eq *eq, struct fi_eq_err_entry *buf,
			size_t len, uint64_t flags);
	ssize_t	(*write)(struct fid_eq *eq, const void *buf, size_t len);
	int	(*reset)(struct fid_eq *eq, const void *cond);
	ssize_t	(*condread)(struct fid_eq *eq, void *buf, size_t len,
			const void *cond);
	ssize_t	(*condreadfrom)(struct fid_eq *eq, void *buf, size_t len,
			void *src_addr, size_t *addrlen, const void *cond);
	const char * (*strerror)(struct fid_eq *eq, int prov_errno,
			const void *prov_data, void *buf, size_t len);
};

struct fid_eq {
	struct fid		fid;
	struct fi_ops_eq	*ops;
};

enum fi_cntr_events {
	FI_CNTR_EVENTS_COMP
};

/* Use fi_control GETWAIT to get underlying wait object */
enum fi_cntr_wait_obj {
	FI_CNTR_WAIT_NONE,
	FI_CNTR_WAIT_MUT_COND /* pthread mutex & cond */
};

enum {
	FI_CNTR_ATTR_EVENTS	= 1 << 0,
	FI_CNTR_ATTR_WAIT_OBJ	= 1 << 1,
	FI_CNTR_ATTR_FLAGS	= 1 << 2,
	FI_CNTR_ATTR_MASK_V1	= (FI_CNTR_ATTR_FLAGS << 1) - 1
};

struct fi_cntr_attr {
	int			mask;
	enum fi_cntr_events	events;
	enum fi_cntr_wait_obj	wait_obj;
	uint64_t		flags;
};

struct fi_ops_cntr {
	size_t	size;
	uint64_t (*read)(struct fid_cntr *cntr);
	int	(*add)(struct fid_cntr *cntr, uint64_t value);
	int	(*set)(struct fid_cntr *cntr, uint64_t value);
	int	(*wait)(struct fid_cntr *cntr, uint64_t threshold);
};

struct fid_cntr {
	struct fid		fid;
	struct fi_ops_cntr	*ops;
};


enum {
	FI_MR_ATTR_IOV		= 1 << 0,
	FI_MR_ATTR_ACCESS	= 1 << 1,
	FI_MR_ATTR_KEY		= 1 << 2,
	FI_MR_ATTR_CONTEXT	= 1 << 3,
	FI_MR_ATTR_MASK_V1	= (FI_MR_ATTR_CONTEXT << 1) - 1
};

struct fi_mr_attr {
	int			mask;
	const struct iovec	*mr_iov;
	size_t			iov_count;
	uint64_t		access;
	uint64_t		requested_key;
	void			*context;
};


/* fi_info domain capabilities */
#define FI_WRITE_COHERENT	(1ULL << 0)
#define FI_CONTEXT		(1ULL << 1)
#define FI_LOCAL_MR		(1ULL << 2)
#define FI_USER_MR_KEY		(1ULL << 3)


/*
 * The thought is that domain attributes should be relative to what it can
 * provide to the applications, and is not intended as a set of available
 * hardware limits.
 */
struct fi_domain_attr {
	/* Note to providers: set prov_attr to static struct */
	size_t			prov_attr_size;
	void			*prov_attr;
	size_t			mem_desc_size;
};

struct fi_ops_domain {
	size_t	size;
	int	(*query)(struct fid_domain *domain, struct fi_domain_attr *attr,
			size_t *attrlen);
	int	(*av_open)(struct fid_domain *domain, struct fi_av_attr *attr,
			struct fid_av **av, void *context);
	int	(*eq_open)(struct fid_domain *domain, struct fi_eq_attr *attr,
			struct fid_eq **eq, void *context);
	int	(*endpoint)(struct fid_domain *domain, struct fi_info *info,
			struct fid_ep **ep, void *context);
	int	(*if_open)(struct fid_domain *domain, const char *name,
			uint64_t flags, struct fid **fif, void *context);
	int	(*cntr_alloc)(struct fid_domain *domain, struct fi_cntr_attr *attr,
			struct fid_cntr **cntr, void *context);
};

struct fi_ops_mr {
	size_t	size;
	int	(*reg)(struct fid_domain *domain, const void *buf, size_t len,
			uint64_t access, uint64_t requested_key,
			uint64_t flags, struct fid_mr **mr, void *context);
	int	(*regv)(struct fid_domain *domain, const struct iovec *iov,
			size_t count, uint64_t access, uint64_t requested_key,
			uint64_t flags, struct fid_mr **mr, void *context);
	int	(*regattr)(struct fid_domain *domain, const struct fi_mr_attr *attr,
			uint64_t flags, struct fid_mr **mr);
};

struct fid_domain {
	struct fid		fid;
	struct fi_ops_domain	*ops;
	struct fi_ops_mr	*mr;
};


#ifndef FABRIC_DIRECT

static inline int
fi_fdomain(struct fid_fabric *fabric, struct fi_info *info,
	   struct fid_domain **domain, void *context)
{
	return fabric->ops->domain(fabric, info, domain, context);
}

static inline int
fi_fec_open(struct fid_fabric *fabric, const struct fi_eq_attr *attr,
	    struct fid_eq **eq, void *context)
{
	return fabric->ops->eq_open(fabric, attr, eq, context);
}

static inline int
fi_eq_open(struct fid_domain *domain, struct fi_eq_attr *attr,
	   struct fid_eq **eq, void *context)
{
	return domain->ops->eq_open(domain, attr, eq, context);
}

static inline ssize_t fi_eq_read(struct fid_eq *eq, void *buf, size_t len)
{
	return eq->ops->read(eq, buf, len);
}

static inline ssize_t
fi_eq_readfrom(struct fid_eq *eq, void *buf, size_t len,
	       void *src_addr, size_t *addrlen)
{
	return eq->ops->readfrom(eq, buf, len, src_addr, addrlen);
}

static inline ssize_t
fi_eq_readerr(struct fid_eq *eq, struct fi_eq_err_entry *buf, size_t len,
	      uint64_t flags)
{
	return eq->ops->readerr(eq, buf, len, flags);
}

static inline int fi_eq_reset(struct fid_eq *eq, void *cond)
{
	return eq->ops->reset(eq, cond);
}

static inline const char *
fi_eq_strerror(struct fid_eq *eq, int prov_errno, void *prov_data,
	       void *buf, size_t len)
{
	return eq->ops->strerror(eq, prov_errno, prov_data, buf, len);
}

static inline int
fi_cntr_alloc(struct fid_domain *domain, struct fi_cntr_attr *attr,
	      struct fid_cntr **cntr, void *context)
{
	return domain->ops->cntr_alloc(domain, attr, cntr, context);
}

static inline uint64_t fi_cntr_read(struct fid_cntr *cntr)
{
	return cntr->ops->read(cntr);
}

static inline int fi_cntr_add(struct fid_cntr *cntr, uint64_t value)
{
	return cntr->ops->add(cntr, value);
}

static inline int fi_cntr_set(struct fid_cntr *cntr, uint64_t value)
{
	return cntr->ops->set(cntr, value);
}

static inline int fi_cntr_wait(struct fid_cntr *cntr, uint64_t threshold)
{
	return cntr->ops->wait(cntr, threshold);
}

static inline int
fi_mr_reg(struct fid_domain *domain, const void *buf, size_t len,
	  uint64_t access, uint64_t requested_key,
	  uint64_t flags, struct fid_mr **mr, void *context)
{
	return domain->mr->reg(domain, buf, len, access, requested_key,
			       flags, mr, context);
}

static inline void *fi_mr_desc(struct fid_mr *mr)
{
	return mr->mem_desc;
}

static inline uint64_t fi_mr_key(struct fid_mr *mr)
{
	return mr->key;
}

static inline int fi_mr_unreg(struct fid_mr *mr)
{
	return fi_close(&mr->fid);
}

static inline int
fi_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
	   struct fid_av **av, void *context)
{
	return domain->ops->av_open(domain, attr, av, context);
}

static inline int
fi_av_map(struct fid_av *av, const void *addr, size_t count,
	  void **fi_addr, uint64_t flags)
{
	return av->ops->insert(av, addr, count, fi_addr, flags);
}

static inline int
fi_av_unmap(struct fid_av *av, void *fi_addr, size_t count, uint64_t flags)
{
	return av->ops->remove(av, fi_addr, count, flags);
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
