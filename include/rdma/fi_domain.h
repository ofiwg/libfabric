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
	FI_AV_ATTR_ADDR_FORMAT	= 1 << 1,
	FI_AV_ATTR_ADDRLEN	= 1 << 2,
	FI_AV_ATTR_COUNT	= 1 << 3,
	FI_AV_ATTR_FLAGS	= 1 << 4,
	FI_AV_ATTR_MASK_V1	= (FI_AV_ATTR_FLAGS << 1) - 1
};

struct fi_av_attr {
	int			mask;
	enum fi_av_type		type;
	enum fi_addr_format	addr_format; /* TODO: remove */
	size_t			addrlen;     /* TODO: remove */
	size_t			count;
	uint64_t		flags;
};

struct fi_ops_av {
	size_t	size;
	int	(*insert)(struct fid_av *av, const void *addr, size_t count,
			void **fi_addr, uint64_t flags);
	int	(*remove)(struct fid_av *av, void *fi_addr, size_t count,
			uint64_t flags);
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
 * EC = Event Collector
 * Used to report various events and the completion of asynchronous
 * operations.
 */
#define FI_AUTO_RESET		(1ULL << 1)

/* #define FI_TRUNC		(1ULL << 1) */
/* #define FI_CTRUNC		(1ULL << 2) */

enum fi_ec_domain {
	FI_EC_DOMAIN_GENERAL,
	FI_EC_DOMAIN_COMP,
	FI_EC_DOMAIN_CM,
	FI_EC_DOMAIN_AV
};

enum fi_ec_type {
	FI_EC_QUEUE,
	FI_EC_COUNTER
};

enum fi_ec_format {
	FI_EC_FORMAT_UNSPEC,
	FI_EC_FORMAT_CONTEXT,
	FI_EC_FORMAT_COMP,
	FI_EC_FORMAT_DATA,
	FI_EC_FORMAT_TAGGED,
	FI_EC_FORMAT_ERR,
	FI_EC_FORMAT_COMP_ERR,
	FI_EC_FORMAT_DATA_ERR,
	FI_EC_FORMAT_TAGGED_ERR,
	FI_EC_FORMAT_CM,
	FI_EC_FORMAT_COUNTER,
	FI_EC_FORMAT_COUNTER_ERR,
};

/* Use fi_control GETECWAIT to get underlying wait object */
enum fi_ec_wait_obj {
	FI_EC_WAIT_NONE,
	FI_EC_WAIT_FD
};

enum fi_ec_wait_cond {
	FI_EC_COND_NONE,
	FI_EC_COND_THRESHOLD	/* size_t threshold */
};

enum {
	FI_EC_ATTR_DOMAIN	= 1 << 0,
	FI_EC_ATTR_TYPE		= 1 << 1,
	FI_EC_ATTR_FORMAT	= 1 << 2,
	FI_EC_ATTR_WAIT_OBJ	= 1 << 3,
	FI_EC_ATTR_WAIT_COND	= 1 << 4,
	FI_EC_ATTR_SIZE		= 1 << 5,
	FI_EC_ATTR_VECTOR	= 1 << 6,
	FI_EC_ATTR_FLAGS	= 1 << 7,
	FI_EC_ATTR_COND		= 1 << 8,
	FI_EC_ATTR_MASK_V1	= (FI_EC_ATTR_COND << 1) - 1
};

struct fi_ec_attr {
	int			mask;
	enum fi_ec_domain	domain;
	enum fi_ec_type		type;
	enum fi_ec_format	format;
	enum fi_ec_wait_obj	wait_obj;
	enum fi_ec_wait_cond	wait_cond;
	size_t			size;
	int			signaling_vector;
	uint64_t		flags;
	/* If AUTO_RESET is enabled, and wait_cond is not NONE */
	void			*cond;
};

struct fi_ec_entry {
	void			*op_context;
};

struct fi_ec_comp_entry {
	void			*op_context;
	uint64_t		flags;
	size_t			len;
};

struct fi_ec_data_entry {
	void			*op_context;
	void			*buf;
	uint64_t		flags;
	size_t			len;
	/* data depends on operation and/or flags - e.g. immediate data */
	uint64_t		data;
};

struct fi_ec_tagged_entry {
	void			*op_context;
	void			*buf;
	uint64_t		flags;
	size_t			len;
	uint64_t		data;
	uint64_t		tag;
	size_t			olen;
};

struct fi_ec_err_entry {
	void			*op_context;
	union {
		void		*fid_context;
		void		*buf;
	};
	uint64_t		flags;
	size_t			len;
	uint64_t		data;
	int			err;
	int			prov_errno;
	/* prov_data is available until the next time the EQ is read */
	void			*prov_data;
};

struct fi_ec_tagged_err_entry {
	int			status;
	union {
		struct fi_ec_tagged_entry	tagged;
		struct fi_ec_err_entry		err;
	};
};

struct fi_ec_counter_entry {
	uint64_t		events;
};

struct fi_ec_counter_err_entry {
	uint64_t		events;
	uint64_t		errors;
};

enum fi_cm_event {
	FI_CONNREQ,
	FI_CONNECTED,
	FI_SHUTDOWN
};

struct fi_ec_cm_entry {
	void			*fid_context;
	uint64_t		flags;
	enum fi_cm_event	event;
	/* user must call fi_freeinfo to release info */
	struct fi_info		*info;
	/* connection data placed here, up to space provided */
	uint8_t			data[0];
};

struct fi_ops_ec {
	size_t	size;
	ssize_t	(*read)(struct fid_ec *ec, void *buf, size_t len);
	ssize_t	(*readfrom)(struct fid_ec *ec, void *buf, size_t len,
			void *src_addr, size_t *addrlen);
	ssize_t	(*readerr)(struct fid_ec *ec, void *buf, size_t len,
			uint64_t flags);
	ssize_t	(*write)(struct fid_ec *ec, const void *buf, size_t len);
	int	(*reset)(struct fid_ec *ec, const void *cond);
	ssize_t	(*condread)(struct fid_ec *ec, void *buf, size_t len,
			const void *cond);
	ssize_t	(*condreadfrom)(struct fid_ec *ec, void *buf, size_t len,
			void *src_addr, size_t *addrlen, const void *cond);
	const char * (*strerror)(struct fid_ec *ec, int prov_errno,
			const void *prov_data, void *buf, size_t len);
};

struct fid_ec {
	struct fid		fid;
	struct fi_ops_ec	*ops;
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


enum fi_progress {
	FI_PROGRESS_AUTO,
	FI_PROGRESS_INDIRECT,	/* progress possible through any domain call */
	FI_PROGRESS_EXPLICIT	/* user must explicitly request progress */
};

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
	enum fi_progress	progress;
};

struct fi_ops_domain {
	size_t	size;
	int	(*progress)(struct fid_domain *domain);
	int	(*query)(struct fid_domain *domain, struct fi_domain_attr *attr,
			size_t *attrlen);
	int	(*av_open)(struct fid_domain *domain, struct fi_av_attr *attr,
			struct fid_av **av, void *context);
	int	(*ec_open)(struct fid_domain *domain, struct fi_ec_attr *attr,
			struct fid_ec **ec, void *context);
	int	(*endpoint)(struct fid_domain *domain, struct fi_info *info,
			struct fid_ep **ep, void *context);
	int	(*if_open)(struct fid_domain *domain, const char *name,
			uint64_t flags, struct fid **fif, void *context);
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
fi_fec_open(struct fid_fabric *fabric, const struct fi_ec_attr *attr,
	    struct fid_ec **ec, void *context)
{
	return fabric->ops->ec_open(fabric, attr, ec, context);
}

static inline int
fi_ec_open(struct fid_domain *domain, struct fi_ec_attr *attr,
	   struct fid_ec **ec, void *context)
{
	return domain->ops->ec_open(domain, attr, ec, context);
}

static inline ssize_t fi_ec_read(struct fid_ec *ec, void *buf, size_t len)
{
	return ec->ops->read(ec, buf, len);
}

static inline ssize_t
fi_ec_readfrom(struct fid_ec *ec, void *buf, size_t len,
	       void *src_addr, size_t *addrlen)
{
	return ec->ops->readfrom(ec, buf, len, src_addr, addrlen);
}

static inline ssize_t
fi_ec_readerr(struct fid_ec *ec, void *buf, size_t len, uint64_t flags)
{
	return ec->ops->readerr(ec, buf, len, flags);
}

static inline int fi_ec_reset(struct fid_ec *ec, void *cond)
{
	return ec->ops->reset(ec, cond);
}

static inline const char *
fi_ec_strerror(struct fid_ec *ec, int prov_errno, void *prov_data,
	       void *buf, size_t len)
{
	return ec->ops->strerror(ec, prov_errno, prov_data, buf, len);
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
