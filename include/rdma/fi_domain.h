/*
 * Copyright (c) 2013 Intel Corporation. All rights reserved.
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


struct fi_iomv {
	void			*addr;
	size_t			len;
	uint64_t		mem_desc;
};

/*
 * AV = Address Vector
 * Maps and stores transport/network addresses.
 */

struct fi_av_addr {
	fid_t			av;
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
	int			av_mask;
	enum fi_av_type		type;
	enum fi_addr_format	addr_format;
	size_t			addrlen;
	size_t			count;
	uint64_t		flags;
};

struct fi_ops_av {
	size_t	size;
	int	(*insert)(fid_t fid, const void *addr, size_t count,
			void **fi_addr, uint64_t flags);
	int	(*remove)(fid_t fid, void *fi_addr, size_t count,
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
	uint64_t		mem_desc;
	be64_t			key;
};


/*
 * EC = Event Collector
 * Used to report various events and the completion of asynchronous
 * operations.
 */
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
	FI_EC_FORMAT_CM
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
	int			ec_mask;
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
	void			*fid_context;
	void			*op_context;
	uint64_t		flags;
	int			err;
	int			prov_errno;
	uint64_t		data;
	/* prov_data is available until the next time the EQ is read */
	void			*prov_data;
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
	ssize_t	(*read)(fid_t fid, void *buf, size_t len);
	ssize_t	(*readfrom)(fid_t fid, void *buf, size_t len,
			    void *src_addr, size_t *addrlen);
	ssize_t	(*readerr)(fid_t fid, void *buf, size_t len, uint64_t flags);
	ssize_t	(*write)(fid_t fid, void *buf, size_t len);
	int	(*reset)(fid_t fid, void *cond);
	ssize_t	(*condread)(fid_t fid, void *buf, size_t len, void *cond);
	ssize_t	(*condreadfrom)(fid_t fid, void *buf, size_t len,
				void *src_addr, size_t *addrlen, void *cond);
	const char * (*strerror)(fid_t fid, int prov_errno, void *prov_data,
			    void *buf, size_t len);
};

struct fid_ec {
	struct fid		fid;
	struct fi_ops_ec	*ops;
};


enum {
	FI_MR_ATTR_ACCESS	= 1 << 0,
	FI_MR_ATTR_FLAGS	= 1 << 1,
	FI_MR_ATTR_KEY		= 1 << 2,
	FI_MR_ATTR_MASK_V1	= (FI_MR_ATTR_KEY << 1) - 1
};

struct fi_mr_attr {
	int			mr_mask;
	uint64_t		access;
	uint64_t		flags;
	uint64_t		requested_key;
};


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
	size_t			max_auth_key_size;
	enum fi_progress	progress;
};

struct fi_ops_domain {
	size_t	size;
	int	(*progress)(fid_t fid);
	int	(*query)(fid_t fid, struct fi_domain_attr *attr, size_t *attrlen);
	int	(*av_open)(fid_t fid, struct fi_av_attr *attr, fid_t *av,
			   void *context);
	int	(*ec_open)(fid_t fid, struct fi_ec_attr *attr, fid_t *ec,
			   void *context);
	int	(*mr_reg)(fid_t fid, const void *buf, size_t len,
			  struct fi_mr_attr *attr, fid_t *mr, void *context);
	int	(*mr_regv)(fid_t fid, const struct iovec *iov, size_t count,
			   struct fi_mr_attr *attr, fid_t *mr, void *context);
};

struct fid_domain {
	struct fid		fid;
	struct fi_ops_domain	*ops;
};

static inline int fi_ec_open(fid_t fid, struct fi_ec_attr *attr, fid_t *ec,
			    void *context)
{
	struct fid_domain *domain = container_of(fid, struct fid_domain, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_RESOURCE_DOMAIN);
	FI_ASSERT_OPS(fid, struct fid_domain, ops);
	FI_ASSERT_OP(domain->ops, struct fi_ops_domain, ec_open);
	return domain->ops->ec_open(fid, attr, ec, context);
}

static inline ssize_t fi_ec_read(fid_t fid, void *buf, size_t len)
{
	struct fid_ec *ec = container_of(fid, struct fid_ec, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_EC);
	FI_ASSERT_OPS(fid, struct fid_ec, ops);
	FI_ASSERT_OP(ec->ops, struct fi_ops_ec, read);
	return ec->ops->read(fid, buf, len);
}

static inline ssize_t fi_ec_readfrom(fid_t fid, void *buf, size_t len,
				     void *src_addr, size_t *addrlen)
{
	struct fid_ec *ec = container_of(fid, struct fid_ec, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_EC);
	FI_ASSERT_OPS(fid, struct fid_ec, ops);
	FI_ASSERT_OP(ec->ops, struct fi_ops_ec, readfrom);
	return ec->ops->readfrom(fid, buf, len, src_addr, addrlen);
}

static inline ssize_t fi_ec_readerr(fid_t fid, void *buf, size_t len, uint64_t flags)
{
	struct fid_ec *ec = container_of(fid, struct fid_ec, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_EC);
	FI_ASSERT_OPS(fid, struct fid_ec, ops);
	FI_ASSERT_OP(ec->ops, struct fi_ops_ec, readerr);
	return ec->ops->readerr(fid, buf, len, flags);
}

static inline int fi_ec_reset(fid_t fid, void *cond)
{
	struct fid_ec *ec = container_of(fid, struct fid_ec, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_EC);
	FI_ASSERT_OPS(fid, struct fid_ec, ops);
	FI_ASSERT_OP(ec->ops, struct fi_ops_ec, reset);
	return ec->ops->reset(fid, cond);
}

static inline const char * fi_ec_strerror(fid_t fid, int prov_errno, void *prov_data,
	void *buf, size_t len)
{
	struct fid_ec *ec = container_of(fid, struct fid_ec, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_EC);
	FI_ASSERT_OPS(fid, struct fid_ec, ops);
	FI_ASSERT_OP(ec->ops, struct fi_ops_ec, strerror);
	return ec->ops->strerror(fid, prov_errno, prov_data, buf, len);
}

static inline int fi_mr_reg(fid_t fid, const void *buf, size_t len,
			    struct fi_mr_attr *attr, fid_t *mr, void *context)
{
	struct fid_domain *domain = container_of(fid, struct fid_domain, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_RESOURCE_DOMAIN);
	FI_ASSERT_OPS(fid, struct fid_domain, ops);
	FI_ASSERT_OP(domain->ops, struct fi_ops_domain, mr_reg);
	return domain->ops->mr_reg(fid, buf, len, attr, mr, context);
}

static inline uint64_t fi_mr_desc(fid_t fid)
{
	struct fid_mr *mr = container_of(fid, struct fid_mr, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_MR);
	FI_ASSERT_FIELD(fid, struct fid_mr, mem_desc);
	return mr->mem_desc;
}

static inline be64_t fi_mr_key(fid_t fid)
{
	struct fid_mr *mr = container_of(fid, struct fid_mr, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_MR);
	FI_ASSERT_FIELD(fid, struct fid_mr, key);
	return mr->key;
}

static inline int fi_mr_unreg(fid_t fid)
{
	FI_ASSERT_CLASS(fid, FID_CLASS_MR);
	return fi_close(fid);
}

static inline int fi_av_open(fid_t fid, struct fi_av_attr *attr, fid_t *av,
			     void *context)
{
	struct fid_domain *domain = container_of(fid, struct fid_domain, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_RESOURCE_DOMAIN);
	FI_ASSERT_OPS(fid, struct fid_domain, ops);
	FI_ASSERT_OP(domain->ops, struct fi_ops_domain, av_open);
	return domain->ops->av_open(fid, attr, av, context);
}

static inline int fi_av_map(fid_t fid, const void *addr, size_t count,
			    void **fi_addr, uint64_t flags)
{
	struct fid_av *av = container_of(fid, struct fid_av, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_AV);
	FI_ASSERT_OPS(fid, struct fid_av, ops);
	FI_ASSERT_OP(av->ops, struct fi_ops_av, insert);
	return av->ops->insert(fid, addr, count, fi_addr, flags);
}

static inline int fi_av_unmap(fid_t fid, void *fi_addr, size_t count,
			      uint64_t flags)
{
	struct fid_av *av = container_of(fid, struct fid_av, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_AV);
	FI_ASSERT_OPS(fid, struct fid_av, ops);
	FI_ASSERT_OP(av->ops, struct fi_ops_av, remove);
	return av->ops->remove(fid, fi_addr, count, flags);
}

static inline int fi_av_sync(fid_t fid, uint64_t flags, void *context)
{
	FI_ASSERT_CLASS(fid, FID_CLASS_AV);
	return fi_sync(fid, flags, context);
}


#ifdef __cplusplus
}
#endif

#endif /* _FI_DOMAIN_H_ */
