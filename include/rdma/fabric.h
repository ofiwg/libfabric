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

#ifndef _FABRIC_H_
#define _FABRIC_H_

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <sys/socket.h>

#ifdef __cplusplus
extern "C" {
#endif


#ifndef container_of
#define container_of(ptr, type, field) \
	((type *) ((char *)ptr - offsetof(type, field)))
#endif

enum {
	FI_PATH_MAX		= 256,
	FI_NAME_MAX		= 64,
	FI_VERSION_MAX		= 64
};

/*
 * Vendor specific protocols/etc. are encoded as OUI, followed by vendor
 * specific data.
 */
#define FI_OUI_SHIFT		48

/* fi_info and operation flags - pass into endpoint ops calls.
 * A user may also set these on a endpoint by using fcntl, which has the
 * affect of applying them to all applicable operations.
 */

/*
 * Flags
 * The 64-bit flag field is divided as follows:
 * bits		use
 *  0 -  9	operation specific (used for a single call)
 * 10 - 55	common (usable with multiple operations)
 * 56 - 59	reserved
 * 60 - 63	provider-domain specific
 */

#define FI_BLOCK		(1ULL << 10)
#define FI_INJECT		(1ULL << 11)
#define FI_MULTI_RECV		(1ULL << 12)
#define FI_SOURCE		(1ULL << 13)

#define FI_READ			(1ULL << 16)
#define FI_WRITE		(1ULL << 17)
#define FI_RECV			(1ULL << 18)
#define FI_SEND			(1ULL << 19)
#define FI_REMOTE_READ		(1ULL << 20)
#define FI_REMOTE_WRITE		(1ULL << 21)

#define FI_IMM			(1ULL << 24)
#define FI_EVENT		(1ULL << 25)
#define FI_REMOTE_SIGNAL	(1ULL << 26)
#define FI_REMOTE_COMPLETE	(1ULL << 27)
#define FI_CANCEL		(1ULL << 28)
#define FI_MORE			(1ULL << 29)
#define FI_PEEK			(1ULL << 30)
#define FI_TRIGGER		(1ULL << 31)


struct fi_ioc {
	void			*addr;
	size_t			count;
};

/*
 * Format for transport addresses: sendto, writeto, etc.
 */
enum fi_addr_format {
	FI_ADDR,		/* void * fi_addr */
	FI_AV,			/* struct fi_av_addr */
	FI_ADDR_INDEX,		/* size_t fi_addr */
	FI_ADDR_PROTO,		/* void * proto_addr */
	FI_SOCKADDR,		/* struct sockaddr */
	FI_SOCKADDR_IN,		/* struct sockaddr_in */
	FI_SOCKADDR_IN6,	/* struct sockaddr_in6 */
	FI_SOCKADDR_IB,		/* struct sockaddr_ib */
};

enum fi_progress {
	FI_PROGRESS_UNSPEC,
	FI_PROGRESS_AUTO,
	FI_PROGRESS_IMPLICIT
};

enum fi_threading {
	FI_THREAD_MULTIPLE,
	FI_THREAD_SINGLE,
	FI_THREAD_FUNNELED,
	FI_THREAD_SERIALIZED
};


struct fi_info {
	struct fi_info		*next;
	size_t			size;
	uint64_t		type;
	uint64_t		protocol;
	uint64_t		ep_cap;
	uint64_t		op_flags;
	uint64_t		domain_cap;
	enum fi_addr_format	addr_format;
	enum fi_addr_format	info_addr_format;
	size_t			src_addrlen;
	size_t			dest_addrlen;
	void			*src_addr;
	void			*dest_addr;
	/* Authorization key is intended to limit communication with only
	 * those endpoints sharing the same key and allows sharing of
	 * data with local processes.
	 */
	size_t			auth_keylen;
	void			*auth_key;
	enum fi_threading	threading;
	enum fi_progress	control_progress;
	enum fi_progress	data_progress;
	char			*fabric_name;
	char			*domain_name;
	size_t			datalen;
	void			*data;
};

enum {
	FID_CLASS_UNSPEC,
	FID_CLASS_FABRIC,
	FID_CLASS_DOMAIN,
	FID_CLASS_EP,
	FID_CLASS_PEP,
	FID_CLASS_INTERFACE,
	FID_CLASS_AV,
	FID_CLASS_MR,
	FID_CLASS_EQ,
	FID_CLASS_CNTR
};

struct fid;
struct fid_fabric;
struct fid_domain;
struct fid_av;
struct fid_eq;
struct fid_cntr;
struct fid_ep;
struct fid_pep;
struct fid_mr;

typedef struct fid *fid_t;
struct fi_eq_attr;

struct fi_resource {
	struct fid		*fid;
	uint64_t		flags;
};

struct fi_ops {
	size_t	size;
	int	(*close)(struct fid *fid);
	int	(*bind)(struct fid *fid, struct fi_resource *fids, int nfids);
	int	(*sync)(struct fid *fid, uint64_t flags, void *context);
	int	(*control)(struct fid *fid, int command, void *arg);
};

/* All fabric interface descriptors must start with this structure */
struct fid {
	int			fclass;
	unsigned int		size;
	void			*context;
	struct fi_ops		*ops;
};

#define FI_NUMERICHOST		(1ULL << 1)

int fi_getinfo(const char *node, const char *service, uint64_t flags,
	       struct fi_info *hints, struct fi_info **info);
void fi_freeinfo(struct fi_info *info);

struct fi_attr {
	uint64_t		version;
	uint64_t		prov_version;
	uint64_t		hw_version;
	uint32_t		oui;
};

void fi_query(const struct fi_info *info, struct fi_attr *attr, size_t *attrlen);

struct fi_ops_fabric {
	size_t	size;
	int	(*domain)(struct fid_fabric *fabric, struct fi_info *info,
			struct fid_domain **dom, void *context);
	int	(*endpoint)(struct fid_fabric *fabric, struct fi_info *info,
			struct fid_pep **pep, void *context);
	int	(*eq_open)(struct fid_fabric *fabric, const struct fi_eq_attr *attr,
			struct fid_eq **eq, void *context);
	int	(*if_open)(struct fid_fabric *fabric, const char *name,
			uint64_t flags, struct fid **fif, void *context);
};

struct fid_fabric {
	struct fid		fid;
	struct fi_ops_fabric	*ops;
};

int fi_fabric(const char *name, uint64_t flags, struct fid_fabric **fabric,
	      void *context);


#define FI_ASSERT_CLASS(fid, f_class)   assert(fid->fclass == f_class)
#define FI_ASSERT_FIELD(ptr, ftype, field) assert(ptr->size > offsetof(ftype, field))
#define FI_ASSERT_OPS(fid, ftype, ops) FI_ASSERT_FIELD(fid, ftype, ops)
#define FI_ASSERT_OP(ops, otype, op)   FI_ASSERT_FIELD(ops, otype, op)

static inline int
fi_fopen(struct fid_fabric *fabric, const char *name, uint64_t flags,
	 struct fid **fif, void *context)
{
	return fabric->ops->if_open(fabric, name, flags, fif, context);
}

static inline int fi_close(struct fid *fid)
{
	return fid->ops->close(fid);
}
#define fi_destroy(fid) fi_close(fid)

static inline int fi_bind(struct fid *fid, struct fi_resource *fids, int nfids)
{
	return fid->ops->bind(fid, fids, nfids);
}

static inline int fi_sync(struct fid *fid, uint64_t flags, void *context)
{
	return fid->ops->sync(fid, flags, context);
}

struct fi_alias {
	struct fid 		**fid;
	uint64_t		flags;
};

/* control commands */
enum {
	FI_GETFIDFLAG,		/* uint64_t flags */
	FI_SETFIDFLAG,		/* uint64_t flags */
	FI_GETOPSFLAG,		/* uint64_t flags */
	FI_SETOPSFLAG,		/* uint64_t flags */

	/* Duplicate a fid_t.  This allows for 2 fids that refer to a single
	 * HW resource.  Each fid may reference functions that are optimized
	 * for different use cases.
	 */
	FI_ALIAS,		/* struct fi_alias * */
	FI_GETWAIT,		/* void * wait object */
};

static inline int fi_control(struct fid *fid, int command, void *arg)
{
	return fid->ops->control(fid, command, arg);
}

static inline int fi_alias(struct fid *fid, struct fid **alias_fid, uint64_t flags)
{
	struct fi_alias alias;
	alias.fid = alias_fid;
	alias.flags = flags;
	return fi_control(fid, FI_ALIAS, &alias);
}


#ifndef FABRIC_DIRECT

struct fi_context {
	void			*internal[4];
};

#else // FABRIC_DIRECT
#include <rdma/fi_direct.h>
#endif

#ifdef __cplusplus
}
#endif

#endif /* _FABRIC_H_ */
