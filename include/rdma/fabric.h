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
 *  0 -  3	operation specific (used for a single call)
 *  4 -  7	reserved
 *  8 - 55	common (usable with multiple operations)
 * 56 - 59	reserved
 * 60 - 63	provider-domain specific
 */

#define FI_BLOCK		(1ULL << 9)
#define FI_BUFFERED_RECV	(1ULL << 10)
#define FI_BUFFERED_SEND	(1ULL << 11)
#define FI_MULTI_RECV		(1ULL << 12)

#define FI_EXCL			(1ULL << 16)
#define FI_READ			(1ULL << 17)
#define FI_WRITE		(1ULL << 18)
#define FI_RECV			(1ULL << 19)
#define FI_SEND			(1ULL << 20)
#define FI_REMOTE_READ		(1ULL << 21)
#define FI_REMOTE_WRITE		(1ULL << 22)

#define FI_IMM			(1ULL << 24)
#define FI_EVENT		(1ULL << 25)
#define FI_REMOTE_SIGNAL	(1ULL << 26)
#define FI_REMOTE_COMPLETE	(1ULL << 27)
#define FI_CANCEL		(1ULL << 28)
#define FI_MORE			(1ULL << 29)
#define FI_PEEK			(1ULL << 30)
#define FI_TRIGGER		(1ULL << 31)


/*
 * Format for 'vectored' data transfer calls: sendv, writev, etc.
 */
enum fi_iov_format {
	FI_IOV,			/* struct iovec */
	FI_IOMV,		/* struct fi_iomv */
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

struct fi_info {
	struct fi_info		*next;
	size_t			size;
	uint64_t		flags;
	uint64_t		type;
	uint64_t		protocol;
	uint64_t		protocol_cap;
	uint64_t		domain_cap;
	enum fi_iov_format	iov_format;
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
	FID_CLASS_EC
};

struct fi_context {
	void			*internal[4];
};

struct fid;
typedef struct fid *fid_t;
struct fi_ec_attr;

struct fi_resource {
	fid_t			fid;
	uint64_t		flags;
};

struct fi_ops {
	size_t	size;
	int	(*close)(fid_t fid);
	int	(*bind)(fid_t fid, struct fi_resource *fids, int nfids);
	int	(*sync)(fid_t fid, uint64_t flags, void *context);
	int	(*control)(fid_t fid, int command, void *arg);
};

/* All fabric interface descriptors must start with this structure */
struct fid {
	int			fclass;
	unsigned int		size;
	void			*context;
	struct fi_ops		*ops;
};

#define FI_PASSIVE		(1ULL << 0)
#define FI_NUMERICHOST		(1ULL << 1)

int fi_getinfo(const char *node, const char *service, struct fi_info *hints,
	       struct fi_info **info);
void fi_freeinfo(struct fi_info *info);

struct fi_ops_fabric {
	size_t	size;
	int	(*domain)(fid_t fid, struct fi_info *info, fid_t *dom,
			void *context);
	int	(*endpoint)(fid_t fid, struct fi_info *info, fid_t *pep,
			void *context);
	int	(*ec_open)(fid_t fid, const struct fi_ec_attr *attr, fid_t *ec,
			void *context);
	int	(*if_open)(fid_t fid, const char *name, uint64_t flags,
			fid_t *fif, void *context);
};

struct fid_fabric {
	struct fid		fid;
	struct fi_ops_fabric	*ops;
};

int fi_fabric(const char *name, uint64_t flags, fid_t *fid, void *context);


#define FI_ASSERT_CLASS(fid, f_class)   assert(fid->fclass == f_class)
#define FI_ASSERT_FIELD(ptr, ftype, field) assert(ptr->size > offsetof(ftype, field))
#define FI_ASSERT_OPS(fid, ftype, ops) FI_ASSERT_FIELD(fid, ftype, ops)
#define FI_ASSERT_OP(ops, otype, op)   FI_ASSERT_FIELD(ops, otype, op)

static inline int
fi_fopen(fid_t fid, const char *name, uint64_t flags, fid_t *fif, void *context)
{
	struct fid_fabric *fab = container_of(fid, struct fid_fabric, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_FABRIC);
	FI_ASSERT_OPS(fid, struct fid_fabric, ops);
	FI_ASSERT_OP(fab->ops, struct fi_ops_fabric, if_open);
	return fab->ops->if_open(fid, name, flags, fif, context);
}

static inline int fi_close(fid_t fid)
{
	FI_ASSERT_OPS(fid, struct fid, ops);
	FI_ASSERT_OP(fid->ops, struct fi_ops, close);
	return fid->ops->close(fid);
}
#define fi_destroy(fid) fi_close(fid)

static inline int fi_bind(fid_t fid, struct fi_resource *fids, int nfids)
{
	FI_ASSERT_OPS(fid, struct fid, ops);
	FI_ASSERT_OP(fid->ops, struct fi_ops, bind);
	return fid->ops->bind(fid, fids, nfids);
}

static inline int fi_sync(fid_t fid, uint64_t flags, void *context)
{
	FI_ASSERT_OPS(fid, struct fid, ops);
	FI_ASSERT_OP(fid->ops, struct fi_ops, sync);
	return fid->ops->sync(fid, flags, context);
}

struct fi_alias {
	fid_t			*fid;
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
	FI_GETECWAIT,		/* void * wait object */

	/* Start/stop an internal progress thread.  This is only needed if the
	 * provider does not support active_progress, and the app does not
	 * want to poll for progress.
	 */
	FI_STARTPROGRESS,	/* NULL - flags? */
	FI_STOPPROGRESS		/* NULL - flags? */
};

/*
 * fi_control may be used to set the flags for data transfer operations.  This
 * is done using the FI_SETOPSFLAG command with arg a uint64_t flags value.  The
 * FI_READ, FI_WRITE, FI_SEND, FI_RECV flags indicate the type of data transfer
 * that the flags should apply to, with other flags OR'ed in.
 */
static inline int fi_control(fid_t fid, int command, void *arg)
{
	FI_ASSERT_OPS(fid, struct fid, ops);
	FI_ASSERT_OP(fid->ops, struct fi_ops, control);
	return fid->ops->control(fid, command, arg);
}

static inline int fi_alias(fid_t fid, fid_t *alias_fid, uint64_t flags)
{
	struct fi_alias alias;
	alias.fid = alias_fid;
	alias.flags = flags;
	return fi_control(fid, FI_ALIAS, &alias);
}


#ifdef FABRIC_DIRECT
#include <rdma/fi_direct.h>
#endif

#ifdef __cplusplus
}
#endif

#endif /* _FABRIC_H_ */
