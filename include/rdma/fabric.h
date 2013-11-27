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

#ifndef _FABRIC_H_
#define _FABRIC_H_

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <sys/socket.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef uint16_t be16_t;
typedef uint32_t be32_t;
typedef uint64_t be64_t;

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

/* PASSIVE - Indicates that the allocated endpoint will be used
 * to listen for connection requests.
 * fi_info
 */
#define FI_PASSIVE		(1ULL << 0)
/* NUMERICHOST - The node parameter passed into fi_getinfo is a
 * numeric IP address or GID.  When set, name resolution is not
 * performed.
 * fi_info
 */
#define FI_NUMERICHOST		(1ULL << 1)
/* FAMILY - If set, then the node parameter passed into fi_getinfo
 * is encoded address.  The format of the address is given by the
 * sa_family field in fi_info.  This flag is needed by providers
 * in order to determine if an address is an IPv6 or GID based
 * address.
 * fi_info
 */
//#define FI_FAMILY		(1ULL << 2)

/* AUTO_RESET - automatically resets the event queue to generate
 * a new wake-up event on the next entry.  Example use:
 * 1. wait on eq wait object -- poll(fd)
 * 2. wait object is ready -- fd is readable
 * 3. read eq to retrieve events
 * 4. continue reading until read returns 0
 */
#define FI_AUTO_RESET		(1ULL << 7)

/* fi_info type, fcntl, fi_open flags */

/* Reserve lower 8-bits for type selection
 * fi_info type, fi_open, fcntl
 */
#define FI_NONBLOCK		(1ULL << 8)
/* Reserve lower 8-bits for type selection
 * fi_info type, fi_open, fcntl
 */
#define FI_SYNC			(1ULL << 9)
/* EXCL - Indicates that the specified domain should not share
 * resources with another opened domain.  By default, resources
 * associated with a resource domain are shared across all open
 * calls by the same process.
 * reserve lower 8-bits for type selection
 * fi_info type, fi_open, fcntl
 */
#define FI_EXCL			(1ULL << 10)
/* BUFFERED_RECV - If set, the provider should attempt to queue inbound
 * data that arrives before a receive buffer has been posted.  In the
 * absence of this flag, any messages that arrive before a receive is
 * posted are lost.
 * When set, the user must use struct fi_context * as their per
 * operation context.
 * reserve lower 8-bits for type selection
 * fi_info type, fi_open, fcntl
 */
/* TODO: Should buffered be its own bit */
#define FI_BUFFERED_RECV	(1ULL << 11)
/* CANCEL - Indicates that the user wants the ability to cancel
 * the operation if it does not complete first.  Providers use this
 * to return a handle to the request, which the user may then cancel.
 * Also used by search to indicate that a request should be canceled.
 * fi_info type, fi_open, fcntl, data transfer ops
 */
#define FI_CANCEL		(1ULL << 12)
/* SHARED_RECV - A endpoint created with this flag will share the same
 * receive queue as other endpoints created on the same domain.
 * fi_info type, fi_open, fcntl
 */
/* TODO: should shared be its own bit? */
#define FI_SHARED_RECV		(1ULL << 13)
/* READ - Used to enable read access to data buffers.
 */
#define FI_READ			(1ULL << 14)
/* WRITE - Used to enable write access to data buffers.
 */
#define FI_WRITE		(1ULL << 15)
/* RECV - Report recv completion EQs
 */
/* TODO: Use with buffered_recv / shared_recv? */
#define FI_RECV			(1ULL << 16)
/* SEND - Report send completion EQs
 */
/* TODO: Use with buffered_send? */
#define FI_SEND			(1ULL << 17)

/* fcntl and data transfer ops */

#define FI_DONTWAIT		FI_NONBLOCK
#define FI_PEEK			(1ULL << 25)
/* ERRQUEUE - A read operation should retrieve any queued error data.
 * In the case of a failure, a read operation may return an error code,
 * indicating that an operation has failed and extended error data is
 * available.  Queued error data must be read before additional
 * completions may be read.
 *
 * Added eq.readerr call, which should eliminate the need for this.
 */
#define FI_ERRQUEUE		(1ULL << 26)
/* TRUNC - Signals that received data has been truncated.
 */
#define FI_TRUNC		(1ULL << 27)
/* CTRUNC - Indicates that control data was truncated.  Use case?
 */
#define FI_CTRUNC		(1ULL << 28)
#define FI_ATRUNC		(1ULL << 29)
/* IMM - Indicates that immediate data is available.  IMM data is
 * communicated to a receiver through completion data, rather than
 * appearing in targeted receive buffers.
 */
#define FI_IMM			(1ULL << 30)
/* NOCOMP - Indicates that no completion should be generated for the
 * specified operation.
 */
#define FI_NOCOMP		(1ULL << 31)
/* MORE: Indicates that additional requests are pending.  Providers may
 * use this to optimize access to hardware.
 */
#define FI_MORE			(1ULL << 32)
/* SIGNAL - Indicates if a completion event should be generated.
 */
#define FI_SIGNAL		(1ULL << 33)
/* BUFFERED_SEND - If set, the outbound data buffer should be returned
 * to user immediately after the call returns, even if the operation is
 * handled asynchronously.  This may require that the provider copy
 * the data into a local buffer and transfer out of that buffer.
 */
#define FI_BUFFERED_SEND	(1ULL << 34)
/* ACK - Indicates that a completion event is not generated until the operation
 * initiated is acknowledged by the remote side */
#define FI_ACK			(1ULL << 35)

/* ERRINLINE - Error events are reported inline with other events, rather
 * than through a separate error queue (see ERRQUEUE).
 */
#define FI_ERRINLINE		(1ULL << 36)
/* REMOTE - Used to indicate capabilities and permissions for remote processes */
#define FI_REMOTE		(1ULL << 37)


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
	FI_INFO_ADDR,		/* struct fi_info_addr */
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
	enum fi_iov_format	iov_format;
	enum fi_addr_format	addr_format;
	enum fi_addr_format	info_addr_format;
	size_t			src_addrlen;
	size_t			dst_addrlen;
	void			*src_addr;
	void			*dst_addr;
	/* Authorization key is intended to limit communication with only
	 * those endpoints sharing the same key.
	 */
	size_t			auth_keylen;
	void			*auth_key;
	/* A shared_fd is intended to allow a domain to share resources
	 * and data with other processes that have access to the same
	 * shared_fd.  Based on XRC work.
	 */
	int			shared_fd;
	char			*domain_name;
	size_t			datalen;
	void			*data;
};

enum {
	FID_CLASS_UNSPEC,
	FID_CLASS_EP,
	FID_CLASS_DOMAIN,
	FID_CLASS_INTERFACE,
	FID_CLASS_AV,
	FID_CLASS_MR,
	FID_CLASS_EC
};

/* See FI_BUFFERED_RECV, FI_CANCEL */
struct fi_context {
	void			*internal[4];
};

struct fid;
typedef struct fid *fid_t;

struct fi_resource {
	fid_t			fid;
	uint64_t		flags;
};

struct fi_ops {
	size_t	size;
	int	(*close)(fid_t fid);
	/* Associate resources with this object */
	int	(*bind)(fid_t fid, struct fi_resource *fids, int nfids);
	/* Operation that completes after all previous async requests complete */
	int	(*sync)(fid_t fid, uint64_t flags, void *context);
	/* low-level control - similar to fcntl & ioctl operations */
	int	(*control)(fid_t fid, int command, void *arg);
};

/* All fabric interface descriptors must start with this structure */
struct fid {
	int			fclass;
	int			size;
	void			*context;
	struct fi_ops		*ops;
};

#define FI_PREFIX		"fi"
#define FI_DOMAIN_NAMES		"domains"
#define FI_UNBOUND_NAME		"local"

int fi_getinfo(char *node, char *service, struct fi_info *hints,
	       struct fi_info **info);
void fi_freeinfo(struct fi_info *info);

int fi_open(const char *name, uint64_t flags, fid_t *fid, void *context);
int fi_domain(struct fi_info *info, fid_t *fid, void *context);
int fi_endpoint(struct fi_info *info, fid_t *fid, void *context);

#define FI_ASSERT_CLASS(fid, f_class)   assert(fid->fclass == f_class)
#define FI_ASSERT_FIELD(ptr, ftype, field) assert(ptr->size > offsetof(ftype, field))
#define FI_ASSERT_OPS(fid, ftype, ops) FI_ASSERT_FIELD(fid, ftype, ops)
#define FI_ASSERT_OP(ops, otype, op)   FI_ASSERT_FIELD(ops, otype, op)

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
	FI_DUPFID,		/* fid_t * */
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


#ifdef __cplusplus
}
#endif

#endif /* _FABRIC_H_ */
