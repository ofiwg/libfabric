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

#ifndef _FI_ENDPOINT_H_
#define _FI_ENDPOINT_H_

#include <sys/socket.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <stddef.h>


#ifdef __cplusplus
extern "C" {
#endif


enum fid_type {
	FID_UNSPEC,
	FID_MSG,
	FID_STREAM,
	FID_DGRAM,
	FID_RAW,
	FID_RDM,
	FID_PACKET,
	FID_MAX
};

/* fi_info protocol field.
 * If two providers support the same protocol, then they shall interoperate
 * when the protocol capabilities match.
 */
enum fi_proto {
	FI_PROTO_UNSPEC,
	FI_PROTO_IB_RC,
	FI_PROTO_IWARP,
	FI_PROTO_IB_UC,
	FI_PROTO_IB_UD,
	FI_PROTO_IB_XRC,
	FI_PROTO_RAW
};

/* fi_info endpoint capabilities */
#define FI_PASSIVE		(1ULL << 0)
#define FI_MSG			(1ULL << 1)
#define FI_RMA			(1ULL << 2)
#define FI_TAGGED		(1ULL << 3)
#define FI_ATOMICS		(1ULL << 4)
#define FI_MULTICAST		(1ULL << 5)	/* multicast uses MSG ops */
#define FI_BUFFERED_RECV	(1ULL << 9)


struct fi_msg {
	const struct iovec	*msg_iov;
	void			**desc;
	size_t			iov_count;
	const void		*addr;
	void			*context;
	uint64_t		data;
	int			flow;
};

/* Endpoint option levels */
enum {
	FI_OPT_ENDPOINT
};

/* FI_OPT_ENDPOINT option names */
enum {
	FI_OPT_MAX_INJECTED_SEND,	/* size_t */
	FI_OPT_TOTAL_BUFFERED_RECV,	/* size_t */
	FI_OPT_MAX_MSG_SIZE,		/* size_t */
	FI_OPT_DATA_FLOW,		/* int */
};

struct fi_ops_ep {
	size_t	size;
	int	(*enable)(struct fid_ep *ep);
	ssize_t	(*cancel)(fid_t fid, void *context);
	int	(*getopt)(fid_t fid, int level, int optname,
			  void *optval, size_t *optlen);
	int	(*setopt)(fid_t fid, int level, int optname,
			  const void *optval, size_t optlen);
};

struct fi_ops_msg {
	size_t	size;
	ssize_t (*recv)(struct fid_ep *ep, void *buf, size_t len, void *desc,
			void *context);
	ssize_t (*recvv)(struct fid_ep *ep, const struct iovec *iov, void **desc,
			size_t count, void *context);
	ssize_t (*recvfrom)(struct fid_ep *ep, void *buf, size_t len, void *desc,
			const void *src_addr, void *context);
	ssize_t (*recvmsg)(struct fid_ep *ep, const struct fi_msg *msg,
			uint64_t flags);
	ssize_t (*send)(struct fid_ep *ep, const void *buf, size_t len, void *desc,
			void *context);
	ssize_t (*sendv)(struct fid_ep *ep, const struct iovec *iov, void **desc,
			size_t count, void *context);
	ssize_t (*sendto)(struct fid_ep *ep, const void *buf, size_t len, void *desc,
			const void *dest_addr, void *context);
	ssize_t (*sendmsg)(struct fid_ep *ep, const struct fi_msg *msg,
			uint64_t flags);
	ssize_t	(*inject)(struct fid_ep *ep, const void *buf, size_t len);
	ssize_t	(*injectto)(struct fid_ep *ep, const void *buf, size_t len,
			const void *dest_addr);
	ssize_t (*senddata)(struct fid_ep *ep, const void *buf, size_t len, void *desc,
			uint64_t data, void *context);
	ssize_t (*senddatato)(struct fid_ep *ep, const void *buf, size_t len, void *desc,
			uint64_t data, const void *dest_addr, void *context);
};

struct fi_ops_cm;
struct fi_ops_rma;
struct fi_ops_tagged;
struct fi_ops_atomic;
/* struct fi_ops_collectives; */

/*
 * Calls which modify the properties of a endpoint (control, setopt, bind, ...)
 * must be serialized against all other operations.  Those calls may modify the
 * operations referenced by a endpoint in order to optimize the data transfer code
 * paths.
 *
 * A provider may allocate the minimal size structure needed to support the
 * ops requested by the user.
 */
struct fid_ep {
	struct fid		fid;
	struct fi_ops_ep	*ops;
	struct fi_ops_cm	*cm;
	struct fi_ops_msg	*msg;
	struct fi_ops_rma	*rma;
	struct fi_ops_tagged	*tagged;
	struct fi_ops_atomic	*atomic;
};

struct fid_pep {
	struct fid		fid;
	struct fi_ops_ep	*ops;
	struct fi_ops_cm	*cm;
};

#ifndef FABRIC_DIRECT

static inline int
fi_pendpoint(struct fid_fabric *fabric, struct fi_info *info,
	     struct fid_pep **pep, void *context)
{
	return fabric->ops->endpoint(fabric, info, pep, context);
}

static inline int
fi_endpoint(struct fid_domain *domain, struct fi_info *info,
	    struct fid_ep **ep, void *context)
{
	return domain->ops->endpoint(domain, info, ep, context);
}

static inline ssize_t fi_enable(struct fid_ep *ep)
{
	return ep->ops->enable(ep);
}

static inline ssize_t fi_cancel(fid_t fid, void *context)
{
	struct fid_ep *ep = container_of(fid, struct fid_ep, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_EP);
	FI_ASSERT_OPS(fid, struct fid_ep, ops);
	FI_ASSERT_OP(ep->ops, struct fi_ops_ep, cancel);
	return ep->ops->cancel(fid, context);
}

static inline ssize_t fi_setopt(fid_t fid, int level, int optname,
				const void *optval, size_t optlen)
{
	struct fid_ep *ep = container_of(fid, struct fid_ep, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_EP);
	FI_ASSERT_OPS(fid, struct fid_ep, ops);
	FI_ASSERT_OP(ep->ops, struct fi_ops_ep, setopt);
	return ep->ops->setopt(fid, level, optname, optval, optlen);
}

static inline ssize_t fi_getopt(fid_t fid, int level, int optname,
				void *optval, size_t *optlen)
{
	struct fid_ep *ep = container_of(fid, struct fid_ep, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_EP);
	FI_ASSERT_OPS(fid, struct fid_ep, ops);
	FI_ASSERT_OP(ep->ops, struct fi_ops_ep, getopt);
	return ep->ops->getopt(fid, level, optname, optval, optlen);
}

static inline ssize_t
fi_recv(struct fid_ep *ep, void *buf, size_t len, void *desc, void *context)
{
	return ep->msg->recv(ep, buf, len, desc, context);
}

static inline ssize_t
fi_recvv(struct fid_ep *ep, const struct iovec *iov, void **desc,
	 size_t count, void *context)
{
	return ep->msg->recvv(ep, iov, desc, count, context);
}

static inline ssize_t
fi_recvfrom(struct fid_ep *ep, void *buf, size_t len, void *desc,
	    const void *src_addr, void *context)
{
	return ep->msg->recvfrom(ep, buf, len, desc, src_addr, context);
}

static inline ssize_t
fi_recvmsg(struct fid_ep *ep, const struct fi_msg *msg, uint64_t flags)
{
	return ep->msg->recvmsg(ep, msg, flags);
}

static inline ssize_t
fi_send(struct fid_ep *ep, const void *buf, size_t len, void *desc, void *context)
{
	return ep->msg->send(ep, buf, len, desc, context);
}

static inline ssize_t
fi_sendv(struct fid_ep *ep, const struct iovec *iov, void **desc,
	 size_t count, void *context)
{
	return ep->msg->sendv(ep, iov, desc, count, context);
}

static inline ssize_t
fi_sendto(struct fid_ep *ep, const void *buf, size_t len, void *desc,
	  const void *dest_addr, void *context)
{
	return ep->msg->sendto(ep, buf, len, desc, dest_addr, context);
}

static inline ssize_t
fi_sendmsg(struct fid_ep *ep, const struct fi_msg *msg, uint64_t flags)
{
	return ep->msg->sendmsg(ep, msg, flags);
}

static inline ssize_t
fi_inject(struct fid_ep *ep, const void *buf, size_t len)
{
	return ep->msg->inject(ep, buf, len);
}

static inline ssize_t
fi_injectto(struct fid_ep *ep, const void *buf, size_t len, const void *dest_addr)
{
	return ep->msg->injectto(ep, buf, len, dest_addr);
}

static inline ssize_t
fi_senddata(struct fid_ep *ep, const void *buf, size_t len, void *desc,
	    uint64_t data, void *context)
{
	return ep->msg->senddata(ep, buf, len, desc, data, context);
}

static inline ssize_t
fi_senddatato(struct fid_ep *ep, const void *buf, size_t len, void *desc,
	      uint64_t data, const void *dest_addr, void *context)
{
	return ep->msg->senddatato(ep, buf, len, desc, data, dest_addr, context);
}

#else // FABRIC_DIRECT
#include <rdma/fi_direct_endpoint.h>
#endif

#ifdef __cplusplus
}
#endif

#endif /* _FI_ENDPOINT_H_ */
