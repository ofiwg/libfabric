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

/* fi_info protocol capabilities */
#define FI_PROTO_CAP_MSG	(1ULL << 0)
#define FI_PROTO_CAP_RMA	(1ULL << 1)
#define FI_PROTO_CAP_TAGGED	(1ULL << 2)
#define FI_PROTO_CAP_ATOMICS	(1ULL << 3)
#define FI_PROTO_CAP_MULTICAST	(1ULL << 4)	/* multicast uses MSG ops */
/*#define FI_PROTO_CAP_COLLECTIVES (1ULL << 5)*/

struct fi_msg {
	const void		*msg_iov;
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
	FI_OPT_MAX_BUFFERED_SEND	/* size_t */
};

struct fi_ops_ep {
	size_t	size;
	int	(*enable)(fid_t fid);
	ssize_t	(*cancel)(fid_t fid, struct fi_context *context);
	int	(*getopt)(fid_t fid, int level, int optname,
			  void *optval, size_t *optlen);
	int	(*setopt)(fid_t fid, int level, int optname,
			  const void *optval, size_t optlen);
};

struct fi_ops_msg {
	size_t	size;
	ssize_t (*recv)(fid_t fid, void *buf, size_t len, void *context);
	ssize_t (*recvmem)(fid_t fid, void *buf, size_t len, uint64_t mem_desc,
			  void *context);
	ssize_t (*recvv)(fid_t fid, const void *iov, size_t count, void *context);
	ssize_t (*recvfrom)(fid_t fid, void *buf, size_t len,
			  const void *src_addr, void *context);
	ssize_t (*recvmemfrom)(fid_t fid, void *buf, size_t len, uint64_t mem_desc,
			  const void *src_addr, void *context);
	ssize_t (*recvmsg)(fid_t fid, const struct fi_msg *msg, uint64_t flags);
	ssize_t (*send)(fid_t fid, const void *buf, size_t len, void *context);
	ssize_t (*sendmem)(fid_t fid, const void *buf, size_t len,
			  uint64_t mem_desc, void *context);
	ssize_t (*sendv)(fid_t fid, const void *iov, size_t count, void *context);
	ssize_t (*sendto)(fid_t fid, const void *buf, size_t len,
			  const void *dest_addr, void *context);
	ssize_t (*sendmemto)(fid_t fid, const void *buf, size_t len, uint64_t mem_desc,
			  const void *dest_addr, void *context);
	ssize_t (*sendmsg)(fid_t fid, const struct fi_msg *msg, uint64_t flags);
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
fi_fendpoint(fid_t fid, struct fi_info *info, fid_t *pep, void *context)
{
	struct fid_fabric *fab = container_of(fid, struct fid_fabric, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_FABRIC);
	FI_ASSERT_OPS(fid, struct fid_fabric, ops);
	FI_ASSERT_OP(fab->ops, struct fi_ops_fabric, endpoint);
	return fab->ops->endpoint(fid, info, pep, context);
}

static inline int
fi_endpoint(fid_t fid, struct fi_info *info, fid_t *ep, void *context)
{
	struct fid_domain *domain = container_of(fid, struct fid_domain, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_DOMAIN);
	FI_ASSERT_OPS(fid, struct fid_domain, ops);
	FI_ASSERT_OP(domain->ops, struct fi_ops_domain, endpoint);
	return domain->ops->endpoint(fid, info, ep, context);
}

static inline ssize_t fi_enable(fid_t fid)
{
	struct fid_ep *ep = container_of(fid, struct fid_ep, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_EP);
	FI_ASSERT_OPS(fid, struct fid_ep, ops);
	FI_ASSERT_OP(ep->ops, struct fi_ops_ep, enable);
	return ep->ops->enable(fid);
}

static inline ssize_t fi_cancel(fid_t fid, struct fi_context *context)
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

static inline ssize_t fi_recvmem(fid_t fid, void *buf, size_t len,
				 uint64_t mem_desc, void *context)
{
	struct fid_ep *ep = container_of(fid, struct fid_ep, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_EP);
	FI_ASSERT_OPS(fid, struct fid_ep, msg);
	FI_ASSERT_OP(ep->msg, struct fi_ops_msg, recvmem);
	return ep->msg->recvmem(fid, buf, len, mem_desc, context);
}

static inline ssize_t fi_sendmem(fid_t fid, void *buf, size_t len,
				 uint64_t mem_desc, void *context)
{
	struct fid_ep *ep = container_of(fid, struct fid_ep, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_EP);
	FI_ASSERT_OPS(fid, struct fid_ep, msg);
	FI_ASSERT_OP(ep->msg, struct fi_ops_msg, sendmem);
	return ep->msg->sendmem(fid, buf, len, mem_desc, context);
}


#else // FABRIC_DIRECT
#include <rdma/fi_direct_endpoint.h>
#endif

#ifdef __cplusplus
}
#endif

#endif /* _FI_ENDPOINT_H_ */
