/*
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
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

#ifndef FI_ENDPOINT_H
#define FI_ENDPOINT_H

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>


#ifdef __cplusplus
extern "C" {
#endif


struct fi_msg {
	const struct iovec	*msgIov;
	void			**desc;
	size_t			iovCount;
	fi_addr_t		addr;
	void			*context;
	uint64_t		data;
};

/* Endpoint option levels */
enum {
	FI_OPT_ENDPOINT
};

/* FI_OPT_ENDPOINT option names */
enum {
	FI_OPT_MIN_MULTI_RECV,		/* size_t */
	FI_OPT_CM_DATA_SIZE,		/* size_t */
	FI_OPT_BUFFERED_MIN,		/* size_t */
	FI_OPT_BUFFERED_LIMIT,		/* size_t */
	FI_OPT_SEND_BUF_SIZE,
	FI_OPT_RECV_BUF_SIZE,
	FI_OPT_TX_SIZE,
	FI_OPT_RX_SIZE,
	FI_OPT_FI_HMEM_P2P,		/* int */
	FI_OPT_XPU_TRIGGER,		/* struct fi_trigger_xpu */
	FI_OPT_CUDA_API_PERMITTED,	/* bool */
};

/*
 * Parameters for FI_OPT_HMEM_P2P to allow endpoint control over peer to peer
 * support and FI_HMEM.
 */
enum {
	FI_HMEM_P2P_ENABLED,	/* Provider decides when to use P2P, default. */
	FI_HMEM_P2P_REQUIRED,	/* Must use P2P for all transfers */
	FI_HMEM_P2P_PREFERRED,	/* Should use P2P for all transfers if available */
	FI_HMEM_P2P_DISABLED	/* Do not use P2P */
};

struct fi_ops_ep {
	size_t	size;
	ssize_t	(*cancel)(fid_t fid, void *context);
	int	(*getopt)(fid_t fid, int level, int optname,
			void *optval, size_t *optlen);
	int	(*setopt)(fid_t fid, int level, int optname,
			const void *optval, size_t optlen);
	int	(*txCtx)(struct fid_ep *sep, int index,
			struct fi_tx_attr *attr, struct fid_ep **txEp,
			void *context);
	int	(*rxCtx)(struct fid_ep *sep, int index,
			struct fi_rx_attr *attr, struct fid_ep **rxEp,
			void *context);
	ssize_t (*rxSizeLeft)(struct fid_ep *ep);
	ssize_t (*txSizeLeft)(struct fid_ep *ep);
};

struct fi_ops_msg {
	size_t	size;
	ssize_t (*recv)(struct fid_ep *ep, void *buf, size_t len, void *desc,
			fi_addr_t srcAddr, void *context);
	ssize_t (*recvv)(struct fid_ep *ep, const struct iovec *iov, void **desc,
			size_t count, fi_addr_t srcAddr, void *context);
	ssize_t (*recvmsg)(struct fid_ep *ep, const struct fi_msg *msg,
			uint64_t flags);
	ssize_t (*send)(struct fid_ep *ep, const void *buf, size_t len, void *desc,
			fi_addr_t destAddr, void *context);
	ssize_t (*sendv)(struct fid_ep *ep, const struct iovec *iov, void **desc,
			size_t count, fi_addr_t destAddr, void *context);
	ssize_t (*sendmsg)(struct fid_ep *ep, const struct fi_msg *msg,
			uint64_t flags);
	ssize_t	(*inject)(struct fid_ep *ep, const void *buf, size_t len,
			fi_addr_t destAddr);
	ssize_t (*senddata)(struct fid_ep *ep, const void *buf, size_t len, void *desc,
			uint64_t data, fi_addr_t destAddr, void *context);
	ssize_t	(*injectdata)(struct fid_ep *ep, const void *buf, size_t len,
			uint64_t data, fi_addr_t destAddr);
};

struct fi_ops_cm;
struct fi_ops_rma;
struct fi_ops_tagged;
struct fi_ops_atomic;
struct fi_ops_collective;

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
	struct fi_ops_collective *collective;
};

struct fid_pep {
	struct fid		fid;
	struct fi_ops_ep	*ops;
	struct fi_ops_cm	*cm;
};

struct fid_stx {
	struct fid		fid;
	struct fi_ops_ep	*ops;
};

#ifdef FABRIC_DIRECT
#include <rdma/fi_direct_endpoint.h>
#endif /* FABRIC_DIRECT */

#ifndef FABRIC_DIRECT_ENDPOINT

static inline int
fiPassiveEp(struct fid_fabric *fabric, struct fi_info *info,
	     struct fid_pep **pep, void *context)
{
	return fabric->ops->passiveEp(fabric, info, pep, context);
}

static inline int
fiEndpoint(struct fid_domain *domain, struct fi_info *info,
	    struct fid_ep **ep, void *context)
{
	return domain->ops->endpoint(domain, info, ep, context);
}

static inline int
fiEndpoint2(struct fid_domain *domain, struct fi_info *info,
	     struct fid_ep **ep, uint64_t flags, void *context)
{
	if (!flags)
		return fiEndpoint(domain, info, ep, context);

	return FI_CHECK_OP(domain->ops, struct fi_ops_domain, endpoint2) ?
		domain->ops->endpoint2(domain, info, ep, flags, context) :
		-FI_ENOSYS;
}

static inline int
fiScalableEp(struct fid_domain *domain, struct fi_info *info,
	    struct fid_ep **sep, void *context)
{
	return domain->ops->scalableEp(domain, info, sep, context);
}

static inline int fiEpBind(struct fid_ep *ep, struct fid *bfid, uint64_t flags)
{
	return ep->fid.ops->bind(&ep->fid, bfid, flags);
}

static inline int fiPepBind(struct fid_pep *pep, struct fid *bfid, uint64_t flags)
{
	return pep->fid.ops->bind(&pep->fid, bfid, flags);
}

static inline int fiScalableEpBind(struct fid_ep *sep, struct fid *bfid, uint64_t flags)
{
	return sep->fid.ops->bind(&sep->fid, bfid, flags);
}

static inline int fiEnable(struct fid_ep *ep)
{
	return ep->fid.ops->control(&ep->fid, FI_ENABLE, NULL);
}

static inline ssize_t fiCancel(fid_t fid, void *context)
{
	struct fid_ep *ep = container_of(fid, struct fid_ep, fid);
	return ep->ops->cancel(fid, context);
}

static inline int
fiSetopt(fid_t fid, int level, int optname,
	  const void *optval, size_t optlen)
{
	struct fid_ep *ep = container_of(fid, struct fid_ep, fid);
	return ep->ops->setopt(fid, level, optname, optval, optlen);
}

static inline int
fiGetopt(fid_t fid, int level, int optname,
	  void *optval, size_t *optlen)
{
	struct fid_ep *ep = container_of(fid, struct fid_ep, fid);
	return ep->ops->getopt(fid, level, optname, optval, optlen);
}

static inline int fiEpAlias(struct fid_ep *ep, struct fid_ep **aliasEp,
			      uint64_t flags)
{
	int ret;
	struct fid *fid;
	ret = fiAlias(&ep->fid, &fid, flags);
	if (!ret)
		*aliasEp = container_of(fid, struct fid_ep, fid);
	return ret;
}

static inline int
fiTxContext(struct fid_ep *ep, int idx, struct fi_tx_attr *attr,
	      struct fid_ep **txEp, void *context)
{
	return ep->ops->txCtx(ep, idx, attr, txEp, context);
}

static inline int
fiRxContext(struct fid_ep *ep, int idx, struct fi_rx_attr *attr,
	      struct fid_ep **rxEp, void *context)
{
	return ep->ops->rxCtx(ep, idx, attr, rxEp, context);
}

static inline FI_DEPRECATED_FUNC ssize_t
fiRxSizeLeft(struct fid_ep *ep)
{
	return ep->ops->rxSizeLeft(ep);
}

static inline FI_DEPRECATED_FUNC ssize_t
fiTxSizeLeft(struct fid_ep *ep)
{
	return ep->ops->txSizeLeft(ep);
}

static inline int
fiStxContext(struct fid_domain *domain, struct fi_tx_attr *attr,
	       struct fid_stx **stx, void *context)
{
	return domain->ops->stxCtx(domain, attr, stx, context);
}

static inline int
fiSrxContext(struct fid_domain *domain, struct fi_rx_attr *attr,
	       struct fid_ep **rxEp, void *context)
{
	return domain->ops->srxCtx(domain, attr, rxEp, context);
}

static inline ssize_t
fiRecv(struct fid_ep *ep, void *buf, size_t len, void *desc, fi_addr_t srcAddr,
	void *context)
{
	return ep->msg->recv(ep, buf, len, desc, srcAddr, context);
}

static inline ssize_t
fiRecvv(struct fid_ep *ep, const struct iovec *iov, void **desc,
	 size_t count, fi_addr_t srcAddr, void *context)
{
	return ep->msg->recvv(ep, iov, desc, count, srcAddr, context);
}

static inline ssize_t
fiRecvmsg(struct fid_ep *ep, const struct fi_msg *msg, uint64_t flags)
{
	return ep->msg->recvmsg(ep, msg, flags);
}

static inline ssize_t
fiSend(struct fid_ep *ep, const void *buf, size_t len, void *desc,
	fi_addr_t destAddr, void *context)
{
	return ep->msg->send(ep, buf, len, desc, destAddr, context);
}

static inline ssize_t
fiSendv(struct fid_ep *ep, const struct iovec *iov, void **desc,
	 size_t count, fi_addr_t destAddr, void *context)
{
	return ep->msg->sendv(ep, iov, desc, count, destAddr, context);
}

static inline ssize_t
fiSendmsg(struct fid_ep *ep, const struct fi_msg *msg, uint64_t flags)
{
	return ep->msg->sendmsg(ep, msg, flags);
}

static inline ssize_t
fiInject(struct fid_ep *ep, const void *buf, size_t len, fi_addr_t destAddr)
{
	return ep->msg->inject(ep, buf, len, destAddr);
}

static inline ssize_t
fiSenddata(struct fid_ep *ep, const void *buf, size_t len, void *desc,
	      uint64_t data, fi_addr_t destAddr, void *context)
{
	return ep->msg->senddata(ep, buf, len, desc, data, destAddr, context);
}

static inline ssize_t
fiInjectdata(struct fid_ep *ep, const void *buf, size_t len,
		uint64_t data, fi_addr_t destAddr)
{
	return ep->msg->injectdata(ep, buf, len, data, destAddr);
}

#endif

#ifdef __cplusplus
}
#endif

#endif /* FI_ENDPOINT_H */
