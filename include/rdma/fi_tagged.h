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

#ifndef FI_TAGGED_H
#define FI_TAGGED_H

#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>


#ifdef __cplusplus
extern "C" {
#endif


struct fi_msg_tagged {
	const struct iovec	*msgIov;
	void			**desc;
	size_t			iovCount;
	fi_addr_t		addr;
	uint64_t		tag;
	uint64_t		ignore;
	void			*context;
	uint64_t		data;
};

struct fi_ops_tagged {
	size_t	size;
	ssize_t (*recv)(struct fid_ep *ep, void *buf, size_t len, void *desc,
			fi_addr_t srcAddr,
			uint64_t tag, uint64_t ignore, void *context);
	ssize_t (*recvv)(struct fid_ep *ep, const struct iovec *iov, void **desc,
			size_t count, fi_addr_t srcAddr,
			uint64_t tag, uint64_t ignore, void *context);
	ssize_t (*recvmsg)(struct fid_ep *ep, const struct fi_msg_tagged *msg,
			uint64_t flags);
	ssize_t (*send)(struct fid_ep *ep, const void *buf, size_t len, void *desc,
			fi_addr_t destAddr, uint64_t tag, void *context);
	ssize_t (*sendv)(struct fid_ep *ep, const struct iovec *iov, void **desc,
			size_t count, fi_addr_t destAddr, uint64_t tag, void *context);
	ssize_t (*sendmsg)(struct fid_ep *ep, const struct fi_msg_tagged *msg,
			uint64_t flags);
	ssize_t	(*inject)(struct fid_ep *ep, const void *buf, size_t len,
			fi_addr_t destAddr, uint64_t tag);
	ssize_t (*senddata)(struct fid_ep *ep, const void *buf, size_t len, void *desc,
			uint64_t data, fi_addr_t destAddr, uint64_t tag, void *context);
	ssize_t	(*injectdata)(struct fid_ep *ep, const void *buf, size_t len,
			uint64_t data, fi_addr_t destAddr, uint64_t tag);
};


#ifdef FABRIC_DIRECT
#include <rdma/fi_direct_tagged.h>
#endif	/* FABRIC_DIRECT */

#ifndef FABRIC_DIRECT_TAGGED

static inline ssize_t
fiTrecv(struct fid_ep *ep, void *buf, size_t len, void *desc,
	 fi_addr_t srcAddr, uint64_t tag, uint64_t ignore, void *context)
{
	return ep->tagged->recv(ep, buf, len, desc, srcAddr, tag, ignore,
				context);
}

static inline ssize_t
fiTrecvv(struct fid_ep *ep, const struct iovec *iov, void **desc,
	  size_t count, fi_addr_t srcAddr, uint64_t tag, uint64_t ignore,
	  void *context)
{
	return ep->tagged->recvv(ep, iov, desc, count, srcAddr, tag, ignore,
				 context);
}

static inline ssize_t
fiTrecvmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg, uint64_t flags)
{
	return ep->tagged->recvmsg(ep, msg, flags);
}

static inline ssize_t
fiTsend(struct fid_ep *ep, const void *buf, size_t len, void *desc,
	 fi_addr_t destAddr, uint64_t tag, void *context)
{
	return ep->tagged->send(ep, buf, len, desc, destAddr, tag, context);
}

static inline ssize_t
fiTsendv(struct fid_ep *ep, const struct iovec *iov, void **desc,
	  size_t count, fi_addr_t destAddr, uint64_t tag, void *context)
{
	return ep->tagged->sendv(ep, iov, desc, count, destAddr,tag, context);
}

static inline ssize_t
fiTsendmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg, uint64_t flags)
{
	return ep->tagged->sendmsg(ep, msg, flags);
}

static inline ssize_t
fiTinject(struct fid_ep *ep, const void *buf, size_t len,
	   fi_addr_t destAddr, uint64_t tag)
{
	return ep->tagged->inject(ep, buf, len, destAddr, tag);
}

static inline ssize_t
fiTsenddata(struct fid_ep *ep, const void *buf, size_t len, void *desc,
	     uint64_t data, fi_addr_t destAddr, uint64_t tag, void *context)
{
	return ep->tagged->senddata(ep, buf, len, desc, data,
				    destAddr, tag, context);
}

static inline ssize_t
fiTinjectdata(struct fid_ep *ep, const void *buf, size_t len,
		uint64_t data, fi_addr_t destAddr, uint64_t tag)
{
	return ep->tagged->injectdata(ep, buf, len, data, destAddr, tag);
}

#endif

#ifdef __cplusplus
}
#endif

#endif /* FI_TAGGED_H */
