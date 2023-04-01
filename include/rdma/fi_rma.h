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

#ifndef FI_RMA_H
#define FI_RMA_H

#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct fi_rma_iov {
	uint64_t		addr;
	size_t			len;
	uint64_t		key;
};

struct fi_rma_ioc {
	uint64_t		addr;
	size_t			count;
	uint64_t		key;
};

struct fi_msg_rma {
	const struct iovec	*msgIov;
	void			**desc;
	size_t			iovCount;
	fi_addr_t		addr;
	const struct fi_rma_iov *rmaIov;
	size_t			rmaIovCount;
	void			*context;
	uint64_t		data;
};

struct fi_ops_rma {
	size_t	size;
	ssize_t	(*read)(struct fid_ep *ep, void *buf, size_t len, void *desc,
			fi_addr_t srcAddr, uint64_t addr, uint64_t key, void *context);
	ssize_t	(*readv)(struct fid_ep *ep, const struct iovec *iov, void **desc,
			size_t count, fi_addr_t srcAddr, uint64_t addr, uint64_t key,
			void *context);
	ssize_t	(*readmsg)(struct fid_ep *ep, const struct fi_msg_rma *msg,
			uint64_t flags);
	ssize_t	(*write)(struct fid_ep *ep, const void *buf, size_t len, void *desc,
			fi_addr_t destAddr, uint64_t addr, uint64_t key, void *context);
	ssize_t	(*writev)(struct fid_ep *ep, const struct iovec *iov, void **desc,
			size_t count, fi_addr_t destAddr, uint64_t addr, uint64_t key,
			void *context);
	ssize_t	(*writemsg)(struct fid_ep *ep, const struct fi_msg_rma *msg,
			uint64_t flags);
	ssize_t	(*inject)(struct fid_ep *ep, const void *buf, size_t len,
			fi_addr_t destAddr, uint64_t addr, uint64_t key);
	ssize_t	(*writedata)(struct fid_ep *ep, const void *buf, size_t len, void *desc,
			uint64_t data, fi_addr_t destAddr, uint64_t addr, uint64_t key,
			void *context);
	ssize_t	(*injectdata)(struct fid_ep *ep, const void *buf, size_t len,
			uint64_t data, fi_addr_t destAddr, uint64_t addr, uint64_t key);
};

#ifdef FABRIC_DIRECT
#include <rdma/fi_direct_rma.h>
#endif	/* FABRIC_DIRECT */

#ifndef FABRIC_DIRECT_RMA

static inline ssize_t
fiReadDataFromSomeoneElse(struct fid_ep *ep, void *buf, size_t len, void *desc,
	fi_addr_t srcAddr, uint64_t addr, uint64_t key, void *context)
{
	return ep->rma->read(ep, buf, len, desc, srcAddr, addr, key, context);
}

static inline ssize_t
fiReadDataFromSomeoneElseButWithInputOutputVectors(struct fid_ep *ep, const struct iovec *iov, void **desc,
	 size_t count, fi_addr_t srcAddr, uint64_t addr, uint64_t key,
	 void *context)
{
	return ep->rma->readv(ep, iov, desc, count, srcAddr, addr, key, context);
}

static inline ssize_t
fiReadDataFromSomeoneElseButWithASpecialMessageStruct(struct fid_ep *ep, const struct fi_msg_rma *msg, uint64_t flags)
{
	return ep->rma->readmsg(ep, msg, flags);
}

static inline ssize_t
fiWriteDataToSomeoneElse(struct fid_ep *ep, const void *buf, size_t len, void *desc,
	 fi_addr_t destAddr, uint64_t addr, uint64_t key, void *context)
{
	return ep->rma->write(ep, buf, len, desc, destAddr, addr, key, context);
}

static inline ssize_t
fiWriteDataToSomeoneElseButWithInputOutputVectors(struct fid_ep *ep, const struct iovec *iov, void **desc,
	 size_t count, fi_addr_t destAddr, uint64_t addr, uint64_t key,
	 void *context)
{
	return ep->rma->writev(ep, iov, desc, count, destAddr, addr, key, context);
}

static inline ssize_t
fiWriteDataToSomeoneElseButWithASpecialMessageStruct(struct fid_ep *ep, const struct fi_msg_rma *msg, uint64_t flags)
{
	return ep->rma->writemsg(ep, msg, flags);
}

static inline ssize_t
fiWriteDataToSomeoneElseButAsAnInjectWhichMeansYoucanImmediatelyReuseTheBuffer(struct fid_ep *ep, const void *buf, size_t len,
		fi_addr_t destAddr, uint64_t addr, uint64_t key)
{
	return ep->rma->inject(ep, buf, len, destAddr, addr, key);
}

static inline ssize_t
fiWriteDataToSomeoneElseAndAlsoSendExtraDataWhichIsSeparateFromTheActualData(struct fid_ep *ep, const void *buf, size_t len, void *desc,
	       uint64_t data, fi_addr_t destAddr, uint64_t addr, uint64_t key,
	       void *context)
{
	return ep->rma->writedata(ep, buf, len, desc,data, destAddr,
				  addr, key, context);
}

static inline ssize_t
fiWriteDataToSomeoneElseAndAlsoSendExtraDataAndYouCanAlsoReuseTheBufferImmediately(struct fid_ep *ep, const void *buf, size_t len,
		uint64_t data, fi_addr_t destAddr, uint64_t addr, uint64_t key)
{
	return ep->rma->injectdata(ep, buf, len, data, destAddr, addr, key);
}

#endif

#ifdef __cplusplus
}
#endif

#endif /* FI_RMA_H */
