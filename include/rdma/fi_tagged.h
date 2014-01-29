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

#ifndef _FI_TAGGED_H_
#define _FI_TAGGED_H_

#include <assert.h>
#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>


#ifdef __cplusplus
extern "C" {
#endif

struct fi_msg_tagged {
	const void		*msg_iov;
	size_t			iov_count;
	const void		*addr;
	uint64_t		tag;
	uint64_t		mask;
	void			*context;
	uint64_t		data;
	int			priority;
};

struct fi_ops_tagged {
	size_t	size;
	ssize_t (*recv)(fid_t fid, void *buf, size_t len,
			uint64_t tag, uint64_t mask, void *context);
	ssize_t (*recvv)(fid_t fid, const void *iov, size_t count,
			 uint64_t tag, uint64_t mask, void *context);
	ssize_t (*recvfrom)(fid_t fid, void *buf, size_t len, const void *src_addr,
			    uint64_t tag, uint64_t mask, void *context);
	ssize_t (*recvmsg)(fid_t fid, const struct fi_msg_tagged *msg, uint64_t flags);
	ssize_t (*send)(fid_t fid, const void *buf, size_t len, uint64_t tag,
			void *context);
	ssize_t (*sendv)(fid_t fid, const void *iov, size_t count, uint64_t tag,
			 void *context);
	ssize_t (*sendto)(fid_t fid, const void *buf, size_t len,
			  const void *dest_addr, uint64_t tag, void *context);
	ssize_t (*sendmsg)(fid_t fid, const struct fi_msg_tagged *msg, uint64_t flags);
	ssize_t (*search)(fid_t fid, uint64_t *tag, uint64_t mask, uint64_t flags,
			  void *src_addr, size_t *src_addrlen, size_t *len, void *context);
};

static inline ssize_t
fi_tsendto(fid_t fid, const void *buf, size_t len,
	   const void *dest_addr, uint64_t tag, void *context)
{
	struct fid_ep *ep = container_of(fid, struct fid_ep, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_EP);
	FI_ASSERT_OPS(fid, struct fid_ep, tagged);
	FI_ASSERT_OP(ep->tagged, struct fi_ops_tagged, sendto);
	return ep->tagged->sendto(fid, buf, len, dest_addr, tag, context);
}

static inline ssize_t
fi_trecvfrom(fid_t fid, void *buf, size_t len, const void *src_addr,
	     uint64_t tag, uint64_t mask, void *context)
{
	struct fid_ep *ep = container_of(fid, struct fid_ep, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_EP);
	FI_ASSERT_OPS(fid, struct fid_ep, tagged);
	FI_ASSERT_OP(ep->tagged, struct fi_ops_tagged, recvfrom);
	return ep->tagged->recvfrom(fid, buf, len, src_addr, tag, mask, context);
}

static inline ssize_t
fi_tsearch(fid_t fid, uint64_t *tag, uint64_t mask, uint64_t flags,
	   void *src_addr, size_t *src_addrlen, size_t *len, void *context)
{
	struct fid_ep *ep = container_of(fid, struct fid_ep, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_EP);
	FI_ASSERT_OPS(fid, struct fid_ep, tagged);
	FI_ASSERT_OP(ep->tagged, struct fi_ops_tagged, search);
	return ep->tagged->search(fid, tag, mask, flags, src_addr, src_addrlen, len, context);
}

#ifdef __cplusplus
}
#endif

#endif /* _FI_TAGGED_H_ */
