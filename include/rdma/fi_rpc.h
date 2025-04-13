/*
 * Copyright (c) Intel Corporation. All rights reserved.
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

#ifndef FI_RPC_H
#define FI_RPC_H

#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>


#ifdef __cplusplus
extern "C" {
#endif


struct fi_msg_rpc {
	const struct iovec	*req_iov;
	void			**req_desc;
	size_t			req_iov_count;
	const struct iovec	*resp_iov;
	void			**resp_desc;
	size_t			resp_iov_count;
	fi_addr_t		addr;
	int			timeout;
	void			*context;
	uint64_t		data;
};

struct fi_msg_rpc_resp {
	const struct iovec	*iov;
	void			**desc;
	size_t			iov_count;
	fi_addr_t		addr;
	uint64_t		rpc_id;
	void			*context;
	uint64_t		data;
};

struct fi_ops_rpc {
	size_t	size;
	ssize_t (*req)(struct fid_ep *ep, const void *req_buf, size_t req_len,
		       void *req_desc, void *resp_buf, size_t resp_len,
		       void *resp_desc, fi_addr_t dest_addr, int timeout,
		       void *context);
	ssize_t (*reqv)(struct fid_ep *ep, const struct iovec *req_iov,
			void **req_desc, size_t req_iov_count,
			struct iovec *resp_iov, void **resp_desc,
			size_t resp_iov_count, fi_addr_t dest_addr, int timeout,
			void *context);
	ssize_t (*reqdata)(struct fid_ep *ep, const void *req_buf,
			   size_t req_len, void *req_desc, void *resp_buf,
			   size_t resp_len, void *resp_desc, uint64_t data,
			   fi_addr_t dest_addr, int timeout, void *context);
	ssize_t (*reqmsg)(struct fid_ep *ep, const struct fi_msg_rpc *msg,
			  uint64_t flags);
	ssize_t (*resp)(struct fid_ep *ep, const void *buf, size_t len,
			void *desc, fi_addr_t dest_addr, uint64_t rpc_id,
			void *context);
	ssize_t (*respv)(struct fid_ep *ep, const struct iovec *iov,
			 void **desc, size_t count, fi_addr_t dest_addr,
			 uint64_t rpc_id, void *context);
	ssize_t (*respdata)(struct fid_ep *ep, const void *buf, size_t len,
			    void *desc, uint64_t data, fi_addr_t dest_addr,
			    uint64_t rpc_id, void *context);
	ssize_t (*respmsg)(struct fid_ep *ep, const struct fi_msg_rpc_resp *msg,
			   uint64_t flags);
	ssize_t (*discard)(struct fid_ep *ep, uint64_t rpc_id);
};


#ifdef FABRIC_DIRECT
#include <rdma/fi_direct_tagged.h>
#endif	/* FABRIC_DIRECT */

#ifndef FABRIC_DIRECT_RPC

static inline ssize_t
fi_rpc(struct fid_ep *ep, const void *req_buf, size_t req_len, void *req_desc,
       void *resp_buf, size_t resp_len, void *resp_desc, fi_addr_t dest_addr,
       int timeout, void *context)
{
	return ep->rpc->req(ep, req_buf, req_len, req_desc, resp_buf, resp_len,
			    resp_desc, dest_addr, timeout, context);
}

static inline ssize_t
fi_rpcv(struct fid_ep *ep, const struct iovec *req_iov, void **req_desc,
	size_t req_count, struct iovec *resp_iov, void **resp_desc,
	size_t resp_count, fi_addr_t dest_addr, int timeout, void *context)
{
	return ep->rpc->reqv(ep, req_iov, req_desc, req_count, resp_iov,
			     resp_desc, resp_count, dest_addr, timeout,
			     context);
}

static inline ssize_t
fi_rpcdata(struct fid_ep *ep, const void *req_buf, size_t req_len,
	   void *req_desc, void *resp_buf, size_t resp_len, void *resp_desc,
	   uint64_t data, fi_addr_t dest_addr, int timeout, void *context)
{
	return ep->rpc->reqdata(ep, req_buf, req_len, req_desc, resp_buf,
				resp_len, resp_desc, data, dest_addr, timeout,
				context);
}

static inline ssize_t
fi_rpcmsg(struct fid_ep *ep, const struct fi_msg_rpc *msg, uint64_t flags)
{
	return ep->rpc->reqmsg(ep, msg, flags);
}

static inline ssize_t
fi_rpc_resp(struct fid_ep *ep, const void *buf, size_t len, void *desc,
            fi_addr_t dest_addr, uint64_t rpc_id, void *context)
{
	return ep->rpc->resp(ep, buf, len, desc, dest_addr, rpc_id, context);
}

static inline ssize_t
fi_rpc_respv(struct fid_ep *ep, const struct iovec *iov, void **desc,
	     size_t count, fi_addr_t dest_addr, uint64_t rpc_id, void *context)
{
	return ep->rpc->respv(ep, iov, desc, count, dest_addr, rpc_id, context);
}

static inline ssize_t
fi_rpc_respdata(struct fid_ep *ep, const void *buf, size_t len, void *desc,
		uint64_t data, fi_addr_t dest_addr, uint64_t rpc_id,
		void *context)
{
	return ep->rpc->respdata(ep, buf, len, desc, data, dest_addr, rpc_id,
				 context);
}

static inline ssize_t
fi_rpc_respmsg(struct fid_ep *ep, const struct fi_msg_rpc_resp *msg,
	       uint64_t flags)
{
	return ep->rpc->respmsg(ep, msg, flags);
}

static inline ssize_t
fi_rpc_discard(struct fid_ep *ep, uint64_t rpc_id)
{
	return ep->rpc->discard(ep, rpc_id);
}

#endif

#ifdef __cplusplus
}
#endif

#endif /* FI_RPC_H */
