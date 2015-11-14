/*
 * Copyright (c) 2015 Cray Inc. All rights reserved.
 * Copyright (c) 2015 Los Alamos National Security, LLC. All rights reserved.
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

#ifndef _GNIX_EP_MSG_H_
#define _GNIX_EP_MSG_H_

#include "gnix.h"

/*
 * Entry points for FI_EP_MSG data motion methods.
 * This file is intended to be included only in gnix_ep.c
 */

static ssize_t gnix_ep_sendv_msg(struct fid_ep *ep, const struct iovec *iov,
				 void **desc, size_t count, fi_addr_t dest_addr,
				 void *context)
{
	return -FI_ENOSYS;
}

static ssize_t gnix_ep_sendmsg_msg(struct fid_ep *ep, const struct fi_msg *msg,
				   uint64_t flags)
{
	return -FI_ENOSYS;
}

static ssize_t gnix_ep_msg_inject_msg(struct fid_ep *ep, const void *buf,
				      size_t len, fi_addr_t dest_addr)
{
	return -FI_ENOSYS;
}


static ssize_t gnix_ep_recvv_msg(struct fid_ep *ep, const struct iovec *iov,
				 void **desc, size_t count, fi_addr_t dest_addr,
				 void *context)
{
	return -FI_ENOSYS;
}

static ssize_t gnix_ep_recvmsg_msg(struct fid_ep *ep, const struct fi_msg *msg,
				   uint64_t flags)
{
	return -FI_ENOSYS;
}

static ssize_t gnix_ep_tsend_msg(struct fid_ep *ep, const void *buf, size_t len,
				 void *desc, fi_addr_t dest_addr, uint64_t tag,
				 void *context)
{
	return -FI_ENOSYS;
}

static ssize_t gnix_ep_tsendv_msg(struct fid_ep *ep, const struct iovec *iov,
				  void **desc, size_t count,
				  fi_addr_t dest_addr,
				  uint64_t tag, void *context)
{
	return -FI_ENOSYS;
}

static ssize_t gnix_ep_tsendmsg_msg(struct fid_ep *ep,
				    const struct fi_msg_tagged *msg,
				    uint64_t flags)
{
	return -FI_ENOSYS;
}

static ssize_t gnix_ep_tinject_msg(struct fid_ep *ep, const void *buf,
				   size_t len, fi_addr_t dest_addr,
				   uint64_t tag)
{
	return -FI_ENOSYS;
}

static ssize_t gnix_ep_trecv_msg(struct fid_ep *ep, void *buf, size_t len,
			  void *desc, fi_addr_t src_addr, uint64_t tag,
			  uint64_t ignore, void *context)
{
	return -FI_ENOSYS;
}

static ssize_t gnix_ep_trecvv_msg(struct fid_ep *ep, const struct iovec *iov,
				  void **desc, size_t count, fi_addr_t src_addr,
				  uint64_t tag, uint64_t ignore, void *context)
{
	return -FI_ENOSYS;
}

static ssize_t gnix_ep_trecvmsg_msg(struct fid_ep *ep,
				    const struct fi_msg_tagged *msg,
				    uint64_t flags)
{
	return -FI_ENOSYS;
}

#endif /* _GNIX_EP_MSG_H_ */
