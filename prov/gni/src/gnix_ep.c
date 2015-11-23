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

/*
 * Endpoint common code
 */
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "gnix.h"
#include "gnix_cm_nic.h"
#include "gnix_nic.h"
#include "gnix_util.h"
#include "gnix_ep.h"
#include "gnix_hashtable.h"
#include "gnix_vc.h"
#include "gnix_msg.h"
#include "gnix_rma.h"
#include "gnix_atomic.h"
#include "gnix_cntr.h"


/*******************************************************************************
 * gnix_fab_req freelist functions
 *
 * These are wrappers around the gnix_s_freelist
 *
 ******************************************************************************/

#define GNIX_FAB_REQ_FL_MIN_SIZE 100
#define GNIX_FAB_REQ_FL_REFILL_SIZE 10

static int __fr_freelist_init(struct gnix_fid_ep *ep)
{
	assert(ep);
	return _gnix_sfl_init_ts(sizeof(struct gnix_fab_req),
				offsetof(struct gnix_fab_req, slist),
				GNIX_FAB_REQ_FL_MIN_SIZE,
				GNIX_FAB_REQ_FL_REFILL_SIZE,
				0, 0, &ep->fr_freelist);
}

static void __fr_freelist_destroy(struct gnix_fid_ep *ep)
{
	assert(ep);
	_gnix_sfl_destroy(&ep->fr_freelist);
}

/*******************************************************************************
 * Forward declaration for ops structures.
 ******************************************************************************/

static struct fi_ops gnix_ep_fi_ops;
static struct fi_ops_ep gnix_ep_ops;
static struct fi_ops_msg gnix_ep_msg_ops;
static struct fi_ops_rma gnix_ep_rma_ops;
struct fi_ops_tagged gnix_ep_tagged_ops;
struct fi_ops_atomic gnix_ep_atomic_ops;

/*******************************************************************************
 * EP common messaging wrappers.
 ******************************************************************************/

static inline ssize_t __ep_recv(struct fid_ep *ep, void *buf, size_t len,
				void *desc, fi_addr_t src_addr, void *context,
				uint64_t flags, uint64_t tag, uint64_t ignore)
{
	struct gnix_fid_ep *ep_priv;

	if (!ep) {
		return -FI_EINVAL;
	}

	ep_priv = container_of(ep, struct gnix_fid_ep, ep_fid);
	assert((ep_priv->type == FI_EP_RDM) || (ep_priv->type == FI_EP_MSG));

	return _gnix_recv(ep_priv, (uint64_t)buf, len, desc, src_addr, context,
			  ep_priv->op_flags | flags, tag, ignore);
}

static inline ssize_t __ep_recvv(struct fid_ep *ep, const struct iovec *iov,
				 void **desc, size_t count, fi_addr_t src_addr,
				 void *context, uint64_t flags, uint64_t tag,
				 uint64_t ignore)
{
	struct gnix_fid_ep *ep_priv;

	if (!ep || !iov || count != 1) {
		return -FI_EINVAL;
	}

	ep_priv = container_of(ep, struct gnix_fid_ep, ep_fid);
	assert((ep_priv->type == FI_EP_RDM) || (ep_priv->type == FI_EP_MSG));

	return _gnix_recv(ep_priv, (uint64_t)iov[0].iov_base, iov[0].iov_len,
			  desc ? desc[0] : NULL, src_addr, context,
			  ep_priv->op_flags | flags, tag, ignore);
}

static inline ssize_t __ep_recvmsg(struct fid_ep *ep, const struct fi_msg *msg,
				   uint64_t flags, uint64_t tag,
				   uint64_t ignore)
{
	struct gnix_fid_ep *ep_priv;

	if (!ep || !msg || !msg->msg_iov || msg->iov_count != 1) {
		return -FI_EINVAL;
	}

	ep_priv = container_of(ep, struct gnix_fid_ep, ep_fid);
	assert((ep_priv->type == FI_EP_RDM) || (ep_priv->type == FI_EP_MSG));

	return _gnix_recv(ep_priv, (uint64_t)msg->msg_iov[0].iov_base,
			  msg->msg_iov[0].iov_len,
			  msg->desc ? msg->desc[0] : NULL,
			  msg->addr, msg->context, flags, tag, ignore);
}

static inline ssize_t __ep_send(struct fid_ep *ep, const void *buf, size_t len,
				void *desc, fi_addr_t dest_addr, void *context,
				uint64_t flags, uint64_t tag)
{
	struct gnix_fid_ep *gnix_ep;

	if (!ep) {
		return -FI_EINVAL;
	}

	gnix_ep = container_of(ep, struct gnix_fid_ep, ep_fid);
	assert((gnix_ep->type == FI_EP_RDM) || (gnix_ep->type == FI_EP_MSG));

	return _gnix_send(gnix_ep, (uint64_t)buf, len, desc, dest_addr, context,
			  gnix_ep->op_flags | flags, 0, tag);
}

static inline ssize_t __ep_sendv(struct fid_ep *ep, const struct iovec *iov,
				 void **desc, size_t count, fi_addr_t dest_addr,
				 void *context, uint64_t flags, uint64_t tag)
{
	struct gnix_fid_ep *gnix_ep;

	if (!ep || !iov || count != 1) {
		return -FI_EINVAL;
	}

	gnix_ep = container_of(ep, struct gnix_fid_ep, ep_fid);
	assert((gnix_ep->type == FI_EP_RDM) || (gnix_ep->type == FI_EP_MSG));

	return _gnix_send(gnix_ep, (uint64_t)iov[0].iov_base, iov[0].iov_len,
			  desc ? desc[0] : NULL, dest_addr, context,
			  gnix_ep->op_flags | flags, 0, tag);
}

static inline ssize_t __ep_sendmsg(struct fid_ep *ep, const struct fi_msg *msg,
				   uint64_t flags, uint64_t tag)
{
	struct gnix_fid_ep *gnix_ep;

	if (!ep || !msg || !msg->msg_iov || msg->iov_count != 1) {
		return -FI_EINVAL;
	}

	gnix_ep = container_of(ep, struct gnix_fid_ep, ep_fid);
	assert((gnix_ep->type == FI_EP_RDM) || (gnix_ep->type == FI_EP_MSG));

	return _gnix_send(gnix_ep, (uint64_t)msg->msg_iov[0].iov_base,
			  msg->msg_iov[0].iov_len,
			  msg->desc ? msg->desc[0] : NULL, msg->addr,
			  msg->context, flags, msg->data, tag);
}

static inline ssize_t __ep_inject(struct fid_ep *ep, const void *buf,
				  size_t len, fi_addr_t dest_addr,
				  uint64_t flags, uint64_t tag)
{
	struct gnix_fid_ep *gnix_ep;
	uint64_t inject_flags;

	if (!ep) {
		return -FI_EINVAL;
	}

	gnix_ep = container_of(ep, struct gnix_fid_ep, ep_fid);
	assert((gnix_ep->type == FI_EP_RDM) || (gnix_ep->type == FI_EP_MSG));

	inject_flags = (gnix_ep->op_flags | FI_INJECT |
			GNIX_SUPPRESS_COMPLETION | flags);

	return _gnix_send(gnix_ep, (uint64_t)buf, len, NULL, dest_addr,
			  NULL, inject_flags, 0, tag);
}

static inline ssize_t __ep_senddata(struct fid_ep *ep, const void *buf,
				    size_t len, void *desc, uint64_t data,
				    fi_addr_t dest_addr, void *context,
				    uint64_t flags, uint64_t tag)
{
	struct gnix_fid_ep *gnix_ep;
	uint64_t sd_flags;

	if (!ep) {
		return -FI_EINVAL;
	}

	gnix_ep = container_of(ep, struct gnix_fid_ep, ep_fid);
	assert((gnix_ep->type == FI_EP_RDM) || (gnix_ep->type == FI_EP_MSG));

	sd_flags = gnix_ep->op_flags | FI_REMOTE_CQ_DATA | flags;

	return _gnix_send(gnix_ep, (uint64_t)buf, len, desc, dest_addr,
			  context, sd_flags, data, tag);
}

/*******************************************************************************
 * EP messaging API function implementations.
 ******************************************************************************/

static ssize_t gnix_ep_recv(struct fid_ep *ep, void *buf, size_t len,
			    void *desc, fi_addr_t src_addr, void *context)
{
	return __ep_recv(ep, buf, len, desc, src_addr, context, 0, 0, 0);
}

static ssize_t gnix_ep_recvv(struct fid_ep *ep, const struct iovec *iov,
			     void **desc, size_t count, fi_addr_t src_addr,
			     void *context)
{
	return __ep_recvv(ep, iov, desc, count, src_addr, context, 0, 0, 0);
}

static ssize_t gnix_ep_recvmsg(struct fid_ep *ep, const struct fi_msg *msg,
			       uint64_t flags)
{
	return __ep_recvmsg(ep, msg, flags & GNIX_RECVMSG_FLAGS, 0, 0);
}

ssize_t gnix_ep_send(struct fid_ep *ep, const void *buf, size_t len,
		     void *desc, fi_addr_t dest_addr, void *context)
{
	return __ep_send(ep, buf, len, desc, dest_addr, context, 0, 0);
}

static ssize_t gnix_ep_sendv(struct fid_ep *ep, const struct iovec *iov,
			     void **desc, size_t count, fi_addr_t dest_addr,
			     void *context)
{
	return __ep_sendv(ep, iov, desc, count, dest_addr, context, 0, 0);
}

static ssize_t gnix_ep_sendmsg(struct fid_ep *ep, const struct fi_msg *msg,
			uint64_t flags)
{
	return __ep_sendmsg(ep, msg, flags & GNIX_SENDMSG_FLAGS, 0);
}

static ssize_t gnix_ep_msg_inject(struct fid_ep *ep, const void *buf,
				  size_t len, fi_addr_t dest_addr)
{
	return __ep_inject(ep, buf, len, dest_addr, 0, 0);
}

ssize_t gnix_ep_senddata(struct fid_ep *ep, const void *buf, size_t len,
			 void *desc, uint64_t data, fi_addr_t dest_addr,
			 void *context)
{
	return __ep_senddata(ep, buf, len, desc, data, dest_addr,
			context, 0, 0);
}

ssize_t gnix_ep_msg_injectdata(struct fid_ep *ep, const void *buf, size_t len,
			       uint64_t data, fi_addr_t dest_addr)
{
	struct gnix_fid_ep *gnix_ep;
	uint64_t flags;

	if (!ep) {
		return -FI_EINVAL;
	}

	gnix_ep = container_of(ep, struct gnix_fid_ep, ep_fid);
	assert((gnix_ep->type == FI_EP_RDM) || (gnix_ep->type == FI_EP_MSG));

	flags = gnix_ep->op_flags | FI_INJECT | FI_REMOTE_CQ_DATA |
			GNIX_SUPPRESS_COMPLETION;

	return _gnix_send(gnix_ep, (uint64_t)buf, len, NULL, dest_addr,
			  NULL, flags, data, 0);
}


/*******************************************************************************
 * EP RMA API function implementations.
 ******************************************************************************/

#define GNIX_RMA_READ_FLAGS_DEF		(FI_RMA | FI_READ)
#define GNIX_RMA_WRITE_FLAGS_DEF	(FI_RMA | FI_WRITE)

static ssize_t gnix_ep_read(struct fid_ep *ep, void *buf, size_t len,
			    void *desc, fi_addr_t src_addr, uint64_t addr,
			    uint64_t key, void *context)
{
	struct gnix_fid_ep *gnix_ep;
	uint64_t flags;

	if (!ep) {
		return -FI_EINVAL;
	}

	gnix_ep = container_of(ep, struct gnix_fid_ep, ep_fid);
	assert((gnix_ep->type == FI_EP_RDM) || (gnix_ep->type == FI_EP_MSG));

	flags = gnix_ep->op_flags | GNIX_RMA_READ_FLAGS_DEF;

	return _gnix_rma(gnix_ep, GNIX_FAB_RQ_RDMA_READ,
			 (uint64_t)buf, len, desc,
			 src_addr, addr, key,
			 context, flags, 0);
}

static ssize_t gnix_ep_readv(struct fid_ep *ep, const struct iovec *iov,
			     void **desc, size_t count, fi_addr_t src_addr,
			     uint64_t addr, uint64_t key, void *context)
{
	struct gnix_fid_ep *gnix_ep;
	uint64_t flags;

	if (!ep || !iov || !desc || count != 1) {
		return -FI_EINVAL;
	}

	gnix_ep = container_of(ep, struct gnix_fid_ep, ep_fid);
	assert((gnix_ep->type == FI_EP_RDM) || (gnix_ep->type == FI_EP_MSG));

	flags = gnix_ep->op_flags | GNIX_RMA_READ_FLAGS_DEF;

	return _gnix_rma(gnix_ep, GNIX_FAB_RQ_RDMA_READ,
			 (uint64_t)iov[0].iov_base, iov[0].iov_len, desc[0],
			 src_addr, addr, key,
			 context, flags, 0);
}

static ssize_t gnix_ep_readmsg(struct fid_ep *ep, const struct fi_msg_rma *msg,
			       uint64_t flags)
{
	struct gnix_fid_ep *gnix_ep;

	if (!ep || !msg || !msg->msg_iov || !msg->rma_iov || !msg->desc ||
	    msg->iov_count != 1 || msg->rma_iov_count != 1 ||
	    msg->rma_iov[0].len > msg->msg_iov[0].iov_len) {
		return -FI_EINVAL;
	}

	gnix_ep = container_of(ep, struct gnix_fid_ep, ep_fid);
	assert((gnix_ep->type == FI_EP_RDM) || (gnix_ep->type == FI_EP_MSG));

	flags = (flags & GNIX_READMSG_FLAGS) | GNIX_RMA_READ_FLAGS_DEF;

	return _gnix_rma(gnix_ep, GNIX_FAB_RQ_RDMA_READ,
			 (uint64_t)msg->msg_iov[0].iov_base,
			 msg->msg_iov[0].iov_len, msg->desc[0],
			 msg->addr, msg->rma_iov[0].addr, msg->rma_iov[0].key,
			 msg->context, flags, msg->data);
}

static ssize_t gnix_ep_write(struct fid_ep *ep, const void *buf, size_t len,
			     void *desc, fi_addr_t dest_addr, uint64_t addr,
			     uint64_t key, void *context)
{
	struct gnix_fid_ep *gnix_ep;
	uint64_t flags;

	if (!ep) {
		return -FI_EINVAL;
	}

	gnix_ep = container_of(ep, struct gnix_fid_ep, ep_fid);
	assert((gnix_ep->type == FI_EP_RDM) || (gnix_ep->type == FI_EP_MSG));

	flags = gnix_ep->op_flags | GNIX_RMA_WRITE_FLAGS_DEF;

	return _gnix_rma(gnix_ep, GNIX_FAB_RQ_RDMA_WRITE,
			 (uint64_t)buf, len, desc, dest_addr, addr, key,
			 context, flags, 0);
}

static ssize_t gnix_ep_writev(struct fid_ep *ep, const struct iovec *iov,
			      void **desc, size_t count, fi_addr_t dest_addr,
			      uint64_t addr, uint64_t key, void *context)
{
	struct gnix_fid_ep *gnix_ep;
	uint64_t flags;

	if (!ep || !iov || !desc || count != 1) {
		return -FI_EINVAL;
	}

	gnix_ep = container_of(ep, struct gnix_fid_ep, ep_fid);
	assert((gnix_ep->type == FI_EP_RDM) || (gnix_ep->type == FI_EP_MSG));

	flags = gnix_ep->op_flags | GNIX_RMA_WRITE_FLAGS_DEF;

	return _gnix_rma(gnix_ep, GNIX_FAB_RQ_RDMA_WRITE,
			 (uint64_t)iov[0].iov_base, iov[0].iov_len, desc[0],
			 dest_addr, addr, key, context, flags, 0);
}

static ssize_t gnix_ep_writemsg(struct fid_ep *ep, const struct fi_msg_rma *msg,
				uint64_t flags)
{
	struct gnix_fid_ep *gnix_ep;

	if (!ep || !msg || !msg->msg_iov || !msg->rma_iov || !msg->desc ||
	    msg->iov_count != 1 || msg->rma_iov_count != 1 ||
	    msg->rma_iov[0].len > msg->msg_iov[0].iov_len) {
		return -FI_EINVAL;
	}

	gnix_ep = container_of(ep, struct gnix_fid_ep, ep_fid);
	assert((gnix_ep->type == FI_EP_RDM) || (gnix_ep->type == FI_EP_MSG));

	flags = (flags & GNIX_WRITEMSG_FLAGS) | GNIX_RMA_WRITE_FLAGS_DEF;

	return _gnix_rma(gnix_ep, GNIX_FAB_RQ_RDMA_WRITE,
			 (uint64_t)msg->msg_iov[0].iov_base,
			 msg->msg_iov[0].iov_len, msg->desc[0],
			 msg->addr, msg->rma_iov[0].addr, msg->rma_iov[0].key,
			 msg->context, flags, msg->data);
}

static ssize_t gnix_ep_rma_inject(struct fid_ep *ep, const void *buf,
				  size_t len, fi_addr_t dest_addr,
				  uint64_t addr, uint64_t key)
{
	struct gnix_fid_ep *gnix_ep;
	uint64_t flags;

	if (!ep) {
		return -FI_EINVAL;
	}

	gnix_ep = container_of(ep, struct gnix_fid_ep, ep_fid);
	assert((gnix_ep->type == FI_EP_RDM) || (gnix_ep->type == FI_EP_MSG));

	flags = gnix_ep->op_flags | FI_INJECT | GNIX_SUPPRESS_COMPLETION |
			GNIX_RMA_WRITE_FLAGS_DEF;

	return _gnix_rma(gnix_ep, GNIX_FAB_RQ_RDMA_WRITE,
			 (uint64_t)buf, len, NULL,
			 dest_addr, addr, key,
			 NULL, flags, 0);
}

static ssize_t gnix_ep_writedata(struct fid_ep *ep, const void *buf,
				 size_t len, void *desc, uint64_t data,
				 fi_addr_t dest_addr, uint64_t addr,
				 uint64_t key, void *context)
{
	struct gnix_fid_ep *gnix_ep;
	uint64_t flags;

	if (!ep) {
		return -FI_EINVAL;
	}

	gnix_ep = container_of(ep, struct gnix_fid_ep, ep_fid);
	assert((gnix_ep->type == FI_EP_RDM) || (gnix_ep->type == FI_EP_MSG));

	flags = gnix_ep->op_flags | FI_REMOTE_CQ_DATA |
			GNIX_RMA_WRITE_FLAGS_DEF;

	return _gnix_rma(gnix_ep, GNIX_FAB_RQ_RDMA_WRITE,
			 (uint64_t)buf, len, desc,
			 dest_addr, addr, key,
			 context, flags, data);
}

static ssize_t gnix_ep_rma_injectdata(struct fid_ep *ep, const void *buf,
				      size_t len, uint64_t data,
				      fi_addr_t dest_addr, uint64_t addr,
				      uint64_t key)
{
	struct gnix_fid_ep *gnix_ep;
	uint64_t flags;

	if (!ep) {
		return -FI_EINVAL;
	}

	gnix_ep = container_of(ep, struct gnix_fid_ep, ep_fid);
	assert((gnix_ep->type == FI_EP_RDM) || (gnix_ep->type == FI_EP_MSG));

	flags = gnix_ep->op_flags | FI_INJECT | FI_REMOTE_CQ_DATA |
			GNIX_SUPPRESS_COMPLETION | GNIX_RMA_WRITE_FLAGS_DEF;

	return _gnix_rma(gnix_ep, GNIX_FAB_RQ_RDMA_WRITE,
			 (uint64_t)buf, len, NULL,
			 dest_addr, addr, key,
			 NULL, flags, data);
}

/*******************************************************************************
 * EP Tag matching API function implementations.
 ******************************************************************************/

static ssize_t gnix_ep_trecv(struct fid_ep *ep, void *buf, size_t len,
			     void *desc, fi_addr_t src_addr, uint64_t tag,
			     uint64_t ignore, void *context)
{
	return __ep_recv(ep, buf, len, desc, src_addr, context,
			FI_TAGGED, tag, ignore);
}

static ssize_t gnix_ep_trecvv(struct fid_ep *ep, const struct iovec *iov,
			      void **desc, size_t count, fi_addr_t src_addr,
			      uint64_t tag, uint64_t ignore, void *context)
{
	return __ep_recvv(ep, iov, desc, count, src_addr, context,
			FI_TAGGED, tag, ignore);
}

static ssize_t gnix_ep_trecvmsg(struct fid_ep *ep,
				const struct fi_msg_tagged *msg, uint64_t flags)
{
	uint64_t clean_flags;

	const struct fi_msg _msg = {
			.msg_iov = msg->msg_iov,
			.desc = msg->desc,
			.iov_count = msg->iov_count,
			.addr = msg->addr,
			.context = msg->context,
			.data = msg->data
	};

	clean_flags = (flags & (GNIX_TRECVMSG_FLAGS)) | FI_TAGGED;

	/* From the fi_tagged man page regarding the use of FI_CLAIM:
	 *
	 * In order to use the FI_CLAIM flag, an  application  must  supply  a
	 * struct  fi_context  structure as the context for the receive opera-
	 * tion.  The same fi_context structure used for an FI_PEEK + FI_CLAIM
	 * operation must be used by the paired FI_CLAIM requests
	 */
	if ((flags & FI_CLAIM) && _msg.context == NULL)
		return -FI_EINVAL;

	/* From the fi_tagged man page regarding the use of FI_DISCARD:
	 *
	 * This  flag  must  be used in conjunction with either
	 * FI_PEEK or FI_CLAIM.
	 *
	 * Note: I suspect the use of all three flags at the same time is invalid,
	 * but the man page does not say that it is.
	 */
	if ((flags & FI_DISCARD) && !(flags & (FI_PEEK | FI_CLAIM)))
		return -FI_EINVAL;

	return __ep_recvmsg(ep, &_msg, clean_flags, msg->tag,
			msg->ignore);
}

static ssize_t gnix_ep_tsend(struct fid_ep *ep, const void *buf, size_t len, void *desc,
			     fi_addr_t dest_addr, uint64_t tag, void *context)
{
	return __ep_send(ep, buf, len, desc, dest_addr, context,
			FI_TAGGED, tag);
}

static ssize_t gnix_ep_tsendv(struct fid_ep *ep, const struct iovec *iov,
			      void **desc, size_t count, fi_addr_t dest_addr,
			      uint64_t tag, void *context)
{
	return __ep_sendv(ep, iov, desc, count, dest_addr, context,
			FI_TAGGED, tag);
}

static ssize_t gnix_ep_tsendmsg(struct fid_ep *ep,
				const struct fi_msg_tagged *msg, uint64_t flags)
{
	uint64_t clean_flags;

	const struct fi_msg _msg = {
			.msg_iov = msg->msg_iov,
			.desc = msg->desc,
			.iov_count = msg->iov_count,
			.addr = msg->addr,
			.context = msg->context,
			.data = msg->data
	};

	clean_flags = (flags & GNIX_SENDMSG_FLAGS) | FI_TAGGED;

	return __ep_sendmsg(ep, &_msg, clean_flags, msg->tag);
}

static ssize_t gnix_ep_tinject(struct fid_ep *ep, const void *buf, size_t len,
			       fi_addr_t dest_addr, uint64_t tag)
{
	return __ep_inject(ep, buf, len, dest_addr, FI_TAGGED, tag);
}

ssize_t gnix_ep_tsenddata(struct fid_ep *ep, const void *buf, size_t len,
			  void *desc, uint64_t data, fi_addr_t dest_addr,
			  uint64_t tag, void *context)
{
	return __ep_senddata(ep, buf, len, desc, data, dest_addr, context,
			FI_TAGGED, tag);
}


/*******************************************************************************
 * EP atomic API implementation.
 ******************************************************************************/

#define GNIX_ATOMIC_WRITE_FLAGS_DEF	(FI_ATOMIC | FI_WRITE)
#define GNIX_ATOMIC_READ_FLAGS_DEF	(FI_ATOMIC | FI_READ)

static int gnix_ep_atomic_valid(struct fid_ep *ep,
				enum fi_datatype datatype,
				enum fi_op op, size_t *count)
{
	if (count)
		*count = 1;

	return _gnix_atomic_cmd(datatype, op, GNIX_FAB_RQ_AMO) >= 0 ?
			0 : -FI_ENOENT;
}

static int gnix_ep_fetch_atomic_valid(struct fid_ep *ep,
				      enum fi_datatype datatype,
				      enum fi_op op, size_t *count)
{
	if (count)
		*count = 1;

	return _gnix_atomic_cmd(datatype, op, GNIX_FAB_RQ_FAMO) >= 0 ?
			0 : -FI_ENOENT;
}

static int gnix_ep_cmp_atomic_valid(struct fid_ep *ep,
				    enum fi_datatype datatype,
				    enum fi_op op, size_t *count)
{
	if (count)
		*count = 1;

	return _gnix_atomic_cmd(datatype, op, GNIX_FAB_RQ_CAMO) >= 0 ?
			0 : -FI_ENOENT;
}

ssize_t gnix_ep_atomic_write(struct fid_ep *ep,
			     const void *buf, size_t count, void *desc,
			     fi_addr_t dest_addr,
			     uint64_t addr, uint64_t key,
			     enum fi_datatype datatype, enum fi_op op, void *context)
{
	struct gnix_fid_ep *gnix_ep;
	struct fi_msg_atomic msg;
	struct fi_ioc msg_iov;
	struct fi_rma_ioc rma_iov;
	uint64_t flags;

	if (gnix_ep_atomic_valid(ep, datatype, op, NULL))
		return -FI_ENOENT;

	if (!ep)
		return -FI_EINVAL;

	gnix_ep = container_of(ep, struct gnix_fid_ep, ep_fid);
	assert((gnix_ep->type == FI_EP_RDM) || (gnix_ep->type == FI_EP_MSG));

	msg_iov.addr = (void *)buf;
	msg_iov.count = count;
	msg.msg_iov = &msg_iov;
	msg.desc = &desc;
	msg.iov_count = 1;
	msg.addr = dest_addr;
	rma_iov.addr = addr;
	rma_iov.count = 1;
	rma_iov.key = key;
	msg.rma_iov = &rma_iov;
	msg.datatype = datatype;
	msg.op = op;
	msg.context = context;

	flags = gnix_ep->op_flags | GNIX_ATOMIC_WRITE_FLAGS_DEF;

	return _gnix_atomic(gnix_ep, GNIX_FAB_RQ_AMO, &msg,
			    NULL, NULL, 0, NULL, NULL, 0, flags);
}

ssize_t gnix_ep_atomic_readwrite(struct fid_ep *ep,
				 const void *buf, size_t count, void *desc,
				 void *result, void *result_desc,
				 fi_addr_t dest_addr,
				 uint64_t addr, uint64_t key,
				 enum fi_datatype datatype, enum fi_op op,
				 void *context)
{
	struct gnix_fid_ep *gnix_ep;
	struct fi_msg_atomic msg;
	struct fi_ioc msg_iov;
	struct fi_rma_ioc rma_iov;
	struct fi_ioc result_iov;
	uint64_t flags;

	if (gnix_ep_fetch_atomic_valid(ep, datatype, op, NULL))
		return -FI_ENOENT;

	if (!ep)
		return -FI_EINVAL;

	gnix_ep = container_of(ep, struct gnix_fid_ep, ep_fid);
	assert((gnix_ep->type == FI_EP_RDM) || (gnix_ep->type == FI_EP_MSG));

	msg_iov.addr = (void *)buf;
	msg_iov.count = count;
	msg.msg_iov = &msg_iov;
	msg.desc = &desc;
	msg.iov_count = 1;
	msg.addr = dest_addr;
	rma_iov.addr = addr;
	rma_iov.count = 1;
	rma_iov.key = key;
	msg.rma_iov = &rma_iov;
	msg.datatype = datatype;
	msg.op = op;
	msg.context = context;
	result_iov.addr = result;
	result_iov.count = 1;

	flags = gnix_ep->op_flags | GNIX_ATOMIC_READ_FLAGS_DEF;

	return _gnix_atomic(gnix_ep, GNIX_FAB_RQ_FAMO, &msg,
			    NULL, NULL, 0,
			    &result_iov, &result_desc, 1,
			    flags);
}

ssize_t gnix_ep_atomic_compwrite(struct fid_ep *ep,
				 const void *buf, size_t count, void *desc,
				 const void *compare, void *compare_desc,
				 void *result, void *result_desc,
				 fi_addr_t dest_addr,
				 uint64_t addr, uint64_t key,
				 enum fi_datatype datatype, enum fi_op op,
				 void *context)
{
	struct gnix_fid_ep *gnix_ep;
	struct fi_msg_atomic msg;
	struct fi_ioc msg_iov;
	struct fi_rma_ioc rma_iov;
	struct fi_ioc result_iov;
	struct fi_ioc compare_iov;
	uint64_t flags;

	if (gnix_ep_cmp_atomic_valid(ep, datatype, op, NULL))
		return -FI_ENOENT;

	if (!ep)
		return -FI_EINVAL;

	gnix_ep = container_of(ep, struct gnix_fid_ep, ep_fid);
	assert((gnix_ep->type == FI_EP_RDM) || (gnix_ep->type == FI_EP_MSG));

	msg_iov.addr = (void *)buf;
	msg_iov.count = count;
	msg.msg_iov = &msg_iov;
	msg.desc = &desc;
	msg.iov_count = 1;
	msg.addr = dest_addr;
	rma_iov.addr = addr;
	rma_iov.count = 1;
	rma_iov.key = key;
	msg.rma_iov = &rma_iov;
	msg.datatype = datatype;
	msg.op = op;
	msg.context = context;
	result_iov.addr = result;
	result_iov.count = 1;
	compare_iov.addr = (void *)compare;
	compare_iov.count = 1;

	flags = gnix_ep->op_flags | GNIX_ATOMIC_READ_FLAGS_DEF;

	return _gnix_atomic(gnix_ep, GNIX_FAB_RQ_CAMO, &msg,
			    &compare_iov, &compare_desc, 1,
			    &result_iov, &result_desc, 1,
			    flags);
}

/*******************************************************************************
 * Base EP API function implementations.
 ******************************************************************************/


static int gnix_ep_control(fid_t fid, int command, void *arg)
{
	int ret = FI_SUCCESS;
	struct gnix_fid_ep *ep;

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	ep = container_of(fid, struct gnix_fid_ep, ep_fid);

	switch (command) {
	/*
	 * for FI_EP_RDM, enable the cm_nic associated
	 * with this ep.
	 */
	case FI_ENABLE:
		if (ep->type == FI_EP_RDM) {
			ret = _gnix_vc_cm_init(ep->cm_nic);
			if (ret != FI_SUCCESS) {
				GNIX_WARN(FI_LOG_EP_CTRL,
				     "_gnix_vc_cm_nic_init call returned %d\n",
					ret);
				goto err;
			}
			ret = _gnix_cm_nic_enable(ep->cm_nic);
			if (ret != FI_SUCCESS) {
				GNIX_WARN(FI_LOG_EP_CTRL,
				     "_gnix_cm_nic_enable call returned %d\n",
					ret);
				goto err;
			}
		}
		break;

	case FI_GETFIDFLAG:
	case FI_SETFIDFLAG:
	case FI_ALIAS:
	default:
		return -FI_ENOSYS;
	}

err:
	return ret;
}


static void __ep_destruct(void *obj)
{
	int __attribute__((unused)) ret;
	struct gnix_fid_domain *domain;
	struct gnix_nic *nic;
	struct gnix_fid_av *av;
	struct gnix_cm_nic *cm_nic;
	gnix_ht_key_t *key_ptr;
	struct gnix_fid_ep *ep = (struct gnix_fid_ep *) obj;

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	if (ep->send_cq) {
		_gnix_cq_poll_nic_rem(ep->send_cq, ep->nic);
		_gnix_ref_put(ep->send_cq);
	}

	if (ep->recv_cq) {
		_gnix_cq_poll_nic_rem(ep->recv_cq, ep->nic);
		_gnix_ref_put(ep->recv_cq);
	}

	if (ep->send_cntr) {
		_gnix_cntr_poll_nic_rem(ep->send_cntr, ep->nic);
		_gnix_ref_put(ep->send_cntr);
	}

	if (ep->recv_cntr) {
		_gnix_cntr_poll_nic_rem(ep->recv_cntr, ep->nic);
		_gnix_ref_put(ep->recv_cntr);
	}

	if (ep->read_cntr) {
		_gnix_cntr_poll_nic_rem(ep->read_cntr, ep->nic);
		_gnix_ref_put(ep->read_cntr);
	}

	if (ep->write_cntr) {
		_gnix_cntr_poll_nic_rem(ep->write_cntr, ep->nic);
		_gnix_ref_put(ep->write_cntr);
	}

	domain = ep->domain;
	assert(domain != NULL);
	_gnix_ref_put(domain);

	cm_nic = ep->cm_nic;
	assert(cm_nic != NULL);

	nic = ep->nic;
	assert(nic != NULL);

	av = ep->av;
	if (av != NULL)
		_gnix_ref_put(av);

	/*
	 * clean up any vc hash table or vector,
	 * remove entry from addr_to_ep ht.
	 */

	if (ep->type == FI_EP_RDM) {

		key_ptr = (gnix_ht_key_t *)&ep->my_name.gnix_addr;
		ret =  _gnix_ht_remove(ep->cm_nic->addr_to_ep_ht,
				       *key_ptr);
		if (ep->vc_ht != NULL) {
			ret = _gnix_ht_destroy(ep->vc_ht);
			if (ret == FI_SUCCESS) {
				free(ep->vc_ht);
				ep->vc_ht = NULL;
			} else {
				GNIX_WARN(FI_LOG_EP_CTRL,
					"_gnix_ht_destroy returned %s\n",
					  fi_strerror(-ret));
			}
		}
	}

	/* There is no other choice here, we need to assert if we can't free */
	ret = _gnix_nic_free(nic);
	assert(ret == FI_SUCCESS);

	ep->nic = NULL;

	/* This currently always returns FI_SUCCESS */
	ret = _gnix_cm_nic_free(cm_nic);
	assert(ret == FI_SUCCESS);

	/*
	 * Free fab_reqs
	 */
	if (atomic_get(&ep->active_fab_reqs) != 0) {
		/* Should we just assert here? */
		GNIX_WARN(FI_LOG_EP_CTRL,
			  "Active requests while closing an endpoint.");
	}
	__fr_freelist_destroy(ep);

	free(ep);
}

static int gnix_ep_close(fid_t fid)
{
	int ret = FI_SUCCESS;
	struct gnix_fid_ep *ep;
	int references_held;

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	ep = container_of(fid, struct gnix_fid_ep, ep_fid.fid);

	references_held = _gnix_ref_put(ep);
	if (references_held)
		GNIX_INFO(FI_LOG_EP_CTRL, "failed to fully close ep due "
			  "to lingering references. references=%i ep=%p\n",
			  references_held, ep);

	return ret;
}

static int gnix_ep_bind(fid_t fid, struct fid *bfid, uint64_t flags)
{
	int ret = FI_SUCCESS;
	struct gnix_fid_ep  *ep;
	struct gnix_fid_av  *av;
	struct gnix_fid_cq  *cq;
	struct gnix_fid_cntr *cntr;

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	ep = container_of(fid, struct gnix_fid_ep, ep_fid.fid);

	if (!bfid)
		return -FI_EINVAL;

	switch (bfid->fclass) {
	case FI_CLASS_EQ:
		ret = -FI_ENOSYS;
		goto err;
		break;
	case FI_CLASS_CQ:
		cq = container_of(bfid, struct gnix_fid_cq, cq_fid.fid);
		if (ep->domain != cq->domain) {
			ret = -FI_EINVAL;
			break;
		}
		if (flags & FI_TRANSMIT) {
			/* don't allow rebinding */
			if (ep->send_cq) {
				ret = -FI_EINVAL;
				break;
			}

			ep->send_cq = cq;
			if (flags & FI_SELECTIVE_COMPLETION) {
				ep->send_selective_completion = 1;
			}

			_gnix_cq_poll_nic_add(cq, ep->nic);
			_gnix_ref_get(cq);
		}
		if (flags & FI_RECV) {
			/* don't allow rebinding */
			if (ep->recv_cq) {
				ret = -FI_EINVAL;
				break;
			}

			ep->recv_cq = cq;
			if (flags & FI_SELECTIVE_COMPLETION) {
				ep->recv_selective_completion = 1;
			}

			_gnix_cq_poll_nic_add(cq, ep->nic);
			_gnix_ref_get(cq);
		}
		break;
	case FI_CLASS_AV:
		av = container_of(bfid, struct gnix_fid_av, av_fid.fid);
		if (ep->domain != av->domain) {
			ret = -FI_EINVAL;
			break;
		}
		ep->av = av;
		_gnix_ref_get(ep->av);
		break;
	case FI_CLASS_CNTR:
		cntr = container_of(bfid, struct gnix_fid_cntr, cntr_fid.fid);
		if (ep->domain != cntr->domain) {
			ret = -FI_EINVAL;
			break;
		}

		if (flags & FI_SEND) {
			/* don't allow rebinding */
			if (ep->send_cntr) {
				ret = -FI_EINVAL;
				break;
			}
			ep->send_cntr = cntr;
			_gnix_cntr_poll_nic_add(cntr, ep->nic);
			_gnix_ref_get(cntr);
		}

		if (flags & FI_RECV) {
			/* don't allow rebinding */
			if (ep->recv_cntr) {
				ret = -FI_EINVAL;
				break;
			}
			ep->recv_cntr = cntr;
			_gnix_cntr_poll_nic_add(cntr, ep->nic);
			_gnix_ref_get(cntr);
		}

		if (flags & FI_READ) {
			/* don't allow rebinding */
			if (ep->read_cntr) {
				ret = -FI_EINVAL;
				break;
			}
			ep->read_cntr = cntr;
			_gnix_cntr_poll_nic_add(cntr, ep->nic);
			_gnix_ref_get(cntr);
		}

		if (flags & FI_WRITE) {
			/* don't allow rebinding */
			if (ep->write_cntr) {
				ret = -FI_EINVAL;
				break;
			}
			ep->write_cntr = cntr;
			_gnix_cntr_poll_nic_add(cntr, ep->nic);
			_gnix_ref_get(cntr);
		}

		/* TODO: don't support this option right now,
		   never should have gotten here since gni provider
		   doesn't claim cap FI_RMA_EVENT.  This
		   option could be supported via Aries atomics
		   or using SMSG cntrl messages */

		if ((flags & FI_REMOTE_WRITE) ||
			(flags & FI_REMOTE_READ))
			ret = -FI_ENOSYS;
		break;

	case FI_CLASS_MR:/*TODO: got to figure this one out */
	default:
		ret = -FI_ENOSYS;
		break;
	}

err:
	return ret;
}

static void __gnix_vc_destroy_ht_entry(void *val)
{
	struct gnix_vc *vc = (struct gnix_vc *) val;

	_gnix_vc_destroy(vc);
}

int gnix_ep_open(struct fid_domain *domain, struct fi_info *info,
		 struct fid_ep **ep, void *context)
{
	int ret = FI_SUCCESS;
	int tsret = FI_SUCCESS;
	uint32_t cdm_id;
	struct gnix_fid_domain *domain_priv;
	struct gnix_fid_ep *ep_priv;
	gnix_hashtable_attr_t gnix_ht_attr;
	gnix_ht_key_t *key_ptr;
	struct gnix_nic_attr *attr = NULL;
	struct gnix_nic_attr ded_nic_attr = {0};
	struct gnix_tag_storage_attr untagged_attr = {
			.type = GNIX_TAG_LIST,
			.use_src_addr_matching = 1,
	};

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	if ((domain == NULL) || (info == NULL) || (ep == NULL) ||
	    (info->ep_attr == NULL))
		return -FI_EINVAL;

	if (info->ep_attr->type != FI_EP_RDM)
		return -FI_ENOSYS;

	domain_priv = container_of(domain, struct gnix_fid_domain, domain_fid);

	ep_priv = calloc(1, sizeof *ep_priv);
	if (!ep_priv)
		return -FI_ENOMEM;

	/* init untagged storages */
	tsret = _gnix_posted_tag_storage_init(
			&ep_priv->posted_recv_queue, &untagged_attr);
	if (tsret)
		return tsret;

	tsret = _gnix_unexpected_tag_storage_init(
			&ep_priv->unexp_recv_queue, &untagged_attr);
	if (tsret)
		return tsret;

	/* init tagged storages */
	tsret = _gnix_posted_tag_storage_init(
			&ep_priv->tagged_posted_recv_queue, NULL);
	if (tsret)
		return tsret;

	tsret = _gnix_unexpected_tag_storage_init(
			&ep_priv->tagged_unexp_recv_queue, NULL);
	if (tsret)
		return tsret;

	ep_priv->ep_fid.fid.fclass = FI_CLASS_EP;
	ep_priv->ep_fid.fid.context = context;

	ep_priv->ep_fid.fid.ops = &gnix_ep_fi_ops;
	ep_priv->ep_fid.ops = &gnix_ep_ops;
	ep_priv->domain = domain_priv;
	ep_priv->type = info->ep_attr->type;

	atomic_initialize(&ep_priv->active_fab_reqs, 0);
	_gnix_ref_init(&ep_priv->ref_cnt, 1, __ep_destruct);

	fastlock_init(&ep_priv->recv_comp_lock);
	fastlock_init(&ep_priv->recv_queue_lock);
	fastlock_init(&ep_priv->tagged_queue_lock);
	slist_init(&ep_priv->pending_recv_comp_queue);

	ep_priv->caps = info->caps & GNIX_EP_RDM_CAPS;

	if (info->tx_attr)
		ep_priv->op_flags = info->tx_attr->op_flags;
	if (info->rx_attr)
		ep_priv->op_flags |= info->rx_attr->op_flags;
	ep_priv->op_flags &= GNIX_EP_OP_FLAGS;

	ret = __fr_freelist_init(ep_priv);
	if (ret != FI_SUCCESS) {
		GNIX_ERR(FI_LOG_EP_CTRL,
			 "Error allocating gnix_fab_req freelist (%s)",
			 fi_strerror(-ret));
		goto err1;
	}

	ep_priv->ep_fid.msg = &gnix_ep_msg_ops;
	ep_priv->ep_fid.rma = &gnix_ep_rma_ops;
	ep_priv->ep_fid.tagged = &gnix_ep_tagged_ops;
	ep_priv->ep_fid.atomic = &gnix_ep_atomic_ops;

	ep_priv->ep_fid.cm = &gnix_cm_ops;

	if (ep_priv->type == FI_EP_RDM) {
		ret = _gnix_cm_nic_alloc(domain_priv,
					 info,
					 &ep_priv->cm_nic);
		if (ret != FI_SUCCESS)
			goto err;

		/*
		 * if we had to dedicate a cm_nic to this ep
		 * because its bound (i.e. specifying the cdm id)
		 * then we had to allocate a GNI cdm/nic handle for
		 * it and thus the underlying CQ hw resources.
		 * Try to reduce CQ hw consumption by passing in
		 * the cm_nic's cdm/nic handles.  In this way,
		 * if a gnix_nic is allocated, it can reuse the
		 * gni cdm/nic hndls used for the previously
		 * allocated gnix_cm_nic.
		 */
		if (info->src_addr != NULL) {
			ded_nic_attr.gni_cdm_hndl =
				ep_priv->cm_nic->gni_cdm_hndl;
			ded_nic_attr.gni_nic_hndl =
				ep_priv->cm_nic->gni_nic_hndl;
			attr = &ded_nic_attr;
			ep_priv->my_name = ep_priv->cm_nic->my_name;
		} else {
			ep_priv->my_name.gnix_addr.device_addr =
				ep_priv->cm_nic->my_name.gnix_addr.device_addr;
			ep_priv->my_name.cm_nic_cdm_id =
				ep_priv->cm_nic->my_name.gnix_addr.cdm_id;
			ret = _gnix_get_new_cdm_id(domain_priv, &cdm_id);
			if (ret != FI_SUCCESS) {
				GNIX_WARN(FI_LOG_EP_CTRL,
					    "gnix_get_new_cdm_id call returned %s\n",
					     fi_strerror(-ret));
				goto err;
			}
			ep_priv->my_name.gnix_addr.cdm_id = cdm_id;
		}

		key_ptr = (gnix_ht_key_t *)&ep_priv->my_name.gnix_addr;
		ret = _gnix_ht_insert(ep_priv->cm_nic->addr_to_ep_ht,
					*key_ptr,
					ep_priv);
		if ((ret != FI_SUCCESS) && (ret != -FI_ENOSPC)) {
			GNIX_WARN(FI_LOG_EP_CTRL,
				  "__gnix_ht_insert returned %d\n",
				  ret);
		}

		gnix_ht_attr.ht_initial_size = domain_priv->params.ct_init_size;
		gnix_ht_attr.ht_maximum_size = domain_priv->params.ct_max_size;
		gnix_ht_attr.ht_increase_step = domain_priv->params.ct_step;
		gnix_ht_attr.ht_increase_type = GNIX_HT_INCREASE_MULT;
		gnix_ht_attr.ht_collision_thresh = 500;
		gnix_ht_attr.ht_hash_seed = 0xdeadbeefbeefdead;
		gnix_ht_attr.ht_internal_locking = 0;
		gnix_ht_attr.destructor = __gnix_vc_destroy_ht_entry;

		ep_priv->vc_ht = calloc(1, sizeof(struct gnix_hashtable));
		if (ep_priv->vc_ht == NULL)
			goto err;
		ret = _gnix_ht_init(ep_priv->vc_ht, &gnix_ht_attr);
		if (ret != FI_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_CTRL,
				    "gnix_ht_init call returned %d\n",
				     ret);
			goto err;
		}
		fastlock_init(&ep_priv->vc_ht_lock);

	} else {
		ep_priv->cm_nic = NULL;
		ep_priv->vc = NULL;
	}

	ep_priv->progress_fn = NULL;
	ep_priv->rx_progress_fn = NULL;

	ret = gnix_nic_alloc(domain_priv, attr, &ep_priv->nic);
	if (ret != FI_SUCCESS) {
		GNIX_WARN(FI_LOG_EP_CTRL,
			    "_gnix_nic_alloc call returned %d\n",
			     ret);
		goto err;
	}

	/*
	 * if smsg callbacks not present hook them up now
	 */

	if (ep_priv->nic->smsg_callbacks == NULL)
		ep_priv->nic->smsg_callbacks = gnix_ep_smsg_callbacks;

	_gnix_ref_get(ep_priv->domain);
	*ep = &ep_priv->ep_fid;
	return ret;

err1:
	__fr_freelist_destroy(ep_priv);
err:
	if (ep_priv->vc_ht != NULL) {
		_gnix_ht_destroy(ep_priv->vc_ht); /* may not be initialized but
						     okay */
		free(ep_priv->vc_ht);
		ep_priv->vc_ht = NULL;
	}
	if (ep_priv->cm_nic != NULL)
		ret = _gnix_cm_nic_free(ep_priv->cm_nic);
	free(ep_priv);
	return ret;

}

static int __match_context(struct slist_entry *item, const void *arg)
{
	struct gnix_fab_req *req;

	req = container_of(item, struct gnix_fab_req, slist);

	return req->user_context == arg;
}

static inline struct gnix_fab_req *__find_tx_req(
		struct gnix_fid_ep *ep,
		void *context)
{
	struct gnix_fab_req *req = NULL;
	struct slist_entry *entry;
	struct gnix_vc *vc;
	GNIX_HASHTABLE_ITERATOR(ep->vc_ht, iter);

	GNIX_DEBUG(FI_LOG_EP_CTRL, "searching VCs for the correct context to"
			" cancel, context=%p", context);

	fastlock_acquire(&ep->vc_ht_lock);
	while ((vc = _gnix_ht_iterator_next(&iter))) {
		fastlock_acquire(&vc->tx_queue_lock);
		entry = slist_remove_first_match(&vc->tx_queue,
				__match_context, context);
		fastlock_release(&vc->tx_queue_lock);
		if (entry) {
			req = container_of(entry, struct gnix_fab_req, slist);
			break;
		}
	}
	fastlock_release(&ep->vc_ht_lock);

	return req;
}

static inline struct gnix_fab_req *__find_rx_req(
		struct gnix_fid_ep *ep,
		void *context)
{
	struct gnix_fab_req *req = NULL;

	fastlock_acquire(&ep->recv_queue_lock);
	req = _gnix_remove_req_by_context(&ep->posted_recv_queue, context);
	fastlock_release(&ep->recv_queue_lock);

	if (req)
		return req;

	fastlock_acquire(&ep->tagged_queue_lock);
	req = _gnix_remove_req_by_context(&ep->tagged_posted_recv_queue,
			context);
	fastlock_release(&ep->tagged_queue_lock);

	return req;
}

static ssize_t gnix_ep_cancel(fid_t fid, void *context)
{
	int ret = FI_SUCCESS;
	struct gnix_fid_ep *ep;
	struct gnix_fab_req *req;
	struct gnix_fid_cq *err_cq = NULL;
	struct gnix_fid_cntr *err_cntr = NULL;
	void *addr;
	uint64_t tag, flags;
	size_t len;
	int is_send = 0;

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	ep = container_of(fid, struct gnix_fid_ep, ep_fid.fid);

	if (!ep->domain)
		return -FI_EDOMAIN;

	/* without context, we will have to find a request that matches
	 * a recv or send request. Try the send requests first.
	 */
	GNIX_INFO(FI_LOG_EP_CTRL, "looking for event to cancel\n");

	req = __find_tx_req(ep, context);
	if (!req) {
		req = __find_rx_req(ep, context);
		if (req) {
			err_cq = ep->recv_cq;
			err_cntr = ep->recv_cntr;
		}
	} else {
		is_send = 1;
		err_cq = ep->send_cq;
		err_cntr = ep->send_cntr;
	}
	GNIX_INFO(FI_LOG_EP_CTRL, "finished searching\n");

	if (!req)
		return -FI_ENOENT;

	if (err_cq) {
		/* add canceled event */
		if (!(req->type == GNIX_FAB_RQ_RDMA_READ ||
				req->type == GNIX_FAB_RQ_RDMA_WRITE)) {
			if (!is_send) {
				addr = (void *) req->msg.recv_addr;
				len = req->msg.recv_len;
			} else {
				addr = (void *) req->msg.send_addr;
				len = req->msg.send_len;
			}
			tag = req->msg.tag;
		} else {
			/* rma information */
			addr = (void *) req->rma.loc_addr;
			len = req->rma.len;
			tag = 0;
		}
		flags = req->flags;

		_gnix_cq_add_error(err_cq, context, flags, len, addr, 0 /* data */,
				tag, len, FI_ECANCELED, FI_ECANCELED, 0);

	}

	if (err_cntr) {
		/* signal increase in cntr errs */
		_gnix_cntr_inc_err(err_cntr);
	}

	_gnix_fr_free(ep, req);

	return ret;
}

ssize_t gnix_cancel(fid_t fid, void *context)
{
	ssize_t ret;

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	switch (fid->fclass) {
	case FI_CLASS_EP:
		ret = gnix_ep_cancel(fid, context);
		break;

	/* not supported yet */
	case FI_CLASS_RX_CTX:
	case FI_CLASS_SRX_CTX:
	case FI_CLASS_TX_CTX:
	case FI_CLASS_STX_CTX:
		return -FI_ENOENT;

	default:
		GNIX_ERR(FI_LOG_EP_CTRL, "Invalid fid type\n");
		return -FI_EINVAL;
	}

	return ret;
}

/*******************************************************************************
 * FI_OPS_* data structures.
 ******************************************************************************/

static struct fi_ops gnix_ep_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = gnix_ep_close,
	.bind = gnix_ep_bind,
	.control = gnix_ep_control,
};

static struct fi_ops_ep gnix_ep_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = gnix_cancel,
	.getopt = fi_no_getopt,
	.setopt = fi_no_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

static struct fi_ops_msg gnix_ep_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = gnix_ep_recv,
	.recvv = gnix_ep_recvv,
	.recvmsg = gnix_ep_recvmsg,
	.send = gnix_ep_send,
	.sendv = gnix_ep_sendv,
	.sendmsg = gnix_ep_sendmsg,
	.inject = gnix_ep_msg_inject,
	.senddata = gnix_ep_senddata,
	.injectdata = gnix_ep_msg_injectdata,
};

static struct fi_ops_rma gnix_ep_rma_ops = {
	.size = sizeof(struct fi_ops_rma),
	.read = gnix_ep_read,
	.readv = gnix_ep_readv,
	.readmsg = gnix_ep_readmsg,
	.write = gnix_ep_write,
	.writev = gnix_ep_writev,
	.writemsg = gnix_ep_writemsg,
	.inject = gnix_ep_rma_inject,
	.writedata = gnix_ep_writedata,
	.injectdata = gnix_ep_rma_injectdata,
};

struct fi_ops_tagged gnix_ep_tagged_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = gnix_ep_trecv,
	.recvv = gnix_ep_trecvv,
	.recvmsg = gnix_ep_trecvmsg,
	.send = gnix_ep_tsend,
	.sendv = gnix_ep_tsendv,
	.sendmsg = gnix_ep_tsendmsg,
	.inject = gnix_ep_tinject,
	.senddata = fi_no_tagged_senddata,
	.senddata = gnix_ep_tsenddata,
	.injectdata = fi_no_tagged_injectdata,
};

struct fi_ops_atomic gnix_ep_atomic_ops = {
	.size = sizeof(struct fi_ops_atomic),
	.write = gnix_ep_atomic_write,
//	.writev = sock_ep_atomic_writev,
//	.writemsg = sock_ep_atomic_writemsg,
//	.inject = sock_ep_atomic_inject,
	.readwrite = gnix_ep_atomic_readwrite,
//	.readwritev = sock_ep_atomic_readwritev,
//	.readwritemsg = sock_ep_atomic_readwritemsg,
	.compwrite = gnix_ep_atomic_compwrite,
//	.compwritev = sock_ep_atomic_compwritev,
//	.compwritemsg = sock_ep_atomic_compwritemsg,
	.writevalid = gnix_ep_atomic_valid,
	.readwritevalid = gnix_ep_fetch_atomic_valid,
	.compwritevalid = gnix_ep_cmp_atomic_valid,
};

