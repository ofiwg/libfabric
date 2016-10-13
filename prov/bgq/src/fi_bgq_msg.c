/*
 * Copyright (C) 2016 by Argonne National Laboratory.
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
#include "rdma/bgq/fi_bgq.h"
#include <fi_enosys.h>

ssize_t fi_bgq_sendmsg(struct fid_ep *ep, const struct fi_msg *msg,
			uint64_t flags)
{
	struct fi_bgq_ep * bgq_ep = container_of(ep, struct fi_bgq_ep, ep_fid);
	const enum fi_threading threading = bgq_ep->threading;
	const enum fi_av_type av_type = bgq_ep->av_type;

	return fi_bgq_send_generic_flags(ep, msg->msg_iov, msg->iov_count,
		msg->desc, msg->addr, 0, msg->context,
		(threading != FI_THREAD_ENDPOINT && threading != FI_THREAD_DOMAIN),	/* "lock required"? */
		av_type,
		1 /* is_msg */,
		0 /* is_contiguous */,
		1 /* override the default tx flags */,
		flags);
}

ssize_t fi_bgq_sendv(struct fid_ep *ep, const struct iovec *iov,
			void **desc, size_t count, fi_addr_t dest_addr,
			void *context)
{
	struct fi_bgq_ep * bgq_ep = container_of(ep, struct fi_bgq_ep, ep_fid);
	const enum fi_threading threading = bgq_ep->threading;
	const enum fi_av_type av_type = bgq_ep->av_type;

	return fi_bgq_send_generic_flags(ep, iov, count,
		desc, dest_addr, 0, context,
		(threading != FI_THREAD_ENDPOINT && threading != FI_THREAD_DOMAIN),	/* "lock required"? */
		av_type,
		1 /* is_msg */,
		0 /* is_contiguous */,
		0 /* do not override flags */,
		0);
}


ssize_t fi_bgq_senddata(struct fid_ep *ep, const void *buf, size_t len, void *desc,
			uint64_t data, void *context)
{
	errno = FI_ENOSYS;
	return -errno;
}

/* "FI_BGQ_MSG_SPECIALIZED_FUNC(0, FI_AV_MAP, FI_PROGRESS_MANUAL)" is already declared via FABRIC_DIRECT */
FI_BGQ_MSG_SPECIALIZED_FUNC(1, FI_AV_MAP, FI_PROGRESS_MANUAL)
FI_BGQ_MSG_SPECIALIZED_FUNC(0, FI_AV_TABLE, FI_PROGRESS_MANUAL)
FI_BGQ_MSG_SPECIALIZED_FUNC(1, FI_AV_TABLE, FI_PROGRESS_MANUAL)
FI_BGQ_MSG_SPECIALIZED_FUNC(0, FI_AV_MAP, FI_PROGRESS_AUTO)
FI_BGQ_MSG_SPECIALIZED_FUNC(1, FI_AV_MAP, FI_PROGRESS_AUTO)
FI_BGQ_MSG_SPECIALIZED_FUNC(0, FI_AV_TABLE, FI_PROGRESS_AUTO)
FI_BGQ_MSG_SPECIALIZED_FUNC(1, FI_AV_TABLE, FI_PROGRESS_AUTO)

#define FI_BGQ_MSG_OPS_STRUCT_NAME(LOCK, AV, PROGRESS)		\
	fi_bgq_ops_msg_ ## LOCK ## _ ## AV ## _ ## PROGRESS	\

#define FI_BGQ_MSG_OPS_STRUCT(LOCK, AV, PROGRESS)		\
static struct fi_ops_msg					\
	FI_BGQ_MSG_OPS_STRUCT_NAME(LOCK, AV, PROGRESS) = {	\
	.size		= sizeof(struct fi_ops_msg),		\
	.recv		=					\
		FI_BGQ_MSG_SPECIALIZED_FUNC_NAME(recv,		\
			LOCK, AV, PROGRESS),			\
	.recvv		= fi_no_msg_recvv,			\
	.recvmsg	=					\
		FI_BGQ_MSG_SPECIALIZED_FUNC_NAME(recvmsg,	\
			LOCK, AV, PROGRESS),			\
	.send		=					\
		FI_BGQ_MSG_SPECIALIZED_FUNC_NAME(send,		\
			LOCK, AV, PROGRESS),			\
	.sendv		= fi_bgq_sendv,				\
	.sendmsg	= fi_bgq_sendmsg,			\
	.inject	=						\
		FI_BGQ_MSG_SPECIALIZED_FUNC_NAME(inject,	\
			LOCK, AV, PROGRESS),			\
	.senddata	= fi_no_msg_senddata,			\
	.injectdata	= fi_no_msg_injectdata			\
}

FI_BGQ_MSG_OPS_STRUCT(0, FI_AV_MAP, FI_PROGRESS_MANUAL);
FI_BGQ_MSG_OPS_STRUCT(1, FI_AV_MAP, FI_PROGRESS_MANUAL);
FI_BGQ_MSG_OPS_STRUCT(0, FI_AV_TABLE, FI_PROGRESS_MANUAL);
FI_BGQ_MSG_OPS_STRUCT(1, FI_AV_TABLE, FI_PROGRESS_MANUAL);
FI_BGQ_MSG_OPS_STRUCT(0, FI_AV_MAP, FI_PROGRESS_AUTO);
FI_BGQ_MSG_OPS_STRUCT(1, FI_AV_MAP, FI_PROGRESS_AUTO);
FI_BGQ_MSG_OPS_STRUCT(0, FI_AV_TABLE, FI_PROGRESS_AUTO);
FI_BGQ_MSG_OPS_STRUCT(1, FI_AV_TABLE, FI_PROGRESS_AUTO);

static struct fi_ops_msg fi_bgq_no_msg_ops = {
	.size		= sizeof(struct fi_ops_msg),
	.recv		= fi_no_msg_recv,
	.recvv		= fi_no_msg_recvv,
	.recvmsg	= fi_no_msg_recvmsg,
	.send		= fi_no_msg_send,
	.sendv		= fi_no_msg_sendv,
	.sendmsg	= fi_no_msg_sendmsg,
	.inject		= fi_no_msg_inject,
	.senddata	= fi_no_msg_senddata,
	.injectdata	= fi_no_msg_injectdata
};

int fi_bgq_init_msg_ops(struct fi_bgq_ep *bgq_ep, struct fi_info *info)
{
	if (!info || !bgq_ep) {
		errno = FI_EINVAL;
		goto err;
	}
	if (info->caps & FI_MSG ||
			(info->tx_attr &&
			 (info->tx_attr->caps & FI_MSG))) {

		bgq_ep->rx.min_multi_recv = sizeof(union fi_bgq_mu_packet_payload);
		bgq_ep->rx.poll.min_multi_recv = bgq_ep->rx.min_multi_recv;

	}

	return 0;

err:
	return -errno;
}

int fi_bgq_enable_msg_ops(struct fi_bgq_ep *bgq_ep)
{
	int lock_required;
	enum fi_av_type av_type;
	enum fi_progress progress;

	if (!bgq_ep || !bgq_ep->domain)
		return -FI_EINVAL;

	if (!(bgq_ep->tx.caps & FI_MSG)) {
		/* Messaging ops not enabled on this endpoint */
		bgq_ep->ep_fid.msg =
			&fi_bgq_no_msg_ops;
		return 0;
	}

	av_type = bgq_ep->av->type;
	progress = bgq_ep->domain->data_progress;

	switch (bgq_ep->domain->threading) {
	case FI_THREAD_ENDPOINT:
	case FI_THREAD_DOMAIN:
	case FI_THREAD_COMPLETION:
		lock_required = 0;
		break;
	case FI_THREAD_FID:
	case FI_THREAD_UNSPEC:
	case FI_THREAD_SAFE:
		lock_required = 1;
		break;
	default:
		return -FI_EINVAL;
	}

	if (lock_required == 0 &&
			progress == FI_PROGRESS_MANUAL &&
			av_type == FI_AV_MAP) {
		bgq_ep->ep_fid.msg =
			&FI_BGQ_MSG_OPS_STRUCT_NAME(0, FI_AV_MAP, FI_PROGRESS_MANUAL);
	} else if (lock_required == 1 &&
			progress == FI_PROGRESS_MANUAL &&
			av_type == FI_AV_MAP) {
		bgq_ep->ep_fid.msg =
			&FI_BGQ_MSG_OPS_STRUCT_NAME(1, FI_AV_MAP, FI_PROGRESS_MANUAL);
	} else if (lock_required == 0 &&
			progress == FI_PROGRESS_MANUAL &&
			av_type == FI_AV_TABLE) {
		bgq_ep->ep_fid.msg =
			&FI_BGQ_MSG_OPS_STRUCT_NAME(0, FI_AV_TABLE, FI_PROGRESS_MANUAL);
	} else if (lock_required == 1 &&
			progress == FI_PROGRESS_MANUAL &&
			av_type == FI_AV_TABLE) {
		bgq_ep->ep_fid.msg =
			&FI_BGQ_MSG_OPS_STRUCT_NAME(1, FI_AV_TABLE, FI_PROGRESS_MANUAL);
	} else if (lock_required == 0 &&
			progress == FI_PROGRESS_AUTO &&
			av_type == FI_AV_MAP) {
		bgq_ep->ep_fid.msg =
			&FI_BGQ_MSG_OPS_STRUCT_NAME(0, FI_AV_MAP, FI_PROGRESS_AUTO);
	} else if (lock_required == 1 &&
			progress == FI_PROGRESS_AUTO &&
			av_type == FI_AV_MAP) {
		bgq_ep->ep_fid.msg =
			&FI_BGQ_MSG_OPS_STRUCT_NAME(1, FI_AV_MAP, FI_PROGRESS_AUTO);
	} else if (lock_required == 0 &&
			progress == FI_PROGRESS_AUTO &&
			av_type == FI_AV_TABLE) {
		bgq_ep->ep_fid.msg =
			&FI_BGQ_MSG_OPS_STRUCT_NAME(0, FI_AV_TABLE, FI_PROGRESS_AUTO);
	} else if (lock_required == 1 &&
			progress == FI_PROGRESS_AUTO &&
			av_type == FI_AV_TABLE) {
		bgq_ep->ep_fid.msg =
			&FI_BGQ_MSG_OPS_STRUCT_NAME(1, FI_AV_TABLE, FI_PROGRESS_AUTO);
	} else {
		bgq_ep->ep_fid.msg = &fi_bgq_no_msg_ops;
		FI_WARN(fi_bgq_global.prov, FI_LOG_EP_DATA,
				"Msg ops not enabled on EP\n");
	}

	return 0;
}

int fi_bgq_finalize_msg_ops(struct fi_bgq_ep *bgq_ep)
{
	if (!bgq_ep) {
		return 0;
	}

	return 0;
}
