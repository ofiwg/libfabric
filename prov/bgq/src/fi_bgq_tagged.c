#include "rdma/bgq/fi_bgq.h"

#include <fi_enosys.h>

ssize_t fi_bgq_trecvmsg(struct fid_ep *ep,
		const struct fi_msg_tagged *msg, uint64_t flags)
{
	struct fi_bgq_ep * bgq_ep = container_of(ep, struct fi_bgq_ep, ep_fid);
	const enum fi_av_type av_type = bgq_ep->av_type;
	const enum fi_progress progress = bgq_ep->domain->data_progress;
	const enum fi_threading threading = bgq_ep->domain->threading;
	const int lock_required =
		(threading == FI_THREAD_FID) ||
		(threading == FI_THREAD_UNSPEC) ||
		(threading == FI_THREAD_SAFE);

	return fi_bgq_trecvmsg_generic(ep, msg, flags, lock_required, av_type, progress);
}



ssize_t fi_bgq_tsendmsg(struct fid_ep *ep,
		const struct fi_msg_tagged *msg, uint64_t flags)
{
	const size_t niov = msg->iov_count;

	if (niov > 32) {

		/* ---------------------------------------------------------
		 * a single torus packet payload can only transfer 32
		 * 'struct fi_bgq_mu_iov' elements - this is the current
		 * limit for non-contiguous rendezvous operations
		 *
		 * TODO - support >32 iov elements?
		 * --------------------------------------------------------- */
		return -FI_EINVAL;

	} else {

		struct fi_bgq_ep * bgq_ep = container_of(ep, struct fi_bgq_ep, ep_fid);
		const enum fi_threading threading = bgq_ep->threading;
		const enum fi_av_type av_type = bgq_ep->av_type;

		return fi_bgq_send_generic_flags(ep, msg->msg_iov, niov,
			msg->desc, msg->addr, msg->tag, msg->context,
			(threading != FI_THREAD_ENDPOINT && threading != FI_THREAD_DOMAIN),
			av_type,
			0 /* is_msg */,
			0 /* is_contiguous */,
			1 /* override flags */,
			flags);
	}
}

ssize_t fi_bgq_tsenddata(struct fid_ep *ep, const void *buf, size_t len,
		void *desc, uint64_t data, fi_addr_t dest_addr, uint64_t tag,
		void *context)
{
	errno = FI_ENOSYS;
	return -errno;
}

/* "FI_BGQ_TAGGED_SPECIALIZED_FUNC(0, FI_AV_MAP, FI_PROGRESS_MANUAL)" is already declared via FABRIC_DIRECT */
FI_BGQ_TAGGED_SPECIALIZED_FUNC(1, FI_AV_MAP, FI_PROGRESS_MANUAL)
FI_BGQ_TAGGED_SPECIALIZED_FUNC(0, FI_AV_TABLE, FI_PROGRESS_MANUAL)
FI_BGQ_TAGGED_SPECIALIZED_FUNC(1, FI_AV_TABLE, FI_PROGRESS_MANUAL)
FI_BGQ_TAGGED_SPECIALIZED_FUNC(0, FI_AV_MAP, FI_PROGRESS_AUTO)
FI_BGQ_TAGGED_SPECIALIZED_FUNC(1, FI_AV_MAP, FI_PROGRESS_AUTO)
FI_BGQ_TAGGED_SPECIALIZED_FUNC(0, FI_AV_TABLE, FI_PROGRESS_AUTO)
FI_BGQ_TAGGED_SPECIALIZED_FUNC(1, FI_AV_TABLE, FI_PROGRESS_AUTO)

#define FI_BGQ_TAGGED_OPS_STRUCT_NAME(LOCK, AV, PROGRESS)		\
	fi_bgq_ops_tagged_ ## LOCK ## _ ## AV ## _ ## PROGRESS		\

#define FI_BGQ_TAGGED_OPS_STRUCT(LOCK, AV, PROGRESS)			\
static struct fi_ops_tagged						\
	FI_BGQ_TAGGED_OPS_STRUCT_NAME(LOCK, AV, PROGRESS) = {		\
	.size		= sizeof(struct fi_ops_tagged),			\
	.recv		=						\
		FI_BGQ_TAGGED_SPECIALIZED_FUNC_NAME(trecv,		\
			LOCK, AV, PROGRESS),				\
	.recvv		= fi_no_tagged_recvv,				\
	.recvmsg	=						\
		FI_BGQ_TAGGED_SPECIALIZED_FUNC_NAME(trecvmsg,		\
			LOCK, AV, PROGRESS),				\
	.send		=						\
		FI_BGQ_TAGGED_SPECIALIZED_FUNC_NAME(tsend,		\
			LOCK, AV, PROGRESS),				\
	.sendv		= fi_no_tagged_sendv,				\
	.sendmsg	= fi_bgq_tsendmsg,				\
	.inject =							\
		FI_BGQ_TAGGED_SPECIALIZED_FUNC_NAME(tinject,		\
			LOCK, AV, PROGRESS),				\
	.senddata	= fi_no_tagged_senddata,			\
	.injectdata	= fi_no_tagged_injectdata			\
}

FI_BGQ_TAGGED_OPS_STRUCT(0, FI_AV_MAP, FI_PROGRESS_MANUAL);
FI_BGQ_TAGGED_OPS_STRUCT(1, FI_AV_MAP, FI_PROGRESS_MANUAL);
FI_BGQ_TAGGED_OPS_STRUCT(0, FI_AV_TABLE, FI_PROGRESS_MANUAL);
FI_BGQ_TAGGED_OPS_STRUCT(1, FI_AV_TABLE, FI_PROGRESS_MANUAL);
FI_BGQ_TAGGED_OPS_STRUCT(0, FI_AV_MAP, FI_PROGRESS_AUTO);
FI_BGQ_TAGGED_OPS_STRUCT(1, FI_AV_MAP, FI_PROGRESS_AUTO);
FI_BGQ_TAGGED_OPS_STRUCT(0, FI_AV_TABLE, FI_PROGRESS_AUTO);
FI_BGQ_TAGGED_OPS_STRUCT(1, FI_AV_TABLE, FI_PROGRESS_AUTO);





ssize_t fi_bgq_tsearch(struct fid_ep *ep, uint64_t *tag,
		uint64_t ignore, uint64_t flags,
		fi_addr_t *src_addr, size_t *len, void *context)
{
	errno = FI_ENOSYS;
	return -errno;
}

static struct fi_ops_tagged fi_bgq_no_tagged_ops = {
        .size           = sizeof(struct fi_ops_tagged),
        .recv           = fi_no_tagged_recv,
        .recvv          = fi_no_tagged_recvv,
        .recvmsg        = fi_no_tagged_recvmsg,
        .send           = fi_no_tagged_send,
        .sendv          = fi_no_tagged_sendv,
        .sendmsg        = fi_no_tagged_sendmsg,
        .inject         = fi_no_tagged_inject,
        .senddata       = fi_no_tagged_senddata,
        .injectdata     = fi_no_tagged_injectdata
};

int fi_bgq_init_tagged_ops(struct fi_bgq_ep *bgq_ep, struct fi_info *info)
{
        if (!info || !bgq_ep)
                goto err;

        if (info->caps & FI_TAGGED ||
                        (info->tx_attr &&
                         (info->tx_attr->caps & FI_TAGGED))) {
        }

        return 0;

err:
        errno = FI_EINVAL;
        return -errno;
}


int fi_bgq_enable_tagged_ops(struct fi_bgq_ep *bgq_ep)
{
        int lock_required;
        enum fi_av_type av_type;
	enum fi_progress progress;

        if (!bgq_ep || !bgq_ep->domain)
                goto err;

        if (!(bgq_ep->tx.caps & FI_TAGGED)) {
                /* Tagged ops not enabled on this endpoint */
                bgq_ep->ep_fid.tagged =
                        &fi_bgq_no_tagged_ops;
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
                bgq_ep->ep_fid.tagged =
                        &FI_BGQ_TAGGED_OPS_STRUCT_NAME(0, FI_AV_MAP, FI_PROGRESS_MANUAL);
        } else if (lock_required == 1 &&
			progress == FI_PROGRESS_MANUAL &&
                        av_type == FI_AV_MAP) {
                bgq_ep->ep_fid.tagged =
                        &FI_BGQ_TAGGED_OPS_STRUCT_NAME(1, FI_AV_MAP, FI_PROGRESS_MANUAL);
        } else if (lock_required == 0 &&
			progress == FI_PROGRESS_MANUAL &&
                        av_type == FI_AV_TABLE) {
                bgq_ep->ep_fid.tagged =
                        &FI_BGQ_TAGGED_OPS_STRUCT_NAME(0, FI_AV_TABLE, FI_PROGRESS_MANUAL);
        } else if (lock_required == 1 &&
			progress == FI_PROGRESS_MANUAL &&
                        av_type == FI_AV_TABLE) {
                bgq_ep->ep_fid.tagged =
                        &FI_BGQ_TAGGED_OPS_STRUCT_NAME(1, FI_AV_TABLE, FI_PROGRESS_MANUAL);
        } else if (lock_required == 0 &&
			progress == FI_PROGRESS_AUTO &&
                        av_type == FI_AV_MAP) {
                bgq_ep->ep_fid.tagged =
                        &FI_BGQ_TAGGED_OPS_STRUCT_NAME(0, FI_AV_MAP, FI_PROGRESS_AUTO);
        } else if (lock_required == 1 &&
			progress == FI_PROGRESS_AUTO &&
                        av_type == FI_AV_MAP) {
                bgq_ep->ep_fid.tagged =
                        &FI_BGQ_TAGGED_OPS_STRUCT_NAME(1, FI_AV_MAP, FI_PROGRESS_AUTO);
        } else if (lock_required == 0 &&
			progress == FI_PROGRESS_AUTO &&
                        av_type == FI_AV_TABLE) {
                bgq_ep->ep_fid.tagged =
                        &FI_BGQ_TAGGED_OPS_STRUCT_NAME(0, FI_AV_TABLE, FI_PROGRESS_AUTO);
        } else if (lock_required == 1 &&
			progress == FI_PROGRESS_AUTO &&
                        av_type == FI_AV_TABLE) {
                bgq_ep->ep_fid.tagged =
                        &FI_BGQ_TAGGED_OPS_STRUCT_NAME(1, FI_AV_TABLE, FI_PROGRESS_AUTO);
        } else {
                bgq_ep->ep_fid.tagged = &fi_bgq_no_tagged_ops;
                FI_WARN(fi_bgq_global.prov, FI_LOG_EP_DATA,
                                "Tagged ops not enabled on EP\n");
        }

	return 0;
err:
        errno = FI_EINVAL;
        return -errno;

}

int fi_bgq_finalize_tagged_ops(struct fi_bgq_ep *bgq_ep)
{
	if (!bgq_ep) {
		return 0;
	}

	return 0;
}
