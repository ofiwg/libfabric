#include "rdma/bgq/fi_bgq.h"
#include <fi_enosys.h>
#include <errno.h>

FI_BGQ_RMA_SPECIALIZED_FUNC(0, FI_AV_MAP, FI_MR_BASIC)
FI_BGQ_RMA_SPECIALIZED_FUNC(1, FI_AV_MAP, FI_MR_BASIC)
FI_BGQ_RMA_SPECIALIZED_FUNC(1, FI_AV_MAP, FI_MR_SCALABLE)

FI_BGQ_RMA_SPECIALIZED_FUNC(0, FI_AV_TABLE, FI_MR_SCALABLE)
FI_BGQ_RMA_SPECIALIZED_FUNC(0, FI_AV_TABLE, FI_MR_BASIC)
FI_BGQ_RMA_SPECIALIZED_FUNC(1, FI_AV_TABLE, FI_MR_BASIC)
FI_BGQ_RMA_SPECIALIZED_FUNC(1, FI_AV_TABLE, FI_MR_SCALABLE)

#define FI_BGQ_RMA_OPS_STRUCT_NAME(LOCK, AV, MR)		\
	fi_bgq_ops_rma_ ## LOCK ## _ ## AV ## _ ## MR		\

#define FI_BGQ_RMA_OPS_STRUCT(LOCK, AV, MR)			\
static struct fi_ops_rma					\
	FI_BGQ_RMA_OPS_STRUCT_NAME(LOCK, AV, MR) = {		\
	.size	= sizeof(struct fi_ops_rma),			\
	.read	= FI_BGQ_RMA_SPECIALIZED_FUNC_NAME(read,	\
			LOCK, AV, MR),				\
	.readv	= fi_no_rma_readv,				\
	.readmsg = FI_BGQ_RMA_SPECIALIZED_FUNC_NAME(readmsg,	\
			LOCK, AV, MR),				\
	.write	= FI_BGQ_RMA_SPECIALIZED_FUNC_NAME(write,	\
			LOCK, AV, MR),				\
	.inject = FI_BGQ_RMA_SPECIALIZED_FUNC_NAME(inject_write,\
			LOCK, AV, MR),				\
	.writev = FI_BGQ_RMA_SPECIALIZED_FUNC_NAME(writev,	\
			LOCK, AV, MR),				\
	.writemsg = FI_BGQ_RMA_SPECIALIZED_FUNC_NAME(writemsg,	\
			LOCK, AV, MR),				\
	.writedata = fi_no_rma_writedata,			\
}

FI_BGQ_RMA_OPS_STRUCT(0, FI_AV_MAP, FI_MR_SCALABLE);
FI_BGQ_RMA_OPS_STRUCT(0, FI_AV_MAP, FI_MR_BASIC);
FI_BGQ_RMA_OPS_STRUCT(1, FI_AV_MAP, FI_MR_SCALABLE);
FI_BGQ_RMA_OPS_STRUCT(1, FI_AV_MAP, FI_MR_BASIC);
FI_BGQ_RMA_OPS_STRUCT(0, FI_AV_TABLE, FI_MR_SCALABLE);
FI_BGQ_RMA_OPS_STRUCT(0, FI_AV_TABLE, FI_MR_BASIC);
FI_BGQ_RMA_OPS_STRUCT(1, FI_AV_TABLE, FI_MR_SCALABLE);
FI_BGQ_RMA_OPS_STRUCT(1, FI_AV_TABLE, FI_MR_BASIC);

static inline ssize_t fi_bgq_rma_read(struct fid_ep *ep,
		void *buf, size_t len, void *desc,
		fi_addr_t src_addr, uint64_t addr,
		uint64_t key, void *context)
{
	int lock_required;
	struct fi_bgq_ep *bgq_ep;

	bgq_ep = container_of(ep, struct fi_bgq_ep, ep_fid);

	switch (bgq_ep->threading) {
	case FI_THREAD_ENDPOINT:
	case FI_THREAD_DOMAIN:
		lock_required = 0;
	default:
		lock_required = 1;
	}

	return fi_bgq_read_generic(ep, buf, len, desc, src_addr,
			addr, key, context, lock_required,
			bgq_ep->av_type,
			bgq_ep->mr_mode);
}

static inline ssize_t fi_bgq_rma_readmsg(struct fid_ep *ep,
		const struct fi_msg_rma *msg, uint64_t flags)
{
	int lock_required;
	struct fi_bgq_ep *bgq_ep;

	bgq_ep = container_of(ep, struct fi_bgq_ep, ep_fid);

	switch (bgq_ep->threading) {
	case FI_THREAD_ENDPOINT:
	case FI_THREAD_DOMAIN:
		lock_required = 0;
	default:
		lock_required = 1;
	}

	return fi_bgq_readmsg_generic(ep, msg, flags,
			lock_required, bgq_ep->av_type,
			bgq_ep->mr_mode);
}

static inline ssize_t fi_bgq_rma_inject_write(struct fid_ep *ep,
		const void *buf, size_t len,
		fi_addr_t dst_addr, uint64_t addr, uint64_t key)
{
	int lock_required;
	struct fi_bgq_ep *bgq_ep;

	bgq_ep = container_of(ep, struct fi_bgq_ep, ep_fid);

	switch (bgq_ep->threading) {
		case FI_THREAD_ENDPOINT:
		case FI_THREAD_DOMAIN:
			lock_required = 0;
		default:
			lock_required = 1;
	}

	return fi_bgq_inject_write_generic(ep, buf, len, dst_addr,
			addr, key, lock_required, bgq_ep->av_type,
			bgq_ep->mr_mode);
}

static inline ssize_t fi_bgq_rma_write(struct fid_ep *ep,
		const void *buf, size_t len, void *desc,
		fi_addr_t dst_addr, uint64_t addr,
		uint64_t key, void *context)
{
	int lock_required;
	struct fi_bgq_ep *bgq_ep;

	bgq_ep = container_of(ep, struct fi_bgq_ep, ep_fid);

	switch (bgq_ep->threading) {
	case FI_THREAD_ENDPOINT:
	case FI_THREAD_DOMAIN:
		lock_required = 0;
	default:
		lock_required = 1;
	}

	return fi_bgq_write_generic(ep, buf, len, desc, dst_addr,
			addr, key, context, lock_required, bgq_ep->av_type,
			bgq_ep->mr_mode);
}

static inline ssize_t fi_bgq_rma_writev(struct fid_ep *ep,
		const struct iovec *iov, void **desc,
		size_t count, fi_addr_t dest_addr, uint64_t addr,
		uint64_t key, void *context)
{
	int lock_required;
	struct fi_bgq_ep *bgq_ep;

	bgq_ep = container_of(ep, struct fi_bgq_ep, ep_fid);

	switch (bgq_ep->threading) {
	case FI_THREAD_ENDPOINT:
	case FI_THREAD_DOMAIN:
		lock_required = 0;
	default:
		lock_required = 1;
	}

	return fi_bgq_writev_generic(ep, iov, desc, count, dest_addr, addr,
			key, context, lock_required, bgq_ep->av_type,
			bgq_ep->mr_mode);
}

static inline ssize_t fi_bgq_rma_writemsg(struct fid_ep *ep,
		const struct fi_msg_rma *msg, uint64_t flags)
{
	int lock_required;
	struct fi_bgq_ep *bgq_ep;

	bgq_ep = container_of(ep, struct fi_bgq_ep, ep_fid);

	switch (bgq_ep->threading) {
	case FI_THREAD_ENDPOINT:
	case FI_THREAD_DOMAIN:
		lock_required = 0;
	default:
		lock_required = 1;
	}

	return fi_bgq_writemsg_generic(ep, msg, flags,
			lock_required, bgq_ep->av_type,
			bgq_ep->mr_mode);
}

static struct fi_ops_rma fi_bgq_ops_rma_default = {
	.size		= sizeof(struct fi_ops_rma),
	.read		= fi_bgq_rma_read,
	.readv		= fi_no_rma_readv,
	.readmsg	= fi_bgq_rma_readmsg,
	.write		= fi_bgq_rma_write,
	.inject		= fi_bgq_rma_inject_write,
	.writev		= fi_bgq_rma_writev,
	.writemsg	= fi_bgq_rma_writemsg,
	.writedata	= fi_no_rma_writedata,
};

int fi_bgq_init_rma_ops(struct fi_bgq_ep *bgq_ep, struct fi_info *info) 
{
	if (!bgq_ep || !info) {
		errno = FI_EINVAL;
		goto err;
	}

	if (info->caps & FI_RMA ||
			(info->tx_attr && (info->tx_attr->caps & FI_RMA))) {
#if 0
		/*
		 * We delay setting bgq_ep->ep_fid.rma until we enable the EP
		 * because the application could change the address vector
		 * type between init and enable, and that may impact which
		 * ops we want to use.
		 */
		if (posix_memalign((void **) &bgq_ep->cmd_inject_write,
				FI_BGQ_CACHE_LINE_SIZE,
				sizeof(*bgq_ep->cmd_inject_write))) {
			errno = FI_ENOMEM;
			goto err;
		}

		if (posix_memalign((void **) &bgq_ep->cmd_write,
					FI_BGQ_CACHE_LINE_SIZE,
					sizeof(*bgq_ep->cmd_write))) {
			errno = FI_ENOMEM;
			goto err;
		}

		if (posix_memalign((void **) &bgq_ep->cmd_inject_write_match,
					FI_BGQ_CACHE_LINE_SIZE,
					sizeof(*bgq_ep->cmd_inject_write_match))) {
			errno = FI_ENOMEM;
			goto err;
		}

		if (posix_memalign((void **) &bgq_ep->cmd_write_match,
					FI_BGQ_CACHE_LINE_SIZE,
					sizeof(*bgq_ep->cmd_write_match))) {
			errno = FI_ENOMEM;
			goto err;
		}
#endif
	}

	return 0;
err:
	return -errno;
}

int fi_bgq_enable_rma_ops(struct fi_bgq_ep *bgq_ep)
{
	int lock_required;
	enum fi_av_type av_type;

	if (!bgq_ep || !bgq_ep->domain) {
		errno = FI_EINVAL;
		goto err;
	}

	if (!(bgq_ep->tx.caps & FI_RMA)) {
		/* rma ops not enabled on this endpoint */
		return 0;
	}

	av_type = bgq_ep->av->type;

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
		errno = FI_EINVAL;
		goto err;
	}

	enum fi_mr_mode mr_mode = bgq_ep->domain->mr_mode;

	if (lock_required == 0 &&
			av_type == FI_AV_MAP &&
			mr_mode == FI_MR_SCALABLE) {
		bgq_ep->ep_fid.rma =
			&FI_BGQ_RMA_OPS_STRUCT_NAME(0, FI_AV_MAP, FI_MR_SCALABLE);
	} else if (lock_required == 0 &&
			av_type == FI_AV_MAP &&
			mr_mode == FI_MR_BASIC) {
		bgq_ep->ep_fid.rma =
			&FI_BGQ_RMA_OPS_STRUCT_NAME(0, FI_AV_MAP, FI_MR_BASIC);
	} else if (lock_required == 1 &&
			av_type == FI_AV_MAP &&
			mr_mode == FI_MR_SCALABLE) {
		bgq_ep->ep_fid.rma =
			&FI_BGQ_RMA_OPS_STRUCT_NAME(1, FI_AV_MAP, FI_MR_SCALABLE);
	} else if (lock_required == 1 &&
			av_type == FI_AV_MAP &&
			mr_mode == FI_MR_BASIC) {
		bgq_ep->ep_fid.rma =
			&FI_BGQ_RMA_OPS_STRUCT_NAME(1, FI_AV_MAP, FI_MR_BASIC);
	} else if (lock_required == 0 &&
			av_type == FI_AV_TABLE &&
			mr_mode == FI_MR_SCALABLE) {
		bgq_ep->ep_fid.rma =
			&FI_BGQ_RMA_OPS_STRUCT_NAME(0, FI_AV_TABLE, FI_MR_SCALABLE);
	} else if (lock_required == 0 &&
			av_type == FI_AV_TABLE &&
			mr_mode == FI_MR_BASIC) {
		bgq_ep->ep_fid.rma =
			&FI_BGQ_RMA_OPS_STRUCT_NAME(0, FI_AV_TABLE, FI_MR_BASIC);
	} else if (lock_required == 1 &&
			av_type == FI_AV_TABLE &&
			mr_mode == FI_MR_SCALABLE) {
		bgq_ep->ep_fid.rma =
			&FI_BGQ_RMA_OPS_STRUCT_NAME(1, FI_AV_TABLE, FI_MR_SCALABLE);
	} else if (lock_required == 1 &&
			av_type == FI_AV_TABLE &&
			mr_mode == FI_MR_BASIC) {
		bgq_ep->ep_fid.rma =
			&FI_BGQ_RMA_OPS_STRUCT_NAME(1, FI_AV_TABLE, FI_MR_BASIC);
	} else {
		bgq_ep->ep_fid.rma = &fi_bgq_ops_rma_default;
	}

	return 0;
err:
	return -errno;
}

int fi_bgq_finalize_rma_ops(struct fi_bgq_ep *bgq_ep)
{
	if (!bgq_ep) {
		return 0;
	}

	return 0;
}
