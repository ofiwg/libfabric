#ifndef _FI_BGQ_DIRECT_TAGGED_H_
#define _FI_BGQ_DIRECT_TAGGED_H_

#define FABRIC_DIRECT_TAGGED 1

#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_direct_eq.h>
#include <rdma/fi_direct_endpoint.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline
ssize_t fi_bgq_tinject(struct fid_ep *ep,
		const void *buf,
		size_t len,
		fi_addr_t dest_addr,
		uint64_t tag,
		int lock_required,
		enum fi_av_type av_type,
		enum fi_progress progress)
{
	return fi_bgq_inject_generic(ep, buf, len, dest_addr, tag,
			lock_required, av_type, progress, 0);
}

/*
 * In FI_PROGRESS_AUTO mode:
 * The bgq 'recv' implementation is THREAD SAFE and LOCKLESS due to its use of
 * the L2 atomic operations to post the match information to the progress thread.
 * The 'fi_bgq_lock_if_required()' utility function is not used.
 *
 * \note The bgq provider asserts the following mode bits which affect
 * 	the behavior of this routine:
 *
 * 	- 'FI_ASYNC_IOV' mode bit which requires the application to maintain
 * 	  the 'msg->msg_iov' iovec array until the operation completes
 *
 * 	- 'FI_LOCAL_MR' mode bit which allows the provider to ignore the 'desc'
 * 	  parameter .. no memory regions are required to access the local
 * 	  memory
 */
static inline
ssize_t fi_bgq_trecvmsg_generic (struct fid_ep *ep,
		const struct fi_msg_tagged *msg,
		uint64_t flags,
		const int lock_required,
		const enum fi_av_type av_type,
		const enum fi_progress progress)
{
	struct fi_bgq_ep * bgq_ep = container_of(ep, struct fi_bgq_ep, ep_fid);
	uint64_t context_rsh3b = 0;

	if (msg->iov_count == 0) {
		assert(msg->context);
		assert(((uintptr_t)msg->context & 0x07ull) == 0);	/* must be 8 byte aligned */

		union fi_bgq_context * bgq_context =
			(union fi_bgq_context *) msg->context;
		bgq_context->flags = flags;
		bgq_context->len = 0;
		bgq_context->buf = NULL;
		bgq_context->byte_counter = (uint64_t)-1;
		if ((flags & (FI_PEEK | FI_CLAIM)) != FI_CLAIM) {
			/* do not overwrite state from a previous "peek|claim" operation */
			bgq_context->tag = msg->tag;
			bgq_context->ignore = msg->ignore;
		}

		context_rsh3b = (uint64_t)bgq_context >> 3;

	} else if (msg->iov_count == 1) {
		assert(msg->context);
		assert(((uintptr_t)msg->context & 0x07ull) == 0);	/* must be 8 byte aligned */

		union fi_bgq_context * bgq_context =
			(union fi_bgq_context *) msg->context;
		bgq_context->flags = flags;
		bgq_context->len = msg->msg_iov[0].iov_len;
		bgq_context->buf = msg->msg_iov[0].iov_base;
		bgq_context->byte_counter = (uint64_t)-1;
		if ((flags & (FI_PEEK | FI_CLAIM)) != FI_CLAIM) {
			/* do not overwrite state from a previous "peek|claim" operation */
			bgq_context->tag = msg->tag;
			bgq_context->ignore = msg->ignore;
		}

		context_rsh3b = (uint64_t)bgq_context >> 3;

	} else {
		assert((flags & (FI_PEEK | FI_CLAIM)) != FI_CLAIM);	/* TODO - why not? */

		struct fi_bgq_context_ext * ext;
		posix_memalign((void**)&ext, 32, sizeof(struct fi_bgq_context_ext));
		flags |= FI_BGQ_CQ_CONTEXT_EXT;

		ext->bgq_context.flags = flags;
		ext->bgq_context.byte_counter = (uint64_t)-1;
		ext->bgq_context.tag = msg->tag;
		ext->bgq_context.ignore = msg->ignore;
		ext->msg.op_context = msg->context;
		ext->msg.iov_count = msg->iov_count;
		ext->msg.iov = (struct iovec *)msg->msg_iov;

		context_rsh3b = (uint64_t)ext >> 3;
		if (FI_PROGRESS_MANUAL == progress) {	/* constant expression will compile out */

			int ret;
			ret = fi_bgq_lock_if_required(&bgq_ep->lock, lock_required);
			if (ret) return ret;

			fi_bgq_ep_progress_manual_recv(bgq_ep,
				0, /* is_msg */
				(union fi_bgq_context *)(context_rsh3b << 3),
				flags,
				1  /* is_context_ext */);

			ret = fi_bgq_unlock_if_required(&bgq_ep->lock, lock_required);
			if (ret) return ret;

			return 0;
		}
	}

	if (FI_PROGRESS_MANUAL == progress) {	/* constant expression will compile out */

		int ret;
		ret = fi_bgq_lock_if_required(&bgq_ep->lock, lock_required);
		if (ret) return ret;

		fi_bgq_ep_progress_manual_recv(bgq_ep,
			0, /* is_msg */
			(union fi_bgq_context *)(context_rsh3b << 3),
			flags,
			0  /* is_context_ext */);

		ret = fi_bgq_unlock_if_required(&bgq_ep->lock, lock_required);
		if (ret) return ret;

	} else {

		/* the *only* difference between a 'tagged' and 'non-tagged' recv is
		 * the L2 atomic fifo used to post the receive information */
		struct l2atomic_fifo_producer * fifo = &bgq_ep->rx.post.match[0];	/* TODO - use enum */

		while (l2atomic_fifo_produce(fifo, context_rsh3b) != 0);		/* spin loop! */
	}

	return 0;
}

/*
 * Declare specialized functions that qualify for FABRIC_DIRECT.
 * - No locks
 * - FI_AV_MAP
 * - FI_PROGRESS_MANUAL
 */
#define FI_BGQ_TAGGED_FABRIC_DIRECT_LOCK	0
#define FI_BGQ_TAGGED_FABRIC_DIRECT_AV		FI_AV_MAP

FI_BGQ_TAGGED_SPECIALIZED_FUNC(FI_BGQ_TAGGED_FABRIC_DIRECT_LOCK,
		FI_BGQ_TAGGED_FABRIC_DIRECT_AV,
		FI_BGQ_FABRIC_DIRECT_PROGRESS)

#ifdef FABRIC_DIRECT
#define fi_tsend(ep, buf, len, desc, dest_addr, tag, context)		\
	(FI_BGQ_TAGGED_SPECIALIZED_FUNC_NAME(tsend,			\
			FI_BGQ_TAGGED_FABRIC_DIRECT_LOCK,		\
			FI_BGQ_TAGGED_FABRIC_DIRECT_AV,			\
			FI_BGQ_FABRIC_DIRECT_PROGRESS)			\
	(ep, buf, len, desc, dest_addr, tag, context))

#define fi_trecv(ep, buf, len, desc, src_addr, tag, ignore, context)	\
	(FI_BGQ_TAGGED_SPECIALIZED_FUNC_NAME(trecv,			\
			FI_BGQ_TAGGED_FABRIC_DIRECT_LOCK,		\
			FI_BGQ_TAGGED_FABRIC_DIRECT_AV,			\
			FI_BGQ_FABRIC_DIRECT_PROGRESS)			\
	(ep, buf, len, desc, src_addr, tag, ignore, context))

#define fi_tinject(ep, buf, len, dest_addr, tag)			\
	(FI_BGQ_TAGGED_SPECIALIZED_FUNC_NAME(tinject,			\
			FI_BGQ_TAGGED_FABRIC_DIRECT_LOCK,		\
			FI_BGQ_TAGGED_FABRIC_DIRECT_AV,			\
			FI_BGQ_FABRIC_DIRECT_PROGRESS)			\
	(ep, buf, len, dest_addr, tag))

#define fi_tsenddata(ep, buf, len, desc, data, dest_addr, tag, context)	\
	(FI_BGQ_TAGGED_SPECIALIZED_FUNC_NAME(tsenddata,			\
			FI_BGQ_TAGGED_FABRIC_DIRECT_LOCK,		\
			FI_BGQ_TAGGED_FABRIC_DIRECT_AV,			\
			FI_BGQ_FABRIC_DIRECT_PROGRESS)			\
	(ep, buf, len, desc, data, dest_addr, tag, context))

#define fi_tinjectdata(ep, buf, len, data, dest_addr, tag)		\
	(FI_BGQ_TAGGED_SPECIALIZED_FUNC_NAME(tinjectdata,		\
			FI_BGQ_TAGGED_FABRIC_DIRECT_LOCK,		\
			FI_BGQ_TAGGED_FABRIC_DIRECT_AV,			\
			FI_BGQ_FABRIC_DIRECT_PROGRESS)			\
	(ep, buf, len, data, dest_addr, tag))

#define fi_trecvmsg(ep, msg, flags)					\
	(FI_BGQ_TAGGED_SPECIALIZED_FUNC_NAME(trecvmsg,			\
			FI_BGQ_TAGGED_FABRIC_DIRECT_LOCK,		\
			FI_BGQ_TAGGED_FABRIC_DIRECT_AV,			\
			FI_BGQ_FABRIC_DIRECT_PROGRESS)			\
	(ep, msg, flags))

static inline ssize_t
fi_tsendmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg, uint64_t flags)
{
        return ep->tagged->sendmsg(ep, msg, flags);
}

#endif

#ifdef __cplusplus
}
#endif

#endif /* _FI_BGQ_DIRECT_TAGGED_H_ */
