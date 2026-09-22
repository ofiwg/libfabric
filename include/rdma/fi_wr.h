/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef FI_WR_H
#define FI_WR_H

#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_trigger.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * OFI Work Request (WR) API.
 *
 * The standard post calls format a work request, hand it to the provider, and
 * initiate it on the fabric in one step.  This interface splits that lifecycle
 * into prepare, modify, queue, and flush, so that formatting can be separated
 * in time and place from submission.  Available when the endpoint reports the
 * FI_WR capability.  See fi_wr(3).
 *
 * An operation is described on input by struct fi_wr_attr, which reuses the
 * struct fi_op_* descriptors of the deferred work queue interface, and is
 * formatted into an fi_wr.  The latter is an opaque handle (void *): the
 * contents are provider defined and the buffer size is reported through
 * fi_ep_attr::max_tx_wr_size and max_rx_wr_size, so sizeof() is never meaningful.
 *
 * The queue calls take the work request by const pointer.  State that belongs
 * to the queue slot rather than to the operation, such as a descriptor phase
 * bit or a device request id, is written to the queue, never back into the work
 * request.  This is what allows one work request to be queued concurrently by
 * many threads, or by many work items of a device kernel.
 */

struct fi_wr_attr {
	enum fi_op_type				op_type;

	union {
		struct fi_op_msg		*msg;
		struct fi_op_tagged		*tagged;
		struct fi_op_rma		*rma;
		struct fi_op_atomic		*atomic;
		struct fi_op_fetch_atomic	*fetch_atomic;
		struct fi_op_compare_atomic	*compare_atomic;
	} op;
};

typedef void *fi_wr;

struct fi_ops_wr {
	size_t	size;
	int	(*prepare)(struct fid_ep *ep, const struct fi_wr_attr *attr,
			fi_wr wr, size_t *wr_len);
	int	(*queue_tx)(struct fid_ep *ep, const fi_wr wr,
			void *context);
	int	(*queue_rx)(struct fid_ep *ep, const fi_wr wr,
			void *context);
	int	(*queue_trx)(struct fid_ep *ep, const fi_wr wr,
			void *context);
	int	(*modify_addr)(struct fid_ep *ep, fi_wr wr,
			fi_addr_t addr);
	int	(*modify_iov)(struct fid_ep *ep, fi_wr wr,
			const struct iovec *iov, void **desc, size_t count);
	int	(*modify_rma_iov)(struct fid_ep *ep, fi_wr wr,
			const struct fi_rma_iov *rma_iov, size_t count);
	int	(*modify_tag)(struct fid_ep *ep, fi_wr wr,
			uint64_t tag, uint64_t ignore);
	int	(*modify_data)(struct fid_ep *ep, fi_wr wr,
			uint64_t data);
	int	(*modify_flags)(struct fid_ep *ep, fi_wr wr,
			uint64_t flags);
};

#ifndef FABRIC_DIRECT_WR

static inline int
fi_wr_prepare(struct fid_ep *ep, const struct fi_wr_attr *attr,
	      fi_wr wr, size_t *wr_len)
{
	return ep->wr->prepare(ep, attr, wr, wr_len);
}

static inline int
fi_wr_queue_tx(struct fid_ep *ep, const fi_wr wr, void *context)
{
	return ep->wr->queue_tx(ep, wr, context);
}

static inline int
fi_wr_queue_rx(struct fid_ep *ep, const fi_wr wr, void *context)
{
	return ep->wr->queue_rx(ep, wr, context);
}

static inline int
fi_wr_queue_trx(struct fid_ep *ep, const fi_wr wr, void *context)
{
	return ep->wr->queue_trx(ep, wr, context);
}

static inline int
fi_wr_modify_addr(struct fid_ep *ep, fi_wr wr, fi_addr_t addr)
{
	return ep->wr->modify_addr(ep, wr, addr);
}

static inline int
fi_wr_modify_iov(struct fid_ep *ep, fi_wr wr,
		 const struct iovec *iov, void **desc, size_t count)
{
	return ep->wr->modify_iov(ep, wr, iov, desc, count);
}

static inline int
fi_wr_modify_rma_iov(struct fid_ep *ep, fi_wr wr,
		     const struct fi_rma_iov *rma_iov, size_t count)
{
	return ep->wr->modify_rma_iov(ep, wr, rma_iov, count);
}

static inline int
fi_wr_modify_tag(struct fid_ep *ep, fi_wr wr, uint64_t tag,
		 uint64_t ignore)
{
	return ep->wr->modify_tag(ep, wr, tag, ignore);
}

static inline int
fi_wr_modify_data(struct fid_ep *ep, fi_wr wr, uint64_t data)
{
	return ep->wr->modify_data(ep, wr, data);
}

static inline int
fi_wr_modify_flags(struct fid_ep *ep, fi_wr wr, uint64_t flags)
{
	return ep->wr->modify_flags(ep, wr, flags);
}

#endif /* FABRIC_DIRECT_WR */

#ifdef __cplusplus
}
#endif

#endif /* FI_WR_H */
