/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. */

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "efa.h"
#include "efa_base_ep.h"
#include "efa_cq.h"
#include "efa_cntr.h"
#include "efa_av.h"
#include "efa_mr.h"
#include "efa_xpu.h"

#include <rdma/fi_xpu_device_efa.h>

/*
 * efadv.h is already pulled in by efa_base_ep.h -> infiniband/efadv.h.
 * No additional include needed here.
 */

/*
 * An XPU drives the direct data path from the device, so it needs what that
 * path needs: efadv_query_qp_wqs() and efadv_query_cq(), which also find the
 * hardware queues to map. Without them the provider still builds - it just has
 * no XPU support to offer, so every entry point below reports that and the rest
 * of the file drops out.
 */
#if HAVE_EFA_DATA_PATH_DIRECT

/* ----------------------------------------------------------------
 * Memory allocation / import helpers
 * ---------------------------------------------------------------- */

static void xpu_put_fd(struct efa_xpu_ctx *xctx, int fd);

/**
 * xpu_alloc - allocate device memory via user-provided ops or HMEM.
 *
 * If the XPU attr has ops->alloc, use that (user-managed allocation).
 * Otherwise fall back to the HMEM device ops, which take the same arguments.
 */
static int xpu_alloc(struct efa_xpu_ctx *xctx, uint64_t size,
		     uint64_t alignment, uint64_t flags,
		     void **addr, int *fd, uint64_t *offset)
{
	struct fi_xpu_attr *attr = &xctx->attr;
	int ret;

	if (!(attr->ops && attr->ops->alloc))
		return ofi_hmem_dev_alloc(attr->iface, attr->device, size,
					  alignment, flags, addr, fd, offset);

	ret = attr->ops->alloc(attr->device, size, alignment, flags, addr, fd,
			       offset);
	if (ret)
		return ret;

	/*
	 * What the application's allocator hands back is placed where the
	 * provider asked for it: a ring the NIC writes and a handle a kernel
	 * reads both assume the alignment they were allocated with, and a CQ
	 * ring is only reachable by the NIC through the dmabuf it was asked to
	 * come with. Neither failure would show up here - only as a NIC or a
	 * kernel misreading the memory later - so refuse the allocation, and
	 * give it back, now.
	 */
	if (!*addr ||
	    (alignment && ((uintptr_t) *addr & (alignment - 1))) ||
	    ((flags & FI_XPU_ALLOC_DMABUF) && (!fd || *fd < 0))) {
		EFA_WARN(FI_LOG_DOMAIN,
			 "fi_xpu_ops alloc returned %p, fd %d for a %" PRIu64
			 "-byte allocation aligned to %" PRIu64
			 "%s; refusing it\n", *addr, fd ? *fd : -1, size,
			 alignment,
			 (flags & FI_XPU_ALLOC_DMABUF) ? " with a dmabuf" : "");
		if (fd && *fd >= 0) {
			xpu_put_fd(xctx, *fd);
			*fd = -1;
		}
		if (*addr)
			attr->ops->free(attr->device, *addr);
		*addr = NULL;
		return -FI_EINVAL;
	}

	return 0;
}

/**
 * xpu_import - map host address into device address space.
 */
static int xpu_import(struct efa_xpu_ctx *xctx, void *host_addr,
		      uint64_t size, uint64_t flags, void **dev_addr)
{
	struct fi_xpu_attr *attr = &xctx->attr;

	/*
	 * EFA maps its queues, doorbells and host pages by address, never
	 * through a dmabuf, so no fd goes with an import (FI_XPU_IMPORT_DMABUF
	 * is not used).
	 */
	if (attr->ops && attr->ops->import)
		return attr->ops->import(attr->device, host_addr,
					 size, -1, flags, dev_addr);

	/*
	 * Provider-managed import: register the host/MMIO region with the
	 * HMEM layer and return a device-accessible pointer (cuMemHostRegister
	 * + cuMemHostGetDevicePointer for CUDA).
	 */
	return ofi_hmem_dev_import(attr->iface, attr->device, host_addr, size,
				   -1, flags, dev_addr);
}

/**
 * xpu_unimport - release a mapping made by xpu_import.
 *
 * Whoever made the mapping releases it. An application that supplied ->import
 * but no ->unimport keeps the mapping: there is nothing the provider can call,
 * and the HMEM ops did not make it.
 */
static void xpu_unimport(struct efa_xpu_ctx *xctx, void *host_addr)
{
	struct fi_xpu_attr *attr = &xctx->attr;
	int ret;

	if (!host_addr)
		return;

	if (attr->ops && attr->ops->import) {
		if (!attr->ops->unimport)
			return;
		ret = attr->ops->unimport(attr->device, host_addr);
	} else {
		ret = ofi_hmem_dev_unimport(attr->iface, attr->device,
					    host_addr);
	}

	if (ret)
		EFA_WARN(FI_LOG_EP_CTRL,
			 "Failed to release XPU mapping of %p: %d\n",
			 host_addr, ret);
}

/**
 * xpu_free - free device memory via user-provided ops or HMEM.
 */
static void xpu_free(struct efa_xpu_ctx *xctx, void *addr)
{
	struct fi_xpu_attr *attr = &xctx->attr;

	if (attr->ops && attr->ops->free) {
		attr->ops->free(attr->device, addr);
		return;
	}

	if (!addr)
		return;

	/* Provider-managed: release the device allocation via HMEM ops. */
	ofi_hmem_dev_free(attr->iface, attr->device, addr);
}

/**
 * xpu_put_fd - release a dmabuf fd xpu_alloc exported.
 *
 * The fd is only needed until the NIC object the memory was allocated for has
 * been created over it and holds its own reference, so it is released there
 * rather than at free time. Whichever side allocated the memory, the fd names a
 * dmabuf of the context iface and is released the way that iface releases one.
 */
static void xpu_put_fd(struct efa_xpu_ctx *xctx, int fd)
{
	int ret;

	if (fd < 0)
		return;

	ret = ofi_hmem_put_dmabuf_fd(xctx->attr.iface, fd);
	if (ret)
		EFA_WARN(FI_LOG_EP_CTRL,
			 "Failed to release dmabuf fd %d: %d\n", fd, ret);
}

/*
 * The CQ ring and the counters are allocated where they are opened, not here,
 * because the NIC wants their dmabuf fd at creation time. They still have to
 * come from - and go back to - whichever side owns the XPU context memory, so
 * those paths reach the same helpers through here.
 */
int efa_xpu_mem_alloc(struct fid_xpu_ctx *ctx, uint64_t size,
		      uint64_t alignment, uint64_t flags, void **addr, int *fd,
		      uint64_t *offset)
{
	return xpu_alloc(container_of(ctx, struct efa_xpu_ctx, ctx), size,
			 alignment, flags, addr, fd, offset);
}

void efa_xpu_mem_free(struct fid_xpu_ctx *ctx, void *addr)
{
	xpu_free(container_of(ctx, struct efa_xpu_ctx, ctx), addr);
}

void efa_xpu_mem_put_fd(struct fid_xpu_ctx *ctx, int fd)
{
	if (fd < 0)
		return;

	xpu_put_fd(container_of(ctx, struct efa_xpu_ctx, ctx), fd);
}

/* ----------------------------------------------------------------
 * XPU Context (fid_xpu_ctx) management
 * ---------------------------------------------------------------- */

static int efa_xpu_ctx_query(struct fid_xpu_ctx *ctx,
			     struct fi_xpu_ctx_attr *attr)
{
	attr->caps = FI_XPU_CAP_EP | FI_XPU_CAP_CQ | FI_XPU_CAP_CNTR;
	attr->av_addr_size = sizeof(struct efa_xpu_peer);
	attr->mr_desc_size = sizeof(struct efa_xpu_desc);
	return 0;
}

static struct fi_ops_xpu_ctx efa_xpu_ctx_ops = {
	.size = sizeof(struct fi_ops_xpu_ctx),
	.query = efa_xpu_ctx_query,
};

static int efa_xpu_ctx_close(struct fid *fid)
{
	struct efa_xpu_ctx *ctx;

	ctx = container_of(fid, struct efa_xpu_ctx, ctx.fid);
	free(ctx);
	return 0;
}

static struct fi_ops efa_xpu_ctx_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = efa_xpu_ctx_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

/*
 * Check the memory callbacks an application supplied, before anything is
 * allocated through them.
 *
 * Every callback in struct fi_xpu_ops is part of its first layout, so a size
 * smaller than that is not an older application - there is none - but a table
 * this provider would read past the end of. A larger size is a newer
 * application, whose extra members this provider does not know and ignores.
 *
 * Memory goes back to whoever allocated it, so alloc and free come as a pair:
 * with only one of them the provider would free application memory through the
 * HMEM ops, or hand its own allocation to the application's free. unimport only
 * releases what import mapped, so it is meaningless without it. import without
 * unimport is allowed, and documented: the application keeps those mappings.
 */
static int efa_xpu_ops_valid(const struct fi_xpu_ops *ops)
{
	if (ops->size < sizeof(struct fi_xpu_ops)) {
		EFA_WARN(FI_LOG_DOMAIN,
			 "fi_xpu_ops size %zu is smaller than the %zu its "
			 "callbacks need\n", ops->size,
			 sizeof(struct fi_xpu_ops));
		return 0;
	}

	if (!ops->alloc != !ops->free) {
		EFA_WARN(FI_LOG_DOMAIN,
			 "fi_xpu_ops must supply alloc and free together\n");
		return 0;
	}

	if (ops->unimport && !ops->import) {
		EFA_WARN(FI_LOG_DOMAIN,
			 "fi_xpu_ops supplies unimport without import\n");
		return 0;
	}

	return 1;
}

int efa_xpu_ctx_open(struct fid_domain *domain, struct fi_xpu_attr *attr,
		     struct fid_xpu_ctx **ctx_out, void *context)
{
	struct efa_xpu_ctx *ctx;
	struct efa_domain *efa_domain;

	if (!domain || !attr || !ctx_out)
		return -FI_EINVAL;

	if (attr->ops && !efa_xpu_ops_valid(attr->ops))
		return -FI_EINVAL;

	/*
	 * Only efa-direct exports its queues to a device. The rdm and dgram
	 * paths keep the send queue behind the provider's own protocol state,
	 * so a kernel posting into it would corrupt them. FI_XPU is advertised
	 * on the efa-direct info alone (see efa_prov.c), but a domain op can be
	 * reached from any domain, so the domain is checked here too.
	 */
	efa_domain = container_of(domain, struct efa_domain,
				  util_domain.domain_fid);
	if (efa_domain->info_type != EFA_INFO_DIRECT) {
		EFA_WARN(FI_LOG_DOMAIN,
			 "FI_XPU is only supported on an efa-direct domain\n");
		return -FI_EOPNOTSUPP;
	}

	ctx = calloc(1, sizeof(*ctx));
	if (!ctx)
		return -FI_ENOMEM;

	ctx->ctx.fid.fclass = FI_CLASS_XPU_CTX;
	ctx->ctx.fid.context = context;
	ctx->ctx.fid.ops = &efa_xpu_ctx_fi_ops;
	ctx->ctx.ops = &efa_xpu_ctx_ops;
	ctx->attr = *attr;
	ctx->domain = domain;

	*ctx_out = &ctx->ctx;
	return 0;
}

/* ----------------------------------------------------------------
 * State lifecycle
 * ---------------------------------------------------------------- */

struct efa_xpu_ep_state *efa_xpu_ep_state_create(struct fid_xpu_ctx *ctx)
{
	struct efa_xpu_ep_state *state;

	state = calloc(1, sizeof(*state));
	if (!state)
		return NULL;

	state->xpu_ctx = ctx;
	return state;
}

struct efa_xpu_cq_state *efa_xpu_cq_state_create(struct fid_xpu_ctx *ctx)
{
	struct efa_xpu_cq_state *state;

	state = calloc(1, sizeof(*state));
	if (!state)
		return NULL;

	state->xpu_ctx = ctx;
	return state;
}

struct efa_xpu_cntr_state *efa_xpu_cntr_state_create(struct fid_xpu_ctx *ctx)
{
	struct efa_xpu_cntr_state *state;

	state = calloc(1, sizeof(*state));
	if (!state)
		return NULL;

	state->xpu_ctx = ctx;
	return state;
}

void efa_xpu_ep_state_destroy(struct efa_xpu_ep_state *state)
{
	struct efa_xpu_ctx *xctx;
	size_t i;

	if (!state)
		return;

	xctx = container_of(state->xpu_ctx, struct efa_xpu_ctx, ctx);

	/* Unmap the hardware queues before the QP they belong to goes away. */
	for (i = 0; i < state->imported_cnt; i++)
		xpu_unimport(xctx, state->imported[i]);

	if (state->dev_ep)
		xpu_free(xctx, state->dev_ep);

	if (state->done_registered)
		ofi_hmem_dev_unregister(xctx->attr.iface, state->done_handle);
	if (state->done_host) {
		xpu_unimport(xctx, state->done_host);
		free(state->done_host);
	} else if (state->done_dev) {
		xpu_free(xctx, state->done_dev);
	}

	for (i = 0; i < EFA_XPU_DONE_CNT; i++) {
		if (state->ctx_host[i]) {
			xpu_unimport(xctx, state->ctx_host[i]);
			free(state->ctx_host[i]);
		}
		if (state->ctx_dev[i])
			xpu_free(xctx, state->ctx_dev[i]);
		if (state->slot_dev[i])
			xpu_free(xctx, state->slot_dev[i]);
		free(state->slot_cnt[i]);
	}
	if (state->ddone_dev)
		xpu_free(xctx, state->ddone_dev);

	free(state);
}

/*
 * Publish one word of the endpoint's done memory, at byte offset @off, from
 * the host copy @val: a store the device sees in its own memory through the
 * registration, or in the mapped host memory. Called with the CQ's
 * ep_list_lock held, so a host copy is never advanced by two readers at once.
 */
static void efa_xpu_publish(struct efa_xpu_ep_state *state, size_t off,
			    const uint32_t *val, size_t cnt)
{
	struct efa_xpu_ctx *xctx = container_of(state->xpu_ctx,
						struct efa_xpu_ctx, ctx);
	size_t i;
	int ret;

	if (state->done_host) {
		for (i = 0; i < cnt; i++)
			__atomic_store_n((uint32_t *) ((char *) state->done_host +
						       off) + i,
					 val[i], __ATOMIC_RELEASE);
		return;
	}

	ret = ofi_hmem_dev_reg_copy_to_hmem(xctx->attr.iface,
					    state->done_handle,
					    (char *) state->done_dev + off,
					    val, cnt * sizeof(uint32_t));
	if (OFI_UNLIKELY(ret))
		EFA_WARN(FI_LOG_CQ, "publishing an XPU done cursor failed: %d\n",
			 ret);
}

/*
 * Publish @queue's done cursor, which the CQ read now ending has advanced, and
 * take it off the read's flush list (see efa_cq_xpu_flush()).
 */
void efa_xpu_ep_publish_done(struct efa_xpu_ep_state *state, int queue)
{
	int other = !queue;

	/* Already published along with the other queue's cursor. */
	if (!state->done_listed[queue])
		return;
	state->done_listed[queue] = false;

	/*
	 * The two cursors are adjacent, so when the read advanced both - both
	 * queues report to this CQ - one copy publishes them. Queues on
	 * different CQs are read under different locks, and copying the other
	 * one's word could put back a value older than what its own CQ read
	 * has published.
	 */
	if (state->done_listed[other] &&
	    state->done_cq[other] == state->done_cq[queue]) {
		state->done_listed[other] = false;
		efa_xpu_publish(state, 0, state->done, EFA_XPU_DONE_CNT);
		return;
	}

	efa_xpu_publish(state, queue * sizeof(uint32_t), &state->done[queue],
			1);
}

/*
 * Count a CQE a host CQ read consumed against @queue, whose descriptor carried
 * @req_id: once against the slot it frees, where the queue has a context
 * table, and once against the queue's done cursor. The context has already
 * been taken out of the table by the time the read gets here, so the slot's
 * count may let a poster reuse the entry at once, and it is published here.
 * The done cursor is published once per read instead: the queue goes on its
 * CQ's flush list, and only a read that touches more queues than the list
 * holds publishes the extra ones at once.
 */
void efa_xpu_ep_note_done(struct efa_xpu_ep_state *state, int queue,
			  uint16_t req_id)
{
	if (!state->done_cq[queue])
		return;

	if (state->slot_cnt[queue]) {
		uint32_t slot = req_id & state->slot_mask[queue];

		state->slot_cnt[queue][slot]++;
		efa_xpu_publish(state, state->slot_off[queue] +
				slot * sizeof(uint32_t),
				&state->slot_cnt[queue][slot], 1);
	}

	if (state->done_bound[queue]) {
		struct efa_cq *cq = state->done_cq[queue];

		state->done[queue]++;
		if (state->done_listed[queue])
			return;
		if (cq->xpu_dirty_cnt == EFA_CQ_XPU_DIRTY_MAX) {
			efa_xpu_publish(state, queue * sizeof(uint32_t),
					&state->done[queue], 1);
			return;
		}
		cq->xpu_dirty[cq->xpu_dirty_cnt].state = state;
		cq->xpu_dirty[cq->xpu_dirty_cnt].queue = queue;
		cq->xpu_dirty_cnt++;
		state->done_listed[queue] = true;
	}
}

/*
 * Take or drop the ep_list_lock of the CQs in @cq, each once. These are the
 * locks a host CQ read holds while it publishes, so holding them keeps a read
 * from seeing a done cursor half set up or half torn down. They are taken from
 * the CQs themselves rather than the endpoint, whose util_ep has already let
 * go of them by the time a close detaches it.
 */
static void efa_xpu_done_lock(struct efa_cq **cq, bool lock)
	/* No lock analysis: clang cannot reason about conditional locking. */
	OFI_TSA_NO_ANALYSIS
{
	int i;

	for (i = 0; i < EFA_XPU_DONE_CNT; i++) {
		if (!cq[i] || (i && cq[i] == cq[0]))
			continue;
		if (lock)
			ofi_genlock_lock(&cq[i]->util_cq.ep_list_lock);
		else
			ofi_genlock_unlock(&cq[i]->util_cq.ep_list_lock);
	}
}

/*
 * The CQ a queue's completions go to, if the host reads it. A CQ opened for an
 * XPU is read only on the device, and nothing on the host would advance a done
 * cursor for it.
 */
static struct efa_cq *efa_xpu_host_cq(struct util_cq *util_cq)
{
	struct efa_cq *cq;

	if (!util_cq)
		return NULL;

	cq = container_of(util_cq, struct efa_cq, util_cq);
	return cq->xpu_state ? NULL : cq;
}

/* The CQ a queue's completions go to, if the device reads it. */
static struct efa_cq *efa_xpu_dev_cq(struct util_cq *util_cq)
{
	struct efa_cq *cq;

	if (!util_cq)
		return NULL;

	cq = container_of(util_cq, struct efa_cq, util_cq);
	return cq->xpu_state ? cq : NULL;
}

/* Point the host poll of the QP's @queue at a context table, or at none. */
static void efa_xpu_qp_set_ctx(struct efa_base_ep *base_ep, struct efa_cq *cq,
			       int queue, const uint64_t *ctx, uint32_t mask)
{
	struct efa_data_path_direct_qp *dqp = &base_ep->qp->data_path_direct_qp;
	struct efa_cq *lock[EFA_XPU_DONE_CNT] = { cq, NULL };

	efa_xpu_done_lock(lock, true);
	if (queue == EFA_XPU_SQ_DONE) {
		dqp->sq.xpu_ctx = ctx;
		dqp->sq.xpu_ctx_mask = mask;
	} else {
		dqp->rq.xpu_ctx = ctx;
		dqp->rq.xpu_ctx_mask = mask;
	}
	efa_xpu_done_lock(lock, false);
}

/*
 * Give the queues whose descriptors cannot carry the application's context a
 * context table: the receive queue always, its request ID being 16 bits, and
 * the send queue when it has no 64-bit request ID (@sq_req_id_64 false). A
 * post stores the context at the slot it is given and puts the slot in the
 * request ID, and a completion takes it back out, so the table holds one
 * 8-byte entry per ring slot, and the 16-bit request ID limits the ring to
 * 64Ki entries.
 *
 * There is one copy of each table, placed where the queue's completions are
 * read, so nothing synchronizes copies. For a CQ the host reads it is a host
 * page mapped for the device, as the receive ring itself is: the device's
 * store is a posted PCIe write, and the host poll reads its own memory, which
 * the write keeps coherent. For a CQ the device reads it is device memory,
 * which the device-side read finds through the CQ's QP map once the CQ is
 * exported. A queue with no CQ bound reports its completions nowhere and needs
 * none.
 *
 * Completions can come back out of order, so a table also needs a count of
 * the completions consumed at each slot, which a poster waits on before it
 * reuses a slot's entry (see efa_xpu_wq_wait_slot()). For a CQ the device
 * reads, the counts are device memory of their own, allocated here and
 * advanced by the device CQ read; for a CQ the host reads,
 * they share the done cursors' memory (efa_xpu_ep_done_setup()).
 */
static int efa_xpu_ep_ctx_setup(struct efa_xpu_ctx *xctx,
				struct efa_base_ep *base_ep,
				struct efa_xpu_ep_state *state,
				struct efa_xpu_ep *h_ep, bool sq_req_id_64)
{
	struct util_cq *ucq[EFA_XPU_DONE_CNT] = { base_ep->util_ep.tx_cq,
						  base_ep->util_ep.rx_cq };
	struct efa_xpu_wq *wq[EFA_XPU_DONE_CNT] = { &h_ep->sq, &h_ep->rq };
	bool need[EFA_XPU_DONE_CNT] = { !sq_req_id_64, h_ep->rq.buf != NULL };
	size_t page = ofi_get_page_size();
	int i, ret;

	for (i = 0; i < EFA_XPU_DONE_CNT; i++)
		state->queue_mask[i] = wq[i]->queue_mask;

	for (i = 0; i < EFA_XPU_DONE_CNT; i++) {
		uint32_t entries = wq[i]->queue_mask + 1;
		size_t size = (size_t) entries * sizeof(uint64_t);
		struct efa_cq *cq;
		void *dev = NULL;

		if (!need[i] || !ucq[i])
			continue;

		if (entries > (1u << 16)) {
			EFA_WARN(FI_LOG_EP_CTRL,
				 "XPU queue %d of %u entries is too deep for a "
				 "16-bit request ID to name its slots\n", i,
				 entries);
			return -FI_EOPNOTSUPP;
		}

		cq = container_of(ucq[i], struct efa_cq, util_cq);
		if (!cq->xpu_state) {
			size = ofi_get_aligned_size(size, page);
			ret = ofi_memalign(&state->ctx_host[i], page, size);
			if (ret)
				return -FI_ENOMEM;
			memset(state->ctx_host[i], 0, size);
			ret = xpu_import(xctx, state->ctx_host[i], size,
					 FI_XPU_IMPORT_DEVICEMAP, &dev);
			if (ret) {
				free(state->ctx_host[i]);
				state->ctx_host[i] = NULL;
				return ret;
			}
			state->ctx_host_cq[i] = cq;
			efa_xpu_qp_set_ctx(base_ep, cq, i, state->ctx_host[i],
					   wq[i]->queue_mask);
		} else {
			size_t slot_size = (size_t) entries * sizeof(uint32_t);
			uint32_t *zero;

			ret = xpu_alloc(xctx, size, 64, 0, &dev, NULL, NULL);
			if (ret)
				return ret;
			state->ctx_dev[i] = dev;

			zero = calloc(1, slot_size);
			if (!zero)
				return -FI_ENOMEM;
			ret = xpu_alloc(xctx, slot_size, 64, 0,
					&state->slot_dev[i], NULL, NULL);
			if (!ret)
				ret = ofi_copy_to_hmem(xctx->attr.iface,
						       xctx->attr.device,
						       state->slot_dev[i], zero,
						       slot_size);
			free(zero);
			if (ret)
				return ret;
			wq[i]->slot_done = state->slot_dev[i];
		}
		wq[i]->ctx = dev;
	}

	return 0;
}

/*
 * Give the endpoint's queues done cursors.
 *
 * Where the device reads a queue's completions, its cursor is plain device
 * memory, which the device-side CQ read advances once the CQ's export has put
 * this QP in its map (see efa_cq_export_xpu()). Where the host reads them:
 *
 * The cursors go in device memory registered for host stores - gdrcopy, for
 * CUDA - which is what keeps a waiting poster's reads on the device. Where the
 * device memory cannot be registered, they go in a host page mapped for the
 * device instead, which a poster then reads across the bus, only when its
 * cached view says the queue is full. If neither is possible the queues stay
 * unbounded and the depth is the application's, as for a queue the device
 * completes itself.
 *
 * The depth is the endpoint's advertised one, so a CQ sized for the tx + rx
 * depths of the queues bound to it cannot overflow while posters wait.
 *
 * A poster waits for descriptors older than the depth, and only the ones older
 * than the staging limit are known to have been rung (see
 * efa_xpu_wq_wait_done()), so a queue whose staging limit is not below its
 * depth has the limit lowered to half the depth.
 *
 * A queue with a context table that the host's CQ reads also gets its
 * per-slot completion counts here, after the cursors in the same memory.
 * Those are not optional: without them a poster could overwrite the context
 * of a completion that came back late, so failing to place them fails the
 * export, where a queue without a table is only left unbounded.
 */
static int efa_xpu_ep_done_setup(struct efa_xpu_ctx *xctx,
				 struct efa_base_ep *base_ep,
				 struct efa_xpu_ep_state *state,
				 struct efa_xpu_ep *h_ep)
{
	struct efa_cq *cq[EFA_XPU_DONE_CNT];
	struct efa_xpu_wq *wq[EFA_XPU_DONE_CNT] = { &h_ep->sq, &h_ep->rq };
	uint32_t depth[EFA_XPU_DONE_CNT];
	bool bound[EFA_XPU_DONE_CNT], slots = false;
	size_t page = ofi_get_page_size();
	size_t size = 64;	/* the cursors, then the slot counts */
	void *dev = NULL, *zero;
	int i, ret;

	struct efa_cq *dcq[EFA_XPU_DONE_CNT];
	bool dbound[EFA_XPU_DONE_CNT];

	cq[EFA_XPU_SQ_DONE] = efa_xpu_host_cq(base_ep->util_ep.tx_cq);
	cq[EFA_XPU_RQ_DONE] = h_ep->rq.buf ?
			      efa_xpu_host_cq(base_ep->util_ep.rx_cq) : NULL;
	dcq[EFA_XPU_SQ_DONE] = efa_xpu_dev_cq(base_ep->util_ep.tx_cq);
	dcq[EFA_XPU_RQ_DONE] = h_ep->rq.buf ?
			       efa_xpu_dev_cq(base_ep->util_ep.rx_cq) : NULL;
	/*
	 * Wait against the depths the endpoint advertises - what the QP was
	 * created with and what an application sizes its CQs from - rather than
	 * the rings, which can hold more: a receive ring of 32768 entries on an
	 * endpoint whose rx_attr->size is 8192 would otherwise let a kernel run
	 * further ahead of its CQ than an application sized for tx + rx depth
	 * has room for, and the CQ would overflow.
	 */
	depth[EFA_XPU_SQ_DONE] = MIN(h_ep->sq_size,
				     (uint32_t) efa_base_ep_get_tx_pool_size(base_ep));
	depth[EFA_XPU_RQ_DONE] = MIN(h_ep->rq.queue_mask + 1,
				     (uint32_t) efa_base_ep_get_rx_pool_size(base_ep));

	/*
	 * Ringing more often than the hardware requires is always safe, so a
	 * queue that stages as many descriptors as it holds rings at least
	 * every half queue instead.
	 */
	for (i = 0; i < EFA_XPU_DONE_CNT; i++) {
		if ((cq[i] || dcq[i]) && depth[i] <= wq[i]->max_batch)
			wq[i]->max_batch = depth[i] / 2;
		/* A queue of one has nothing to bound. */
		bound[i] = cq[i] && wq[i]->max_batch;
		dbound[i] = dcq[i] && wq[i]->max_batch;

		if (state->ctx_host[i]) {
			uint32_t entries = wq[i]->queue_mask + 1;

			state->slot_cnt[i] = calloc(entries, sizeof(uint32_t));
			if (!state->slot_cnt[i])
				return -FI_ENOMEM;
			state->slot_mask[i] = wq[i]->queue_mask;
			state->slot_off[i] = size;
			size += ofi_get_aligned_size(entries * sizeof(uint32_t),
						     64);
			cq[i] = state->ctx_host_cq[i];
			slots = true;
		} else if (!bound[i]) {
			cq[i] = NULL;
		}
	}
	if (dbound[EFA_XPU_SQ_DONE] || dbound[EFA_XPU_RQ_DONE]) {
		uint32_t dzero[EFA_XPU_DONE_CNT] = { 0 };

		ret = xpu_alloc(xctx, 64, 64, 0, &state->ddone_dev, NULL, NULL);
		if (!ret)
			ret = ofi_copy_to_hmem(xctx->attr.iface,
					       xctx->attr.device,
					       state->ddone_dev, dzero,
					       sizeof(dzero));
		if (ret) {
			EFA_WARN(FI_LOG_EP_CTRL,
				 "no device memory for the XPU done cursors: "
				 "%d\n", ret);
			return ret;
		}
		for (i = 0; i < EFA_XPU_DONE_CNT; i++) {
			if (!dbound[i])
				continue;
			wq[i]->done = (uint32_t *) state->ddone_dev + i;
			wq[i]->depth = depth[i];
			wq[i]->done_seen = 0;
			state->dev_cq[i] = dcq[i];
		}
	}

	if (!cq[EFA_XPU_SQ_DONE] && !cq[EFA_XPU_RQ_DONE])
		return 0;

	zero = calloc(1, size);
	if (!zero)
		return -FI_ENOMEM;
	ret = xpu_alloc(xctx, size, 64, 0, &dev, NULL, NULL);
	if (!ret)
		ret = ofi_copy_to_hmem(xctx->attr.iface, xctx->attr.device,
				       dev, zero, size);
	free(zero);
	if (!ret)
		ret = ofi_hmem_dev_register(xctx->attr.iface, dev, size,
					    &state->done_handle);
	if (!ret) {
		state->done_registered = true;
		state->done_dev = dev;
	} else {
		if (dev)
			xpu_free(xctx, dev);
		dev = NULL;

		size = ofi_get_aligned_size(size, page);
		ret = ofi_memalign(&state->done_host, page, size);
		if (!ret) {
			memset(state->done_host, 0, size);
			ret = xpu_import(xctx, state->done_host, size,
					 FI_XPU_IMPORT_DEVICEMAP, &dev);
		}
		if (ret) {
			free(state->done_host);
			state->done_host = NULL;
			if (slots) {
				EFA_WARN(FI_LOG_EP_CTRL,
					 "no device memory for the XPU slot "
					 "counts: %d\n", ret);
				return ret;
			}
			EFA_WARN(FI_LOG_EP_CTRL,
				 "no device memory for the XPU done cursors: "
				 "%d; queues left unbounded\n", ret);
			return 0;
		}
		state->done_dev = dev;
	}

	for (i = 0; i < EFA_XPU_DONE_CNT; i++) {
		if (state->slot_cnt[i])
			wq[i]->slot_done = (uint32_t *)
				((char *) state->done_dev + state->slot_off[i]);
		if (!bound[i])
			continue;
		wq[i]->done = (uint32_t *) state->done_dev + i;
		wq[i]->depth = depth[i];
		wq[i]->done_seen = 0;
	}

	/*
	 * Start publishing under the CQs' lock, so a read in progress never
	 * sees half of it.
	 */
	efa_xpu_done_lock(cq, true);
	for (i = 0; i < EFA_XPU_DONE_CNT; i++) {
		if (!cq[i])
			continue;
		state->done_bound[i] = bound[i];
		state->done_cq[i] = cq[i];
		cq[i]->xpu_publishers++;
	}
	efa_xpu_done_lock(cq, false);
	return 0;
}

/*
 * Release the endpoint's XPU state, first taking it out of reach of a host CQ
 * read that may be publishing through it.
 */
void efa_xpu_ep_detach(struct efa_base_ep *base_ep)
{
	struct efa_xpu_ep_state *state = base_ep->xpu_state;
	struct efa_cq *cq[EFA_XPU_DONE_CNT];
	int i;

	if (!state)
		return;

	memcpy(cq, state->done_cq, sizeof(cq));
	efa_xpu_done_lock(cq, true);
	for (i = 0; i < EFA_XPU_DONE_CNT; i++) {
		if (cq[i])
			cq[i]->xpu_publishers--;
		state->done_cq[i] = NULL;
	}
	base_ep->xpu_state = NULL;
	efa_xpu_done_lock(cq, false);

	/*
	 * Take the context tables out of the host poll's reach before they are
	 * freed. A device CQ's map is not touched: it is a snapshot the
	 * application rebuilds by exporting the CQ again, which it must do
	 * before the device reads the CQ after an endpoint bound to it closes.
	 */
	for (i = 0; i < EFA_XPU_DONE_CNT; i++) {
		if (state->ctx_host[i] && base_ep->qp)
			efa_xpu_qp_set_ctx(base_ep, state->ctx_host_cq[i], i,
					   NULL, 0);
	}

	efa_xpu_ep_state_destroy(state);
}

void efa_xpu_cq_state_destroy(struct efa_xpu_cq_state *state)
{
	struct efa_xpu_ctx *xctx;

	if (!state)
		return;

	xctx = container_of(state->xpu_ctx, struct efa_xpu_ctx, ctx);

	if (state->dev_cq)
		xpu_free(xctx, state->dev_cq);

	if (state->cq_buf_dev)
		xpu_free(xctx, state->cq_buf_dev);

	if (state->qp_map_dev)
		xpu_free(xctx, state->qp_map_dev);
	free(state->qp_map_host);

	free(state);
}

void efa_xpu_cntr_state_destroy(struct efa_xpu_cntr_state *state)
{
	struct efa_xpu_ctx *xctx;

	if (!state)
		return;

	xctx = container_of(state->xpu_ctx, struct efa_xpu_ctx, ctx);

	if (state->dev_cntr)
		xpu_free(xctx, state->dev_cntr);

	if (state->cntr_alloc_addr)
		xpu_free(xctx, state->cntr_alloc_addr);
	if (state->cntr_err_alloc_addr)
		xpu_free(xctx, state->cntr_err_alloc_addr);

	free(state);
}

/* ----------------------------------------------------------------
 * EP export
 * ---------------------------------------------------------------- */

int efa_ep_export_xpu(struct fid_ep *ep, uint64_t flags,
		      struct fid_xpu_ep *xpu_ep)
{
	struct efa_base_ep *base_ep;
	struct efa_xpu_ctx *xctx;
	struct efa_xpu_ep *dev_ep;
	struct fid_xpu_ctx *fid_ctx;
	struct efa_xpu_ep_state *state;
	void *dev_buf = NULL;
	int ret;

	if (!ep || !xpu_ep)
		return -FI_EINVAL;

	base_ep = container_of(ep, struct efa_base_ep, util_ep.ep_fid);

	/*
	 * Only an endpoint opened for an XPU has anything to export. It was
	 * opened with fi_endpoint2(FI_XPU), which recorded the flag, and with
	 * the context in info->ep_attr->xpu_ctx (not the endpoint context
	 * argument), preserved in base_ep->info by efa_base_ep_construct().
	 */
	if (!(base_ep->util_ep.flags & FI_XPU))
		return -FI_EINVAL;

	fid_ctx = (base_ep->info && base_ep->info->ep_attr) ?
		  base_ep->info->ep_attr->xpu_ctx : NULL;
	if (!fid_ctx)
		return -FI_EINVAL;

	xctx = container_of(fid_ctx, struct efa_xpu_ctx, ctx);

#if HAVE_EFADV_QUERY_QP_WQS
	{
		struct efadv_wq_attr qp_sq_attr = {0};
		struct efadv_wq_attr qp_rq_attr = {0};
		struct efa_xpu_ep h_ep = {0};
		bool sq_req_id_64;

		if (!base_ep->qp || !base_ep->qp->ibv_qp)
			return -FI_EINVAL;

		/*
		 * fi_getinfo() refuses FI_XPU when the direct data path is off,
		 * but the endpoint is what actually decides whether this QP uses
		 * it, so check the QP rather than trust the negotiation. Handing
		 * the rings to a kernel while the host posts through rdma-core
		 * would have both halves driving the same send queue by
		 * different rules.
		 */
		if (!base_ep->qp->data_path_direct_enabled) {
			EFA_WARN(FI_LOG_EP_CTRL,
				 "FI_XPU requires an endpoint using EFA's direct "
				 "data path\n");
			return -FI_EOPNOTSUPP;
		}

		/* Query hardware queue geometry for this QP. */
		ret = efadv_query_qp_wqs(base_ep->qp->ibv_qp, &qp_sq_attr,
					 &qp_rq_attr, sizeof(qp_sq_attr));
		if (ret) {
			EFA_WARN(FI_LOG_EP_CTRL,
				 "efadv_query_qp_wqs failed: %d\n", ret);
			return (ret == EOPNOTSUPP) ? -FI_EOPNOTSUPP : -FI_EINVAL;
		}

		/*
		 * Where the send queue has 64-bit request IDs, a device-built
		 * WQE carries the application's context in its request ID, as a
		 * host-built one does. Without them - the device may not offer
		 * the feature, or FI_EFA_USE_SQ_REQ_ID_64_BIT may have turned it
		 * off, and the host data path follows the same knob - the
		 * request ID is 16 bits, so it carries the send queue slot and
		 * the context goes in a context table, as for the receive queue
		 * (efa_xpu_ep_ctx_setup()).
		 */
#if HAVE_EFADV_WQ_ATTR_CAPS
		sq_req_id_64 = (qp_sq_attr.caps & EFADV_WQ_CAPS_64_BIT_REQ_ID) &&
			       efa_env.use_sq_req_id_64_bit;
#else
		sq_req_id_64 = false;
#endif

		/*
		 * Re-export: release the descriptor, the queue mappings and the
		 * done cursors the previous export took.
		 */
		efa_xpu_ep_detach(base_ep);

		state = efa_xpu_ep_state_create(fid_ctx);
		if (!state)
			return -FI_ENOMEM;

		/*
		 * Allocate the device-side EP state struct on the XPU. It is
		 * plain device memory that only the XPU kernel reads, so it
		 * needs no dmabuf fd - unlike the CQ ring and the counters,
		 * which the NIC itself writes. dev_buf must NOT be dereferenced
		 * from the host; we build the contents in the host struct h_ep
		 * and copy it to the device below.
		 */
		ret = xpu_alloc(xctx, sizeof(struct efa_xpu_ep), 64, 0,
				&dev_buf, NULL, NULL);
		if (ret)
			goto err_destroy_state;
		state->dev_ep = dev_buf;
		dev_ep = (struct efa_xpu_ep *)dev_buf;

		/* Populate the XPU header (host side). prov_ctx points at the
		 * device-accessible copy of this struct. The version lets a
		 * kernel built against a newer header refuse, or adapt to, a
		 * handle exported by an older library. */
		h_ep.version = fi_version();

		/* SQ geometry */
		h_ep.sq.queue_mask = qp_sq_attr.num_entries - 1;
		h_ep.sq.queue_size_shift = __builtin_ctz(qp_sq_attr.num_entries);
		h_ep.sq.max_batch = qp_sq_attr.max_batch;
		h_ep.sq.entry_size = qp_sq_attr.entry_size;
		h_ep.sq.pc = 0;
		h_ep.sq.released = 0;
		h_ep.sq.db_rung = 0;
		h_ep.sq.init_phase = 0;
		h_ep.sq_size = qp_sq_attr.num_entries;

		/* RQ geometry */
		if (qp_rq_attr.num_entries) {
			h_ep.rq.queue_mask = qp_rq_attr.num_entries - 1;
			h_ep.rq.queue_size_shift =
				__builtin_ctz(qp_rq_attr.num_entries);
			h_ep.rq.max_batch = qp_rq_attr.max_batch;
			h_ep.rq.entry_size = qp_rq_attr.entry_size;
			h_ep.rq.pc = 0;
			h_ep.rq.released = 0;
			h_ep.rq.db_rung = 0;
			h_ep.rq.init_phase = 0;
		}

		/*
		 * Map the SQ ring buffer (BAR MMIO) into device-accessible
		 * memory so the XPU kernel can build WQEs directly.
		 */
		{
			void *sq_buf_dev = NULL;

			ret = xpu_import(xctx, qp_sq_attr.buffer,
					 (size_t) qp_sq_attr.num_entries *
						 qp_sq_attr.entry_size,
					 FI_XPU_IMPORT_IOMEMORY |
						 FI_XPU_IMPORT_DEVICEMAP,
					 &sq_buf_dev);
			if (ret) {
				EFA_WARN(FI_LOG_EP_CTRL,
					 "map SQ buffer failed: %d\n", ret);
				goto err_destroy_state;
			}
			state->imported[state->imported_cnt++] =
				qp_sq_attr.buffer;
			h_ep.sq.buf = (uint8_t *) sq_buf_dev;
		}

		/*
		 * Map the SQ doorbell (BAR MMIO). Register only the 4-byte
		 * doorbell register: the doorbell VA is not page-aligned and
		 * registering a full page would cross into an unmapped page.
		 */
		{
			void *sq_db_dev = NULL;

			ret = xpu_import(xctx, qp_sq_attr.doorbell,
					 sizeof(uint32_t),
					 FI_XPU_IMPORT_IOMEMORY |
						 FI_XPU_IMPORT_DEVICEMAP,
					 &sq_db_dev);
			if (ret) {
				EFA_WARN(FI_LOG_EP_CTRL,
					 "map SQ doorbell failed: %d\n", ret);
				goto err_destroy_state;
			}
			state->imported[state->imported_cnt++] =
				qp_sq_attr.doorbell;
			h_ep.sq.db = (uint32_t *) sq_db_dev;
		}

		/* Map RQ buffer (host RAM) and doorbell (BAR MMIO) if present. */
		if (qp_rq_attr.buffer) {
			void *rq_buf_dev = NULL;
			void *rq_db_dev = NULL;

			ret = xpu_import(xctx, qp_rq_attr.buffer,
					 (size_t) qp_rq_attr.num_entries *
						 qp_rq_attr.entry_size,
					 FI_XPU_IMPORT_DEVICEMAP, &rq_buf_dev);
			if (ret) {
				EFA_WARN(FI_LOG_EP_CTRL,
					 "map RQ buffer failed: %d\n", ret);
				goto err_destroy_state;
			}
			state->imported[state->imported_cnt++] =
				qp_rq_attr.buffer;
			h_ep.rq.buf = (uint8_t *) rq_buf_dev;

			ret = xpu_import(xctx, qp_rq_attr.doorbell,
					 sizeof(uint32_t),
					 FI_XPU_IMPORT_IOMEMORY |
						 FI_XPU_IMPORT_DEVICEMAP,
					 &rq_db_dev);
			if (ret) {
				EFA_WARN(FI_LOG_EP_CTRL,
					 "map RQ doorbell failed: %d\n", ret);
				goto err_destroy_state;
			}
			state->imported[state->imported_cnt++] =
				qp_rq_attr.doorbell;
			h_ep.rq.db = (uint32_t *) rq_db_dev;
		}

		ret = efa_xpu_ep_ctx_setup(xctx, base_ep, state, &h_ep,
					   sq_req_id_64);
		if (ret)
			goto err_destroy_state;

		ret = efa_xpu_ep_done_setup(xctx, base_ep, state, &h_ep);
		if (ret)
			goto err_destroy_state;

		/*
		 * A device-side CQ read finds this QP only through a map built
		 * when the CQ is exported, so a CQ exported before this
		 * endpoint has to be exported again.
		 */
		for (int q = 0; q < EFA_XPU_DONE_CNT; q++) {
			if (state->dev_cq[q] &&
			    state->dev_cq[q]->xpu_state->qp_map_dev)
				EFA_WARN(FI_LOG_EP_CTRL,
					 "XPU CQ exported before this endpoint: "
					 "export it again, or its device-side "
					 "reads will not account for QP %u\n",
					 base_ep->qp->qp_num);
		}

		/* Transfer the fully-populated host struct to device memory. */
		ret = ofi_copy_to_hmem(xctx->attr.iface, xctx->attr.device,
				       dev_ep, &h_ep, sizeof(h_ep));
		if (ret) {
			EFA_WARN(FI_LOG_EP_CTRL,
				 "ofi_copy_to_hmem failed for XPU ep: %d\n",
				 ret);
			goto err_destroy_state;
		}

		xpu_ep->fid.fclass = FI_CLASS_EP;
		xpu_ep->fid.prov_id = FI_XPU_PROV_EFA;
		xpu_ep->fid.prov_ctx = (uint64_t)(uintptr_t)dev_ep;

		/* The endpoint owns these resources until it is closed. */
		base_ep->xpu_state = state;

		return FI_SUCCESS;

err_destroy_state:
		/* Detach rather than destroy: the done cursors may already be
		 * published through the endpoint's CQs. */
		base_ep->xpu_state = state;
		efa_xpu_ep_detach(base_ep);
		return ret;
	}
#else
	return -FI_EOPNOTSUPP;
#endif
}

/* ----------------------------------------------------------------
 * CQ export
 * ---------------------------------------------------------------- */

/*
 * Build the QP map of an XPU CQ: EFA_XPU_CQ_QP_MAP_SIZE rows indexed by the QP
 * number's low bits, filled from every endpoint bound to the CQ that has been
 * exported, and refused if two of those QPs would share a row. A row carries, for each of
 * the QP's queues that reports to this CQ, the done cursor a device-side read
 * advances and the context table and slot counts where the queue uses them.
 *
 * The map is a snapshot, and keeping it current is the application's: an
 * endpoint bound or exported after this has no row, and one closed after this
 * keeps its row, so after either the CQ has to be exported again before the
 * device reads it, and it is not to be read on the device while that happens.
 */
static int efa_xpu_cq_map_build(struct efa_cq *efa_cq)
	/* No lock analysis: the CQ's ep_list_lock is taken by hand. */
	OFI_TSA_NO_ANALYSIS
{
	struct efa_xpu_cq_state *cs = efa_cq->xpu_state;
	struct efa_xpu_ctx *xctx = container_of(cs->xpu_ctx, struct efa_xpu_ctx,
						ctx);
	struct fid_list_entry *item;
	struct efa_xpu_cq_qp *rows;
	size_t size = EFA_XPU_CQ_QP_MAP_SIZE * sizeof(*rows);
	int ret = 0;

	if (!cs->qp_map_dev) {
		void *dev = NULL;

		cs->qp_map_host = calloc(1, size);
		if (!cs->qp_map_host)
			return -FI_ENOMEM;
		ret = xpu_alloc(xctx, size, 64, 0, &dev, NULL, NULL);
		if (ret) {
			EFA_WARN(FI_LOG_CQ, "no device memory for the XPU CQ's "
				 "QP map: %d\n", ret);
			free(cs->qp_map_host);
			cs->qp_map_host = NULL;
			return ret;
		}
		cs->qp_map_dev = dev;
	}
	rows = cs->qp_map_host;

	ofi_genlock_lock(&efa_cq->util_cq.ep_list_lock);
	memset(rows, 0, size);
	dlist_foreach_container(&efa_cq->util_cq.ep_list, struct fid_list_entry,
				item, entry) {
		struct efa_base_ep *ep = container_of(item->fid,
						      struct efa_base_ep,
						      util_ep.ep_fid.fid);
		struct efa_xpu_ep_state *state = ep->xpu_state;
		struct util_cq *ucq[EFA_XPU_DONE_CNT] = { ep->util_ep.tx_cq,
							  ep->util_ep.rx_cq };
		struct efa_xpu_cq_qp *row;
		int q;

		if (!state || !ep->qp)
			continue;

		row = &rows[ep->qp->qp_num & (EFA_XPU_CQ_QP_MAP_SIZE - 1)];
		if (row->state == EFA_XPU_CQ_QP_LIVE &&
		    row->qpn != ep->qp->qp_num) {
			EFA_WARN(FI_LOG_CQ, "QP %u and QP %u share an XPU CQ map "
				 "row\n", row->qpn, ep->qp->qp_num);
			ret = -FI_EBUSY;
			break;
		}
		row->qpn = ep->qp->qp_num;
		row->state = EFA_XPU_CQ_QP_LIVE;

		for (q = 0; q < EFA_XPU_DONE_CNT; q++) {
			uint32_t *done = NULL;
			uint64_t *ctx = state->ctx_dev[q];
			uint32_t *slot = state->slot_dev[q];

			if (ucq[q] != &efa_cq->util_cq)
				continue;
			if (state->dev_cq[q] == efa_cq)
				done = (uint32_t *) state->ddone_dev + q;
			if (q == EFA_XPU_SQ_DONE) {
				row->sq_done = done;
				row->sq_ctx = ctx;
				row->sq_slot_done = slot;
				row->sq_mask = state->queue_mask[q];
			} else {
				row->rq_done = done;
				row->rq_ctx = ctx;
				row->rq_slot_done = slot;
				row->rq_mask = state->queue_mask[q];
			}
		}
	}
	ofi_genlock_unlock(&efa_cq->util_cq.ep_list_lock);
	if (ret)
		return ret;

	return ofi_copy_to_hmem(xctx->attr.iface, xctx->attr.device,
				cs->qp_map_dev, rows, size);
}

int efa_cq_export_xpu(struct fid_cq *cq, uint64_t flags,
		      struct fid_xpu_cq *xpu_cq)
{
	struct efa_cq *efa_cq;
	struct efa_xpu_ctx *xctx;
	struct efa_xpu_cq *dev_cq;
	struct fid_xpu_ctx *fid_ctx;
	void *dev_buf = NULL;
	int ret;

	if (!cq || !xpu_cq)
		return -FI_EINVAL;

	efa_cq = container_of(cq, struct efa_cq, util_cq.cq_fid);

	if (!efa_cq->xpu_state)
		return -FI_ENODATA;
	fid_ctx = efa_cq->xpu_state->xpu_ctx;
	if (!fid_ctx)
		return -FI_EINVAL;

	xctx = container_of(fid_ctx, struct efa_xpu_ctx, ctx);

#if HAVE_EFADV_QUERY_CQ
	{
		struct efadv_cq_attr efadv_attr = {0};
		struct efa_xpu_cq h_cq = {0};

		if (!efa_cq->xpu_state->cq_buf_dev)
			return -FI_ENODATA;

		ret = efadv_query_cq(ibv_cq_ex_to_cq(efa_cq->ibv_cq.ibv_cq_ex),
				     &efadv_attr, sizeof(efadv_attr));
		if (ret)
			return (ret == EOPNOTSUPP) ? -FI_EOPNOTSUPP : -FI_EINVAL;

		/*
		 * Allocate the device-side CQ descriptor. Plain device memory:
		 * only the ring buffer it points at, which the NIC writes, was
		 * allocated as a dmabuf.
		 */
		if (efa_cq->xpu_state->dev_cq) {
			/* Re-export: release the previous descriptor. */
			xpu_free(xctx, efa_cq->xpu_state->dev_cq);
			efa_cq->xpu_state->dev_cq = NULL;
		}

		ret = xpu_alloc(xctx, sizeof(struct efa_xpu_cq), 64, 0,
				&dev_buf, NULL, NULL);
		if (ret)
			return ret;
		efa_cq->xpu_state->dev_cq = dev_buf;
		dev_cq = (struct efa_xpu_cq *)dev_buf;

		/*
		 * Build the descriptor on the host. The CQ ring buffer is the
		 * device-resident buffer allocated at open time (EXT_MEM_DMABUF),
		 * so the XPU kernel polls the hardware CQ directly.
		 */
		h_cq.version = fi_version();
		h_cq.buf = (uint8_t *)efa_cq->xpu_state->cq_buf_dev;
		h_cq.entry_size = efadv_attr.entry_size;
		h_cq.queue_mask = efadv_attr.num_entries - 1;
		h_cq.queue_size_shift = __builtin_ctz(efadv_attr.num_entries);
		h_cq.cc = 0;
		h_cq.init_phase = 1;
		/*
		 * What a device poll writes into the caller's buffer, as
		 * opposed to entry_size above, which is what the NIC writes
		 * into the ring.
		 */
		h_cq.format = efa_cq->format;
		h_cq.user_entry_size = efa_cq->entry_size;
		/*
		 * The QP map, built from the endpoints bound to this CQ and
		 * exported by now (see efa_xpu_cq_map_build()).
		 */
		ret = efa_xpu_cq_map_build(efa_cq);
		if (ret) {
			xpu_free(xctx, dev_buf);
			efa_cq->xpu_state->dev_cq = NULL;
			return ret;
		}
		h_cq.qp_map = efa_cq->xpu_state->qp_map_dev;

		ret = ofi_copy_to_hmem(xctx->attr.iface, xctx->attr.device,
				       dev_cq, &h_cq, sizeof(h_cq));
		if (ret) {
			EFA_WARN(FI_LOG_CQ,
				 "ofi_copy_to_hmem failed for XPU cq: %d\n",
				 ret);
			xpu_free(xctx, dev_buf);
			efa_cq->xpu_state->dev_cq = NULL;
			return ret;
		}

		xpu_cq->fid.fclass = FI_CLASS_CQ;
		xpu_cq->fid.prov_id = FI_XPU_PROV_EFA;
		xpu_cq->fid.prov_ctx = (uint64_t)(uintptr_t)dev_cq;

		return FI_SUCCESS;
	}
#else
	return -FI_EOPNOTSUPP;
#endif
}

/* ----------------------------------------------------------------
 * Counter export
 * ---------------------------------------------------------------- */

int efa_cntr_export_xpu(struct fid_cntr *cntr, uint64_t flags,
			struct fid_xpu_cntr *xpu_cntr)
{
	struct efa_cntr *efa_cntr;
	struct efa_xpu_ctx *xctx;
	struct efa_xpu_cntr *dev_cntr;
	struct fid_xpu_ctx *fid_ctx;
	void *dev_buf = NULL;
	int ret;

	if (!cntr || !xpu_cntr)
		return -FI_EINVAL;

	efa_cntr = container_of(cntr, struct efa_cntr,
				util_cntr.cntr_fid);

	if (!efa_cntr->xpu_state || !efa_cntr->xpu_state->cntr_value_dev)
		return -FI_ENODATA;
	fid_ctx = efa_cntr->xpu_state->xpu_ctx;
	if (!fid_ctx)
		return -FI_EINVAL;

	xctx = container_of(fid_ctx, struct efa_xpu_ctx, ctx);

	{
		struct efa_xpu_cntr h_cntr = {0};

		/*
		 * Allocate the device-side counter descriptor. Plain device
		 * memory: only the counter values it points at, which the NIC
		 * writes, were allocated as dmabufs.
		 */
		if (efa_cntr->xpu_state->dev_cntr) {
			/* Re-export: release the previous descriptor. */
			xpu_free(xctx, efa_cntr->xpu_state->dev_cntr);
			efa_cntr->xpu_state->dev_cntr = NULL;
		}

		ret = xpu_alloc(xctx, sizeof(struct efa_xpu_cntr), 64, 0,
				&dev_buf, NULL, NULL);
		if (ret)
			return ret;
		efa_cntr->xpu_state->dev_cntr = dev_buf;
		dev_cntr = (struct efa_xpu_cntr *)dev_buf;

		/*
		 * Build the descriptor on the host. The counter values live in
		 * the device-resident buffers allocated at open time (EFADV
		 * external DMABUF); the NIC writes them and the XPU kernel reads
		 * them directly.
		 */
		h_cntr.version = fi_version();
		h_cntr.value = (volatile uint64_t *)
			efa_cntr->xpu_state->cntr_value_dev;
		h_cntr.err_value = (volatile uint64_t *)
			efa_cntr->xpu_state->cntr_err_dev;

		ret = ofi_copy_to_hmem(xctx->attr.iface, xctx->attr.device,
				       dev_cntr, &h_cntr, sizeof(h_cntr));
		if (ret) {
			EFA_WARN(FI_LOG_CNTR,
				 "ofi_copy_to_hmem failed for XPU cntr: %d\n",
				 ret);
			xpu_free(xctx, dev_buf);
			efa_cntr->xpu_state->dev_cntr = NULL;
			return ret;
		}

		xpu_cntr->fid.fclass = FI_CLASS_CNTR;
		xpu_cntr->fid.prov_id = FI_XPU_PROV_EFA;
		xpu_cntr->fid.prov_ctx = (uint64_t)(uintptr_t)dev_cntr;

		return FI_SUCCESS;
	}
}

#else /* !HAVE_EFA_DATA_PATH_DIRECT */

int efa_xpu_ctx_open(struct fid_domain *domain, struct fi_xpu_attr *attr,
		     struct fid_xpu_ctx **ctx, void *context)
{
	EFA_WARN(FI_LOG_DOMAIN,
		 "FI_XPU is not supported: libfabric was built against an "
		 "rdma-core without efadv_query_qp_wqs()/efadv_query_cq()\n");
	return -FI_EOPNOTSUPP;
}

int efa_ep_export_xpu(struct fid_ep *ep, uint64_t flags,
		      struct fid_xpu_ep *xpu_ep)
{
	return -FI_EOPNOTSUPP;
}

int efa_cq_export_xpu(struct fid_cq *cq, uint64_t flags,
		      struct fid_xpu_cq *xpu_cq)
{
	return -FI_EOPNOTSUPP;
}

int efa_cntr_export_xpu(struct fid_cntr *cntr, uint64_t flags,
			struct fid_xpu_cntr *xpu_cntr)
{
	return -FI_EOPNOTSUPP;
}

int efa_xpu_mem_alloc(struct fid_xpu_ctx *ctx, uint64_t size,
		      uint64_t alignment, uint64_t flags, void **addr, int *fd,
		      uint64_t *offset)
{
	return -FI_EOPNOTSUPP;
}

void efa_xpu_mem_free(struct fid_xpu_ctx *ctx, void *addr)
{
}

void efa_xpu_mem_put_fd(struct fid_xpu_ctx *ctx, int fd)
{
}

/*
 * An XPU context can never be created, so no object can ever hold XPU state:
 * the creates always fail and the destroys always get NULL.
 */
struct efa_xpu_ep_state *efa_xpu_ep_state_create(struct fid_xpu_ctx *ctx)
{
	return NULL;
}

struct efa_xpu_cq_state *efa_xpu_cq_state_create(struct fid_xpu_ctx *ctx)
{
	return NULL;
}

struct efa_xpu_cntr_state *efa_xpu_cntr_state_create(struct fid_xpu_ctx *ctx)
{
	return NULL;
}

void efa_xpu_ep_state_destroy(struct efa_xpu_ep_state *state)
{
}

void efa_xpu_ep_detach(struct efa_base_ep *base_ep)
{
	base_ep->xpu_state = NULL;
}

void efa_xpu_ep_note_done(struct efa_xpu_ep_state *state, int queue,
			  uint16_t req_id)
{
}

void efa_xpu_ep_publish_done(struct efa_xpu_ep_state *state, int queue)
{
}

void efa_xpu_cq_state_destroy(struct efa_xpu_cq_state *state)
{
}

void efa_xpu_cntr_state_destroy(struct efa_xpu_cntr_state *state)
{
}

#endif /* HAVE_EFA_DATA_PATH_DIRECT */
