/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2017 DataDirect Networks, Inc. All rights reserved.
 * Copyright (c) 2018,2020-2023,2026 Hewlett Packard Enterprise Development LP
 */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "ofi_util.h"
#include "cxip.h"
#include "fi_cxi_gpu_ext.h"

#define CXIP_DBG(...) _CXIP_DBG(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXIP_INFO(...) _CXIP_INFO(FI_LOG_EP_CTRL, __VA_ARGS__)

extern struct fi_ops_rma cxip_ep_rma_ops;
extern struct fi_ops_rma cxip_ep_rma_no_ops;
extern struct fi_ops_rma cxip_ep_rma_writedata_ops;

extern struct fi_ops_msg cxip_ep_msg_ops;
extern struct fi_ops_msg cxip_ep_msg_no_ops;
extern struct fi_ops_msg cxip_ep_msg_no_tx_ops;
extern struct fi_ops_msg cxip_ep_msg_no_rx_ops;
extern struct fi_ops_msg cxip_ep_msg_ops_no_writedata;
extern struct fi_ops_msg cxip_ep_msg_no_rx_ops_no_writedata;

extern struct fi_ops_tagged cxip_ep_tagged_ops;
extern struct fi_ops_tagged cxip_ep_tagged_no_ops;
extern struct fi_ops_tagged cxip_ep_tagged_no_tx_ops;
extern struct fi_ops_tagged cxip_ep_tagged_no_rx_ops;
extern struct fi_ops_tagged cxip_ep_tagged_ops_no_writedata;
extern struct fi_ops_tagged cxip_ep_tagged_no_rx_ops_no_writedata;

extern struct fi_ops_atomic cxip_ep_atomic_ops;
extern struct fi_ops_atomic cxip_ep_atomic_no_ops;

extern struct fi_ops_collective cxip_collective_ops;
extern struct fi_ops_collective cxip_collective_no_ops;

extern struct fi_ops_cm cxip_ep_cm_ops;
extern struct fi_ops_ep cxip_ep_ops;
extern struct fi_ops cxip_ep_fi_ops;
extern struct fi_ops_ep cxip_ctx_ep_ops;

/*
 * cxip_ep_cmdq() - Open a shareable TX or Target command queue.
 *
 * Caller must hold ep_obj->lock
 */
int cxip_ep_cmdq(struct cxip_ep_obj *ep_obj, bool transmit, uint32_t tclass,
		 struct cxi_eq *evtq, struct cxip_cmdq **cmdq)
{
	struct cxi_cq_alloc_opts cq_opts = {};
	struct cxip_cmdq **ep_obj_cmdq;
	ofi_atomic32_t *ep_obj_ref;
	int ret;
	size_t size;
	const struct fi_cxi_ki_cmdq_buf *ubuf = NULL;

	if (transmit) {
		ep_obj_cmdq = &ep_obj->txq;
		ep_obj_ref = &ep_obj->txq_ref;
		size = ep_obj->txq_size;
		/* Check for KI user-provided TX cmdq buffer */
		if (ep_obj->ki_hints.tx_cmdq.buf)
			ubuf = &ep_obj->ki_hints.tx_cmdq;
	} else {
		ep_obj_cmdq = &ep_obj->tgq;
		ep_obj_ref = &ep_obj->tgq_ref;
		size = ep_obj->tgq_size;
	}

	if (*ep_obj_cmdq) {
		ofi_atomic_inc32(ep_obj_ref);
		CXIP_DBG("Reusing %s base CMDQ: %p\n",
			 transmit ? "TX" : "RX", *ep_obj_cmdq);
		*cmdq = *ep_obj_cmdq;

		return FI_SUCCESS;
	}

	/* An IDC command can use up to 4 64 byte slots. */
	cq_opts.count = size * 4;
	cq_opts.flags = transmit ? CXI_CQ_IS_TX : 0;
	cq_opts.policy = cxip_env.cq_policy;

	/* Use KI user-provided buffer if available */
	if (ubuf) {
		cq_opts.count = ubuf->len / C_CQ_CMD_SIZE;
		CXIP_DBG("Using KI user-provided cmdq buffer: %p (%zu bytes)\n",
			 ubuf->buf, ubuf->len);
	}

	ret = cxip_cmdq_alloc(ep_obj->domain->lni, evtq, &cq_opts,
			      ep_obj->auth_key.vni, cxip_ofi_to_cxi_tc(tclass),
			      CXI_TC_TYPE_DEFAULT, ubuf, cmdq);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Unable to allocate CMDQ, ret: %d\n", ret);
		return ret;
	}
	*ep_obj_cmdq = *cmdq;
	ofi_atomic_inc32(ep_obj_ref);

	CXIP_DBG("Allocated %s CMDQ: %p CP: %u\n",
		 transmit ? "TX" : "RX", *cmdq, cq_opts.lcid);
	return ret;
}

/*
 * cxip_ep_cmdq_put() - Release reference to shareable TX or Target command
 * queue.
 *
 * Caller must hold ep_obj->lock.
 */
void cxip_ep_cmdq_put(struct cxip_ep_obj *ep_obj, bool transmit)
{
	struct cxip_cmdq **ep_obj_cmdq;
	ofi_atomic32_t *ep_obj_ref;

	if (transmit) {
		ep_obj_cmdq = &ep_obj->txq;
		ep_obj_ref = &ep_obj->txq_ref;
	} else {
		ep_obj_cmdq = &ep_obj->tgq;
		ep_obj_ref = &ep_obj->tgq_ref;
	}

	if (!ofi_atomic_dec32(ep_obj_ref)) {
		cxip_cmdq_free(*ep_obj_cmdq);

		CXIP_DBG("Freed %s CMDQ: %p\n",
			 transmit ? "TX" : "RX", *ep_obj_cmdq);
		*ep_obj_cmdq = NULL;
	} else {
		CXIP_DBG("Put %s CMDQ: %p\n",
			 transmit ? "TX" : "RX", *ep_obj_cmdq);
	}
}

static int cxip_ep_cm_getname(fid_t fid, void *addr, size_t *addrlen)
{
	struct cxip_ep *cxip_ep = container_of(fid, struct cxip_ep, ep.fid);
	size_t len;

	len = MIN(*addrlen, sizeof(struct cxip_addr));

	if (!cxip_ep->ep_obj->enabled)
		return -FI_EOPBADSTATE;

	CXIP_DBG("NIC: 0x%x PID: %u\n", cxip_ep->ep_obj->src_addr.nic,
		 cxip_ep->ep_obj->src_addr.pid);

	memcpy(addr, &cxip_ep->ep_obj->src_addr, len);
	*addrlen = sizeof(struct cxip_addr);

	return (len == sizeof(struct cxip_addr)) ? FI_SUCCESS : -FI_ETOOSMALL;
}

static int _join_collective(struct fid_ep *ep, const void *addr,
			    uint64_t flags, struct fid_mc **mc, void *context)
{
	struct fi_collective_addr *arg = (struct fi_collective_addr *)addr;

	return cxip_join_collective(ep, arg->coll_addr, arg->set,
				    flags, mc, context);
}

struct fi_ops_cm cxip_ep_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = fi_no_setname,
	.getname = cxip_ep_cm_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = fi_no_listen,
	.accept = fi_no_accept,
	.reject = fi_no_reject,
	.shutdown = fi_no_shutdown,
	.join = _join_collective,
};

/*
 * cxip_ep_progress() - Progress an endpoint. Libfabric API calls
 * that initiate/require progress call this routine to initiate
 * progress of an endpoint.
 */
void cxip_ep_progress(struct fid *fid)
{
	struct cxip_ep *ep = container_of(fid, struct cxip_ep, ep.fid);
	struct cxip_ep_obj *ep_obj = ep->ep_obj;

	if (ep_obj->enabled) {

		ofi_genlock_lock(&ep_obj->lock);
		ep_obj->rxc->ops.progress(ep_obj->rxc, false);
		ep_obj->txc->ops.progress(ep_obj->txc, false);
		cxip_ep_ctrl_progress_locked(ep_obj, false);
		ofi_genlock_unlock(&ep_obj->lock);
	}
}

/*
 * fi_ep_get_unexpected_msgs() - Get unexpected message information, exposed
 * via domain open ops.
 */
size_t cxip_ep_get_unexp_msgs(struct fid_ep *fid_ep,
			      struct fi_cq_tagged_entry *entry, size_t count,
			      fi_addr_t *src_addr, size_t *ux_count)
{
	struct cxip_ep *ep = container_of(fid_ep, struct cxip_ep, ep);
	size_t ret_count = 0;

	/* Synchronous implementation to return a snapshot of the unexpected
	 * message queue for the endpoint.
	 */
	if (!ux_count)
		return -FI_EINVAL;

	if (ep->ep_obj->rxc->state == RXC_DISABLED)
		return -FI_EOPBADSTATE;

	if (!ofi_recv_allowed(ep->rx_attr.caps)) {
		CXIP_WARN("FI_RECV not enabled\n");
		return -FI_EINVAL;
	}

	/* If in flow control, let that complete since
	 * on-loading could be in progress.
	 */
	if (ep->ep_obj->rxc->state != RXC_ENABLED &&
	    ep->ep_obj->rxc->state != RXC_ENABLED_SOFTWARE) {
		cxip_cq_progress(ep->ep_obj->rxc->recv_cq);
		return -FI_EAGAIN;
	}

	ofi_genlock_lock(&ep->ep_obj->lock);
	if (cxip_evtq_saturated(&ep->ep_obj->rxc->rx_evtq)) {
		RXC_DBG(ep->ep_obj->rxc, "Target HW EQ saturated\n");
		ofi_genlock_unlock(&ep->ep_obj->lock);

		return -FI_EAGAIN;
	}

	/* Fill in supplied memory with what can fit */
	ret_count = cxip_build_ux_entry_info(ep, entry, count, src_addr,
					     ux_count);
	ofi_genlock_unlock(&ep->ep_obj->lock);

	return ret_count;
}

/*
 * cxip_ep_flush_trig_reqs() - Free triggered request for the EP.
 */
void cxip_ep_flush_trig_reqs(struct cxip_ep_obj *ep_obj)
{
	ofi_genlock_lock(&ep_obj->lock);
	cxip_evtq_flush_trig_reqs(&ep_obj->txc->tx_evtq);
	ofi_genlock_unlock(&ep_obj->lock);
}

/*
 * cxip_txc_close() - close the TX side of endpoint object.
 */
void cxip_txc_close(struct cxip_ep *ep)
{
	struct cxip_txc *txc = ep->ep_obj->txc;

	if (txc->send_cq) {
		ofi_genlock_lock(&txc->send_cq->ep_list_lock);
		fid_list_remove2(&txc->send_cq->util_cq.ep_list,
				&txc->send_cq->util_cq.ep_list_lock,
				&ep->ep.fid);
		ofi_genlock_unlock(&txc->send_cq->ep_list_lock);

		ofi_atomic_dec32(&txc->send_cq->util_cq.ref);
	}

	if (txc->send_cntr) {
		fid_list_remove2(&txc->send_cntr->ctx_list,
				 &txc->send_cntr->lock,
				 &ep->ep.fid);
		ofi_atomic_dec32(&txc->send_cntr->ref);
	}

	if (txc->read_cntr) {
		fid_list_remove2(&txc->read_cntr->ctx_list,
				 &txc->read_cntr->lock,
				 &ep->ep.fid);
		ofi_atomic_dec32(&txc->read_cntr->ref);
	}

	if (txc->write_cntr) {
		fid_list_remove2(&txc->write_cntr->ctx_list,
				 &txc->write_cntr->lock,
				 &ep->ep.fid);
		ofi_atomic_dec32(&txc->write_cntr->ref);
	}

	cxip_domain_remove_txc(txc->domain, txc);

	cxip_txc_disable(txc);
}

/*
 * cxip_rxc_close() - close the RX side of the endpoint object.
 */
void cxip_rxc_close(struct cxip_ep *ep)
{
	struct cxip_rxc *rxc = ep->ep_obj->rxc;

	if (rxc->recv_cq) {
		/* EP FID may not be found in the list if recv_cq == send_cq,
		 * but we still need to decrement reference.
		 */
		ofi_genlock_lock(&rxc->recv_cq->ep_list_lock);
		fid_list_remove2(&rxc->recv_cq->util_cq.ep_list,
				&rxc->recv_cq->util_cq.ep_list_lock,
				&ep->ep.fid);
		ofi_genlock_unlock(&rxc->recv_cq->ep_list_lock);

		ofi_atomic_dec32(&rxc->recv_cq->util_cq.ref);
	}

	if (rxc->recv_cntr) {
		fid_list_remove2(&rxc->recv_cntr->ctx_list,
				 &rxc->recv_cntr->lock,
				 &ep->ep.fid);
		ofi_atomic_dec32(&rxc->recv_cntr->ref);
	}

	cxip_rxc_disable(rxc);
}

/**
 * Get TX/RX option flags.
 *
 * Support TX/RX context control(FI_GETOPSFLAG).
 *
 * @param tx_attr : TX attributes, or NULL
 * @param rx_attr : RX attributes, or NULL
 * @param flags : storage for returned flags
 *
 * @return int : 0 on success, -errno on failure
 */
int cxip_getopflags(struct fi_tx_attr *tx_attr, struct fi_rx_attr *rx_attr,
		    uint64_t *flags)
{
	if ((*flags & FI_TRANSMIT) && (*flags & FI_RECV)) {
		CXIP_WARN("Both Tx/Rx flags cannot be specified\n");
		return -FI_EINVAL;
	} else if (tx_attr && (*flags & FI_TRANSMIT)) {
		*flags = tx_attr->op_flags;
	} else if (rx_attr && (*flags & FI_RECV)) {
		*flags = rx_attr->op_flags;
	} else {
		CXIP_WARN("Tx/Rx flags not specified\n");
		return -FI_EINVAL;
	}

	return 0;
}

/**
 * Set TX/RX option flags.
 *
 * Support TX/RX control(FI_SETOPSFLAG).
 *
 * @param tx_attr : TX attributes, or NULL
 * @param rx_attr : RX attributes, or NULL
 * @param flags : flags to set
 *
 * @return int : 0 on success, -errno on failure
 */
int cxip_setopflags(struct fi_tx_attr *tx_attr, struct fi_rx_attr *rx_attr,
		    uint64_t flags)
{
	if ((flags & FI_TRANSMIT) && (flags & FI_RECV)) {
		CXIP_WARN("Both Tx/Rx flags cannot be specified\n");
		return -FI_EINVAL;
	} else if (tx_attr && (flags & FI_TRANSMIT)) {
		tx_attr->op_flags = flags;
		tx_attr->op_flags &= ~FI_TRANSMIT;
		if (!(flags & (FI_INJECT_COMPLETE | FI_TRANSMIT_COMPLETE |
			       FI_DELIVERY_COMPLETE)))
			tx_attr->op_flags |= FI_TRANSMIT_COMPLETE;
	} else if (rx_attr && (flags & FI_RECV)) {
		rx_attr->op_flags = flags;
		rx_attr->op_flags &= ~FI_RECV;
	} else {
		CXIP_WARN("Tx/Rx flags not specified\n");
		return -FI_EINVAL;
	}

	return 0;
}

/**
 * Cancel RX operation
 *
 * Support TX/RX context cancel().
 *
 * Searches the RX queue for a pending async operation with the specified
 * 'context', and cancels it if still pending.
 *
 * @param rxc : RX context to search
 * @param context : user context pointer to search for
 *
 * @return ssize_t : 0 on success, -errno on failure
 */
ssize_t cxip_rxc_cancel(struct cxip_rxc *rxc, void *context)
{
	if (rxc->state == RXC_DISABLED)
		return -FI_EOPBADSTATE;

	return cxip_evtq_req_cancel(&rxc->rx_evtq, rxc, context, true);
}

/**
 * Cancel TX operation
 *
 * Support TX/RX context cancel().
 *
 * Searches the TX queue for a pending async operation with the specified
 * 'context', and request it be canceled.
 *
 * @param rxc : TX context to search
 * @param context : user context pointer to search for
 *
 * @return ssize_t : 0 on success, -errno on failure
 */
ssize_t cxip_txc_cancel(struct cxip_txc *txc, void *context)
{
	struct cxip_req *req;
	struct dlist_entry *tmp;

	if (!txc->enabled)
		return -FI_EOPBADSTATE;

	if (!context)
		return -FI_ENOENT;

	/* Only messaging may be canceled at this time */
	dlist_foreach_container_safe(&txc->msg_queue, struct cxip_req, req,
				     send.txc_entry, tmp) {
		if (req->type != CXIP_REQ_SEND ||
		    req->context != (uint64_t)context)
			continue;
		return txc->ops.cancel_msg_send(req);
	}

	return -FI_ENOENT;
}

/*
 * cxip_ep_cancel() - Cancel TX/RX operation for EP.
 */
ssize_t cxip_ep_cancel(fid_t fid, void *context)
{
	struct cxip_ep *ep = container_of(fid, struct cxip_ep, ep.fid);
	ssize_t ret;

	/* TODO: Remove this since it requires malicious programming to
	 * create this condition.
	 */
	if (fid->fclass != FI_CLASS_EP)
		return -FI_EINVAL;

	if (!ofi_recv_allowed(ep->ep_obj->caps))
		return -FI_ENOENT;

	ofi_genlock_lock(&ep->ep_obj->lock);

	ret = cxip_rxc_cancel(ep->ep_obj->rxc, context);
	if (ret != -FI_ENOENT)
		goto out_unlock;

	ret = cxip_txc_cancel(ep->ep_obj->txc, context);

out_unlock:
	ofi_genlock_unlock(&ep->ep_obj->lock);

	return ret;
}

/*
 * cxip_ep_destroy_priv_wait - Free an internal wait channel for the EP.
 */
static void cxip_ep_destroy_priv_wait(struct cxip_ep_obj *ep_obj)
{
	assert(ep_obj->priv_wait);

	if (ep_obj->txc->send_cq && ep_obj->txc->send_cq->attr.wait_obj)
		cxip_cq_del_wait_fd(ep_obj->txc->send_cq, ep_obj->wait_fd);

	if (ep_obj->rxc->recv_cq && ep_obj->rxc->recv_cq->attr.wait_obj &&
	    ep_obj->rxc->recv_cq != ep_obj->txc->send_cq)
		cxip_cq_del_wait_fd(ep_obj->rxc->recv_cq, ep_obj->wait_fd);

	cxil_destroy_wait_obj(ep_obj->priv_wait);

	ep_obj->priv_wait = NULL;
	ep_obj->wait_fd = -1;
}

/*
 * cxip_ep_alloc_priv_wait - Allocate an internal wait channel for the EP.
 */
static int cxip_ep_alloc_priv_wait(struct cxip_ep_obj *ep_obj)
{
	bool tx_cq_added = false;
	int ret;

	assert(ep_obj->priv_wait == NULL);

	ret = cxil_alloc_wait_obj(ep_obj->domain->lni->lni, &ep_obj->priv_wait);
	if (ret) {
		CXIP_WARN("Alloc of EP internal wait object failed %d\n",
			  ret);
		return ret;
	}

	ep_obj->wait_fd = cxil_get_wait_obj_fd(ep_obj->priv_wait);
	ret = fi_fd_nonblock(ep_obj->wait_fd);
	if (ret) {
		CXIP_WARN("Unable to set EP wait non-blocking mode: %d\n", ret);
		goto destroy_wait;
	}

	if (ep_obj->txc->send_cq && ep_obj->txc->send_cq->attr.wait_obj) {
		ret = cxip_cq_add_wait_fd(ep_obj->txc->send_cq, ep_obj->wait_fd,
					  EPOLLPRI | POLLERR);
		if (ret)
			goto destroy_wait;

		tx_cq_added = true;
	}

	if (ep_obj->rxc->recv_cq && ep_obj->rxc->recv_cq->attr.wait_obj &&
	    ep_obj->rxc->recv_cq != ep_obj->txc->send_cq) {
		ret = cxip_cq_add_wait_fd(ep_obj->rxc->recv_cq, ep_obj->wait_fd,
					  EPOLLPRI | POLLERR);
		if (ret) {
			if (tx_cq_added)
				cxip_cq_del_wait_fd(ep_obj->txc->send_cq,
						    ep_obj->wait_fd);
			goto destroy_wait;
		}
	}

	CXIP_DBG("Add EP private wait object, EP intr FD: %d\n",
		 ep_obj->wait_fd);

	return FI_SUCCESS;

destroy_wait:
	cxil_destroy_wait_obj(ep_obj->priv_wait);
	ep_obj->priv_wait = NULL;
	ep_obj->wait_fd = -1;

	return ret;
}

/*
 * cxip_ep_trywait() - Determine if hardware events are waiting to be processed
 * for EP based on CQ.
 */
int cxip_ep_trywait(struct cxip_ep_obj *ep_obj, struct cxip_cq *cq)
{
	struct cxip_txc_rnr *txc_rnr = container_of(ep_obj->txc,
						    struct cxip_txc_rnr, base);

	assert(ep_obj->priv_wait);

	ofi_genlock_lock(&ep_obj->lock);

	/* Keep progressing if RNR retry queues are not null */
	if (ep_obj->protocol == FI_PROTO_CXI_RNR &&
	    txc_rnr->next_retry_wait_us != UINT64_MAX &&
	    ofi_atomic_get32(&txc_rnr->time_wait_reqs))
		goto ready;

	cxil_clear_wait_obj(ep_obj->priv_wait);

	/* Enable any currently disabled EQ interrupts, if events are
	 * ready shortcut and return.
	 */
	if ((ep_obj->txc->send_cq == cq ||
	     ep_obj->rxc->recv_cq == cq) && ep_obj->txc->tx_evtq.eq) {
		cxi_eq_int_enable(ep_obj->txc->tx_evtq.eq);
		ep_obj->txc->tx_evtq.unacked_events = 0;

		if (cxi_eq_peek_event(ep_obj->txc->tx_evtq.eq))
			goto ready;
	}

	if (ep_obj->rxc->recv_cq == cq && ep_obj->rxc->rx_evtq.eq) {
		cxi_eq_int_enable(ep_obj->rxc->rx_evtq.eq);
		ep_obj->rxc->rx_evtq.unacked_events = 0;

		if (cxi_eq_peek_event(ep_obj->rxc->rx_evtq.eq))
			goto ready;
	}

	/* Side band control messages can also require progress */
	cxi_eq_int_enable(ep_obj->ctrl.tx_evtq);
	if (cxi_eq_peek_event(ep_obj->ctrl.tx_evtq))
		goto ready;

	cxi_eq_int_enable(ep_obj->ctrl.tgt_evtq);
	if (cxi_eq_peek_event(ep_obj->ctrl.tgt_evtq))
		goto ready;

	ofi_genlock_unlock(&ep_obj->lock);

	return FI_SUCCESS;

ready:
	ofi_genlock_unlock(&ep_obj->lock);

	return -FI_EAGAIN;
}

/*
 * cxip_ep_enable() - Enable standard EP.
 */
static int cxip_ep_enable(struct fid_ep *fid_ep)
{
	struct cxip_ep *ep = container_of(fid_ep, struct cxip_ep, ep);
	struct cxip_ep_obj *ep_obj = ep->ep_obj;
	int ret = FI_SUCCESS;

	ofi_genlock_lock(&ep_obj->lock);
	if (ep_obj->enabled)
		goto unlock;

	/* Allocate an EP internal wait object if a CQ is bound with a
	 * wait object specified.
	 */
	if ((ep_obj->txc->send_cq && ep_obj->txc->send_cq->attr.wait_obj) ||
	    (ep_obj->rxc->recv_cq && ep_obj->rxc->recv_cq->attr.wait_obj)) {
		ret = cxip_ep_alloc_priv_wait(ep_obj);
		if (ret) {
			CXIP_WARN("EP internal wait alloc failed %s\n",
				  fi_strerror(-ret));
			goto unlock;
		}
	}

	if (!ep_obj->av) {
		CXIP_WARN("Endpoint must be bound to an AV\n");
		ret = -FI_ENOAV;
		goto free_wait;
	}

	assert(ep_obj->domain->enabled);

	/* src_addr.pid may be C_PID_ANY at this point. */
	if (ep_obj->av_auth_key) {
		ret = cxip_av_auth_key_get_vnis(ep_obj->av, &ep_obj->vnis,
						&ep_obj->vni_count);
		if (ret)
			goto free_wait;

		ret = cxip_portals_table_alloc(ep_obj->domain->lni,
					       ep_obj->vnis, ep_obj->vni_count,
					       ep_obj->src_addr.pid,
					       &ep_obj->ptable);
		if (ret != FI_SUCCESS) {
			CXIP_WARN("Failed to allocate auth key ring portals table: %d\n",
				  ret);
			goto free_vnis;
		}

		/* This is unfortunately needed to allocate a command queue.
		 * But, this can be changed later.
		 */
		ep_obj->auth_key.vni = ep_obj->vnis[0];
	} else {
		ret = cxip_portals_table_alloc(ep_obj->domain->lni,
					       &ep_obj->auth_key.vni, 1,
					       ep_obj->src_addr.pid,
					       &ep_obj->ptable);
		if (ret != FI_SUCCESS) {
			CXIP_WARN("Failed to allocate portals table: %d\n",
				  ret);
			goto free_wait;
		}
	}

	ep_obj->src_addr.pid = ep_obj->ptable->pid;

	ret = cxip_ep_ctrl_init(ep_obj);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("cxip_ep_ctrl_init returned: %d\n", ret);
		goto free_portals_table;
	}

	ret = cxip_zbcoll_init(ep_obj);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("cxip_zbcoll_init returned: %d\n", ret);
		goto free_ep_ctrl;
	}

	CXIP_DBG("EP assigned NIC: %#x VNI: %u PID: %u\n",
		 ep_obj->src_addr.nic,
		 ep_obj->auth_key.vni,
		 ep_obj->src_addr.pid);

	ret = cxip_txc_enable(ep_obj->txc);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("cxip_txc_enable returned: %d\n", ret);
		goto free_zbcoll;
	}

	/* Allocate KI triggered cmdq if user provided a buffer. */
	if (ep_obj->ki_hints.trig_cmdq.buf) {
		const struct fi_cxi_ki_cmdq_buf *ubuf =
			&ep_obj->ki_hints.trig_cmdq;
		struct cxi_cq_alloc_opts cq_opts = {};

		cq_opts.count = ubuf->len / C_CQ_CMD_SIZE;
		cq_opts.flags = CXI_CQ_IS_TX | CXI_CQ_TX_WITH_TRIG_CMDS;
		cq_opts.policy = CXI_CQ_UPDATE_ALWAYS;

		ret = cxip_cmdq_alloc(ep_obj->domain->lni,
				      ep_obj->txc->tx_evtq.eq, &cq_opts,
				      ep_obj->auth_key.vni,
				      cxip_ofi_to_cxi_tc(ep_obj->tx_attr.tclass),
				      CXI_TC_TYPE_DEFAULT, ubuf,
				      &ep_obj->ki_trig_cmdq);
		if (ret != FI_SUCCESS) {
			CXIP_WARN("Failed to allocate KI triggered cmdq: %d\n",
				  ret);
			/* Match the TX cmdq path (cxip_ep_ctrl_init). */
			ret = -FI_EDOMAIN;
			goto disable_txc;
		}
		CXIP_DBG("Allocated KI triggered cmdq: %p (buf=%p, len=%zu, eq=%p eqn=%u)\n",
			 ep_obj->ki_trig_cmdq, ubuf->buf, ubuf->len,
			 ep_obj->txc->tx_evtq.eq,
			 ep_obj->txc->tx_evtq.eq ?
			 cxip_evtq_eqn(&ep_obj->txc->tx_evtq) : 0);
	}

	ret = cxip_rxc_enable(ep_obj->rxc);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("cxip_rxc_enable returned: %d\n", ret);
		goto free_ki_trig;
	}

	ret = cxip_coll_enable(ep);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("cxip_coll_enable returned: %d\n", ret);
		/* collectives will not function, but EP will */
	}

	/* Enable only appropriate API functions based on primary/secondary
	 * capabilities. Send/Receive requires FI_MSG or FI_TAGGED.
	 *
	 * For FI_TAGGED operations, check if writedata is supported.
	 * If tx_attr.caps includes FI_TAGGED and domain->msg_cq_data_size is non-zero,
	 * use ops table with fi_tagged_senddata/fi_tagged_injectdata implementations.
	 * Otherwise, use ops table with fi_no_tagged_senddata/fi_no_tagged_injectdata.
	 */
	if (ofi_send_allowed(ep->tx_attr.caps & ~FI_MSG) &&
	    ofi_recv_allowed(ep->rx_attr.caps & ~FI_MSG)) {
		if ((ep->tx_attr.caps & FI_TAGGED) && ep_obj->domain->msg_cq_data_size)
			ep->ep.tagged = &cxip_ep_tagged_ops;
		else
			ep->ep.tagged = &cxip_ep_tagged_ops_no_writedata;
	} else if (ofi_send_allowed(ep->tx_attr.caps & ~FI_MSG)) {
		if ((ep->tx_attr.caps & FI_TAGGED) && ep_obj->domain->msg_cq_data_size)
			ep->ep.tagged = &cxip_ep_tagged_no_rx_ops;
		else
			ep->ep.tagged = &cxip_ep_tagged_no_rx_ops_no_writedata;
	} else if (ofi_recv_allowed(ep->rx_attr.caps & ~FI_MSG)) {
		ep->ep.tagged = &cxip_ep_tagged_no_tx_ops;
	}

	/* For FI_MSG operations, check if writedata is supported.
	 * If tx_attr.caps includes FI_MSG and domain->msg_cq_data_size is non-zero,
	 * use ops table with fi_senddata/fi_injectdata implementations.
	 * Otherwise, use ops table with fi_no_msg_senddata/fi_no_msg_injectdata.
	 */
	if (ofi_send_allowed(ep->tx_attr.caps & ~FI_TAGGED) &&
	    ofi_recv_allowed(ep->rx_attr.caps & ~FI_TAGGED)) {
		if ((ep->tx_attr.caps & FI_MSG) && ep_obj->domain->msg_cq_data_size)
			ep->ep.msg = &cxip_ep_msg_ops;
		else
			ep->ep.msg = &cxip_ep_msg_ops_no_writedata;
	} else if (ofi_send_allowed(ep->tx_attr.caps & ~FI_TAGGED)) {
		if ((ep->tx_attr.caps & FI_MSG) && ep_obj->domain->msg_cq_data_size)
			ep->ep.msg = &cxip_ep_msg_no_rx_ops;
		else
			ep->ep.msg = &cxip_ep_msg_no_rx_ops_no_writedata;
	} else if (ofi_recv_allowed(ep->rx_attr.caps & ~FI_TAGGED)) {
		ep->ep.msg = &cxip_ep_msg_no_tx_ops;
	}

	/* Initiate requires FI_RMA or FI_ATOMIC */
	if (ofi_rma_initiate_allowed(ep->tx_attr.caps & ~FI_RMA))
		ep->ep.atomic = &cxip_ep_atomic_ops;

	if (ofi_rma_initiate_allowed(ep->tx_attr.caps & ~FI_ATOMIC)) {
		/* Select RMA ops variant. Enable writedata/injectdata only if:
		 *  - FI_RMA is present in tx_attr.caps
		 *  - domain has non-zero rma_cq_data_size (remote CQ data supported)
		 */
		if ((ep->tx_attr.caps & FI_RMA) && ep_obj->domain->rma_cq_data_size) {
			ep->ep.rma = &cxip_ep_rma_writedata_ops;
		} else {
			ep->ep.rma = &cxip_ep_rma_ops;
		}
	}

	/* An EP with a KI TX command queue buffer is driven by the GPU.
	 * Host data-transfer operations would share the ring and its
	 * write pointer with the GPU, so install the no-op tables.
	 */
	if (ep_obj->ki_hints.tx_cmdq.buf) {
		ep->ep.msg = &cxip_ep_msg_no_ops;
		ep->ep.tagged = &cxip_ep_tagged_no_ops;
		ep->ep.rma = &cxip_ep_rma_no_ops;
		ep->ep.atomic = &cxip_ep_atomic_no_ops;
	}

	ep_obj->enabled = true;
	goto unlock;

free_ki_trig:
	if (ep_obj->ki_trig_cmdq) {
		/* Before txc_disable: holds a ref on txc->tx_evtq */
		cxip_cmdq_free(ep_obj->ki_trig_cmdq);
		ep_obj->ki_trig_cmdq = NULL;
	}
disable_txc:
	cxip_txc_disable(ep_obj->txc);
free_zbcoll:
	cxip_zbcoll_fini(ep_obj);
free_ep_ctrl:
	cxip_ep_ctrl_fini(ep_obj);
free_portals_table:
	cxip_portals_table_free(ep_obj->ptable);
	ep_obj->ptable = NULL;
free_vnis:
	if (ep_obj->vnis) {
		cxip_av_auth_key_put_vnis(ep_obj->av, ep_obj->vnis,
					  ep_obj->vni_count);
		ep_obj->vnis = NULL;
	}
free_wait:
	if (ep_obj->priv_wait) {
		cxip_ep_destroy_priv_wait(ep_obj);
	}
unlock:
	ofi_genlock_unlock(&ep_obj->lock);
	return ret;
}

/*
 * cxip_ep_disable() - Disable the base EP if enabled.
 */
static void cxip_ep_disable(struct cxip_ep_obj *ep_obj)
{
	if (ep_obj->enabled) {
		cxip_coll_disable(ep_obj);
		cxip_zbcoll_fini(ep_obj);
		cxip_ep_ctrl_fini(ep_obj);
		cxip_portals_table_free(ep_obj->ptable);
		if (ep_obj->vnis) {
			cxip_av_auth_key_put_vnis(ep_obj->av, ep_obj->vnis,
						  ep_obj->vni_count);
			ep_obj->vnis = NULL;
		}
		ep_obj->ptable = NULL;
		ep_obj->enabled = false;
	}
}

/*
 * cxip_free_endpoint() - Release base EP object resources and free object.
 */
int cxip_free_endpoint(struct cxip_ep *ep)
{
	struct cxip_ep_obj *ep_obj = ep->ep_obj;
	int count;

	/* Each bound MR increments ref, so MRs must be removed.
	 */
	count = ofi_atomic_get32(&ep_obj->ref);
	if (count) {
		CXIP_WARN("EP refcount non-zero:%d\n", count);
		return -FI_EBUSY;
	}

	count = ofi_atomic_get32(&ep_obj->coll.num_mc);
	if (count) {
		CXIP_WARN("EP num_mc non-zero: %d\n", count);
		return -FI_EBUSY;
	}

	if (ep_obj->av)
		cxip_av_unbind_ep(ep_obj->av, ep);

	if (ep->ep_obj->eq) {
		ofi_mutex_lock(&ep_obj->eq->list_lock);
		dlist_remove(&ep_obj->eq_link);
		ofi_mutex_unlock(&ep_obj->eq->list_lock);
		ofi_atomic_dec32(&ep_obj->eq->util_eq.ref);
	}

	ofi_genlock_lock(&ep_obj->lock);
	/* The KI triggered command queue holds a reference on the TXC's
	 * TX event queue, so it must be destroyed before cxip_txc_close()
	 * destroys that event queue.
	 */
	if (ep_obj->ki_trig_cmdq) {
		cxip_cmdq_free(ep_obj->ki_trig_cmdq);
		ep_obj->ki_trig_cmdq = NULL;
	}
	cxip_coll_close(ep_obj);
	cxip_txc_close(ep);
	cxip_rxc_close(ep);
	cxip_ep_disable(ep_obj);
	if (ep_obj->priv_wait)
		cxip_ep_destroy_priv_wait(ep_obj);
	ofi_genlock_unlock(&ep_obj->lock);

	ofi_atomic_dec32(&ep_obj->domain->ref);
	ofi_genlock_destroy(&ep_obj->lock);

	cxip_txc_free(ep_obj->txc);
	cxip_rxc_free(ep_obj->rxc);

	free(ep_obj);
	ep->ep_obj = NULL;

	return FI_SUCCESS;
}

/*
 * cxip_ep_close() - Close (destroy) the base EP.
 */
static int cxip_ep_close(struct fid *fid)
{
	struct cxip_ep *ep = container_of(fid, struct cxip_ep, ep.fid);
	int ret;
	int count;

	if (ep->is_alias) {
		count = ofi_atomic_get32(&ep->ep_obj->ref);

		if (count > 0) {
			ofi_atomic_dec32(&ep->ep_obj->ref);
			free(ep);
			return FI_SUCCESS;
		}

		CXIP_WARN("EP alias %p, invalid EP object refcnt %d\n",
			  ep, count);
		return -FI_EINVAL;
	}

	ret = cxip_free_endpoint(ep);
	if (ret) {
		CXIP_WARN("Unable to free EP object %d : %s\n",
			  ret, fi_strerror(-ret));
		return ret;
	}
	free(ep);

	return FI_SUCCESS;
}

/*
 * cxip_ep_bind_cq() - Bind the EP to the CQ resource.
 */
static int cxip_ep_bind_cq(struct cxip_ep *ep, struct cxip_cq *cq,
			   uint64_t flags)
{
	struct cxip_txc *txc;
	struct cxip_rxc *rxc;
	int ret;

	if (ep->ep_obj->domain != cq->domain) {
		CXIP_WARN("Invalid CQ domain for EP\n");
		return -FI_EINVAL;
	}

	if ((flags | CXIP_EP_CQ_FLAGS) != CXIP_EP_CQ_FLAGS) {
		CXIP_WARN("Invalid CQ flags\n");
		return -FI_EINVAL;
	}

	if (flags & FI_TRANSMIT) {
		txc = ep->ep_obj->txc;
		if (txc->send_cq) {
			CXIP_WARN("SEND CQ previously bound\n");
			return -FI_EINVAL;
		}

		ofi_atomic_inc32(&cq->util_cq.ref);
		txc->send_cq = cq;

		if (flags & FI_SELECTIVE_COMPLETION)
			txc->selective_completion = 1;
		if (!txc->selective_completion)
			txc->attr.op_flags |= FI_COMPLETION;

		ep->tx_attr.op_flags = txc->attr.op_flags;

		/* Use CXI ep_list_lock that can be selectively optimized */
		ofi_genlock_lock(&cq->ep_list_lock);
		ret = fid_list_insert2(&cq->util_cq.ep_list,
				      &cq->util_cq.ep_list_lock,
				      &ep->ep.fid);
		ofi_genlock_unlock(&cq->ep_list_lock);

		if (ret) {
			CXIP_WARN("EP CQ fid insert failed %d\n", ret);
			ofi_atomic_dec32(&cq->util_cq.ref);
			txc->send_cq = NULL;
		}
	}

	if (flags & FI_RECV) {
		rxc = ep->ep_obj->rxc;
		if (rxc->recv_cq) {
			CXIP_WARN("RECV CQ previously bound\n");
			return -FI_EINVAL;
		}

		ofi_atomic_inc32(&cq->util_cq.ref);
		rxc->recv_cq = cq;

		if (flags & FI_SELECTIVE_COMPLETION)
			rxc->selective_completion = 1;
		if (!rxc->selective_completion)
			rxc->attr.op_flags |= FI_COMPLETION;

		ep->rx_attr.op_flags = rxc->attr.op_flags;

		/* Use CXI ep_list_lock that can be selectively optimized */
		ofi_genlock_lock(&cq->ep_list_lock);
		ret = fid_list_insert2(&cq->util_cq.ep_list,
				      &cq->util_cq.ep_list_lock,
				      &ep->ep.fid);
		ofi_genlock_unlock(&cq->ep_list_lock);

		if (ret) {
			CXIP_WARN("EP CQ fid insert failed %d\n", ret);
			ofi_atomic_dec32(&cq->util_cq.ref);
			rxc->recv_cq = NULL;
		}
	}
	return FI_SUCCESS;
}

/*
 * cxip_ep_bind_cntr() - Bind EP to counter resource
 */
static int cxip_ep_bind_cntr(struct cxip_ep *ep, struct cxip_cntr *cntr,
			     uint64_t flags)
{
	int ret;

	if (ep->ep_obj->domain != cntr->domain) {
		CXIP_WARN("Counter domain invalid for EP\n");
		return -FI_EINVAL;
	}

	if (!(flags & CXIP_EP_CNTR_FLAGS))
		return FI_SUCCESS;

	if ((flags & FI_SEND && ep->ep_obj->txc->send_cntr) ||
	    (flags & FI_READ && ep->ep_obj->txc->read_cntr) ||
	    (flags & FI_WRITE && ep->ep_obj->txc->write_cntr) ||
	    (flags & FI_RECV && ep->ep_obj->rxc->recv_cntr)) {
		CXIP_WARN("EP previously bound to counter\n");
		return -FI_EINVAL;
	}

	bool use_ded_trig_cmdq = cxip_env.cntr_trig_cmdq &&
		cntr->domain->util_domain.threading != FI_THREAD_DOMAIN;

	if (use_ded_trig_cmdq) {
		ofi_genlock_lock(&cntr->lock);
		if (!cntr->trig_cmdq) {
			struct cxip_domain *dom = cntr->domain;
			struct cxi_cq_alloc_opts cq_opts = {
				.policy = CXI_CQ_UPDATE_ALWAYS,
			};
			cq_opts.count = MAX(dom->max_trig_op_in_use, 64);
			cq_opts.flags = CXI_CQ_IS_TX | CXI_CQ_TX_WITH_TRIG_CMDS;
			cq_opts.policy = CXI_CQ_UPDATE_ALWAYS;
			cxip_cmdq_alloc(dom->lni, NULL, &cq_opts,
					dom->auth_key.vni,
					cxip_ofi_to_cxi_tc(dom->tclass),
					CXI_TC_TYPE_DEFAULT, NULL,
					&cntr->trig_cmdq);
			if (cntr->trig_cmdq == NULL) {
				CXIP_WARN("Alloc of cntr trig_cmdq failed\n");
			}
		}
		ofi_genlock_unlock(&cntr->lock);
	}

	ret = fid_list_insert2(&cntr->ctx_list, &cntr->lock, &ep->ep.fid);
	if (ret) {
		CXIP_WARN("Add of EP to cntr EP list failed: %d:%s\n",
			  ret, fi_strerror(-ret));
		return ret;
	}

	if (flags & FI_SEND) {
		ep->ep_obj->txc->send_cntr = cntr;
		ofi_atomic_inc32(&cntr->ref);
	}
	if (flags & FI_READ) {
		ep->ep_obj->txc->read_cntr = cntr;
		ofi_atomic_inc32(&cntr->ref);
	}
	if (flags & FI_WRITE) {
		ep->ep_obj->txc->write_cntr = cntr;
		ofi_atomic_inc32(&cntr->ref);
	}
	if (flags & FI_RECV) {
		ep->ep_obj->rxc->recv_cntr = cntr;
		ofi_atomic_inc32(&cntr->ref);
	}

	return FI_SUCCESS;
}

/*
 * cxip_ep_bind() - Bind EP resources.
 */
int cxip_ep_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	int ret;
	struct cxip_ep *ep = container_of(fid, struct cxip_ep, ep.fid);
	struct cxip_eq *eq;
	struct cxip_cq *cq;
	struct cxip_av *av;
	struct cxip_cntr *cntr;

	/* TODO: Remove this since it requires malicious programming to
	 * create this condition.
	 */
	if (fid->fclass != FI_CLASS_EP)
		return -FI_EINVAL;

	ret = ofi_ep_bind_valid(&cxip_prov, bfid, flags);
	if (ret)
		return ret;

	switch (bfid->fclass) {
	case FI_CLASS_EQ:
		eq = container_of(bfid, struct cxip_eq, util_eq.eq_fid.fid);
		ofi_atomic_inc32(&eq->util_eq.ref);
		ofi_mutex_lock(&eq->list_lock);
		dlist_insert_tail(&ep->ep_obj->eq_link, &eq->ep_list);
		ofi_mutex_unlock(&eq->list_lock);
		ep->ep_obj->eq = eq;
		break;

	case FI_CLASS_CQ:
		cq = container_of(bfid, struct cxip_cq, util_cq.cq_fid.fid);
		ret = cxip_ep_bind_cq(ep, cq, flags);
		if (ret)
			return ret;
		break;

	case FI_CLASS_CNTR:
		cntr = container_of(bfid, struct cxip_cntr, cntr_fid.fid);
		ret = cxip_ep_bind_cntr(ep, cntr, flags);
		if (ret)
			return ret;
		break;

	case FI_CLASS_AV:
		av = container_of(bfid, struct cxip_av, av_fid.fid);
		ret = cxip_av_bind_ep(av, ep);
		if (ret)
			return ret;
		ep->ep_obj->av = av;

		break;

	case FI_CLASS_SRX_CTX:
		ep->ep_obj->owner_srx = ep->ep_obj->domain->owner_srx;
		break;

	default:
		return -FI_EINVAL;
	}

	return FI_SUCCESS;
}

/*
 * cxip_set_tclass()
 */
int cxip_set_tclass(uint32_t desired_tc, uint32_t default_tc, uint32_t *new_tc)
{
	assert(new_tc != NULL);

	if (desired_tc != FI_TC_UNSPEC) {
		if (desired_tc >= FI_TC_LABEL &&
		    desired_tc <= FI_TC_SCAVENGER) {
			*new_tc = desired_tc;
		} else {
			CXIP_WARN("Invalid tclass\n");
			return -FI_EINVAL;
		}
	} else {
		*new_tc = default_tc;
	}

	CXIP_DBG("Set tclass to %d\n", *new_tc);
	return FI_SUCCESS;
}

/**
 * provider fi_set_val()/FI_SET_VAL implementation for EP
 *
 * @param fid : EP fid
 * @param val : parameter structure for set value operations.
 *
 * @return int : 0 on success, -errno on failure
 */
static inline int cxip_ep_set_val(struct cxip_ep *cxi_ep,
				  struct fi_fid_var *val)
{
	struct cxip_txc_rnr *txc_rnr;
	uint64_t *req_order;
	uint64_t *req_rnr_max_time;
	uint32_t *req_tclass;
	uint32_t new_tclass = FI_TC_UNSPEC;

	if (!val->val)
		return -FI_EINVAL;

	switch (val->name) {
	case FI_OPT_CXI_SET_TCLASS:
		req_tclass = (uint32_t *) val->val;

		if (cxip_set_tclass(*req_tclass, cxi_ep->tx_attr.tclass,
				    &new_tclass))
			return -FI_EINVAL;

		cxi_ep->tx_attr.tclass = new_tclass;
		break;
	case FI_OPT_CXI_SET_MSG_ORDER:
		req_order = (uint64_t *) val->val;

		if (*req_order & ~CXIP_MSG_ORDER) {
			CXIP_WARN("Invalid message order 0x%" PRIx64 "\n",
				  *req_order);
			return -FI_EINVAL;
		}

		cxi_ep->tx_attr.msg_order = *req_order;
		break;
	case FI_OPT_CXI_SET_RNR_MAX_RETRY_TIME:
		req_rnr_max_time = (uint64_t *) val->val;

		if (cxi_ep->ep_obj->protocol != FI_PROTO_CXI_RNR) {
			CXIP_WARN("Not FI_PROTO_CXI_RNR EP\n");
			return -FI_EINVAL;
		}

		txc_rnr = container_of(cxi_ep->ep_obj->txc, struct cxip_txc_rnr,
				       base);
		txc_rnr->max_retry_wait_us = *req_rnr_max_time;
		CXIP_DBG("RNR maximum timeout set to %ld usec\n",
			 txc_rnr->max_retry_wait_us);
		break;
	default:
		return -FI_EINVAL;
	}

	return FI_SUCCESS;
}

/*
 * cxip_ki_cmdq_buf_validate() - Validate a user-provided KI command queue
 * buffer.
 *
 * Validate a user-provided KI command queue buffer against the
 * constraints the kernel applies when the endpoint is enabled.
 *
 * Omitting a buffer entirely is valid: a NULL pointer with a zero
 * length means no queue of that kind is requested, and the triggered
 * queue in particular is only allocated when a buffer is supplied.
 * Setting just one of the two is a caller error and is rejected.
 *
 * The kernel requires a supplied buffer to be page aligned and
 * physically contiguous, so a misaligned buffer is rejected here
 * rather than at enable time.
 *
 * The kernel also sizes the ring from the command count by rounding
 * up to a power-of-two number of pages, and maps that rounded length
 * starting at the buffer.  A length that is not itself a power-of-two
 * number of pages would therefore have the kernel map, and the
 * producer write, memory beyond the caller's buffer.  Such lengths
 * are rejected here, along with lengths above the hardware maximum.
 *
 * Physical contiguity cannot be checked from user space; it is only
 * detected by the kernel at allocation time.
 *
 * The dmabuf fields must agree with is_host_mem.  The kernel ignores
 * them for host memory, so a host buffer must carry the "not dmabuf"
 * values (fd -1, offset 0) to catch callers that filled them by
 * mistake.  A device buffer is either a dmabuf (fd >= 0) or not
 * (fd -1 with offset 0).
 */
static int cxip_ki_cmdq_buf_validate(const char *name,
				     const struct fi_cxi_ki_cmdq_buf *ubuf)
{
	long page_size = ofi_get_page_size();
	size_t max_len = (size_t)CXI_MAX_CQ_COUNT * C_CQ_CMD_SIZE;
	int ret = FI_SUCCESS;

	if ((ubuf->len == 0) && !ubuf->buf) {
		/* No buffer supplied for this queue; nothing to validate. */
	} else if ((ubuf->len == 0) || !ubuf->buf) {
		CXIP_WARN("KI %s cmdq buffer %p and length %zu must both be "
			  "set or both be unset\n", name, ubuf->buf, ubuf->len);
		ret = -FI_EINVAL;
	} else if ((uintptr_t)ubuf->buf % (uintptr_t)page_size) {
		CXIP_WARN("KI %s cmdq buffer %p is not aligned to the %ld "
			  "byte page size\n", name, ubuf->buf, page_size);
		ret = -FI_EINVAL;
	} else if ((ubuf->len < (size_t)page_size) ||
		   ((ubuf->len & (ubuf->len - 1)) != 0)) {
		CXIP_WARN("KI %s cmdq buffer length %zu must be a power-of-two "
			  "multiple of the %ld byte page size\n", name,
			  ubuf->len, page_size);
		ret = -FI_EINVAL;
	} else if (ubuf->len > max_len) {
		CXIP_WARN("KI %s cmdq buffer length %zu exceeds the maximum "
			  "%zu\n", name, ubuf->len, max_len);
		ret = -FI_EINVAL;
	} else if (ubuf->is_host_mem &&
		   ((ubuf->dmabuf_fd != -1) || (ubuf->dmabuf_offset != 0))) {
		CXIP_WARN("KI %s cmdq host buffer must have dmabuf_fd -1 and "
			  "dmabuf_offset 0, got %d and %lu\n", name,
			  ubuf->dmabuf_fd, ubuf->dmabuf_offset);
		ret = -FI_EINVAL;
	} else if (!ubuf->is_host_mem &&
		   ((ubuf->dmabuf_fd < -1) ||
		    ((ubuf->dmabuf_fd == -1) && (ubuf->dmabuf_offset != 0)))) {
		CXIP_WARN("KI %s cmdq device buffer must have dmabuf_fd >= 0 "
			  "or dmabuf_fd -1 with dmabuf_offset 0, got %d and "
			  "%lu\n", name, ubuf->dmabuf_fd, ubuf->dmabuf_offset);
		ret = -FI_EINVAL;
	}

	return ret;
}

/*
 * cxip_ep_control() - Provider EP control implementation.
 */
static int cxip_ep_control(struct fid *fid, int command, void *arg)
{
	struct cxip_ep *ep = container_of(fid, struct cxip_ep, ep.fid);
	struct cxip_ep *new_ep;
	struct fi_alias *alias;
	int ret;

	/* TODO: Remove this since it requires malicious programming to
	 * create this condition.
	 */
	if (fid->fclass != FI_CLASS_EP)
		return -FI_EINVAL;

	switch (command) {
	case FI_ALIAS:
		if (!arg)
			return -FI_EINVAL;
		alias = (struct fi_alias *)arg;
		if (!alias->fid)
			return -FI_EINVAL;
		new_ep = calloc(1, sizeof(*new_ep));
		if (!new_ep)
			return -FI_ENOMEM;

		memcpy(&new_ep->tx_attr, &ep->tx_attr,
		       sizeof(struct fi_tx_attr));
		memcpy(&new_ep->rx_attr, &ep->rx_attr,
		       sizeof(struct fi_rx_attr));
		ret = cxip_setopflags(&new_ep->tx_attr, &new_ep->rx_attr,
				      alias->flags);
		if (ret) {
			free(new_ep);
			return -FI_EINVAL;
		}
		new_ep->ep_obj = ep->ep_obj;
		new_ep->is_alias = 1;
		memcpy(&new_ep->ep, &ep->ep, sizeof(struct fid_ep));
		*alias->fid = &new_ep->ep.fid;
		ofi_atomic_inc32(&new_ep->ep_obj->ref);
		break;
	case FI_GETOPSFLAG:
		if (!arg)
			return -FI_EINVAL;
		ret = cxip_getopflags(&ep->tx_attr, &ep->rx_attr,
				      (uint64_t *)arg);
		if (ret)
			return -FI_EINVAL;
		break;
	case FI_SETOPSFLAG:
		if (!arg)
			return -FI_EINVAL;
		ret = cxip_setopflags(&ep->tx_attr, &ep->rx_attr,
				      *(uint64_t *)arg);
		if (ret)
			return -FI_EINVAL;
		break;
	case FI_ENABLE:
		return cxip_ep_enable(&ep->ep);
	case FI_SET_VAL:
		if (!arg)
			return -FI_EINVAL;
		return cxip_ep_set_val(ep, (struct fi_fid_var *) arg);
	case FI_OPT_CXI_SET_KI_EP_HINTS_1:
		if (!arg)
			return -FI_EINVAL;
		if (ep->ep_obj->enabled) {
			CXIP_WARN("KI EP hints must be set before enable\n");
			return -FI_EOPBADSTATE;
		}
		/* The KI command queues are allocated for the EP's single
		 * VNI and the GPU cannot switch them to another, so KI
		 * cannot reach peers with per-peer VNIs.
		 */
		if (ep->ep_obj->av_auth_key) {
			CXIP_WARN("KI requires a single-VNI EP; "
				  "FI_AV_AUTH_KEY is not supported\n");
			return -FI_EOPNOTSUPP;
		}
		{
			struct fi_cxi_ki_ep_hints *hints =
				(struct fi_cxi_ki_ep_hints *)arg;

			if (hints->flags) {
				CXIP_WARN("KI EP hints flags %#lx must be 0\n",
					  hints->flags);
				return -FI_EINVAL;
			}

			/* Report this here: the first command queue
			 * allocation happens in cxip_ep_ctrl_init(), which
			 * maps every failure to -FI_EDOMAIN.
			 */
			if ((hints->tx_cmdq.buf || hints->trig_cmdq.buf) &&
			    !cxip_libcxi_ops.alloc_buf_cmdq) {
				CXIP_WARN("User cmdq buffer unsupported by "
					  "this libcxi\n");
				return -FI_ENOSYS;
			}

			/* A KI TX command queue buffer becomes the ring for
			 * the EP's shared TX command queue, which messaging
			 * flow control and collectives also emit on without
			 * the application issuing host operations.  Those
			 * emits would collide with the GPU's write pointer,
			 * so refuse the capabilities that enable them.
			 */
			if (hints->tx_cmdq.buf &&
			    (ep->ep_obj->caps &
			     (FI_MSG | FI_TAGGED | FI_COLLECTIVE))) {
				CXIP_WARN("KI TX cmdq buffer requires an EP "
					  "without FI_MSG, FI_TAGGED and "
					  "FI_COLLECTIVE\n");
				return -FI_EINVAL;
			}

			ret = cxip_ki_cmdq_buf_validate("TX", &hints->tx_cmdq);
			if (ret != FI_SUCCESS) {
				return ret;
			}

			ret = cxip_ki_cmdq_buf_validate("triggered",
							&hints->trig_cmdq);
			if (ret != FI_SUCCESS) {
				return ret;
			}

			ep->ep_obj->ki_hints = *hints;

			CXIP_DBG("KI EP hints set: tx_buf=%p (%zu), trig_buf=%p (%zu)\n",
				 hints->tx_cmdq.buf, hints->tx_cmdq.len,
				 hints->trig_cmdq.buf, hints->trig_cmdq.len);
		}
		break;
	default:
		return -FI_EINVAL;
	}

	return FI_SUCCESS;
}

/* KI (GPU-initiated) Unified Operations Implementation */

/*
 * struct cxip_ki_peer_blob
 *
 * Internal view of the opaque bytes in struct fi_cxi_ki_peer_info.
 *
 * The public type is a bare uint8_t[32] so that callers treat it as
 * an opaque payload to hand to PMI_Allgather.  Within the provider the
 * layout is known, so peer blobs are cast to this structure rather
 * than assembled and taken apart one memcpy at a time.
 *
 * The blob crosses process boundaries, so it starts with a magic
 * byte and a version that get_gpu_remote_mrs() checks before
 * decoding.  The MR key is carried whole; the initiator-side values
 * that depend on it (match bits, offset adjustment, restricted or
 * unrestricted) are derived when the blob is decoded.
 *
 * Packed for alignment, not for layout.  Every member already sits at
 * its natural offset, so the attribute changes nothing except the
 * alignment the structure demands of its address.  That matters
 * because the public type is only 1-byte aligned, and peer blobs
 * routinely arrive in the middle of an allgather receive buffer.
 */
#define CXIP_KI_PEER_BLOB_MAGIC		0xC2
#define CXIP_KI_PEER_BLOB_VERSION	1

struct cxip_ki_peer_blob {
	uint8_t  magic;		/* CXIP_KI_PEER_BLOB_MAGIC */
	uint8_t  version;	/* CXIP_KI_PEER_BLOB_VERSION */
	uint8_t  padding[2];	/* Align the subsequent fields */
	uint16_t write_ptl_idx;	/* Portal table index for Puts */
	uint16_t read_ptl_idx;	/* Portal table index for Gets */
	uint64_t iova;		/* IOVA of the MR start */
	uint64_t key;		/* MR key, as passed to fi_mr_reg() */
	uint8_t  unused[8];	/* Fills a fi_cxi_ki_peer_info */
} __attribute__((packed));

_Static_assert(sizeof(struct cxip_ki_peer_blob) <=
	       sizeof(struct fi_cxi_ki_peer_info),
	       "KI peer blob must fit within the opaque payload");

/*
 * Helper: populate a fi_cxi_gpu_cmdq wrapper from a cxip_cmdq.
 *
 * Copies the command queue state the GPU needs into the wrapper and sets
 * host pointers.  Device pointers are set later by the user via inline
 * helpers.
 */
static void cxip_ki_fill_cmdq(struct fi_cxi_gpu_cmdq *out,
			      struct cxip_cmdq *cmdq)
{
	struct cxi_cq *dev = cmdq->dev_cmdq;

	/* Host pointers - user needs these for GPU registration */
	out->hptr_buff = (void *)dev->cmds32;
	out->hptr_mmio_db = dev->wp_addr;

	/* Device pointers - not set yet (user calls set_buf/set_db) */
	out->dptr_buff = NULL;
	out->dptr_mmio_db = NULL;

	/* Copy HW command queue state */
	out->cq.cmds32 = dev->cmds32;
	out->cq.wp_addr = dev->wp_addr;
	out->cq.wp32 = dev->wp32;
	out->cq.hw_wp32 = dev->hw_wp32;
	out->cq.rp32 = dev->rp32;
	out->cq.size32 = dev->size32;
}

/*
 * cxip_ki_selective_completion_cb() - KI completion callback.
 *
 * KI commands must suppress send and success events, so this normally
 * only runs for failures.  The request carries no per-operation state,
 * so error entries reported here have flags 0 and carry the endpoint's
 * context (the context passed to fi_endpoint()) as op_context.
 */
static int cxip_ki_selective_completion_cb(struct cxip_req *req,
					   const union c_event *event)
{
	int event_rc;
	int ret_err;
	int ret;

	event_rc = cxi_init_event_rc(event);

	if (event->hdr.event_type == C_EVENT_SEND) {
		/*
		 * When errors happen, send events can occur before the
		 * put/get event.  The put/get event is authoritative and
		 * is reported on its own, so the send event is dropped
		 * whatever its return code, as cxip_rma_cb() does.
		 * Reporting a failed send as well would post two error
		 * entries for one command.
		 */
		CXIP_WARN(CXIP_UNEXPECTED_EVENT, cxi_event_to_str(event),
			  cxi_rc_to_str(cxi_event_rc(event)));
		ret = FI_SUCCESS;
	} else if (event_rc == C_RC_OK) {
		/*
		 * The command left event_success_disable clear.  A
		 * completion written here would carry no operation
		 * information (flags 0, endpoint context), so drop the
		 * event and warn.
		 */
		CXIP_WARN("Unexpected successful %s event\n",
			  cxi_event_to_str(event));
		ret = FI_SUCCESS;
	} else {
		ret_err = proverr2errno(event_rc);

		CXIP_WARN("KI operation failed: %s rc=%s\n",
			  cxi_event_to_str(event),
			  cxi_rc_to_str(cxi_event_rc(event)));

		ret = cxip_cq_req_error(req, 0, ret_err, cxi_event_rc(event),
					NULL, 0, FI_ADDR_UNSPEC);
	}

	return ret;
}

/*
 * cxip_ki_selective_completion_req() - Return the shared KI error
 * completion request.
 *
 * Allocates on first use and caches on the EP.  Every KI command, of
 * any operation type, stamps this same request as its user_ptr, so
 * the resulting error completion carries no operation-type
 * information (flags is 0).  The request belongs to the TXC event
 * queue and is reclaimed with its request pool when cxip_txc_disable()
 * calls cxip_evtq_fini().
 *
 * Takes ep_obj->lock: the request pool is shared with the data path,
 * and the application may call this concurrently with progress.
 *
 * Returns NULL if no request could be allocated.
 */
static struct cxip_req *
cxip_ki_selective_completion_req(struct cxip_ep_obj *ep_obj)
{
	struct cxip_req **cached = &ep_obj->ki_selective_completion_req;
	struct cxip_req *req;

	ofi_genlock_lock(&ep_obj->lock);
	if (!*cached) {
		req = cxip_evtq_req_alloc(&ep_obj->txc->tx_evtq, 0,
					  ep_obj->txc);
		if (!req) {
			CXIP_WARN("Failed to allocate KI completion request\n");
		} else {
			req->cb = cxip_ki_selective_completion_cb;
			req->context = (uint64_t)ep_obj->txc->context;
			req->flags = 0;
			req->addr = FI_ADDR_UNSPEC;

			*cached = req;
		}
	}
	req = *cached;
	ofi_genlock_unlock(&ep_obj->lock);

	return req;
}

/*
 * cxip_ki_get_gpu_ep_info() - Get GPU-ready EP info in one call.
 *
 * Replaces: get_tx_cmdq_info + get_trig_cmdq_info + get_ep_info + get_cntr_info
 */
static int cxip_ki_get_gpu_ep_info(struct fid *fid,
				   struct fi_cxi_gpu_ep_info *gpu_ep)
{
	struct cxip_ep *ep;

	if (!fid || !gpu_ep)
		return -FI_EINVAL;

	if (fid->fclass != FI_CLASS_EP)
		return -FI_EINVAL;

	ep = container_of(fid, struct cxip_ep, ep.fid);

	if (!ep->ep_obj->enabled) {
		CXIP_WARN("EP must be enabled before getting GPU EP info\n");
		return -FI_EOPBADSTATE;
	}

	memset(gpu_ep, 0, sizeof(*gpu_ep));

	/* TX command queue */
	if (!ep->ep_obj->txq || !ep->ep_obj->txq->dev_cmdq) {
		CXIP_WARN("TX cmdq not available\n");
		return -FI_EOPNOTSUPP;
	}

	cxip_ki_fill_cmdq(&gpu_ep->tx_cmdq, ep->ep_obj->txq);

	/*
	 * Triggered command queue (optional).
	 *
	 * Left zero-filled when no triggered buffer was supplied via
	 * FI_OPT_CXI_SET_KI_EP_HINTS_1.  Callers detect this by testing
	 * cq.wp_addr (or hptr_mmio_db) for NULL.
	 */
	if (ep->ep_obj->ki_trig_cmdq && ep->ep_obj->ki_trig_cmdq->dev_cmdq) {
		cxip_ki_fill_cmdq(&gpu_ep->trig_cmdq, ep->ep_obj->ki_trig_cmdq);
	} else {
		CXIP_DBG("KI trig cmdq not present, trig_cmdq left zeroed\n");
	}

	/*
	 * EQ number from the TXC's TX event queue (cxip_txc.tx_evtq), not
	 * the similarly-named control event queue (cxip_ctrl.tx_evtq).
	 */
	if (ep->ep_obj->txc && ep->ep_obj->txc->tx_evtq.eq) {
		gpu_ep->eqn = cxip_evtq_eqn(&ep->ep_obj->txc->tx_evtq);
	} else {
		gpu_ep->eqn = C_EQ_NONE;
		CXIP_DBG("KI: no TX event queue, eqn=C_EQ_NONE\n");
	}

	/*
	 * Error reporting handle.
	 *
	 * KI commands suppress success events, so this is used only when
	 * an operation fails.  Left zero when the endpoint has no event
	 * queue to report against, in which case a failure is visible only
	 * through the counter's failure count.
	 */
	if (gpu_ep->eqn != C_EQ_NONE) {
		struct cxip_req *req;

		req = cxip_ki_selective_completion_req(ep->ep_obj);
		if (!req)
			return -FI_ENOMEM;

		gpu_ep->user_ptr = (uint64_t)req;
	} else {
		CXIP_DBG("KI: no event queue, error reporting disabled\n");
	}

	/* Counter number from the EP's bound send counter. */
	if (ep->ep_obj->txc && ep->ep_obj->txc->send_cntr &&
	    ep->ep_obj->txc->send_cntr->ct) {
		gpu_ep->ctn = ep->ep_obj->txc->send_cntr->ct->ctn;
	} else {
		gpu_ep->ctn = C_CT_NONE;
		CXIP_DBG("KI: no bound send counter, ctn=C_CT_NONE\n");
	}

	/* PID bits */
	gpu_ep->pid_bits = ep->ep_obj->domain->iface->info->pid_bits;

	CXIP_DBG("KI get_gpu_ep_info: eqn=%u ctn=%u pid_bits=%u\n",
		 gpu_ep->eqn, gpu_ep->ctn, gpu_ep->pid_bits);

	return FI_SUCCESS;
}

/*
 * cxip_ki_get_peer_info() - Get opaque blob for PMI exchange.
 *
 * Encodes MR's {iova, write_ptl_idx, read_ptl_idx} into an
 * opaque blob that can be exchanged via PMI_Allgather.
 */
static int cxip_ki_get_peer_info(struct fid *ep_fid, struct fid *fid,
				 struct fi_cxi_ki_peer_info *info)
{
	struct cxip_ki_peer_blob *blob;
	struct cxip_ep *ep;
	struct cxip_mr *mr;

	if (!ep_fid || !fid || !info)
		return -FI_EINVAL;

	if (ep_fid->fclass != FI_CLASS_EP || fid->fclass != FI_CLASS_MR)
		return -FI_EINVAL;

	ep = container_of(ep_fid, struct cxip_ep, ep.fid);
	mr = container_of(fid, struct cxip_mr, mr_fid.fid);

	/* Peers reach the MR through the PID of the EP it is bound to,
	 * so the blob is only valid alongside that EP's address.
	 */
	if (!mr->ep || mr->ep->ep_obj != ep->ep_obj) {
		CXIP_WARN("MR not bound to EP\n");
		return -FI_EINVAL;
	}

	/* Only a remotely accessible MR gets a key, and so portal indexes,
	 * and they are assigned at enable.
	 */
	if (!(mr->attr.access & (FI_REMOTE_READ | FI_REMOTE_WRITE))) {
		CXIP_WARN("MR has no remote access\n");
		return -FI_EOPBADSTATE;
	}

	if (!mr->enabled) {
		CXIP_WARN("MR not enabled\n");
		return -FI_EOPBADSTATE;
	}

	if (!mr->md || !mr->md->md) {
		CXIP_WARN("MR not mapped\n");
		return -FI_EOPBADSTATE;
	}

	memset(info, 0, sizeof(*info));

	blob = (struct cxip_ki_peer_blob *)info->opaque;

	blob->magic = CXIP_KI_PEER_BLOB_MAGIC;
	blob->version = CXIP_KI_PEER_BLOB_VERSION;
	blob->write_ptl_idx = cxip_generic_mr_key_to_ptl_idx(mr->domain,
							     mr->key, true);
	blob->read_ptl_idx = cxip_generic_mr_key_to_ptl_idx(mr->domain,
							    mr->key, false);
	/* The IOVA of the MR itself, as linked in its LE, not of the
	 * possibly larger cached MD that backs it.
	 */
	blob->iova = mr->len ? CXI_VA_TO_IOVA(mr->md->md, mr->buf) : 0;
	blob->key = mr->key;

	CXIP_DBG("KI get_peer_info: iova=0x%lx wptl=%u rptl=%u key=0x%lx\n",
		 blob->iova, blob->write_ptl_idx, blob->read_ptl_idx,
		 blob->key);

	return FI_SUCCESS;
}

/*
 * cxip_ki_get_gpu_target_info() - Bulk resolve fi_addr_t → NID/PID.
 *
 * Pure addressing - no DFA, no MR info.  A command carries no VNI; the
 * KI command queues send on the EP's VNI, so per-peer VNIs
 * (FI_AV_AUTH_KEY) are not supported.
 */
static int cxip_ki_get_gpu_target_info(struct fid *fid,
				       fi_addr_t *addrs,
				       int npes,
				       struct fi_cxi_gpu_target_info *targets)
{
	struct cxip_ep *ep;
	struct cxip_addr caddr;
	int i, ret;

	if (!fid || !addrs || !targets || npes <= 0)
		return -FI_EINVAL;

	if (fid->fclass != FI_CLASS_EP)
		return -FI_EINVAL;

	ep = container_of(fid, struct cxip_ep, ep.fid);

	if (ep->ep_obj->av_auth_key) {
		CXIP_WARN("KI requires a single-VNI EP; "
			  "FI_AV_AUTH_KEY is not supported\n");
		return -FI_EOPNOTSUPP;
	}

	if (!ep->ep_obj->av) {
		CXIP_WARN("EP has no bound AV\n");
		return -FI_EOPBADSTATE;
	}

	for (i = 0; i < npes; i++) {
		ret = cxip_av_lookup_addr_checked(ep->ep_obj->av, addrs[i],
						  &caddr);
		if (ret != FI_SUCCESS) {
			CXIP_WARN("Failed to lookup fi_addr %lu: %d\n",
				  addrs[i], ret);
			return ret;
		}

		targets[i].nid = caddr.nic;
		targets[i].pid = caddr.pid;
	}

	CXIP_DBG("KI get_gpu_target_info: resolved %d targets\n", npes);

	return FI_SUCCESS;
}

/*
 * cxip_ki_get_gpu_remote_mrs() - Decode remote MRs from peer blobs.
 *
 * Extracts {iova, write_ptl_idx, read_ptl_idx} from each peer
 * blob and derives from its MR key the values an initiator needs to
 * target the MR: the DMA match bits (used only by a standard MR's
 * matching LE), the offset to add to every MR-relative remote offset
 * (nonzero for a cached provider key), and whether the MR must be
 * targeted with unrestricted commands.  DFA is NOT computed here -
 * GPU computes it on-device.
 */
static int cxip_ki_get_gpu_remote_mrs(struct fid *fid,
				      const struct fi_cxi_ki_peer_info *peer_info,
				      int npes,
				      struct fi_cxi_gpu_remote_mr *remote_mrs)
{
	const struct cxip_ki_peer_blob *blob;
	struct cxip_mr_key key;
	struct cxip_ep *ep;
	int i;

	if (!fid || !peer_info || !remote_mrs || npes <= 0)
		return -FI_EINVAL;

	if (fid->fclass != FI_CLASS_EP)
		return -FI_EINVAL;

	ep = container_of(fid, struct cxip_ep, ep.fid);

	for (i = 0; i < npes; i++) {
		blob = (const struct cxip_ki_peer_blob *)peer_info[i].opaque;

		if (blob->magic != CXIP_KI_PEER_BLOB_MAGIC ||
		    blob->version != CXIP_KI_PEER_BLOB_VERSION) {
			CXIP_WARN("Peer blob %d has magic 0x%02x version %u, "
				  "expected 0x%02x version %u\n", i,
				  blob->magic, blob->version,
				  CXIP_KI_PEER_BLOB_MAGIC,
				  CXIP_KI_PEER_BLOB_VERSION);
			return -FI_EINVAL;
		}

		key.raw = blob->key;

		memset(&remote_mrs[i], 0, sizeof(remote_mrs[i]));
		remote_mrs[i].iova = blob->iova;
		remote_mrs[i].write_ptl_idx = blob->write_ptl_idx;
		remote_mrs[i].read_ptl_idx = blob->read_ptl_idx;
		remote_mrs[i].match_bits = CXIP_KEY_MATCH_BITS(blob->key);
		remote_mrs[i].offset_adjust = key.cached ? key.lac_off : 0;
		remote_mrs[i].unrestricted =
			!cxip_generic_is_mr_key_opt(blob->key) ||
			cxip_generic_is_mr_key_events(ep->ep_obj->caps,
						      blob->key);
	}

	CXIP_DBG("KI get_gpu_remote_mrs: decoded %d remote MRs\n", npes);

	return FI_SUCCESS;
}

/*
 * cxip_ki_get_gpu_mr_info() - Get local MR info for GPU access.
 *
 * Returns the LAC and the IOVA of buf, which must lie within the MR.
 */
static int cxip_ki_get_gpu_mr_info(struct fid *ep_fid, struct fid *fid,
				   void *buf,
				   struct fi_cxi_gpu_mr_info *info)
{
	struct cxip_ep *ep;
	struct cxip_mr *mr;

	if (!ep_fid || !fid || !buf || !info)
		return -FI_EINVAL;

	if (ep_fid->fclass != FI_CLASS_EP || fid->fclass != FI_CLASS_MR)
		return -FI_EINVAL;

	ep = container_of(ep_fid, struct cxip_ep, ep.fid);
	mr = container_of(fid, struct cxip_mr, mr_fid.fid);

	/* The LAC and IOVA are only meaningful on the EP's LNI. */
	if (mr->domain != ep->ep_obj->domain) {
		CXIP_WARN("MR domain does not match EP\n");
		return -FI_EINVAL;
	}

	if (!mr->md || !mr->md->md) {
		CXIP_WARN("MR not mapped\n");
		return -FI_EOPBADSTATE;
	}

	if ((uintptr_t)buf < (uintptr_t)mr->buf ||
	    (uintptr_t)buf >= (uintptr_t)mr->buf + mr->len) {
		CXIP_WARN("buf %p outside MR [%p, %p)\n", buf, mr->buf,
			  (void *)((uintptr_t)mr->buf + mr->len));
		return -FI_EINVAL;
	}

	info->buf = buf;
	info->iova = CXI_VA_TO_IOVA(mr->md->md, buf);
	info->lac = mr->md->md->lac;
	info->flags = 0;

	CXIP_DBG("KI get_gpu_mr_info: buf=%p iova=0x%lx lac=%u\n",
		 buf, info->iova, info->lac);

	return FI_SUCCESS;
}

/*
 * cxip_ki_get_cntr_number() - Get HW counter number from a counter FID.
 */
static int cxip_ki_get_cntr_number(struct fid *ep_fid, struct fid *fid,
				   uint32_t *ctn)
{
	struct cxip_ep *ep;
	struct cxip_cntr *cntr;

	if (!ep_fid || !fid || !ctn)
		return -FI_EINVAL;

	if (ep_fid->fclass != FI_CLASS_EP || fid->fclass != FI_CLASS_CNTR)
		return -FI_EINVAL;

	ep = container_of(ep_fid, struct cxip_ep, ep.fid);
	cntr = container_of(fid, struct cxip_cntr, cntr_fid);

	/* The counter number is only meaningful on the EP's LNI. */
	if (cntr->domain != ep->ep_obj->domain) {
		CXIP_WARN("Counter domain does not match EP\n");
		return -FI_EINVAL;
	}

	if (!cntr->ct)
		return -FI_EOPBADSTATE;

	*ctn = cntr->ct->ctn;

	CXIP_DBG("KI get_cntr_number: ctn=%u\n", *ctn);

	return FI_SUCCESS;
}

/* KI unified operations structure */
static struct fi_cxi_ki_ops cxip_ki_ops = {
	.get_gpu_ep_info = cxip_ki_get_gpu_ep_info,
	.get_peer_info = cxip_ki_get_peer_info,
	.get_gpu_target_info = cxip_ki_get_gpu_target_info,
	.get_gpu_remote_mrs = cxip_ki_get_gpu_remote_mrs,
	.get_gpu_mr_info = cxip_ki_get_gpu_mr_info,
	.get_cntr_number = cxip_ki_get_cntr_number,
};

/*
 * cxip_ep_ops_open() - Open provider-specific operations for EP.
 *
 * No flags are defined; returns -FI_ENOSYS for an unknown ops name so
 * that an application built against a newer interface can detect an
 * older provider.
 */
static int cxip_ep_ops_open(struct fid *fid, const char *name,
			    uint64_t flags, void **ops, void *context)
{
	if (!name || !ops || flags)
		return -FI_EINVAL;

	if (strcmp(name, FI_CXI_KI_OPS_1) == 0) {
		*ops = &cxip_ki_ops;
		return FI_SUCCESS;
	}

	return -FI_ENOSYS;
}

struct fi_ops cxip_ep_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = cxip_ep_close,
	.bind = cxip_ep_bind,
	.control = cxip_ep_control,
	.ops_open = cxip_ep_ops_open,
};

int cxip_ep_getopt_priv(struct cxip_ep *ep, int level, int optname,
			void *optval, size_t *optlen)
{
	struct cxip_rxc_rnr *rxc_rnr;

	if (level != FI_OPT_ENDPOINT)
		return -FI_ENOPROTOOPT;

	switch (optname) {
	case FI_OPT_MIN_MULTI_RECV:
		if (!optval || !optlen)
			return -FI_EINVAL;
		if (*optlen < sizeof(size_t))
			return -FI_ETOOSMALL;

		*(size_t *)optval = ep->ep_obj->rxc->min_multi_recv;
		*optlen = sizeof(size_t);
		break;

	case FI_OPT_CUDA_API_PERMITTED:
		if (!optval || !optlen)
			return -FI_EINVAL;
		if (*optlen < sizeof(bool))
			return -FI_ETOOSMALL;

		*(bool *)optval =
			!ep->ep_obj->require_dev_reg_copy[FI_HMEM_CUDA];
		break;

	case FI_OPT_CXI_GET_RNR_APPEND_RETRY_ATTEMPTS:
		if (!optval || !optlen)
			return -FI_EINVAL;
		if (*optlen < sizeof(u_int64_t))
			return -FI_ETOOSMALL;
		if (ep->ep_obj->protocol != FI_PROTO_CXI_RNR) {
			CXIP_WARN("Not FI_PROTO_CXI_RNR EP\n");
			return -FI_EINVAL;
		}

		rxc_rnr = container_of(ep->ep_obj->rxc, struct cxip_rxc_rnr,
				       base);
		*(u_int64_t *)optval = ofi_atomic_get64(&rxc_rnr->total_append_retries);
		CXIP_WARN("rxc_rnr total append retry count %ld\n",
			ofi_atomic_get64(&rxc_rnr->total_append_retries));
		break;

	default:
		return -FI_ENOPROTOOPT;
	}

	return FI_SUCCESS;
}

/*
 * cxip_ep_getopt() - Return endpoint option value if supported.
 */
static int cxip_ep_getopt(fid_t fid, int level, int optname, void *optval,
			  size_t *optlen)
{
	struct cxip_ep *ep = container_of(fid, struct cxip_ep, ep.fid);

	return cxip_ep_getopt_priv(ep, level, optname, optval, optlen);
}

int cxip_ep_setopt_priv(struct cxip_ep *ep, int level, int optname,
			const void *optval, size_t optlen)
{
	size_t min_multi_recv;
	bool cuda_api_permitted;

	if (level != FI_OPT_ENDPOINT)
		return -FI_ENOPROTOOPT;

	switch (optname) {
	case FI_OPT_MIN_MULTI_RECV:
		if (!optval)
			return -FI_EINVAL;

		min_multi_recv = *(size_t *)optval;

		if (min_multi_recv > CXIP_EP_MAX_MULTI_RECV) {
			CXIP_WARN("Maximum min_multi_recv value is: %u\n",
				  CXIP_EP_MAX_MULTI_RECV);
			return -FI_EINVAL;
		}
		ep->ep_obj->rxc->min_multi_recv = min_multi_recv;
		break;
	/*
	 * If GDRCopy is required by the application (ie. it has set
	 * FI_OPT_CUDA_API_PERMITTED), and is not available, return not
	 * supported.
	 */
	case FI_OPT_CUDA_API_PERMITTED:
		if (optlen != sizeof(bool))
			return -FI_EINVAL;

		if (!hmem_ops[FI_HMEM_CUDA].initialized) {
			CXIP_WARN("FI_OPT_CUDA_API_PERMITTED cannot be set when CUDA library or CUDA device is not available\n");
			return -FI_EOPNOTSUPP;
		}

		cuda_api_permitted = *(bool *)optval;

		if (!cuda_api_permitted && !cuda_is_gdrcopy_enabled())
			return -FI_EOPNOTSUPP;

		if (!cxip_env.force_dev_reg_copy) {
			ep->ep_obj->require_dev_reg_copy[FI_HMEM_CUDA] =
				!cuda_api_permitted;
		}
		break;

	default:
		return -FI_ENOPROTOOPT;
	}

	return FI_SUCCESS;
}

/*
 * cxip_ep_setopt() - Set endpoint option value if supported.
 */
static int cxip_ep_setopt(fid_t fid, int level, int optname, const void *optval,
			  size_t optlen)
{
	struct cxip_ep *ep = container_of(fid, struct cxip_ep, ep.fid);

	return cxip_ep_setopt_priv(ep, level, optname, optval, optlen);
}

struct fi_ops_ep cxip_ep_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = cxip_ep_cancel,
	.getopt = cxip_ep_getopt,
	.setopt = cxip_ep_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

/*
 * cxip_alloc_endpoint() - Allocate and initialize base EP object.
 */
int cxip_alloc_endpoint(struct cxip_domain *cxip_dom, struct fi_info *hints,
			struct cxip_ep_obj **ep_base_obj, void *context)
{
	int ret;
	struct cxip_ep_obj *ep_obj;
	uint32_t txc_tclass = FI_TC_UNSPEC;
	struct cxip_addr ep_addr;
	int i;

	if (!hints || !hints->ep_attr || !hints->tx_attr || !hints->rx_attr)
		return -FI_EINVAL;

	ret = ofi_prov_check_info(&cxip_util_prov, CXIP_FI_VERSION, hints);
	if (ret != FI_SUCCESS)
		return -FI_EINVAL;

	if (cxip_dom->auth_key_entry_max > 1 &&
	    ((hints->caps & FI_DIRECTED_RECV) ||
	     (hints->rx_attr->caps & FI_DIRECTED_RECV))) {
		CXIP_WARN("FI_DIRECTED_RECV not supported with multiple auth key per EP\n");
		return -FI_EINVAL;
	}

	ret = cxip_check_auth_key_info(hints);
	if (ret)
		return ret;

	ep_addr.nic = cxip_dom->nic_addr;
	if (hints->src_addr) {
		struct cxip_addr *src = hints->src_addr;
		if (src->nic != ep_addr.nic) {
			CXIP_WARN("bad src_addr NIC value\n");
			return -FI_EINVAL;
		}
		ep_addr.pid = src->pid;
	} else {
		ep_addr.pid = C_PID_ANY;
	}

	ep_obj = calloc(1, sizeof(struct cxip_ep_obj));
	if (!ep_obj)
		return -FI_ENOMEM;

	/* Save EP attributes from hints */
	ep_obj->protocol = hints->ep_attr->protocol;
	ep_obj->caps = hints->caps;
	ep_obj->ep_attr = *hints->ep_attr;
	ep_obj->txq_size = hints->tx_attr->size;
	ep_obj->tgq_size = hints->rx_attr->size;
	ep_obj->tx_attr = *hints->tx_attr;
	ep_obj->rx_attr = *hints->rx_attr;
	ep_obj->wait_fd = -1;

	ep_obj->asic_ver = cxip_dom->iface->info->cassini_version;

	ofi_atomic_initialize32(&ep_obj->ref, 0);

	/* Allow FI_THREAD_DOMAIN optimizaiton */
	if (cxip_dom->util_domain.threading == FI_THREAD_DOMAIN ||
	    cxip_dom->util_domain.threading == FI_THREAD_COMPLETION)
		ofi_genlock_init(&ep_obj->lock, OFI_LOCK_NONE);
	else
		ofi_genlock_init(&ep_obj->lock, OFI_LOCK_SPINLOCK);

	ep_obj->domain = cxip_dom;
	ep_obj->src_addr.nic = ep_addr.nic;
	ep_obj->src_addr.pid = ep_addr.pid;
	ep_obj->fi_addr = FI_ADDR_NOTAVAIL;

	/* Default to allowing non-dev reg copy APIs unless the caller
	 * disables it.
	 */
	for (i = 0; i < OFI_HMEM_MAX; i++)
		ep_obj->require_dev_reg_copy[i] = cxip_env.force_dev_reg_copy;

	ofi_atomic_initialize32(&ep_obj->txq_ref, 0);
	ofi_atomic_initialize32(&ep_obj->tgq_ref, 0);

	for (i = 0; i < CXIP_NUM_CACHED_KEY_LE; i++) {
		ofi_atomic_initialize32(&ep_obj->ctrl.std_mr_cache[i].ref, 0);
		ofi_atomic_initialize32(&ep_obj->ctrl.opt_mr_cache[i].ref, 0);
	}

	dlist_init(&ep_obj->ctrl.mr_list);
	ep_obj->ep_attr.tx_ctx_cnt = 1;
	ep_obj->ep_attr.rx_ctx_cnt = 1;

	if (hints->ep_attr->auth_key) {
		/* Auth key size is verified in ofi_prov_check_info(). */
		assert(hints->ep_attr->auth_key_size ==
		       sizeof(struct cxi_auth_key));

		memcpy(&ep_obj->auth_key, hints->ep_attr->auth_key,
		       sizeof(struct cxi_auth_key));

		/* All EPs that share a Domain must use the same Service ID. */
		if (ep_obj->auth_key.svc_id != cxip_dom->auth_key.svc_id) {
			CXIP_WARN("Invalid svc_id: %u\n",
				  ep_obj->auth_key.svc_id);
			ret = -FI_EINVAL;
			goto err;
		}
	} else {
		if (cxip_dom->av_auth_key) {
			ep_obj->av_auth_key = true;
		} else {
			/* Inherit auth_key from Domain. */
			ep_obj->auth_key = cxip_dom->auth_key;
			CXIP_DBG("Inherited domain auth_key\n");
		}
	}

	if (cxip_set_tclass(ep_obj->tx_attr.tclass,
			    cxip_dom->tclass, &txc_tclass)) {
		CXIP_WARN("Invalid tclass\n");
		ret = -FI_EINVAL;
		goto err;
	}

	ep_obj->tx_attr.tclass = txc_tclass;

	ep_obj->txc = cxip_txc_calloc(ep_obj, context);
	if (!ep_obj->txc) {
		ret = -FI_ENOMEM;
		goto err;
	}

	ep_obj->rxc = cxip_rxc_calloc(ep_obj, context);
	if (!ep_obj->rxc) {
		ret = -FI_ENOMEM;
		goto err;
	}

	ofi_atomic_inc32(&cxip_dom->ref);
	*ep_base_obj = ep_obj;

	return FI_SUCCESS;

err:
	/* handles null check */
	cxip_txc_free(ep_obj->txc);
	cxip_rxc_free(ep_obj->rxc);
	free(ep_obj);

	return ret;
}

int cxip_ep_obj_map(struct cxip_ep_obj *ep, const void *buf, unsigned long len,
		    uint64_t access, uint64_t flags, struct cxip_md **md)
{
	struct cxip_domain *dom = ep->domain;
	int ret;

	ret = cxip_map(dom, buf, len, access, flags, md);
	if (ret != FI_SUCCESS)
		return ret;

	if (ep->require_dev_reg_copy[(*md)->info.iface] &&
	    !((*md)->handle_valid)) {
		CXIP_WARN("Required dev registration copy failed\n");
		cxip_unmap(*md);
		return -FI_EOPNOTSUPP;
	}

	return FI_SUCCESS;
}

/*
 * cxip_endpoint() - Provider fi_endpoint() implementation.
 */
int cxip_endpoint(struct fid_domain *domain, struct fi_info *info,
		  struct fid_ep **fid_ep, void *context)
{
	int ret;
	struct cxip_ep *ep;
	struct cxip_domain *cxip_dom = container_of(domain, struct cxip_domain,
						    util_domain.domain_fid);
	if (!fid_ep)
		return -FI_EINVAL;

	ep = calloc(1, sizeof(*ep));
	if (!ep)
		return -FI_ENOMEM;

	/* Allocate and initialize the base endpoint */
	ret = cxip_alloc_endpoint(cxip_dom, info, &ep->ep_obj, context);
	if (ret) {
		free(ep);
		return ret;
	}

	/* Store EP attribures with the wrapper since values can be
	 * overridden by alias EP that share the same EP object.
	 */
	ep->tx_attr = ep->ep_obj->tx_attr;
	ep->rx_attr = ep->ep_obj->rx_attr;

	ep->ep.fid.fclass = FI_CLASS_EP;
	ep->ep.fid.context = context;
	ep->ep.fid.ops = &cxip_ep_fi_ops;
	ep->ep.ops = &cxip_ep_ops;
	ep->ep.cm = &cxip_ep_cm_ops;

	/* Initialize API to not supported until EP is enabled */
	ep->ep.msg = &cxip_ep_msg_no_ops;
	ep->ep.tagged = &cxip_ep_tagged_no_ops;
	ep->ep.rma = &cxip_ep_rma_no_ops;
	ep->ep.atomic = &cxip_ep_atomic_no_ops;
	ep->ep.collective = &cxip_collective_no_ops;

	*fid_ep = &ep->ep;

	cxip_coll_init(ep->ep_obj);
	cxip_domain_add_txc(ep->ep_obj->domain, ep->ep_obj->txc);

	return FI_SUCCESS;
}

