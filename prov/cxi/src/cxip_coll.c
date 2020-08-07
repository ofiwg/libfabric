/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2020 Cray Inc. All rights reserved.
 */

/* Support for Restricted Nomatch Put.
 */


/****************************************************************************
 * Exported:
 *
 * cxip_coll_init()
 * cxip_coll_enable()
 * cxip_coll_close()
 * cxip_coll_send()
 *
 */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>

#include <ofi_list.h>
#include <ofi.h>

#include "cxip.h"

#define CXIP_LOG_DBG(...) _CXIP_LOG_DBG(FI_LOG_EP_CTRL, \
		"COLL " __VA_ARGS__)
#define CXIP_LOG_ERROR(...) _CXIP_LOG_ERROR(FI_LOG_EP_CTRL, \
		"COLL " __VA_ARGS__)

static ssize_t _coll_recv(struct cxip_ep_obj *ep_obj,
			  struct cxip_coll_buf *buf);

/****************************************************************************
 * SEND operations (send data to a remote PTE)
 */

/**
 * Issue a restricted IDC Put to the destination address.
 *
 * Tailored for collectives. SEND events disabled.
 *
 * @param ep_obj - endpoint object
 * @param buffer - buffer containing data to send
 * @param buflen - byte count of data in buffer
 * @param dest_addr - fabric address for destination
 *
 * @return int - return code
 */
int cxip_coll_send(struct cxip_ep_obj *ep_obj,
		   const void *buffer, size_t buflen,
		   fi_addr_t dest_addr)
{
	struct cxip_addr dest_caddr;
	union c_fab_addr dfa;
	uint8_t pid_bits;
	uint8_t idx_ext;
	uint32_t pid_idx;
	struct cxip_cmdq *cmdq = ep_obj->coll.tx_cmdq;
	int ret;

	if (!ep_obj->coll.enabled)
		return -FI_EOPBADSTATE;

	if (buflen && !buffer)
		return -FI_EINVAL;

	if (buflen > 49)	// TODO sizeof(struct reduction_packet)
		return -FI_EMSGSIZE;

	/* Look up target CXI address */
	ret = _cxip_av_lookup(ep_obj->av, dest_addr, &dest_caddr);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Failed to look up FI addr: %d\n", ret);
		return ret;
	}

	/* Calculate DFA */
	pid_idx = CXIP_PTL_IDX_RXC(CXIP_PTL_IDX_COLL);
	pid_bits = ep_obj->domain->iface->dev->info.pid_bits;
	if (ep_obj->coll.is_netsim) {
		cxi_build_dfa(dest_caddr.nic, dest_caddr.pid, pid_bits,
			      pid_idx, &dfa, &idx_ext);
	} else {
		// TODO: when multicast is ready
		// cxi_build_multicast_dfa();
		CXIP_LOG_ERROR("Cassini multicast not supported\n");
		return -FI_EOPNOTSUPP;
	}

	/* Submit command */
	union c_cmdu cmd = {};
	cmd.c_state.event_send_disable = 1;
	cmd.c_state.event_success_disable = 1;
	cmd.c_state.restricted = 1;
	cmd.c_state.reduction = !ep_obj->coll.is_netsim;
	cmd.c_state.index_ext = idx_ext;
	cmd.c_state.eq = ep_obj->coll.tx_cq->evtq->eqn;
	cmd.c_state.initiator = CXI_MATCH_ID(pid_bits, ep_obj->src_addr.pid,
					     ep_obj->src_addr.nic);

	fastlock_acquire(&cmdq->lock);
	if (memcmp(&cmdq->c_state, &cmd.c_state, sizeof(cmd.c_state))) {
		ret = cxi_cq_emit_c_state(cmdq->dev_cmdq, &cmd.c_state);
		if (ret) {
			CXIP_LOG_DBG("Failed to issue C_STATE command: %d\n",
				     ret);
			/* Return error according to Domain Resource
			 * Management
			 */
			ret = -FI_EAGAIN;
			goto err_unlock;
		}

		/* Update TXQ C_STATE */
		cmdq->c_state = cmd.c_state;
	}

	memset(&cmd.idc_put, 0, sizeof(cmd.idc_put));
	cmd.idc_put.idc_header.dfa = dfa;
	ret = cxi_cq_emit_idc_put(cmdq->dev_cmdq, &cmd.idc_put,
				  buffer, buflen);
	if (ret) {
		CXIP_LOG_DBG("Failed to write IDC: %d\n", ret);

		/* Return error according to Domain Resource Management
		 */
		ret = -FI_EAGAIN;
		goto err_unlock;
	}

	cxi_cq_ring(cmdq->dev_cmdq);
	ret = FI_SUCCESS;

err_unlock:
	fastlock_release(&cmdq->lock);
	return ret;
}

/****************************************************************************
 * Receive operations (posting buffers to collective PTE)
 */

static void _coll_rx_progress(struct cxip_req *req, const union c_event *event)
{
	/* stub */
}

/* Report results of an event.
 */
static void _coll_rx_req_report(struct cxip_req *req)
{
	size_t overflow;
	int err, ret;

	req->flags &= (FI_RECV | FI_COMPLETION | FI_COLLECTIVE);

	/* Interpret results */
	overflow = req->coll.hw_req_len - req->data_len;
	if (req->coll.rc == C_RC_OK && !overflow) {
		if (req->flags & FI_COMPLETION) {
			ret = cxip_cq_req_complete(req);
			if (ret != FI_SUCCESS)
				CXIP_LOG_ERROR(
				    "Failed to report completion: %d\n",
				    ret);
		}

		if (req->coll.ep_obj->coll.rx_cntr) {
			ret = cxip_cntr_mod(req->coll.ep_obj->coll.rx_cntr, 1,
					    false, false);
			if (ret)
				CXIP_LOG_ERROR("cxip_cntr_mod returned: %d\n",
					       ret);
		}
	} else {
		if (overflow) {
			err = FI_EMSGSIZE;
			CXIP_LOG_DBG("Request truncated: %p (err: %d, %s)\n",
				     req, err,
				     cxi_rc_to_str(req->coll.rc));
		} else {
			err = FI_EIO;
			CXIP_LOG_ERROR("Request error: %p (err: %d, %s)\n",
				       req, err,
				       cxi_rc_to_str(req->coll.rc));
		}

		ret = cxip_cq_req_error(req, overflow, err, req->coll.rc,
					NULL, 0);
		if (ret != FI_SUCCESS)
			CXIP_LOG_ERROR("Failed to report error: %d\n", ret);

		if (req->coll.ep_obj->coll.rx_cntr) {
			ret = cxip_cntr_mod(req->coll.ep_obj->coll.rx_cntr, 1,
					    false, true);
			if (ret)
				CXIP_LOG_ERROR("cxip_cntr_mod returned: %d\n",
					       ret);
		}
	}

	if (req->coll.mrecv_space < req->coll.ep_obj->coll.min_multi_recv) {
		struct cxip_ep_obj *ep_obj = req->coll.ep_obj;
		struct cxip_coll_buf *buf = req->coll.buf;

		/* Useful for testing */
		ep_obj->coll.buf_swap_cnt++;

		/* Hardware has silently unlinked this */
		cxip_cq_req_free(req);

		/* Re-use this buffer in the hardware */
		ret = _coll_recv(ep_obj, buf);
		if (ret != FI_SUCCESS) {
			CXIP_LOG_ERROR("Re-link buffer failed: %d\n", ret);
		}
	}
}

/* Callback for receive events.
 */
static int _coll_recv_cb(struct cxip_req *req, const union c_event *event)
{
	req->coll.rc = cxi_tgt_event_rc(event);
	switch (event->hdr.event_type) {
	case C_EVENT_LINK:
		/* Normally disabled */
		if (req->coll.rc != C_RC_OK) {
			CXIP_LOG_ERROR("LINK event error = %d\n",
				       req->coll.rc);
			break;
		}
		CXIP_LOG_DBG("LINK event seen\n");
		break;
	case C_EVENT_UNLINK:
		req->coll.rc = cxi_tgt_event_rc(event);
		if (req->coll.rc != C_RC_OK) {
			CXIP_LOG_ERROR("UNLINK event error = %d\n",
				       req->coll.rc);
			break;
		}
		CXIP_LOG_DBG("UNLINK event seen\n");
		break;
	case C_EVENT_PUT:
		req->coll.rc = cxi_tgt_event_rc(event);
		if (req->coll.rc != C_RC_OK) {
			CXIP_LOG_ERROR("PUT event error = %d\n",
				       req->coll.rc);
			break;
		}
		CXIP_LOG_DBG("PUT event seen\n");
		req->buf = (uint64_t)(CXI_IOVA_TO_VA(
					req->coll.buf->cxi_md->md,
					event->tgt_long.start));
		req->coll.mrecv_space -= event->tgt_long.mlength;
		req->coll.hw_req_len = event->tgt_long.rlength;
		req->data_len = event->tgt_long.mlength;
		_coll_rx_progress(req, event);
		_coll_rx_req_report(req);
		break;
	default:
		req->coll.rc = cxi_tgt_event_rc(event);
		CXIP_LOG_ERROR("Unexpected event type %d, error = %d\n",
			       event->hdr.event_type, req->coll.rc);
		break;
	}

	return FI_SUCCESS;
}

/* Inject a hardware LE append. Does not generate HW event.
 */
static int _hw_coll_recv(struct cxip_ep_obj *ep_obj, struct cxip_req *req)
{
	uint32_t le_flags;
	uint64_t recv_iova;
	int ret;

	/* Always set manage_local in Receive LEs. This makes Cassini ignore
	 * initiator remote_offset in all Puts.
	 */
	le_flags = C_LE_EVENT_LINK_DISABLE | C_LE_EVENT_UNLINK_DISABLE |
		   C_LE_OP_PUT | C_LE_MANAGE_LOCAL;

	recv_iova = CXI_VA_TO_IOVA(req->coll.buf->cxi_md->md,
				   (uint64_t)req->coll.buf->buffer);

	ret = cxip_pte_append(ep_obj->coll.pte,
			      recv_iova,
			      req->coll.buf->bufsiz,
			      req->coll.buf->cxi_md->md->lac,
			      C_PTL_LIST_PRIORITY,
			      req->req_id,
			      0, 0, 0,
			      ep_obj->coll.min_multi_recv,
			      le_flags, ep_obj->coll.rx_cntr,
			      ep_obj->coll.rx_cmdq,
			      true);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_ERROR("PTE append inject failed: %d\n", ret);
		return ret;
	}

	return FI_SUCCESS;
}

/* Append a receive buffer to the PTE.
 */
static ssize_t _coll_recv(struct cxip_ep_obj *ep_obj,
			  struct cxip_coll_buf *buf)
{
	struct cxip_req *req;
	int ret;

	if (buf->bufsiz && !buf->buffer)
		return -FI_EINVAL;

	/* Allocate and populate a new request
	 * Sets:
	 * - req->cq
	 * - req->req_id to request index
	 * - req->req_ctx to passed context (buf)
	 * - req->discard to false
	 * - Inserts into the cq->req_list
	 */
	req = cxip_cq_req_alloc(ep_obj->coll.rx_cq, true, buf);
	if (!req) {
		CXIP_LOG_DBG("Failed to allocate request\n");
		ret = -FI_ENOMEM;
		goto recv_unmap;
	}

	/* CQ event fields, set according to fi_cq.3
	 *   - set by provider
	 *   - returned to user in completion event
	 * uint64_t context;	// operation context
	 * uint64_t flags;	// operation flags
	 * uint64_t data_len;   // received data length
	 * uint64_t buf;	// receive buf offset
	 * uint64_t data;	// receive REMOTE_CQ_DATA
	 * uint64_t tag;	// receive tag value on matching interface
	 * fi_addr_t addr;	// sender address (if known) ???
	 */

	/* Request parameters */
	req->type = CXIP_REQ_COLL;
	req->flags = (FI_RECV | FI_COMPLETION | FI_COLLECTIVE);
	req->cb = _coll_recv_cb;
	req->triggered = false;
	req->trig_thresh = 0;
	req->trig_cntr = NULL;
	req->context = (uint64_t)buf;
	req->data_len = (uint64_t)buf->buflen;
	req->buf = (uint64_t)buf->buffer;
	req->data = 0;
	req->tag = 0;
	req->coll.ep_obj = ep_obj;
	req->coll.buf = buf;
	req->coll.mrecv_space = req->coll.buf->bufsiz;

	/* Returns FI_SUCCESS or FI_EAGAIN */
	ret = _hw_coll_recv(ep_obj, req);
	if (ret != FI_SUCCESS)
		goto recv_dequeue;

	return FI_SUCCESS;

recv_dequeue:
	cxip_cq_req_free(req);

recv_unmap:
	cxip_unmap(buf->cxi_md);
	return ret;
}

/* PTE state-change callback.
 */
static void _coll_pte_cb(struct cxip_pte *pte, enum c_ptlte_state state)
{
	struct cxip_ep_obj *ep_obj = (struct cxip_ep_obj *)pte->ctx;

	switch (state) {
	case C_PTLTE_ENABLED:
		ep_obj->coll.pte_state = C_PTLTE_ENABLED;
		break;
	case C_PTLTE_DISABLED:
		ep_obj->coll.pte_state = C_PTLTE_DISABLED;
		break;
	default:
		CXIP_LOG_ERROR("Unexpected state received: %u\n", state);
	}
}

/* Allocate a new PTE for collectives.
 */
static int _coll_pte_alloc(struct cxip_ep_obj *ep_obj,
			   struct cxip_pte **pte)
{
	struct cxi_pt_alloc_opts pt_opts = {
		.use_long_event = 1,
		.do_space_check = 1,
		.en_restricted_unicast_lm = 1,
	};
	uint64_t pid_idx;
	int ret;

	if (ep_obj->coll.is_netsim) {
		pid_idx = CXIP_PTL_IDX_RXC(CXIP_PTL_IDX_COLL);
		ret = cxip_pte_alloc(ep_obj->if_dom, ep_obj->coll.rx_cq->evtq,
				     pid_idx, false, &pt_opts, _coll_pte_cb,
				     ep_obj, pte);
		if (ret != FI_SUCCESS) {
			CXIP_LOG_ERROR("PTE allocation error = %d\n", ret);
			return ret;
		}
	} else {
		CXIP_LOG_ERROR("Cassini multicast not supported\n");
		return -FI_EOPNOTSUPP;
	}

	return FI_SUCCESS;
}

/* Free an existing PTE for collectives
 */
static void _coll_pte_free(struct cxip_ep_obj *ep_obj)
{
	if (ep_obj->coll.pte)
		cxip_pte_free(ep_obj->coll.pte);
	ep_obj->coll.pte = NULL;
}

/* Connect all TX/RX resources to STD EP resources. Enable the PTE and other
 * control structures. Wait for completion.
 */
static int _coll_pte_enable(struct cxip_ep_obj *ep_obj, uint32_t drop_count)
{
	union c_cmdu cmd = {};
	int ret;

	/* First call allocates a new PTE, persists until close */
	if (!ep_obj->coll.pte) {
		ret = _coll_pte_alloc(ep_obj, &ep_obj->coll.pte);
		if (ret != FI_SUCCESS)
			return ret;
	}

	cmd.command.opcode = C_CMD_TGT_SETSTATE;
	cmd.set_state.ptlte_index = ep_obj->coll.pte->pte->ptn;
	cmd.set_state.ptlte_state = C_PTLTE_ENABLED;
	cmd.set_state.drop_count = drop_count;

	ret = cxi_cq_emit_target(ep_obj->coll.rx_cmdq->dev_cmdq, &cmd);
	if (ret) {
		CXIP_LOG_ERROR("Failed to enqueue PTE ENABLE: %d\n", ret);
		return -FI_EAGAIN;
	}

	cxi_cq_ring(ep_obj->coll.rx_cmdq->dev_cmdq);
	CXIP_LOG_DBG("PTE enable started\n");

	do {
		sched_yield();
		cxip_cq_progress(ep_obj->coll.rx_cq);
	} while (ep_obj->coll.pte_state != C_PTLTE_ENABLED);
	CXIP_LOG_DBG("PTE enable completed\n");

	return FI_SUCCESS;
}

/* Disable the collectives PTE. Wait for completion.
 */
static int _coll_pte_disable(struct cxip_ep_obj *ep_obj)
{
	union c_cmdu cmd = {};
	int ret;

	if (!ep_obj->coll.pte)
		return FI_SUCCESS;

	cmd.command.opcode = C_CMD_TGT_SETSTATE;
	cmd.set_state.ptlte_index = ep_obj->coll.pte->pte->ptn;
	cmd.set_state.ptlte_state = C_PTLTE_DISABLED;

	ret = cxi_cq_emit_target(ep_obj->coll.rx_cmdq->dev_cmdq, &cmd);
	if (ret) {
		CXIP_LOG_ERROR("Failed to enqueue PTE DISABLE: %d\n", ret);
		return -FI_EAGAIN;
	}

	cxi_cq_ring(ep_obj->coll.rx_cmdq->dev_cmdq);
	CXIP_LOG_DBG("PTE disable started\n");

	do {
		sched_yield();
		cxip_cq_progress(ep_obj->coll.rx_cq);
	} while (ep_obj->coll.pte_state != C_PTLTE_DISABLED);
	CXIP_LOG_DBG("PTE disable completed\n");

	return FI_SUCCESS;
}

/* Destroy and unmap all buffers used by the collectives PTE.
 */
static void _coll_destroy_buffers(struct cxip_ep_obj *ep_obj)
{
	struct dlist_entry *list = &ep_obj->coll.buf_list;
	struct cxip_coll_buf *buf;

	while (!dlist_empty(list)) {
		dlist_pop_front(list, struct cxip_coll_buf, buf, buf_entry);
		cxip_unmap(buf->cxi_md);
		free(buf);
	}
}

/* Adds 'count' buffers of 'size' bytes to the collecives PTE. This succeeds
 * fully, or it fails and removes all buffers.
 */
static int _coll_add_buffers(struct cxip_ep_obj *ep_obj, size_t size,
			     size_t count)
{
	struct cxip_coll_buf *buf;
	int ret, i;

	CXIP_LOG_DBG("Adding %ld buffers of size %ld\n", count, size);
	for (i = 0; i < count; i++) {
		buf = calloc(1, sizeof(*buf) + size);
		if (!buf) {
			ret = -FI_ENOMEM;
			goto out;
		}
		ret = cxip_map(ep_obj->domain, (void *)buf->buffer, size,
			       &buf->cxi_md);
		if (ret)
			goto del_msg;
		buf->bufsiz = size;
		dlist_insert_tail(&buf->buf_entry, &ep_obj->coll.buf_list);

		ret = _coll_recv(ep_obj, buf);
		if (ret) {
			CXIP_LOG_ERROR("Add buffer %d of %ld: %d\n",
				       i, count, ret);
			goto out;
		}
	}
	return FI_SUCCESS;
del_msg:
	free(buf);
out:
	_coll_destroy_buffers(ep_obj);
	return ret;
}

/****************************************************************************
 * Initialize, configure, enable, disable, and close the collective PTE.
 */

/**
 * Initialize the collectives structures.
 *
 * Must be done during EP initialization.
 *
 * @param ep_obj - EP object
 *
 * @return int - FI return code
 */
int cxip_coll_init(struct cxip_ep_obj *ep_obj)
{
	ep_obj->coll.rx_cmdq = NULL;
	ep_obj->coll.tx_cmdq = NULL;
	ep_obj->coll.rx_cntr = NULL;
	ep_obj->coll.tx_cntr = NULL;
	ep_obj->coll.rx_cq = NULL;
	ep_obj->coll.tx_cq = NULL;
	ep_obj->coll.pte = NULL;

	ofi_atomic_initialize32(&ep_obj->coll.mc_count, 0);
	dlist_init(&ep_obj->coll.buf_list);
	dlist_init(&ep_obj->coll.mc_list);
	fastlock_init(&ep_obj->coll.lock);
	ep_obj->coll.min_multi_recv = CXIP_COLL_MIN_FREE;
	ep_obj->coll.pte_state = C_PTLTE_DISABLED;
	ep_obj->coll.buffer_count = CXIP_COLL_MIN_RX_BUFS;
	ep_obj->coll.buffer_size = CXIP_COLL_MIN_RX_SIZE;
	ep_obj->coll.buf_swap_cnt = 0;
	ep_obj->coll.enabled = false;

	return FI_SUCCESS;
}

/**
 * Enable collectives.
 *
 * Must be preceded by cxip_coll_init().
 *
 * This uses the coll.buffer_count and coll.buffer_size fields to set up the
 * reusable locally_managed buffers for performing collectives.
 *
 * There is only one collectives object. It can be safely enabled multiple
 * times in a multithreaded environment. It may return -FI_EAGAIN if another
 * thread is attempting to enable it.
 *
 * @param ep_obj - EP object
 *
 * @return int - FI return code
 */
int cxip_coll_enable(struct cxip_ep_obj *ep_obj)
{
	int ret;

	if (ep_obj->coll.buffer_count < CXIP_COLL_MIN_RX_BUFS) {
		CXIP_LOG_ERROR("Buffer count %ld < minimum (%d):\n",
			       ep_obj->coll.buffer_count,
			       CXIP_COLL_MIN_RX_BUFS);
		return -FI_EINVAL;
	}

	if (ep_obj->coll.buffer_size < CXIP_COLL_MIN_RX_SIZE) {
		CXIP_LOG_ERROR("Buffer size %ld < minimum (%d):\n",
			       ep_obj->coll.buffer_size,
			       CXIP_COLL_MIN_RX_SIZE);
		return -FI_EINVAL;
	}

	if (!ep_obj->enabled) {
		CXIP_LOG_ERROR("Parent STD EP not enabled\n");
		return -FI_EOPBADSTATE;
	}

	/* A read-only or write-only endpoint is legal */
	if (!(ofi_recv_allowed(ep_obj->rxcs[0]->attr.caps) &&
	      ofi_send_allowed(ep_obj->txcs[0]->attr.caps))) {
		CXIP_LOG_DBG("EP not recv/send, collectives not enabled\n");
		return FI_SUCCESS;
	}

	ep_obj->coll.is_netsim =
			(ep_obj->domain->iface->info->device_platform ==
			 CXI_PLATFORM_NETSIM);

	fastlock_acquire(&ep_obj->coll.lock);

	if (ep_obj->coll.enabled) {
		ret = FI_SUCCESS;
		goto rls_lock;
	}

	/* Bind all STD EP objects to the coll object */
	ep_obj->coll.rx_cmdq = ep_obj->rxcs[0]->rx_cmdq;
	ep_obj->coll.tx_cmdq = ep_obj->txcs[0]->tx_cmdq;
	ep_obj->coll.rx_cntr = ep_obj->rxcs[0]->recv_cntr;
	ep_obj->coll.tx_cntr = ep_obj->txcs[0]->send_cntr;
	ep_obj->coll.rx_cq = ep_obj->rxcs[0]->recv_cq;
	ep_obj->coll.tx_cq = ep_obj->txcs[0]->send_cq;

	ret = _coll_pte_enable(ep_obj, CXIP_PTE_IGNORE_DROPS);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_ERROR("PTE enable failed = %d\n", ret);
		ret = -FI_EDOMAIN;
		goto rls_lock;
	}

	ret = _coll_add_buffers(ep_obj,
				ep_obj->coll.buffer_size,
				ep_obj->coll.buffer_count);
	if (ret != FI_SUCCESS)
		goto rls_lock;

	ep_obj->coll.enabled = true;

rls_lock:
	fastlock_release(&ep_obj->coll.lock);
	return ret;
}

/**
 * Disable collectives.
 *
 * @param ep_obj - EP object
 *
 * @return int - FI return code
 */
int cxip_coll_disable(struct cxip_ep_obj *ep_obj)
{
	if (!ep_obj->coll.enabled)
		return FI_SUCCESS;

	ep_obj->coll.enabled = false;

	return FI_SUCCESS;
}

/**
 * Closes collectives.
 *
 * Must be done during EP close.
 *
 * @param ep_obj - EP object
 */
int cxip_coll_close(struct cxip_ep_obj *ep_obj)
{
	ep_obj->coll.enabled = false;

	if (ofi_atomic_get32(&ep_obj->coll.mc_count) != 0) {
		CXIP_LOG_ERROR("MC objects pending\n");
		return -FI_EBUSY;
	}

	_coll_pte_disable(ep_obj);
	_coll_destroy_buffers(ep_obj);
	_coll_pte_free(ep_obj);
	fastlock_destroy(&ep_obj->coll.lock);

	return FI_SUCCESS;
}

struct fi_ops_collective cxip_collective_ops = {
	.size = sizeof(struct fi_ops_collective),
	.barrier = fi_coll_no_barrier,
	.broadcast = fi_coll_no_broadcast,
	.alltoall = fi_coll_no_alltoall,
	.allreduce = fi_coll_no_allreduce,
	.allgather = fi_coll_no_allgather,
	.reduce_scatter = fi_coll_no_reduce_scatter,
	.reduce = fi_coll_no_reduce,
	.scatter = fi_coll_no_scatter,
	.gather = fi_coll_no_gather,
	.msg = fi_coll_no_msg,
};

struct fi_ops_collective cxip_no_collective_ops = {
	.size = sizeof(struct fi_ops_collective),
	.barrier = fi_coll_no_barrier,
	.broadcast = fi_coll_no_broadcast,
	.alltoall = fi_coll_no_alltoall,
	.allreduce = fi_coll_no_allreduce,
	.allgather = fi_coll_no_allgather,
	.reduce_scatter = fi_coll_no_reduce_scatter,
	.reduce = fi_coll_no_reduce,
	.scatter = fi_coll_no_scatter,
	.gather = fi_coll_no_gather,
	.msg = fi_coll_no_msg,
};

int cxip_join_collective(struct fid_ep *ep, fi_addr_t coll_addr,
			 const struct fid_av_set *coll_av_set,
			 uint64_t flags, struct fid_mc **mc, void *context)
{
	/* stub */
	return FI_SUCCESS;
}
