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

static inline bool is_netsim(struct cxip_ep_obj *ep_obj)
{
	return (ep_obj->domain->iface->info->device_platform == 1);
}

static ssize_t _coll_recv(struct cxip_coll_pte *coll_pte,
			  struct cxip_coll_buf *buf);

/****************************************************************************
 * SEND operation (restricted IDC Put to a remote PTE)
 */

/* Generate a dfa and index extension for a reduction.
 */
static int _gen_tx_dfa(struct cxip_coll_reduction *reduction,
		       fi_addr_t dest_addr, union c_fab_addr *dfa,
		       uint8_t *index_ext, bool *is_mcast)
{
	struct cxip_ep_obj *ep_obj;
	struct cxip_av_set *av_set;
	struct cxip_addr dest_caddr;
	int pid_bits;
	int idx_ext;
	int ret;

	ep_obj = reduction->mc_obj->ep_obj;
	av_set = reduction->mc_obj->av_set;

	switch (av_set->comm_key.type) {
	case COMM_KEY_MULTICAST:
		/* - dest_addr == multicast ID
		 * - idx_ext == 0
		 * - dfa == multicast destination
		 * - index_ext == 0
		 */
		if (is_netsim(ep_obj))
			return -FI_EINVAL;
		dest_addr = av_set->comm_key.dest_addr;
		idx_ext = 0;
		cxi_build_mcast_dfa(dest_addr,
				    reduction->red_id, idx_ext,
				    dfa, index_ext);
		*is_mcast = true;
		break;
	case COMM_KEY_UNICAST:
		/* - dest_addr == av_set rank of destination NIC
		 * - idx_ext == CXIP_PTL_IDX_COLL
		 * - dfa = remote nic
		 * - index_ext == CXIP_PTL_IDX_COLL
		 */
		if (dest_addr >= av_set->fi_addr_cnt)
			return -FI_EINVAL;
		dest_addr = av_set->fi_addr_ary[dest_addr];
		ret = _cxip_av_lookup(ep_obj->av, dest_addr, &dest_caddr);
		if (ret != FI_SUCCESS)
			return ret;
		pid_bits = ep_obj->domain->iface->dev->info.pid_bits;
		idx_ext = CXIP_PTL_IDX_RXC(CXIP_PTL_IDX_COLL);
		cxi_build_dfa(dest_caddr.nic, dest_caddr.pid, pid_bits,
			      idx_ext, dfa, index_ext);
		*is_mcast = false;
		break;
	case COMM_KEY_RANK:
		/* - dest_addr == reduction rank of mc_obj
		 * - idx_ext == CXIP_PTL_IDX_COLL
		 * - dfa == source NIC
		 * - index_ext == 16+idx_ext
		 */
		if (dest_addr >= av_set->fi_addr_cnt)
			return -FI_EINVAL;
		dest_caddr = ep_obj->src_addr;
		pid_bits = ep_obj->domain->iface->dev->info.pid_bits;
		idx_ext = 16 + dest_addr;
		cxi_build_dfa(dest_caddr.nic, dest_caddr.pid, pid_bits,
			      idx_ext, dfa, index_ext);
		*is_mcast = false;
		break;
	default:
		return -FI_EINVAL;
	}
	return FI_SUCCESS;
}

/**
 * Issue a restricted IDC Put to the destination address.
 *
 * Exported for unit testing.
 *
 * @param reduction - reduction object
 * @param dest_addr - comm_key destination address
 * @param buffer - buffer containing data to send
 * @param buflen - byte count of data in buffer
 *
 * @return int - return code
 */
int cxip_coll_send(struct cxip_coll_reduction *reduction,
		   fi_addr_t dest_addr, const void *buffer, size_t buflen)
{
	struct cxip_ep_obj *ep_obj;
	struct cxip_cmdq *cmdq;
	union c_fab_addr dfa;
	uint8_t index_ext;
	uint8_t pid_bits;
	bool is_mcast;
	int ret;

	if (buflen && !buffer)
		return -FI_EINVAL;

	ep_obj = reduction->mc_obj->ep_obj;
	if (buflen >= ep_obj->coll.min_multi_recv)
		return -FI_EMSGSIZE;

	ret = _gen_tx_dfa(reduction, dest_addr, &dfa, &index_ext, &is_mcast);
	if (ret)
		return ret;

	/* pid_bits needed to obtain initiator address */
	pid_bits = ep_obj->domain->iface->dev->info.pid_bits;
	cmdq = ep_obj->coll.tx_cmdq;

	union c_cmdu cmd = {};
	cmd.c_state.event_send_disable = 1;
	cmd.c_state.event_success_disable = 1;
	cmd.c_state.restricted = 1;
	cmd.c_state.reduction = is_mcast;
	cmd.c_state.index_ext = index_ext;
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

	ofi_atomic_inc32(&reduction->mc_obj->send_cnt);

err_unlock:
	fastlock_release(&cmdq->lock);
	return ret;
}

/****************************************************************************
 * RECV operation (restricted IDC Put to a local PTE)
 */

/* Report success/error results of an RX event through RX CQ / counters, and
 * roll over the buffers if appropriate.
 *
 * NOTE: req may be invalid after this call.
 */
static void _coll_rx_req_report(struct cxip_req *req)
{
	size_t overflow;
	int err, ret;

	req->flags &= (FI_RECV | FI_COMPLETION | FI_COLLECTIVE);

	/* Interpret results */
	overflow = req->coll.hw_req_len - req->data_len;
	if (req->coll.rc == C_RC_OK && !overflow) {
		/* success */
		if (req->flags & FI_COMPLETION) {
			ret = cxip_cq_req_complete(req);
			if (ret != FI_SUCCESS)
				CXIP_LOG_ERROR(
				    "Failed to report completion: %d\n",
				    ret);
		}

		if (req->coll.coll_pte->ep_obj->coll.rx_cntr) {
			ret = cxip_cntr_mod(
				req->coll.coll_pte->ep_obj->coll.rx_cntr, 1,
				false, false);
			if (ret)
				CXIP_LOG_ERROR("cxip_cntr_mod returned: %d\n",
					       ret);
		}
	} else {
		/* failure */
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

		if (req->coll.coll_pte->ep_obj->coll.rx_cntr) {
			ret = cxip_cntr_mod(
				req->coll.coll_pte->ep_obj->coll.rx_cntr, 1,
				false, true);
			if (ret)
				CXIP_LOG_ERROR("cxip_cntr_mod returned: %d\n",
					       ret);
		}
	}

	if (req->coll.mrecv_space <
	    req->coll.coll_pte->ep_obj->coll.min_multi_recv) {
		struct cxip_coll_pte *coll_pte = req->coll.coll_pte;
		struct cxip_coll_buf *buf = req->coll.coll_buf;

		/* Useful for testing */
		coll_pte->buf_swap_cnt++;

		/* Re-use this buffer in the hardware */
		ret = _coll_recv(coll_pte, buf);
		if (ret != FI_SUCCESS)
			CXIP_LOG_ERROR("Re-link buffer failed: %d\n", ret);

		/* Hardware has silently unlinked this */
		cxip_cq_req_free(req);
	}
}

/* Evaluate PUT request to see if this is a reduction packet.
 */
static void _coll_rx_progress(struct cxip_req *req,
			      const union c_event *event)
{
	struct cxip_coll_mc *mc_obj;

	/* sanity check */
	if (req->type != CXIP_REQ_COLL)
		return;

	mc_obj = req->coll.coll_pte->mc_obj;
	ofi_atomic_inc32(&mc_obj->recv_cnt);
}

/* Event-handling callback for posted receive buffers.
 */
static int _coll_recv_cb(struct cxip_req *req, const union c_event *event)
{
	req->coll.rc = cxi_tgt_event_rc(event);
	switch (event->hdr.event_type) {
	case C_EVENT_LINK:
		/* Normally disabled, errors only */
		if (req->coll.rc != C_RC_OK) {
			CXIP_LOG_ERROR("LINK event error = %d\n",
				       req->coll.rc);
			break;
		}
		CXIP_LOG_DBG("LINK event seen\n");
		break;
	case C_EVENT_UNLINK:
		/* Normally disabled, errors only */
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
					req->coll.coll_buf->cxi_md->md,
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

/* Inject a hardware LE append. Does not generate HW LINK event unless error.
 */
static int _hw_coll_recv(struct cxip_coll_pte *coll_pte, struct cxip_req *req)
{
	uint32_t le_flags;
	uint64_t recv_iova;
	int ret;

	/* Always set manage_local in Receive LEs. This makes Cassini ignore
	 * initiator remote_offset in all Puts.
	 */
	le_flags = C_LE_EVENT_LINK_DISABLE | C_LE_EVENT_UNLINK_DISABLE |
		   C_LE_OP_PUT | C_LE_MANAGE_LOCAL;

	recv_iova = CXI_VA_TO_IOVA(req->coll.coll_buf->cxi_md->md,
				   (uint64_t)req->coll.coll_buf->buffer);

	ret = cxip_pte_append(coll_pte->pte,
			      recv_iova,
			      req->coll.coll_buf->bufsiz,
			      req->coll.coll_buf->cxi_md->md->lac,
			      C_PTL_LIST_PRIORITY,
			      req->req_id,
			      0, 0, 0,
			      coll_pte->ep_obj->min_multi_recv,
			      le_flags, coll_pte->ep_obj->coll.rx_cntr,
			      coll_pte->ep_obj->coll.rx_cmdq,
			      true);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_ERROR("PTE append inject failed: %d\n", ret);
		return ret;
	}

	return FI_SUCCESS;
}

/* Append a receive buffer to the PTE, with callback to handle receives.
 */
static ssize_t _coll_recv(struct cxip_coll_pte *coll_pte,
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
	req = cxip_cq_req_alloc(coll_pte->ep_obj->coll.rx_cq, 1, buf);
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
	req->data_len = (uint64_t)0;
	req->buf = (uint64_t)buf->buffer;
	req->data = 0;
	req->tag = 0;
	req->coll.coll_pte = coll_pte;
	req->coll.coll_buf = buf;
        req->coll.mrecv_space = req->coll.coll_buf->bufsiz;

	/* Returns FI_SUCCESS or FI_EAGAIN */
	ret = _hw_coll_recv(coll_pte, req);
	if (ret != FI_SUCCESS)
		goto recv_dequeue;

	return FI_SUCCESS;

recv_dequeue:
	cxip_cq_req_free(req);

recv_unmap:
	cxip_unmap(buf->cxi_md);
	return ret;
}

/****************************************************************************
 * PTE management functions.
 */

/* PTE state-change callback.
 */
static void _coll_pte_cb(struct cxip_pte *pte, enum c_ptlte_state state)
{
	struct cxip_coll_pte *coll_pte = (struct cxip_coll_pte *)pte->ctx;

	switch (state) {
	case C_PTLTE_ENABLED:
		coll_pte->pte_state = C_PTLTE_ENABLED;
		break;
	case C_PTLTE_DISABLED:
		coll_pte->pte_state = C_PTLTE_DISABLED;
		break;
	default:
		CXIP_LOG_ERROR("Unexpected state received: %u\n", state);
	}
}

/* Destroy a collective multicast PTE.
 */
static void _coll_pte_free(struct cxip_coll_pte *coll_pte)
{
	if (coll_pte->pte)
		cxip_pte_free(coll_pte->pte);
	free(coll_pte);
}

/* Enable a collective PTE. Wait for completion.
 */
static int _coll_pte_enable(struct cxip_coll_pte *coll_pte,
			    uint32_t drop_count)
{
	union c_cmdu cmd = {};
	int ret;

	if (!coll_pte->pte) {
		CXIP_LOG_ERROR("Collective PTE not allocated\n");
		return -FI_EINVAL;
	}

	cmd.command.opcode = C_CMD_TGT_SETSTATE;
	cmd.set_state.ptlte_index = coll_pte->pte->pte->ptn;
	cmd.set_state.ptlte_state = C_PTLTE_ENABLED;
	cmd.set_state.drop_count = drop_count;

	ret = cxi_cq_emit_target(coll_pte->ep_obj->coll.rx_cmdq->dev_cmdq,
				 &cmd);
	if (ret) {
		CXIP_LOG_ERROR("Failed to enqueue PTE ENABLE: %d\n", ret);
		return -FI_EAGAIN;
	}

	cxi_cq_ring(coll_pte->ep_obj->coll.rx_cmdq->dev_cmdq);
	CXIP_LOG_DBG("PTE enable started\n");

	do {
		sched_yield();
		cxip_cq_progress(coll_pte->ep_obj->coll.rx_cq);
	} while (coll_pte->pte_state != C_PTLTE_ENABLED);
	CXIP_LOG_DBG("PTE enable completed\n");

	return FI_SUCCESS;
}

/* Disable a collective PTE. Wait for completion.
 */
static int _coll_pte_disable(struct cxip_coll_pte *coll_pte)
{
	union c_cmdu cmd = {};
	int ret;

	if (!coll_pte->pte) {
		CXIP_LOG_ERROR("Collective PTE not allocated\n");
		return -FI_EINVAL;
	}

	if (coll_pte->pte_state != C_PTLTE_ENABLED)
		return FI_SUCCESS;

	cmd.command.opcode = C_CMD_TGT_SETSTATE;
	cmd.set_state.ptlte_index = coll_pte->pte->pte->ptn;
	cmd.set_state.ptlte_state = C_PTLTE_DISABLED;

	ret = cxi_cq_emit_target(coll_pte->ep_obj->coll.rx_cmdq->dev_cmdq,
				 &cmd);
	if (ret) {
		CXIP_LOG_ERROR("Failed to enqueue PTE DISABLE: %d\n", ret);
		return -FI_EAGAIN;
	}

	cxi_cq_ring(coll_pte->ep_obj->coll.rx_cmdq->dev_cmdq);

	do {
		sched_yield();
		cxip_cq_progress(coll_pte->ep_obj->coll.rx_cq);
	} while (coll_pte->pte_state != C_PTLTE_DISABLED);
	CXIP_LOG_DBG("PTE disable completed\n");

	return FI_SUCCESS;
}

/* Destroy and unmap all buffers used by the collectives PTE.
 */
static void _coll_destroy_buffers(struct cxip_coll_pte *coll_pte)
{
	struct dlist_entry *list = &coll_pte->buf_list;
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
static int _coll_add_buffers(struct cxip_coll_pte *coll_pte, size_t size,
			     size_t count)
{
	struct cxip_coll_buf *buf;
	int ret, i;

	if (count < CXIP_COLL_MIN_RX_BUFS) {
		CXIP_LOG_ERROR("Buffer count %ld < minimum (%d):\n",
			       count, CXIP_COLL_MIN_RX_BUFS);
		return -FI_EINVAL;
	}

	if (size < CXIP_COLL_MIN_RX_SIZE) {
		CXIP_LOG_ERROR("Buffer size %ld < minimum (%d):\n",
			       size, CXIP_COLL_MIN_RX_SIZE);
		return -FI_EINVAL;
	}

	CXIP_LOG_DBG("Adding %ld buffers of size %ld\n", count, size);
	for (i = 0; i < count; i++) {
		buf = calloc(1, sizeof(*buf) + size);
		if (!buf) {
			ret = -FI_ENOMEM;
			goto out;
		}
		ret = cxip_map(coll_pte->ep_obj->domain, (void *)buf->buffer,
			       size, &buf->cxi_md);
		if (ret)
			goto del_msg;
		buf->bufsiz = size;
		dlist_insert_tail(&buf->buf_entry, &coll_pte->buf_list);

		ret = _coll_recv(coll_pte, buf);
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
	_coll_destroy_buffers(coll_pte);
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
	ep_obj->coll.min_multi_recv = CXIP_COLL_MIN_FREE;
	ep_obj->coll.buffer_count = CXIP_COLL_MIN_RX_BUFS;
	ep_obj->coll.buffer_size = CXIP_COLL_MIN_RX_SIZE;

	ofi_atomic_initialize32(&ep_obj->coll.mc_count, 0);
	fastlock_init(&ep_obj->coll.lock);

	return FI_SUCCESS;
}

/**
 * Enable collectives.
 *
 * Must be preceded by cxip_coll_init(), and EP must be enabled.
 *
 * There is only one collectives object associated with an EP. It can be safely
 * enabled multiple times.
 *
 * @param ep_obj - EP object
 *
 * @return int - FI return code
 */
int cxip_coll_enable(struct cxip_ep_obj *ep_obj)
{
	if (ep_obj->coll.enabled)
		return FI_SUCCESS;

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

	/* Sanity checks */
	if (ep_obj->coll.buffer_size == 0)
		return -FI_EINVAL;
	if (ep_obj->coll.buffer_count == 0)
		return -FI_EINVAL;
	if (ep_obj->coll.min_multi_recv == 0)
		return -FI_EINVAL;
	if (ep_obj->coll.min_multi_recv >= ep_obj->coll.buffer_size)
		return -FI_EINVAL;

	/* Bind all STD EP objects to the coll object */
	ep_obj->coll.rx_cmdq = ep_obj->rxcs[0]->rx_cmdq;
	ep_obj->coll.tx_cmdq = ep_obj->txcs[0]->tx_cmdq;
	ep_obj->coll.rx_cntr = ep_obj->rxcs[0]->recv_cntr;
	ep_obj->coll.tx_cntr = ep_obj->txcs[0]->send_cntr;
	ep_obj->coll.rx_cq = ep_obj->rxcs[0]->recv_cq;
	ep_obj->coll.tx_cq = ep_obj->txcs[0]->send_cq;

	ep_obj->coll.enabled = true;

	return FI_SUCCESS;
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
 * Closes collectives and cleans up.
 *
 * Must be done during EP close.
 *
 * @param ep_obj - EP object
 */
int cxip_coll_close(struct cxip_ep_obj *ep_obj)
{
	if (ofi_atomic_get32(&ep_obj->coll.mc_count) != 0) {
		CXIP_LOG_ERROR("MC objects pending\n");
		return -FI_EBUSY;
	}

	fastlock_destroy(&ep_obj->coll.lock);

	return FI_SUCCESS;
}

/* Post a join completion request to the EP EQ.
 */
static int _post_join_complete(struct cxip_coll_mc *mc_obj, void *context)
{
	/* signals join completion by writing to the endpoint EQ */
	struct fi_eq_entry entry;
	int ret;

	memset(&entry, 0, sizeof(entry));
	entry.fid = &mc_obj->mc_fid.fid;
	entry.context = context;

	ret = ofi_eq_write(&mc_obj->ep_obj->eq->util_eq.eq_fid,
			   FI_JOIN_COMPLETE, &entry,
			   sizeof(entry), FI_COLLECTIVE);
	if (ret < 0)
		return ret;

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

/* Close a multicast object.
 */
static int _close_mc(struct fid *fid)
{
	struct cxip_coll_mc *mc_obj;
	int ret;

	mc_obj = container_of(fid, struct cxip_coll_mc, mc_fid.fid);

	do {
		ret = _coll_pte_disable(mc_obj->coll_pte);
	} while (ret == -FI_EAGAIN);

	_coll_destroy_buffers(mc_obj->coll_pte);
	_coll_pte_free(mc_obj->coll_pte);

	mc_obj->av_set->mc_obj = NULL;
	ofi_atomic_dec32(&mc_obj->av_set->ref);
	ofi_atomic_dec32(&mc_obj->ep_obj->coll.mc_count);
	free(mc_obj);

	return FI_SUCCESS;
}

static struct fi_ops mc_ops = {
	.size = sizeof(struct fi_ops),
	.close = _close_mc,
};

/* Find rank of a NIC identifier in the av_set.
 */
static int _find_nic_rank(struct cxip_av_set *av_set, struct cxip_addr *myaddr)
{
	struct cxip_addr caddr;
	int rank;
	int ret;

	for (rank = 0; rank < av_set->fi_addr_cnt; rank++) {
		ret = _cxip_av_lookup(av_set->cxi_av,
				      av_set->fi_addr_ary[rank], &caddr);
		if (ret) {
			// make noise
			return -FI_EINVAL;
		}
		if (CXIP_ADDR_EQUAL(caddr, *myaddr))
			return rank;
	}
	return -1;
}

/* Allocate a multicast object.
 */
static int _alloc_mc(struct cxip_ep_obj *ep_obj, struct cxip_av_set *av_set,
		     struct cxip_coll_mc **mc)
{
	struct cxi_pt_alloc_opts pt_opts = {
		.use_long_event = 1,
		.do_space_check = 1,
		.en_restricted_unicast_lm = 1,
	};
	struct cxip_coll_mc *mc_obj;
	struct cxip_coll_pte *coll_pte;
	uint64_t pid_idx;
	bool is_multicast;
	int mynode_rank;
	int mcast_id;
	int red_id;
	int state;
	int ret;

	if (av_set->mc_obj)
		return -FI_EINVAL;

	switch (av_set->comm_key.type) {
	case COMM_KEY_MULTICAST:
		mynode_rank = _find_nic_rank(av_set, &ep_obj->src_addr);
		if (mynode_rank < 0)
			return -FI_EINVAL;
		mcast_id = av_set->comm_key.dest_addr;
		pid_idx = av_set->comm_key.dest_addr;
		is_multicast = true;
		break;
	case COMM_KEY_UNICAST:
		/* Use unicast addresses to a known PTE index */
		mynode_rank = _find_nic_rank(av_set, &ep_obj->src_addr);
		if (mynode_rank < 0)
			return -FI_EINVAL;
		mcast_id = av_set->comm_key.dest_addr;
		pid_idx = CXIP_PTL_IDX_RXC(CXIP_PTL_IDX_COLL);
		is_multicast = false;
		break;
	case COMM_KEY_RANK:
		mynode_rank = av_set->comm_key.dest_addr;
		mcast_id = av_set->comm_key.dest_addr;
		pid_idx = 16 + av_set->comm_key.dest_addr;
		is_multicast = false;
		break;
	default:
		return -FI_EINVAL;
	}

	ret = -FI_ENOMEM;
	mc_obj = calloc(1, sizeof(*av_set->mc_obj));
	if (!mc_obj)
		return ret;

	coll_pte = calloc(1, sizeof(*coll_pte));
	if (!coll_pte)
		goto free_mc_obj;

	dlist_init(&coll_pte->buf_list);
	coll_pte->ep_obj = ep_obj;
	coll_pte->pte_state = C_PTLTE_DISABLED;

	ret = cxip_pte_alloc(ep_obj->if_dom, ep_obj->coll.rx_cq->evtq,
			     pid_idx, is_multicast, &pt_opts, _coll_pte_cb, coll_pte,
			     &coll_pte->pte);
	if (ret)
		goto free_coll_pte;

	do {
		ret = _coll_pte_enable(coll_pte, CXIP_PTE_IGNORE_DROPS);
	} while (ret == -FI_EAGAIN);
	if (ret)
		goto disable_coll_pte;

	ret = _coll_add_buffers(coll_pte,
				ep_obj->coll.buffer_size,
				ep_obj->coll.buffer_count);
	if (ret)
		goto disable_coll_pte;

	mc_obj->mc_fid.fid.fclass = FI_CLASS_MC;
	mc_obj->mc_fid.fid.context = mc_obj;
	mc_obj->mc_fid.fid.ops = &mc_ops;
	mc_obj->coll_pte = coll_pte;
	mc_obj->ep_obj = ep_obj;
	mc_obj->av_set = av_set;
	mc_obj->mcast_id = mcast_id;
	mc_obj->mynode_rank = mynode_rank;
	mc_obj->is_joined = true;
	state = CXIP_COLL_STATE_NONE;
	for (red_id = 0; red_id < CXIP_COLL_MAX_CONCUR; red_id++) {
		struct cxip_coll_reduction *reduction;

		reduction = &mc_obj->reduction[red_id];
		reduction->op_state = state;
		reduction->mc_obj = mc_obj;
		reduction->red_id = red_id;
		reduction->in_use = false;
	}
	fastlock_init(&mc_obj->lock);
	ofi_atomic_initialize32(&mc_obj->send_cnt, 0);
	ofi_atomic_initialize32(&mc_obj->recv_cnt, 0);
	ofi_atomic_initialize32(&mc_obj->pkt_cnt, 0);

	ofi_atomic_inc32(&ep_obj->coll.mc_count);
	ofi_atomic_inc32(&av_set->ref);
	av_set->mc_obj = mc_obj;
	coll_pte->mc_obj = mc_obj;

	*mc = av_set->mc_obj;

	return FI_SUCCESS;

disable_coll_pte:
	_coll_pte_disable(coll_pte);
free_coll_pte:
	_coll_pte_free(coll_pte);
free_mc_obj:
	free(mc_obj);
	return ret;
}

void cxip_coll_reset_mc_ctrs(struct cxip_coll_mc *mc_obj)
{
	fastlock_acquire(&mc_obj->lock);
	ofi_atomic_set32(&mc_obj->send_cnt, 0);
	ofi_atomic_set32(&mc_obj->recv_cnt, 0);
	ofi_atomic_set32(&mc_obj->pkt_cnt, 0);
	fastlock_release(&mc_obj->lock);
}

/**
 * fi_join_collective() implementation.
 *
 * @param ep - endpoint
 * @param addr - pointer to struct fi_collective_addr
 * @param flags - collective flags
 * @param mc - returned multicast object
 * @param context - user-defined context
 *
 * @return int - return code
 */
int cxip_join_collective(struct fid_ep *ep, fi_addr_t coll_addr,
			 const struct fid_av_set *coll_av_set,
			 uint64_t flags, struct fid_mc **mc, void *context)
{
	struct cxip_ep *cxi_ep;
	struct cxip_av_set *av_set;
	struct cxip_coll_mc *mc_obj;
	int ret;

	cxi_ep = container_of(ep, struct cxip_ep, ep.fid);
	av_set = container_of(coll_av_set, struct cxip_av_set, av_set_fid);

	if (!cxi_ep->ep_obj->coll.enabled) {
		CXIP_LOG_ERROR("Multicast not enabled\n");
		return -FI_EOPBADSTATE;
	}

	if (coll_addr != FI_ADDR_NOTAVAIL) {
		/* provider-managed not supported */
		return -FI_EINVAL;
	} else if (av_set->comm_key.type == COMM_KEY_NONE) {
		CXIP_LOG_ERROR("av_set comm_key not provided\n");
		return -FI_EINVAL;
	}

	ret = _alloc_mc(cxi_ep->ep_obj, av_set, &mc_obj);
	if (ret)
		return ret;

	ret = _post_join_complete(mc_obj, context);
	if (ret)
		return ret;

	*mc = &mc_obj->mc_fid;

	return FI_SUCCESS;
}
