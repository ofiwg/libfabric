/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2021 Hewlett Packard Enterprise Development LP
 */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <endian.h>
#include <sys/time.h>
#include <sys/types.h>

#include <ofi_list.h>
#include <ofi.h>
#include <fenv.h>
#include <xmmintrin.h>

#include "cxip.h"

#define CXIP_DBG(...) _CXIP_DBG(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXIP_INFO(...) _CXIP_INFO(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_EP_CTRL, __VA_ARGS__)

#ifdef	_do_trace_
#define _trc_(...) printf(__VA_ARGS__)
#else
#define _trc_(...)
#endif

/****************************************************************************
 * Point-to-point multicast broadcast for cxip_join_collective() use.
 *
 * We lay out all of the node indices (0..maxnodes-1) in layers, as follows:
 *
 * RADIX 1:
 * row: nodeidx
 *   0: 0
 *   1: 1
 *   2: 2
 * ...
 *
 * RADIX 2:
 * row: nodeidx
 *   0: 0
 *   1: 1, 2
 *   2: 3, 4, 5, 6
 *   3: 7, 8, 9, 10, 11, 12, 13, 14
 * ...
 *
 * RADIX 3:
 * row: nodeidx
 *   0: 0
 *   1: 1, 2, 3
 *   2: 4, 5, 6, 7, 8, 9, 10, 11, 12
 *   3: 13, 14, 15, 16, 17, 18, ... 38, 39
 * ...
 *
 * The parent of any node is in the row above it, and the children are in the
 * row below it. The width of any row is (RADIX ^ row), so for every node, there
 * can be up to RADIX children, and one parent, with exception of the root node
 * (no parent).
 */

/**
 * @brief Compute row and column for a given node index.
 *
 * @param radix   : radix of tree
 * @param nodeidx : node index
 * @param row     : returned row of this node
 * @param col     : returned offset of this node in the row
 * @param siz     : returned size of the row, (0 <= col < siz)
 */
void cxip_tree_rowcol(int radix, int nodeidx, int *row, int *col, int *siz)
{
	int rownum = 0;
	int rowcum = 0;
	int rowsiz = 1;

	*row = 0;
	*col = 0;
	*siz = rowsiz;
	if (radix < 1)
		return;
	while (nodeidx > rowcum) {
		rowsiz *= radix;
		*row = rownum + 1;
		*col = nodeidx - rowcum - 1;
		*siz = rowsiz;
		rowcum += rowsiz;
		rownum += 1;
	}
}

/**
 * @brief Compute the node index for a give row and column.
 *
 * Note that illegal columns can be specified for a row, which results
 * in a return index of -1.
 *
 * @param radix   : radix of tree
 * @param row     : row of node
 * @param col     : column of node
 * @param nodeidx : returned node index, or -1 if illegal
 */
void cxip_tree_nodeidx(int radix, int row, int col, int *nodeidx)
{
	int rownum = 0;
	int rowcum = 0;
	int rowsiz = 1;

	*nodeidx = 0;
	while (radix && rownum < row) {
		rowsiz *= radix;
		*nodeidx = rowcum + col + 1;
		rowcum += rowsiz;
		rownum += 1;
	}
	if (col >= rowsiz)
		*nodeidx = -1;
}

/**
 * @brief Provide the relatives (parent, children) of a node
 *
 * The rels array must be provided, and must have RADIX+1 entries.
 *
 * The parent position [0] will always be populated, but with -1 if the node is
 * the root node.
 *
 * Only valid child positions [1..RADIX] will be populated.
 *
 * This returns the total number of positions populated.
 *
 * If radix < 1, there can be no relatives, and this returns 0.
 *
 * @param radix    : radix of tree
 * @param nodeidx  : node to find relatives for
 * @param maxnodes : maximum valid node indices available
 * @param rels     : relatives array
 * @return int return number of valid relatives found
 */
int cxip_tree_relatives(int radix, int nodeidx, int maxnodes, int *rels)
{
	int row, col, siz, idx, n;

	if (radix < 1 || !maxnodes || !rels)
		return 0;

	cxip_tree_rowcol(radix, nodeidx, &row, &col, &siz);

	idx = 0;
	if (row)
		cxip_tree_nodeidx(radix, row - 1, col / radix, &rels[idx++]);
	else
		rels[idx++] = -1;

	cxip_tree_nodeidx(radix, row+1, col*radix, &nodeidx);
	for (n = 0; n < radix; n++) {
		if ((nodeidx + n) >= maxnodes)
			break;
		rels[idx++] = nodeidx + n;
	}

	return idx;
}

/**
 * @brief Destroy the zbcoll structure.
 *
 * @param ep_obj : endpoint
 */
void cxip_zbcoll_fini(struct cxip_ep_obj *ep_obj)
{
	struct cxip_zbcoll_obj *zbcoll;
	int i;

	zbcoll = &ep_obj->zbcoll;
	for (i = 0; i < zbcoll->count; i++)
		free(zbcoll->state[i].relatives);
	free(zbcoll->state);
	zbcoll->count = 0;
	zbcoll->state = NULL;
}

/**
 * @brief Initialize the zbcoll
 *
 * @param ep_obj : endpoint object
 * @return int FI_SUCCESS
 */
int cxip_zbcoll_init(struct cxip_ep_obj *ep_obj)
{
	ep_obj->zbcoll.count = 0;
	ep_obj->zbcoll.state = NULL;
	return FI_SUCCESS;
}

/**
 * @brief Simulation testing: packs pseudo src/dst into mb.zb_data.
 *
 * Note that when using NETSIM for testing, zb_data payloads are functionally
 * limited to 29 bits. Any additional bits will be discarded.
 *
 * Production code limits the zb_data payloads to 61 bits.
 *
 * Only 13 bits are needed for a multicast address value.
 */
union packer {
	struct {
		uint64_t dat: 29;
		uint64_t src: 16;
		uint64_t dst: 16;
	} __attribute__((packed));
	uint64_t raw;
};

/* Pack testing data */
static inline uint64_t zbpack(int src, int dst, uint64_t dat)
{
	union packer x = {.raw = 0};
	x.src = src;
	x.dst = dst;
	x.dat = dat;
	return x.raw;
}

/* Unpack testing data */
static inline void zbunpack(uint64_t data, int *src, int *dst, uint64_t *dat)
{
	union packer x = {.raw = 0};
	x.raw = data;
	*src = x.src;
	*dst = x.dst;
	*dat = x.dat;
}

/* Mark a collective operation done, prep for next */
static inline void zbdone(struct cxip_zbcoll_state *zbs)
{
	zbs->contribs = 0;
	zbs->complete = true;
}

/* Mark a collective operation failure */
static void zbsend_fail(struct cxip_zbcoll_state *zbs,
			struct cxip_ctrl_req *req, int ret)
{
	ofi_atomic_inc32(&zbs->err_count);
	zbs->error = ret;
	zbdone(zbs);
	free(req);
}

/* Send a zbcoll packet robustly, with failure handling */
static int zbsend(struct cxip_zbcoll_state *zbs, struct cxip_ctrl_req *req)
{
	int ret;

	do {
		sched_yield();
		ret = cxip_ctrl_msg_send(req);
	} while (ret == -FI_EAGAIN);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Send failure, ret = %d\n", ret);
		zbsend_fail(zbs, req, ret);
	}
	return ret;
}

/* root has no parent */
static inline bool isroot(struct cxip_zbcoll_state *zbs)
{
	return (zbs->relatives[0] == (uint32_t)-1);
}

/* receive is complete when all contributors have spoken */
static inline bool rcvcomplete(struct cxip_zbcoll_state *zbs)
{
	return (zbs->contribs == zbs->num_relatives);
}

/* send upstream to the parent */
static void zbsend_up(struct cxip_ep_obj *ep_obj,
		      struct cxip_zbcoll_state *zbs,
		      uint64_t mbv)
{
	union cxip_match_bits mb = {.raw = mbv};
	_trc_("%d->%d: %10s %10s %d/%d\n",
		zbs->nid, zbs->relatives[0], "", __func__,
		zbs->contribs, zbs->num_relatives);
	cxip_zbcoll_send(ep_obj, zbs->nid, zbs->relatives[0], mb.raw);
}

/* send downstream to all of the children */
static void zbsend_dn(struct cxip_ep_obj *ep_obj,
		      struct cxip_zbcoll_state *zbs,
		      uint64_t mbv)
{
	union cxip_match_bits mb = {.raw = mbv};
	int relidx;

 	 for (relidx = 1; relidx < zbs->num_relatives; relidx++) {
		_trc_("%d->%d: %10s %10s\n",
			zbs->nid, zbs->relatives[relidx], __func__, "");
		cxip_zbcoll_send(ep_obj, zbs->nid,
				 zbs->relatives[relidx], mb.raw);
	}
}

/* advance the upstream collective engine */
static void advance_up(struct cxip_ep_obj *ep_obj,
		       struct cxip_zbcoll_state *zbs,
		       uint64_t mbv)
{
	union cxip_match_bits mb = {.raw = mbv};

	if (!rcvcomplete(zbs))
		return;

	if (isroot(zbs)) {
		/* The root always reflects down */
		mb.zb_data = zbs->dataval;
		zbsend_dn(ep_obj, zbs, mb.raw);
		zbdone(zbs);
	} else {
		/* completed children send up */
		zbsend_up(ep_obj, zbs, mbv);
	}
}

/**
 * @brief Send callback function to manage source ACK.
 *
 * The request must be retried, or freed.
 *
 * NETSIM will simply drop packets sent to non-existent addresses, which
 * leaks the request packet. Real networks monitor the hardware with the retry
 * handler, and will perform automatic retries before giving up. This will then
 * be called with an ACK failure response.
 *
 * Libfabric caller does not handle error returns, handle them here.
 *
 * @param req   : original request
 * @param event : CXI driver event
 * @return int FI_SUCCESS
 */
static int zbdata_send_cb(struct cxip_ctrl_req *req, const union c_event *event)
{
	struct cxip_zbcoll_state *zbs;

	/* resolve NETSIM testcase -- find state pointer */
	if (req->ep_obj->zbcoll.count > 1) {
		int src, dst;
		uint64_t dat;
		zbunpack(req->send.mb.zb_data, &src, &dst, &dat);
		zbs = &req->ep_obj->zbcoll.state[src];
	} else {
		zbs = req->ep_obj->zbcoll.state;
	}

	switch (event->hdr.event_type) {
	case C_EVENT_ACK:
		switch (cxi_event_rc(event)) {
		case C_RC_OK:
			ofi_atomic_inc32(&zbs->ack_count);
			free(req);
			break;
		case C_RC_ENTRY_NOT_FOUND:
			/* likely a target queue is full, retry */
			CXIP_WARN("Target dropped packet, retry\n");
			zbsend(zbs, req);
			break;
		case C_RC_PTLTE_NOT_FOUND:
			/* may be a race during setup, retry */
			CXIP_WARN("Target connection failed, retry\n");
			zbsend(zbs, req);
			break;
		default:
			CXIP_WARN("ACK return code = %d, failed\n",
				  cxi_event_rc(event));
			zbsend_fail(zbs, req, -FI_EIO);
			break;
		}
		break;
	default:
		/* fail the send */
		CXIP_WARN("%s: Unexpected event type: %s\n",
		       	 __func__, cxi_event_to_str(event));
		zbsend_fail(zbs, req, -FI_EIO);
	}
	return FI_SUCCESS;
}

/**
 * @brief Send a zero-buffer collective packet.
 *
 * Creates a request packet that must be freed (or retried) in callback.
 *
 * @param ep_obj : endpoint
 * @param srcnid : source address (ignored unless simulating)
 * @param dstnid : destination address
 * @param mb     : packet to send
 */
int cxip_zbcoll_send(struct cxip_ep_obj *ep_obj, uint32_t srcnid,
		     uint32_t dstnid, uint64_t mbv)
{
	struct cxip_zbcoll_state *zbs;
	union cxip_match_bits mb = {.raw = mbv};
	struct cxip_ctrl_req *req;

	/* resolve NETSIM testcase */
	if (ep_obj->zbcoll.count > 1) {
		int srcidx, dstidx;
		/* find destination state index */
		for (dstidx = 0; dstidx < ep_obj->zbcoll.count; dstidx++)
			if (ep_obj->zbcoll.state[dstidx].nid == dstnid)
				break;
		/* find source state index */
		for (srcidx = 0; srcidx < ep_obj->zbcoll.count; srcidx++)
			if (ep_obj->zbcoll.state[srcidx].nid == srcnid)
				break;
		/* alter the data to pass srcidx/dstidx */
		mb.zb_data = zbpack(srcidx, dstidx, mb.zb_data);

		/* replace destination with self for this test */
		dstnid = ep_obj->src_addr.nic;

		zbs = &ep_obj->zbcoll.state[srcidx];
	} else {
		zbs = &ep_obj->zbcoll.state[0];
	}

	req = calloc(1, sizeof(*req));
	if (! req) {
		CXIP_WARN("%s failed request allocation\n", __func__);
		ofi_atomic_inc32(&zbs->err_count);
		return -FI_ENOMEM;
	}
	req->ep_obj = ep_obj;
	req->cb = zbdata_send_cb;
	req->send.nic_addr = dstnid;
	req->send.pid = ep_obj->src_addr.pid;
	req->send.mb.raw = mb.raw;
	req->send.mb.ctrl_le_type = CXIP_CTRL_LE_TYPE_CTRL_MSG;
	req->send.mb.ctrl_msg_type = CXIP_CTRL_MSG_ZB_DATA;

	return zbsend(zbs, req);
}

/**
 * @brief zbcoll message receive callback.
 *
 * This is called when a ZB collective packet is received. This does the full
 * processing of the packet, including forwarding it up and downstream.
 *
 * Caller does not handle error returns gracefully, handle them here.
 *
 * @param ep_obj    : endpoint
 * @param init_nic  : received initiator NIC
 * @param init_pid  : received initiator PID
 * @param mbv       : received match bits
 * @return int FI_SUCCESS
 */
int cxip_zbcoll_recv(struct cxip_ep_obj *ep_obj, uint32_t init_nic,
		      uint32_t init_pid, uint64_t mbv)
{
	struct cxip_zbcoll_state *zbs;
	union cxip_match_bits mb = {.raw = mbv};
	uint64_t dataval;
	int relidx;

	/* resolve NETSIM testcase */
	if (ep_obj->zbcoll.count > 1) {
		int src, dst;
		uint64_t dat;

		zbunpack(mb.zb_data, &src, &dst, &dat);
		init_nic = ep_obj->zbcoll.state[src].nid;
		zbs = &ep_obj->zbcoll.state[dst];
		dataval = dat;		// 29 bits
	} else {
		zbs = &ep_obj->zbcoll.state[0];
		dataval = mb.zb_data;	// 61 bits
	}

	/* Reject if this isn't our pid */
	if (ep_obj->src_addr.pid != init_pid) {
		CXIP_WARN("ZBCOLL bad pid, saw %d, exp %d\n",
			  init_pid, ep_obj->src_addr.pid);
		return -1;
	}

	/* Data received, increment the counter */
	ofi_atomic_inc32(&zbs->rcv_count);

	/* Raw send test case, we are done */
	if (! zbs->num_relatives) {
		CXIP_DBG("ZBCOLL no relatives: test case\n");
		return 0;
	}

	/* State machine can be disabled for testing */
	if (ep_obj->zbcoll.disable) {
		CXIP_DBG("ZBCOLL state machine disabled\n");
		return 0;
	}

	/* Check to see if this came from a relative */
	for (relidx = 0; relidx < zbs->num_relatives; relidx++)
		if (zbs->relatives[relidx] == init_nic)
			break;
	if (relidx >= zbs->num_relatives) {
		CXIP_WARN("ZBCOLL initiator %08x not a relative\n",
			  init_nic);
		ofi_atomic_inc32(&zbs->err_count);
		return 0;
	}

	if (relidx == 0) {
		/* downstream recv from parent */

		/* Copy the data to the user pointer, if any */
		if (zbs->dataptr)
			*zbs->dataptr = dataval;

		_trc_("%d<-%d: %10s %10s %d/%d (%ld)\n",
			zbs->nid, zbs->relatives[0], "dn_recvd", "",
			zbs->contribs, zbs->num_relatives, dataval);

		/* Send downstream to children */
		zbsend_dn(ep_obj, zbs, mb.raw);
		zbdone(zbs);
	} else {
		/* upstream recv from child */

		/* upstream packets contribute */
		zbs->contribs += 1;
		_trc_("%d<-x: %10s %10s %d/%d\n",
			zbs->nid, "", "up_recvd", zbs->contribs,
			zbs->num_relatives);

		/* if all contributions received, process this */
		advance_up(ep_obj, zbs, mb.raw);
	}
	return 0;
}

/**
 * @brief Perform the counter reset for all states.
 *
 * @param zbcoll   : zbcoll structure
 */
static void state_reset_counters(struct cxip_zbcoll_obj *zbcoll)
{
	struct cxip_zbcoll_state *zbs;
	int n;

	for (n = 0; n < zbcoll->count; n++) {
		zbs = &zbcoll->state[n];
		ofi_atomic_initialize32(&zbs->err_count, 0);
		ofi_atomic_initialize32(&zbs->ack_count, 0);
		ofi_atomic_initialize32(&zbs->rcv_count, 0);
	}
}

/**
 * @brief Perform the zbcoll configuration.
 *
 * This creates at least one zbcoll state.
 *
 * The nids[] map is used to create the "relatives" of this node, i.e. which
 * node is the parent, and which nodes are the children.
 *
 * For production, it creates only one state, which applies to this node.
 *
 * For testing, it creates num_nids states, one for each simulated node. This
 * allows bench-testing of the model by encoding the src/dst addresses in the
 * zero-buffer data (at the expense of payload bits), and 'sending' data to
 * the different states, using the actual mynid address to send the data to
 * itself.
 *
 * @param zbcoll    : zbcoll structure
 * @param count     : number of state structures to create, >= 1
 * @param mynid     : the hardware NID
 * @param mypid     : the hardware PID
 * @param num_nids  : the number of entries in nids[]
 * @param nids      : the array of nids
 */
static void state_config(struct cxip_zbcoll_obj *zbcoll, int count,
		         uint32_t mynid, uint32_t mypid,
		         int num_nids, uint32_t *nids)
{
	struct cxip_zbcoll_state *zbs;
	int relidx, n;

	zbcoll->count = count;
	zbcoll->state = calloc(zbcoll->count,
				sizeof(struct cxip_zbcoll_state));
	state_reset_counters(zbcoll);
	for (n = 0; n < zbcoll->count; n++) {
		zbs = &zbcoll->state[n];
		zbs->nid = (nids) ? nids[n] : mynid;
		zbs->pid = mypid;
		zbs->contribs = 0;
		zbs->error = FI_SUCCESS;
		zbs->running = false;
		zbs->complete = false;
		zbs->relatives = NULL;
		if (!num_nids || !nids)
			continue;

		zbs->relatives =
			calloc(cxip_env.zbcoll_radix + 1, sizeof(uint32_t));
		zbs->num_relatives =
			cxip_tree_relatives(cxip_env.zbcoll_radix, n, num_nids,
						(int *)zbs->relatives);
		for (relidx = 0; relidx < zbs->num_relatives; relidx++)
			if (zbs->relatives[relidx] != (uint32_t)-1)
				zbs->relatives[relidx] =
					nids[zbs->relatives[relidx]];
	}
}

/**
 * @brief Reset the endpoint counters.
 *
 * @param ep : endpoint
 */
void cxip_zbcoll_reset_counters(struct fid_ep *ep)
{
	struct cxip_ep *cxip_ep;

	cxip_ep = container_of(ep, struct cxip_ep, ep.fid);
	state_reset_counters(&cxip_ep->ep_obj->zbcoll);
}

/* sort compare */
static int _nidcmp(const void *v1, const void *v2)
{
	const uint32_t *nid1 = (uint32_t *)v1;
	const uint32_t *nid2 = (uint32_t *)v2;

	if (*nid1 < *nid2)
		return -1;
	if (*nid1 > *nid2)
		return 1;
	return 0;
}

/**
 * @brief Allow user to configure they zbcoll system.
 *
 * Normally, this should specify nids[0:num_nids-1] as actual NID addresses in
 * fabric, which comprise the zbcoll reduction group, and myidx == -1.
 *
 * There is a test case where 0 <= myidx < num_nids, which can be used (only)
 * with NETSIM. This supports sending from nids[myidx] to any nid in nids[],
 * including itself. This is simulated by creating num_nids state structures,
 * each associated with the corresponding simulated nids[] addresses, and using
 * the src address for this simulated NIC as the packet destination.
 *
 * @param ep       : endpoint
 * @param num_nids : number of network addresses to configure
 * @param nids     : array of network addresses to configure
 * @param sim      : true to perform single-node simulation
 * @return int     :
 */
int cxip_zbcoll_config(struct fid_ep *ep, int num_nids, uint32_t *nids,
			bool sim)
{
	struct cxip_ep *cxip_ep;
	struct cxip_ep_obj *ep_obj;
	struct cxip_zbcoll_obj *zbcoll;
	uint32_t mynid;
	uint32_t mypid;
	int N;

	cxip_ep = container_of(ep, struct cxip_ep, ep.fid);
	ep_obj = cxip_ep->ep_obj;

	if (num_nids && !nids) {
		CXIP_WARN("Non-zero NID count with NULL NID pointer\n");
		return -FI_EINVAL;
	}

	zbcoll = &ep_obj->zbcoll;

	// TODO: determine if this is necessary
	/* Sort these to ensure they are in the same order on all nodes */
	if (nids && num_nids)
		qsort(nids, num_nids, sizeof(uint32_t), _nidcmp);

	/* Determine if my actual address is in the NIC list */
	mypid = ep_obj->src_addr.pid;
	mynid = ep_obj->src_addr.nic;
	for (N = 0; N < num_nids; N++)
		if (mynid == nids[N])
			break;

	if (!num_nids) {
		/* Test case: no nids, send-to-self only */
		cxip_zbcoll_fini(ep_obj);
		state_config(zbcoll, 1, mynid, mypid, 0, NULL);
	} else if (sim) {
		/* Simulation: create a state for each item in nids[] */
		cxip_zbcoll_fini(ep_obj);
		state_config(zbcoll, num_nids, mynid, mypid, num_nids, nids);
	} else if (N < num_nids) {
		/* We want to participate as nids[N] == my real address */
		cxip_zbcoll_fini(ep_obj);
		state_config(zbcoll, 1, mynid, mypid, num_nids, nids);
	} else {
		/* This endpoint address is not included in the nids[] list */
		CXIP_WARN("NIC %08x is not a participant in group\n", mynid);
		return -FI_EADDRNOTAVAIL;
	}
	return FI_SUCCESS;
}

/**
 * @brief Initiate a data broadcast.
 *
 * All participants call this.
 *
 * The data is supplied by the root node data buffer.
 * The data is delivered to the child node data buffers.
 *
 * @param ep       : endpoint
 * @param data     : pointer to data buffer
 */
int cxip_zbcoll_bcast(struct fid_ep *ep, uint64_t *dataptr)
{
	struct cxip_ep *cxip_ep;
	struct cxip_zbcoll_state *zbs;
	union cxip_match_bits mb = {.raw = 0};
	int n, ret;

	cxip_ep = container_of(ep, struct cxip_ep, ep.fid);
	sched_yield();
	cxip_ep_ctrl_progress(cxip_ep->ep_obj);

	/* Loop is for testing. Production has count == 1. */
	ret = -FI_EAGAIN;
	for (n = 0; n < cxip_ep->ep_obj->zbcoll.count; n++) {
		zbs = &cxip_ep->ep_obj->zbcoll.state[n];
		if (zbs->running)
			continue;
		/* found a free slot, start the collective */
		zbs->error = FI_SUCCESS;
		zbs->complete = false;
		zbs->running = true;
		zbs->dataptr = dataptr;
		zbs->contribs++;
		if (isroot(zbs) && zbs->dataptr)
			zbs->dataval = *zbs->dataptr;
		else
			zbs->dataval = 0;
		/* if terminal leaf node, will send up immediately */
		mb.zb_data = 0;
		advance_up(cxip_ep->ep_obj, zbs, mb.raw);
		ret = FI_SUCCESS;
		break;
	}
	return ret;
}

/**
 * @brief Initiate a no-data barrier.
 *
 * All participants call this.
 *
 * @param ep       : endpoint
 */
int cxip_zbcoll_barrier(struct fid_ep *ep)
{
	return cxip_zbcoll_bcast(ep, NULL);
}

/**
 * @brief Progress the zbcollective state.
 *
 * Returns FI_SUCCESS on successful completion
 * Returns FI_EAGAIN if still running
 * Returns error code on collective failure
 *
 * @param ep   : endpoint
 * @return int : return code
 */
int cxip_zbcoll_progress(struct fid_ep *ep)
{
	struct cxip_ep *cxip_ep;
	struct cxip_zbcoll_state *zbs;
	int n, ret;

	cxip_ep = container_of(ep, struct cxip_ep, ep.fid);

	sched_yield();
	cxip_ep_ctrl_progress(cxip_ep->ep_obj);

	ret = FI_SUCCESS;
	for (n = 0; n < cxip_ep->ep_obj->zbcoll.count; n++) {
		zbs = &cxip_ep->ep_obj->zbcoll.state[n];
		if (! zbs->complete)
			return -FI_EAGAIN;
		zbs->running = false;
		if (zbs->error)
			ret = zbs->error;
	}
	return ret;
}
