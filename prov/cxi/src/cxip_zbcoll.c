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

#define	trc CXIP_TRACE

/* see data packing structures below */
#define	ZB_MAP_BITS	54
#define	ZB_GRPID_BITS	6
#define	ZB_SIM_BITS	5
#define	ZB_NEG_BIT	(ZB_MAP_BITS - 1)

static int zbdata_send_cb(struct cxip_ctrl_req *req,
			  const union c_event *event);

/****************************************************************************
 * OVERVIEW
 *
 * There are two related components in this file.
 * - An abstract radix tree constructor
 * - A collective implemention built on the Zero-Buffer Put control channel.
 *
 * The basic operational flow is as follows:
 * - cxip_zbcoll_init() prepares the system for zbcoll collectives.
 * - cxip_zbcoll_alloc() allocates and configures a collective structure.
 * - cxip_zbcoll_getgroup() negotiates a collective identifier (one time).
 * - cxip_zbcoll_barrier() performs a barrier (can be repeated).
 * - cxip_zbcoll_broadcast() performs a broadcast (can be repeated).
 * - cxip_zbcoll_progress() progresses getgroup/barrier/broadcast.
 * - cxip_zbcoll_free() releases the collective structure and identifier.
 * - cxip_zbcoll_fini() releases all collectives and cleans up.
 *
 * Any number of collective structures can be created, spanning the same, or
 * different node-sets.
 *
 * To enable the structure, it must acquire a group identifier using the
 * getgroup operation, which is itself a collective operation. Getgroup is
 * serialized over the endpoint, and acquires one of 53 possible group
 * identifiers. It will fail with -FI_BUSY if all 53 identifiers are in use. You
 * cannot enable another structure until a prior structure is freed.
 *
 * Each enabled structure can be used concurrently with any other enabled
 * structures. Collective operations are serialized over the enabled structure.
 * An attempt to start a new collective operation on an enabled structure that
 * is already performing a collective operation will return -FI_EAGAIN.
 *
 * The getgroup, barrier, and broadcast functions support a callback stack that
 * allows caller-defined callback functions to be stacked for execution upon
 * completion of a collective. The callback can initiate a new collective on the
 * same object.
 *
 * Note that this is NOT a general-purpose collective implementation.
 */

/****************************************************************************
 * ABSTRACT RADIX TREE
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
 * can be up to RADIX children, and one parent, with the exception of the root
 * node (no parent).
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
 * Only valid child positions in [1..RADIX] will be populated.
 *
 * This returns the total number of positions populated.
 *
 * If radix < 1, there can be no relatives, and this returns 0.
 *
 * @param radix    : radix of tree
 * @param nodeidx  : index of node to find relatives for
 * @param maxnodes : maximum valid node indices available
 * @param rels     : relative index array
 * @return int : number of valid relatives found
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

/****************************************************************************
 * @brief Zero-buffer collectives.
 *
 * ZB collectives are intended for implementation of the fi_join_collective()
 * function.
 *
 * The ep_obj has a container structure of type cxip_ep_zbcoll_obj, which
 * maintains endpoint-global state for all zb collectives. We refer to this as
 * the zbcoll object, and it is an extension of the endpoint itself.
 *
 * The zbcoll object contains up to ZB_NEG_BIT dynamic zb objects, each
 * representing a collective group.
 *
 * Each zb object contains one or more state structures. These are purely for
 * doing single-endpoint bench testing. Production code will use only one state
 * for the NID.
 *
 * Each zb object contains a callback stack, which can be used to execute a
 * sequence of user-defined operations that chain automatically. The next
 * callback on the stack is triggered by completion of the previous callback.
 *
 * Diagnostic counters are maintained:
 *
 * - ack_count == successful sends
 * - err_count == failed sends
 * - rcv_count == successful receives
 * - dsc_count == discarded receives
 */

static inline void _setbit(uint64_t *mask, int bit)
{
	*mask |= (1ULL << bit);
}

static inline void _clrbit(uint64_t *mask, int bit)
{
	*mask &= ~(1ULL << bit);
}

void cxip_zbcoll_get_counters(struct cxip_ep_obj *ep_obj, uint32_t *dsc,
			      uint32_t *err, uint32_t *ack, uint32_t *rcv)
{
	struct cxip_ep_zbcoll_obj *zbcoll;

	zbcoll = &ep_obj->zbcoll;
	if (dsc)
		*dsc = ofi_atomic_get32(&zbcoll->dsc_count);
	if (err)
		*err = ofi_atomic_get32(&zbcoll->err_count);
	if (ack)
		*ack = ofi_atomic_get32(&zbcoll->ack_count);
	if (rcv)
		*rcv = ofi_atomic_get32(&zbcoll->rcv_count);
}

/**
 * @brief Free zbcoll object.
 *
 * This flushes the callback stack, and releases the group identifier
 * associated with this zbcoll object.
 *
 * @param zb : zb object to free
 */
void cxip_zbcoll_free(struct cxip_zbcoll_obj *zb)
{
	int i;

	if (!zb)
		return;

	zb->flush = true;
	cxip_zbcoll_pop_cb(zb);
	cxip_zbcoll_rlsgroup(zb);
	if (zb->state) {
		for (i = 0; i < zb->simcount; i++)
			free(zb->state[i].relatives);
	}
	free(zb->caddrs);
	free(zb->state);
	free(zb->shuffle);
	free(zb);
}

/* configure the zbcoll object */
static int _state_config(struct cxip_zbcoll_obj *zb, int simcount, int grp_rank)
{
	struct cxip_zbcoll_state *zbs;
	int radix, n;

	radix = cxip_env.zbcoll_radix;

	zb->simcount = simcount;
	zb->state = calloc(simcount, sizeof(*zbs));
	if (!zb->state)
		goto nomem;

	free(zb->shuffle);
	zb->shuffle = NULL;

	for (n = 0; n < simcount; n++) {
		zbs = &zb->state[n];
		zbs->zb = zb;

		/* do not create relatives if no addrs */
		if (!zb->num_caddrs)
			continue;

		/* if simulating, override grp_rank with each state index */
		if (simcount > 1)
			grp_rank = n;

		/* create space for relatives */
		zbs->grp_rank = grp_rank;
		zbs->relatives = calloc(radix + 1, sizeof(*zbs->relatives));
		if (!zbs->relatives)
			goto fail;

		/* This produces indices in an abstract tree */
		zbs->num_relatives =
			cxip_tree_relatives(radix, grp_rank, zb->num_caddrs,
					    zbs->relatives);
	}
	return FI_SUCCESS;
fail:
	for (n = 0; n < simcount; n++) {
		zbs = &zb->state[n];
		free(zbs->relatives);
	}
nomem:
	free(zb->state);
	zb->state = NULL;
	return -FI_ENOMEM;
}

/* sort out the various configuration cases */
static int _zbcoll_config(struct cxip_zbcoll_obj *zb, int num_addrs,
			  fi_addr_t *fiaddrs, bool sim)
{
	int grp_rank, i, ret;

	if (num_addrs && !fiaddrs) {
		CXIP_WARN("Non-zero addr count with NULL addr pointer\n");
		return -FI_EINVAL;
	}

	/* do a lookup on all of the fiaddrs */
	if (num_addrs) {
		zb->num_caddrs = num_addrs;
		zb->caddrs = calloc(num_addrs, sizeof(*zb->caddrs));
		if (!zb->caddrs)
			return -FI_ENOMEM;
		for (i = 0; i < num_addrs; i++) {
			ret = _cxip_av_lookup(zb->ep_obj->av, fiaddrs[i],
					      &zb->caddrs[i]);
			if (ret) {
				CXIP_WARN("Lookup on fiaddr=%ld failed]n",
					  fiaddrs[i]);
				return -FI_EINVAL;
			}
		}
	}

	/* find the index of the source address in the address list */
	for (grp_rank = 0; !sim && grp_rank < num_addrs; grp_rank++)
		if (CXIP_ADDR_EQUAL(zb->caddrs[grp_rank], zb->ep_obj->src_addr))
			break;

	if (!num_addrs) {
		/* test case: no nics, send-to-self only */
		ret = _state_config(zb, 1, grp_rank);
	} else if (sim && num_addrs <= (1 << ZB_SIM_BITS)) {
		/* simulation: create a state for each item in addrs[] */
		ret = _state_config(zb, num_addrs, grp_rank);
	} else if (sim) {
		CXIP_WARN("Simulation maximum size = %d\n", (1 << ZB_SIM_BITS));
		ret = -FI_EADDRNOTAVAIL;
	} else if (grp_rank < num_addrs) {
		/* we want to participate as addrs[src_idx] == myaddr */
		ret = _state_config(zb, 1, grp_rank);
	} else {
		CXIP_WARN("Endpoint addr not in addrs[]\n");
		ret = -FI_EADDRNOTAVAIL;
	}
	return ret;
}

/**
 * @brief Allocate and configure a zbcoll object.
 *
 * The zb object represents a radix tree through multiple nics that can perform
 * sequential synchronizing collectives. It can be reused.
 *
 * This supports several test modes.
 *
 * If num_nics == 0, the zb object can only be used to test cxip_zbcoll_send(),
 * to exercise a send-to-self using the ctrl channel, and will work with NETSIM.
 *
 * If sim is true, this can be used to perform a simulated collective on a
 * single node, and will work with NETSIM. num_nics is limited to 2^ZB_SIM_BITS
 * simulated endpoints.
 *
 * Otherwise this will be used to perform real collectives over the group
 * specified by the specified nics. The self-address of the node calling this
 * must be a member of this set, or the function will return -FI_EADDRNOTAVAIL
 * and *zbp will return NULL.
 *
 * nid[0] is defined as the collective root nid.
 *
 * @param zbcoll   : endpoint zbcoll object
 * @param num_nics : number of nics in the list
 * @param nics     : fabric nic addresses
 * @param sim      : true if nics are simulated
 * @param zbp      : returned zb object
 * @return int : FI_SUCCESS or error value
 */
int cxip_zbcoll_alloc(struct cxip_ep_obj *ep_obj,
		      int num_addrs, fi_addr_t *fiaddrs, bool sim,
		      struct cxip_zbcoll_obj **zbp)
{
	struct cxip_zbcoll_obj *zb;
	int ret;

	if (!zbp)
		return -FI_EINVAL;

	/* allocate the zbcoll object */
	*zbp = NULL;
	zb = calloc(1, sizeof(*zb));
	if (!zb)
		return -FI_ENOMEM;
	zb->ep_obj = ep_obj;
	zb->grpid = ZB_NEG_BIT;

	/* configure the zbcoll object */
	ret = _zbcoll_config(zb, num_addrs, fiaddrs, sim);
	if (ret) {
		cxip_zbcoll_free(zb);
		CXIP_WARN("Failed to configure zbcoll object = %s\n",
			  fi_strerror(-ret));
		return ret;
	}

	/* return the zbcoll object */
	*zbp = zb;
	return FI_SUCCESS;
}

/**
 * Data packing structures.
 *
 * This defines the specific bit meanings in the 64-bit zb put packet. Bit
 * mapping could be modified, see considerations below.
 *
 * Considerations for the (production) network field:
 *
 * - dat MUST hold a multicast address and hardware root data
 * - grpid size limits the number of concurrent zbcoll operations
 * - sim requires only one bit and applies only to devel testing
 * - pad is fixed by the control channel implementation
 *
 * Implementation of the negotiation operation requires that dat contain a
 * bitmap. The choice of 54 allows for 54 grpid values (0-53), which will fit
 * into a 6-bit grpid value. This is a large number for concurrencies. The grpid
 * field could be reduced to 5 bits, offering only 32 concurrent operations. The
 * map bits should then be reduced to 32, which would free up 23 bits for other
 * information during negotiation, should extra bits be required.
 *
 * For broadcast, the full dat field is available for multicast information. The
 * multicast address is currently 13 bits. Future revisions of Rosetta may
 * increase this. The remaining bits can be used for a representation of the
 * root node. A full caddr would require 32 bits, while using a 32-bit index
 * into the fi_av_set would allow for a collective spanning up to 4 billion
 * endpoints. This allows the multicast address to expand by another 9 bits, for
 * a total of 22 bits, or 4 million multicast addresses.
 *
 * Considerations for the simulation fields:
 *
 * - src and dst must have the same number of bits
 * - src/dst bits constrain the size of the simulated zbcoll tree
 * - multi-NID simulations are not likely to be supported
 */
union packer {
	struct {
		uint64_t dat: (ZB_MAP_BITS - 2*ZB_SIM_BITS);
		uint64_t src: ZB_SIM_BITS;
		uint64_t dst: ZB_SIM_BITS;
		uint64_t grpid: ZB_GRPID_BITS;
		uint64_t sim: 1;
		uint64_t pad: 3;
	} sim __attribute__((__packed__));
	struct {
		uint64_t dat: ZB_MAP_BITS;
		uint64_t grpid: ZB_GRPID_BITS;
		uint64_t sim: 1;
		uint64_t pad: 3;
	} net __attribute__((__packed__));
	uint64_t raw;
};


/* pack data */
static inline uint64_t zbpack(int sim, int src, int dst, int grpid,
			      uint64_t dat)
{
	union packer x = {.raw = 0};
	if (sim) {
		x.sim.sim = 1;
		x.sim.src = src;
		x.sim.dst = dst;
		x.sim.grpid = grpid;
		x.sim.dat = dat;
	} else {
		x.sim.sim = 0;
		x.net.grpid = grpid;
		x.net.dat = dat;
	}
	return x.raw;
}

/* unpack data */
static inline int zbunpack(uint64_t data, int *src, int *dst, int *grpid,
			   uint64_t *dat)
{
	union packer x = {.raw = data};
	if (x.sim.sim) {
		*src = x.sim.src;
		*dst = x.sim.dst;
		*grpid = x.sim.grpid;
		*dat = x.sim.dat;
	} else {
		*src = 0;
		*dst = 0;
		*grpid = x.net.grpid;
		*dat = x.net.dat;
	}
	return x.sim.sim;
}

/**
 * zbcoll state machine.
 *
 * The zbcollectives are intended to perform necessary synchronization among all
 * NIDs participating in a fi_join_collective() operation. Every join will have
 * its own set of NIDs, which may overlap with the NIDs used in another
 * concurrently-executing fi_join_collective(). Thus, every NID may be
 * participating simultaneously in a different number of join operations.
 *
 * Every process (NID) in the collective sits somewhere in a radix tree, with
 * one parent as relative[0] (except for the root), and up to RADIX-1 children
 * at relative[1,...].
 *
 * The collective follows a two-stage data flow, first from children toward the
 * root (upstream), then from root toward the children (downstream).
 *
 * Processes (NIDs) must wait for all children to report before forwarding their
 * own contribution toward the root. When the children of the root all report,
 * the root reflects the result back to its children, and completes. As each
 * child receives from its parent, it propagates the result to its children, and
 * completes.
 *
 * Packets are unrestricted, and thus receive confirmation ACK messages from the
 * hardware, or NAK and retry if delivery fails.
 *
 * The leaf (childless) NIDs contribute immediately and send the zb->dataval
 * data upstream. Each parent collects data from its children and bitwise-ANDs
 * the data with its own zb->dataval. When all children have reported to the
 * root, the root sends *zb->dataval downstream, and the children simply
 * propagate the data to the leaves. This fixed behavior covers all our
 * use-cases.
 *
 * For the barrier operation, zb->dataptr is set to NULL, and both it and
 * zb->dataval are ignored.
 *
 * For the broadcast operation, zb->dataptr is a caller-supplied pointer,
 * containing the data to be broadcast on the root and unspecified on the other
 * NIDs, and zb->dataval is ignored.
 *
 * Both barrier and broadcast must be preceded by a getgroup operation, to
 * obtain a grpid value for the uninitialized zb object.
 *
 * For the getgroup operation, zb->dataval is a copy of the endpoint zbcoll
 * grpmsk, which has a bit set to 1 for every grpid that is available for that
 * NID. NIDs may have different grpmsk values. All of these masks are passed
 * upstream through zb->dataval in a bitwise-AND reduction. When it reaches the
 * root, the set bits in zb->dataval are the grpid values still available across
 * all of the NIDs in the group. Because zb->dataptr is set to &zb->dataval,
 * downstream propagation automatically distributes this bitmask back to all of
 * the other NIDs. The negotiated group id is the lowest numbered bit still set,
 * and every NID computes this from the bitmask.
 *
 * It is possible for all group ID values to be exhausted. In this case, the
 * getgroup operation will report -FI_EBUSY, and the caller should retry until a
 * join operation completes, releasing one of the group ID values. If zb
 * collective objects are never released, new operations will be blocked
 * indefinitely.
 *
 * Getgroup operations are always serialized across the entire endpoint.
 * Attempting a second getgroup on any (new) zb object before the first has
 * completed will return -FI_EAGAIN. This is required to prevent race conditions
 * that would issue the same group id to multiple zbcoll objects.
 *
 * We are externally guaranteed that all fi_join_collective() operations will
 * observe proper collective ordering. Specifically, if any two joins share two
 * or more NIDs, those joins will be initiated in the same order on all shared
 * NIDs (possibly interspersed with other joins for unrelated groups). This
 * behavior is necessary to ensure that all NIDs in a group obtain the same
 * grpid value.
 */

/* send a zbcoll packet -- wrapper for cxip_ctrl_msg_send() */
static void zbsend(struct cxip_ep_obj *ep_obj, uint32_t dstnic, uint32_t dstpid,
		   uint64_t mbv)
{
	struct cxip_ep_zbcoll_obj *zbcoll;
	struct cxip_ctrl_req *req;
	int ret;

	zbcoll = &ep_obj->zbcoll;

	req = calloc(1, sizeof(*req));
	if (!req) {
		CXIP_WARN("%s failed request allocation\n", __func__);
		ofi_atomic_inc32(&zbcoll->err_count);
		return;
	}

	req->ep_obj = ep_obj;
	req->cb = zbdata_send_cb;
	req->send.nic_addr = dstnic;
	req->send.pid = dstpid;
	req->send.mb.raw = mbv;
	req->send.mb.ctrl_le_type = CXIP_CTRL_LE_TYPE_CTRL_MSG;
	req->send.mb.ctrl_msg_type = CXIP_CTRL_MSG_ZB_DATA;

	/* If we can't send, collective cannot complete, just spin */
	do {
		ret =  cxip_ctrl_msg_send(req);
		if (ret == -FI_EAGAIN)
			cxip_ep_ctrl_progress(ep_obj);
	} while (ret == -FI_EAGAIN);
	if (ret) {
		CXIP_WARN("%s failed CTRL message send\n", __func__);
		ofi_atomic_inc32(&zbcoll->err_count);
	}
}

/* reject a getgroup request from a non-group entity */
static void reject(struct cxip_ep_obj *ep_obj, int dstnic, int dstpid,
		   int sim, int src, int dst, int grpid)
{
	union cxip_match_bits mb;

	mb.raw = zbpack(sim, dst, src, grpid, 0);
	zbsend(ep_obj, dstnic, dstpid, mb.raw);
}

/**
 * @brief Send a zero-buffer collective packet.
 *
 * Creates a request packet that must be freed (or retried) in callback.
 *
 * This can physically send ONLY from endpoint source address, but the src
 * address can be provided for simulation.
 *
 * Only the lower bits of the 64-bit payload will be delivered, depending on the
 * specific packing model. Upper control bits will be overwritten as necessary.
 *
 * @param zb      : indexed zb structure
 * @param srcidx  : source address index (ignored unless simulating)
 * @param dstidx  : destination address index (required)
 * @param payload : packet value to send
 */
void cxip_zbcoll_send(struct cxip_zbcoll_obj *zb, int srcidx, int dstidx,
		      uint64_t payload)
{
	union cxip_match_bits mb = {.raw = 0};
	struct cxip_addr dstaddr;

	/* resolve NETSIM testcase */
	trc("SND %04x->%04x %016lx\n", srcidx, dstidx, payload);
	if (zb->simcount > 1) {
		if (dstidx >= zb->simcount) {
			ofi_atomic_inc32(&zb->ep_obj->zbcoll.err_count);
			return;
		}
		/* alter the data to pass srcaddr/dstaddr */
		mb.zb_data = zbpack(1, srcidx, dstidx, zb->grpid, payload);
		dstaddr = zb->ep_obj->src_addr;
	} else {
		/* srcidx, dstaddr are discarded in zbpack() */
		if (dstidx >= zb->num_caddrs) {
			ofi_atomic_inc32(&zb->ep_obj->zbcoll.err_count);
			return;
		}
		mb.zb_data = zbpack(0, 0, 0, zb->grpid, payload);
		dstaddr = zb->caddrs[dstidx];
	}
	zbsend(zb->ep_obj, dstaddr.nic, dstaddr.pid, mb.raw);
}

/* mark a collective operation done, pop the callback */
static inline void zbdone(struct cxip_zbcoll_state *zbs)
{
	struct cxip_zbcoll_obj *zb = zbs->zb;
	zbs->contribs = 0;
	if (zb->busy && !--zb->busy)
		cxip_zbcoll_pop_cb(zb);
}

/* mark a collective send failure and end the collective */
static void zbsend_fail(struct cxip_zbcoll_state *zbs,
			struct cxip_ctrl_req *req, int ret)
{
	struct cxip_ep_zbcoll_obj *zbcoll;

	zbcoll = &zbs->zb->ep_obj->zbcoll;
	ofi_atomic_inc32(&zbcoll->err_count);
	zbs->zb->error = ret;
	zbdone(zbs);
	free(req);
}

/* root has no parent */
static inline bool isroot(struct cxip_zbcoll_state *zbs)
{
	return (zbs->relatives[0] < 0);
}

/* receive is complete when all contributors have spoken */
static inline bool rcvcomplete(struct cxip_zbcoll_state *zbs)
{
	return (zbs->contribs == zbs->num_relatives);
}

/* send upstream to the parent */
static void zbsend_up(struct cxip_zbcoll_state *zbs,
		      uint64_t mbv)
{
	union cxip_match_bits mb = {.raw = mbv};
	int sim, src, dst, grpid;
	uint64_t dat;

	trc("%04x->%04x: %-10s %-10s %d/%d\n",
		zbs->grp_rank, zbs->relatives[0], "", __func__,
		zbs->contribs, zbs->num_relatives);
	sim = zbunpack(mb.raw, &src, &dst, &grpid, &dat);
	dat &= zbs->dataval;
	zbpack(sim, src, dst, grpid, dat);
	cxip_zbcoll_send(zbs->zb, zbs->grp_rank, zbs->relatives[0], mb.raw);
}

/* send downstream to all of the children */
static void zbsend_dn(struct cxip_zbcoll_state *zbs,
		      uint64_t mbv)
{
	union cxip_match_bits mb = {.raw = mbv};
	int relidx;

 	for (relidx = 1; relidx < zbs->num_relatives; relidx++) {
		trc("%04x->%04x: %-10s %-10s\n",
			zbs->grp_rank, zbs->relatives[relidx],
			__func__, "");
		cxip_zbcoll_send(zbs->zb,
			zbs->grp_rank, zbs->relatives[relidx], mb.raw);
	}
}

/* advance the collective engine */
static void advance(struct cxip_zbcoll_state *zbs, uint64_t mbv)
{
	union cxip_match_bits mb = {.raw = mbv};

	if (!rcvcomplete(zbs))
		return;

	if (isroot(zbs)) {
		/* The root always reflects bcast data down */
		mb.zb_data = (zbs->dataptr) ? (*zbs->dataptr) : 0;
		zbsend_dn(zbs, mb.raw);
		zbdone(zbs);
	} else {
		/* completed children send up */
		zbsend_up(zbs, mbv);
	}
}

/**
 * @brief zbcoll message receive callback.
 *
 * This is called by the cxip_ctrl handler when a ZB collective packet is
 * received. This is "installed" at ep initialization, so it can begin receiving
 * packets before a zb object has been allocated to receive the data. Races are
 * handled by issuing a rejection packet back to the sender, which results in a
 * retry.
 *
 * Caller does not handle error returns gracefully. Handle all errors, and
 * return FI_SUCCESS.
 *
 * @param ep_obj    : endpoint
 * @param init_nic  : received (actual) initiator NIC
 * @param init_pid  : received (actual) initiator PID
 * @param mbv       : received match bits
 * @return int : FI_SUCCESS (formal return)
 */
int cxip_zbcoll_recv_cb(struct cxip_ep_obj *ep_obj, uint32_t init_nic,
			uint32_t init_pid, uint64_t mbv)
{
	struct cxip_ep_zbcoll_obj *zbcoll;
	struct cxip_zbcoll_obj *zb;
	struct cxip_zbcoll_state *zbs;
	int sim, src, dst, grpid;
	uint32_t inic, ipid;
	uint64_t dat;
	union cxip_match_bits mb = {.raw = mbv};
	int relidx;

	zbcoll = &ep_obj->zbcoll;
	sim = zbunpack(mbv, &src, &dst, &grpid, &dat);
	/* determine the initiator to use */
	if (sim) {
		inic = src;
		ipid = ep_obj->src_addr.pid;
	} else {
		inic = init_nic;
		ipid = init_pid;
	}
	trc("RCV INI=%04x PID=%04x sim=%d %d->%d grp=%d dat=%016lx\n",
	    inic, ipid, sim, src, dst, grpid, dat);

	/* discard if grpid is explicitly invalid (bad packet) */
	if (grpid > ZB_NEG_BIT) {
		CXIP_WARN("Invalid group ID value = %d\n", grpid);
		trc("Invalid group ID value = %d\n", grpid);
		ofi_atomic_inc32(&zbcoll->dsc_count);
		return FI_SUCCESS;
	}
	/* low-level packet test */
	if (zbcoll->disable) {
		/* Attempting a low-level test */
		ofi_atomic_inc32(&zbcoll->rcv_count);
		return FI_SUCCESS;
	}
	/* resolve the zb object */
	zb = zbcoll->grptbl[grpid];
	if (!zb) {
		if (grpid == ZB_NEG_BIT) {
			/* someone else is negotiating, we aren't ready */
			reject(ep_obj, init_nic, init_pid,
			       sim, dst, src, grpid);
		} else {
			/* illegal, attempting collective without grpid */
			trc("discard: collective with no grpid\n");
			ofi_atomic_inc32(&zbcoll->dsc_count);
		}
		return FI_SUCCESS;
	}
	/* reject bad state indices */
	if (src >= zb->simcount || dst >= zb->simcount) {
		CXIP_WARN("Bad simulation: src=%d dst=%d max=%d\n",
			  src, dst, zb->simcount);
		trc("discard: simsrc=%d simdst=%d\n", src, dst);
		ofi_atomic_inc32(&zbcoll->dsc_count);
		return FI_SUCCESS;
	}
	/* set the state object, and modify initiator for simulation */
	zbs = &zb->state[dst];
	/* raw send test case, we are done */
	if (!zbs->num_relatives) {
		CXIP_DBG("ZBCOLL no relatives: test case\n");
		return FI_SUCCESS;
	}
	/* determine which relative this came from */
	for (relidx = 0; relidx < zbs->num_relatives; relidx++) {
		if (inic == zb->caddrs[zbs->relatives[relidx]].nic &&
		    ipid == zb->caddrs[zbs->relatives[relidx]].pid)
			break;
	}
	if (relidx == zbs->num_relatives) {
		/* not a relative, reject or discard */
		if (grpid == ZB_NEG_BIT) {
			trc("getgroup: src not a relative\n");
			reject(ep_obj, init_nic, init_pid,
			       sim, dst, src, grpid);
		} else {
			trc("discard: src not a relative\n");
			ofi_atomic_inc32(&zbcoll->dsc_count);
		}
		return FI_SUCCESS;
	}
	/* data received, increment the counter */
	ofi_atomic_inc32(&zbcoll->rcv_count);

	/* advance the state */
	if (relidx == 0) {
		/* downstream recv from parent */

		/* copy the data to the user pointer, if any */
		if (zbs->dataptr)
			*zbs->dataptr = dat;

		trc("%04x<-%04x: %-10s %-10s %d/%d (%016lx)\n",
			zbs->grp_rank, zbs->relatives[0], "dn_recvd", "",
			zbs->contribs, zbs->num_relatives, dat);

		/* send downstream to children */
		zbsend_dn(zbs, mb.raw);
		zbdone(zbs);
	} else {
		/* upstream recv from child */

		/* bitwise-AND the upstream data value */
		zbs->dataval &= mb.raw;

		/* upstream packets contribute */
		zbs->contribs += 1;
		trc("%04x<-%04x: %-10s %-10s %d/%d\n",
			zbs->grp_rank, inic, "", "up_recvd", zbs->contribs,
			zbs->num_relatives);

		/* send upstream to parent */
		advance(zbs, mb.raw);
	}
	return FI_SUCCESS;
}

/**
 * @brief Send callback function to manage source ACK.
 *
 * The request must be retried, or freed.
 *
 * NETSIM will simply drop packets sent to non-existent addresses, which leaks
 * the request packet.
 *
 * Caller does not handle error returns gracefully. Handle all errors, and
 * return FI_SUCCESS.
 *
 * @param req   : original request
 * @param event : CXI driver event
 * @return int  : FI_SUCCESS (formal return)
 */
static int zbdata_send_cb(struct cxip_ctrl_req *req, const union c_event *event)
{
	struct cxip_ep_zbcoll_obj *zbcoll;
	struct cxip_zbcoll_obj *zb;
	struct cxip_zbcoll_state *zbs;
	int src, dst, grpid;
	int sim __attribute__((unused));
	uint64_t dat;
	int ret;

	sim = zbunpack(req->send.mb.zb_data, &src, &dst, &grpid, &dat);
	trc("ACK sim=%d %d->%d grp=%d dat=%016lx\n",
	    sim, src, dst, grpid, dat);
	zbcoll = &req->ep_obj->zbcoll;
	if (grpid > ZB_NEG_BIT) {
		/* ill-formed packet, discard with no regrets */
		CXIP_WARN("Invalid group ID value = %d\n", grpid);
		goto discard;
	}
	zb = zbcoll->grptbl[grpid];
	if (!zb) {
		/* Low-level testing */
		if (zbcoll->disable) {
			ofi_atomic_inc32(&zbcoll->ack_count);
			goto done;
		}
		/* ACK is late on negotiation */
		if (grpid == ZB_NEG_BIT) {
			ofi_atomic_inc32(&zbcoll->ack_count);
			goto done;
		}
		/* ACK is late */
		CXIP_WARN("ACK src=%d dst=%d grp=%d\n", src, dst, grpid);
		goto discard;
	}
	if (src >= zb->simcount || dst >= zb->simcount) {
		/* ill-formed packet, discard with no regrets */
		CXIP_WARN("Bad simulation: src=%d dst=%d max=%d\n",
			  src, dst, zb->simcount);
		goto discard;
	}
	zbs = &zb->state[dst];

	ret = FI_SUCCESS;
	switch (event->hdr.event_type) {
	case C_EVENT_ACK:
		switch (cxi_event_rc(event)) {
		case C_RC_OK:
			ofi_atomic_inc32(&zbcoll->ack_count);
			free(req);
			break;
		case C_RC_ENTRY_NOT_FOUND:
			/* likely a target queue is full, retry */
			CXIP_WARN("Target dropped packet, retry\n");
			usleep(cxip_env.fc_retry_usec_delay);
			ret = cxip_ctrl_msg_send(req);
			break;
		case C_RC_PTLTE_NOT_FOUND:
			/* may be a race during setup, retry */
			CXIP_WARN("Target connection failed, retry\n");
			usleep(cxip_env.fc_retry_usec_delay);
			ret = cxip_ctrl_msg_send(req);
			break;
		default:
			CXIP_WARN("ACK return code = %d, failed\n",
				  cxi_event_rc(event));
			ret = -FI_EIO;
			break;
		}
		break;
	default:
		/* fail the send */
		CXIP_WARN("%s: Unexpected event type: %s\n",
		       	 __func__, cxi_event_to_str(event));
		ret = -FI_EIO;
		break;
	}
	if (ret != FI_SUCCESS)
		zbsend_fail(zbs, req, ret);

	return FI_SUCCESS;
discard:
	ofi_atomic_inc32(&zbcoll->err_count);
done:
	free(req);
	return FI_SUCCESS;
}

/**
 * @brief Push callback and data onto stack for a zb object.
 *
 * @param zb      : zb object
 * @param usrfunc : user-defined callback function
 * @param usrptr  : user-defined callback data
 */
int cxip_zbcoll_push_cb(struct cxip_zbcoll_obj *zb,
			 zbcomplete_t usrfunc, void *usrptr)
{
	struct cxip_zbcoll_stack *stk;

	stk = calloc(1, sizeof(*stk));
	if (!stk)
		return -FI_ENOMEM;
	stk->cbstack = zb->cbstack;	// save old stack pointer
	stk->usrfunc = usrfunc;
	stk->usrptr = usrptr;
	zb->cbstack = stk;		// replace stack pointer
	return FI_SUCCESS;
}

/**
 * @brief Pop and execute callback function for a zb object.
 *
 * Callback functions should examine the zb->error flag and take appropriate
 * corrective action if it is not FI_SUCCESS.
 *
 * @param zb : zb object
 */
void cxip_zbcoll_pop_cb(struct cxip_zbcoll_obj *zb)
{
	struct cxip_zbcoll_stack *stk;
	zbcomplete_t usrfunc;
	void *usrptr;

	/* nothing more to do */
	if (!zb || !zb->cbstack)
		return;
	/* pop the next callback and execute */
	stk = zb->cbstack;		// get old stack pointer
	usrfunc = stk->usrfunc;
	usrptr = stk->usrptr;
	zb->cbstack = stk->cbstack;	// replace stack pointer
	free(stk);
	if (zb->flush)
		cxip_zbcoll_pop_cb(zb);
	else
		usrfunc(zb, usrptr);
}

/**
 * @brief Return the maximum group ID for concurrent zbcoll operations.
 *
 * Maximum slots are ZB_NEG_BIT+1, with one reserved for negotiation.
 *
 * @param sim  : true if nics are simulated
 * @return int maximum group ID value
 */
int cxip_zbcoll_max_grps(bool sim)
{
	return (!sim) ? ZB_NEG_BIT : ZB_NEG_BIT - 2*ZB_SIM_BITS;
}

/* callback function for completion of getgroup collective */
static void _getgroup_done(struct cxip_zbcoll_obj *zb, void *usrptr)
{
	struct cxip_ep_zbcoll_obj *zbcoll;
	uint64_t v;
	int grpid;

	zbcoll = &zb->ep_obj->zbcoll;

	/* find the LSBit in returned data */
	for (grpid = 0, v= 1ULL; grpid <= ZB_NEG_BIT; grpid++, v<<=1)
		if (v & zb->state[0].dataval)
			break;

	/* manage a rejection due to a transient race condition */
	if (grpid > ZB_NEG_BIT) {
		/* race condition reported */
		zbcoll->grptbl[ZB_NEG_BIT] = NULL;
		zb->error = -FI_EAGAIN;
		return;
	}

	/* manage failure due to all grpid values in-use */
	if (grpid == ZB_NEG_BIT) {
		/* no group IDs available */
		zbcoll->grptbl[ZB_NEG_BIT] = NULL;
		zb->error = -FI_EBUSY;
		return;
	}

	/* we found our group ID */
	fastlock_acquire(&zbcoll->lock);
	zb->grpid = grpid;
	zbcoll->grptbl[grpid] = zb;
	_clrbit(&zbcoll->grpmsk, grpid);
	zbcoll->grptbl[ZB_NEG_BIT] = NULL;
	fastlock_release(&zbcoll->lock);

	zb->error = FI_SUCCESS;

	/* chain to the next callback */
	cxip_zbcoll_pop_cb(zb);
}

/**
 * @brief Negotiate a group id among participants.
 *
 * Note that -FI_EAGAIN is self-clearing with progression, and indicates that
 * another negotiation is already in progress for a different group. To preserve
 * negotiation ordering, any call to this function must be repeated until it
 * stops returning -FI_EAGAIN, before attempting a different negotiation.
 *
 * @param zb : zbcoll structure
 * @return int : FI_SUCCESS or error value
 */
int cxip_zbcoll_getgroup(struct cxip_zbcoll_obj *zb)
{
	struct cxip_ep_zbcoll_obj *zbcoll;
	struct cxip_zbcoll_state *zbs;
	union cxip_match_bits mb = {.raw = 0};
	int i, n, ret;

	zbcoll = &zb->ep_obj->zbcoll;
	if (zbcoll->disable) {
		CXIP_DBG("ZBCOLL disabled\n");
		return FI_SUCCESS;
	}

	/* function could be called by non-participating nodes */
	if (!zb)
		return -FI_EADDRNOTAVAIL;

	/* check for malformed object */
	if (zb->grpid > ZB_NEG_BIT)
		return -FI_EINVAL;

	/* getgroup operations must be serialized */
	ret = FI_SUCCESS;
	fastlock_acquire(&zbcoll->lock);
	if (!zbcoll->grptbl[ZB_NEG_BIT])
		zbcoll->grptbl[ZB_NEG_BIT] = zb;
	else
		ret = -FI_EAGAIN;
	fastlock_release(&zbcoll->lock);
	if (ret)
		return ret;

	/* completion handler -- execution precedes existing handlers */
	ret = cxip_zbcoll_push_cb(zb, _getgroup_done, NULL);
	if (ret) {
		/* stop negotiating, inherently atomic */
		zbcoll->grptbl[ZB_NEG_BIT] = NULL;
		return ret;
	}

	/* Loop is for testing. Production has simcount == 1. The shuffle array
	 * allows the test to process simulated nics out-of-order, to probe
	 * for problems based on sequencing of operations.
	 */
	zb->error = FI_SUCCESS;
	zb->busy = zb->simcount;
	for (i = 0; i < zb->simcount; i++) {
		n = (zb->simcount > 1 && zb->shuffle) ? zb->shuffle[i] : i;
		zbs = &zb->state[n];
		zbs->dataval = zbcoll->grpmsk;
		zbs->dataptr = &zbs->dataval;
		zbs->contribs++;
		/* if terminal leaf node, will send up immediately */
		mb.zb_data = zbcoll->grpmsk;
		advance(zbs, mb.raw);
	}
	return FI_SUCCESS;
}

/**
 * @brief Release negotiated group id.
 *
 * @param zb : zbcoll structure
 */
void cxip_zbcoll_rlsgroup(struct cxip_zbcoll_obj *zb)
{
	struct cxip_ep_zbcoll_obj *zbcoll;

	if (!zb || zb->grpid > ZB_NEG_BIT)
		return;

	zbcoll = &zb->ep_obj->zbcoll;

	fastlock_acquire(&zbcoll->lock);
	_setbit(&zbcoll->grpmsk, zb->grpid);
	zbcoll->grptbl[zb->grpid] = NULL;
	zb->grpid = ZB_NEG_BIT;
	fastlock_release(&zbcoll->lock);
}

/**
 * @brief Initiate a data broadcast.
 *
 * All participants call this.
 *
 * The data is supplied by the root node data buffer.
 * The data is delivered to the child node data buffers.
 *
 * @param zb      : zbcoll structure
 * @param dataptr : pointer to data buffer
 * @param usrfunc : completion callback function
 * @param usrptr  : data pointer for callback
 * @return int : FI_SUCCESS or error value
 */
int cxip_zbcoll_broadcast(struct cxip_zbcoll_obj *zb, uint64_t *dataptr)
{
	struct cxip_ep_zbcoll_obj *zbcoll;
	struct cxip_zbcoll_state *zbs;
	union cxip_match_bits mb = {.raw = 0};
	int i, n;

	/* low level testing */
	zbcoll = &zb->ep_obj->zbcoll;
	if (zbcoll->disable) {
		CXIP_DBG("ZBCOLL disabled\n");
		return FI_SUCCESS;
	}

	/* function could be called on non-participating NIDs */
	if (!zb)
		return -FI_EADDRNOTAVAIL;

	/* operations on a single zb_obj are serialized */
	if (zb->busy)
		return -FI_EAGAIN;

	/* Loop is for testing. Production has simcount == 1. The shuffle array
	 * allows the test to process simulated nics out-of-order, to probe
	 * for problems based on sequencing of operations.
	 */
	zb->error = FI_SUCCESS;
	zb->busy = zb->simcount;
	for (i = 0; i < zb->simcount; i++) {
		n = (zb->simcount > 1 && zb->shuffle) ? zb->shuffle[i] : i;
		zbs = &zb->state[n];
		zbs->dataptr = dataptr;
		zbs->contribs++;
		/* if terminal leaf node, will send up immediately */
		advance(zbs, mb.raw);
	}

	return FI_SUCCESS;
}

/**
 * @brief Initiate a no-data barrier.
 *
 * All participants call this.
 *
 * @param zb      : zbcoll structure
 * @param usrfunc : completion callback function
 * @param usrptr  : data pointer for callback
 * @return int : FI_SUCCESS or error value
 */
int cxip_zbcoll_barrier(struct cxip_zbcoll_obj *zb)
{
	return cxip_zbcoll_broadcast(zb, NULL);
}

/**
 * @brief Poll for zbcoll completion completion.
 *
 * @param ep_obj
 */
void cxip_zbcoll_progress(struct cxip_ep_obj *ep_obj)
{
	cxip_ep_ctrl_progress(ep_obj);
}

/**
 * @brief Intialize the zbcoll system.
 *
 * @param ep_obj : endpoint
 * @return int : FI_SUCCESS or error value
 */
int cxip_zbcoll_init(struct cxip_ep_obj *ep_obj)
{
	struct cxip_ep_zbcoll_obj *zbcoll;

	zbcoll = &ep_obj->zbcoll;
	zbcoll->grpmsk = -1ULL;
	zbcoll->grptbl = calloc(ZB_MAP_BITS, sizeof(void *));
	if (!zbcoll->grptbl)
		return -FI_ENOMEM;
	fastlock_init(&zbcoll->lock);
	ofi_atomic_initialize32(&zbcoll->dsc_count, 0);
	ofi_atomic_initialize32(&zbcoll->err_count, 0);
	ofi_atomic_initialize32(&zbcoll->ack_count, 0);
	ofi_atomic_initialize32(&zbcoll->rcv_count, 0);

	return FI_SUCCESS;
}

/**
 * @brief Cleanup all operations in progress.
 *
 * @param ep_obj : endpoint
 */
void cxip_zbcoll_fini(struct cxip_ep_obj *ep_obj)
{
	struct cxip_ep_zbcoll_obj *zbcoll;
	int i;

	zbcoll = &ep_obj->zbcoll;
	for (i = 0; i < ZB_MAP_BITS; i++)
		cxip_zbcoll_free(zbcoll->grptbl[i]);
	free(zbcoll->grptbl);
	zbcoll->grptbl = NULL;
}

/**
 * @brief Reset the endpoint counters.
 *
 * @param ep : endpoint
 */
void cxip_zbcoll_reset_counters(struct cxip_ep_obj *ep_obj)
{
	struct cxip_ep_zbcoll_obj *zbcoll;

	zbcoll = &ep_obj->zbcoll;
	ofi_atomic_set32(&zbcoll->dsc_count, 0);
	ofi_atomic_set32(&zbcoll->err_count, 0);
	ofi_atomic_set32(&zbcoll->ack_count, 0);
	ofi_atomic_set32(&zbcoll->rcv_count, 0);
}
