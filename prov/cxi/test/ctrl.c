/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2021 Hewlett Packard Enterprise Development LP
 */
#include <stdio.h>
#include <stdlib.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>

#include <ofi.h>

#include "cxip.h"
#include "cxip_test_common.h"

TestSuite(ctrl, .init = cxit_setup_rma, .fini = cxit_teardown_rma,
	  .timeout = CXIT_DEFAULT_TIMEOUT);

/**
 * Test reversibility of N <-> (r,c), error conditions
 *
 * For a range of radix values, select a node number (N), convert to
 * a (row,column) pair, and then convert back to node number. These
 * should match, unless an invalid column (for a row) is specified,
 * in which case we see an error.
 */
Test(ctrl, radix_tree_reversible)
{
	int radix, N, M, row, col, siz, rowold, rowwid;

	for (radix = 1; radix < 8; radix++) {
		rowold = -1;
		rowwid = 1;
		for (N = 0; N < 256; N++) {
			/* test reversibility */
			cxip_tree_rowcol(radix, N, &row, &col, &siz);
			cxip_tree_nodeidx(radix, row, col, &M);
			cr_assert(M == N, "M=%d != N=%d\n", M, N);
			if (rowold != row) {
				rowold = row;
				rowwid *= radix;
			}
			/* test invalid column */
			col = rowwid + 1;
			cxip_tree_nodeidx(radix, row, col, &M);
			cr_assert(M == -1,
				  "radix=%d N=%d row=%d col=%d"
				  " M=%d != -1\n",
				  radix, N, row, col, M);
		}
	}
}

/**
 * Test parent/child mapping.
 *
 * For a range of radix values, generate the relatives in the tree (one
 * parent, multiple children), and confirm that these relatives have the
 * expected position in the tree, which guarantees that we have no loops
 * in the tree, and that every node has a parent (except the root), and
 * is a child of its parent.
 */
Test(ctrl, radix_tree_mapping)
{
	int radix, nodes, N, M, *rels;
	int count, parent, child, i;

	/* Test radix zero case */
	M = cxip_tree_relatives(0, 0, 0, NULL);
	cr_assert(M == 0);

	/* Test expected pattern of parent/child indices */
	for (radix = 1; radix < 8; radix++) {
		/* only needs radix+1, but for test, provide extra space */
		rels = calloc(radix+2, sizeof(int));
		for (nodes = 0; nodes < 256; nodes++) {
			count = 0;
			parent = -1;
			child = 1;
			for (N = 0; N < nodes; N++) {
				M = cxip_tree_relatives(radix, N, nodes, rels);
				cr_assert(M >= 0);
				cr_assert(M <= radix+1);
				if (M > 0) {
					/* test parent node index */
					cr_assert(rels[0] == parent,
						"radix=%d nodes=%d index=%d"
						" parent=%d != rels[0]=%d\n",
						radix, nodes, N, parent, rels[0]);
					/* test child node indices */
					for (i = 1; i < M; i++, child++)
						cr_assert(rels[i] == child,
							"radix=%d nodes=%d"
							" index=%d child=%d"
							" != rels[%d]=%d\n",
							radix, nodes, N,
							child, i, rels[i]);
				}
				count++;
				if (N == 0 || count >= radix) {
					count = 0;
					parent++;
				}
			}
		}
		free(rels);
	}
}

/**
 * @brief Test the valid and invalid configurations.
 *
 */
Test(ctrl, zb_config)
{
	struct cxip_ep *cxip_ep;
	struct cxip_zbcoll_obj *zb;
	uint32_t nids[] = {2, 3, 4, 5};
	int numnids = sizeof(nids)/sizeof(uint32_t);
	int ret;

	cxip_ep = container_of(cxit_ep, struct cxip_ep, ep.fid);
	zb = &cxip_ep->ep_obj->zbcoll;

	/* cannot specify numnids > 0 with nids == NULL */
	ret = cxip_zbcoll_config(cxit_ep, numnids, NULL, false);
	cr_assert(ret == -FI_EINVAL, "numnids && !nids, ret=%d\n", ret);

	/* test case, do not generate a tree */
	ret = cxip_zbcoll_config(cxit_ep, 0, NULL, false);
	cr_assert(ret == 0, "!numnids, ret=%d\n", ret);
	cr_assert(zb->count == 1, "!numnids, cnt=%d\n", zb->count);

	/* request simulation */
	ret = cxip_zbcoll_config(cxit_ep, numnids, nids, true);
	cr_assert(ret == 0, "sim1, ret=%d\n", ret);
	cr_assert(zb->count == numnids, "sim1, cnt=%d\n", zb->count);

	/* resize simulation */
	ret = cxip_zbcoll_config(cxit_ep, numnids-1, nids, true);
	cr_assert(ret == 0, "sim2, ret=%d\n", ret);
	cr_assert(zb->count == numnids-1, "sim2, cnt=%d\n", zb->count);

	/* exercise real setup failure, real addr not in list */
	ret = cxip_zbcoll_config(cxit_ep, numnids-1, nids, false);
	cr_assert(ret == -FI_EADDRNOTAVAIL, "fail, ret=%d\n", ret);
	cr_assert(zb->count == numnids-1, "fail, cnt=%d\n", zb->count);

	/* exercise real setup success, real addr is in list */
	nids[1] = cxip_ep->ep_obj->src_addr.nic;
	ret = cxip_zbcoll_config(cxit_ep, numnids, nids, false);
	cr_assert(ret == 0, "good, ret=%d\n", ret);
	cr_assert(zb->count == 1, "good, cnt=%d\n", zb->count);
}

/**
 * Send a single packet using a pure-send (self to self) configuration.
 */
Test(ctrl, zb_send0)
{
	struct cxip_ep *cxip_ep;
	struct cxip_zbcoll_obj *zb;
	struct cxip_zbcoll_state *zbs;
	union cxip_match_bits mb = {.raw = 0};
	uint32_t mynid;
	uint32_t err, ack, rcv, cnt;
	int ret;

	cr_assert(sizeof(union cxip_match_bits) == 8);

	cxip_ep = container_of(cxit_ep, struct cxip_ep, ep.fid);
	zb = &cxip_ep->ep_obj->zbcoll;
	zb->disable = true;

	/* Send-to-self configuration */
	ret = cxip_zbcoll_config(cxit_ep, 0, NULL, false);
	cr_assert(ret == 0);
	cr_assert(zb->count == 1, "zb->count = %d, != %d\n",
		  zb->count, 1);

	/* Legitimate send to self */
	cxip_zbcoll_reset_counters(cxit_ep);
	mynid = cxip_ep->ep_obj->src_addr.nic;
	cxip_zbcoll_send(cxip_ep->ep_obj, mynid, mynid, mb.raw);
	zbs = &zb->state[0];
	cnt = 0;
	do {
		usleep(10);
		cxip_ep_ctrl_progress(cxip_ep->ep_obj);
		err = ofi_atomic_get32(&zbs->err_count);
		ack = ofi_atomic_get32(&zbs->ack_count);
		rcv = ofi_atomic_get32(&zbs->rcv_count);
		ret = (err || (ack && rcv));
		cnt++;
	} while (!ret && cnt < 1000);
	cr_assert(cnt < 1000, "repeat count = %d\n", cnt);
	cr_assert(err == 0, "err = %d, != 0\n", err);
	cr_assert(ack == 1, "ack = %d, != 1\n", ack);
	cr_assert(rcv == 1, "rcv = %d, != 1\n", rcv);

	/* Invalid send to non-existent address */
	cxip_zbcoll_reset_counters(cxit_ep);
	mynid = cxip_ep->ep_obj->src_addr.nic + 1;
	cxip_zbcoll_send(cxip_ep->ep_obj, mynid, mynid, mb.raw);
	zbs = &zb->state[0];
	cnt = 0;
	do {
		usleep(10);
		cxip_ep_ctrl_progress(cxip_ep->ep_obj);
		err = ofi_atomic_get32(&zbs->err_count);
		ack = ofi_atomic_get32(&zbs->ack_count);
		rcv = ofi_atomic_get32(&zbs->rcv_count);
		ret = (err || (ack && rcv));
		cnt++;
	} while (!ret && cnt < 1000);
	cr_assert(cnt >= 1000, "repeat count = %d\n", cnt);
	cr_assert(err == 0, "err = %d, != 0\n", err);
	cr_assert(ack == 0, "ack = %d, != 0\n", ack);
	cr_assert(rcv == 0, "rcv = %d, != 0\n", rcv);
}

/* Send a single packet from src to dst in NETSIM simulation */
static void _sendN(struct cxip_ep_obj *ep_obj,
		   int num_nids, uint32_t *nids, int srcidx, int dstidx)
{
	struct cxip_zbcoll_obj *zb = &ep_obj->zbcoll;
	union cxip_match_bits mb = {.zb_data=1234};
	int i, ret, cnt, err, ack, rcv;

	/* send to dstidx simulated address */
	zb->disable = true;
	cxip_zbcoll_reset_counters(cxit_ep);
	cxip_zbcoll_send(ep_obj, nids[srcidx], nids[dstidx], mb.raw);

	/* wait for errors, or completion */
	cnt = 0;
	do {
		usleep(10);
		cxip_ep_ctrl_progress(ep_obj);
		err = ofi_atomic_get32(&zb->state[srcidx].err_count) +
		      ofi_atomic_get32(&zb->state[dstidx].err_count);
		ack = ofi_atomic_get32(&zb->state[srcidx].ack_count);
		rcv = ofi_atomic_get32(&zb->state[dstidx].rcv_count);
		ret = (err || (ack && rcv));
		cnt++;
	} while (!ret && cnt < 1000);
	cr_assert(cnt < 1000, "repeat count = %d\n", cnt);
	cr_assert(err == 0, "err = %d, != 0\n", err);
	cr_assert(ack == 1, "ack = %d, != 1\n", ack);
	cr_assert(rcv == 1, "rcv = %d, != 1\n", rcv);

	/* make sure no bleed-over into other states */
	for (i = 0; i < num_nids; i++) {
		if (i == srcidx || i == dstidx)
			continue;
		err = ofi_atomic_get32(&zb->state[i].err_count);
		ack = ofi_atomic_get32(&zb->state[i].ack_count);
		rcv = ofi_atomic_get32(&zb->state[i].rcv_count);
		cr_assert(err == 0, "uninvolved[%d] err = %d\n", i, err);
		cr_assert(ack == 0, "uninvolved[%d] ack = %d\n", i, ack);
		cr_assert(rcv == 0, "uninvolved[%d] rcv = %d\n", i, rcv);
	}
	zb->disable = false;
}

#define	NUM_NIDS(nids)	sizeof(nids)/sizeof(uint32_t)

/* Send packet from every src to every dst in simulation */
Test(ctrl, zb_sendN)
{
	struct cxip_ep *cxip_ep;
	struct cxip_zbcoll_obj *zb;
	int srcidx, dstidx, ret;

	uint32_t nids[] = {10, 11, 12, 13};
	int num_nids = NUM_NIDS(nids);

	cxip_ep = container_of(cxit_ep, struct cxip_ep, ep.fid);
	zb = &cxip_ep->ep_obj->zbcoll;
	zb->disable = true;

	ret = cxip_zbcoll_config(cxit_ep, num_nids, nids, true);
	cr_assert(ret == 0);
	cr_assert(zb->count == num_nids,
		  "zb->count = %d, != %d\n", zb->count, num_nids);

	for (srcidx = 0; srcidx < num_nids; srcidx++)
		for (dstidx = 0; dstidx < num_nids; dstidx++)
			_sendN(cxip_ep->ep_obj, num_nids, nids, srcidx, dstidx);
	zb->disable = false;
}

__attribute__((unused))
static void dumpmap(struct cxip_zbcoll_obj *zb)
{
	int i, j;

	printf("MAP=======\n");
	for (i = 0; i < zb->count; i++) {
		printf("%2d:", i);
		for (j = 0; j < zb->state[i].num_relatives; j++)
			printf(" %2d", zb->state[i].relatives[j]);
		printf("\n");
	}
	printf("\n");
}

static int _await_complete(struct cxip_ep *cxip_ep, int num_nids)
{
	struct cxip_zbcoll_obj *zb;
	struct cxip_zbcoll_state *zbs;
	int i, rep;

	zb = &cxip_ep->ep_obj->zbcoll;
	for (rep = 0; rep < 1000; rep++) {
		cxip_ep_ctrl_progress(cxip_ep->ep_obj);
		for (i = 0; i < num_nids; i++) {
			zbs = &zb->state[i];
			if (! zbs->complete)
				break;
		}
		if (i >= num_nids)
			return 0;
		usleep(1000);
	}
	return -1;
}

/* Test barrier in simulation */
Test(ctrl, zb_barrier)
{
	struct cxip_ep *cxip_ep;
	struct cxip_zbcoll_obj *zb;
	int i, rep, ret;

	uint32_t nids[] = {10, 11, 12, 13, 14, 15, 16, 17};
	int num_nids = NUM_NIDS(nids);

	cxip_ep = container_of(cxit_ep, struct cxip_ep, ep.fid);
	zb = &cxip_ep->ep_obj->zbcoll;

	ret = cxip_zbcoll_config(cxit_ep, num_nids, nids, true);
	cr_assert(ret == 0);
	cr_assert(zb->count == num_nids,
		  "zb->count = %d, != %d\rep", zb->count, num_nids);

	/* Do this test twice */
	for (rep = 0; rep < 2; rep++) {
		/* Issue num_nids Barrier operations */
		for (i = 0; i < num_nids; i++) {
			ret = cxip_zbcoll_barrier(cxit_ep);
			cr_assert(ret == 0, "%d barrier[%d] = %d\n",
				  rep, i, ret);
		}
		/* Allow this time to complete */
		ret = _await_complete(cxip_ep, num_nids);
		cr_assert(ret == FI_SUCCESS, "%d barrier = %d\n",
			  rep, ret);
		/* Try to issue another barrier -- should fail */
		ret = cxip_zbcoll_barrier(cxit_ep);
		cr_assert(ret == -FI_EAGAIN, "%d barrier = %d\n",
			  rep, ret);
		/* Officially progress */
		ret = cxip_zbcoll_progress(cxit_ep);
		cr_assert(ret == FI_SUCCESS, "%d barrier = %d\n",
			  rep, ret);
	}
}

/* Test bcast in simulation */
Test(ctrl, zb_bcast)
{
	struct cxip_ep *cxip_ep;
	struct cxip_zbcoll_obj *zb;
	int i, rep, ret;

	uint32_t nids[] = {10, 11, 12, 13, 14, 15, 16, 17};
	uint64_t data[NUM_NIDS(nids)];
	int num_nids = NUM_NIDS(nids);

	cxip_ep = container_of(cxit_ep, struct cxip_ep, ep.fid);
	zb = &cxip_ep->ep_obj->zbcoll;

	ret = cxip_zbcoll_config(cxit_ep, num_nids, nids, true);
	cr_assert(ret == 0);
	cr_assert(zb->count == num_nids,
		  "zb->count = %d, != %d\n", zb->count, num_nids);

	for (rep = 0; rep < 2; rep++) {
		memset(data, 0, sizeof(data));
		data[0] = (rand() & ((1 << 29) - 1)) | (1 << 28);
		for (i = 0; i < num_nids; i++) {
			ret = cxip_zbcoll_bcast(cxit_ep, &data[i]);
			cr_assert(ret == 0, "%d bcast[%d] = %d\n",
				  rep, i, ret);
		}
		/* Allow this to complete */
		ret = _await_complete(cxip_ep, num_nids);
		cr_assert(ret == FI_SUCCESS, "%d barrier = %d\n",
			  rep, ret);
		/* Try to issue another barrier -- should fail */
		ret = cxip_zbcoll_bcast(cxit_ep, &data[0]);
		cr_assert(ret == -FI_EAGAIN, "%d bcast = %d\n",
			  rep, ret);
		/* Officially progress */
		ret = cxip_zbcoll_progress(cxit_ep);
		cr_assert(ret == FI_SUCCESS, "%d bcast = %d\n",
			  rep, ret);
		for (i = 0; i < num_nids; i++)
			cr_assert(data[0] == data[i],
				  "%d ret data[%d] = %ld\n", rep, i, data[i]);
	}
}
