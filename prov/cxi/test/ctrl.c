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

#define	trc CXIP_TRACE

TestSuite(ctrl, .init = cxit_setup_rma, .fini = cxit_teardown_rma,
	  .timeout = CXIT_DEFAULT_TIMEOUT);

/**
 * @brief Test reversibility of N <-> (r,c), error conditions
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
 * @brief Test parent/child mapping.
 *
 * For a range of radix values, generate the relatives in the tree (one
 * parent, multiple children), and confirm that these relatives have the
 * expected position in the tree, which guarantees that we have no loops
 * in the tree, and that every node has a parent (except the root), and
 * is a child of its parent.
 */
Test(ctrl, radix_tree_mapping)
{
	int *rels, parent, child;
	int radix, nodes, N, M;
	int count, i;

	/* Test radix zero case */
	M = cxip_tree_relatives(0, 0, 0, NULL);
	cr_assert(M == 0);

	/* Test expected pattern of parent/child indices */
	for (radix = 1; radix < 8; radix++) {
		/* only needs radix+1, but for test, provide extra space */
		rels = calloc(radix+2, sizeof(*rels));
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

/* Utility to show the node relatives */
__attribute__((unused))
static void dumpmap(struct cxip_zbcoll_obj *zb)
{
	int i, j;

	printf("MAP=======\n");
	for (i = 0; i < zb->simcount; i++) {
		printf("%2d:", i);
		for (j = 0; j < zb->state[i].num_relatives; j++)
			printf(" %2d", zb->state[i].relatives[j]);
		printf("\n");
	}
	printf("\n");
}

/**
 * Generate simualted addresses.
 *
 * This generates size + 1 addresses.
 *
 * fiaddr[0..size-1] are simulated addresses.
 *
 * fiaddr[size] is the real hardware address of this node.
 *
 * Note that under NETSIM, the real NIC address is 0, which coincidentally
 * matches the simulation NIC address of 0.
 */
static int _generate_sim_addrs(struct cxip_ep_obj *ep_obj, int size,
			       fi_addr_t **fiaddrp)
{
	struct cxip_addr *caddrs;
	fi_addr_t *fiaddrs;
 	int i, ret;

	if (fiaddrp)
		*fiaddrp = NULL;
	if (size < 1)
		return -FI_EINVAL;

	/* NOTE: creates a "hidden" final addr == actual NIC address */
	ret = -FI_ENOMEM;
	caddrs = calloc(size + 1, sizeof(*caddrs));
	fiaddrs = calloc(size + 1, sizeof(*fiaddrs));
	if (!caddrs || !fiaddrs)
		goto cleanup;

	/* Prepare simulated addresses, including "hidden" addr */
	for (i = 0; i < size; i++) {
		caddrs[i].nic = i;
		caddrs[i].pid = ep_obj->src_addr.pid;
	}
	caddrs[i++] = ep_obj->src_addr;

	/* Register these with the av */
	ret = fi_av_insert(&ep_obj->av->av_fid, caddrs, size + 1, fiaddrs,
			   0L, NULL);
	if (ret < 1)
		goto cleanup;

	/* returning */
	if (fiaddrp) {
		*fiaddrp = fiaddrs;
		fiaddrs = NULL;
	}
	ret = FI_SUCCESS;
cleanup:
	free(fiaddrs);
	free(caddrs);
	return ret;
}

/**
 * @brief Test the valid and invalid cxip_zbcoll_obj configurations.
 */
Test(ctrl, zb_config)
{
	struct cxip_ep *cxip_ep;
	struct cxip_ep_obj *ep_obj;
	struct cxip_zbcoll_obj *zb;
	fi_addr_t *fiaddrs;
	int ret;

	int num_addrs = 5;

	cxip_ep = container_of(cxit_ep, struct cxip_ep, ep.fid);
	ep_obj = cxip_ep->ep_obj;

	trc("Generating addresses\n");
	ret = _generate_sim_addrs(ep_obj, num_addrs, &fiaddrs);
	cr_assert(!ret, "out of memory\n");
	cr_assert(fiaddrs, "no fi_addrs\n");

	/* cannot specify num_addrs > 0 with addrs == NULL */
	trc("case: null\n");
	ret = cxip_zbcoll_alloc(ep_obj, num_addrs, NULL, false, &zb);
	cr_assert(ret == -FI_EINVAL,
		  "illegal config: num_addrs && !addrs, ret=%d\n", ret);
	cxip_zbcoll_free(zb);

	/* test case, do not generate a tree */
	trc("case: no tree\n");
	ret = cxip_zbcoll_alloc(ep_obj, 0, NULL, false, &zb);
	cr_assert(ret == 0,
		  "no tree: ret=%d\n", ret);
	cr_assert(zb->simcount == 1,
		  "no tree: simcnt=%d\n", zb->simcount);
	cxip_zbcoll_free(zb);

	/* request simulation */
	trc("case: simulated\n");
	ret = cxip_zbcoll_alloc(ep_obj, num_addrs, fiaddrs, true, &zb);
	cr_assert(ret == 0,
		  "sim tree 4: ret=%d\n", ret);
	cr_assert(zb->simcount == num_addrs,
		  "sim tree 4: cnt=%d\n", zb->simcount);
	cxip_zbcoll_free(zb);

	/* resize simulation */
	trc("case: resize simulation\n");
	ret = cxip_zbcoll_alloc(ep_obj, num_addrs-1, fiaddrs, true, &zb);
	cr_assert(ret == 0,
		  "sim tree 3: ret=%d\n", ret);
	cr_assert(zb->simcount == num_addrs-1,
		  "sim tree 3: cnt=%d\n", zb->simcount);
	cxip_zbcoll_free(zb);

	/* exercise real setup success, ensure real addr is in list */
	trc("case: real addresses success\n");
	ret = cxip_zbcoll_alloc(ep_obj, num_addrs, &fiaddrs[1], false, &zb);
	cr_assert(ret == 0,
		  "real tree, in list: ret=%d\n", ret);
	cr_assert(zb->simcount == 1,
		  "real tree, in list: simcnt=%d\n", zb->simcount);
	cxip_zbcoll_free(zb);

	/* exercise real setup failure, ensure real addr not in list */
	trc("case: real addresses failure\n");
	ret = cxip_zbcoll_alloc(ep_obj, num_addrs-1, &fiaddrs[1], false, &zb);
	cr_assert(ret == -FI_EADDRNOTAVAIL,
		  "real tree, not in list: ret=%d\n", ret);
	cxip_zbcoll_free(zb);

	free(fiaddrs);
}

/**
 * @brief Send a single packet using a self to self send-only configuration.
 */
Test(ctrl, zb_send0)
{
	struct cxip_ep *cxip_ep;
	struct cxip_ep_obj *ep_obj;
	struct cxip_zbcoll_obj *zb;
	union cxip_match_bits mb = {.raw = 0};
	uint32_t dsc, err, ack, rcv, cnt;
	int ret;

	fi_addr_t *fiaddrs;
	int num_addrs = 2;

	cr_assert(sizeof(union cxip_match_bits) == 8);

	cxip_ep = container_of(cxit_ep, struct cxip_ep, ep.fid);
	ep_obj = cxip_ep->ep_obj;
	ret = _generate_sim_addrs(ep_obj, num_addrs, &fiaddrs);
	cr_assert(!ret, "out of memory\n");

	/* Set up the send-only zbcoll */
	ret = cxip_zbcoll_alloc(ep_obj, 1, &fiaddrs[num_addrs], false, &zb);
	cr_assert(ret == 0, "cxip_zbcoll_alloc() = %d\n", ret);
	cr_assert(zb != NULL);
	cr_assert(zb->simcount == 1);
	cr_assert(zb->state != NULL);

	/* Test that if disabled, getgroup is no-op */
	ep_obj->zbcoll.disable = true;
	ret = cxip_zbcoll_getgroup(zb);
	cr_assert(ret == 0, "getgroup = %d\n", ret);

	/* Legitimate send to self */
	cxip_zbcoll_reset_counters(ep_obj);
	cxip_zbcoll_send(zb, 0, 0, mb.raw);
	cnt = 0;
	do {
		usleep(1);
		cxip_zbcoll_progress(ep_obj);
		cxip_zbcoll_get_counters(ep_obj, &dsc, &err, &ack, &rcv);
		ret = (dsc || err || (ack && rcv));
		cnt++;
	} while (!ret && cnt < 1000);
	cr_assert(cnt < 1000, "repeat count = %d >= %d\n", cnt, 1000);
	cr_assert(dsc == 0, "dsc = %d, != 0\n", dsc);
	cr_assert(err == 0, "err = %d, != 0\n", err);
	cr_assert(ack == 1, "ack = %d, != 1\n", ack);
	cr_assert(rcv == 1, "rcv = %d, != 1\n", rcv);

	/* Invalid send to out-of-range address index */
	cxip_zbcoll_reset_counters(ep_obj);
	cxip_zbcoll_send(zb, 0, num_addrs, mb.raw);
	cnt = 0;
	do {
		usleep(1);
		cxip_zbcoll_progress(ep_obj);
		cxip_zbcoll_get_counters(ep_obj, &dsc, &err, &ack, &rcv);
		ret = (err || dsc || (ack && rcv));
		cnt++;
	} while (!ret && cnt < 1000);
	cr_assert(cnt < 1000, "repeat count = %d < %d\n", cnt, 1000);
	cr_assert(dsc == 0, "dsc = %d, != 0\n", dsc);
	cr_assert(err == 1, "err = %d, != 1\n", err);
	cr_assert(ack == 0, "ack = %d, != 0\n", ack);
	cr_assert(rcv == 0, "rcv = %d, != 0\n", rcv);

	cxip_zbcoll_free(zb);
	free(fiaddrs);
}

/* utility to send from src to dst */
static void _send(struct cxip_zbcoll_obj *zb, int srcidx, int dstidx)
{
	struct cxip_ep_obj *ep_obj;
	union cxip_match_bits mb = {.zb_data=0};
	int ret, cnt;
	uint32_t dsc, err, ack, rcv;

	/* send to dstidx simulated address */
	ep_obj = zb->ep_obj;
	cxip_zbcoll_reset_counters(ep_obj);
	cxip_zbcoll_send(zb, srcidx, dstidx, mb.raw);

	/* wait for errors, or completion */
	cnt = 0;
	do {
		usleep(1);
		cxip_zbcoll_progress(ep_obj);
		cxip_zbcoll_get_counters(ep_obj, &dsc, &err, &ack, &rcv);
		ret = (err || dsc || (ack && rcv));
		cnt++;
	} while (!ret && cnt < 1000);
	cr_assert(cnt < 1000, "repeat count = %d\n", cnt);

	cr_assert(dsc == 0, "dsc = %d, != 0\n", dsc);
	cr_assert(err == 0, "err = %d, != 0\n", err);
	cr_assert(ack == 1, "ack = %d, != 1\n", ack);
	cr_assert(rcv == 1, "rcv = %d, != 1\n", rcv);
}

/**
 * @brief Send a single packet from each src to dst in NETSIM simulation.
 *
 * Scales as O(N^2), so don't go nuts on the number of addrs.
 */
Test(ctrl, zb_sendN)
{
	struct cxip_ep *cxip_ep;
	struct cxip_ep_obj *ep_obj;
	struct cxip_zbcoll_obj *zb;
	int srcidx, dstidx, ret;

	fi_addr_t *fiaddrs;
	int num_addrs = 5;

	cxip_ep = container_of(cxit_ep, struct cxip_ep, ep.fid);
	ep_obj = cxip_ep->ep_obj;

	ret = _generate_sim_addrs(ep_obj, num_addrs, &fiaddrs);
	cr_assert(!ret, "out of memory\n");

	ret = cxip_zbcoll_alloc(ep_obj, num_addrs, fiaddrs, true, &zb);
	cr_assert(ret == 0, "cxip_zbcoll_alloc() = %d\n", ret);
	cr_assert(zb != NULL);
	cr_assert(zb->simcount == num_addrs);
	cr_assert(zb->state != NULL);

	/* Test that if disabled, getgroup is no-op */
	ep_obj->zbcoll.disable = true;
	ret = cxip_zbcoll_getgroup(zb);
	cr_assert(ret == 0, "getgroup = %d\n", ret);

	for (srcidx = 0; srcidx < num_addrs; srcidx++)
		for (dstidx = 0; dstidx < num_addrs; dstidx++)
			_send(zb, srcidx, dstidx);
	cxip_zbcoll_free(zb);
	free(fiaddrs);
}

/* Utility to wait until a collective has completed */
static int _await_complete(struct cxip_zbcoll_obj *zb)
{
	uint32_t rep;

	/* We only wait for 1 sec */
	for (rep = 0; rep < 10000; rep++) {
		usleep(100);
		cxip_zbcoll_progress(zb->ep_obj);
		if (!zb->busy || zb->error)
			break;
	}
	return zb->error;
}

void _shuffle_array32(uint32_t *array, size_t size)
{
	uint32_t i, j, t;

	for (i = 0; i < size-1; i++) {
		j = i + rand() / (RAND_MAX / (size - i) + 1);
		t = array[j];
		array[j] = array[i];
		array[i] = t;
	}
}

/* create a randomized shuffle array */
void _addr_shuffle(struct cxip_zbcoll_obj *zb, bool shuffle)
{
	struct timespec tv;
	int i;

	clock_gettime(CLOCK_MONOTONIC, &tv);
	srand((unsigned int)tv.tv_nsec);
	free(zb->shuffle);
	zb->shuffle = calloc(zb->simcount, sizeof(uint32_t));
	for (i = 0; i < zb->simcount; i++)
		zb->shuffle[i] = i;
	if (shuffle)
		_shuffle_array32(zb->shuffle, zb->simcount);
}

/*****************************************************************/
/**
 * @brief Test simulated getgroup.
 *
 * This exercises the basic broad operation, the user callback, and the
 * non-concurrency lockout.
 *
 * This is simulated in a single thread, so it tests only a single barrier
 * across multiple addrs. It randomizes the nid processing order, and performs
 * multiple barriers to uncover any ordering issues.
 */

struct getgroup_data {
	int count;
};
static void getgroup_func(struct cxip_zbcoll_obj *zb, void *usrptr)
{
	struct getgroup_data *data = (struct getgroup_data *)usrptr;
	data->count++;
}
Test(ctrl, zb_getgroup)
{
	struct cxip_ep *cxip_ep;
	struct cxip_ep_obj *ep_obj;
	struct cxip_zbcoll_obj **zb;
	struct getgroup_data zbd;
	int i, ret;
	uint32_t dsc, err, ack, rcv;
	fi_addr_t *fiaddrs;
	int num_addrs = 9;	// arbitrary
	int num_zb = 43;	// limit for simulation
	int cnt = 0;

	cxip_ep = container_of(cxit_ep, struct cxip_ep, ep.fid);
	ep_obj = cxip_ep->ep_obj;

	ret = _generate_sim_addrs(ep_obj, num_addrs, &fiaddrs);
	cr_assert(!ret, "addrs out of memory\n");

	zb = calloc(num_zb, sizeof(struct cxip_zbcoll_obj *));
	cr_assert(zb, "zb out of memory\n");

	memset(&zbd, 0, sizeof(zbd));

	for (i = 0; i < num_zb; i++) {
		/* Verify multiple allocations */
		ret = cxip_zbcoll_alloc(ep_obj, num_addrs, fiaddrs,
					true, &zb[i]);
		cr_assert(ret == 0, "cxip_zbcoll_alloc() = %d\n", ret);
		cr_assert(zb[i]->simcount == num_addrs,
			"zb->simcount = %d, != %d\n",
			zb[i]->simcount, num_addrs);
		/* Initialize the address shuffling */
		_addr_shuffle(zb[i], false);
		/* Test getgroup operation */
		cxip_zbcoll_push_cb(zb[i], getgroup_func, &zbd);
		ret = cxip_zbcoll_getgroup(zb[i]);
		cr_assert(ret == FI_SUCCESS, "%d getgroup = %s\n",
			  i, fi_strerror(-ret));
		/* Test getgroup non-concurrency */
		ret = cxip_zbcoll_getgroup(zb[i]);
		cr_assert(ret == -FI_EAGAIN, "%d getgroup = %s\n",
			  i, fi_strerror(-ret));
		/* Poll until complete */
		ret = _await_complete(zb[i]);
		cr_assert(ret == FI_SUCCESS, "%d getgroup = %s\n",
			  i, fi_strerror(-ret));
		/* Check user callback completion count result */
		cr_assert(zbd.count == i+1, "%d zbdcount = %d\n",
			  i, zbd.count);
		/* Confirm expected grpid */
		cr_assert(zb[i]->grpid == i, "%d grpid = %d\n",
			  i, zb[i]->grpid);

		cnt += 2 * (num_addrs - 1);
	}

	/* Free item [0] and try again */
	i = 0;
	cxip_zbcoll_free(zb[i]);
	ret = cxip_zbcoll_alloc(ep_obj, num_addrs, fiaddrs, true, &zb[i]);
	cr_assert(ret == 0, "cxip_zbcoll_alloc() = %d\n", ret);
	ret = cxip_zbcoll_getgroup(zb[i]);
	cr_assert(ret == FI_SUCCESS, "retry %d getgroup = %s\n",
		  i, fi_strerror(-ret));
	ret = cxip_zbcoll_getgroup(zb[i]);
	cr_assert(ret == -FI_EAGAIN, "retry %d getgroup = %s\n",
		  i, fi_strerror(-ret));
	ret = _await_complete(zb[i]);
	cr_assert(ret == FI_SUCCESS, "retry %d getgroup = %s\n",
		  i, fi_strerror(-ret));
	cr_assert(zb[i]->grpid == i,
			"%d grpid = %d\n", i, zb[i]->grpid);
	cnt += 2 * (num_addrs - 1);

	cxip_zbcoll_get_counters(ep_obj, &dsc, &err, &ack, &rcv);
	cr_assert(dsc == 0 && err == 0,
		  "FAILED dsc=%d err=%d ack=%d rcv=%d cnt=%d\n",
		  dsc, err, ack, rcv, cnt);
	/* cleanup */
	for (i = 0; i < num_zb; i++)
		cxip_zbcoll_free(zb[i]);
	free(zb);
	free(fiaddrs);
}

/*****************************************************************/
/**
 * @brief Test simulated barrier.
 *
 * This exercises the basic broad operation, the user callback, and the
 * non-concurrency lockout.
 *
 * This is done in a single thread, so it tests only a single barrier across
 * multiple addrs. It randomizes the nid processing order, and performs multiple
 * barriers to uncover any ordering issues.
 */
struct barrier_data {
	int count;
};
static void barrier_func(struct cxip_zbcoll_obj *zb, void *usrptr)
{
	struct barrier_data *data = (struct barrier_data *)usrptr;

	/* increment the user completion count */
	data->count++;
}

Test(ctrl, zb_barrier)
{
	struct cxip_ep *cxip_ep;
	struct cxip_ep_obj *ep_obj;
	struct cxip_zbcoll_obj *zb;
	struct barrier_data zbd;
	int rep, ret;

	fi_addr_t *fiaddrs;
	int num_addrs = 9;

	cxip_ep = container_of(cxit_ep, struct cxip_ep, ep.fid);
	ep_obj = cxip_ep->ep_obj;

	ret = _generate_sim_addrs(ep_obj, num_addrs, &fiaddrs);
	cr_assert(!ret, "out of memory\n");

	ret = cxip_zbcoll_alloc(ep_obj, num_addrs, fiaddrs, true, &zb);
	cr_assert(ret == 0, "cxip_zbcoll_alloc() = %d\n", ret);
	cr_assert(zb->simcount == num_addrs,
		  "zb->simcount = %d, != %d\n", zb->simcount, num_addrs);
	/* Initialize the addresses */
	_addr_shuffle(zb, false);

	/* Acquire a group id */
	ret = cxip_zbcoll_getgroup(zb);
	cr_assert(ret == 0, "getgroup = %d\n", ret);
	ret = _await_complete(zb);
	cr_assert(ret == 0, "getgroup done = %d\n", ret);

	memset(&zbd, 0, sizeof(zbd));
	for (rep = 0; rep < 20; rep++) {
		/* Shuffle the addresses */
		_addr_shuffle(zb, true);
		/* Perform a barrier */
		cxip_zbcoll_push_cb(zb, barrier_func, &zbd);
		ret = cxip_zbcoll_barrier(zb);
		cr_assert(ret == 0, "%d barrier = %s\n",
			  rep, fi_strerror(-ret));
		/* Try again immediately, should show BUSY */
		cxip_zbcoll_push_cb(zb, barrier_func, &zbd);
		ret = cxip_zbcoll_barrier(zb);
		cr_assert(ret == -FI_EAGAIN, "%d barrier = %s\n",
			  rep, fi_strerror(-ret));
		/* Poll until complete */
		ret = _await_complete(zb);
		cr_assert(ret == FI_SUCCESS, "%d barrier = %s\n",
			  rep, fi_strerror(-ret));
	}
	/* Confirm completion count */
	cr_assert(zbd.count == rep);

	uint32_t dsc, err, ack, rcv;
	cxip_zbcoll_get_counters(ep_obj, &dsc, &err, &ack, &rcv);
	cr_assert(dsc == 0 && err == 0,
		  "FAILED dsc=%d err=%d ack=%d rcv=%d\n",
		  dsc, err, ack, rcv);

	cxip_zbcoll_free(zb);
	free(fiaddrs);
}

/*****************************************************************/
/**
 * @brief Perform a simulated broadcast.
 *
 * This exercises the basic broad operation, the user callback, and the
 * non-concurrency lockout. The user callback captures all of the results and
 * ensures they all match the broadcast value.
 *
 * This is done in a single thread, so it tests only a single bcast across
 * multiple addrs. It randomizes the nid processing order, and performs multiple
 * barriers to uncover any ordering issues.
 */
struct bcast_data {
	uint64_t *data;
	int count;
};

static void bcast_func(struct cxip_zbcoll_obj *zb, void *usrptr)
{
	struct bcast_data *data = (struct bcast_data *)usrptr;
	int i;

	for (i = 0; i < zb->simcount; i++)
		data->data[i] = *zb->state[i].dataptr;
	data->count++;
}

/* Test bcast in simulation */
Test(ctrl, zb_broadcast)
{
	struct cxip_ep *cxip_ep;
	struct cxip_ep_obj *ep_obj;
	struct cxip_zbcoll_obj *zb;
	struct bcast_data zbd;
	int i, rep, ret;
	uint64_t data;

	fi_addr_t *fiaddrs;
	int num_addrs = 25;

	cxip_ep = container_of(cxit_ep, struct cxip_ep, ep.fid);
	ep_obj = cxip_ep->ep_obj;

	ret = _generate_sim_addrs(ep_obj, num_addrs, &fiaddrs);
	cr_assert(!ret, "out of memory\n");

	ret = cxip_zbcoll_alloc(ep_obj, num_addrs, fiaddrs, true, &zb);
	cr_assert(ret == 0, "cxip_zbcoll_alloc() = %d\n", ret);
	cr_assert(zb->simcount == num_addrs,
		  "zb->simcount = %d, != %d\n", zb->simcount, num_addrs);
	_addr_shuffle(zb, false);

	/* Acquire a group id */
	ret = cxip_zbcoll_getgroup(zb);
	cr_assert(ret == 0, "getgroup = %d\n", ret);
	ret = _await_complete(zb);
	cr_assert(ret == 0, "getgroup done = %d\n", ret);

	memset(&zbd, 0, sizeof(zbd));
	zbd.data = calloc(num_addrs, sizeof(uint64_t));
	for (rep = 0; rep < 20; rep++) {
		_addr_shuffle(zb, true);
		memset(zbd.data, -1, num_addrs*sizeof(uint64_t));
		/* Perform a broadcast */
		data = (rand() & ((1 << 29) - 1)) | (1 << 28);
		cxip_zbcoll_push_cb(zb, bcast_func, &zbd);
		ret = cxip_zbcoll_broadcast(zb, &data);
		cr_assert(ret == 0, "%d bcast = %s\n",
			  rep, fi_strerror(-ret));
		/* Try again immediately, should fail */
		cxip_zbcoll_push_cb(zb, bcast_func, &zbd);
		ret = cxip_zbcoll_broadcast(zb, &data);
		cr_assert(ret == -FI_EAGAIN, "%d bcast = %s\n",
			  rep, fi_strerror(-ret));
		/* Poll until complete */
		ret = _await_complete(zb);
		cr_assert(ret == FI_SUCCESS, "%d bcast = %s\n",
			  rep, fi_strerror(-ret));
		/* Validate the data */
		for (i = 0; i < num_addrs; i++)
			cr_assert(zbd.data[i] == data, "[%d] %ld != %ld\n",
				  i, zbd.data[i], data);
	}
	cr_assert(zbd.count == rep);

	uint32_t dsc, err, ack, rcv;
	cxip_zbcoll_get_counters(ep_obj, &dsc, &err, &ack, &rcv);
	cr_assert(dsc == 0 && err == 0,
		  "FAILED dsc=%d err=%d ack=%d rcv=%d\n",
		  dsc, err, ack, rcv);

	free(zbd.data);
	cxip_zbcoll_free(zb);
	free(fiaddrs);
}
