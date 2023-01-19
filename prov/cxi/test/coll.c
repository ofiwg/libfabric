/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2017-2019 Intel Corporation. All rights reserved.
 * Copyright (c) 2020-2022 Cray Inc. All rights reserved.
 * Copyright (c) 2021-2023 Hewlett Packard Enterprise Development LP
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHWARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. const NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER const AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS const THE
 * SOFTWARE.
 */

/*
 * NOTE: This is a standalone test that uses the COMM_KEY_RANK model, and thus
 * consists of a single process driving multiple data objects sequentially to
 * simulate network transfers. It can be run under NETSIM, and is part of the
 * standard Jenkins validation integration with Git check-in, allowing this to
 * serve as and automated regression test.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <complex.h>
#include <time.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#include <fenv.h>

#include <ofi.h>

#include "cxip.h"
#include "cxip_test_common.h"

/* see cxit_trace_enable() in each test framework */
#define	TRACE CXIP_TRACE

#define	MIN(a,b) (((a)<(b))?(a):(b))

/***************************************/
/**
 * Sanity tests for proper integration with EP, enable/disable checks.
 */

TestSuite(coll_init, .disabled = false, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test EP close without explicitly enabling collectives.
 */
Test(coll_init, noop)
{
	struct cxip_ep *ep;

	cxit_setup_rma();
	ep = container_of(cxit_ep, struct cxip_ep, ep);

	cr_assert(ep->ep_obj->coll.enabled,
		  "coll not enabled on startup\n");
	cxit_teardown_rma();
}

/* Test EP close after explicitly enabling collectives.
 */
Test(coll_init, enable)
{
	struct cxip_ep *ep;
	int ret;

	cxit_setup_rma();
	ep = container_of(cxit_ep, struct cxip_ep, ep);

	ret = cxip_coll_enable(ep->ep_obj);
	cr_assert(ret == 0, "cxip_coll_enable failed: %d\n", ret);
	cr_assert(ep->ep_obj->coll.enabled,
		  "coll not enabled after enabling\n");
	cxit_teardown_rma();
}

/* Test EP close after disabling collectives.
 */
Test(coll_init, disable)
{
	struct cxip_ep *ep;
	int ret;

	cxit_setup_rma();
	ep = container_of(cxit_ep, struct cxip_ep, ep);

	ret = cxip_coll_enable(ep->ep_obj);
	cr_assert(ret == 0, "cxip_coll_enable failed: %d\n", ret);
	ret = cxip_coll_disable(ep->ep_obj);
	cr_assert(ret == 0, "cxip_coll_disable failed: %d\n", ret);
	cr_assert(!ep->ep_obj->coll.enabled,
		  "coll enabled after disabling\n");
	cxit_teardown_rma();
}

/* Test EP close after disabling/re-enabling collectives.
 */
Test(coll_init, reenable)
{
	struct cxip_ep *ep;
	int ret;

	cxit_setup_rma();
	ep = container_of(cxit_ep, struct cxip_ep, ep);

	ret = cxip_coll_enable(ep->ep_obj);
	cr_assert(ret == 0, "cxip_coll_enable failed: %d\n", ret);
	ret = cxip_coll_disable(ep->ep_obj);
	cr_assert(ret == 0, "cxip_coll_disable failed: %d\n", ret);
	ret = cxip_coll_enable(ep->ep_obj);
	cr_assert(ret == 0, "cxip_coll_enable failed: %d\n", ret);
	cr_assert(ep->ep_obj->coll.enabled,
		  "coll not enabled after enabling\n");
	cxit_teardown_rma();
}

/***************************************/
/**
 * JOIN testing.
 */
TestSuite(coll_join, .init = cxit_setup_rma, .fini = cxit_teardown_rma,
	  .disabled = false, .timeout = CXIT_DEFAULT_TIMEOUT);

/* expand AV and create av_sets for collectives */
static void _create_av_set(int count, int rank, struct fid_av_set **av_set_fid)
{
	struct cxip_ep *ep;
	struct cxip_comm_key comm_key = {
		.keytype = COMM_KEY_RANK,
		.rank.rank = rank,
		.rank.hwroot_idx = 0,
	};
	struct fi_av_set_attr attr = {
		.count = 0,
		.start_addr = FI_ADDR_NOTAVAIL,
		.end_addr = FI_ADDR_NOTAVAIL,
		.stride = 1,
		.comm_key_size = sizeof(comm_key),
		.comm_key = (void *)&comm_key,
		.flags = 0,
	};
	struct cxip_addr caddr;
	int i, ret;

	ep = container_of(cxit_ep, struct cxip_ep, ep);

	/* lookup initiator caddr as set in test framework */
	ret = _cxip_av_lookup(ep->ep_obj->av, cxit_ep_fi_addr, &caddr);
	cr_assert(ret == 0, "bad lookup on address %ld: %d\n",
		  cxit_ep_fi_addr, ret);

	/* create empty av_set */
	ret = fi_av_set(&ep->ep_obj->av->av_fid, &attr, av_set_fid, NULL);
	cr_assert(ret == 0, "av_set creation failed: %d\n", ret);

	/* add source address as multiple av entries */
	for (i = count - 1; i >= 0; i--) {
		fi_addr_t fi_addr;

		ret = fi_av_insert(&ep->ep_obj->av->av_fid, &caddr, 1,
				   &fi_addr, 0, NULL);
		cr_assert(ret == 1, "%d cxip_av_insert failed: %d\n", i, ret);
		ret = fi_av_set_insert(*av_set_fid, fi_addr);
		cr_assert(ret == 0, "%d fi_av_set_insert failed: %d\n", i, ret);
	}
}

void _create_netsim_collective(int count, int exp)
{
	int i, ret;

	TRACE("========================\n%s: entry\n", __func__);
	TRACE("%s: count=%d\n", __func__, count);
	cxit_coll_mc_list.count = count;
	cxit_coll_mc_list.av_set_fid = calloc(cxit_coll_mc_list.count,
					      sizeof(struct fid_av_set *));
	cxit_coll_mc_list.mc_fid = calloc(cxit_coll_mc_list.count,
					  sizeof(struct fid_mc *));

	for (i = 0; i < cxit_coll_mc_list.count; i++) {
		TRACE("%s: ==== create %d\n", __func__, i);
		TRACE("create av_set rank %d\n", i);
		_create_av_set(cxit_coll_mc_list.count, i,
			       &cxit_coll_mc_list.av_set_fid[i]);
		TRACE("join collective\n");
		ret = cxip_join_collective(cxit_ep, FI_ADDR_NOTAVAIL,
					   cxit_coll_mc_list.av_set_fid[i],
					   0, &cxit_coll_mc_list.mc_fid[i],
					   NULL);
		TRACE("ret=%d\n", ret);
		cr_assert(ret == exp,
			  "cxip_coll_enable failed: exp %s saw %s\n",
			  fi_strerror(-exp), fi_strerror(-ret));
	}
	TRACE("%s: exit\n========================\n", __func__);
}

void _destroy_netsim_collective(void)
{
	int i;

	for (i = cxit_coll_mc_list.count - 1; i >= 0; i--) {
		TRACE("closing %d\n", i);
		if (cxit_coll_mc_list.mc_fid[i])
			fi_close(&cxit_coll_mc_list.mc_fid[i]->fid);
		if (cxit_coll_mc_list.av_set_fid[i])
			fi_close(&cxit_coll_mc_list.av_set_fid[i]->fid);
	}
	TRACE("cleanup\n");
	free(cxit_coll_mc_list.mc_fid);
	free(cxit_coll_mc_list.av_set_fid);
	cxit_coll_mc_list.mc_fid = NULL;
	cxit_coll_mc_list.av_set_fid = NULL;
}

static void _wait_for_join(int count, int exp_err)
{
	struct cxip_ep *ep;
	struct fid_cq *txcq, *rxcq;
	struct fid_eq *eq;
	struct fi_cq_err_entry cqd = {};
	struct fi_eq_err_entry eqd = {};
	uint32_t event;
	int ret, err;

	ep = container_of(cxit_ep, struct cxip_ep, ep);
	rxcq = &ep->ep_obj->coll.rx_cq->util_cq.cq_fid;
	txcq = &ep->ep_obj->coll.tx_cq->util_cq.cq_fid;
	eq = &ep->ep_obj->coll.eq->util_eq.eq_fid;

	do {
		sched_yield();
		err = -FI_EINVAL;
		ret = fi_eq_read(eq, &event, &eqd, sizeof(eqd), 0);
		if (ret == -FI_EAVAIL) {
			TRACE("=== error available!\n");
			ret = fi_eq_readerr(eq, &eqd, 0);
			if (ret >= 0) {
				TRACE("  event   = %d\n", event);
				TRACE("  fid     = %p\n", eqd.fid);
				TRACE("  context = %p\n", eqd.context);
				TRACE("  data    = %lx\n", eqd.data);
				TRACE("  err     = %s\n",
					fi_strerror(-eqd.err));
				TRACE("  prov_err= 0x%04x\n", eqd.prov_errno);
				TRACE("  err_data= %p\n", eqd.err_data);
				TRACE("  err_size= %ld\n", eqd.err_data_size);
				TRACE("  readerr = %d\n", ret);
				err = eqd.err;
				event = eqd.data;
			}
			TRACE("===\n");
		} else if (ret >= 0) {
			TRACE("=== EQ SUCCESS!\n");
			err = FI_SUCCESS;
		}
		if (ret != -FI_EAGAIN) {
			if (event == FI_JOIN_COMPLETE) {
				TRACE("FI_JOIN_COMPLETE seen\n");
				count--;
			}
			if (exp_err != err) {
			cr_assert(exp_err == err,
				  "FAILED TEST: exp_err = '%s' saw '%s'\n",
				  fi_strerror(-exp_err), fi_strerror(-err));
				break;
			}
		}

		ret = fi_cq_read(rxcq, &cqd, sizeof(cqd));
		if (ret == -FI_EAVAIL) {
			ret = fi_cq_readerr(rxcq, &cqd, sizeof(cqd));
			break;
		}

		ret = fi_cq_read(txcq, &cqd, sizeof(cqd));
		if (ret == -FI_EAVAIL) {
			ret = fi_cq_readerr(txcq, &cqd, sizeof(cqd));
			break;
		}
	} while (count > 0);
	TRACE("wait done\n");
}

/* Basic test of single NETSIM join.
 */
Test(coll_join, join1)
{
	TRACE("=========================\n");
	TRACE("join1\n");
	_create_netsim_collective(1, FI_SUCCESS);
	_wait_for_join(1, FI_SUCCESS);
	_destroy_netsim_collective();
}

/* Basic test of two NETSIM joins.
 */
Test(coll_join, join2)
{
	TRACE("=========================\n");
	TRACE("join2\n");
	_create_netsim_collective(2, FI_SUCCESS);
	_wait_for_join(2, FI_SUCCESS);
	_destroy_netsim_collective();
}

/* Basic test of three NETSIM joins.
 */
Test(coll_join, join3)
{
	TRACE("=========================\n");
	TRACE("join3\n");
	_create_netsim_collective(3, FI_SUCCESS);
	_wait_for_join(3, FI_SUCCESS);
	_destroy_netsim_collective();
}

/* Basic test of maximum NETSIM joins.
 */
Test(coll_join, join32)
{
	TRACE("=========================\n");
	TRACE("join32\n");
	_create_netsim_collective(32, FI_SUCCESS);
	_wait_for_join(32, FI_SUCCESS);
	_destroy_netsim_collective();
}

/***************************************/
/**
 * Basic send/receive testing.
 */

TestSuite(coll_put, .init = cxit_setup_rma, .fini = cxit_teardown_rma,
	  .disabled = false, .timeout = CXIT_DEFAULT_TIMEOUT);

/* 50-byte packet */
struct fakebuf {
	uint64_t count[6];
	uint16_t pad;
} __attribute__((packed));

/* Progression is needed because the test runs in a single execution thread with
 * NETSIM. This waits for completion of PROGRESS_COUNT messages on the simulated
 * (loopback) target. It needs to be called periodically during the test run, or
 * the netsim resources run out and this gets blocked.
 */
#define	PROGRESS_COUNT	10
void _progress_put(struct cxip_cq *cq, int sendcnt, uint64_t *dataval)
{
	struct fi_cq_tagged_entry entry[PROGRESS_COUNT];
	struct fi_cq_err_entry err;
	int i, ret;

	while (sendcnt > 0) {
		do {
			int cnt = MIN(PROGRESS_COUNT, sendcnt);
			sched_yield();
			ret = fi_cq_read(&cq->util_cq.cq_fid, entry, cnt);
		} while (ret == -FI_EAGAIN);
		if (ret == -FI_EAVAIL) {
			ret = fi_cq_readerr(&cq->util_cq.cq_fid, &err, 0);
			memcpy(&entry[0], &err, sizeof(entry[0]));
		}
		for (i = 0; i < ret; i++) {
			struct fakebuf *fb = entry[i].buf;
			cr_assert(entry[i].len == sizeof(*fb),
				  "fb->len exp %ld, saw %ld\n",
				  sizeof(*fb), entry[i].len);
			cr_assert(fb->count[0] == *dataval,
				  "fb->count[0] exp %ld, saw %ld\n",
				  fb->count[0], *dataval);
			cr_assert(fb->count[5] == *dataval,
				  "fb->count[5] exp %ld, saw %ld\n",
				  fb->count[5], *dataval);
			cr_assert(fb->pad == (uint16_t)*dataval,
				  "fb_pad exp %x, saw %x\n",
				  fb->pad, (uint16_t)*dataval);
			(*dataval)++;
		}
		sendcnt -= ret;
	}
}

/* Put count packets, and verify them. This sends count packets from one
 * NETSIM multicast resource to another.
 */
void _put_data(int count, int from_rank, int to_rank)
{
	struct cxip_coll_mc *mc_obj_send, *mc_obj_recv;
	struct cxip_coll_reduction *reduction;
	struct cxip_ep *ep;
	struct fakebuf *buf;
	void *buffers;
	int sendcnt, cnt;
	uint64_t dataval;
	int i, j, ret;

	ep = container_of(cxit_ep, struct cxip_ep, ep);

	/* from and to (may be the same mc_obj) */
	mc_obj_send = container_of(cxit_coll_mc_list.mc_fid[from_rank],
				 struct cxip_coll_mc, mc_fid);
	mc_obj_recv = container_of(cxit_coll_mc_list.mc_fid[to_rank],
				 struct cxip_coll_mc, mc_fid);

	TRACE("%s: mc_obj_send = %p\n", __func__, mc_obj_send);
	TRACE("%s: mc_obj_recv = %p\n", __func__, mc_obj_recv);

	/* clear any prior values */
	TRACE("%s: reset mc_ctrs\n", __func__);
	cxip_coll_reset_mc_ctrs(&mc_obj_send->mc_fid);
	cxip_coll_reset_mc_ctrs(&mc_obj_recv->mc_fid);

	/* from_rank reduction */
	reduction = &mc_obj_send->reduction[0];

	/* must persist until _progress called, for validation */
	buffers = calloc(PROGRESS_COUNT, sizeof(*buf));

	buf = buffers;
	sendcnt = 0;
	dataval = 0;
	TRACE("%s: iteration over %p\n", __func__, buf);
	for (i = 0; i < count; i++) {
		for (j = 0; j < 6; j++)
			buf->count[j] = i;
		buf->pad = i;
		TRACE("call cxip_coll_send()\n");
		ret = cxip_coll_send(reduction, to_rank, buf, sizeof(*buf),
				     NULL);
		cr_assert(ret == 0, "cxip_coll_send failed: %d\n", ret);

		buf++;
		sendcnt++;
		if (sendcnt >= PROGRESS_COUNT) {
			_progress_put(ep->ep_obj->coll.rx_cq, sendcnt,
				      &dataval);
			buf = buffers;
			sendcnt = 0;
		}
	}
	TRACE("call _progress_put\n");
	_progress_put(ep->ep_obj->coll.rx_cq, sendcnt, &dataval);

	/* check final counts */
	TRACE("check counts\n");
	if (count * sizeof(*buf) >
	    ep->ep_obj->coll.buffer_size - ep->ep_obj->rxc.min_multi_recv) {
		cnt = ofi_atomic_get32(&mc_obj_recv->coll_pte->buf_swap_cnt);
		cr_assert(cnt > 0, "Did not recirculate buffers\n");
	}

	TRACE("check atomic counts\n");
	cnt = ofi_atomic_get32(&mc_obj_send->send_cnt);
	cr_assert(cnt == count,
		  "Expected mc_obj[%d] send_cnt == %d, saw %d",
		  from_rank, count, cnt);

	cnt = ofi_atomic_get32(&mc_obj_recv->recv_cnt);
	cr_assert(cnt == count,
		  "Expected mc_obj[%d]->[%d] recv_cnt == %d, saw %d",
		  from_rank, to_rank, count, cnt);
	cnt = ofi_atomic_get32(&mc_obj_recv->pkt_cnt);
	cr_assert(cnt == 0,
		  "Expected mc_obj[%d]->[%d] pkt_cnt == %d, saw %d",
		  from_rank, to_rank, 0, cnt);

	TRACE("free buffers\n");
	free(buffers);
}

/* Attempt to send from rank 0 to rank 3 (does not exist).
 */
Test(coll_put, put_bad_rank)
{
	struct cxip_coll_mc *mc_obj;
	struct cxip_coll_reduction *reduction;
	struct fakebuf buf;
	int ret;

	_create_netsim_collective(2, FI_SUCCESS);
	_wait_for_join(2, FI_SUCCESS);

	mc_obj = container_of(cxit_coll_mc_list.mc_fid[0],
			      struct cxip_coll_mc, mc_fid);
	reduction = &mc_obj->reduction[0];

	ret = cxip_coll_send(reduction, 3, &buf, sizeof(buf), NULL);
	cr_assert(ret == -FI_EINVAL, "cxip_coll_set bad error = %d\n", ret);

	_destroy_netsim_collective();
}

/* Basic test with one packet from rank 0 to rank 0.
 */
Test(coll_put, put_one)
{
	_create_netsim_collective(1, FI_SUCCESS);
	_wait_for_join(1, FI_SUCCESS);
	_put_data(1, 0, 0);
	_destroy_netsim_collective();
}

/* Basic test with one packet from each rank to another rank.
 * Exercises NETSIM rank-based target addressing.
 */
Test(coll_put, put_ranks)
{
	_create_netsim_collective(2, FI_SUCCESS);
	_wait_for_join(2, FI_SUCCESS);
	TRACE("call _put_data()\n");
	_put_data(1, 0, 0);
	_put_data(1, 0, 1);
	_put_data(1, 1, 0);
	_put_data(1, 1, 1);
	_destroy_netsim_collective();
}

/* Test a lot of packets to force buffer rollover.
 */
Test(coll_put, put_many)
{
	_create_netsim_collective(1, FI_SUCCESS);
	_wait_for_join(1, FI_SUCCESS);
	_put_data(4000, 0, 0);
	_destroy_netsim_collective();
}

/* Progress the reduction packet send.
 */
void _progress_red_pkt(struct cxip_cq *cq, int sendcnt, uint64_t *dataval)
{
	struct fi_cq_tagged_entry entry[PROGRESS_COUNT];
	struct fi_cq_err_entry err;
	int i, ret;

	while (sendcnt > 0) {
		do {
			int cnt = MIN(PROGRESS_COUNT, sendcnt);
			sched_yield();
			ret = fi_cq_read(&cq->util_cq.cq_fid, entry, cnt);
		} while (ret == -FI_EAGAIN);
		if (ret == -FI_EAVAIL) {
			ret = fi_cq_readerr(&cq->util_cq.cq_fid, &err, 0);
			memcpy(&entry[0], &err, sizeof(entry[0]));
		}
		for (i = 0; i < ret; i++)
			(*dataval)++;
		sendcnt -= ret;
	}
}

/* Test red_pkt sends. With only one node, root sends to self.
 */
void _put_red_pkt(int count)
{
	struct cxip_coll_mc *mc_obj;
	struct cxip_coll_reduction *reduction;
	struct cxip_coll_data coll_data = {.red_cnt = 1};
	int sendcnt, cnt;
	uint64_t dataval;
	int i, ret;

	_create_netsim_collective(1, FI_SUCCESS);
	_wait_for_join(1, FI_SUCCESS);

	mc_obj = container_of(cxit_coll_mc_list.mc_fid[0],
			      struct cxip_coll_mc, mc_fid);

	/* clear counters */
	cxip_coll_reset_mc_ctrs(&mc_obj->mc_fid);

	sendcnt = 0;
	dataval = 0;
	coll_data.intval.ival[0] = dataval;
	reduction = &mc_obj->reduction[0];
	reduction->coll_state = CXIP_COLL_STATE_NONE;
	for (i = 0; i < count; i++) {
		ret = cxip_coll_send_red_pkt(reduction, &coll_data,
					     false, false);
		cr_assert(ret == FI_SUCCESS,
			  "Packet send from root failed: %d\n", ret);

		sendcnt++;
		if (sendcnt >= PROGRESS_COUNT) {
			_progress_red_pkt(mc_obj->ep_obj->coll.rx_cq, sendcnt,
					  &dataval);
			sendcnt = 0;
		}
	}
	_progress_red_pkt(mc_obj->ep_obj->coll.rx_cq, sendcnt, &dataval);

	cnt = ofi_atomic_get32(&mc_obj->send_cnt);
	cr_assert(cnt == count, "Bad send counter on root: %d, exp %d\n", cnt, count);
	cnt = ofi_atomic_get32(&mc_obj->recv_cnt);
	cr_assert(cnt == count, "Bad recv counter on root: %d, exp %d\n", cnt, count);
	cnt = ofi_atomic_get32(&mc_obj->pkt_cnt);
	cr_assert(cnt == count, "Bad pkt counter on root: %d, exp %d\n", cnt, count);

	_destroy_netsim_collective();
}

/* Test of a single red_pkt from root to root.
 */
Test(coll_put, put_red_pkt_one)
{
	_put_red_pkt(1);
}

/* Test of a many red_pkts from root to root.
 */
Test(coll_put, put_red_pkt_many)
{
	_put_red_pkt(4000);
}

/* Test of the reduction packet code distribution under NETSIM.
 * Exercises distribution root->leaves, leaves->root, single packet.
 */
Test(coll_put, put_red_pkt_distrib)
{
	struct cxip_coll_mc *mc_obj[5];
	struct cxip_cq *rx_cq;
	struct cxip_coll_reduction *reduction;
	struct cxip_coll_data coll_data = {.red_cnt = 1};
	struct fi_cq_data_entry entry;
	int i, cnt, ret;

	_create_netsim_collective(5, FI_SUCCESS);
	_wait_for_join(5, FI_SUCCESS);

	for (i = 0; i < 5; i++) {
		mc_obj[i] = container_of(cxit_coll_mc_list.mc_fid[i],
					 struct cxip_coll_mc, mc_fid);
		mc_obj[i]->reduction[0].coll_state = CXIP_COLL_STATE_NONE;
		cxip_coll_reset_mc_ctrs(&mc_obj[i]->mc_fid);
	}

	rx_cq = mc_obj[0]->ep_obj->coll.rx_cq;

	coll_data.intval.ival[0] = 0;
	reduction = &mc_obj[0]->reduction[0];
	ret = cxip_coll_send_red_pkt(reduction, &coll_data,
				     false, false);
	cr_assert(ret == FI_SUCCESS,
		  "Packet send from root failed: %d\n", ret);
	cnt = ofi_atomic_get32(&mc_obj[0]->send_cnt);
	cr_assert(cnt == 4, "Bad send counter on root: %d\n", cnt);
	for (i = 1; i < 5; i++) {
		do {
			sched_yield();
			ret = fi_cq_read(&rx_cq->util_cq.cq_fid, &entry, 1);
		} while (ret == -FI_EAGAIN);
		cr_assert(ret == 1, "Bad CQ response[%d]: %d\n", i, ret);
		cnt = ofi_atomic_get32(&mc_obj[i]->recv_cnt);
		cr_assert(cnt == 1,
			  "Bad recv counter on leaf[%d]: %d\n", i, cnt);
	}

	/* Send data from leaf (!0) to root */
	for (i = 0; i < 5; i++)
		cxip_coll_reset_mc_ctrs(&mc_obj[i]->mc_fid);
	for (i = 1; i < 5; i++) {
		coll_data.intval.ival[0] = i;
		reduction = &mc_obj[i]->reduction[0];
		ret = cxip_coll_send_red_pkt(reduction, &coll_data,
					     false, false);
		cr_assert(ret == FI_SUCCESS,
			  "Packet send from leaf[%d] failed: %d\n", i, ret);
		cnt = ofi_atomic_get32(&mc_obj[i]->send_cnt);
		cr_assert(cnt == 1,
			  "Bad send counter on leaf[%d]: %d\n", i, cnt);
		do {
			sched_yield();
			ret = fi_cq_read(&rx_cq->util_cq.cq_fid, &entry, 1);
		} while (ret == -FI_EAGAIN);
		cr_assert(ret == 1, "Bad CQ response[%d]: %d\n", i, ret);
	}

	cnt = ofi_atomic_get32(&mc_obj[0]->recv_cnt);
	cr_assert(cnt == 4,
		  "Bad recv counter on root: %d\n", cnt);

	_destroy_netsim_collective();
}

/***************************************/
/**
 * Test reduction concurrency.
 */
TestSuite(coll_reduce, .init = cxit_setup_rma, .fini = cxit_teardown_rma,
	  .disabled = false, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Simulated user context, specifically to return error codes */
struct user_context {
	struct dlist_entry entry;
	int node;		// reduction simulated node (MC object)
	int seqno;		// reduction sequence number
	int red_id;		// reduction ID
	int errcode;		// reduction error code
	int hw_rc;		// reduction hardware failure code
	uint64_t expval;	// expected reduction value
};

static struct dlist_entry done_list;
static int dlist_initialized;
static int max_queue_depth;
static int queue_depth;

/* Wrapper for fi_allreduce() (injection), returns the reduction ID used. */
ssize_t _fi_allreduce(struct fid_ep *ep, const void *buf, size_t count,
		      void *desc, void *result, void *result_desc,
		      fi_addr_t coll_addr, enum fi_datatype datatype,
		      enum fi_op op, uint64_t flags, void *context,
		      int *reduction_id)
{
	struct cxip_ep *cxi_ep;
	struct cxip_coll_mc *mc_obj;
	int cxi_opcode;
	int ret;

	cxi_ep = container_of(ep, struct cxip_ep, ep.fid);
	mc_obj = (struct cxip_coll_mc *) ((uintptr_t) coll_addr);

	if (mc_obj->ep_obj != cxi_ep->ep_obj)
		return -FI_EINVAL;

	cxi_opcode = cxip_fi2cxi_opcode(op, datatype);
	if (cxi_opcode < 0)
		return cxi_opcode;

	ret = cxip_coll_inject(mc_obj, cxi_opcode, buf, result,
			       count, flags, context, reduction_id);

	/* Event queue should be one deeper */
	if (ret != -FI_EAGAIN && max_queue_depth < ++queue_depth)
		max_queue_depth = queue_depth;

	return ret;
}

/**
 * @brief Progress state machine and record completions.
 *
 * If context == NULL, this polls once to advance the state and returns.
 *
 * If context != NULL, this polls until that context is seen.
 *
 * Any context seen that isn't what we were looking for goes on the queue. A
 * subsequent call with context != NULL will search the queue first.
 *
 * This returns once for every TX completion event, and polls the RX CQ only to
 * keep it empty. This test is not interested in the reduction result.
 *
 * @param rx_cq_fid - RX completion queue
 * @param tx_cq_fid - TX completion queue
 * @param context - context to wait for, or NULL
 */
static void _allreduce_wait(struct fid_cq *rx_cq_fid, struct fid_cq *tx_cq_fid,
			    struct user_context *context)
{
	struct dlist_entry *done;
	struct fi_cq_data_entry entry;
	struct fi_cq_err_entry err_entry;
	struct user_context *ctx;
	int ret;

	/* initialize the static locals on first use */
	if (! dlist_initialized) {
		dlist_init(&done_list);
		dlist_initialized = 1;
	}

	/* search for prior detection of context */
	dlist_foreach(&done_list, done) {
		if ((void *)context == (void *)done) {
			dlist_remove(done);
			return;
		}
	}

	do {
		/* Wait for a tx CQ completion event */
		do {
			sched_yield();
			/* read the receive queue and discard */
			ret = fi_cq_read(rx_cq_fid, &entry, 1);
			/* read tx CQ to see a single completion event */
			ret = fi_cq_read(tx_cq_fid, &entry, 1);
			if (!(ret == -FI_EAGAIN && context))
				break;
		} while (true);

		ctx = NULL;
		if (ret == -FI_EAVAIL) {
			/* tx CQ posted an error, copy to user context */
			ret = fi_cq_readerr(tx_cq_fid, &err_entry, 1);
			cr_assert(ret == 1, "fi_cq_readerr failed: %d\n", ret);
			ctx = err_entry.op_context;
			ctx->errcode = err_entry.err;
			ctx->hw_rc = err_entry.prov_errno;
			cr_assert(err_entry.err != 0,
				  "Failure with good return\n");
			queue_depth--;
		} else if (ret == 1) {
			/* tx CQ posted a normal completion */
			ctx = entry.op_context;
			ctx->errcode = 0;
			ctx->hw_rc = 0;
			queue_depth--;
		} else {
			/* We should only see a 'no-event' error */
			cr_assert(ret == -FI_EAGAIN, "Improper return %d\n", 		  ret);
		}

		/* context we are looking for, NULL matches no-event */
		if (ctx == context)
			return;

		/* if we did see a ctx == context, record it  */
		if (ctx)
			dlist_insert_tail(&ctx->entry, &done_list);

	} while (context);
}

/* extract and verify endpoint across NETSIM collective group */
void _get_endpoint(const char *label, int nodes,
		struct cxip_coll_mc **mc_obj,
		struct cxip_ep_obj **ep_obj,
		struct fid_cq **rx_cq_fid,
		struct fid_cq **tx_cq_fid)
{
	int node;

	/* scan mc_fid[], convert to mc_obj[], and extract ep_obj pointer */
	*ep_obj = NULL;
	for (node = 0; node < nodes; node++) {
		mc_obj[node] = container_of(cxit_coll_mc_list.mc_fid[node],
					     struct cxip_coll_mc, mc_fid);
		if (!*ep_obj)
			*ep_obj = mc_obj[node]->ep_obj;
		cr_assert(mc_obj[node]->ep_obj == *ep_obj,
			  "%s Mismatched endpoints\n", label);
	}
	cr_assert(*ep_obj != NULL,
		  "%s Did not find an endpoint object\n", label);
	/* extract rx and tx cq fids */
	*rx_cq_fid = &(*ep_obj)->coll.rx_cq->util_cq.cq_fid;
	*tx_cq_fid = &(*ep_obj)->coll.tx_cq->util_cq.cq_fid;
}

/**
 * @brief Exercise the collective state machine.
 *
 * This is a single-threaded test, intended for use with NETSIM.
 *
 * We initiate the collective in sequence, beginning with 'start_node', and
 * wrapping around. If start_node is zero, the root node initiates first,
 * otherwise a leaf node initiates first.
 *
 * We perform 'concur' reductions concurrently. When we hit the maximum of
 * concurrent injections, the reduction attempt should return -FI_EAGAIN. When
 * this happens, we poll to see if a completion has occurred, then try again.
 * Since we don't know the order of completions, we wait for ANY completion,
 * which is then saved in a queue. We can then (later) look for a specific
 * completion, which searches the queue before waiting for new completions.
 *
 * We inject an error by specifying a 'bad' node in the range of nodes. If
 * bad_node is outside the range (e.g. -1), no errors will be injected. The
 * error injection is done by choosing to send the wrong reduction operation
 * code for the bad node, which causes the entire reduction to fail.
 *
 * We perform 'concur' reductions to exercise the round-robin reduction ID
 * handling and blocking. This should be tested for values > 8.
 *
 * We generate different results for each concurrent reduction, to ensure that
 * there is no mixing of the packets in each reduction channel.
 *
 * @param start_node - node (rank) to start the reduction
 * @param bad_node - node to inject a bad reduction, or -1 to succeed
 * @param concur - number of reductions to start before polling
 */
void _allreduce(int start_node, int bad_node, int concur)
{
	struct cxip_ep_obj *ep_obj;
	struct cxip_coll_mc **mc_obj;
	struct user_context **context;
	struct cxip_intval **rslt;
	struct cxip_intval *data;
	struct fid_cq *rx_cq_fid, *tx_cq_fid;
	int nodes, first, last, base;
	char label[128];
	uint64_t result;
	ssize_t size;
	int i, node, ret;

	TRACE("\n===== %s rank=%d bad=%d concur=%d\n",
		__func__, start_node, bad_node, concur);
	concur = MAX(concur, 1);
	nodes = cxit_coll_mc_list.count;
	context = calloc(nodes, sizeof(**context));
	mc_obj = calloc(nodes, sizeof(**mc_obj));
	rslt = calloc(nodes, sizeof(**rslt));
	data = calloc(nodes, sizeof(*data));
	start_node %= nodes;
	snprintf(label, sizeof(label), "{%2d,%2d,%2d}",
		 start_node, bad_node, concur);

	_get_endpoint(label, nodes, mc_obj, &ep_obj, &rx_cq_fid, &tx_cq_fid);
	for (node = 0; node < nodes; node++) {
		context[node] = calloc(concur, sizeof(struct user_context));
		rslt[node] = calloc(concur, sizeof(struct cxip_intval));
	}

	/* Inject all of the collectives */
	first = 0;
	last = 0;
	base = 1;
	result = 0;

	/* last advances from 0 to concur */
	while (last < concur) {
		uint64_t undone = (1 << nodes) - 1;

		/* use different values on each concurrency */
		base <<= 1;
		if (base > 16)
			base = 1;

		/* FI_EAGAIN results will force reordering */
		result = 0;
		while (undone) {
			/* Polls once if we have free reduction IDs */
			_allreduce_wait(rx_cq_fid, tx_cq_fid, NULL);
			/* Initiates a single BAND reduction across the nodes */
			for (i = 0; i < nodes; i++) {
				enum fi_op op;
				uint64_t mask;
				int red_id;

				node = (start_node + i) % nodes;
				mask = 1LL << node;
				op = (node == bad_node) ? FI_BAND : FI_BOR;

				/* Don't repeat nodes that succeeded */
				if (! (mask & undone))
					continue;

				/* Each node contributes a bit */
				data[node].ival[0] = (base << node);
				result |= data[node].ival[0];
				context[node][last].node = node;
				context[node][last].seqno = last;

				size = _fi_allreduce(cxit_ep,
					&data[node], 1, NULL,
					&rslt[node][last], NULL,
					(fi_addr_t)mc_obj[node],
					FI_UINT64, op, 0,
					&context[node][last],
					&red_id);
				if (size == -FI_EAGAIN)
					continue;

				/* Completed this one */
				undone &= ~mask;

				/* record reduction id used */
				context[node][last].red_id = red_id;
			}
		}

		/* record the final expected result */
		for (node = 0; node < nodes; node++)
			context[node][last].expval = result;

		/* Ensure these all used the same reduction ID */
		ret = 0;
		for (node = 1; node < nodes; node++)
			if (context[0][last].red_id !=
			    context[node][last].red_id)
				ret = -1;
		if (ret)
			cr_assert(true, "%s reduction ID mismatch\n", label);

		last++;
	}

	/* Wait for all reductions to complete */
	while (first < last) {
		struct user_context *ctx;
		int red_id0, fi_err0, rc_err0;
		uint64_t expval, actval;

		/* If there was a bad node, all reductions should fail */
		rc_err0 = (bad_node < 0) ? 0 : CXIP_COLL_RC_OP_MISMATCH;
		for (node = 0; node < nodes; node++) {
			_allreduce_wait(rx_cq_fid, tx_cq_fid, &context[node][first]);
			ctx = &context[node][first];

			/* Use the root values as definitive */
			if (node == 0) {
				red_id0 = ctx->red_id;
				fi_err0 = ctx->errcode;
				expval = ctx->expval;
			}
			actval = rslt[node][first].ival[0];

			/* Test values */
			if (ctx->node != node ||
			    ctx->seqno != first  ||
			    ctx->red_id != red_id0 ||
			    ctx->errcode != fi_err0 ||
			    ctx->hw_rc != rc_err0 ||
			    (!fi_err0 && expval != actval)) {
				TRACE("%s =====\n", label);
				TRACE("  node    %3d, exp %3d\n",
				       ctx->node, node);
				TRACE("  seqno   %3d, exp %3d\n",
				       ctx->seqno, first);
				TRACE("  red_id  %3d, exp %3d\n",
				       ctx->red_id, red_id0);
				TRACE("  errcode %3d, exp %3d\n",
				       ctx->errcode, fi_err0);
				TRACE("  hw_rc   %3d, exp %3d\n",
				       ctx->hw_rc, rc_err0);
				TRACE("  value   %08lx, exp %08lx\n",
				       actval, expval);
				cr_assert(true, "%s context failure\n",
					  label);
			}
		}

		first++;
	}
	for (node = 0; node < nodes; node++) {
		TRACE("tmout[%d] = %d\n", node,
		    ofi_atomic_get32(&mc_obj[node]->tmout_cnt));
	}

	/* make sure we got them all */
	cr_assert(dlist_empty(&done_list), "Pending contexts\n");
	cr_assert(queue_depth == 0, "queue_depth = %d\n", queue_depth);
	TRACE("completed\n");

	for (node = 0; node < nodes; node++) {
		free(rslt[node]);
		free(context[node]);
	}
	free(context);
	free(rslt);
	free(data);
	free(mc_obj);
}

void _reduce_test_set(int concur)
{
	_create_netsim_collective(5, FI_SUCCESS);
	TRACE("========================\n%s with %d concurrencies\n",
	    __func__, concur);
	_wait_for_join(5, FI_SUCCESS);
	/* success with each of the nodes starting */
	_allreduce(0, -1, concur);
	_allreduce(1, -1, concur);
	_allreduce(2, -1, concur);
	_allreduce(3, -1, concur);
	_allreduce(4, -1, concur);
	_allreduce(0, -1, concur);
	_allreduce(1, -1, concur);
	_allreduce(2, -1, concur);
	_allreduce(3, -1, concur);
	/* failure with root starting */
	_allreduce(0, 0, concur);
	_allreduce(0, 1, concur);
	/* failure with leaf starting */
	_allreduce(1, 0, concur);
	_allreduce(1, 1, concur);
	_destroy_netsim_collective();
}

Test(coll_reduce, concur1)
{
	_reduce_test_set(1);
}

Test(coll_reduce, concur2)
{
	_reduce_test_set(2);
}

Test(coll_reduce, concur8)
{
	_reduce_test_set(8);
}

Test(coll_reduce, concurN)
{
	_reduce_test_set(29);
}

/***************************************/
/* Collective operation testing */
#define	REDUCE_NODES	10

void _setup_reduce(void)
{
	cxit_setup_rma();
	_create_netsim_collective(REDUCE_NODES, FI_SUCCESS);
	_wait_for_join(REDUCE_NODES, FI_SUCCESS);
}

void _teardown_reduce(void) {
	_destroy_netsim_collective();
	cxit_teardown_rma();
}

TestSuite(coll_reduce_ops, .init = _setup_reduce, .fini = _teardown_reduce,
	  .disabled = false, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Perform reduction operation with data, wait for result */
int _allreduceop(int opcode, enum fi_datatype typ, uint64_t flags,
		 int width, void *data, void *rslt,
		 struct user_context *context)
{
	struct cxip_ep_obj *ep_obj;
	struct cxip_coll_mc **mc_obj;
	struct fid_cq *rx_cq_fid, *tx_cq_fid;
	int nodes, node, ret;
	ssize_t size;

	nodes = cxit_coll_mc_list.count;
	mc_obj = calloc(nodes, sizeof(**mc_obj));
	_get_endpoint("reduce", nodes, mc_obj, &ep_obj, &rx_cq_fid, &tx_cq_fid);

	/* 'parallel' injection across nodes */
	ret = 0;
	for (node = 0; node < nodes; node++) {
		size = _fi_allreduce(cxit_ep,
			(char *)data + (node*width), width, NULL,
			(char *)rslt + (node*width), NULL,
			(fi_addr_t)mc_obj[node],
			typ, opcode, 0,
			&context[node], NULL);
			if (size != FI_SUCCESS) {
				printf("%s _fi_allreduce()[%d]=%ld\n",
					__func__, node, size);
				ret = 1;
				goto done;
			}
 	}

	/* 'parallel' wait for all to complete */
	for (node = 0; node < nodes; node++)
		_allreduce_wait(rx_cq_fid, tx_cq_fid, &context[node]);

done:
	free(mc_obj);
	return ret;
}

/* Signaling NaN generation, for testing.
 * Linux feature requires GNU_SOURCE.
 * This generates a specific sNaN value.
 */
static inline double _snan64(void)
{
	return _bits2dbl(0x7ff4000000000000);
}

/* Returns true if this is a signalling NAN */
static inline bool _is_snan64(double d)
{
	/* This detection is universal IEEE */
	return isnan(d) && !(_dbl2bits(d) & 0x0008000000000000);
}

/* Converts a signalling NAN to a non-signalling NAN */
static void _quiesce_nan(double *d)
{
	if (isnan(*d))
		*d = NAN;
}

/* random generation for doubles */
static inline double _frand(double range)
{
	return ((double)rand()/(double)RAND_MAX) * range;
}

/* float equality measure, accommodates snan */
static inline bool _feq(double a, double b)
{
	if (_is_snan64(a) && _is_snan64(b))
		return true;
	if (_is_snan64(a) || _is_snan64(b))
		return false;
	if (isnan(a) && isnan(b))
		return true;
	if (isnan(a) || isnan(b))
		return false;
	return (a == b);
}

/* returns true if a is preferred, false if b is preferred.
 * preference is determined by prefer_nan and prefer_min.
 * if (a==b), a is preferred.
 */
static inline bool _fcmp(double a, double b, bool prefer_min, bool prefer_nan)
{
	if (prefer_nan) {
		/* leftmost snan places first */
		if (_is_snan64(a))
			return false;
		/* rightmost snan places second */
		if (_is_snan64(b))
			return true;
		/* leftmost nan places third */
		if (isnan(a))
			return false;
		/* rightmost nan places last */
		if (isnan(b))
			return true;
	}
	/* right argument is nan, give preference to left (possibly nan) */
	if (isnan(b))
		return false;
	/* left argument is nan and right argument is not, use right */
	if (isnan(a))
		return true;
	/* neither argument is nan, return left or right by preference */
	return (a > b) ? prefer_min : !prefer_min;
}

/* Sanity test for the above */
Test(coll_reduce_ops, fcmp)
{
	cr_assert(!_fcmp(1.0, 2.0, true, true));
	cr_assert( _fcmp(1.0, 2.0, false, true));
	cr_assert(!_fcmp(1.0, 2.0, true, false));
	cr_assert( _fcmp(1.0, 2.0, false, false));
	cr_assert( _fcmp(2.0, NAN, true, true));
	cr_assert( _fcmp(2.0, NAN, false, true));
	cr_assert(!_fcmp(2.0, NAN, true, false));
	cr_assert(!_fcmp(2.0, NAN, false, false));
	cr_assert(!_fcmp(NAN, NAN, true, true));
	cr_assert(!_fcmp(NAN, NAN, false, true));
	cr_assert(!_fcmp(NAN, NAN, true, false));
	cr_assert(!_fcmp(NAN, NAN, false, false));
	cr_assert( _fcmp(2.0, _snan64(), true, true));
	cr_assert( _fcmp(2.0, _snan64(), false, true));
	cr_assert(!_fcmp(2.0, _snan64(), true, false));
	cr_assert(!_fcmp(2.0, _snan64(), false, false));
	cr_assert( _fcmp(NAN, _snan64(), true, true));
	cr_assert( _fcmp(NAN, _snan64(), false, true));
	cr_assert(!_fcmp(NAN, _snan64(), true, false));
	cr_assert(!_fcmp(NAN, _snan64(), false, false));
	cr_assert(!_fcmp(_snan64(), _snan64(), true, true));
	cr_assert(!_fcmp(_snan64(), _snan64(), false, true));
	cr_assert(!_fcmp(_snan64(), _snan64(), true, false));
	cr_assert(!_fcmp(_snan64(), _snan64(), false, false));
}

/* finds MIN(a, b) with two NAN models */
static inline double _fmin(double a, double b, bool prefer_nan)
{
	return (!_fcmp(a, b, true, prefer_nan)) ? a : b;
}

/* finds MAX(a, b) with two NAN models */
static inline double _fmax(double a, double b, bool prefer_nan)
{
	return (!_fcmp(a, b, false, prefer_nan)) ? a : b;
}

/* Prediction of results takes into account the two NAN models and accounts
 * for the distinction between NAN and sNAN. After collective processing, the
 * sNAN will be quiesced, so after accounting for its effect, we need to
 * quiesce it here for comparison.
 */

/* computes fmin result */
static void _predict_fmin(int nodes, struct cxip_fltval *data,
			struct cxip_fltval *check, bool prefer_nan)
{
	int i, j;

	memcpy(check, &data[0], sizeof(*check));
	for (i = 1; i < nodes; i++)
		for (j = 0; j < 4; j++)
			check->fval[j] =
				_fmin(data[i].fval[j], check->fval[j],
					prefer_nan);
	for (i = 0; i < nodes; i++)
		for (j = 0; j < 4; j++)
			_quiesce_nan(&check->fval[j]);
}

/* computes fmax result */
static void _predict_fmax(int nodes, struct cxip_fltval *data,
			struct cxip_fltval *check, bool prefer_nan)
{
	int i, j;

	memcpy(check, &data[0], sizeof(*check));
	for (i = 1; i < nodes; i++)
		for (j = 0; j < 4; j++)
			check->fval[j] =
				_fmax(data[i].fval[j], check->fval[j],
					prefer_nan);
	for (i = 0; i < nodes; i++)
		for (j = 0; j < 4; j++)
			_quiesce_nan(&check->fval[j]);
}

/* computes minmax result */
static void _predict_fminmax(int nodes, struct cxip_fltminmax *data,
				struct cxip_fltminmax *check, bool prefer_nan)
{
	double a, b;
	int i;

	memcpy(check, &data[0], sizeof(*check));
	for (i = 1; i < nodes; i++) {
		a = data[i].fminval;
		b = check->fminval;
		if (_feq(a, b)) {
			/* if equal, choose lowest index */
			if (data[i].fminidx < check->fminidx)
				check->fminidx = data[i].fminidx;
		} else if (!_fcmp(a, b, true, prefer_nan)) {
			check->fminval = a;
			check->fminidx = i;
		}
		a = data[i].fmaxval;
		b = check->fmaxval;
		if (_feq(a, b)) {
			/* if equal, choose lowest index */
			if (data[i].fmaxidx < check->fmaxidx)
				check->fmaxidx = data[i].fmaxidx;
		} else if (!_fcmp(a, b, false, prefer_nan)) {
			check->fmaxval = a;
			check->fmaxidx = i;
		}
	}
	for (i = 0; i < nodes; i++) {
		_quiesce_nan(&check->fminval);
		_quiesce_nan(&check->fmaxval);
	}
}

/* Routines to dump error messages on failure */
static int _dump_ival(int nodes, int i0, int j0,
		      struct cxip_intval *rslt,
		      struct cxip_intval *check)
{
	int i, j;

	for (i = 0; i < nodes; i++)
		for (j = 0; j < 4; j++)
			printf("[%2d][%2d] rslt=%016lx expect=%016lx%s\n",
				i, j, rslt[i].ival[j], check->ival[j],
				(i==i0 && j==j0) ? "<-failed" : "");
	return 1;
}

static int _dump_fval(int nodes, int i0, int j0,
		      struct cxip_fltval *rslt,
		      struct cxip_fltval *check)
{
	int i, j;

	for (i = 0; i < nodes; i++)
		for (j = 0; j < 4; j++)
			printf("[%2d][%2d] rslt=%016g expect=%016g%s\n",
				i, j, rslt[i].fval[j], check->fval[j],
				(i==i0 && j==j0) ? "<-failed" : "");
	return 1;
}

static int _dump_iminmax(int nodes, int i0,
			struct cxip_iminmax *rslt,
			struct cxip_iminmax *check)
{
	int i;

	for (i = 0; i < nodes; i++) {
		printf("[%2d] iminval=%16lx expect=%16lx%s\n",
			i, rslt[i].iminval, check->iminval,
			(i==i0) ? "<-failed" : "");
		printf("[%2d] iminidx=%16ld expect=%16ld%s\n",
			i, rslt[i].iminidx, check->iminidx,
			(i==i0) ? "<-failed" : "");
		printf("[%2d] imaxval=%16lx expect=%16lx%s\n",
			i, rslt[i].imaxval, check->imaxval,
			(i==i0) ? "<-failed" : "");
		printf("[%2d] imaxidx=%16ld expect=%16ld%s\n",
			i, rslt[i].imaxidx, check->imaxidx,
			(i==i0) ? "<-failed" : "");
	}
	return 1;
}

static int _dump_fminmax(int nodes, int i0,
			struct cxip_fltminmax *rslt,
			struct cxip_fltminmax *check)
{
	int i;

	for (i = 0; i < nodes; i++) {
		printf("[%2d] fminval=%16g expect=%16g%s\n",
			i, rslt[i].fminval, check->fminval,
			(i==i0) ? "<-failed" : "");
		printf("[%2d] fminidx=%16ld expect=%16ld%s\n",
			i, rslt[i].fminidx, check->fminidx,
			(i==i0) ? "<-failed" : "");
		printf("[%2d] fmaxval=%16g expect=%16g%s\n",
			i, rslt[i].fmaxval, check->fmaxval,
			(i==i0) ? "<-failed" : "");
		printf("[%2d] fmaxidx=%16ld expect=%16ld%s\n",
			i, rslt[i].fmaxidx, check->fmaxidx,
			(i==i0) ? "<-failed" : "");
	}
	return 1;
}

/* compares collective integer rslt with computed check */
static int _check_ival(int nodes, struct cxip_intval *rslt,
			struct cxip_intval *check)
{
	int i, j;

	for (i = 0; i < nodes; i++)
		for (j = 0; j < 4; j++)
			if (rslt[i].ival[j] != check->ival[j])
				return _dump_ival(nodes, i, j, rslt, check);
	return 0;
}

/* compares collective double rslt with computed check */
static int _check_fval(int nodes, struct cxip_fltval *rslt,
			struct cxip_fltval *check)
{
	int i, j;

	for (i = 0; i < nodes; i++)
		for (j = 0; j < 4; j++)
			if (!_feq(rslt[i].fval[j], check->fval[j]))
				return _dump_fval(nodes, i, j, rslt, check);
	return 0;
}

/* compares collective integer minmax rslt with computed check */
static int _check_iminmax(int nodes, struct cxip_iminmax *rslt,
			  struct cxip_iminmax *check)
{
	int i;

	for (i = 0; i < nodes; i++) {
		if (rslt[i].iminval != check->iminval ||
		    rslt[i].iminidx != check->iminidx ||
		    rslt[i].imaxval != check->imaxval ||
		    rslt[i].imaxidx != check->imaxidx)
			return _dump_iminmax(nodes, i, rslt, check);
	}
	return 0;
}

/* compares collective double minmax rslt with computed check */
static int _check_fminmax(int nodes, struct cxip_fltminmax *rslt,
			  struct cxip_fltminmax *check)
{
	int i;

	for (i = 0; i < nodes; i++)
		if (!_feq(rslt[i].fminval, check->fminval) ||
		    !_feq(rslt[i].fmaxval, check->fmaxval) ||
		    rslt[i].fminidx != check->fminidx ||
		    rslt[i].fmaxidx != check->fmaxidx)
			return _dump_fminmax(nodes, i, rslt, check);
	return 0;
}

/* compares returned RC code with expected value */
static int _check_rc(int nodes, struct user_context *context, int rc)
{
	int i, ret;

	ret = 0;
	for (i = 0; i < nodes; i++)
		if (context[i].hw_rc != rc) {
			printf("hw_rc[%d]=%d!=%d\n", i, context[i].hw_rc, rc);
			ret = 1;
		}
	return ret;
}

/* keeps code easier to read */
#define STDINTSETUP \
	struct user_context *context; \
	struct cxip_intval *data; \
 	struct cxip_intval *rslt; \
	struct cxip_intval check; \
	int i, j, ret, width, nodes; \
	width = sizeof(*data); \
	nodes = cxit_coll_mc_list.count; \
	data = calloc(nodes, sizeof(*data)); \
	rslt = calloc(nodes, sizeof(*rslt)); \
	context = calloc(nodes, sizeof(*context));

#define STDILOCSETUP \
	struct user_context *context; \
	struct cxip_iminmax *data; \
 	struct cxip_iminmax *rslt; \
	struct cxip_iminmax check; \
	int i, ret, width, nodes; \
	width = sizeof(*data); \
	nodes = cxit_coll_mc_list.count; \
	data = calloc(nodes, sizeof(*data)); \
	rslt = calloc(nodes, sizeof(*rslt)); \
	context = calloc(nodes, sizeof(*context));

#define STDFLTSETUP \
	struct user_context *context; \
	struct cxip_fltval *data; \
	struct cxip_fltval *rslt; \
	struct cxip_fltval check; \
	int i, ret, width, nodes; \
	width = sizeof(*data); \
	nodes = cxit_coll_mc_list.count; \
	data = calloc(nodes, sizeof(*data)); \
	rslt = calloc(nodes, sizeof(*rslt)); \
	context = calloc(nodes, sizeof(*context));

#define STDFLOCSETUP \
	struct user_context *context; \
	struct cxip_fltminmax *data; \
	struct cxip_fltminmax *rslt; \
	struct cxip_fltminmax check; \
	int i, ret, width, nodes; \
	width = sizeof(*data); \
	nodes = cxit_coll_mc_list.count; \
	data = calloc(nodes, sizeof(*data)); \
	rslt = calloc(nodes, sizeof(*rslt)); \
	context = calloc(nodes, sizeof(*context));

#define	STDCLEANUP \
	free(context); \
	free(rslt); \
	free(data); \

/* Test binary OR */
Test(coll_reduce_ops, bor)
{
	STDINTSETUP
	/* max nodes == 32 under NETSIM */
	for (i = 0; i < nodes; i++) {
		data[i].ival[0] = 1 << i;
		data[i].ival[1] = i << 2*i;
		data[i].ival[2] = i;
		data[i].ival[3] = 2*i;
	}
	memcpy(&check, &data[0], sizeof(check));
	for (i = 1; i < nodes; i++)
		for (j = 0; j < 4; j++)
			check.ival[j] |= data[i].ival[j];

	ret = _allreduceop(FI_BOR, FI_UINT64, 0L, width, data, rslt, context);
	cr_assert(!ret, "_allreduceop() failed\n");
	ret = _check_ival(nodes, rslt, &check);
	cr_assert(!ret, "compare failed\n");
	ret = _check_rc(nodes, context, CXIP_COLL_RC_SUCCESS);
	cr_assert(!ret, "rc failed\n");
	STDCLEANUP
}

/* Test binary AND */
Test(coll_reduce_ops, band)
{
	STDINTSETUP
	/* max nodes == 32 under NETSIM */
	for (i = 0; i < nodes; i++) {
		data[i].ival[0] = ~(1 << i);
		data[i].ival[1] = ~(i << 2*i);
		data[i].ival[2] = ~i;
		data[i].ival[3] = ~(2*i);
	}
	memcpy(&check, &data[0], sizeof(check));
	for (i = 1; i < nodes; i++)
		for (j = 0; j < 4; j++)
			check.ival[j] &= data[i].ival[j];

	ret = _allreduceop(FI_BAND, FI_UINT64, 0L, width, data, rslt, context);
	cr_assert(!ret, "_allreduceop() failed\n");
	ret = _check_ival(nodes, rslt, &check);
	cr_assert(!ret, "compare failed\n");
	ret = _check_rc(nodes, context, CXIP_COLL_RC_SUCCESS);
	cr_assert(!ret, "rc failed\n");
	STDCLEANUP
}

/* Test binary XOR */
Test(coll_reduce_ops, bxor)
{
	STDINTSETUP
	/* max nodes == 32 under NETSIM */
	for (i = 0; i < nodes; i++) {
		data[i].ival[0] = 1 << i;
		data[i].ival[1] = ~(i << i);
		data[i].ival[2] = i;
		data[i].ival[3] = ~i;
	}
	memcpy(&check, &data[0], sizeof(check));
	for (i = 1; i < nodes; i++)
		for (j = 0; j < 4; j++)
			check.ival[j] ^= data[i].ival[j];

	ret = _allreduceop(FI_BXOR, FI_UINT64, 0L, width, data, rslt, context);
	cr_assert(!ret, "_allreduceop() failed\n");
	ret = _check_ival(nodes, rslt, &check);
	cr_assert(!ret, "compare failed\n");
	ret = _check_rc(nodes, context, CXIP_COLL_RC_SUCCESS);
	cr_assert(!ret, "rc failed\n");
	STDCLEANUP
}

/* Tests int64 minimum */
Test(coll_reduce_ops, imin)
{
	STDINTSETUP
	/* max nodes == 32 under NETSIM */
	for (i = 0; i < nodes; i++) {
		data[i].ival[0] = rand();
		data[i].ival[1] = -rand();
		data[i].ival[2] = rand();
		data[i].ival[3] = -rand();
	}
	memcpy(&check, &data[0], sizeof(check));
	for (i = 1; i < nodes; i++)
		for (j = 0; j < 4; j++)
			check.ival[j] = MIN(check.ival[j], data[i].ival[j]);

	ret = _allreduceop(FI_MIN, FI_UINT64, 0L, width, data, rslt, context);
	cr_assert(!ret, "_allreduceop() failed\n");
	ret = _check_ival(nodes, rslt, &check);
	cr_assert(!ret, "compare failed\n");
	ret = _check_rc(nodes, context, CXIP_COLL_RC_SUCCESS);
	cr_assert(!ret, "rc failed\n");
	STDCLEANUP
}

/* Tests int64 maximum */
Test(coll_reduce_ops, imax)
{
	STDINTSETUP
	/* max nodes == 32 under NETSIM */
	for (i = 0; i < nodes; i++) {
		data[i].ival[0] = rand();
		data[i].ival[1] = -rand();
		data[i].ival[2] = rand();
		data[i].ival[3] = -rand();
	}
	memcpy(&check, &data[0], sizeof(check));
	for (i = 1; i < nodes; i++)
		for (j = 0; j < 4; j++)
			check.ival[j] = MAX(check.ival[j], data[i].ival[j]);

	ret = _allreduceop(FI_MAX, FI_UINT64, 0L, width, data, rslt, context);
	cr_assert(!ret, "_allreduceop() failed\n");
	ret = _check_ival(nodes, rslt, &check);
	cr_assert(!ret, "compare failed\n");
	ret = _check_rc(nodes, context, CXIP_COLL_RC_SUCCESS);
	cr_assert(!ret, "rc failed\n");
	STDCLEANUP
}

/* Tests int64 SUM */
Test(coll_reduce_ops, isum)
{
	STDINTSETUP
	/* max nodes == 32 under NETSIM */
	for (i = 0; i < nodes; i++) {
		data[i].ival[0] = rand();
		data[i].ival[1] = -rand();
		data[i].ival[2] = rand();
		data[i].ival[3] = -rand();
	}
	memcpy(&check, &data[0], sizeof(check));
	for (i = 1; i < nodes; i++)
		for (j = 0; j < 4; j++)
			check.ival[j] += data[i].ival[j];

	ret = _allreduceop(FI_SUM, FI_UINT64, 0L, width, data, rslt, context);
	cr_assert(!ret, "_allreduceop() failed\n");
	ret = _check_ival(nodes, rslt, &check);
	cr_assert(!ret, "compare failed\n");
	ret = _check_rc(nodes, context, CXIP_COLL_RC_SUCCESS);
	cr_assert(!ret, "rc failed\n");
	STDCLEANUP
}

/* Tests int64 minmaxloc */
Test(coll_reduce_ops, iminmaxloc)
{
	STDILOCSETUP
	/* max nodes == 32 under NETSIM */
	for (i = 0; i < nodes; i++) {
		data[i].iminval = rand();
		data[i].iminidx = i;
		data[i].imaxval = rand();
		data[i].imaxidx = i;
	}
	memcpy(&check, &data[0], sizeof(check));
	for (i = 1; i < nodes; i++) {
		if (check.iminval > data[i].iminval) {
			check.iminval = data[i].iminval;
			check.iminidx = data[i].iminidx;
		}
		if (check.imaxval < data[i].imaxval) {
			check.imaxval = data[i].imaxval;
			check.imaxidx = data[i].imaxidx;
		}
	}

	ret = _allreduceop(CXI_FI_MINMAXLOC, FI_UINT64, 0L, width, data, rslt, context);
	cr_assert(!ret, "_allreduceop() failed\n");
	ret = _check_iminmax(nodes, rslt, &check);
	cr_assert(!ret, "compare failed\n");
	ret = _check_rc(nodes, context, CXIP_COLL_RC_SUCCESS);
	cr_assert(!ret, "rc failed\n");
	STDCLEANUP
}

/* Tests double sum */
Test(coll_reduce_ops, fsum)
{
	STDFLTSETUP
	int j;

	/* max nodes == 32 under NETSIM */
	data[0].fval[0] = 1.0e-53;
	data[0].fval[1] = 1.0e-53;
	data[0].fval[2] = 1.0e-53;
	data[0].fval[3] = 1.0e-53;
	for (i = 1; i < nodes; i++) {
		data[i].fval[0] = _frand(1.0);
		data[i].fval[1] = -_frand(1.0);
		data[i].fval[2] = _frand(1.0);
		data[i].fval[3] = -_frand(1.0);
	}
	memcpy(&check, &data[0], sizeof(check));
	for (i = 1; i < nodes; i++)
		for (j = 0; j < 4; j++)
			check.fval[j] += data[i].fval[j];

	ret = _allreduceop(FI_SUM, FI_DOUBLE, 0L, width, data, rslt, context);
	cr_assert(!ret, "_allreduceop() failed\n");
	ret = _check_fval(nodes, rslt, &check);
	cr_assert(!ret, "compare failed\n");
	ret = _check_rc(nodes, context, CXIP_COLL_RC_FLT_INEXACT);
	cr_assert(!ret, "rc failed\n");

	/* Note: inexact computation is guaranteed by the small value included
	 * in the data set. There is a hidden trick when performing the
	 * comparison that relies on the prediction and the NETSIM allreduce
	 * operation both occuring in the same order, due to the nature of the
	 * simulated endpoints. In a real collective, ordering will be random,
	 * and the results will vary according to the ordering.
	 */
	STDCLEANUP
}

/* Test double minimum -- this should be exact */
Test(coll_reduce_ops, fmin)
{
	STDFLTSETUP
	/* max nodes == 32 under NETSIM */
	for (i = 0; i < nodes; i++) {
		data[i].fval[0] = _frand(1.0);
		data[i].fval[1] = -_frand(1.0);
		data[i].fval[2] = _frand(1.0);
		data[i].fval[3] = -_frand(1.0);
	}

	/* normal floating point */
	_predict_fmin(nodes, data, &check, true);
	ret = _allreduceop(FI_MIN, FI_DOUBLE, 0L, width, data, rslt, context);
	cr_assert(!ret, "_allreduceop failed normal");
	ret = _check_fval(nodes, rslt, &check);
	cr_assert(!ret, "compare failed normal\n");
	ret = _check_rc(nodes, context, CXIP_COLL_RC_SUCCESS);
	cr_assert(!ret, "rc failed normal\n");

	data[1].fval[1] = NAN;
	_predict_fmin(nodes, data, &check, true);
	ret = _allreduceop(FI_MIN, FI_DOUBLE, 0L, width, data, rslt, context);
	cr_assert(!ret, "_allreduceop failed NAN");
	ret = _check_fval(nodes, rslt, &check);
	cr_assert(!ret, "compare failed NAN\n");
	ret = _check_rc(nodes, context, CXIP_COLL_RC_FLT_OVERFLOW);
	cr_assert(!ret, "rc failed NAN\n");

	data[1].fval[1] = _snan64();
	_predict_fmin(nodes, data, &check, true);
	ret = _allreduceop(FI_MIN, FI_DOUBLE, 0L, width, data, rslt, context);
	cr_assert(!ret, "_allreduceop failed sNAN");
	ret = _check_fval(nodes, rslt, &check);
	cr_assert(!ret, "compare failed sNAN\n");
	ret = _check_rc(nodes, context, CXIP_COLL_RC_FLT_INVALID);
	cr_assert(!ret, "rc failed sNAN\n");
	STDCLEANUP
}

/* Test double maximum -- this should be exact */
Test(coll_reduce_ops, fmax)
{
	STDFLTSETUP
	/* max nodes == 32 under NETSIM */
	for (i = 0; i < nodes; i++) {
		data[i].fval[0] = _frand(1.0);
		data[i].fval[1] = -_frand(1.0);
		data[i].fval[2] = _frand(1.0);
		data[i].fval[3] = -_frand(1.0);
	}

	_predict_fmax(nodes, data, &check, true);
	ret = _allreduceop(FI_MAX, FI_DOUBLE, 0L, width, data, rslt, context);
	cr_assert(!ret, "_allreduceop failed normal");
	ret = _check_fval(nodes, rslt, &check);
	cr_assert(!ret, "compare failed normal\n");
	ret = _check_rc(nodes, context, CXIP_COLL_RC_SUCCESS);
	cr_assert(!ret, "rc failed normal\n");

	data[1].fval[1] = NAN;
	_predict_fmax(nodes, data, &check, true);
	ret = _allreduceop(FI_MAX, FI_DOUBLE, 0L, width, data, rslt, context);
	cr_assert(!ret, "_allreduceop failed NAN");
	ret = _check_fval(nodes, rslt, &check);
	cr_assert(!ret, "compare failed NAN\n");
	ret = _check_rc(nodes, context, CXIP_COLL_RC_FLT_OVERFLOW);
	cr_assert(!ret, "rc failed NAN\n");

	data[1].fval[1] = _snan64();
	_predict_fmax(nodes, data, &check, true);
	ret = _allreduceop(FI_MAX, FI_DOUBLE, 0L, width, data, rslt, context);
	cr_assert(!ret, "_allreduceop failed sNAN");
	ret = _check_fval(nodes, rslt, &check);
	cr_assert(!ret, "compare failed sNAN\n");
	ret = _check_rc(nodes, context, CXIP_COLL_RC_FLT_INVALID);
	cr_assert(!ret, "rc failed sNAN\n");
	STDCLEANUP
}

/* Test double minmax with index -- should be exact */
Test(coll_reduce_ops, fminmaxloc)
{
	STDFLOCSETUP
	/* max nodes == 32 under NETSIM */
	for (i = 0; i < nodes; i++) {
		data[i].fminval = _frand(1.0);
		data[i].fminidx = i;
		data[i].fmaxval = _frand(1.0);
		data[i].fmaxidx = i;
	}
	memcpy(&check, &data[0], sizeof(check));
	for (i = 1; i < nodes; i++) {
		if (check.fminval > data[i].fminval) {
			check.fminval = data[i].fminval;
			check.fminidx = data[i].fminidx;
		}
		if (check.fmaxval < data[i].fmaxval) {
			check.fmaxval = data[i].fmaxval;
			check.fmaxidx = data[i].fmaxidx;
		}
	}

	_predict_fminmax(nodes, data, &check, true);
	ret = _allreduceop(CXI_FI_MINMAXLOC, FI_DOUBLE, 0L, width, data, rslt, context);
	cr_assert(!ret, "_allreduceop failed normal");
	ret = _check_fminmax(nodes, rslt, &check);
	cr_assert(!ret, "compare failed normal\n");
	ret = _check_rc(nodes, context, CXIP_COLL_RC_SUCCESS);
	cr_assert(!ret, "rc failed normal\n");

	/* NAN is given preference over number */
	data[1].fminval = NAN;
	data[3].fmaxval = NAN;
	_predict_fminmax(nodes, data, &check, true);
	ret = _allreduceop(CXI_FI_MINMAXLOC, FI_DOUBLE, 0L, width, data, rslt, context);
	cr_assert(!ret, "_allreduceop failed NAN");
	ret = _check_fminmax(nodes, rslt, &check);
	cr_assert(!ret, "compare failed NAN\n");
	ret = _check_rc(nodes, context, CXIP_COLL_RC_SUCCESS);	// BUG?
	cr_assert(!ret, "rc failed NAN\n");

	/* SNAN is given preference over NAN */
	data[1].fminval = NAN;
	data[2].fminval = _snan64();
	data[3].fmaxval = NAN;
	_predict_fminmax(nodes, data, &check, true);
	ret = _allreduceop(CXI_FI_MINMAXLOC, FI_DOUBLE, 0L, width, data, rslt, context);
	cr_assert(!ret, "_allreduceop failed sNAN");
	ret = _check_fminmax(nodes, rslt, &check);
	cr_assert(!ret, "compare failed sNAN\n");
	ret = _check_rc(nodes, context, CXIP_COLL_RC_FLT_INVALID);
	cr_assert(!ret, "rc failed sNAN\n");
	STDCLEANUP
}

/* Test double minimum ignoring NAN -- should be exact */
Test(coll_reduce_ops, fminnum)
{
	STDFLTSETUP
	/* max nodes == 32 under NETSIM */
	for (i = 0; i < nodes; i++) {
		data[i].fval[0] = _frand(1.0);
		data[i].fval[1] = -_frand(1.0);
		data[i].fval[2] = _frand(1.0);
		data[i].fval[3] = -_frand(1.0);
	}

	_predict_fmin(nodes, data, &check, false);
	ret = _allreduceop(CXI_FI_MINNUM, FI_DOUBLE, 0L, width, data, rslt, context);
	cr_assert(!ret, "_allreduceop failed normal");
	ret = _check_fval(nodes, rslt, &check);
	cr_assert(!ret, "compare failed normal\n");
	ret = _check_rc(nodes, context, CXIP_COLL_RC_SUCCESS);
	cr_assert(!ret, "rc failed normal\n");

	/* number is given preference over NAN */
	data[1].fval[1] = NAN;
	_predict_fmin(nodes, data, &check, false);
	ret = _allreduceop(CXI_FI_MINNUM, FI_DOUBLE, 0L, width, data, rslt, context);
	cr_assert(!ret, "_allreduceop failed NAN");
	ret = _check_fval(nodes, rslt, &check);
	cr_assert(!ret, "compare failed NAN\n");
	ret = _check_rc(nodes, context, CXIP_COLL_RC_FLT_OVERFLOW);
	cr_assert(!ret, "rc failed NAN\n");

	/* number is given preference over NAN */
	data[1].fval[1] = _snan64();
	_predict_fmin(nodes, data, &check, false);
	ret = _allreduceop(CXI_FI_MINNUM, FI_DOUBLE, 0L, width, data, rslt, context);
	cr_assert(!ret, "_allreduceop failed sNAN");
	ret = _check_fval(nodes, rslt, &check);
	cr_assert(!ret, "compare failed sNAN\n");
	ret = _check_rc(nodes, context, CXIP_COLL_RC_FLT_INVALID);
	cr_assert(!ret, "rc failed sNAN\n");
	STDCLEANUP
}

/* Test double maximum ignoring NAN -- should be exact */
Test(coll_reduce_ops, fmaxnum)
{
	STDFLTSETUP
	/* max nodes == 32 under NETSIM */
	for (i = 0; i < nodes; i++) {
		data[i].fval[0] = _frand(1.0);
		data[i].fval[1] = -_frand(1.0);
		data[i].fval[2] = _frand(1.0);
		data[i].fval[3] = -_frand(1.0);
	}

	_predict_fmax(nodes, data, &check, false);
	ret = _allreduceop(CXI_FI_MAXNUM, FI_DOUBLE, 0L, width, data, rslt, context);
	cr_assert(!ret, "_allreduceop failed normal");
	ret = _check_fval(nodes, rslt, &check);
	cr_assert(!ret, "compare failed normal\n");
	ret = _check_rc(nodes, context, CXIP_COLL_RC_SUCCESS);
	cr_assert(!ret, "rc failed normal\n");

	/* number is given preference over NAN */
	data[1].fval[1] = NAN;
	_predict_fmax(nodes, data, &check, false);
	ret = _allreduceop(CXI_FI_MAXNUM, FI_DOUBLE, 0L, width, data, rslt, context);
	cr_assert(!ret, "_allreduceop failed NAN");
	ret = _check_fval(nodes, rslt, &check);
	cr_assert(!ret, "compare failed NAN\n");
	ret = _check_rc(nodes, context, CXIP_COLL_RC_FLT_OVERFLOW);
	cr_assert(!ret, "rc failed NAN\n");

	/* SNAN is given preference over number */
	data[1].fval[1] = _snan64();
	_predict_fmax(nodes, data, &check, false);
	ret = _allreduceop(CXI_FI_MAXNUM, FI_DOUBLE, 0L, width, data, rslt, context);
	cr_assert(!ret, "_allreduceop failed sNAN");
	ret = _check_fval(nodes, rslt, &check);
	cr_assert(!ret, "compare failed sNAN\n");
	ret = _check_rc(nodes, context, CXIP_COLL_RC_FLT_INVALID);
	cr_assert(!ret, "rc failed sNAN\n");
	STDCLEANUP
}

/* Test double minmax with index ignoring NAN -- should be exact */
Test(coll_reduce_ops, fminmaxnumloc)
{
	STDFLOCSETUP
	/* max nodes == 32 under NETSIM */
	for (i = 0; i < nodes; i++) {
		data[i].fminval = _frand(1.0);
		data[i].fminidx = i;
		data[i].fmaxval = _frand(1.0);
		data[i].fmaxidx = i;
	}
	memcpy(&check, &data[0], sizeof(check));
	for (i = 1; i < nodes; i++) {
		if (check.fminval > data[i].fminval) {
			check.fminval = data[i].fminval;
			check.fminidx = data[i].fminidx;
		}
		if (check.fmaxval < data[i].fmaxval) {
			check.fmaxval = data[i].fmaxval;
			check.fmaxidx = data[i].fmaxidx;
		}
	}

	_predict_fminmax(nodes, data, &check, false);
	ret = _allreduceop(CXI_FI_MINMAXNUMLOC, FI_DOUBLE, 0L, width, data, rslt, context);
	cr_assert(!ret, "_allreduceop failed normal");
	ret = _check_fminmax(nodes, rslt, &check);
	cr_assert(!ret, "compare failed normal\n");
	ret = _check_rc(nodes, context, CXIP_COLL_RC_SUCCESS);
	cr_assert(!ret, "rc failed normal\n");

	/* NAN is given preference over number */
	data[1].fminval = NAN;
	data[3].fmaxval = NAN;
	_predict_fminmax(nodes, data, &check, false);
	ret = _allreduceop(CXI_FI_MINMAXNUMLOC, FI_DOUBLE, 0L, width, data, rslt, context);
	cr_assert(!ret, "_allreduceop failed NAN");
	ret = _check_fminmax(nodes, rslt, &check);
	cr_assert(!ret, "compare failed NAN\n");
	ret = _check_rc(nodes, context, CXIP_COLL_RC_SUCCESS);
	cr_assert(!ret, "rc failed NAN\n");

	/* SNAN is given preference over NAN */
	data[1].fminval = NAN;
	data[2].fminval = _snan64();
	data[3].fmaxval = NAN;
	_predict_fminmax(nodes, data, &check, false);
	ret = _allreduceop(CXI_FI_MINMAXNUMLOC, FI_DOUBLE, 0L, width, data, rslt, context);
	cr_assert(!ret, "_allreduceop failed sNAN");
	ret = _check_fminmax(nodes, rslt, &check);
	cr_assert(!ret, "compare failed sNAN\n");
	ret = _check_rc(nodes, context, CXIP_COLL_RC_FLT_INVALID);
	cr_assert(!ret, "rc failed sNAN\n");
	STDCLEANUP
}
