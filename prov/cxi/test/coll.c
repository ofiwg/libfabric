/*
 * Copyright (c) 2017-2019 Intel Corporation. All rights reserved.
 * Copyright (c) 2020 Cray Inc. All rights reserved.
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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <complex.h>
#include <time.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>

#include <ofi.h>

#include "cxip.h"
#include "cxip_test_common.h"

#define	MIN(a,b) (((a)<(b))?(a):(b))

TestSuite(coll_join, .disabled = false, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test EP close without explicitly enabling collectives.
 */
Test(coll_join, noop)
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
Test(coll_join, enable)
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
Test(coll_join, disable)
{
	struct cxip_ep *ep;
	int ret;

	cxit_setup_rma();
	ep = container_of(cxit_ep, struct cxip_ep, ep);

	ret = cxip_coll_disable(ep->ep_obj);
	cr_assert(ret == 0, "cxip_coll_disable failed: %d\n", ret);
	cr_assert(!ep->ep_obj->coll.enabled,
		  "coll enabled after disabling\n");
	cxit_teardown_rma();
}

/* Test EP close after disabling/re-enabling collectives.
 */
Test(coll_join, reenable)
{
	struct cxip_ep *ep;
	int ret;

	cxit_setup_rma();
	ep = container_of(cxit_ep, struct cxip_ep, ep);

	ret = cxip_coll_disable(ep->ep_obj);
	cr_assert(ret == 0, "cxip_coll_disable failed: %d\n", ret);
	cr_assert(!ep->ep_obj->coll.enabled,
		  "coll enabled after disabling\n");

	ret = cxip_coll_enable(ep->ep_obj);
	cr_assert(ret == 0, "cxip_coll_enable failed: %d\n", ret);
	cr_assert(ep->ep_obj->coll.enabled,
		  "coll not enabled after enabling\n");
	cxit_teardown_rma();
}

/***************************************/

TestSuite(coll_put, .init = cxit_setup_rma, .fini = cxit_teardown_rma,
	  .disabled = false, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Basic test of single join.
 */
Test(coll_put, join1)
{
	cxit_create_netsim_collective(1);
	cxit_destroy_netsim_collective();
}

/* Basic test of two joins.
 */
Test(coll_put, join2)
{
	cxit_create_netsim_collective(2);
	cxit_destroy_netsim_collective();
}

/***************************************/

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
 * multicast resource to another.
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

	/* clear any prior values */
	cxip_coll_reset_mc_ctrs(mc_obj_send);
	cxip_coll_reset_mc_ctrs(mc_obj_recv);

	/* from_rank reduction */
	reduction = &mc_obj_send->reduction[0];

	/* must persist until _progress called, for validation */
	buffers = calloc(PROGRESS_COUNT, sizeof(*buf));

	buf = buffers;
	sendcnt = 0;
	dataval = 0;
	for (i = 0; i < count; i++) {
		for (j = 0; j < 6; j++)
			buf->count[j] = i;
		buf->pad = i;
		ret = cxip_coll_send(reduction, to_rank, buf, sizeof(*buf));
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
	_progress_put(ep->ep_obj->coll.rx_cq, sendcnt, &dataval);

	/* check final counts */
	if (count * sizeof(*buf) >
	    ep->ep_obj->coll.buffer_size - ep->ep_obj->min_multi_recv) {
		cnt = ofi_atomic_get32(&mc_obj_recv->coll_pte->buf_swap_cnt);
		cr_assert(cnt > 0, "Did not recirculate buffers\n");
	}

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

	cxit_create_netsim_collective(2);

	mc_obj = container_of(cxit_coll_mc_list.mc_fid[0],
			      struct cxip_coll_mc, mc_fid);
	reduction = &mc_obj->reduction[0];

	ret = cxip_coll_send(reduction, 3, &buf, sizeof(buf));
	cr_assert(ret == -FI_EINVAL, "cxip_coll_set bad error = %d\n", ret);

	cxit_destroy_netsim_collective();
}

/* Basic test with one packet from rank 0 to rank 0.
 */
Test(coll_put, put_one)
{
	cxit_create_netsim_collective(1);
	_put_data(1, 0, 0);
	cxit_destroy_netsim_collective();
}

/* Basic test with one packet from each rank to another rank.
 * Exercises NETSIM rank-based target addressing.
 */
Test(coll_put, put_ranks)
{
	cxit_create_netsim_collective(2);
	_put_data(1, 0, 0);
	_put_data(1, 0, 1);
	_put_data(1, 1, 0);
	_put_data(1, 1, 1);
	cxit_destroy_netsim_collective();
}

/* Test a lot of packets to force buffer rollover.
 */
Test(coll_put, put_many, .timeout = 30)
{
	cxit_create_netsim_collective(1);
	_put_data(4000, 0, 0);
	cxit_destroy_netsim_collective();
}

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
	int sendcnt, cnt;
	uint64_t dataval;
	int i, ret;

	cxit_create_netsim_collective(1);

	mc_obj = container_of(cxit_coll_mc_list.mc_fid[0],
			      struct cxip_coll_mc, mc_fid);

	/* clear counters */
	cxip_coll_reset_mc_ctrs(mc_obj);

	sendcnt = 0;
	dataval = 0;
	reduction = &mc_obj->reduction[0];
	reduction->op_state = CXIP_COLL_STATE_NONE;
	for (i = 0; i < count; i++) {
		ret = cxip_coll_send_red_pkt(reduction, 1, 0,
					     &dataval, sizeof(uint64_t), false);
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
	cr_assert(cnt == count, "Bad send counter on root: %d\n", cnt);
	cnt = ofi_atomic_get32(&mc_obj->recv_cnt);
	cr_assert(cnt == count, "Bad recv counter on root: %d\n", cnt);
	cnt = ofi_atomic_get32(&mc_obj->pkt_cnt);
	cr_assert(cnt == count, "Bad pkt counter on root: %d\n", cnt);

	cxit_destroy_netsim_collective();
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
	struct fi_cq_data_entry entry;
	uint64_t data;
	int i, cnt, ret;

	cxit_create_netsim_collective(5);

	for (i = 0; i < 5; i++) {
		mc_obj[i] = container_of(cxit_coll_mc_list.mc_fid[i],
					 struct cxip_coll_mc, mc_fid);
		mc_obj[i]->reduction[0].op_state = CXIP_COLL_STATE_NONE;
		cxip_coll_reset_mc_ctrs(mc_obj[i]);
	}

	data = 0;
	rx_cq = mc_obj[0]->ep_obj->coll.rx_cq;

	reduction = &mc_obj[0]->reduction[0];
	ret = cxip_coll_send_red_pkt(reduction, 1, 0,
				     &data, sizeof(uint64_t), false);
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
		cxip_coll_reset_mc_ctrs(mc_obj[i]);
	for (i = 1; i < 5; i++) {
		data = i;
		reduction = &mc_obj[i]->reduction[0];
		ret = cxip_coll_send_red_pkt(reduction, 1, 0,
					     &data, sizeof(uint64_t), false);
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

	cxit_destroy_netsim_collective();
}

/***************************************/

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

/* Simulated user data type */
struct int_data {
	uint64_t ival[4];
};

/* Test makes a direct call into cxip_coll_inject(), to obtain red_id */
ssize_t _allreduce(struct fid_ep *ep, const void *buf, size_t count,
		   void *desc, void *result, void *result_desc,
		   fi_addr_t coll_addr, enum fi_datatype datatype,
		   enum fi_op op, uint64_t flags, void *context,
		   int *reduction_id)
{
	struct cxip_ep *cxi_ep;
	struct cxip_coll_mc *mc_obj;
	int ret;

	cxi_ep = container_of(ep, struct cxip_ep, ep.fid);
	mc_obj = (struct cxip_coll_mc *) ((uintptr_t) coll_addr);

	if (mc_obj->ep_obj != cxi_ep->ep_obj)
		return -FI_EINVAL;

	ret = cxip_coll_inject(mc_obj, FI_ALLREDUCE,
			       datatype, op, buf, result, count,
			       context, reduction_id);

	return ret;
}

/* Poll rx and tx CQs until user context seen.
 *
 * If context == NULL, this polls to advance the state and returns.
 */
void _reduce_wait(struct fid_cq *rx_cq, struct fid_cq *tx_cq,
		  struct user_context *context)
{
	static struct dlist_entry done_list, *done;
	static int initialized = 0;

	struct fi_cq_data_entry entry;
	struct fi_cq_err_entry err_entry;
	struct user_context *ctx;
	int ret;

	/* initialize the static locals */
	if (! initialized) {
		dlist_init(&done_list);
		initialized = 1;
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
			/* read rx CQ and progress state until nothing to do */
			ret = fi_cq_read(rx_cq, &entry, 1);
			if (ret == -FI_EAGAIN && context)
				continue;
			/* read tx CQ to see a completion event */
			ret = fi_cq_read(tx_cq, &entry, 1);
			if (ret != -FI_EAGAIN || !context)
				break;
		} while (true);
		ctx = NULL;
		if (ret == -FI_EAVAIL) {
			/* tx CQ posted an error, copy to user context */
			ret = fi_cq_readerr(tx_cq, &err_entry, 1);
			cr_assert(ret == 1, "fi_cq_readerr failed: %d\n", ret);
			ctx = err_entry.op_context;
			ctx->errcode = err_entry.err;
			ctx->hw_rc = err_entry.prov_errno;
			cr_assert(err_entry.err != 0,
				  "Failure with good return\n");
		} else if (ret == 1) {
			/* tx CQ posted a normal completion */
			ctx = entry.op_context;
			ctx->errcode = 0;
			ctx->hw_rc = 0;
		}
		if (ctx == context)
			return;
		if (ctx)
			dlist_insert_tail(&ctx->entry, &done_list);
	} while (context);
}

/* Exercise the collective state machine.
 *
 * This presumes NETSIM, and considers the size of the collective to be the
 * number of mc_objects in the cxit_coll_mc_list. The first node (zero) is
 * always the root node.
 *
 * We initiate the collective in sequence, beginning with 'start'. If start is
 * zero, the root node initiates first, otherwise a leaf node initiates first.
 *
 * We inject an error by specifying a 'bad' node in the range of nodes. If bad
 * is outside the range (e.g. -1), no errors will be injected. The injection is
 * done by choosing to send the wrong reduction operation code for the bad node,
 * which causes the entire reduction to fail.
 *
 * We perform 'count' reductions to exercise the round-robin reduction ID
 * handling and blocking.
 *
 * We generate different results for each reduction.
 */
void _reduce(int start, int bad, int count)
{
	struct cxip_ep_obj *ep_obj;
	struct cxip_coll_mc **mc_obj;
	struct user_context **context;
	struct int_data **rslt;
	struct int_data *data;
	struct fid_cq *rx_cq, *tx_cq;
	ssize_t size;
	int nodes, first, last, base;
	char label[128];
	int i, ret;

	count = MAX(count, 1);
	nodes = cxit_coll_mc_list.count;
	context = calloc(nodes, sizeof(**context));
	mc_obj = calloc(nodes, sizeof(**mc_obj));
	rslt = calloc(nodes, sizeof(**rslt));
	data = calloc(nodes, sizeof(*data));
	start %= nodes;
	snprintf(label, sizeof(label), "{%2d,%2d,%2d}",
		 start, bad, count);
	ep_obj = NULL;
	for (i = 0; i < nodes; i++) {
		context[i] = calloc(count, sizeof(struct user_context));
		rslt[i] = calloc(count, sizeof(struct int_data));
		mc_obj[i] = container_of(cxit_coll_mc_list.mc_fid[i],
					 struct cxip_coll_mc, mc_fid);
		if (!ep_obj)
			ep_obj = mc_obj[i]->ep_obj;
		cr_assert(mc_obj[i]->ep_obj == ep_obj,
			  "%s Mismatched endpoints\n", label);
	}
	cr_assert(ep_obj != NULL,
		  "%s Did not find an endpoint object\n", label);
	rx_cq = &ep_obj->coll.rx_cq->util_cq.cq_fid;
	tx_cq = &ep_obj->coll.tx_cq->util_cq.cq_fid;
	first = last = 0;

	/* Issue all of the collectives */
	base = 1;
	while (last < count) {
		uint64_t result = 0;
		for (i = 0; i < nodes; i++) {
			enum fi_op op = (start == bad) ? FI_BAND : FI_BOR;
			int red_id;

			data[start].ival[0] = (base << start);
			base <<= 1;
			if (base > 16)
				base = 1;
			context[start][last].node = start;
			context[start][last].seqno = last;
			result |= data[start].ival[0];
			do {
				_reduce_wait(rx_cq, tx_cq, NULL);
				size = _allreduce(cxit_ep, &data[start], 1,
						  NULL, &rslt[start][last],
						  NULL,
						  (fi_addr_t)mc_obj[start],
						  FI_UINT64, op, 0,
						  &context[start][last],
						  &red_id);
			} while (size == -FI_EBUSY);
			context[start][last].red_id = red_id;
			start = (start + 1) % nodes;
		}
		for (i = 0; i < nodes; i++)
			context[i][last].expval = result;

		/* Ensure these all used the same reduction ID */
		ret = 0;
		for (i = 1; i < nodes; i++)
			if (context[0][last].red_id != context[i][last].red_id)
				ret = -1;
		if (ret) {
			for (i = 0; i < nodes; i++)
				printf("%s [%d, %d] red_id = %d\n",
				       label, i, last,
				       context[i][last].red_id);
			cr_assert(true, "%s reduction ID mismatch\n", label);
		}
		last++;
	}

	/* Wait for all reductions to complete */
	while (first < last) {
		struct user_context *ctx;
		int red_id0, errcode0, hw_rc0;
		uint64_t actval0;

		for (i = 0; i < nodes; i++) {
			_reduce_wait(rx_cq, tx_cq, &context[i][first]);
			if (i == 0) {
				red_id0 = context[i][first].red_id;
				errcode0 = context[i][first].errcode;
				hw_rc0 = context[i][first].hw_rc;
				actval0 = rslt[i][first].ival[0];
			}
			ctx = &context[i][first];
			if (ctx->node != i ||
			    ctx->seqno != first  ||
			    ctx->red_id != red_id0 ||
			    ctx->errcode != errcode0 ||
			    ctx->hw_rc != hw_rc0 ||
			    (!errcode0 && ctx->expval != actval0)) {
				printf("%s =====\n", label);
				printf("  node    %3d, exp %3d\n",
				       ctx->node, i);
				printf("  seqno   %3d, exp %3d\n",
				       ctx->seqno, first);
				printf("  red_id  %3d, exp %3d\n",
				       ctx->red_id, red_id0);
				printf("  errcode %3d, exp %3d\n",
				       ctx->errcode, errcode0);
				printf("  hw_rc   %3d, exp %3d\n",
				       ctx->hw_rc, hw_rc0);
				printf("  value   %08lx, exp %08lx\n",
				       ctx->expval, actval0);
				cr_assert(true, "%s context failure\n",
					  label);
			}
		}
		first++;
	}
	for (i = 0; i < nodes; i++) {
		free(rslt[i]);
		free(context[i]);
	}
	free(context);
	free(rslt);
	free(data);
	free(mc_obj);
}

Test(coll_reduce, permute)
{
	printf("\n");
	cxit_create_netsim_collective(5);
	_reduce(0, -1, 1);
	_reduce(1, -1, 1);
	_reduce(2, -1, 1);
	_reduce(3, -1, 1);
	_reduce(4, -1, 1);
	_reduce(0, 0, 1);
	_reduce(0, 1, 1);
	_reduce(1, 0, 1);
	cxit_destroy_netsim_collective();
}

Test(coll_reduce, concur)
{
	printf("\n");
	cxit_create_netsim_collective(5);
	_reduce(0, -1, 29);
	_reduce(0,  0, 29);
	_reduce(0,  1, 29);
	_reduce(1, -1, 29);
	_reduce(1,  0, 29);
	_reduce(1,  1, 29);
	cxit_destroy_netsim_collective();
}
