/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * This software is available to you under the BSD license
 * below:
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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Test aborting in-flight RMA operations by closing MRs.
 *
 * The initiator posts the operations; the target is the remote peer.
 *
 * Test modes:
 *   - Initiator close: initiator posts RMA ops, then closes its local MRs
 *   - Target close: initiator posts RMA ops, target closes its remote MRs
 *   - Multi-op per MR: N ops share 1 MR, close aborts all remaining
 *   - Partial close: 2 MRs on same buffer, close only 1, other completes
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <time.h>

#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_cm.h>

#include "shared.h"

enum close_order_mode {
	CLOSE_ORDER_REVERSE,
	CLOSE_ORDER_RANDOM,
};

enum close_side {
	CLOSE_INITIATOR,
	CLOSE_TARGET,
};

enum test_mode {
	TEST_ABORT,
	TEST_PARTIAL,
	TEST_SEND,
	TEST_TAGGED,
};

/*
 * Each MR slot can have ops_per_mr operations posted against it.
 * op_ctx tracks per-operation state; mr_slot tracks per-MR state.
 */
struct op_ctx {
	struct fi_context2 context;
	int mr_idx;	/* which mr_slot this op belongs to */
	int completed;
	int status;	/* 0 = success, negative = error code */
	int prov_errno;
};

struct mr_slot {
	char *buf;
	struct fid_mr *mr;
	void *desc;
	uint64_t key;
	unsigned int posted;	/* number of ops posted using this MR */
	int mr_closed;
};

struct expected_err {
	int err;
	int prov_errno;
};

static struct mr_slot *slots;
static struct op_ctx *op_arr;
static int *close_order;
static int num_mrs = 8192;
static int ops_per_mr = 1;
static enum close_order_mode close_order_mode = CLOSE_ORDER_REVERSE;
static enum close_side close_side = CLOSE_INITIATOR;
static enum test_mode test_mode = TEST_ABORT;
static int close_ep_first;

/* Remote side MR info */
static struct fi_rma_iov *remote_arr;

#define MR_ABORT_KEY_BASE 0x1000
#define CQ_TIMEOUT_MS 30000

static int max_ops(void)
{
	int n = num_mrs * ops_per_mr;

	/* Partial test always uses 2 ops regardless of num_mrs */
	if (test_mode == TEST_PARTIAL && n < 2)
		n = 2;

	/* Target-close posts an extra signal write before filling the queue */
	if (close_side == CLOSE_TARGET)
		n++;

	return n;
}

static int alloc_test_res(void)
{
	int i;

	slots = calloc(num_mrs, sizeof(*slots));
	if (!slots)
		return -FI_ENOMEM;

	op_arr = calloc(max_ops(), sizeof(*op_arr));
	if (!op_arr)
		return -FI_ENOMEM;

	close_order = calloc(num_mrs, sizeof(*close_order));
	if (!close_order)
		return -FI_ENOMEM;

	remote_arr = calloc(num_mrs, sizeof(*remote_arr));
	if (!remote_arr)
		return -FI_ENOMEM;

	for (i = 0; i < num_mrs; i++) {
		slots[i].buf = calloc(1, opts.transfer_size);
		if (!slots[i].buf)
			return -FI_ENOMEM;
		slots[i].key = MR_ABORT_KEY_BASE + i;
	}

	return 0;
}

static int free_test_res(void)
{
	int i, ret, err = 0;

	if (slots) {
		for (i = 0; i < num_mrs; i++) {
			if (slots[i].mr) {
				ret = fi_close(&slots[i].mr->fid);
				if (ret) {
					FT_ERR("Cleanup: fi_close(mr) slot %d "
						"failed: %d (%s)",
						i, ret, fi_strerror(-ret));
					err = ret;
				}
				slots[i].mr = NULL;
			}
			free(slots[i].buf);
		}
		free(slots);
		slots = NULL;
	}
	free(op_arr);
	op_arr = NULL;
	free(close_order);
	close_order = NULL;
	free(remote_arr);
	remote_arr = NULL;
	return err;
}

static int register_mrs(uint64_t access)
{
	int i, ret;

	for (i = 0; i < num_mrs; i++) {
		if (slots[i].mr)
			continue;

		ret = ft_reg_mr(fi, slots[i].buf, opts.transfer_size,
				access, slots[i].key, opts.iface,
				opts.device, &slots[i].mr, &slots[i].desc);
		if (ret) {
			FT_PRINTERR("ft_reg_mr", ret);
			return ret;
		}
	}
	return 0;
}

static int exchange_mr_keys(void)
{
	struct fi_rma_iov *local_info;
	int i, ret;

	local_info = calloc(num_mrs, sizeof(*local_info));
	if (!local_info)
		return -FI_ENOMEM;

	for (i = 0; i < num_mrs; i++) {
		local_info[i].key = fi_mr_key(slots[i].mr);
		local_info[i].addr =
			(fi->domain_attr->mr_mode & FI_MR_VIRT_ADDR) ?
			(uintptr_t) slots[i].buf : 0;
		local_info[i].len = opts.transfer_size;
	}

	if (opts.dst_addr) {
		ret = ft_sock_send(oob_sock, local_info,
				   num_mrs * sizeof(*local_info));
		if (ret)
			goto out;
		ret = ft_sock_recv(oob_sock, remote_arr,
				   num_mrs * sizeof(*remote_arr));
	} else {
		ret = ft_sock_recv(oob_sock, remote_arr,
				   num_mrs * sizeof(*remote_arr));
		if (ret)
			goto out;
		ret = ft_sock_send(oob_sock, local_info,
				   num_mrs * sizeof(*local_info));
	}

out:
	free(local_info);
	return ret;
}

static void reset_test_state(void)
{
	int i;

	for (i = 0; i < num_mrs; i++) {
		slots[i].posted = 0;
		slots[i].mr_closed = 0;
	}
	for (i = 0; i < max_ops(); i++) {
		op_arr[i].completed = 0;
		op_arr[i].status = 0;
		op_arr[i].mr_idx = -1;
	}
}

/*
 * Post helpers use per-slot MR descriptors and buffers rather than the
 * shared ft_post_rma/ft_post_tx/ft_post_rx helpers, which rely on the
 * global mr_desc/tx_buf/rx_buf. Each operation needs its own MR so we
 * can close them individually.
 */
static ssize_t post_rma_op(int op_idx, int mr_idx)
{
	struct mr_slot *s = &slots[mr_idx];
	struct op_ctx *o = &op_arr[op_idx];

	o->mr_idx = mr_idx;

	switch (opts.rma_op) {
	case FT_RMA_WRITE:
		return fi_write(ep, s->buf, opts.transfer_size, s->desc,
				remote_fi_addr, remote_arr[mr_idx].addr,
				remote_arr[mr_idx].key, &o->context);
	case FT_RMA_WRITEDATA:
		return fi_writedata(ep, s->buf, opts.transfer_size, s->desc,
				    remote_cq_data, remote_fi_addr,
				    remote_arr[mr_idx].addr,
				    remote_arr[mr_idx].key, &o->context);
	case FT_RMA_READ:
		return fi_read(ep, s->buf, opts.transfer_size, s->desc,
			       remote_fi_addr, remote_arr[mr_idx].addr,
			       remote_arr[mr_idx].key, &o->context);
	default:
		return -FI_EINVAL;
	}
}

static ssize_t post_send_op(int op_idx, int mr_idx)
{
	struct mr_slot *s = &slots[mr_idx];
	struct op_ctx *o = &op_arr[op_idx];

	o->mr_idx = mr_idx;

	if (test_mode == TEST_TAGGED)
		return fi_tsend(ep, s->buf, opts.transfer_size, s->desc,
				remote_fi_addr, 0xCAFE, &o->context);
	else
		return fi_send(ep, s->buf, opts.transfer_size, s->desc,
			       remote_fi_addr, &o->context);
}

static ssize_t post_recv_op(int op_idx, int mr_idx)
{
	struct mr_slot *s = &slots[mr_idx];
	struct op_ctx *o = &op_arr[op_idx];

	o->mr_idx = mr_idx;

	if (test_mode == TEST_TAGGED)
		return fi_trecv(ep, s->buf, opts.transfer_size, s->desc,
				remote_fi_addr, 0xCAFE, 0, &o->context);
	else
		return fi_recv(ep, s->buf, opts.transfer_size, s->desc,
			       remote_fi_addr, &o->context);
}

static void shuffle(int *arr, int n)
{
	int i, j, tmp;

	for (i = n - 1; i > 0; i--) {
		j = rand() % (i + 1);
		tmp = arr[i];
		arr[i] = arr[j];
		arr[j] = tmp;
	}
}

static void build_close_order(int num_mrs)
{
	int i, tmp;

	for (i = 0; i < num_mrs; i++)
		close_order[i] = i;

	switch (close_order_mode) {
	case CLOSE_ORDER_REVERSE:
		for (i = 0; i < num_mrs / 2; i++) {
			tmp = close_order[i];
			close_order[i] = close_order[num_mrs - 1 - i];
			close_order[num_mrs - 1 - i] = tmp;
		}
		break;
	case CLOSE_ORDER_RANDOM:
		shuffle(close_order, num_mrs);
		break;
	}
}

static int is_expected_err(struct fi_cq_err_entry *err,
			   struct expected_err *list, int count)
{
	int i;

	if (count < 0)
		return 1; /* negative count = accept anything */

	for (i = 0; i < count; i++) {
		if (err->err == list[i].err &&
		    err->prov_errno == list[i].prov_errno)
			return 1;
	}
	return 0;
}

static int drain_cq(struct fid_cq *cq, int expected,
		    struct expected_err *err_list, int err_count)
{
	struct fi_cq_tagged_entry comp;
	struct fi_cq_err_entry err;
	struct op_ctx *o;
	uint64_t deadline;
	int remaining, ret;

	remaining = expected;
	deadline = ft_gettime_ms() + CQ_TIMEOUT_MS;

	while (remaining > 0 && ft_gettime_ms() < deadline) {
		ret = fi_cq_read(cq, &comp, 1);
		if (ret > 0) {
			o = container_of(comp.op_context,
					 struct op_ctx, context);
			o->completed = 1;
			o->status = 0;
			remaining--;
		} else if (ret == -FI_EAVAIL) {
			memset(&err, 0, sizeof(err));
			ret = fi_cq_readerr(cq, &err, 0);
			if (ret < 0 && ret != -FI_EAGAIN) {
				FT_PRINTERR("fi_cq_readerr", ret);
				return ret;
			}
			if (ret == 1) {
				o = container_of(err.op_context,
						 struct op_ctx, context);
				o->completed = 1;
				o->status = -err.err;
				o->prov_errno = err.prov_errno;
				if (!is_expected_err(&err, err_list,
						     err_count)) {
					FT_ERR("Unexpected CQ error:");
					FT_CQ_ERR(cq, err, NULL, 0);
					return -FI_EOTHER;
				}
				remaining--;
			}
		} else if (ret < 0 && ret != -FI_EAGAIN) {
			FT_PRINTERR("fi_cq_read", ret);
			return ret;
		}
	}

	if (expected > remaining) {
		struct {
			int err;
			int prov_errno;
			int count;
		} buckets[16];
		int i, j, nb = 0;

		for (i = 0; i < expected; i++) {
			if (!op_arr[i].completed || op_arr[i].status == 0)
				continue;
			for (j = 0; j < nb; j++) {
				if (buckets[j].err == -op_arr[i].status &&
				    buckets[j].prov_errno == op_arr[i].prov_errno) {
					buckets[j].count++;
					break;
				}
			}
			if (j == nb && nb < 16) {
				buckets[nb].err = -op_arr[i].status;
				buckets[nb].prov_errno = op_arr[i].prov_errno;
				buckets[nb].count = 1;
				nb++;
			}
		}
		if (nb > 0) {
			fprintf(stderr, "  CQ error breakdown:");
			for (j = 0; j < nb; j++)
				fprintf(stderr, " [err=%d(%s) prov_errno=%d x%d]",
				       buckets[j].err,
				       fi_strerror(buckets[j].err),
				       buckets[j].prov_errno,
				       buckets[j].count);
			fprintf(stderr, "\n");
		}
	}

	return remaining;
}

static const char *op_str(void)
{
	switch (opts.rma_op) {
	case FT_RMA_WRITE: return "write";
	case FT_RMA_WRITEDATA: return "writedata";
	case FT_RMA_READ: return "read";
	default: return "unknown";
	}
}

static const char *close_order_str(void)
{
	return close_order_mode == CLOSE_ORDER_REVERSE ? "reverse" : "random";
}

static const char *side_str(void)
{
	return close_side == CLOSE_INITIATOR ? "initiator" : "target";
}

/* Drain all available rxcq entries without blocking. */
static int flush_rxcq(void)
{
	struct fi_cq_data_entry wc;
	struct fi_cq_err_entry wc_err;
	int ret;

	for (;;) {
		ret = fi_cq_read(rxcq, &wc, 1);
		if (ret == -FI_EAVAIL) {
			memset(&wc_err, 0, sizeof(wc_err));
			fi_cq_readerr(rxcq, &wc_err, 0);
			FT_ERR("Unexpected target rxcq error:");
			FT_CQ_ERR(rxcq, wc_err, NULL, 0);
			return -FI_EOTHER;
		} else if (ret <= 0) {
			break;
		}
	}
	return 0;
}

/*
 * Fill the TX work queue by posting RMA operations until EAGAIN.
 * Posts ops_per_mr operations per MR slot. Returns the total number
 * of operations posted via total_posted and MR slots used via mrs_used.
 */
static int fill_rma_queue(int start_idx, int *total_posted, int *mrs_used)
{
	int i, mr_idx, op_idx = start_idx, ret;
	int tp = 0, mu = 0;

	for (mr_idx = 0; mr_idx < num_mrs; mr_idx++) {
		unsigned int posted_this_mr = 0;
		int eagain = 0;

		for (i = 0; i < ops_per_mr; i++) {
			ret = post_rma_op(op_idx, mr_idx);
			if (ret == -FI_EAGAIN) {
				eagain = 1;
				break;
			}
			if (ret) {
				FT_PRINTERR("post_rma_op", ret);
				return ret;
			}
			posted_this_mr++;
			op_idx++;
			tp++;
		}
		if (posted_this_mr > 0) {
			slots[mr_idx].posted = posted_this_mr;
			mu++;
		}
		if (eagain)
			break;
	}

	*total_posted = tp;
	*mrs_used = mu;
	return 0;
}

/*
 * Test 1: Fill-and-abort (RMA)
 *
 * Initiator mode: fill TX queue with RMA ops, then close local MRs.
 * Target mode: send a 0-byte write-with-imm as a "go" signal, then
 *   fill the TX queue. The target closes all remote MRs as soon as
 *   it receives the signal, racing the in-flight writes/reads.
 *
 * Pass criteria: every posted op produces a completion (success or
 * error). No silent drops (missing == 0).
 */
static int run_fill_abort_client(int iter)
{
	int i, ret;
	int total_posted, mrs_used;
	int completed_ok, completed_err, missing;

	reset_test_state();

	total_posted = 0;
	mrs_used = 0;

	if (close_side == CLOSE_TARGET) {
		/*
		 * Target-close: send a single 0-byte write-with-imm
		 * as a "go" signal, then fill the TX queue with large
		 * writes. The target starts closing all MRs as soon as
		 * it gets the signal, racing the large writes.
		 *
		 * Use op_arr[0] as context so drain_cq's container_of
		 * resolves to a valid op_ctx.
		 */
		op_arr[0].mr_idx = 0;
		ret = fi_writedata(ep, NULL, 0, NULL,
				   (uint64_t) 0xFFFF,
				   remote_fi_addr,
				   remote_arr[0].addr,
				   remote_arr[0].key,
				   &op_arr[0].context);
		if (ret) {
			FT_PRINTERR("fi_writedata (signal)", ret);
			return ret;
		}
		total_posted++; /* count the signal */
	}

	/*
	 * Initiator-close with writedata: sync so the target is
	 * already draining its rxcq before we flood it.
	 */
	if (close_side == CLOSE_INITIATOR && opts.rma_op == FT_RMA_WRITEDATA) {
		ret = ft_sync();
		if (ret)
			return ret;
	}

	ret = fill_rma_queue(total_posted, &i, &mrs_used);
	if (ret)
		return ret;
	total_posted += i;

	if (total_posted == 0) {
		FT_ERR("could not post any operations");
		return -FI_EINVAL;
	}

	/* Close MRs (initiator mode only) */
	if (close_side == CLOSE_INITIATOR) {
		int close_failures = 0;

		build_close_order(mrs_used);
		for (i = 0; i < mrs_used; i++) {
			int idx = close_order[i];

			if (!slots[idx].mr)
				continue;

			ret = fi_close(&slots[idx].mr->fid);
			if (ret)
				close_failures++;
			slots[idx].mr = NULL;
			slots[idx].mr_closed = 1;
		}
		if (close_failures) {
			FT_ERR("MR close: %d/%d failed", close_failures, mrs_used);
			return -FI_EOTHER;
		}
	}

	/* Drain CQ */
	if (close_side == CLOSE_TARGET) {
		struct expected_err target_errs[] = {
			{ .err = FI_EINVAL, .prov_errno = 7 },  /* remote MR invalid */
		};
		missing = drain_cq(txcq, total_posted, target_errs, 1);
	} else {
		struct expected_err initiator_errs[] = {
			{ .err = FI_ECANCELED, .prov_errno = 1 },    /* device flush */
			{ .err = FI_ECANCELED, .prov_errno = 4100 },  /* RDM pkt post fail */
			{ .err = FI_EINVAL, .prov_errno = 5 },        /* local MR invalid */
		};
		missing = drain_cq(txcq, total_posted, initiator_errs, 3);
	}

	if (missing < 0)
		return missing; /* drain_cq hit unexpected error */

	/* Report */
	completed_ok = 0;
	completed_err = 0;
	for (i = 0; i < total_posted; i++) {
		if (op_arr[i].completed) {
			if (op_arr[i].status == 0)
				completed_ok++;
			else
				completed_err++;
		}
	}

	fprintf(stderr, "Iteration %d: op=%s size=%zu posted=%d mrs=%d "
	       "ops_per_mr=%d ok=%d err=%d missing=%d "
	       "close_order=%s side=%s ... %s\n",
	       iter, op_str(), opts.transfer_size, total_posted,
	       mrs_used, ops_per_mr, completed_ok, completed_err,
	       missing, close_order_str(), side_str(),
	       missing == 0 ? "PASS" : "FAIL");

	return missing == 0 ? 0 : -FI_EOTHER;
}

/*
 * Server side for target-close mode.
 *
 * Polls rxcq for write-with-imm completions. The imm data carries
 * the MR index. On each completion, immediately close that MR.
 * No recv posting needed — fi_writedata is an RMA op that generates
 * a remote CQ entry directly.
 */
static int run_fill_abort_server(void)
{
	struct fi_cq_data_entry comp;
	struct fi_cq_err_entry err;
	uint64_t deadline;
	int i, ret;

	if (close_side != CLOSE_TARGET) {
		/*
		 * Initiator-close with writedata: the target must drain
		 * remote CQ entries generated by fi_writedata, otherwise
		 * the device RX CQ can overflow and cause QP errors on
		 * the initiator.
		 *
		 * The actual drain happens via flush_rxcq() calls in
		 * the iteration loop in run_mr_abort_test(). Here we just do
		 * the pre-sync so the target is ready before the initiator
		 * starts posting operations.
		 */
		if (opts.rma_op == FT_RMA_WRITEDATA) {
			ret = ft_sync();
			if (ret)
				return ret;
		}
		return 0;
	}

	/* Wait for the single go signal */
	deadline = ft_gettime_ms() + CQ_TIMEOUT_MS;
	while (ft_gettime_ms() < deadline) {
		ret = fi_cq_read(rxcq, &comp, 1);
		if (ret > 0)
			break;
		else if (ret == -FI_EAVAIL) {
			memset(&err, 0, sizeof(err));
			fi_cq_readerr(rxcq, &err, 0);
		} else if (ret < 0 && ret != -FI_EAGAIN) {
			FT_PRINTERR("fi_cq_read (target rx)", ret);
			return ret;
		}
	}

	/* Close all MRs as fast as possible */
	build_close_order(num_mrs);
	for (i = 0; i < num_mrs; i++) {
		int idx = close_order[i];

		if (!slots[idx].mr)
			continue;
		ret = fi_close(&slots[idx].mr->fid);
		if (ret) {
			FT_ERR("Server MR close failed for slot %d:", idx);
			FT_PRINTERR("fi_close(mr)", ret);
			return ret;
		}
		slots[idx].mr = NULL;
		slots[idx].mr_closed = 1;
	}

	return 0;
}

/*
 * Test 2: Partial close
 *
 * Register 2 MRs on the same buffer. Post 1 write with each MR.
 * Close only the first MR. Verify: one op errors, the other completes.
 * Only runs on the initiator side.
 */
static int run_partial_close_client(void)
{
	struct mr_slot extra_slot = {0};
	struct fi_rma_iov local_iov, remote_iov;
	struct expected_err partial_errs[] = {
		{ .err = FI_ECANCELED, .prov_errno = 1 },    /* device flush */
		{ .err = FI_ECANCELED, .prov_errno = 4100 },  /* RDM pkt post fail */
		{ .err = FI_EINVAL, .prov_errno = 5 },        /* local MR invalid */
		{ .err = FI_EINVAL, .prov_errno = 7 },        /* remote MR invalid */
	};
	int i, completed_ok = 0, completed_err = 0, completed;
	int missing;
	int ret;

	/* Use slot 0's buffer for both MRs */
	extra_slot.buf = slots[0].buf;
	extra_slot.key = MR_ABORT_KEY_BASE + num_mrs; /* unique key */

	ret = ft_reg_mr(fi, extra_slot.buf, opts.transfer_size,
			ft_info_to_mr_access(fi), extra_slot.key, opts.iface,
			opts.device, &extra_slot.mr, &extra_slot.desc);
	if (ret) {
		FT_PRINTERR("ft_reg_mr (extra)", ret);
		return ret;
	}

	/* Exchange the extra key with target */
	local_iov.key = fi_mr_key(extra_slot.mr);
	local_iov.addr =
		(fi->domain_attr->mr_mode & FI_MR_VIRT_ADDR) ?
		(uintptr_t) extra_slot.buf : 0;
	local_iov.len = opts.transfer_size;

	ret = ft_sock_send(oob_sock, &local_iov, sizeof(local_iov));
	if (ret)
		goto close_extra;
	ret = ft_sock_recv(oob_sock, &remote_iov, sizeof(remote_iov));
	if (ret)
		goto close_extra;

	/* Post write using slot 0's MR (will be closed) */
	reset_test_state();
	ret = fi_write(ep, slots[0].buf, opts.transfer_size,
		       slots[0].desc, remote_fi_addr,
		       remote_arr[0].addr, remote_arr[0].key,
		       &op_arr[0].context);
	if (ret) {
		FT_PRINTERR("fi_write (slot 0)", ret);
		goto close_extra;
	}

	/* Post write using extra MR (will survive) */
	ret = fi_write(ep, extra_slot.buf, opts.transfer_size,
		       extra_slot.desc, remote_fi_addr,
		       remote_iov.addr, remote_iov.key,
		       &op_arr[1].context);
	if (ret) {
		FT_PRINTERR("fi_write (extra)", ret);
		goto close_extra;
	}

	/* Close only the extra MR */
	ret = fi_close(&extra_slot.mr->fid);
	if (ret) {
		FT_ERR("Partial: MR close failed");
		goto close_extra;
	}
	extra_slot.mr = NULL;

	/* Drain both completions */
	missing = drain_cq(txcq, 2, partial_errs, 4);
	if (missing < 0) {
		ret = missing;
		goto close_extra;
	}

	for (i = 0; i < 2; i++) {
		if (op_arr[i].completed) {
			if (op_arr[i].status == 0)
				completed_ok++;
			else
				completed_err++;
		}
	}
	completed = completed_ok + completed_err;

	/*
	 * op_arr[0] used slots[0].mr (not closed) — must succeed.
	 * op_arr[1] used extra_slot.mr (closed) — may succeed or fail.
	 */
	fprintf(stderr, "Partial close: posted=2 ok=%d err=%d missing=%d "
	       "surviving_op=%s closed_op=%s ... ",
	       completed_ok, completed_err, 2 - completed,
	       op_arr[0].completed ?
		       (op_arr[0].status == 0 ? "ok" : "FAIL") : "missing",
	       op_arr[1].completed ?
		       (op_arr[1].status == 0 ? "ok" : "err") : "missing");

	if (completed != 2) {
		fprintf(stderr, "FAIL (missing completions)\n");
		ret = -FI_EOTHER;
	} else if (op_arr[0].status != 0) {
		fprintf(stderr, "FAIL (surviving MR op must not fail)\n");
		ret = -FI_EOTHER;
	} else {
		fprintf(stderr, "PASS\n");
		ret = 0;
	}

	/* Sync with target so it can safely close its extra MR */
	if (!ret)
		ret = ft_sync();

close_extra:
	FT_CLOSE_FID(extra_slot.mr);
	return ret;
}

static int run_partial_close_server(void)
{
	struct mr_slot extra_slot = {0};
	struct fi_rma_iov local_iov, remote_iov;
	int ret;

	/* Register an extra MR for the second write target */
	extra_slot.buf = calloc(1, opts.transfer_size);
	if (!extra_slot.buf)
		return -FI_ENOMEM;
	extra_slot.key = MR_ABORT_KEY_BASE + num_mrs;

	ret = ft_reg_mr(fi, extra_slot.buf, opts.transfer_size,
			ft_info_to_mr_access(fi), extra_slot.key, opts.iface,
			opts.device, &extra_slot.mr, &extra_slot.desc);
	if (ret) {
		FT_PRINTERR("ft_reg_mr (extra)", ret);
		free(extra_slot.buf);
		return ret;
	}

	/* Exchange the extra key with initiator */
	local_iov.key = fi_mr_key(extra_slot.mr);
	local_iov.addr = (fi->domain_attr->mr_mode & FI_MR_VIRT_ADDR) ?
		(uintptr_t) extra_slot.buf : 0;
	local_iov.len = opts.transfer_size;

	ret = ft_sock_recv(oob_sock, &remote_iov, sizeof(remote_iov));
	if (!ret)
		ret = ft_sock_send(oob_sock, &local_iov, sizeof(local_iov));
	if (ret)
		goto cleanup;

	/* Wait for initiator to finish the partial close test */
	ret = ft_sync();

cleanup:
	FT_CLOSE_FID(extra_slot.mr);
	free(extra_slot.buf);
	return ret;
}

/*
 * Test 3: Endpoint reuse after abort
 *
 * Re-register MRs, do a normal write + read round-trip.
 */
static int reuse_check_client(void)
{
	struct fi_context2 reuse_ctx;
	struct fi_cq_tagged_entry comp;
	struct fi_cq_err_entry err;
	int i, ret;

	/* Drain any residual error entries from the abort test */
	do {
		ret = fi_cq_read(txcq, &comp, 1);
		if (ret == -FI_EAVAIL) {
			memset(&err, 0, sizeof(err));
			fi_cq_readerr(txcq, &err, 0);
			fprintf(stderr, "Reuse drain: residual error %d (%s)\n",
			       err.err, fi_strerror(err.err));
		}
	} while (ret != -FI_EAGAIN);

	/* Close old MRs and re-register with both write and read access */
	for (i = 0; i < num_mrs; i++)
		FT_CLOSE_FID(slots[i].mr);

	ret = register_mrs(FI_WRITE | FI_READ);
	if (ret)
		return ret;

	ret = exchange_mr_keys();
	if (ret)
		return ret;

	/* Write */
	memset(slots[0].buf, 0xAB, opts.transfer_size);
	ret = fi_write(ep, slots[0].buf, opts.transfer_size,
		       slots[0].desc, remote_fi_addr,
		       remote_arr[0].addr, remote_arr[0].key, &reuse_ctx);
	if (ret) {
		FT_PRINTERR("fi_write (reuse)", ret);
		return ret;
	}

	do {
		ret = fi_cq_read(txcq, &comp, 1);
		if (ret == -FI_EAVAIL) {
			memset(&err, 0, sizeof(err));
			fi_cq_readerr(txcq, &err, 0);
			FT_ERR("Unexpected CQ error during reuse write:");
			FT_CQ_ERR(txcq, err, NULL, 0);
			return -err.err;
		}
	} while (ret == -FI_EAGAIN);
	if (ret < 0) {
		FT_PRINTERR("fi_cq_read (reuse write)", ret);
		return ret;
	}

	ret = ft_sync();
	if (ret)
		return ret;

	/* Read */
	memset(slots[0].buf, 0, opts.transfer_size);
	ret = fi_read(ep, slots[0].buf, opts.transfer_size,
		      slots[0].desc, remote_fi_addr,
		      remote_arr[0].addr, remote_arr[0].key, &reuse_ctx);
	if (ret) {
		FT_PRINTERR("fi_read (reuse)", ret);
		return ret;
	}

	do {
		ret = fi_cq_read(txcq, &comp, 1);
		if (ret == -FI_EAVAIL) {
			memset(&err, 0, sizeof(err));
			fi_cq_readerr(txcq, &err, 0);
			FT_ERR("Unexpected CQ error during reuse read:");
			FT_CQ_ERR(txcq, err, NULL, 0);
			return -err.err;
		}
	} while (ret == -FI_EAGAIN);
	if (ret < 0) {
		FT_PRINTERR("fi_cq_read (reuse read)", ret);
		return ret;
	}

	fprintf(stderr, "Reuse: write ok, read ok ... PASS\n");
	return 0;
}

static int reuse_check_server(void)
{
	int i, ret;

	/* Close existing MRs and re-register with both read+write access */
	for (i = 0; i < num_mrs; i++)
		FT_CLOSE_FID(slots[i].mr);

	ret = register_mrs(FI_REMOTE_WRITE | FI_REMOTE_READ);
	if (ret)
		return ret;

	ret = exchange_mr_keys();
	if (ret)
		return ret;

	/* Sync after initiator's write */
	ret = ft_sync();
	if (ret)
		return ret;

	/* Initiator does read, no sync needed — target just keeps MRs alive */
	return 0;
}

/*
 * Test 4: Send/Tagged abort
 *
 * Initiator fills TX queue with fi_send/fi_tsend, then closes sender MRs.
 * Target pre-posts fi_recv/fi_trecv. If target-close, target closes
 * its recv MRs instead.
 */
static int run_send_abort_client(int iter)
{
	int i, mr_idx, op_idx, ret;
	int total_posted, mrs_used;
	int completed_ok, completed_err, missing;
	const char *mode_str = (test_mode == TEST_TAGGED) ? "tagged" : "send";

	reset_test_state();

	total_posted = 0;
	mrs_used = 0;
	op_idx = 0;

	/* Sync so target has recvs posted before we start sending */
	ret = ft_sync();
	if (ret)
		return ret;

	/* Fill TX queue with sends */
	for (mr_idx = 0; mr_idx < num_mrs; mr_idx++) {
		unsigned int posted_this_mr = 0;
		int eagain = 0;

		for (i = 0; i < ops_per_mr; i++) {
			ret = post_send_op(op_idx, mr_idx);
			if (ret == -FI_EAGAIN) {
				eagain = 1;
				break;
			}
			if (ret) {
				FT_PRINTERR("post_send_op", ret);
				return ret;
			}
			posted_this_mr++;
			op_idx++;
			total_posted++;
		}
		if (posted_this_mr > 0) {
			slots[mr_idx].posted = posted_this_mr;
			mrs_used++;
		}
		if (eagain)
			break;
	}

	if (total_posted == 0) {
		FT_ERR("could not post any send operations");
		return -FI_EINVAL;
	}

	/* Close sender MRs (initiator mode) */
	if (close_side == CLOSE_INITIATOR) {
		int close_failures = 0;

		build_close_order(mrs_used);
		for (i = 0; i < mrs_used; i++) {
			int idx = close_order[i];

			ret = fi_close(&slots[idx].mr->fid);
			if (ret)
				close_failures++;
			slots[idx].mr = NULL;
			slots[idx].mr_closed = 1;
		}
		if (close_failures) {
			FT_ERR("MR close: %d/%d failed", close_failures, mrs_used);
			return -FI_EOTHER;
		}
	}

	/* Drain TX CQ */
	if (close_side == CLOSE_INITIATOR) {
		struct expected_err send_initiator_errs[] = {
			{ .err = FI_ECANCELED, .prov_errno = 1 },     /* device flush */
			{ .err = FI_ECANCELED, .prov_errno = 4100 },  /* RDM pkt post fail */
			{ .err = FI_EINVAL, .prov_errno = 5 },        /* local MR invalid */
			{ .err = FI_EINVAL, .prov_errno = 7 },        /* long-read source-MR cancel (peer abort), with REMOTE_ERROR_BAD_ADDRESS (7) */
		};
		missing = drain_cq(txcq, total_posted,
				   send_initiator_errs, 4);
	} else {
		/* Target-close: initiator didn't close any MRs, no errors expected */
		missing = drain_cq(txcq, total_posted, NULL, 0);
	}

	completed_ok = 0;
	completed_err = 0;
	for (i = 0; i < total_posted; i++) {
		if (op_arr[i].completed) {
			if (op_arr[i].status == 0)
				completed_ok++;
			else
				completed_err++;
		}
	}

	fprintf(stderr, "Iteration %d: mode=%s size=%zu posted=%d mrs=%d "
	       "ok=%d err=%d missing=%d side=%s ... %s\n",
	       iter, mode_str, opts.transfer_size, total_posted,
	       mrs_used, completed_ok, completed_err,
	       missing, side_str(),
	       missing == 0 ? "PASS" : "FAIL");

	return missing == 0 ? 0 : -FI_EOTHER;
}

/*
 * Server side for send/tagged abort.
 *
 * Pre-posts receives until EAGAIN. In target-close mode, closes all
 * recv MRs after syncing with the initiator, then drains rxcq. In
 * initiator-close mode, just drains rxcq to free RQ space (accepts
 * errors caused
 * by sender's MR close on the RDM read-back path).
 */
static int run_send_abort_server(int iter)
{
	int i, mr_idx, op_idx, ret;
	int total_posted, mrs_used;
	int missing;

	total_posted = 0;
	mrs_used = 0;
	op_idx = 0;

	/*
	 * Pre-post receives until EAGAIN.
	 *
	 * Only needed to seed the RQ on the first iteration. On later
	 * iterations the buffers are kept populated by the
	 * repost-on-completion path in the drain loop below, so we must
	 * NOT bulk-post again: those buffers may still back live recvs
	 * the provider holds (e.g. entries silently re-queued into the
	 * SRX by the peer-abort recovery path), and re-posting them here
	 * would leave two SRX entries aliasing the same buffer/context.
	 */
	if (iter == 1) {
		for (mr_idx = 0; mr_idx < num_mrs; mr_idx++) {
			unsigned int posted_this_mr = 0;
			int eagain = 0;

			for (i = 0; i < ops_per_mr; i++) {
				ret = post_recv_op(op_idx, mr_idx);
				if (ret == -FI_EAGAIN) {
					eagain = 1;
					break;
				}
				if (ret) {
					FT_PRINTERR("post_recv_op", ret);
					return ret;
				}
				posted_this_mr++;
				op_idx++;
				total_posted++;
			}
			if (posted_this_mr > 0) {
				slots[mr_idx].posted = posted_this_mr;
				mrs_used++;
			}
			if (eagain)
				break;
		}

		fprintf(stderr,
			"[MRABORT-SRVPOST] iter=%d posted_recvs=%d mrs_used=%d hit_eagain_at_mr=%d\n",
			iter, total_posted, mrs_used,
			(mr_idx < num_mrs) ? mr_idx : -1);
	}

	/* Sync to let initiator start sending */
	ret = ft_sync();
	if (ret)
		return ret;

	/* Close recv MRs (target mode) */
	if (close_side == CLOSE_TARGET) {
		build_close_order(mrs_used);
		for (i = 0; i < mrs_used; i++) {
			int idx = close_order[i];

			ret = fi_close(&slots[idx].mr->fid);
			if (ret) {
				FT_ERR("Server MR close failed for slot %d:", idx);
				FT_PRINTERR("fi_close(mr)", ret);
				return ret;
			}
			slots[idx].mr = NULL;
			slots[idx].mr_closed = 1;
		}
	}

	/* Drain RX CQ */
	if (close_side == CLOSE_TARGET) {
		missing = drain_cq(rxcq, total_posted, NULL, -1);

		fprintf(stderr, "Server iter %d: recvs=%d missing=%d ... %s\n",
		       iter, total_posted, missing,
		       missing == 0 ? "PASS" : "FAIL");

		return missing == 0 ? 0 : -FI_EOTHER;
	}

	/*
	 * Initiator-close: drain whatever completions arrived to free
	 * RQ space. All recv completions should be successful — the
	 * target didn't close any MRs.
	 */
	struct fi_cq_tagged_entry comp;
	struct fi_cq_err_entry err;
	struct op_ctx *o;
	int repost_idx;

	for (;;) {
		ret = fi_cq_read(rxcq, &comp, 1);
		if (ret > 0) {
			/*
			 * A recv completed successfully, so its buffer is
			 * owned by the application again. Immediately re-post
			 * that buffer to the device before continuing. Only a
			 * buffer whose recv reached a terminal completion may
			 * be re-posted: re-posting a buffer whose recv is still
			 * live in the provider (e.g. one silently re-queued
			 * into the SRX by the peer-abort recovery path) leaves
			 * two SRX entries aliasing the same buffer/context.
			 */
			o = container_of(comp.op_context, struct op_ctx,
					 context);
			repost_idx = (int) (o - op_arr);
			ret = post_recv_op(repost_idx, o->mr_idx);
			if (ret && ret != -FI_EAGAIN) {
				FT_PRINTERR("post_recv_op (repost)", ret);
				return ret;
			}
			continue;
		} else if (ret == -FI_EAVAIL) {
			memset(&err, 0, sizeof(err));
			fi_cq_readerr(rxcq, &err, 0);
			FT_ERR("Unexpected target recv error:");
			FT_CQ_ERR(rxcq, err, NULL, 0);
			return -FI_EOTHER;
		} else {
			break;
		}
	}

	return 0;
}

/*
 * Top-level per-endpoint flow, shared by initiator and target.
 *
 * Switches on test_mode and, for each iteration, registers MRs (and
 * exchanges keys for RMA modes), runs the relevant initiator/target
 * test function, then syncs. Send/tagged modes need no key exchange.
 * After all iterations pass, RMA modes run the endpoint reuse check.
 *
 * The initiator posts the operations; the target holds the remote MRs
 * and, in target-close mode, closes them when signaled.
 */
static int run_mr_abort_test(int is_initiator)
{
	int i, ret = 0;

	for (i = 0; i < opts.iterations; i++) {
		switch (test_mode) {
		case TEST_ABORT:
		case TEST_PARTIAL:
			ret = register_mrs(ft_info_to_mr_access(fi));
			if (!ret)
				ret = exchange_mr_keys();
			break;
		case TEST_SEND:
		case TEST_TAGGED:
			ret = register_mrs(is_initiator ? FI_SEND : FI_RECV);
			break;
		}
		if (ret)
			return ret;

		switch (test_mode) {
		case TEST_ABORT:
			if (is_initiator) {
				ret = run_fill_abort_client(i + 1);
			} else {
				ret = run_fill_abort_server();
				if (!ret)
					ret = flush_rxcq();
			}
			break;
		case TEST_PARTIAL:
			ret = is_initiator ? run_partial_close_client() :
					     run_partial_close_server();
			break;
		case TEST_SEND:
		case TEST_TAGGED:
			ret = is_initiator ? run_send_abort_client(i + 1) :
					     run_send_abort_server(i + 1);
			break;
		}
		if (ret)
			return ret;

		ret = ft_sync();
		if (ret)
			return ret;
	}

	/* Endpoint reuse check — only for RMA tests */
	if (test_mode == TEST_ABORT || test_mode == TEST_PARTIAL) {
		ret = is_initiator ? reuse_check_client() :
				     reuse_check_server();
		if (!ret)
			ret = ft_sync();
	}

	return ret;
}

static int run(void)
{
	int ret;

	/* writedata generates remote CQ entries with immediate data */
	if (close_side == CLOSE_TARGET || opts.rma_op == FT_RMA_WRITEDATA)
		cq_attr.format = FI_CQ_FORMAT_DATA;

	if (hints->ep_attr->type == FI_EP_MSG)
		ret = ft_init_fabric_cm();
	else
		ret = ft_init_fabric();
	if (ret)
		return ret;

	ret = alloc_test_res();
	if (ret)
		return ret;

	ret = run_mr_abort_test(opts.dst_addr != NULL);

	/*
	 * Server teardown order: EFA requires the endpoint to be closed
	 * before recv buffer MRs can be deregistered. Use -A ep_first
	 * to close the EP before freeing test MRs.
	 */
	if (close_ep_first && !opts.dst_addr)
		FT_CLOSE_FID(ep);

	if (!ret)
		ret = free_test_res();
	else
		free_test_res();

	if (!ret)
		ft_finalize();

	return ret;
}

int main(int argc, char **argv)
{
	int op, ret, cleanup_ret;

	opts = INIT_OPTS;
	opts.options |= FT_OPT_OOB_SYNC | FT_OPT_SKIP_MSG_ALLOC | FT_OPT_SIZE;
	opts.transfer_size = 4096; /* 4KB default — override with -S */
	opts.iterations = 10;
	opts.cqdata_op = 0;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	srand(time(NULL));

	while ((op = getopt(argc, argv,
			    "W:N:C:R:T:A:h" CS_OPTS INFO_OPTS API_OPTS)) != -1) {
		switch (op) {
		case 'W':
			num_mrs = atoi(optarg);
			break;
		case 'N':
			ops_per_mr = atoi(optarg);
			break;
		case 'C':
			if (!strcmp(optarg, "reverse"))
				close_order_mode = CLOSE_ORDER_REVERSE;
			else if (!strcmp(optarg, "random"))
				close_order_mode = CLOSE_ORDER_RANDOM;
			else {
				FT_ERR("Unknown close order: %s", optarg);
				return EXIT_FAILURE;
			}
			break;
		case 'R':
			if (!strcmp(optarg, "initiator"))
				close_side = CLOSE_INITIATOR;
			else if (!strcmp(optarg, "target"))
				close_side = CLOSE_TARGET;
			else {
				FT_ERR("Unknown close side: %s", optarg);
				return EXIT_FAILURE;
			}
			break;
		case 'T':
			if (!strcmp(optarg, "abort"))
				test_mode = TEST_ABORT;
			else if (!strcmp(optarg, "partial"))
				test_mode = TEST_PARTIAL;
			else if (!strcmp(optarg, "send"))
				test_mode = TEST_SEND;
			else if (!strcmp(optarg, "tagged"))
				test_mode = TEST_TAGGED;
			else {
				FT_ERR("Unknown test mode: %s", optarg);
				return EXIT_FAILURE;
			}
			break;
		case 'A':
			if (!strcmp(optarg, "ep_first"))
				close_ep_first = 1;
			else if (!strcmp(optarg, "mr_first"))
				close_ep_first = 0;
			else {
				FT_ERR("Unknown teardown order: %s", optarg);
				return EXIT_FAILURE;
			}
			break;
		default:
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			ret = ft_parse_api_opts(op, optarg, hints, &opts);
			if (ret)
				return ret;
			break;
		case '?':
		case 'h':
			ft_csusage(argv[0],
				"Test aborting in-flight operations by "
				"closing MRs.");
			FT_PRINT_OPTS_USAGE("-T <test>",
				"Test mode: abort|partial|send|tagged "
				"(default: abort)");
			FT_PRINT_OPTS_USAGE("-o <op>",
				"RMA op: write|read|writedata "
				"(default: write)");
			FT_PRINT_OPTS_USAGE("-W <count>",
				"Number of MR/buffer pairs "
				"(default: 8192)");
			FT_PRINT_OPTS_USAGE("-N <count>",
				"Operations per MR before close "
				"(default: 1)");
			FT_PRINT_OPTS_USAGE("-C <mode>",
				"MR close order: reverse|random "
				"(default: reverse)");
			FT_PRINT_OPTS_USAGE("-R <side>",
				"Which side closes MRs: "
				"initiator|target (default: initiator)");
			FT_PRINT_OPTS_USAGE("-A <order>",
				"Server teardown order: ep_first|mr_first "
				"(default: mr_first)");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	if (num_mrs <= 0 || ops_per_mr <= 0) {
		FT_ERR("-W and -N must be positive");
		return EXIT_FAILURE;
	}

	/*
	 * Target-close for send/tagged would require canceling posted
	 * receive buffers when the receiver closes its recv MRs
	 * mid-flight. EFA has no supported primitive to abort an
	 * in-flight posted receive, so this combination is unsupported.
	 * Fail explicitly rather than run the stale/incorrect path.
	 */
	if (close_side == CLOSE_TARGET &&
	    (test_mode == TEST_SEND || test_mode == TEST_TAGGED)) {
		FT_ERR("Target-close (-R target) is not supported for "
		       "send/tagged (-T send|tagged): canceling posted "
		       "receive buffers is unsupported");
		return EXIT_FAILURE;
	}

	/* Partial test only uses slot 0 + one extra MR */
	if (test_mode == TEST_PARTIAL)
		num_mrs = 1;

	hints->caps = FI_MSG;
	switch (test_mode) {
	case TEST_ABORT:
	case TEST_PARTIAL:
		hints->caps |= FI_RMA;
		break;
	case TEST_TAGGED:
		hints->caps |= FI_TAGGED;
		break;
	case TEST_SEND:
		break;
	}
	hints->mode = FI_CONTEXT | FI_CONTEXT2;
	hints->domain_attr->mr_mode = opts.mr_mode;
	hints->domain_attr->resource_mgmt = FI_RM_ENABLED;
	hints->addr_format = opts.address_format;

	/*
	 * For writedata, fabtests' ft_getinfo() auto-adds FI_RX_CQ_DATA to
	 * hints->mode because opts.cqdata_op is set. On efa-direct, requesting
	 * FI_RX_CQ_DATA disables the device's unsolicited write recv feature
	 * (see efa_base_ep_create_qp), which then requires the target to
	 * pre-post a recv buffer for every incoming write-with-immediate.
	 * This test's target side posts no recvs (it only polls the rxcq for
	 * the imm completion), so suppress the auto FI_RX_CQ_DATA request and
	 * rely on unsolicited write recv to deliver the imm data to the CQ.
	 * cq_attr.format = FI_CQ_FORMAT_DATA (set in run()) still delivers the
	 * immediate data without FI_RX_CQ_DATA.
	 */
	allow_rx_cq_data = false;

	ret = run();

	cleanup_ret = ft_free_res();
	return ft_exit_code(ret ? ret : cleanup_ret);
}
