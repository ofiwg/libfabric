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
#include <limits.h>
#include <errno.h>

#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_ext.h>

#include "shared.h"
#include "hmem.h"

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
	int completions;	/* must end == 1 for every posted op */
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

/*
 * Abort-class CQ errors shared by the RMA abort and partial tests. Which
 * set is "expected" depends on which side tears its MR down while ops are
 * in flight:
 *
 *   local close  - the INITIATOR closes its own source MR while ops are in
 *                  flight: initiator-close abort, and the partial test
 *                  (which closes one of two local MRs aliasing the same
 *                  buffer). Every error is local-origin.
 *   remote close - the TARGET closes the remote MR (target-close abort).
 *                  The op that lands on the invalidated MR takes a remote
 *                  access error (remote MR invalid).
 */
static struct expected_err mr_abort_local_close_errs[] = {
	{ .err = FI_ECANCELED, .prov_errno = 1 },     /* device flush */
	{ .err = FI_ECANCELED, .prov_errno = 4100 },  /* RDM pkt post failure */
	{ .err = FI_ECANCELED, .prov_errno = 4127 },  /* Peer Aborted */
	{ .err = FI_EINVAL,    .prov_errno = 5 },     /* local MR invalid */
};

static struct expected_err mr_abort_remote_close_errs[] = {
	{ .err = FI_EINVAL,    .prov_errno = 7 },     /* remote MR invalid */
	{ .err = FI_ECANCELED, .prov_errno = 1 },     /* device flush */
};

static struct mr_slot *slots;
static struct op_ctx *op_arr;
static int *close_order;
static int close_order_len;	/* valid entries in close_order[] (last build) */
static int num_mrs = 8192;
static int ops_per_mr = 1;
static enum close_order_mode close_order_mode = CLOSE_ORDER_REVERSE;
static enum close_side close_side = CLOSE_INITIATOR;
static enum test_mode test_mode = TEST_ABORT;
static int close_ep_first;
static int set_homogeneous_peers;

/*
 * -r <file>: replay a previously dumped close order instead of generating
 * one. Overrides -C reverse|random. Set on the close side to reproduce a
 * specific failing close sequence (see dump_close_order).
 */
static const char *close_order_replay_path;

/*
 * Whether an aborted send/tagged operation is owed a terminal recv
 * completion on the target (-X). True for protocols where the receiver
 * matches and takes partial data before the abort (LONGCTS, RUNTREAD
 * with a tail READ / LONGREAD): the matched rxe must complete with a
 * clean FI_ECANCELED. False for EAGER and MEDIUM: an aborted message
 * delivered nothing the receiver must complete (a stray FI_ECANCELED
 * may still arrive and is accepted, but is never required). Only
 * meaningful for -T send|tagged.
 */
static int abort_owes_rx_completion;

/* Remote side MR info */
static struct fi_rma_iov *remote_arr;

#define MR_ABORT_KEY_BASE 0x1000

/*
 * Timeout-based CQ drain budget, used ONLY by the send/tagged target RX
 * drain in run_send_abort_target() for the INDETERMINATE case (EAGER /
 * MEDIUM / runt-only without -X, where an aborted send may or may not yield
 * a recv completion, so the exact reaped count cannot be known in advance).
 * Allow up to CQ_FIRST_TIMEOUT_MS for the first completion to appear
 * (covers device-flush latency after an MR close, or the peer starting to
 * send/signal), then only CQ_IDLE_TIMEOUT_MS between consecutive
 * completions: the deadline is seeded to "first" and reset to "idle" on
 * every reaped completion. A drain is declared finished once the CQ has
 * stayed idle for one CQ_IDLE_TIMEOUT_MS window.
 *
 * Every other drain knows the exact required completion count and blocks on
 * it instead: the TX side via drain_cq_counted(), and the target RX drain's
 * slack == 0 branch (the -X owed protocols, where required is guaranteed).
 */
#define CQ_FIRST_TIMEOUT_MS 10000
#define CQ_IDLE_TIMEOUT_MS  1000

/*
 * Non-blocking accumulate of `len` bytes from the OOB socket into `buf`.
 * *got tracks bytes received so far across calls. Returns 1 once all
 * `len` bytes have arrived, 0 if still waiting (would block), or a
 * negative error. Lets the caller keep draining the RX CQ while the
 * peer is still sending instead of blocking on the count.
 */
static int oob_recv_nonblock(int fd, void *buf, size_t len, size_t *got)
{
	ssize_t ret;

	while (*got < len) {
		ret = ofi_recv_socket(fd, (char *) buf + *got, len - *got,
				      MSG_DONTWAIT);
		if (ret > 0) {
			*got += ret;
		} else if (ret == 0) {
			return -FI_ENOTCONN;
		} else if (ofi_sockerr() == EAGAIN ||
			   ofi_sockerr() == EWOULDBLOCK) {
			return 0; /* nothing available right now */
		} else {
			return -ofi_sockerr();
		}
	}
	return 1; /* full value received */
}

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
	int i, ret;

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
		/*
		 * One MR per slot, but each of the ops_per_mr operations on
		 * that slot must land in its own buffer range -- N recvs may
		 * share one memory region, but must NOT share one buffer (that
		 * would alias N concurrent recvs onto the same memory). So size
		 * the slot buffer to hold ops_per_mr * transfer_size and give
		 * each op a distinct transfer_size slice (see slot_op_buf()).
		 */
		ret = ft_hmem_alloc(opts.iface, opts.device,
				    (void **) &slots[i].buf,
				    (size_t) ops_per_mr * opts.transfer_size);
		if (ret)
			return ret;
		ret = ft_hmem_memset(opts.iface, opts.device, slots[i].buf,
				     0, (size_t) ops_per_mr * opts.transfer_size);
		if (ret)
			return ret;
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
			if (slots[i].buf) {
				ft_hmem_free(opts.iface, slots[i].buf);
				slots[i].buf = NULL;
			}
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

		ret = ft_reg_mr(fi, slots[i].buf,
				(size_t) ops_per_mr * opts.transfer_size,
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
		op_arr[i].completions = 0;
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

/*
 * Return the distinct transfer_size buffer slice for one operation within
 * its MR. ops_per_mr operations share a single MR (one registration), but
 * each must use its own buffer range: the within-MR op index is
 * (op_idx - mr_idx * ops_per_mr) because the fill loops post op_idx in
 * lockstep as mr_idx * ops_per_mr + i.
 */
static char *slot_op_buf(int op_idx, int mr_idx)
{
	int op_in_mr = op_idx - (mr_idx * ops_per_mr);

	assert(op_in_mr >= 0 && op_in_mr < ops_per_mr);
	return slots[mr_idx].buf + (size_t) op_in_mr * opts.transfer_size;
}

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
	char *buf = slot_op_buf(op_idx, mr_idx);

	o->mr_idx = mr_idx;

	if (test_mode == TEST_TAGGED)
		return fi_tsend(ep, buf, opts.transfer_size, s->desc,
				remote_fi_addr, 0xCAFE, &o->context);
	else
		return fi_send(ep, buf, opts.transfer_size, s->desc,
			       remote_fi_addr, &o->context);
}

static ssize_t post_recv_op(int op_idx, int mr_idx)
{
	struct mr_slot *s = &slots[mr_idx];
	struct op_ctx *o = &op_arr[op_idx];
	char *buf = slot_op_buf(op_idx, mr_idx);

	o->mr_idx = mr_idx;

	if (test_mode == TEST_TAGGED)
		return fi_trecv(ep, buf, opts.transfer_size, s->desc,
				remote_fi_addr, 0xCAFE, 0, &o->context);
	else
		return fi_recv(ep, buf, opts.transfer_size, s->desc,
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

static const char *close_order_str(void);

/*
 * On failure, persist the close order that triggered it so it can be
 * replayed with -r. Writes a unique /tmp file (one MR slot index per line,
 * '#' header comment) and logs the path. When the run is itself a replay
 * (-r) the order is already on disk, so we just point back at that file.
 *
 * Only called when a test fails -- especially useful for -C random, whose
 * order differs every run.
 */
static void dump_close_order(void)
{
	char path[] = "/tmp/fi_mr_abort_close_order-XXXXXX";
	FILE *f;
	int fd, i;

	if (close_order_len <= 0)
		return;

	if (close_order_replay_path) {
		fprintf(stderr,
			"close order (%s, %d MRs) reproduced from %s\n",
			close_order_str(), close_order_len,
			close_order_replay_path);
		return;
	}

	fd = mkstemp(path);
	if (fd < 0) {
		FT_ERR("could not create close-order file (replay disabled): %s",
		       strerror(errno));
		return;
	}
	f = fdopen(fd, "w");
	if (!f) {
		FT_ERR("could not open close-order file '%s': %s", path,
		       strerror(errno));
		close(fd);
		return;
	}

	fprintf(f, "# fi_mr_abort close order: %s, %d MRs\n",
		close_order_str(), close_order_len);
	for (i = 0; i < close_order_len; i++)
		fprintf(f, "%d\n", close_order[i]);
	fclose(f);

	fprintf(stderr,
		"close order (%s, %d MRs) written to %s -- reproduce with -r %s\n",
		close_order_str(), close_order_len, path, path);
}

/*
 * Load a close order previously dumped by dump_close_order. Reads exactly n
 * slot indices, skipping blank lines and '#' comments; every index must be
 * unique and in [0, n) so the result is a valid permutation. A length
 * mismatch usually means the replay file was produced with different
 * -W/-N/-S settings than the current run.
 */
static int read_close_order(int n)
{
	FILE *f;
	char line[64];
	char *seen;
	int count = 0, idx;

	f = fopen(close_order_replay_path, "r");
	if (!f) {
		FT_ERR("could not open replay file '%s': %s",
		       close_order_replay_path, strerror(errno));
		return -FI_EINVAL;
	}

	seen = calloc(n, 1);
	if (!seen) {
		fclose(f);
		return -FI_ENOMEM;
	}

	while (count < n && fgets(line, sizeof(line), f)) {
		if (line[0] == '#' || line[0] == '\n' || line[0] == '\r' ||
		    line[0] == '\0')
			continue;
		idx = atoi(line);
		if (idx < 0 || idx >= n || seen[idx]) {
			FT_ERR("replay file '%s': invalid or duplicate index "
			       "%d (need unique values in [0,%d))",
			       close_order_replay_path, idx, n);
			free(seen);
			fclose(f);
			return -FI_EINVAL;
		}
		seen[idx] = 1;
		close_order[count++] = idx;
	}
	fclose(f);
	free(seen);

	if (count != n) {
		FT_ERR("replay file '%s': got %d indices, expected %d "
		       "(do -W/-N/-S match the run that produced it?)",
		       close_order_replay_path, count, n);
		return -FI_EINVAL;
	}

	return 0;
}

static int build_close_order(int n)
{
	int i, tmp;

	if (close_order_replay_path) {
		/* Replay overrides the reverse/random generator. */
		int ret = read_close_order(n);

		if (ret)
			return ret;
		close_order_len = n;
		return 0;
	}

	for (i = 0; i < n; i++)
		close_order[i] = i;

	switch (close_order_mode) {
	case CLOSE_ORDER_REVERSE:
		for (i = 0; i < n / 2; i++) {
			tmp = close_order[i];
			close_order[i] = close_order[n - 1 - i];
			close_order[n - 1 - i] = tmp;
		}
		break;
	case CLOSE_ORDER_RANDOM:
		shuffle(close_order, n);
		break;
	}

	close_order_len = n;
	return 0;
}

static int is_expected_err(struct fi_cq_err_entry *err,
			   struct expected_err *list, int count)
{
	int i;

	if (count < 0)
		return 1; /* negative count = accept anything */

	for (i = 0; i < count; i++) {
		/*
		 * A negative prov_errno in the list is a wildcard: match on
		 * the libfabric err code alone, ignoring the protocol/path-
		 * specific prov_errno. Existing exact-match callers pass
		 * non-negative prov_errno and are unaffected.
		 */
		if (err->err == list[i].err &&
		    (list[i].prov_errno < 0 ||
		     err->prov_errno == list[i].prov_errno))
			return 1;
	}
	return 0;
}

/*
 * Counted CQ drain for the TX-side abort/partial tests.
 *
 * Unlike the timeout-based target RX drain, this uses NO idle/first-
 * completion timeout. The initiator-side abort and partial tests post a known, exact number of
 * operations and every one of them is required to produce a terminal
 * completion (success or an expected abort-class error) -- there is no
 * straggler/slack ambiguity. Rather than guessing "the CQ has gone idle, so
 * the rest must be missing" with a short window -- which can falsely fail
 * when a fast local completion (e.g. a just-closed MR's device-flush error)
 * races ahead of a slower but still-valid completion arriving over the wire
 * -- we simply block until all `expected` completions have been reaped.
 *
 * Returns 0 once every expected completion has arrived, or a negative error
 * on an unexpected CQ error / hard fi_cq_read failure (still a clean,
 * immediate return, not a hang). The only way a genuine silent drop
 * manifests is as a hang, which is bounded by the caller's overall test
 * timeout (the pytest ClientServerTest timeout).
 */
static int drain_cq_counted(struct fid_cq *cq, int expected,
			    struct expected_err *err_list, int err_count)
{
	struct fi_cq_tagged_entry comp;
	struct fi_cq_err_entry err;
	struct op_ctx *o;
	int idx, remaining, ret;

	remaining = expected;

	while (remaining > 0) {
		ret = fi_cq_read(cq, &comp, 1);
		if (ret > 0) {
			o = container_of(comp.op_context,
					 struct op_ctx, context);
			/*
			 * Each posted op submits exactly one op_context and is
			 * owed exactly one terminal completion. Validate the
			 * context resolves to a posted op and that this is its
			 * first completion: a second completion for the same op
			 * (or a context outside the posted range) is a provider
			 * contract violation. Counting it toward `remaining`
			 * would mask a genuinely missing completion, so fail.
			 */
			idx = (int) (o - op_arr);
			if (idx < 0 || idx >= expected) {
				FT_ERR("TX completion for out-of-range "
				       "op_context (idx=%d, expected < %d)",
				       idx, expected);
				return -FI_EOTHER;
			}
			if (++o->completions > 1) {
				FT_ERR("op idx=%d tied to %d completions "
				       "(expected 1)", idx, o->completions);
				return -FI_EOTHER;
			}
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
				idx = (int) (o - op_arr);
				if (idx < 0 || idx >= expected) {
					FT_ERR("TX error completion for "
					       "out-of-range op_context "
					       "(idx=%d, expected < %d)",
					       idx, expected);
					return -FI_EOTHER;
				}
				if (++o->completions > 1) {
					FT_ERR("op idx=%d tied to %d "
					       "completions (expected 1)",
					       idx, o->completions);
					return -FI_EOTHER;
				}
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

	return 0;
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

/*
 * Fill-and-abort (RMA) -- initiator.
 *
 * Phases:
 *   1. Target-close only: post a single 0-byte write-with-imm as the
 *      "go" signal that tells the target to start closing its MRs.
 *   2. writedata + initiator-close only: sync so the target is already
 *      draining its rxcq before we flood it.
 *   3. Fill the TX queue with RMA ops until -FI_EAGAIN.
 *   4. Initiator-close only: close the local MRs in the configured
 *      cancel order to abort the in-flight ops.
 *   5. Drain the TX CQ, accepting the expected abort errors, and pass if
 *      every posted op produced a completion.
 */
static int run_fill_abort_initiator(int iter)
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
		 * Use op_arr[0] as context so drain_cq_counted's container_of
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

		ret = build_close_order(mrs_used);
		if (ret)
			return ret;
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
	if (close_side == CLOSE_TARGET)
		ret = drain_cq_counted(txcq, total_posted,
				       mr_abort_remote_close_errs,
				       ARRAY_SIZE(mr_abort_remote_close_errs));
	else
		ret = drain_cq_counted(txcq, total_posted,
				       mr_abort_local_close_errs,
				       ARRAY_SIZE(mr_abort_local_close_errs));

	if (ret != 0)
		return ret; /* drain_cq_counted hit unexpected error */

	/* Report */
	completed_ok = 0;
	completed_err = 0;
	for (i = 0; i < total_posted; i++) {
		if (op_arr[i].completions) {
			if (op_arr[i].status == 0)
				completed_ok++;
			else
				completed_err++;
		}
	}

	/*
	 * The TX side is deterministic: every posted op must reach exactly one
	 * terminal completion. drain_cq_counted() already blocked until
	 * total_posted CQ entries were read, so a nonzero "missing" here means
	 * the per-op tally disagrees with the raw count -- i.e. an op_context
	 * was completed more than once (or a stale completion aliased a reused
	 * op_arr slot).
	 */
	missing = total_posted - (completed_ok + completed_err);

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
 * Fill-and-abort (RMA) -- target.
 *
 * Initiator-close mode: no MRs to close here. For writedata, pre-sync so
 * the target is ready before the initiator floods (the actual rxcq drain
 * happens via flush_rxcq() in run_mr_abort_test()); otherwise return
 * immediately.
 *
 * Target-close mode:
 *   1. Wait for the single go-signal write-with-imm on the rxcq.
 *   2. Close ALL remote MRs as fast as possible (in the configured
 *      cancel order), racing the initiator's in-flight writes/reads.
 *
 * No recv posting is needed -- fi_writedata is an RMA op that generates a
 * remote CQ entry directly.
 */
static int run_fill_abort_target(void)
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
	deadline = ft_gettime_ms() + CQ_FIRST_TIMEOUT_MS;
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
	ret = build_close_order(num_mrs);
	if (ret)
		return ret;
	for (i = 0; i < num_mrs; i++) {
		int idx = close_order[i];

		if (!slots[idx].mr)
			continue;
		ret = fi_close(&slots[idx].mr->fid);
		if (ret) {
			FT_ERR("Target MR close failed for slot %d:", idx);
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
static int run_partial_close_initiator(void)
{
	struct mr_slot extra_slot = {0};
	struct fi_rma_iov local_iov, remote_iov;
	int i, completed_ok = 0, completed_err = 0, completed;
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
	ret = drain_cq_counted(txcq, 2, mr_abort_local_close_errs,
			       ARRAY_SIZE(mr_abort_local_close_errs));
	if (ret != 0)
		goto close_extra;

	for (i = 0; i < 2; i++) {
		if (op_arr[i].completions) {
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
	       op_arr[0].completions == 0 ? "missing" :
	       op_arr[0].completions > 1  ? "DOUBLE"  :
		       (op_arr[0].status == 0 ? "ok" : "FAIL"),
	       op_arr[1].completions == 0 ? "missing" :
	       op_arr[1].completions > 1  ? "DOUBLE"  :
		       (op_arr[1].status == 0 ? "ok" : "err"));

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

static int run_partial_close_target(void)
{
	struct mr_slot extra_slot = {0};
	struct fi_rma_iov local_iov, remote_iov;
	int ret;

	/* Register an extra MR for the second write target */
	ret = ft_hmem_alloc(opts.iface, opts.device, (void **) &extra_slot.buf,
			    opts.transfer_size);
	if (ret)
		return ret;
	extra_slot.key = MR_ABORT_KEY_BASE + num_mrs;

	ret = ft_reg_mr(fi, extra_slot.buf, opts.transfer_size,
			ft_info_to_mr_access(fi), extra_slot.key, opts.iface,
			opts.device, &extra_slot.mr, &extra_slot.desc);
	if (ret) {
		FT_PRINTERR("ft_reg_mr (extra)", ret);
		ft_hmem_free(opts.iface, extra_slot.buf);
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
	ft_hmem_free(opts.iface, extra_slot.buf);
	return ret;
}

/*
 * Test 3: Endpoint reuse after abort
 *
 * Re-register MRs, do a normal write + read round-trip.
 */
static int reuse_check_initiator(void)
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
	ret = ft_hmem_memset(opts.iface, opts.device, slots[0].buf,
			     0xAB, opts.transfer_size);
	if (ret)
		return ret;
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
	ret = ft_hmem_memset(opts.iface, opts.device, slots[0].buf,
			     0, opts.transfer_size);
	if (ret)
		return ret;
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

static int reuse_check_target(void)
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
 * Send/tagged abort -- initiator.
 *
 * Phases (each ft_sync pairs with the matching one on the target):
 *   1. Send MRs are registered by the caller; sync so the target has its
 *      recvs posted before we transmit.
 *   2. Fill the TX queue with fi_send/fi_tsend until -FI_EAGAIN.
 *   3. Initiator-close only: close all sender MRs in the configured
 *      cancel order to abort the in-flight sends.
 *   4. Drain the TX CQ, accepting the expected abort errors.
 *   5. Compute (required, slack) from the drained results and send them
 *      to the target over OOB (see the contract in the section banner).
 *
 * Pass criteria: no silent drops -- every posted op reaches a terminal completion.
 */
static int run_send_abort_initiator(int iter)
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
	int close_failures = 0;

	ret = build_close_order(mrs_used);
	if (ret)
		return ret;
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

	/*
	 * Drain TX CQ.
	 *
	 * Every posted send is guaranteed exactly one terminal TX completion
	 * (a success, or an abort-class error) regardless of wire protocol --
	 * the TX side is never indeterminate. So use the counted drain with
	 * the known posted count rather than a timeout: a genuinely missing
	 * completion manifests as a hang bounded by the pytest timeout, never
	 * a flaky short-window failure.
	 */
	struct expected_err send_initiator_errs[] = {
		{ .err = FI_ECANCELED, .prov_errno = 1 },     /* device flush */
		{ .err = FI_ECANCELED, .prov_errno = 4100 },  /* RDM pkt post fail */
		{ .err = FI_EINVAL, .prov_errno = 5 },        /* local MR invalid */
		{ .err = FI_ECANCELED, .prov_errno = 4127 },  /* peer abort: receiver detected the yanked source MR on its RDMA read and notified us via PEER_ERROR_PKT */
	};
	ret = drain_cq_counted(txcq, total_posted, send_initiator_errs,
			       ARRAY_SIZE(send_initiator_errs));
	if (ret != 0)
		return ret; /* drain_cq_counted hit unexpected error */

	completed_ok = 0;
	completed_err = 0;
	for (i = 0; i < total_posted; i++) {
		if (op_arr[i].completions) {
			if (op_arr[i].status == 0)
				completed_ok++;
			else
				completed_err++;
		}
	}

	/*
	 * The TX side is deterministic: every posted send must reach exactly
	 * one terminal completion. drain_cq_counted() already blocked until
	 * total_posted CQ entries were read, so a nonzero "missing" here means
	 * the per-op tally disagrees with the raw count -- i.e. an op_context
	 * was completed more than once (or a stale completion aliased a reused
	 * op_arr slot).
	 */
	missing = total_posted - (completed_ok + completed_err);

	/*
	 * Send the (required, slack) completion counts to the target over
	 * OOB so it can reconcile its RX completions. See the required/slack
	 * contract in the "Test 4" section banner for what each represents.
	 * Only the initiator-close path exchanges these; the target-close
	 * path's drain does not read them, keeping the OOB stream symmetric.
	 */
	int counts[2];

	counts[0] = completed_ok +	/* required */
			(abort_owes_rx_completion ? completed_err : 0);
	counts[1] = abort_owes_rx_completion ? 0 : completed_err; /* slack */

	ret = ft_sock_send(oob_sock, counts, sizeof(counts));
	if (ret)
		return ret;

	fprintf(stderr, "Iteration %d: mode=%s size=%zu posted=%d mrs=%d "
	       "ok=%d err=%d missing=%d side=%s ... %s\n",
	       iter, mode_str, opts.transfer_size, total_posted,
	       mrs_used, completed_ok, completed_err,
	       missing, side_str(),
	       missing == 0 ? "PASS" : "FAIL");

	return missing == 0 ? 0 : -FI_EOTHER;
}

/*
 * Repost a terminally-completed recv buffer back to the RQ so it stays
 * at its seeded depth (the buffer is app-owned and MUST go back). Retries
 * -FI_EAGAIN, draining the CQ to make room, until the deadline.
 */
static int mr_abort_repost_recv(struct fid_cq *rxcq, struct op_ctx *o,
				uint64_t deadline)
{
	int repost_idx = (int) (o - op_arr);
	int ret;

	do {
		ret = post_recv_op(repost_idx, o->mr_idx);
		if (ret == -FI_EAGAIN)
			(void) fi_cq_read(rxcq, NULL, 0);
	} while (ret == -FI_EAGAIN && ft_gettime_ms() < deadline);

	if (ret && ret != -FI_EAGAIN) {
		FT_PRINTERR("post_recv_op (repost)", ret);
		return ret;
	}
	return 0;
}

/*
 * Read and process a single RX CQ entry for the send/tagged target drain.
 *
 * Returns 1 if a terminal completion was reaped (counted, and its app-owned
 * buffer reposted to keep the RQ at depth), 0 if the CQ was momentarily
 * empty (-FI_EAGAIN), or a negative error on a hard fi_cq_read failure or an
 * unexpected (non-abort) error completion. A success increments *recv_ok; an
 * accepted abort-class error (matching @p errs) increments *recv_canceled.
 *
 * The repost is given a fresh CQ_FIRST_TIMEOUT_MS budget so it succeeds even
 * in the counted (slack == 0) branch, which has no rolling idle deadline.
 */
static int target_recv_drain_one(struct expected_err *errs, int nerr,
				 int *recv_ok, int *recv_canceled)
{
	struct fi_cq_tagged_entry comp;
	struct fi_cq_err_entry err;
	struct op_ctx *o;
	int ret;

	ret = fi_cq_read(rxcq, &comp, 1);
	if (ret > 0) {
		(*recv_ok)++;
		o = container_of(comp.op_context, struct op_ctx, context);
		ret = mr_abort_repost_recv(rxcq, o,
					   ft_gettime_ms() + CQ_FIRST_TIMEOUT_MS);
		if (ret)
			return ret;
		return 1;
	} else if (ret == -FI_EAVAIL) {
		memset(&err, 0, sizeof(err));
		ret = fi_cq_readerr(rxcq, &err, 0);
		if (ret < 0) {
			FT_PRINTERR("fi_cq_readerr", ret);
			return ret;
		}
		if (!is_expected_err(&err, errs, nerr)) {
			FT_ERR("Unexpected target recv error:");
			FT_CQ_ERR(rxcq, err, NULL, 0);
			return -FI_EOTHER;
		}
		(*recv_canceled)++;
		o = container_of(err.op_context, struct op_ctx, context);
		ret = mr_abort_repost_recv(rxcq, o,
					   ft_gettime_ms() + CQ_FIRST_TIMEOUT_MS);
		if (ret)
			return ret;
		return 1;
	} else if (ret < 0 && ret != -FI_EAGAIN) {
		FT_PRINTERR("fi_cq_read", ret);
		return ret;
	}

	return 0; /* -FI_EAGAIN: CQ momentarily empty */
}

/*
 * Send/tagged abort -- target.
 *
 * Phases (mirror the initiator):
 *   1. Seed the RQ on the first iteration only (later iterations stay full
 *      via the repost-on-completion path in the drain loop).
 *   2. Sync to release the sender, then drain in two phases. First wait for
 *      the initiator's (required, slack) counts over OOB while reaping RX
 *      completions -- with NO internal timeout, since our RX progress is
 *      what lets the sender finish and send the counts. Then drain the rest:
 *      slack == 0 blocks for exactly `required` (no timeout); slack > 0
 *      reaps stragglers under a short idle window. Accepted completions are
 *      successes or abort-class errors (see recv_abort_errs).
 *   3. Pass when reaped is at least required and no more than required plus
 *      slack. The slack > 0 straggler window is the ONLY internal timeout;
 *      every other wait is bounded only by the outer test timeout.
 *
 * Note: target-close (-R target) for send/tagged is not supported
 *
 * The required/slack contract
 * ---------------------------
 * The target cannot derive how many recv completions it is owed: in-flight
 * sends are TX-queue-depth bound (not num_mrs * ops_per_mr), and an aborted
 * send may or may not have delivered enough to build a matched rx entry. So
 * the initiator drains its own TX CQ and ships two counts over OOB:
 *
 *   required - completions the receiver MUST produce: every fully delivered
 *              send, plus -- for the -X protocols where the receiver matched
 *              and took partial data before the abort (LONGCTS, RUNTREAD-
 *              with-tail-READ / LONGREAD) -- every aborted send.
 *   slack    - indeterminate extras (EAGER / MEDIUM / runt-only): an aborted
 *              send may have delivered before the local flush (success),
 *              delivered nothing, or surfaced a stray FI_ECANCELED, so each
 *              contributes 0 or 1 allowed-but-not-required completion. For
 *              the -X protocols the abort is already in `required`, so
 *              slack == 0 (an exact count).
 */
static int run_send_abort_target(int iter)
{
	int i, mr_idx, op_idx, ret;

	op_idx = 0;

	/*
	 * Seed the RQ on the first iteration only. Later iterations stay
	 * populated via the repost-on-completion path in the drain loop, so we
	 * must only bulk-post once.
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
			}
			if (posted_this_mr > 0)
				slots[mr_idx].posted = posted_this_mr;
			if (eagain)
				break;
		}
	}

	/* Sync to let initiator start sending */
	ret = ft_sync();
	if (ret)
		return ret;

	/*
	 * The two abort-class errors a recv may legitimately complete with
	 * when the sender closes its source MR mid-transfer: FI_ECANCELED
	 * (clean peer abort) or FI_EINVAL (the receiver's RDMA read hit the
	 * yanked source MR before the provider remapped it to a clean cancel).
	 * Any other err code is a real failure.
	 *
	 * prov_errno is wildcarded (-1) for now. TODO: tighten to the exact
	 * codes the provider emits on the RX abort path:
	 *   FI_ECANCELED -> FI_EFA_ERR_PEER_ABORTED (4127)
	 *   FI_EINVAL    -> EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS (7)
	 */
	struct expected_err recv_abort_errs[] = {
		{ .err = FI_ECANCELED, .prov_errno = -1 }, /* TODO: tighten this */
		{ .err = FI_EINVAL,    .prov_errno = -1 }, /* TODO: tighten this */
	};
	int recv_ok = 0, recv_canceled = 0;
	int counts[2] = {0};
	size_t counts_got = 0;
	int required = -1;	/* terminal completions the receiver MUST see */
	int slack = 0;		/* indeterminate extra completions allowed */
	int have_counts = 0;
	int reaped = 0;		/* terminal completions counted so far */

	/*
	 * Phase 1: poll OOB (non-blocking) for the initiator's
	 * (required, slack) counts and, under FI_PROGRESS_MANUAL, drain RX
	 * completions while we wait. NO internal timeout: under
	 * FI_PROGRESS_MANUAL our RX progress is what lets the initiator's TX
	 * drain finish and ship the counts, so a deadline here would deadlock
	 * the pair. Under FI_PROGRESS_AUTO the provider drives RX progress
	 * itself, so we skip the manual poll and reap in phase 2.
	 */
	while (!have_counts) {
		ret = oob_recv_nonblock(oob_sock, counts, sizeof(counts),
					&counts_got);
		if (ret < 0) {
			FT_PRINTERR("oob_recv_nonblock", ret);
			return ret;
		}
		if (ret == 1) {
			required = counts[0];
			slack = counts[1];
			have_counts = 1;
			break;
		}

		/*
		 * Only poll the RX CQ here under FI_PROGRESS_MANUAL.
		 */
		if (fi->domain_attr->data_progress != FI_PROGRESS_MANUAL)
			continue;

		ret = target_recv_drain_one(recv_abort_errs,
					    ARRAY_SIZE(recv_abort_errs),
					    &recv_ok, &recv_canceled);
		if (ret < 0)
			return ret;
		if (ret == 1)
			reaped++;
	}

	/*
	 * Phase 2: with the counts known, drain the rest.
	 *
	 *  - slack == 0 (the -X owed protocols plus fully delivered sends):
	 *    every `required` completion is GUARANTEED, so block until
	 *    reaped == required with NO internal timeout -- the same contract
	 *    as the TX-side drain_cq_counted(). A genuinely missing completion
	 *    becomes a hang bounded by the outer test timeout, not a flaky
	 *    short-window failure.
	 *
	 *  - slack > 0 (EAGER / MEDIUM / runt-only): INDETERMINATE -- an
	 *    aborted send may have delivered before the local flush (success),
	 *    delivered nothing, or surfaced a stray FI_ECANCELED. This is the
	 *    ONLY case that needs a timeout: reap stragglers within a rolling
	 *    idle window (seeded to CQ_FIRST_TIMEOUT_MS, reset to
	 *    CQ_IDLE_TIMEOUT_MS on each reap), then accept any total in range.
	 */
	if (slack == 0) {
		while (reaped < required) {
			ret = target_recv_drain_one(recv_abort_errs,
						    ARRAY_SIZE(recv_abort_errs),
						    &recv_ok, &recv_canceled);
			if (ret < 0)
				return ret;
			if (ret == 1)
				reaped++;
		}
	} else {
		uint64_t deadline = ft_gettime_ms() + CQ_IDLE_TIMEOUT_MS;

		while (ft_gettime_ms() < deadline) {
			ret = target_recv_drain_one(recv_abort_errs,
						    ARRAY_SIZE(recv_abort_errs),
						    &recv_ok, &recv_canceled);
			if (ret < 0)
				return ret;
			if (ret == 1) {
				reaped++;
				deadline = ft_gettime_ms() + CQ_IDLE_TIMEOUT_MS;
			}
		}
	}

	/* Pass when reaped lands in [required, required + slack]. */
	if (reaped < required || reaped > required + slack) {
		int missing = required - reaped; /* >0 short, <0 over */

		fprintf(stderr,
			"Target iter %d: required=%d slack=%d reaped=%d "
			"recv_ok=%d peer_aborted=%d missing=%d ... FAIL\n",
			iter, required, slack, reaped, recv_ok,
			recv_canceled, missing);
		return -FI_EOTHER;
	}

	fprintf(stderr,
		"Target iter %d: required=%d slack=%d reaped=%d recv_ok=%d "
		"peer_aborted=%d ... PASS\n",
		iter, required, slack, reaped, recv_ok, recv_canceled);

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
				ret = run_fill_abort_initiator(i + 1);
			} else {
				ret = run_fill_abort_target();
				if (!ret)
					ret = flush_rxcq();
			}
			break;
		case TEST_PARTIAL:
			ret = is_initiator ? run_partial_close_initiator() :
					     run_partial_close_target();
			break;
		case TEST_SEND:
		case TEST_TAGGED:
			ret = is_initiator ? run_send_abort_initiator(i + 1) :
					     run_send_abort_target(i + 1);
			break;
		}
		if (ret) {
			dump_close_order();
			return ret;
		}

		ret = ft_sync();
		if (ret)
			return ret;
	}

	/* Endpoint reuse check — only for RMA tests */
	if (test_mode == TEST_ABORT || test_mode == TEST_PARTIAL) {
		ret = is_initiator ? reuse_check_initiator() :
				     reuse_check_target();
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

	if (set_homogeneous_peers) {
		bool homogeneous = true;
		/*
		 * Tell the EP all peers are homogeneous so it skips the
		 * handshake requirement before using an extra-feature,
		 * read-based protocol (e.g. LONGREAD). Skipping the handshake
		 * makes protocol selection deterministic from the very first
		 * send, so the target is reliably owed -- and can enforce via
		 * -X -- exactly one completion per op.
		 *
		 * Normally this option is set before fi_enable(), but
		 * ft_init_fabric() has already enabled the EP. Setting it here
		 * is still correct: EFA's setopt handler has no enable-state
		 * guard, and homogeneous_peers is read lazily at RTM-post time
		 * (efa_rdm_msg_post_rtm), not at enable. No send has been posted
		 * yet at this point, so the value is in effect for every op.
		 */
		ret = fi_setopt(&ep->fid, FI_OPT_ENDPOINT,
				FI_OPT_EFA_HOMOGENEOUS_PEERS,
				&homogeneous, sizeof(homogeneous));
		if (ret) {
			FT_PRINTERR("fi_setopt(HOMOGENEOUS_PEERS)", ret);
			return ret;
		}
	}

	ret = alloc_test_res();
	if (ret)
		return ret;

	ret = run_mr_abort_test(opts.dst_addr != NULL);

	/*
	 * Target teardown order: EFA requires the endpoint to be closed
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
			    "W:N:C:R:T:A:r:XHh" CS_OPTS INFO_OPTS API_OPTS)) != -1) {
		switch (op) {
		case 'W':
			num_mrs = atoi(optarg);
			break;
		case 'N':
			ops_per_mr = atoi(optarg);
			break;
		case 'r':
			close_order_replay_path = optarg;
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
		case 'X':
			abort_owes_rx_completion = 1;
			break;
		case 'H':
			set_homogeneous_peers = 1;
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
			FT_PRINT_OPTS_USAGE("-r <file>",
				"Replay MR close order from <file> "
				"(overrides -C; written to /tmp on failure)");
			FT_PRINT_OPTS_USAGE("-R <side>",
				"Which side closes MRs: "
				"initiator|target (default: initiator)");
			FT_PRINT_OPTS_USAGE("-A <order>",
				"Target teardown order: ep_first|mr_first "
				"(default: mr_first)");
			FT_PRINT_OPTS_USAGE("-X",
				"Aborted send/tagged ops are owed a target "
				"recv completion (LONGCTS/LONGREAD); only "
				"valid with -T send|tagged");
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

	/*
	 * The partial test always closes a local MR on the initiator; it has
	 * no target-close path (run_partial_close_target() only closes its
	 * extra MR during cleanup, after the transfer). Reject -R target for
	 * -T partial so the unsupported combination fails explicitly rather
	 * than silently running the initiator-close path mislabeled as
	 * "target".
	 */
	if (close_side == CLOSE_TARGET && test_mode == TEST_PARTIAL) {
		FT_ERR("Target-close (-R target) is not supported for "
		       "the partial test (-T partial): it only closes a "
		       "local MR on the initiator");
		return EXIT_FAILURE;
	}

	/*
	 * -X (aborted ops owe a target recv completion) only applies to the
	 * two-sided send/tagged protocols. RMA/atomic and partial modes have
	 * no posted recv to complete, so reject it there.
	 */
	if (abort_owes_rx_completion &&
	    test_mode != TEST_SEND && test_mode != TEST_TAGGED) {
		FT_ERR("-X is only valid with -T send|tagged");
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
