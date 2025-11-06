/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2021-2025 Hewlett Packard Enterprise Development LP
 */

/* CXI benchmark common functions */

#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <getopt.h>
#include <err.h>
#include <fcntl.h>
#include <unistd.h>
#include <limits.h>

#include "utils_common.h"

static const char *sep = "--------------------------------------------------"
			 "--------------------------------------------------";
static int emu_mode;

int (*g_malloc)(void **devPtr, size_t size);
int (*g_free)(void *devPtr);
int (*g_memset)(void *devPtr, int value, size_t size);
int (*g_memcpy)(void *dst, const void *src, size_t size, int kind);
int (*g_set_device)(int deviceId);
int (*g_get_device)(int *deviceId);
int (*g_device_count)(int *count);
int (*g_mem_properties)(const void *addr, void **base, size_t *size, int *dma_buf_fd);
int g_memcpy_kind_htod;

// If emu_mode was given then read the time elapsed from C2 counters
// otherwise default to gettimeofday()
static uint64_t get_time_usec(struct cxil_dev *dev)
{
	uint64_t ret_value;
	uint64_t value;
	int res;

	if (emu_mode == 0)
		return gettimeofday_usec();

	res = cxil_read_csr(dev, C_HNI_ERR_ELAPSED_TIME, &value, sizeof(value));
	if (res < 0) {
		fprintf(stderr, "ERROR: cxil_read_csr failed %d\n", res);
		// fall back to using gettimeofday()
		return gettimeofday_usec();
	}

	unsigned int seconds = value >> 30;
	unsigned int nanoseconds = value & ((1 << 30) - 1);

	// aproximate usec as nanoseconds >> 10
	ret_value = seconds * SEC2USEC + (nanoseconds >> 10);
	return ret_value;
}

uint64_t gettimeofday_usec(void)
{
	struct timeval tv;

	gettimeofday(&tv, 0);
	return (tv.tv_sec * 1.0e+6) + tv.tv_usec;
}

// Initialize the cxi interface to read csrs
int init_time_counters(struct cxil_dev *dev)
{
	int rc = 0;

	emu_mode = 1;
	rc = cxil_map_csr(dev);
	if (rc < 0) {
		fprintf(stderr, "ERROR: cxil_map_csr failed (%d)\n", rc);
		return rc;
	}
	return rc;
}

// If emu_mode was given then read the time elapsed from C2 counters
// otherwise default to clock_gettime()
int clock_gettime_or_counters(clockid_t clock_mode, struct timespec *ts,
							  struct cxil_dev *dev)
{
	int res = 0;
	uint64_t value;

	if (emu_mode == 0 || dev == NULL)
		return clock_gettime(clock_mode, ts);

	res = cxil_read_csr(dev, C_HNI_ERR_ELAPSED_TIME, &value, sizeof(value));
	if (res < 0) {
		fprintf(stderr, "ERROR: cxil_read_csr failed %d\n", res);
		// fall back to using clock_gettime()
		return clock_gettime(clock_mode, ts);
	}

	unsigned int seconds = value >> 30;
	unsigned int nanoseconds = value & ((1 << 30) - 1);

	ts->tv_sec = seconds;
	ts->tv_nsec = nanoseconds;

	return res;
}

void print_separator(size_t len)
{
	printf("%.*s\n", (int)(len < strlen(sep) ? len : strlen(sep)), sep);
}

/* Sleep, but avoid looking idle */
int active_sleep(uint64_t usec, struct cxil_dev *dev)
{
	int rc = 0;
	struct timespec lat_ts_0;
	struct timespec lat_ts_1;
	uint64_t count = 0;

	rc = clock_gettime_or_counters(CLOCK_MONOTONIC_RAW, &lat_ts_0, dev);
	if (rc < 0) {
		rc = -errno;
		fprintf(stderr, "clock_gettime() failed: %s\n", strerror(-rc));
		return rc;
	}
	while (count < (usec * 1000)) {
		rc = clock_gettime_or_counters(CLOCK_MONOTONIC_RAW, &lat_ts_1,
					       dev);
		if (rc < 0) {
			rc = -errno;
			fprintf(stderr, "clock_gettime() failed: %s\n",
				strerror(-rc));
			return rc;
		}
		count += TV_NSEC_DIFF(lat_ts_0, lat_ts_1);
	}

	return rc;
}

/* Allocate initiator resources */
int alloc_ini(struct cxi_context *ctx, struct cxi_ctx_ini_opts *opts)
{
	int rc;
	struct cxi_cq_alloc_opts rdzv_pte_cq_opts = { .count = 1 };
	struct cxi_pt_alloc_opts rdzv_pt_opts = { .is_matching = 1 };
	struct cxi_cq_alloc_opts trig_cq_opts = {
		.count = 1, .flags = CXI_CQ_IS_TX | CXI_CQ_TX_WITH_TRIG_CMDS
	};

	if (!ctx || !opts)
		return -EINVAL;

	cxi_build_dfa(ctx->rmt_addr.nic, ctx->rmt_addr.pid,
		      ctx->dev->info.pid_bits, opts->pid_offset, &ctx->dfa,
		      &ctx->index_ext);

	/* communication profiles */
	rc = ctx_alloc_cp(ctx, CXI_TC_BEST_EFFORT, CXI_TC_TYPE_DEFAULT,
			  &ctx->cp);
	if (rc < 0) {
		fprintf(stderr,
			"Failed to allocate Communication Profile: %s\n",
			strerror(-rc));
		return rc;
	}
	if (opts->alloc_hrp) {
		rc = ctx_alloc_cp(ctx, CXI_TC_BEST_EFFORT, CXI_TC_TYPE_HRP,
				  &ctx->hrp_cp);
		if (rc < 0) {
			fprintf(stderr,
				"Failed to allocate HRP Communication Profile: %s\n",
				strerror(-rc));
			return rc;
		}
	}

	/* event queue and buffer */
	rc = ctx_alloc_buf(ctx, opts->eq_attr.queue_len, CTX_BUF_PAT_ZERO,
			   HP_DISABLED, &ctx->ini_eq_buf);
	if (rc < 0) {
		fprintf(stderr, "Failed to allocate Initiator EQ buffer: %s\n",
			strerror(-rc));
		return rc;
	}
	opts->eq_attr.queue = ctx->ini_eq_buf->buf;
	rc = ctx_alloc_eq(ctx, &opts->eq_attr, ctx->ini_eq_buf->md,
			  &ctx->ini_eq);
	if (rc < 0) {
		fprintf(stderr, "Failed to allocate Initiator EQ: %s\n",
			strerror(-rc));
		return rc;
	}

	/* buffer */
	if (opts->use_gpu_buf)
		rc = ctx_alloc_gpu_buf(ctx, opts->buf_opts.length,
				       opts->buf_opts.pattern, &ctx->ini_buf,
				       opts->gpu_id);
	else
		rc = ctx_alloc_buf(ctx, opts->buf_opts.length,
				   opts->buf_opts.pattern, opts->buf_opts.hp,
				   &ctx->ini_buf);
	if (rc < 0) {
		fprintf(stderr, "Failed to allocate Initiator buffer: %s\n",
			strerror(-rc));
		return rc;
	}

	/* command queue */
	opts->cq_opts.lcid = ctx->cp->lcid;
	rc = ctx_alloc_cq(ctx, NULL, &opts->cq_opts, &ctx->ini_cq);
	if (rc < 0) {
		fprintf(stderr,
			"Failed to allocate Initiator Command Queue: %s\n",
			strerror(-rc));
		return rc;
	}

	if (opts->alloc_ct) {
		/* event counter */
		rc = ctx_alloc_ct(ctx, &ctx->ini_ct);
		if (rc < 0) {
			fprintf(stderr,
				"Failed to allocate Initiator Event Counter: %s\n",
				strerror(-rc));
			return rc;
		}

		/* CT triggered op command queue */
		rc = ctx_alloc_cq(ctx, ctx->ini_eq, &trig_cq_opts,
				  &ctx->ini_trig_cq);
		if (rc < 0) {
			fprintf(stderr,
				"Failed to allocate Initiator Trig. Command Queue: %s\n",
				strerror(-rc));
			return rc;
		}
	}

	if (!opts->alloc_rdzv)
		return rc;

	/* RDZV event queue and buffer */
	rc = ctx_alloc_buf(ctx, opts->eq_attr.queue_len, CTX_BUF_PAT_ZERO,
			   HP_DISABLED, &ctx->ini_rdzv_eq_buf);
	if (rc < 0) {
		fprintf(stderr,
			"Failed to allocate Initiator RDZV EQ buffer: %s\n",
			strerror(-rc));
		return rc;
	}
	opts->eq_attr.queue = ctx->ini_rdzv_eq_buf->buf;
	rc = ctx_alloc_eq(ctx, &opts->eq_attr, ctx->ini_rdzv_eq_buf->md,
			  &ctx->ini_rdzv_eq);
	if (rc < 0) {
		fprintf(stderr, "Failed to allocate Initiator RDZV EQ: %s\n",
			strerror(-rc));
		return rc;
	}

	if (opts->alloc_ct) {
		/* RDZV event counter */
		rc = ctx_alloc_ct(ctx, &ctx->ini_rdzv_ct);
		if (rc < 0) {
			fprintf(stderr,
				"Failed to allocate Initiator RDZV Event Counter: %s\n",
				strerror(-rc));
			return rc;
		}

		/* RDZV CT triggered op command queue */
		rc = ctx_alloc_cq(ctx, ctx->ini_rdzv_eq, &trig_cq_opts,
				  &ctx->ini_rdzv_trig_cq);
		if (rc < 0) {
			fprintf(stderr,
				"Failed to allocate Initiator RDZV Trig. CQ: %s\n",
				strerror(-rc));
			return rc;
		}
	}

	/* RDZV PTE command queue */
	rc = ctx_alloc_cq(ctx, NULL, &rdzv_pte_cq_opts, &ctx->ini_rdzv_pte_cq);
	if (rc < 0) {
		fprintf(stderr,
			"Failed to allocate Initiator RDZV PTE CQ: %s\n",
			strerror(-rc));
		return rc;
	}

	/* RDZV Portals table entry */
	rc = ctx_alloc_pte(ctx, ctx->ini_rdzv_eq, &rdzv_pt_opts,
			   ctx->dev->info.rdzv_get_idx, &ctx->ini_rdzv_pte);
	if (rc < 0) {
		fprintf(stderr, "Failed to allocate and map PTE: %s\n",
			strerror(-rc));
		return rc;
	}
	rc = enable_pte(ctx->ini_rdzv_pte_cq, ctx->ini_rdzv_eq,
			ctx->ini_rdzv_pte->ptn);
	if (rc) {
		fprintf(stderr, "Failed to enable PTE: %s\n", strerror(-rc));
		return rc;
	}

	return rc;
}

/* Allocate target resources */
int alloc_tgt(struct cxi_context *ctx, struct cxi_ctx_tgt_opts *opts)
{
	int rc;
	struct cxi_cq_alloc_opts trig_cq_opts = {
		.count = 1, .flags = CXI_CQ_IS_TX | CXI_CQ_TX_WITH_TRIG_CMDS
	};

	if (!ctx || !opts)
		return -EINVAL;

	/* event queue and buffer */
	rc = ctx_alloc_buf(ctx, opts->eq_attr.queue_len, CTX_BUF_PAT_ZERO,
			   HP_DISABLED, &ctx->tgt_eq_buf);
	if (rc < 0) {
		fprintf(stderr, "Failed to allocate Target EQ buffer: %s\n",
			strerror(-rc));
		return rc;
	}
	opts->eq_attr.queue = ctx->tgt_eq_buf->buf;
	rc = ctx_alloc_eq(ctx, &opts->eq_attr, ctx->tgt_eq_buf->md,
			  &ctx->tgt_eq);
	if (rc < 0) {
		fprintf(stderr, "Failed to allocate Target EQ: %s\n",
			strerror(-rc));
		return rc;
	}

	/* buffer */
	if (opts->use_gpu_buf)
		rc = ctx_alloc_gpu_buf(ctx, opts->buf_opts.length,
				       opts->buf_opts.pattern, &ctx->tgt_buf,
				       opts->gpu_id);
	else
		rc = ctx_alloc_buf(ctx, opts->buf_opts.length,
				   opts->buf_opts.pattern, opts->buf_opts.hp,
				   &ctx->tgt_buf);
	if (rc < 0) {
		fprintf(stderr, "Failed to allocate Target buffer: %s\n",
			strerror(-rc));
		return rc;
	}

	/* command queue */
	rc = ctx_alloc_cq(ctx, NULL, &opts->cq_opts, &ctx->tgt_cq);
	if (rc < 0) {
		fprintf(stderr, "Failed to allocate Target Command Queue: %s\n",
			strerror(-rc));
		return rc;
	}

	/* Portals table entry */
	rc = ctx_alloc_pte(ctx, ctx->tgt_eq, &opts->pt_opts, opts->pte_index,
			   &ctx->tgt_pte);
	if (rc < 0) {
		fprintf(stderr, "Failed to allocate and map PTE: %s\n",
			strerror(-rc));
		return rc;
	}
	rc = enable_pte(ctx->tgt_cq, ctx->tgt_eq, ctx->tgt_pte->ptn);
	if (rc) {
		fprintf(stderr, "Failed to enable PTE: %s\n", strerror(-rc));
		return rc;
	}

	/* Separate endpoint for cxi_send_lat to signal completion */
	if (opts->use_final_lat) {
		rc = ctx_alloc_pte(ctx, ctx->tgt_eq, &opts->pt_opts,
				   (opts->pte_index + 1),
				   &ctx->tgt_final_lat_pte);
		if (rc < 0) {
			fprintf(stderr, "Failed to allocate and map PTE: %s\n",
				strerror(-rc));
			return rc;
		}
		rc = enable_pte(ctx->tgt_cq, ctx->tgt_eq,
				ctx->tgt_final_lat_pte->ptn);
		if (rc) {
			fprintf(stderr, "Failed to enable PTE: %s\n",
				strerror(-rc));
			return rc;
		}
	}

	if (opts->alloc_ct) {
		/* event counter */
		rc = ctx_alloc_ct(ctx, &ctx->tgt_ct);
		if (rc < 0) {
			fprintf(stderr,
				"Failed to allocate Target Event Counter: %s\n",
				strerror(-rc));
			return rc;
		}

		/* CT triggered op command queue */
		rc = ctx_alloc_cq(ctx, ctx->tgt_eq, &trig_cq_opts,
				  &ctx->tgt_trig_cq);
		if (rc < 0) {
			fprintf(stderr,
				"Failed to allocate Target Trig. Command Queue: %s\n",
				strerror(-rc));
			return rc;
		}
	}

	return rc;
}

int sw_rdzv_get(struct util_context *util, struct c_event_target_long ev)
{
	int rc;
	struct cxi_context *cxi = &util->cxi;

	util->tgt_rdzv_get_cmd.full_dma.header_data = ev.header_data;
	util->tgt_rdzv_get_cmd.full_dma.match_bits = ev.match_bits;
	util->tgt_rdzv_get_cmd.full_dma.request_len = ev.rlength - ev.mlength;
	util->tgt_rdzv_get_cmd.full_dma.local_addr = ev.start;
	util->tgt_rdzv_get_cmd.full_dma.remote_offset = ev.remote_offset;
	rc = cxi_cq_emit_dma(cxi->ini_cq, &util->tgt_rdzv_get_cmd.full_dma);
	if (rc) {
		fprintf(stderr, "Failed to issue RX RDZV DMA command: %s\n",
			strerror(-rc));
		return rc;
	}
	cxi_cq_ring(cxi->ini_cq);

	return rc;
}

static int print_bw_results(struct util_context *util, uint64_t elapsed)
{
	int rc = 0;
	struct ctrl_connection *ctrl = &util->ctrl;
	struct util_opts *opts = &util->opts;
	uint64_t pkt_per_count = 0;
	size_t size = 0;
	uint64_t count = 0;
	struct bw_results {
		uint64_t elapsed;
		uint64_t count;
	};
	struct bw_results res;
	struct bw_results peer;
	long double bw = 0;
	long double rate = 0;

	/* Calculate rates */
	size = util->size;
	pkt_per_count = ((size - 1) / PORTALS_MTU) + 1;
	if (!ctrl->is_server || opts->bidirectional) {
		count = util->count * opts->list_size;
		if (elapsed) {
			bw = ((long double)size / elapsed) * count;
			rate = ((long double)count / elapsed) * pkt_per_count;
		}
	}

	/* Combine with peer */
	if (ctrl->connected) {
		res.elapsed = elapsed;
		res.count = count;

		rc = ctrl_exchange_data(ctrl, &res, sizeof(res), &peer,
					sizeof(peer));
		if (rc < 0) {
			fprintf(stderr, "Failed to share peer results: %s\n",
				strerror(-rc));
			return rc;
		}
		if (rc != sizeof(peer)) {
			fprintf(stderr, "Peer results size %d (expected %lu)\n",
				rc, sizeof(peer));
			return -ENOMSG;
		}
		rc = 0;

		if (peer.elapsed) {
			bw += ((long double)size / peer.elapsed) * peer.count;
			rate += ((long double)peer.count / peer.elapsed) *
				pkt_per_count;
		}
	}

	/* Bi-directional loopback counts double */
	if (ctrl->is_loopback && opts->bidirectional) {
		bw *= 2;
		rate *= 2;
	}

	if (opts->print_gbits)
		bw = bw * 8 / 1000; /* MB to Gbit */
	if (!ctrl->is_server || opts->bidirectional)
		printf("%*lu  %*lu  %*.*Lf  %*.*Lf\n", SIZE_W, size, COUNT_W,
		       count, BW_W, BW_FRAC_W, bw, MRATE_W, MRATE_FRAC_W, rate);
	else
		printf("%*lu  %*s  %*.*Lf  %*.*Lf\n", SIZE_W, size, COUNT_W,
		       "-", BW_W, BW_FRAC_W, bw, MRATE_W, MRATE_FRAC_W, rate);

	return rc;
}

/*
 * Calculate a reasonable post-run timeout value based on elapsed run time.
 * Allow the default post-run timeout to be overridden by specifying the
 * CXIUTIL_POST_RUN_TIMEOUT environment variable.
 */
uint64_t get_post_run_timeout(uint64_t elapsed)
{
	uint64_t postrun_timeout;
	char *pr_timeout_str;
	char *endptr;
	long pr_timeout_secs;

	/*
	 * Use a reasonable timeout that allows enough time for
	 * short running jobs to complete while limiting the
	 * time to wait for long running jobs to 20% of elapsed time.
	 */
	pr_timeout_str = getenv("CXIUTIL_POST_RUN_TIMEOUT");
	if (pr_timeout_str) {
		errno = 0;
		pr_timeout_secs = (uint64_t)strtol(pr_timeout_str, &endptr, 10);
		if (*endptr != 0 || errno != 0 || pr_timeout_secs < 0 || pr_timeout_secs > INT_MAX) {
			postrun_timeout = DFLT_HANDSHAKE_TIMEOUT;
			fprintf(stderr,
				"Invalid CXIUTIL_POST_RUN_TIMEOUT value found: %s.\n",
				pr_timeout_str);
			fprintf(stderr,
				"Using the default timeout of %ld usec.\n",
				postrun_timeout);
		} else if (pr_timeout_secs == 0) {
			postrun_timeout = NO_TIMEOUT;
			fprintf(stdout, "Disabling post-run timeout.\n");
		} else {
			postrun_timeout = pr_timeout_secs * SEC2USEC;
			fprintf(stdout,
				"Overriding default post-run timeout to %ld usec.\n",
				postrun_timeout);
		}
	} else {
		postrun_timeout = fmax(DFLT_HANDSHAKE_TIMEOUT, elapsed/5);
	}

	return postrun_timeout;
}

/* Repeatedly call the provided do_iter function until the specified number of
 * iterations or duration has passed. Then calculate bandwidth.
 */
int run_bw_active(struct util_context *util,
		  int (*do_iter)(struct util_context *util))
{
	int rc = 0;
	struct ctrl_connection *ctrl;
	struct util_opts *opts;
	uint64_t start_time;
	uint64_t elapsed;
	uint64_t duration_usec;

	if (!util || !do_iter)
		return -EINVAL;

	ctrl = &util->ctrl;
	opts = &util->opts;

	/* When measuring bidirectional bandwidth, we want to sync immediately
	 * after getting the start time and before getting the end time. This
	 * ensures that if the two directions don't stay in sync, we end up
	 * reporting lower BW rather than higher BW. This should not matter
	 * when running for a duration rather than a number of iterations.
	 */
	duration_usec = opts->duration * SEC2USEC;

        if (ctrl->connected) {
		rc = ctrl_barrier(ctrl, DFLT_HANDSHAKE_TIMEOUT, "Pre-run");
		if (rc)
			return rc;
	}

	start_time = get_time_usec(util->cxi.dev);

	util->count = 0;
	elapsed = 0;
	while ((!opts->duration && util->count < opts->iters) ||
			(opts->duration && elapsed < duration_usec)) {
		rc = do_iter(util);
		if (!rc)
			util->count++;
		else if (rc != EAGAIN) {
			fprintf(stderr, "Iteration failed: %s\n",
				strerror(abs(rc)));
			return rc;
		}

		if (opts->duration)
			elapsed = get_time_usec(util->cxi.dev) - start_time;
	}

        if (ctrl->connected) {
		rc = ctrl_barrier(ctrl, DFLT_HANDSHAKE_TIMEOUT, "Post-run");
		if (rc)
			return rc;
	}

	/* Final elapsed time update */
	elapsed = get_time_usec(util->cxi.dev) - start_time;

	rc = print_bw_results(util, elapsed);

	return rc;
}

/* Wait to hear from server */
int run_bw_passive(struct util_context *util)
{
	int rc;

	if (!util)
		return -EINVAL;

	rc = ctrl_barrier(&util->ctrl, NO_TIMEOUT, "Pre-run");
	if (rc)
		return rc;

	rc = ctrl_barrier(&util->ctrl, NO_TIMEOUT, "Post-run");
	if (rc)
		return rc;

	rc = print_bw_results(util, 0);

	return rc;
}

/* cxi_send_lat needs to ensure that neither peer completes before the other
 * when running in duration mode. It does this by sending an "I'm done" signal
 * by performing a final, unmeasured iteration that targets a different
 * endpoint. This function flips the initiator between targeting one or the
 * other endpoint.
 */
static int flip_final_lat_send(struct util_context *util)
{
	int rc = 0;
	uint8_t index_ext;
	struct cxi_context *cxi;
	union c_cmdu c_st_cmd = { 0 };

	cxi = &util->cxi;
	if (util->final_lat_sent)
		index_ext = cxi->index_ext;
	else
		index_ext = cxi->index_ext + 1;

	util->dma_cmd.full_dma.index_ext = index_ext;

	c_st_cmd.c_state.command.opcode = C_CMD_CSTATE;
	c_st_cmd.c_state.index_ext = index_ext;
	c_st_cmd.c_state.event_send_disable = 1;
	c_st_cmd.c_state.event_success_disable = 1;
	c_st_cmd.c_state.eq = cxi->ini_eq->eqn;
	rc = cxi_cq_emit_c_state(cxi->ini_cq, &c_st_cmd.c_state);
	if (rc) {
		fprintf(stderr, "Failed to issue C State command: %s\n",
			strerror(-rc));
		return rc;
	}
	cxi_cq_ring(cxi->ini_cq);

	util->final_lat_sent = !util->final_lat_sent;
	return 0;
}

/* Repeatedly call the provided do_iter function until the specified number of
 * iterations or duration has passed. Then calculate latency.
 */
int run_lat_active(struct util_context *util,
		   int (*do_iter)(struct util_context *util))
{
	int rc = 0;
	struct ctrl_connection *ctrl;
	struct util_opts *opts;
	struct cxi_context *cxi;
	uint64_t i;
	uint64_t warmup_count;
	uint64_t start_time;
	uint64_t duration_usec;
	uint64_t elapsed;
	uint64_t lat_min;
	uint64_t lat_max;
	uint64_t lat_avg;
	uint64_t lat_sdev;
	uint64_t lat_sum;
	uint64_t lat_sum2;
	uint64_t *lats = NULL;

	if (!util || !do_iter)
		return -EINVAL;

	ctrl = &util->ctrl;
	opts = &util->opts;
	cxi = &util->cxi;

	warmup_count = 0;
	start_time = 0;
	elapsed = 0;
	duration_usec = opts->duration * 1e+6;
	lat_min = UINT64_MAX;
	lat_max = 0;
	lat_sum = 0;
	lat_sum2 = 0;

	if (opts->report_all) {
		lats = malloc(opts->iters * sizeof(lats[0]));
		if (!lats)
			err(1, "Failed to allocate results buffer");
	}

	/* Reset final_lat state when measuring multiple sizes (applies to
	 * cxi_send_lat only)
	 */
	if (opts->duration && cxi->tgt_final_lat_pte) {
		if (util->final_lat_sent) {
			rc = flip_final_lat_send(util);
			if (rc < 0)
				goto done;
		}
		if (util->final_lat_recv)
			util->final_lat_recv = false;
	}

	/* Measure latencies */
	util->count = 0;
	while (((!opts->duration && util->count < opts->iters) ||
	       (opts->duration && elapsed < duration_usec))) {
		rc = do_iter(util);
		if (rc < 0) {
			fprintf(stderr, "Iteration failed: %s\n",
				strerror(-rc));
			goto done;
		} else if (util->final_lat_recv) {
			/* Peer is done (applies to cxi_send_lat only) */
			break;
		}
		if (warmup_count < opts->warmup) {
			warmup_count++;
		} else {
			if (!util->count)
				start_time = get_time_usec(
					util->cxi.dev);

			if (opts->report_all)
				lats[util->count] = util->last_lat;
			if (util->last_lat < lat_min)
				lat_min = util->last_lat;
			if (util->last_lat > lat_max)
				lat_max = util->last_lat;
			lat_sum += util->last_lat;
			lat_sum2 += (util->last_lat * util->last_lat);
			util->count++;
		}

		if (opts->iter_delay) {
			rc = active_sleep(opts->iter_delay,
					  util->cxi.dev);
			if (rc < 0)
				goto done;
		}

		if (opts->duration && start_time)
			elapsed = get_time_usec(util->cxi.dev) - start_time;
	}

	/* Let peer know that we're done (applies to cxi_send_lat only) */
	if (opts->duration && !rc && cxi->tgt_final_lat_pte &&
	    !util->final_lat_recv) {
		rc = flip_final_lat_send(util);
		if (rc < 0)
			goto done;
		rc = do_iter(util);
		if (rc < 0)
			goto done;
	}

	if (ctrl->connected) {
		rc = ctrl_barrier(ctrl, DFLT_HANDSHAKE_TIMEOUT, "Post-run");
		if (rc < 0)
			goto done;
	}
	rc = 0;

	lat_avg = lat_sum / util->count;
	lat_sdev = sqrt((lat_sum2 / util->count) - (lat_avg * lat_avg));

	/* Print results */
	if (!ctrl->connected || !ctrl->is_server) {
		if (opts->report_all) {
			print_separator(strlen(util->header));
			printf("%*s  %*s\n", COUNT_W, "WriteNum", LAT_ALL_W,
			       "Latency[us]");
			for (i = 0; i < util->count; i++)
				printf("%*lu  %*lu.%0*lu\n", COUNT_W, i,
				       LAT_ALL_DEC_W, lats[i] / 1000,
				       LAT_ALL_FRAC_W, (lats[i] % 1000));
			print_separator(strlen(util->header));
			printf("%s\n", util->header);
		}
		// clang-format off
		printf("%*lu  %*lu  %*lu.%0*lu  %*lu.%0*lu  %*lu.%0*lu  %*lu.%0*lu\n",
		       SIZE_W, util->size, COUNT_W, util->count,
		       LAT_DEC_W, lat_min / 1000,
		       LAT_FRAC_W, (lat_min % 1000) / 10,
		       LAT_DEC_W, lat_max / 1000,
		       LAT_FRAC_W, (lat_max % 1000) / 10,
		       LAT_DEC_W, lat_avg / 1000,
		       LAT_FRAC_W, (lat_avg % 1000) / 10,
		       LAT_DEC_W, lat_sdev / 1000,
		       LAT_FRAC_W, (lat_sdev % 1000) / 10);
		// clang-format on
	} else {
		print_separator(strlen(util->header));
		printf("See client for results.\n");
	}

done:
	if (lats)
		free(lats);
	return rc;
}

/* Wait to hear from server */
int run_lat_passive(struct util_context *util)
{
	int rc;

	if (!util)
		return -EINVAL;

	rc = ctrl_barrier(&util->ctrl, NO_TIMEOUT, "Post-run");
	if (rc < 0)
		return rc;
	else
		rc = 0;
	print_separator(strlen(util->header));
	printf("See client for results.\n");

	return rc;
}

void parse_common_opt(char c, struct util_opts *opts, const char *name,
		      const char *version, void (*usage)(void))
{
	int rc;
	int i;
	char *endptr;
	int hp;
	long tmp;

	switch (c) {
	case VAL_NO_HRP:
		opts->use_hrp = 0;
		break;
	case VAL_NO_IDC:
		opts->use_idc = 0;
		break;
	case VAL_BUF_SZ:
		errno = 0;
		endptr = NULL;
		opts->max_buf_size = strtoul(optarg, &endptr, 0);
		if (errno != 0 || *endptr != 0 || opts->max_buf_size == 0)
			errx(1, "Invalid buffer size: %s", optarg);
		break;
	case VAL_BUF_ALIGN:
		errno = 0;
		endptr = NULL;
		opts->buf_align = strtoul(optarg, &endptr, 0);
		if (errno != 0 || *endptr != 0 || opts->buf_align == 0)
			errx(1, "Invalid buffer alignment: %s", optarg);
		break;
	case VAL_NO_LL:
		opts->use_ll = 0;
		break;
	case VAL_REPORT_ALL:
		opts->report_all = 1;
		break;
	case VAL_WARMUP:
		errno = 0;
		endptr = NULL;
		opts->warmup = strtoul(optarg, &endptr, 0);
		if (errno != 0 || *endptr != 0)
			errx(1, "Invalid warmup count: %s", optarg);
		break;
	case VAL_ITER_DELAY:
		errno = 0;
		endptr = NULL;
		opts->iter_delay = strtoul(optarg, &endptr, 0);
		if (errno != 0 || *endptr != 0)
			errx(1, "Invalid iteration gap: %s", optarg);
		break;
	case VAL_FETCHING:
		opts->fetching = 1;
		break;
	case VAL_MATCHING:
		opts->matching = 1;
		break;
	case VAL_UNRESTRICTED:
		opts->unrestricted = 1;
		break;
	case VAL_USE_HP:
		hp = get_hugepage_type(optarg);
		if (hp < 0)
			errx(1, "Invalid hugepage type: %s. Must be 2M or 1G",
			     optarg);
		else
			opts->hugepages = hp;
		break;
	case 'd':
		if (strlen(optarg) < 4 || strncmp(optarg, "cxi", 3))
			errx(1,
			     "Device name must include cxi prefix (ex: cxi0): %s",
			     optarg);
		optarg += 3;
		errno = 0;
		endptr = NULL;
		opts->loc_opts.dev_id = strtoul(optarg, &endptr, 10);
		if (errno != 0 || *endptr != 0)
			errx(1, "Invalid device name: %s", optarg);
		break;
	case 'v':
		errno = 0;
		endptr = NULL;
		tmp = strtol(optarg, &endptr, 10);
		if (errno != 0 || *endptr != 0 || endptr == optarg || tmp < 1 ||
		    tmp > INT_MAX)
			errx(1, "Invalid svc_id: %s", optarg);
		opts->loc_opts.svc_id = tmp;
		break;
	case 'p':
		opts->loc_opts.port = strtoul(optarg, &endptr, 0);
		if (errno != 0 || *endptr != 0)
			errx(1, "Invalid port: %s", optarg);
		break;
	case 't':
		errno = 0;
		endptr = NULL;
		opts->loc_opts.tx_gpu = strtoul(optarg, &endptr, 0);
		if (errno != 0 || *endptr != 0)
			errx(1, "Invalid src gpu device: %s", optarg);
		opts->loc_opts.use_tx_gpu = 1;
		opts->use_idc = 0;
		break;
	case 'r':
		errno = 0;
		endptr = NULL;
		opts->loc_opts.rx_gpu = strtoul(optarg, &endptr, 0);
		if (errno != 0 || *endptr != 0)
			errx(1, "Invalid dest gpu device: %s", optarg);
		opts->loc_opts.use_rx_gpu = 1;
		break;
	case 'n':
		errno = 0;
		endptr = NULL;
		opts->iters = strtoul(optarg, &endptr, 0);
		if (errno != 0 || *endptr != 0 || opts->iters == 0)
			errx(1, "Invalid iteration count: %s", optarg);
		break;
	case 'D':
		errno = 0;
		endptr = NULL;
		opts->duration = strtoul(optarg, &endptr, 0);
		if (errno != 0 || *endptr != 0 || opts->duration == 0)
			errx(1, "Invalid duration: %s", optarg);
		break;
	case 'A':
		opts->atomic_op = C_AMO_OP_AXOR + 1;
		for (i = C_AMO_OP_MIN; i <= C_AMO_OP_AXOR; i++) {
			if (i == 3) /* No valid op with this value */
				continue;
			if (!strcasecmp(optarg, amo_op_strs[i])) {
				opts->atomic_op = i;
				break;
			}
		}
		if (opts->atomic_op > C_AMO_OP_AXOR)
			errx(1, "Invalid atomic operation: %s", optarg);
		break;
	case 'C':
		opts->cswap_op = C_AMO_OP_CSWAP_GT + 1;
		for (i = C_AMO_OP_CSWAP_EQ; i <= C_AMO_OP_CSWAP_GT; i++) {
			if (!strcasecmp(optarg, amo_cswap_op_strs[i])) {
				opts->cswap_op = i;
				break;
			}
		}
		if (opts->cswap_op > C_AMO_OP_CSWAP_GT)
			errx(1, "Invalid CSWAP operation: %s", optarg);
		break;
	case 'T':
		opts->atomic_type = C_AMO_TYPE_UINT128_T + 1;
		for (i = C_AMO_TYPE_INT8_T; i <= C_AMO_TYPE_UINT128_T; i++) {
			if (!strcasecmp(optarg, amo_type_strs[i])) {
				opts->atomic_type = i;
				break;
			}
		}
		if (opts->atomic_type > C_AMO_TYPE_UINT128_T)
			errx(1, "Invalid atomic type: %s", optarg);
		break;
	case 's':
		rc = sscanf(optarg, "%lu:%lu", &opts->min_size,
			    &opts->max_size);
		if ((rc == 0 || rc == EOF) ||
		    (opts->min_size == 0 || opts->min_size > MAX_MSG_SIZE) ||
		    (rc == 2 && (opts->max_size < opts->min_size ||
				 opts->max_size > MAX_MSG_SIZE)))
			errx(1, "Invalid size: %s", optarg);
		else if (rc == 2 &&
			 ((opts->min_size & (opts->min_size - 1)) != 0 ||
			  (opts->max_size & (opts->max_size - 1)) != 0))
			errx(1, "Invalid size: %s (ranges must be powers of 2)",
			     optarg);
		else if (rc == 1)
			opts->max_size = opts->min_size;
		break;
	case 'l':
		errno = 0;
		endptr = NULL;
		opts->list_size = strtoul(optarg, &endptr, 0);
		if (errno != 0 || *endptr != 0 || opts->list_size == 0 ||
		    (opts->list_size > ((MAX_CQ_DEPTH / 4) - 1)))
			errx(1, "Invalid list size: %s", optarg);
		break;
	case 'b':
		opts->bidirectional = 1;
		break;
	case VAL_EMU_MODE:
		opts->emu_mode = 1;
		break;
	case 'R':
		opts->use_rdzv = 1;
		break;
	case 'g':
		opts->loc_opts.gpu_type = get_gpu_type(optarg);
		if (opts->loc_opts.gpu_type < 0)
			errx(1, "Invalid gpu type: %s. Must be AMD or NVIDIA",
			     optarg);
		break;
	case 'c':
		if (strcmp(optarg, "clock_gettime") == 0)
			opts->clock = CLOCK_GETTIME;
		else if (strcmp(optarg, "cycles") == 0)
			opts->clock = CYCLES;
		else
			errx(1, "Invalid clock: %s", optarg);
		break;

	case VAL_IGNORE_CPU_FREQ_MISMATCH:
		opts->ignore_cpu_freq_mismatch = 1;
		break;

	case 'h':
		usage();
		exit(0);
	case 'V':
		printf("%s version: %s\n", name, version);
		exit(0);
	case '?':
		usage();
		exit(1);
	default:
		errx(1, "Invalid argument, value = %d", c);
	};
}

void parse_server_addr(int argc, char **argv, struct ctrl_connection *ctrl,
		       uint16_t port)
{
	if (optind < argc) {
		ctrl->dst_addr = argv[optind++];
		ctrl->dst_port = port;
		if (optind < argc)
			errx(1, "Unexpected argument: %s", argv[optind]);
	} else {
		ctrl->src_port = port;
	}
}

int get_hugepage_type(char *type)
{
	if (strcasecmp(type, hugepage_names[HP_2M]) == 0)
		return HP_2M;
	else if (strcasecmp(type, hugepage_names[HP_1G]) == 0)
		return HP_1G;
	else
		return -1;
}

/* Return the number of free hugepages of size hp_size_in_bytes */
uint32_t get_free_hugepages(size_t hp_size_in_bytes)
{
	int rc;
	uint32_t total = 0;
	char buf[9];
	int fd;
	char *f_name;

	rc = asprintf(&f_name,
		      "/sys/kernel/mm/hugepages/hugepages-%ldkB/free_hugepages",
		      hp_size_in_bytes / 1024);
	if (rc <= 0)
		err(1, "String allocation failed");

	fd = open(f_name, O_RDONLY);
	if (fd < 0) {
		free(f_name);
		err(1, "Unable to open %s", f_name);
	}

	buf[8] = '\0';
	rc = read(fd, buf, 8);
	if (rc < 0) {
		free(f_name);
		close(fd);
		err(1, "Unable to read %s", f_name);
	}

	rc = sscanf(buf, "%u", &total);
	if (rc != 1) {
		free(f_name);
		close(fd);
		err(1, "Unable to determine total free hugepages from %s",
		    f_name);
	}

	free(f_name);
	close(fd);
	return total;
}

int get_hugepages_needed(struct util_opts *opts, bool ini_buf, bool tgt_buf)
{
	int bufs_needed = 0;
	uint32_t num_2m;
	uint32_t num_1g;
	int hp_2m_needed;
	int hp_1g_needed;

	/* Determine number of huge pages needed */
	if (opts->hugepages == HP_DISABLED)
		goto no_hp;
	if (ini_buf && !opts->loc_opts.use_tx_gpu)
		bufs_needed++;
	if (tgt_buf && !opts->loc_opts.use_rx_gpu)
		bufs_needed++;
	if (!bufs_needed) {
		opts->hugepages = HP_NOT_APPLIC;
		goto no_hp;
	}

	/* Determine if available and update buffer size to match */
	if (opts->hugepages == HP_2M) {
		num_2m = get_free_hugepages(TWO_MB);
		hp_2m_needed =
			(opts->buf_size / TWO_MB) + !!(opts->buf_size % TWO_MB);
		hp_2m_needed *= bufs_needed;
		if (hp_2m_needed > num_2m) {
			errx(1, "Insufficient 2M huge pages. Need %d to run",
			     hp_2m_needed);
		} else {
			opts->buf_size = NEXT_MULTIPLE(opts->buf_size, TWO_MB);
			return hp_2m_needed;
		}
	} else if (opts->hugepages == HP_1G) {
		num_1g = get_free_hugepages(ONE_GB);
		hp_1g_needed =
			(opts->buf_size / ONE_GB) + !!(opts->buf_size % ONE_GB);
		hp_1g_needed *= bufs_needed;
		if (hp_1g_needed > num_1g) {
			errx(1, "Insufficient 1G huge pages. Need %d to run",
			     hp_1g_needed);
		} else {
			opts->buf_size = NEXT_MULTIPLE(opts->buf_size, ONE_GB);
			return hp_1g_needed;
		}
	}

no_hp:
	opts->buf_size = NEXT_MULTIPLE(opts->buf_size, sysconf(_SC_PAGE_SIZE));
	return 0;
}

void print_loc_opts(struct util_opts *opts, bool is_server)
{
	struct loc_util_opts *cli_opts;
	struct loc_util_opts *srv_opts;

	if (is_server) {
		cli_opts = &opts->rmt_opts;
		srv_opts = &opts->loc_opts;
	} else {
		cli_opts = &opts->loc_opts;
		srv_opts = &opts->rmt_opts;
	}

	printf("Device           : cxi%u\n", opts->loc_opts.dev_id);
	printf("Service ID       : %u\n", opts->loc_opts.svc_id);
	if (cli_opts->use_tx_gpu)
		printf("Client TX Mem    : GPU %d\n", cli_opts->tx_gpu);
	else
		printf("Client TX Mem    : System\n");
	if (opts->bidirectional) {
		if (cli_opts->use_rx_gpu)
			printf("Client RX Mem    : GPU %d\n", cli_opts->rx_gpu);
		else
			printf("Client RX Mem    : System\n");
	}
	if (cli_opts->use_tx_gpu ||
	    (cli_opts->use_rx_gpu && opts->bidirectional))
		printf("Client GPU Type  : %s\n",
		       gpu_names[cli_opts->gpu_type]);
	if (opts->bidirectional) {
		if (srv_opts->use_tx_gpu)
			printf("Server TX Mem    : GPU %d\n", srv_opts->tx_gpu);
		else
			printf("Server TX Mem    : System\n");
	}
	if (srv_opts->use_rx_gpu)
		printf("Server RX Mem    : GPU %d\n", srv_opts->rx_gpu);
	else
		printf("Server RX Mem    : System\n");
	if (srv_opts->use_rx_gpu ||
	    (srv_opts->use_tx_gpu && opts->bidirectional))
		printf("Server GPU Type  : %s\n",
		       gpu_names[srv_opts->gpu_type]);
}

void print_hugepage_opts(struct util_opts *opts, int num_hp)
{
	printf("Hugepages        : %s",
	       opts->hugepages == HP_DISABLED	? "Disabled" :
	       opts->hugepages == HP_NOT_APPLIC ? "Disabled - Not Applicable" :
	       "Enabled");
	if (opts->hugepages == HP_2M)
		printf(" (2M - %d pages)\n", num_hp);
	else if (opts->hugepages == HP_1G)
		printf(" (1G - %d pages)\n", num_hp);
	else
		printf("\n");
}

/* Poll for a specific type of event until the timeout expires. Once received,
 * either ACK the event, or optionally set the pointer param if it is not NULL
 * (in which case the caller must ACK the event). If the timespec pointer
 * param is not NULL, return the final timestamp to the user.
 */
int get_event(struct cxi_eq *eq, enum c_event_type type,
	      const union c_event **ret_event, struct timespec *ts,
	      uint64_t timeout_usec, struct cxil_dev *dev)
{
	int rc = 0;
	int ev_rc;
	const union c_event *event = NULL;
	struct timespec start_ts;
	struct timespec end_ts;
	uint64_t delta_usec;

	if (!eq)
		return -ENODEV;

	/* Poll for Event */
	rc = clock_gettime_or_counters(CLOCK_MONOTONIC_RAW, &start_ts, dev);
	if (rc < 0) {
		rc = -errno;
		fprintf(stderr, "clock_gettime() failed: %s\n", strerror(-rc));
		goto done;
	}
	while (!(event = cxi_eq_get_event(eq))) {
		rc = clock_gettime_or_counters(CLOCK_MONOTONIC_RAW, &end_ts,
					       dev);
		if (rc < 0) {
			rc = -errno;
			fprintf(stderr, "clock_gettime() failed: %s\n",
				strerror(-rc));
			goto done;
		}
		delta_usec = (end_ts.tv_sec - start_ts.tv_sec) * 1000000;
		delta_usec += (end_ts.tv_nsec - start_ts.tv_nsec) / 1000;
		if (timeout_usec != 0 && delta_usec >= timeout_usec) {
			/* If polling once, a timeout is not unexpected
			 * (suppress error message)
			 */
			if (timeout_usec != POLL_ONCE) {
				fprintf(stderr,
					"Timed out waiting for %s event\n",
					cxi_event_type_to_str(type));
			}
			return -ETIME;
		}
	}

	if (ts) {
		rc = clock_gettime_or_counters(CLOCK_MONOTONIC_RAW, &end_ts,
					       dev);
		if (rc < 0) {
			rc = -errno;
			fprintf(stderr, "clock_gettime() failed: %s\n",
				strerror(-rc));
			goto done;
		}
	}

	/* Verify */
	if (event->hdr.event_type != type) {
		fprintf(stderr, "Unexpected event type: %s (%u) Expected %s\n",
			cxi_event_type_to_str(event->hdr.event_type),
			event->hdr.event_type, cxi_event_type_to_str(type));
		rc = -ENOMSG;
	}
	ev_rc = cxi_event_rc(event);
	if (ev_rc != C_RC_OK) {
		fprintf(stderr, "event RC != RC_OK: %s\n",
			cxi_rc_to_str(ev_rc));
		rc = -ENOMSG;
	}

	/* Set timestamp */
	if (!rc && event && ts)
		*ts = end_ts;

done:
	/* ACK or return to be ACKed by caller */
	if (!rc && ret_event)
		*ret_event = event;
	else if (event)
		cxi_eq_ack_events(eq);

	return rc;
}

/* Wait for a CT event until the timeout expires. */
int wait_for_ct(struct cxi_eq *eq, uint64_t timeout_usec, char *label)
{
	int rc;

	rc = get_event(eq, C_EVENT_TRIGGERED_OP, NULL, NULL, timeout_usec,
		       NULL);
	if (rc)
		/* If polling only once, timeout is not unexpected (suppress
		 * error message)
		 */
		if (rc != -ECANCELED &&
		    (rc != -ETIME || timeout_usec != POLL_ONCE))
			fprintf(stderr,
				"Failed to get %s TRIGGERED_OP event: %s\n",
				label, strerror(-rc));

	return rc;
}

/* Increment a CT threshold and set up a triggered event */
int inc_ct(struct cxi_cq *cq, struct c_ct_cmd *cmd, size_t inc)
{
	int rc;

	if (!cq || !cmd)
		return -EINVAL;

	cmd->threshold += inc;
	rc = cxi_cq_emit_ct(cq, C_CMD_CT_TRIG_EVENT, cmd);
	if (rc) {
		fprintf(stderr, "Failed to issue CT_TRIG_EVENT command: %s\n",
			strerror(-rc));
		return rc;
	}
	cxi_cq_ring(cq);

	return rc;
}

/* Increment local and remote offset values with wraparound based on the main
 * initiator buffer.
 */
void inc_tx_buf_offsets(struct util_context *util, uint64_t *rmt, uint64_t *loc)
{
	uint64_t loc_addr_end_offset;

	*rmt += util->buf_granularity;
	*loc += util->buf_granularity;
	loc_addr_end_offset = *loc + util->buf_granularity -
			      (uintptr_t)util->cxi.ini_buf->buf - 1;
	if ((loc_addr_end_offset >= util->cxi.ini_buf->md->len) ||
	    (util->opts.max_buf_size != 0 &&
	     loc_addr_end_offset >= util->opts.max_buf_size)) {
		*rmt = 0;
		*loc = (uintptr_t)util->cxi.ini_buf->buf;
	}
}

/* Enable a Portals Table Entry */
int enable_pte(struct cxi_cq *cq, struct cxi_eq *eq, uint16_t ptn)
{
	int rc;
	union c_cmdu cmd;
	const union c_event *event;
	enum c_ptlte_state state;
	uint64_t tmo_usec = 1e+6;

	if (!cq || !eq)
		return -EINVAL;

	memset(&cmd, 0, sizeof(cmd));
	cmd.set_state.command.opcode = C_CMD_TGT_SETSTATE;
	cmd.set_state.ptlte_index = ptn;
	cmd.set_state.ptlte_state = C_PTLTE_ENABLED;

	rc = cxi_cq_emit_target(cq, &cmd);
	if (rc) {
		fprintf(stderr, "Failed to issue TGT_SETSTATE command: %s\n",
			strerror(-rc));
		return rc;
	}
	cxi_cq_ring(cq);

	rc = get_event(eq, C_EVENT_STATE_CHANGE, &event, NULL, tmo_usec, NULL);
	if (rc) {
		fprintf(stderr, "Failed to get STATE_CHANGE target event: %s\n",
			strerror(-rc));
		return rc;
	}
	state = event->tgt_long.initiator.state_change.ptlte_state;
	if (state != C_PTLTE_ENABLED) {
		fprintf(stderr, "Unexpected STATE_CHANGE state: %d\n", state);
		rc = -ENOMSG;
	}
	cxi_eq_ack_events(eq);

	return rc;
}

/* Append a List Entry to the Priority List of the specified PTE */
int append_le(struct cxi_cq *cq, struct cxi_eq *eq, struct ctx_buffer *ctx_buf,
	      size_t offset, uint32_t flags, uint16_t ptlte_index, uint16_t ct,
	      uint16_t buffer_id)
{
	/* Errata-2902: Setting ignore_bits[0]=1 to force Relaxed Ordering for
	 * the last TLP of Restricted packets
	 */
	return append_me(cq, eq, ctx_buf, offset, flags, ptlte_index, ct, 0, 0,
			 0x1, buffer_id);
}

/* Append a Matching List Entry to the Priority List of the specified PTE */
int append_me(struct cxi_cq *cq, struct cxi_eq *eq, struct ctx_buffer *ctx_buf,
	      size_t offset, uint32_t flags, uint16_t ptlte_index, uint16_t ct,
	      uint32_t match_id, uint64_t match_bits, uint64_t ignore_bits,
	      uint16_t buffer_id)
{
	int rc;
	union c_cmdu cmd = { 0 };
	uint64_t addr;

	if (!cq || !ctx_buf || !ctx_buf->buf || !ctx_buf->md)
		return -EINVAL;

	if (offset >= ctx_buf->md->len)
		return -EINVAL;

	cxi_target_cmd_setopts(&cmd.target, flags);
	cmd.command.opcode = C_CMD_TGT_APPEND;
	cmd.target.ptl_list = C_PTL_LIST_PRIORITY;
	cmd.target.ptlte_index = ptlte_index;
	cmd.target.lac = ctx_buf->md->lac;
	cmd.target.ct = ct;
	cmd.target.match_id = match_id;
	cmd.target.match_bits = match_bits;
	cmd.target.ignore_bits = ignore_bits;

	addr = CXI_VA_TO_IOVA(ctx_buf->md, ctx_buf->buf) + offset;
	cmd.target.start = addr;
	cmd.target.length = ctx_buf->md->len - offset;
	cmd.target.buffer_id = buffer_id;

	rc = cxi_cq_emit_target(cq, &cmd);
	if (rc) {
		fprintf(stderr, "Failed to issue APPEND command: %s\n",
			strerror(-rc));
		return rc;
	}
	cxi_cq_ring(cq);
	if (eq) {
		rc = get_event(eq, C_EVENT_LINK, NULL, NULL, SEC2USEC, NULL);
		if (rc) {
			fprintf(stderr, "Failed to get LINK event: %s\n",
				strerror(-rc));
			return rc;
		}
	}

	return rc;
}

/* Set the given CQ to use the default communication profile */
int set_to_dflt_cp(struct util_context *util, struct cxi_cq *cq)
{
	int rc;

	if (!util || !cq || !util->cxi.cp)
		return -EINVAL;

	rc = cxi_cq_emit_cq_lcid(cq, util->cxi.cp->lcid);
	if (rc)
		fprintf(stderr, "Failed to change CQ to default LCID: %s\n",
			strerror(-rc));
	return rc;
}

/* Set the given CQ to use the HRP communication profile */
int set_to_hrp_cp(struct util_context *util, struct cxi_cq *cq)
{
	int rc;

	if (!util || !cq || !util->cxi.hrp_cp)
		return -EINVAL;

	rc = cxi_cq_emit_cq_lcid(cq, util->cxi.hrp_cp->lcid);
	if (rc)
		fprintf(stderr, "Failed to change CQ to HRP LCID: %s\n",
			strerror(-rc));
	return rc;
}

/* Validate an AMO op and type combination */
void amo_validate_op_and_type(int atomic_op, int cswap_op, int atomic_type)
{
	bool invalid = false;

	switch (atomic_type) {
	case C_AMO_TYPE_FLOAT_T:
	case C_AMO_TYPE_DOUBLE_T:
		if ((atomic_op >= C_AMO_OP_LOR && atomic_op <= C_AMO_OP_BXOR) ||
		    atomic_op == C_AMO_OP_AXOR)
			invalid = true;
		break;
	case C_AMO_TYPE_FLOAT_COMPLEX_T:
	case C_AMO_TYPE_DOUBLE_COMPLEX_T:
		if (atomic_op != C_AMO_OP_SUM && atomic_op != C_AMO_OP_SWAP &&
		    !(atomic_op == C_AMO_OP_CSWAP &&
		      (cswap_op == C_AMO_OP_CSWAP_EQ ||
		       cswap_op == C_AMO_OP_CSWAP_NE)))
			invalid = true;
		break;
	case C_AMO_TYPE_UINT128_T:
		if (atomic_op != C_AMO_OP_CSWAP ||
		    cswap_op != C_AMO_OP_CSWAP_EQ)
			invalid = true;
		break;
	default:
		/* i8 through u64 valid for all */
		break;
	}

	if (invalid) {
		if (atomic_op == C_AMO_OP_CSWAP)
			errx(1, "Invalid Op and Type combination: %s %s %s",
			     amo_op_strs[atomic_op],
			     amo_cswap_op_strs[cswap_op],
			     amo_type_strs[atomic_type]);
		else
			errx(1, "Invalid Op and Type combination: %s %s",
			     amo_op_strs[atomic_op],
			     amo_type_strs[atomic_type]);
	}
}

/* Write operand words based on the AMO type */
static void amo_write_operand(struct amo_operand *op, int atomic_type,
			      uint8_t *word1_buf, uint8_t *word2_buf)
{
	switch (atomic_type) {
	case C_AMO_TYPE_UINT8_T:
		*(uint8_t *)word1_buf = op->op_uint;
		break;
	case C_AMO_TYPE_UINT16_T:
		*(uint16_t *)word1_buf = op->op_uint;
		break;
	case C_AMO_TYPE_UINT32_T:
		*(uint32_t *)word1_buf = op->op_uint;
		break;
	case C_AMO_TYPE_UINT64_T:
		*(uint64_t *)word1_buf = op->op_uint;
		break;
	case C_AMO_TYPE_INT8_T:
		*(int8_t *)word1_buf = op->op_int;
		break;
	case C_AMO_TYPE_INT16_T:
		*(int16_t *)word1_buf = op->op_int;
		break;
	case C_AMO_TYPE_INT32_T:
		*(int32_t *)word1_buf = op->op_int;
		break;
	case C_AMO_TYPE_INT64_T:
		*(int64_t *)word1_buf = op->op_int;
		break;
	case C_AMO_TYPE_FLOAT_COMPLEX_T:
		((float *)word1_buf)[1] = op->op_fp_imag;
	case C_AMO_TYPE_FLOAT_T:
		((float *)word1_buf)[0] = op->op_fp_real;
		break;
	case C_AMO_TYPE_DOUBLE_COMPLEX_T:
		*(double *)word2_buf = op->op_fp_imag;
	case C_AMO_TYPE_DOUBLE_T:
		*(double *)word1_buf = op->op_fp_real;
		break;
	case C_AMO_TYPE_UINT128_T:
		*(uint64_t *)word1_buf = op->op_uint;
		*(uint64_t *)word2_buf = op->op_uint_w2;
		break;
	}
}

static void amo_write_op1(struct util_context *util)
{
	struct util_opts *opts = &util->opts;
	struct amo_operand *op1 = &util->op1;
	struct cxi_context *cxi = &util->cxi;
	int i;
	uint64_t offset;

	if (!opts->use_idc)
		for (i = 0, offset = 0; i < opts->list_size &&
					cxi->ini_buf->md->len > (offset + 16);
		     i++, offset += util->buf_granularity) {
			amo_write_operand(
				op1, opts->atomic_type,
				(uint8_t *)((uintptr_t)cxi->ini_buf->buf +
					    offset),
				(uint8_t *)((uintptr_t)cxi->ini_buf->buf +
					    offset + 8));
		}
	else
		amo_write_operand(op1, opts->atomic_type,
				  (uint8_t *)&util->idc_cmd.idc_amo.op1_word1,
				  (uint8_t *)&util->idc_cmd.idc_amo.op1_word2);
}

static void amo_write_op2(struct util_context *util)
{
	struct util_opts *opts = &util->opts;
	struct amo_operand *op2 = &util->op2;

	if (!opts->use_idc)
		amo_write_operand(op2, opts->atomic_type,
				  (uint8_t *)&util->dma_cmd.dma_amo.op2_word1,
				  (uint8_t *)&util->dma_cmd.dma_amo.op2_word2);
	else
		amo_write_operand(op2, opts->atomic_type,
				  (uint8_t *)&util->idc_cmd.idc_amo.op2_word1,
				  (uint8_t *)&util->idc_cmd.idc_amo.op2_word2);
}

static void amo_write_tgt_op(struct util_context *util)
{
	struct util_opts *opts = &util->opts;
	struct amo_operand *tgt_op = &util->tgt_op;
	struct cxi_context *cxi = &util->cxi;
	int i;
	uint64_t offset;

	for (i = 0, offset = 0;
	     i < opts->list_size && cxi->tgt_buf->md->len > (offset + 16);
	     i++, offset += util->buf_granularity) {
		amo_write_operand(
			tgt_op, opts->atomic_type,
			(uint8_t *)((uintptr_t)cxi->tgt_buf->buf + offset),
			(uint8_t *)((uintptr_t)cxi->tgt_buf->buf + offset + 8));
	}
}

/* Get initial target 'operand' buffer value. These were chosen to maximize
 * AMO writes.
 */
void amo_init_tgt_op(struct util_context *util)
{
	struct util_opts *opts = &util->opts;
	struct amo_operand *op = &util->tgt_op;

	switch (opts->atomic_op) {
	case C_AMO_OP_MIN:
		/* MIN decrements from max value at target */
		switch (opts->atomic_type) {
		case C_AMO_TYPE_INT8_T:
			op->op_int = INT8_MAX;
			break;
		case C_AMO_TYPE_INT16_T:
			op->op_int = INT16_MAX;
			break;
		case C_AMO_TYPE_INT32_T:
			op->op_int = INT32_MAX;
			break;
		case C_AMO_TYPE_INT64_T:
			op->op_int = INT64_MAX;
			break;
		case C_AMO_TYPE_FLOAT_T:
			op->op_fp_real = FLT_MAX;
			break;
		case C_AMO_TYPE_DOUBLE_T:
			op->op_fp_real = DBL_MAX;
			break;
		default:
			op->op_uint = 0xFFFFFFFFFFFFFFFF;
			break;
		}
		break;
	case C_AMO_OP_MAX:
		/* MAX increments from min value at target */
		switch (opts->atomic_type) {
		case C_AMO_TYPE_INT8_T:
			op->op_int = INT8_MIN;
			break;
		case C_AMO_TYPE_INT16_T:
			op->op_int = INT16_MIN;
			break;
		case C_AMO_TYPE_INT32_T:
			op->op_int = INT32_MIN;
			break;
		case C_AMO_TYPE_INT64_T:
			op->op_int = INT64_MIN;
			break;
		case C_AMO_TYPE_FLOAT_T:
			op->op_fp_real = -FLT_MAX;
			break;
		case C_AMO_TYPE_DOUBLE_T:
			op->op_fp_real = -DBL_MAX;
			break;
		}
		break;
	case C_AMO_OP_CSWAP:
		/* CSWAP alternates values using always-true conditional */
		switch (opts->cswap_op) {
		case C_AMO_OP_CSWAP_EQ:
			if (opts->atomic_type == C_AMO_TYPE_UINT128_T)
				op->op_uint_w2 = 2;
		case C_AMO_OP_CSWAP_GE:
		case C_AMO_OP_CSWAP_LE:
			op->op_int = 2;
			op->op_uint = 2;
			op->op_fp_real = 2;
			if (opts->atomic_type == C_AMO_TYPE_FLOAT_COMPLEX_T ||
			    opts->atomic_type == C_AMO_TYPE_DOUBLE_COMPLEX_T)
				op->op_fp_imag = 2;
			break;
		case C_AMO_OP_CSWAP_LT:
			op->op_int = 3;
			op->op_uint = 3;
			op->op_fp_real = 3;
			break;
		}
		break;
	case C_AMO_OP_LAND:
		/* LAND saturates after a single op */
		op->op_int = 1;
		op->op_uint = 1;
		break;
	case C_AMO_OP_BAND:
		/* BAND saturates after flipping each bit */
		op->op_int = 0xFFFFFFFFFFFFFFFF;
		op->op_uint = 0xFFFFFFFFFFFFFFFF;
		break;
	default:
		/* All others start with target at 0 */
		break;
	}

	amo_write_tgt_op(util);
}

/* Get initial operand 1 value. These were chosen to maximize AMO writes */
void amo_init_op1(struct util_context *util)
{
	struct util_opts *opts = &util->opts;
	struct amo_operand *op1 = &util->op1;

	switch (opts->atomic_op) {
	case C_AMO_OP_MIN:
		/* MIN decrements from max value at target */
		switch (opts->atomic_type) {
		case C_AMO_TYPE_UINT8_T:
			op1->op_uint = 0xFE;
			break;
		case C_AMO_TYPE_UINT16_T:
			op1->op_uint = 0xFFFE;
			break;
		case C_AMO_TYPE_UINT32_T:
			op1->op_uint = 0xFFFFFFFE;
			break;
		case C_AMO_TYPE_UINT64_T:
			op1->op_uint = 0xFFFFFFFFFFFFFFFE;
			break;
		case C_AMO_TYPE_INT8_T:
			op1->op_int = INT8_MAX - 1;
			break;
		case C_AMO_TYPE_INT16_T:
			op1->op_int = INT16_MAX - 1;
			break;
		case C_AMO_TYPE_INT32_T:
			op1->op_int = INT32_MAX - 1;
			break;
		case C_AMO_TYPE_INT64_T:
			op1->op_int = INT64_MAX - 1;
			break;
		case C_AMO_TYPE_FLOAT_T:
			op1->op_fp_real = nextafter(FLT_MAX, -FLT_MAX);
			break;
		case C_AMO_TYPE_DOUBLE_T:
			op1->op_fp_real = nextafter(DBL_MAX, -DBL_MAX);
			break;
		}
		break;
	case C_AMO_OP_MAX:
		/* MAX increments from min value at target */
		switch (opts->atomic_type) {
		case C_AMO_TYPE_INT8_T:
			op1->op_int = INT8_MIN + 1;
			break;
		case C_AMO_TYPE_INT16_T:
			op1->op_int = INT16_MIN + 1;
			break;
		case C_AMO_TYPE_INT32_T:
			op1->op_int = INT32_MIN + 1;
			break;
		case C_AMO_TYPE_INT64_T:
			op1->op_int = INT64_MIN + 1;
			break;
		case C_AMO_TYPE_FLOAT_T:
			op1->op_fp_real = nextafter(-FLT_MAX, FLT_MAX);
			break;
		case C_AMO_TYPE_DOUBLE_T:
			op1->op_fp_real = nextafter(-DBL_MAX, DBL_MAX);
			break;
		default:
			op1->op_uint = 1;
		}
		break;
	case C_AMO_OP_LAND:
		/* LAND saturates to 0 after a single op */
		break;
	case C_AMO_OP_BAND:
		/* BAND saturates after flipping each bit to 0 */
		switch (opts->atomic_type) {
		case C_AMO_TYPE_UINT8_T:
		case C_AMO_TYPE_INT8_T:
			op1->op_int = 0xFE;
			op1->op_uint = 0xFE;
			break;
		case C_AMO_TYPE_UINT16_T:
		case C_AMO_TYPE_INT16_T:
			op1->op_int = 0xFFFE;
			op1->op_uint = 0xFFFE;
			break;
		case C_AMO_TYPE_UINT32_T:
		case C_AMO_TYPE_INT32_T:
			op1->op_int = 0xFFFFFFFE;
			op1->op_uint = 0xFFFFFFFE;
			break;
		case C_AMO_TYPE_UINT64_T:
		case C_AMO_TYPE_INT64_T:
			op1->op_int = 0xFFFFFFFFFFFFFFFE;
			op1->op_uint = 0xFFFFFFFFFFFFFFFE;
			break;
		}
		break;
	default:
		/* All others will start Op1 at 1 */
		op1->op_int = 1;
		op1->op_uint = 1;
		if (opts->atomic_type == C_AMO_TYPE_UINT128_T)
			op1->op_uint_w2 = 1;
		op1->op_fp_real = 1;
		if (opts->atomic_type == C_AMO_TYPE_FLOAT_COMPLEX_T ||
		    opts->atomic_type == C_AMO_TYPE_DOUBLE_COMPLEX_T)
			op1->op_fp_imag = 1;
		break;
	}

	amo_write_op1(util);
}

/* Get initial operand 2 value. These were chosen to maximize AMO writes */
void amo_init_op2(struct util_context *util)
{
	struct util_opts *opts = &util->opts;
	struct amo_operand *op2 = &util->op2;

	if (opts->atomic_op == C_AMO_OP_AXOR) {
		/* AXOR starts Op2 at 1 */
		op2->op_int = 1;
		op2->op_uint = 1;
	} else if (opts->atomic_op == C_AMO_OP_CSWAP) {
		/* CSWAP will always be true when comparing to target */
		if (opts->cswap_op == C_AMO_OP_CSWAP_GT ||
		    opts->cswap_op == C_AMO_OP_CSWAP_NE) {
			op2->op_int = 3;
			op2->op_uint = 3;
			op2->op_fp_real = 3;
		} else if (opts->cswap_op != C_AMO_OP_CSWAP_LT) {
			op2->op_int = 2;
			op2->op_uint = 2;
			if (opts->atomic_type == C_AMO_TYPE_UINT128_T)
				op2->op_uint_w2 = 2;
			op2->op_fp_real = 2;
			if (opts->atomic_type == C_AMO_TYPE_FLOAT_COMPLEX_T ||
			    opts->atomic_type == C_AMO_TYPE_DOUBLE_COMPLEX_T)
				op2->op_fp_imag = 2;
		}
	}

	amo_write_op2(util);
}

/* Update AMO operand 1 for the next iteration */
void amo_update_op1(struct util_context *util)
{
	struct util_opts *opts = &util->opts;
	struct amo_operand *op1 = &util->op1;
	bool updated = false;

	switch (opts->atomic_op) {
	case C_AMO_OP_SWAP:
	case C_AMO_OP_CSWAP:
		/* Alternate Op1 between 1 and 2 */
		switch (opts->atomic_type) {
		case C_AMO_TYPE_INT8_T:
		case C_AMO_TYPE_INT16_T:
		case C_AMO_TYPE_INT32_T:
		case C_AMO_TYPE_INT64_T:
			if (op1->op_int == 1)
				op1->op_int = 2;
			else
				op1->op_int = 1;
			break;
		case C_AMO_TYPE_UINT128_T:
			if (op1->op_uint_w2 == 1)
				op1->op_uint_w2 = 2;
			else
				op1->op_uint_w2 = 1;
		case C_AMO_TYPE_UINT8_T:
		case C_AMO_TYPE_UINT16_T:
		case C_AMO_TYPE_UINT32_T:
		case C_AMO_TYPE_UINT64_T:
			if (op1->op_uint == 1)
				op1->op_uint = 2;
			else
				op1->op_uint = 1;
			break;
		case C_AMO_TYPE_FLOAT_COMPLEX_T:
		case C_AMO_TYPE_DOUBLE_COMPLEX_T:
			if (op1->op_fp_imag == 1)
				op1->op_fp_imag = 2;
			else
				op1->op_fp_imag = 1;
		case C_AMO_TYPE_FLOAT_T:
		case C_AMO_TYPE_DOUBLE_T:
			if (op1->op_fp_real == 1)
				op1->op_fp_real = 2;
			else
				op1->op_fp_real = 1;
			break;
		}
		updated = true;
		break;
	case C_AMO_OP_MIN:
		/* Decrement */
		switch (opts->atomic_type) {
		case C_AMO_TYPE_UINT8_T:
		case C_AMO_TYPE_UINT16_T:
		case C_AMO_TYPE_UINT32_T:
		case C_AMO_TYPE_UINT64_T:
			op1->op_uint--;
			break;
		case C_AMO_TYPE_INT8_T:
		case C_AMO_TYPE_INT16_T:
		case C_AMO_TYPE_INT32_T:
		case C_AMO_TYPE_INT64_T:
			op1->op_int--;
			break;
		case C_AMO_TYPE_FLOAT_T:
			op1->op_fp_real = nextafter(op1->op_fp_real, -FLT_MAX);
			break;
		case C_AMO_TYPE_DOUBLE_T:
			op1->op_fp_real = nextafter(op1->op_fp_real, -DBL_MAX);
			break;
		}
		updated = true;
		break;
	case C_AMO_OP_MAX:
		/* Increment */
		switch (opts->atomic_type) {
		case C_AMO_TYPE_UINT8_T:
		case C_AMO_TYPE_UINT16_T:
		case C_AMO_TYPE_UINT32_T:
		case C_AMO_TYPE_UINT64_T:
			op1->op_uint++;
			break;
		case C_AMO_TYPE_INT8_T:
		case C_AMO_TYPE_INT16_T:
		case C_AMO_TYPE_INT32_T:
		case C_AMO_TYPE_INT64_T:
			op1->op_int++;
			break;
		case C_AMO_TYPE_FLOAT_T:
			op1->op_fp_real = nextafter(op1->op_fp_real, FLT_MAX);
			break;
		case C_AMO_TYPE_DOUBLE_T:
			op1->op_fp_real = nextafter(op1->op_fp_real, DBL_MAX);
			break;
		}
		updated = true;
		break;
	case C_AMO_OP_BOR:
	case C_AMO_OP_BAND:
		/* LShift */
		switch (opts->atomic_type) {
		case C_AMO_TYPE_UINT8_T:
		case C_AMO_TYPE_UINT16_T:
		case C_AMO_TYPE_UINT32_T:
		case C_AMO_TYPE_UINT64_T:
			op1->op_uint = op1->op_uint << 1;
			break;
		case C_AMO_TYPE_INT8_T:
		case C_AMO_TYPE_INT16_T:
		case C_AMO_TYPE_INT32_T:
		case C_AMO_TYPE_INT64_T:
			op1->op_int = op1->op_int << 1;
			break;
		}
		updated = true;
		break;
	}

	if (updated)
		amo_write_op1(util);
}

/* Update AMO operand 2 for the next iteration */
void amo_update_op2(struct util_context *util)
{
	struct util_opts *opts = &util->opts;
	struct amo_operand *op2 = &util->op2;
	bool updated = false;

	if (opts->atomic_op == C_AMO_OP_AXOR ||
	    (opts->atomic_op == C_AMO_OP_CSWAP &&
	     (opts->cswap_op == C_AMO_OP_CSWAP_EQ ||
	      opts->cswap_op == C_AMO_OP_CSWAP_GE ||
	      opts->cswap_op == C_AMO_OP_CSWAP_LE))) {
		/* Alternate Op2 between 1 and 2 */
		switch (opts->atomic_type) {
		case C_AMO_TYPE_INT8_T:
		case C_AMO_TYPE_INT16_T:
		case C_AMO_TYPE_INT32_T:
		case C_AMO_TYPE_INT64_T:
			if (op2->op_int == 1)
				op2->op_int = 2;
			else
				op2->op_int = 1;
			break;
		case C_AMO_TYPE_UINT8_T:
		case C_AMO_TYPE_UINT16_T:
		case C_AMO_TYPE_UINT32_T:
		case C_AMO_TYPE_UINT64_T:
			if (op2->op_uint == 1)
				op2->op_uint = 2;
			else
				op2->op_uint = 1;
			break;
		}
		updated = true;
	}

	if (updated)
		amo_write_op2(util);
}

/* GPU buffer malloc */
int gpu_malloc(struct ctx_buffer *win, size_t len)
{
	int rc;

	rc = g_malloc((void **)&win->buf, len);
	if (rc != 0) {
		fprintf(stderr, "%s failed %d\n", __func__, rc);
		return -ENOMEM;
	}
	return 0;
}

/* GPU buffer free */
int gpu_free(void *devPtr)
{
	int rc;

	rc = g_free(devPtr);
	if (rc != 0) {
		fprintf(stderr, "%s failed %d\n", __func__, rc);
		return -1;
	}
	return 0;
}

/* GPU buffer memset */
int gpu_memset(void *devPtr, int value, size_t size)
{
	int rc;

	rc = g_memset(devPtr, value, size);
	if (rc != 0) {
		fprintf(stderr, "%s failed %d\n", __func__, rc);
		return -ENOMEM;
	}
	return 0;
}

/* GPU buffer memcpy */
int gpu_memcpy(void *dst, const void *src, size_t size, int kind)
{
	int rc;

	rc = g_memcpy(dst, src, size, kind);
	if (rc != 0) {
		fprintf(stderr, "%s failed %d\n", __func__, rc);
		return -ENOMEM;
	}
	return 0;
}

/* Returns the number of GPUs present */
int get_gpu_device_count(void)
{
	int ret;
	int count;

	ret = g_device_count(&count);

	/* ret != 0 indicates no devices found */
	if (ret != 0)
		return 0;

	return count;
}

/* Returns the current default GPU device number */
int get_gpu_device(void)
{
	int rc;
	int dev_id;

	rc = g_get_device(&dev_id);
	if (rc != 0) {
		fprintf(stderr, "%s failed %d\n", __func__, rc);
		return -1;
	}
	return dev_id;
}

/* Set GPU device to be used for subsequent hip API calls */
int set_gpu_device(int dev_id)
{
	int rc;
	int cnt;
	int current_gpu_device;

	/* verify dev_id is a valid device number */
	cnt = get_gpu_device_count();
	if ((dev_id < 0) || (dev_id >= cnt)) {
		fprintf(stderr, "GPU must be in the range: 0 to %d.\n",
			(cnt - 1));
		return -1;
	}

	/* set the GPU device */
	rc = g_set_device(dev_id);
	if (rc != 0) {
		fprintf(stderr, "%s failed %d\n", __func__, rc);
		return -1;
	}

	/* verify GPU device number matches the expected value */
	current_gpu_device = get_gpu_device();
	if (current_gpu_device != dev_id) {
		fprintf(stderr, "%s failed. current device %d != dev_id %d\n",
			__func__, current_gpu_device, dev_id);
		return -1;
	}
	return 0;
}

/* Initialize GPU library functions for a particular GPU type */
int gpu_lib_init(enum gpu_types g_type)
{
#if defined(HAVE_HIP_SUPPORT) || defined(HAVE_CUDA_SUPPORT)

	int ret = -1;
	int count;

	switch (g_type) {
	/* AMD only supports HIP */
	case AMD:
#if defined(HAVE_HIP_SUPPORT)
		ret = hip_lib_init();
#endif
		break;

	/* NVIDIA supports CUDA and HIP when compiled for NVIDIA */
	case NVIDIA:
#if defined(HAVE_CUDA_SUPPORT)
		ret = cuda_lib_init();
#elif defined(HAVE_HIP_SUPPORT) && (defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_NVCC__))
		ret = hip_lib_init();
#endif
		break;
	case INTEL:
#if defined(HAVE_ZE_SUPPORT)
		ret = ze_lib_init();
#endif
		break;

	default:
		fprintf(stderr, "Unknown GPU Type\n");
		break;
	}

	if (ret != 0)
		return ret;

	ret = g_device_count(&count);
	if (ret != 0) {
		fprintf(stderr, "Get device count request failed\n");
		return -1;
	} else if (count > 0) {
		return 0;
	}
	fprintf(stderr, "No GPUs found\n");
	return -1;

#else
	printf("No GPU libraries found\n");
	return -1;
#endif /* #if defined(HAVE_HIP_SUPPORT) || defined(HAVE_CUDA_SUPPORT) || defined(HAVE_ZE_SUPPORT) */
}

/* clean up GPU library for the given GPU type */
void gpu_lib_fini(enum gpu_types g_type)
{
#if defined(HAVE_HIP_SUPPORT) || defined(HAVE_CUDA_SUPPORT) || defined(HAVE_ZE_SUPPORT)

	switch (g_type) {
	/* AMD only supports HIP */
	case AMD:
#if defined(HAVE_HIP_SUPPORT)
		hip_lib_fini();
#endif
		break;

	/* NVIDIA supports CUDA and HIP when compiled for NVIDIA */
	case NVIDIA:
#if defined(HAVE_CUDA_SUPPORT)
		cuda_lib_fini();
#elif defined(HAVE_HIP_SUPPORT) && (defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_NVCC__))
		hip_lib_fini();
#endif
		break;
	case INTEL:
#if defined(HAVE_ZE_SUPPORT)
		ze_lib_fini();
#endif
		break;

	default:
		fprintf(stderr, "Unknown GPU Type\n");
		break;
	}
#else
	fprintf(stderr, "No GPU libraries found\n");
#endif /* #if defined(HAVE_HIP_SUPPORT) || defined(HAVE_CUDA_SUPPORT) || defined(HAVE_ZE_SUPPORT) */
}

/* return the GPU type for the given string */
int get_gpu_type(char *gpu_name)
{
	if (strcasecmp(gpu_name, gpu_names[AMD]) == 0)
		return AMD;
	else if (strcasecmp(gpu_name, gpu_names[NVIDIA]) == 0)
		return NVIDIA;
	else if (strcasecmp(gpu_name, gpu_names[INTEL]) == 0)
		return INTEL;
	else
		return -1;
}
