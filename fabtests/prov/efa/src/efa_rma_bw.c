/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

/*
 * EFA-specific RMA bandwidth test.
 *
 * This test measures RMA bandwidth with support for EFA-specific features
 * such as the FI_EFA_WR_HIGH_PPS flag. It currently supports write,
 * writedata, and read operations.
 *
 * Unlike fi_rma_bw, this test uses a nonblocking benchmark loop that
 * interleaves posting and completion polling to keep the pipeline full,
 * similar to the approach used by rdma-core/perftest. This avoids blocking
 * at window boundaries and maximizes throughput.
 *
 * Multi-EP support:
 *   Multiple endpoints (--num-eps / -q) share a single CQ pair (txcq/rxcq)
 *   and AV. Each EP independently tracks its own posted and completed
 *   operation counts (per_ep_posted[] / per_ep_completed[]), enabling
 *   per-EP flow control: an EP may only have up to window_size operations
 *   in flight at any time.
 *
 * Per-EP completion attribution:
 *   Each posted operation carries an efa_rma_bw_ctx containing the EP index.
 *   On completion, the CQ entry's op_context is used (via container_of) to
 *   recover the efa_rma_bw_ctx and attribute the completion to the correct EP.
 *
 *   Context pool slots are partitioned per-EP:
 *     slot = ep_idx * window_size + (per_ep_posted[ep_idx] % window_size)
 *   This ensures an EP's in-flight contexts are never overwritten by another
 *   EP's posts.
 *
 *   For the writedata receiver (unsolicited write-with-imm without posted
 *   receives, i.e. !FI_RX_CQ_DATA), completions have no user context, so
 *   per_ep_completed is passed as NULL and only global counting is performed.
 *
 * FI_MORE batching (--post-list):
 *   When post_list > 1, FI_MORE is set on consecutive posts to the same EP
 *   to batch doorbell rings. FI_MORE is cleared (doorbell fires) when:
 *     - The post count hits a post_list boundary (per_ep_posted % post_list == 0)
 *     - The EP's per-iteration quota is reached (last post for that EP)
 *     - The EP's window is full (next post would exceed window_size outstanding)
 *   This guarantees the provider always receives a final non-FI_MORE post to
 *   flush the batch, preventing hangs from un-rung doorbells.
 *
 * Loop structure (initiator / tx side):
 *   Follows the perftest pattern: the outer while loop runs until all posts
 *   are issued and completed. Each pass iterates over ALL EPs (for-loop),
 *   posting as many operations as each EP's window allows. This prevents
 *   starvation — a full EP doesn't block others from making progress. After
 *   the posting pass, the shared CQ is polled once to harvest completions.
 *
 * Loop structure (writedata server / rx side):
 *   Pre-posts window_size receives per EP. Then polls rxcq and reposts to
 *   EPs that have room, using the same per-EP window logic. FI_MORE is
 *   applied to consecutive reposts on the same EP.
 *
 * Usage:
 *   Server: fi_efa_rma_bw
 *   Client: fi_efa_rma_bw -H <server_addr>
 *
 * Options:
 *   --high-pps        Enable FI_EFA_WR_HIGH_PPS flag on writes.
 *   -o write|writedata|read  Select RMA operation (default: write).
 *   --post-list <n>   Batch n posts per doorbell using FI_MORE (default: 1).
 *   -q <n>, --num-eps <n>  Number of endpoints/QPs (default: 1).
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <getopt.h>

#include <rdma/fi_errno.h>
#include <rdma/fi_ext.h>
#include <rdma/fi_ext_efa.h>

#include <shared.h>
#include "benchmarks/benchmark_shared.h"


#define EFA_RMA_BW_CQ_POLL_BATCH 16
#define EFA_RMA_BW_MAX_EPS 64

/*
 * Per-operation context that embeds the EP index so CQ completions
 * can be attributed to specific EPs.
 * ep_idx is placed before fi_context2 to avoid provider clobbering it.
 */
struct efa_rma_bw_ctx {
	int ep_idx;
	int pad;
	struct fi_context2 context;
};

#define EFA_RMA_BW_CTX_FROM_OP_CONTEXT(ptr) \
	container_of(ptr, struct efa_rma_bw_ctx, context)

static struct efa_rma_bw_ctx *tx_ctx_pool;
static struct efa_rma_bw_ctx *rx_ctx_pool;

static int use_high_pps;
static int post_list = 1;
static int num_eps = 1;
static struct fid_ep *eps[EFA_RMA_BW_MAX_EPS];
static fi_addr_t remote_addrs[EFA_RMA_BW_MAX_EPS];

static ssize_t post_rma(struct fid_ep *target_ep, fi_addr_t target_addr,
			char *buf, size_t size,
			struct fi_rma_iov *remote, void *context,
			uint64_t base_flags)
{
	struct fi_msg_rma msg;
	struct iovec msg_iov;
	struct fi_rma_iov rma_iov;
	uint64_t flags = base_flags;
	ssize_t ret;

	msg_iov.iov_base = buf;
	msg_iov.iov_len = size;
	msg.msg_iov = &msg_iov;
	msg.desc = &mr_desc;
	msg.iov_count = 1;
	rma_iov.addr = remote->addr + (buf - (opts.rma_op == FT_RMA_READ ? rx_buf : tx_buf));
	rma_iov.len = size;
	rma_iov.key = remote->key;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.addr = target_addr;
	msg.context = context;

	if (opts.rma_op == FT_RMA_READ) {
		msg.data = 0;
		ret = fi_readmsg(target_ep, &msg, flags);
	} else {
		if (use_high_pps)
			flags |= FI_EFA_WR_HIGH_PPS;
		if (opts.rma_op == FT_RMA_WRITEDATA) {
			flags |= FI_REMOTE_CQ_DATA;
			msg.data = remote_cq_data;
		} else {
			msg.data = 0;
		}
		ret = fi_writemsg(target_ep, &msg, flags);
	}

	return ret;
}

/*
 * Poll CQ for completions in a nonblocking manner.
 * If per_ep_completed is non-NULL, each completion's op_context is used to
 * recover the efa_rma_bw_ctx and credit the correct EP. For the target side
 * of unsolicited write recv, there is no rx buffer post and no per-EP completion
 * needed (because it is used as the credit to repost rx buffer), the per_ep_completed
 * is not needed and will be passed as NULL.
 * Returns the number of completions harvested, or negative on error.
 */
static int bw_comp_nonblocking(struct fid_cq *cq, uint64_t *cq_cntr,
			       int *completed_cnt,
			       int *per_ep_completed)
{
	int ret, cnt = 0, i;
	struct fi_cq_data_entry comp[EFA_RMA_BW_CQ_POLL_BATCH];
	struct efa_rma_bw_ctx *ctx;

	while ((ret = fi_cq_read(cq, comp, EFA_RMA_BW_CQ_POLL_BATCH)) > 0) {
		if (per_ep_completed) {
			for (i = 0; i < ret; i++) {
				ctx = EFA_RMA_BW_CTX_FROM_OP_CONTEXT(comp[i].op_context);
				per_ep_completed[ctx->ep_idx]++;
			}
		}
		(*completed_cnt) += ret;
		(*cq_cntr) += ret;
		cnt += ret;
	}

	if (ret == -FI_EAVAIL) {
		ret = ft_cq_readerr(cq);
		return ret;
	}

	if (ret < 0 && ret != -FI_EAGAIN) {
		FT_PRINTERR("fi_cq_read", ret);
		return ret;
	}

	return cnt;
}

/*
 * Post a receive buffer on the given EP. The context slot is determined
 * per-EP (ep_idx * window_size + posted % window_size) to avoid cross-EP
 * slot reuse. The ep_idx is stamped into the context so completions can
 * be attributed back to this EP.
 */
static int post_rx(int ep_idx, int *per_ep_posted, int *per_ep_completed,
		   int *posted_cnt, uint64_t rx_flags)
{
	int slot = ep_idx * opts.window_size +
		   (per_ep_posted[ep_idx] % opts.window_size);
	struct iovec iov = {
		.iov_base = rx_buf,
		.iov_len = FT_MAX_CTRL_MSG + ft_rx_prefix_size(),
	};
	struct fi_msg msg = {
		.msg_iov = &iov,
		.desc = &mr_desc,
		.iov_count = 1,
		.addr = FI_ADDR_UNSPEC,
		.context = &rx_ctx_pool[slot].context,
	};
	int ret;

	rx_ctx_pool[slot].ep_idx = ep_idx;
	per_ep_posted[ep_idx]++;

	ret = fi_recvmsg(eps[ep_idx], &msg, rx_flags);
	if (ret) {
		per_ep_posted[ep_idx]--;
		return ret;
	}
	(*posted_cnt)++;
	return 0;
}

/*
 * Unified post/poll loop for both TX (initiator) and RX (writedata server).
 *
 * TX side: posts RMA ops round-robin across EPs, polls txcq.
 * RX side: pre-posts recvs, polls rxcq, reposts on completion.
 *
 * iters_per_ep: number of operations per EP to complete.
 * do_measure: if true, calls ft_start()/ft_stop() around the loop.
 */
static int run_loop(struct fi_rma_iov *remote, size_t rma_start_offset,
		    int iters_per_ep, bool do_measure)
{
	int ret, posted_cnt = 0, completed_cnt = 0;
	int per_ep_posted[EFA_RMA_BW_MAX_EPS] = {0};
	int per_ep_completed[EFA_RMA_BW_MAX_EPS] = {0};
	int total_posts = iters_per_ep * num_eps;
	size_t offset;
	char *buf;
	uint64_t flags;

	if (opts.rma_op == FT_RMA_WRITEDATA && !opts.dst_addr) {
		/* Server side for writedata: pre-post rx buffers round-robin. */
		if (fi->rx_attr->mode & FI_RX_CQ_DATA) {
			int pre_post_limit = MIN(opts.window_size, iters_per_ep);
			for (int ep_idx = 0; ep_idx < num_eps; ep_idx++) {
				while (per_ep_posted[ep_idx] < pre_post_limit) {
					ret = post_rx(ep_idx, per_ep_posted,
						      per_ep_completed,
						      &posted_cnt, 0);
					if (ret == -FI_EAGAIN)
						break;
					if (ret)
						return ret;
				}
			}
		}

		if (do_measure)
			ft_start();

		/* Poll rxcq for completions, reposting to same EP. */
		while (completed_cnt < total_posts) {
			ret = bw_comp_nonblocking(rxcq, &rx_cq_cntr,
						  &completed_cnt,
						  (fi->rx_attr->mode & FI_RX_CQ_DATA) ?
						  per_ep_completed : NULL);
			if (ret < 0)
				return ret;

			if ((fi->rx_attr->mode & FI_RX_CQ_DATA) && ret > 0) {
				/* Repost to EPs that have room in their window */
				for (int ep_idx = 0; ep_idx < num_eps; ep_idx++) {
					while (per_ep_posted[ep_idx] <
					       iters_per_ep &&
					       (per_ep_posted[ep_idx] -
					        per_ep_completed[ep_idx]) <
					       opts.window_size) {
						/*
						 * FI_MORE for rx: peek ahead to
						 * check if the next post on this
						 * EP will also proceed (not at
						 * post_list boundary, iteration
						 * limit, or window limit).
						 */
						uint64_t rx_flags = 0;

						if (post_list > 1 &&
						    (per_ep_posted[ep_idx] + 1) % post_list &&
						    per_ep_posted[ep_idx] + 1 <
						    iters_per_ep &&
						    (per_ep_posted[ep_idx] + 1 -
						     per_ep_completed[ep_idx]) <
						    opts.window_size)
							rx_flags = FI_MORE;

						ret = post_rx(ep_idx,
							      per_ep_posted,
							      per_ep_completed,
							      &posted_cnt,
							      rx_flags);
						if (ret == -FI_EAGAIN)
							break;
						if (ret)
							return ret;
					}
				}
			}
		}
	} else {
		if (do_measure)
			ft_start();

		/* Initiator side: post RMA ops, try all EPs each pass (perftest style) */
		while (posted_cnt < total_posts ||
		       completed_cnt < total_posts) {
			for (int ep_idx = 0; ep_idx < num_eps; ep_idx++) {
				while (per_ep_posted[ep_idx] < iters_per_ep &&
				       (per_ep_posted[ep_idx] -
				        per_ep_completed[ep_idx]) <
				       opts.window_size) {
					int slot = ep_idx * opts.window_size +
						   (per_ep_posted[ep_idx] %
						    opts.window_size);

					offset = rma_start_offset +
						 (per_ep_posted[ep_idx] %
						  opts.window_size) *
							opts.transfer_size;

					buf = (opts.rma_op == FT_RMA_READ) ?
					      rx_buf + offset : tx_buf + offset;

					tx_ctx_pool[slot].ep_idx = ep_idx;
					per_ep_posted[ep_idx]++;

					/*
					 * FI_MORE: batch posts within the same EP.
					 * Set only when all three conditions hold:
					 * 1. Not at a post_list boundary
					 * 2. Not at the EP's iteration limit
					 * 3. Window won't be full after this post
					 * This ensures the next iteration of the
					 * inner while will also execute, so a
					 * non-FI_MORE post always follows to ring
					 * the doorbell.
					 */
					flags = 0;
					if (post_list > 1 &&
					    per_ep_posted[ep_idx] % post_list &&
					    per_ep_posted[ep_idx] <
					    iters_per_ep &&
					    (per_ep_posted[ep_idx] -
					     per_ep_completed[ep_idx]) <
					    opts.window_size)
						flags = FI_MORE;

					ret = post_rma(eps[ep_idx],
							remote_addrs[ep_idx],
							buf, opts.transfer_size,
							remote,
							&tx_ctx_pool[slot].context,
							flags);
					if (ret == -FI_EAGAIN) {
						per_ep_posted[ep_idx]--;
						break;
					}
					if (ret)
						return ret;
					posted_cnt++;
				}
			}

			ret = bw_comp_nonblocking(txcq, &tx_cq_cntr,
						  &completed_cnt,
						  per_ep_completed);
			if (ret < 0)
				return ret;
		}
	}

	if (do_measure)
		ft_stop();

	return 0;
}

static int bandwidth_rma_efa(struct fi_rma_iov *remote)
{
	int ret;
	size_t rma_start_offset;
	int pool_size = opts.window_size * num_eps;

	tx_ctx_pool = calloc(pool_size, sizeof(*tx_ctx_pool));
	rx_ctx_pool = calloc(pool_size, sizeof(*rx_ctx_pool));
	if (!tx_ctx_pool || !rx_ctx_pool) {
		ret = -FI_ENOMEM;
		goto out_free;
	}

	rma_start_offset = FT_RMA_SYNC_MSG_BYTES +
			   MAX(ft_tx_prefix_size(), ft_rx_prefix_size());

	/* Warmup */
	ret = ft_sync();
	if (ret)
		goto out_free;

	ret = run_loop(remote, rma_start_offset,
		       opts.warmup_iterations, false);
	if (ret)
		goto out_free;

	/* Measurement */
	ret = ft_sync();
	if (ret)
		goto out_free;

	ret = run_loop(remote, rma_start_offset,
		       opts.iterations, true);
	if (ret)
		goto out_free;

	if (opts.machr)
		show_perf_mr(opts.transfer_size, opts.iterations * num_eps,
			     &start, &end, 1, opts.argc, opts.argv);
	else
		show_perf(NULL, opts.transfer_size, opts.iterations * num_eps,
			  &start, &end, 1);

	ret = 0;
out_free:
	free(tx_ctx_pool);
	free(rx_ctx_pool);
	tx_ctx_pool = NULL;
	rx_ctx_pool = NULL;
	return ret;
}

static int run(void)
{
	int i, ret;

	/*
	 * Use FI_CQ_FORMAT_DATA so the CQ entry type matches the
	 * fi_cq_data_entry buffer in bw_comp_nonblocking. Without this,
	 * ft_init_fabric defaults to FI_CQ_FORMAT_CONTEXT (no FI_TAGGED cap),
	 * causing a mismatch with our fi_cq_read calls.
	 */
	cq_attr.format = FI_CQ_FORMAT_DATA;

	ret = ft_init_fabric();
	if (ret)
		return ret;

	/* eps[0] is the default ep created by ft_init_fabric */
	eps[0] = ep;
	remote_addrs[0] = remote_fi_addr;

	/* Create additional EPs, all bound to the same CQs and AV */
	for (i = 1; i < num_eps; i++) {
		ret = fi_endpoint(domain, fi, &eps[i], NULL);
		if (ret) {
			FT_PRINTERR("fi_endpoint", ret);
			return ret;
		}
		ret = ft_enable_ep(eps[i], eq, av, txcq, rxcq,
				   txcntr, rxcntr, rma_cntr);
		if (ret)
			return ret;
		ret = ft_init_av_addr(av, eps[i], &remote_addrs[i]);
		if (ret)
			return ret;
	}

	ret = ft_exchange_keys(&remote);
	if (ret)
		return ret;

	/*
	 * ft_exchange_keys() leaves a pre-posted receive (context = &rx_ctx)
	 * on eps[0]. Consume it with a dummy message so it doesn't get
	 * matched by writedata completions producing a bogus op_context.
	 */
	if (opts.dst_addr) {
		ret = ft_post_tx(ep, remote_fi_addr, 1, NO_CQ_DATA, &tx_ctx);
		if (ret)
			return ret;
		ret = ft_get_tx_comp(tx_seq);
	} else {
		ret = ft_get_rx_comp(rx_seq);
	}
	if (ret)
		return ret;

	if (!(opts.options & FT_OPT_SIZE)) {
		for (i = 0; i < TEST_CNT; i++) {
			if (!ft_use_size(i, opts.sizes_enabled))
				continue;
			opts.transfer_size = test_size[i].size;
			init_test(&opts, test_name, sizeof(test_name));
			ret = bandwidth_rma_efa(&remote);
			if (ret)
				goto out;
		}
	} else {
		init_test(&opts, test_name, sizeof(test_name));
		ret = bandwidth_rma_efa(&remote);
		if (ret)
			goto out;
	}

	ft_finalize();
out:
	for (i = 1; i < num_eps; i++) {
		if (eps[i])
			fi_close(&eps[i]->fid);
	}
	return ret;
}

enum {
	OPT_HIGH_PPS = 256,
	OPT_POST_LIST,
	OPT_NUM_EPS,
};

static struct option efa_extra_opts[] = {
	{"high-pps", no_argument, NULL, OPT_HIGH_PPS},
	{"post-list", required_argument, NULL, OPT_POST_LIST},
	{"num-eps", required_argument, NULL, OPT_NUM_EPS},
	{0, 0, 0, 0}
};

static struct option *efa_long_opts;

/*
 * Build a merged long options table by prepending EFA-specific options
 * to the shared fabtests long_opts. This allows getopt_long to parse
 * both EFA-specific (e.g. --high-pps) and shared (e.g. --no-rx-cq-data)
 * long options in a single call.
 */
static void build_long_opts(void)
{
	int shared_cnt, i;
	int extra_cnt = sizeof(efa_extra_opts) / sizeof(efa_extra_opts[0]) - 1;

	for (shared_cnt = 0; long_opts[shared_cnt].name; shared_cnt++)
		;
	efa_long_opts = calloc(shared_cnt + extra_cnt + 1, sizeof(struct option));
	for (i = 0; i < extra_cnt; i++)
		efa_long_opts[i] = efa_extra_opts[i];
	for (i = 0; i < shared_cnt; i++)
		efa_long_opts[extra_cnt + i] = long_opts[i];
}

int main(int argc, char **argv)
{
	int op, ret, cleanup_ret;

	opts = INIT_OPTS;
	opts.options |= FT_OPT_BW;
	opts.rma_op = FT_RMA_WRITE;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	hints->caps = FI_MSG | FI_RMA;
	hints->domain_attr->resource_mgmt = FI_RM_ENABLED;
	hints->mode = FI_CONTEXT | FI_CONTEXT2;
	hints->domain_attr->threading = FI_THREAD_DOMAIN;
	hints->addr_format = opts.address_format;

	build_long_opts();

	while ((op = getopt_long(argc, argv, "hq:" CS_OPTS INFO_OPTS API_OPTS
			    BENCHMARK_OPTS, efa_long_opts,
			    &lopt_idx)) != -1) {
		switch (op) {
		case OPT_HIGH_PPS:
			use_high_pps = 1;
			break;
		case OPT_POST_LIST:
			post_list = atoi(optarg);
			break;
		case OPT_NUM_EPS:
		case 'q':
			num_eps = atoi(optarg);
			if (num_eps < 1 || num_eps > EFA_RMA_BW_MAX_EPS) {
				fprintf(stderr, "num-eps must be 1-%d\n",
					EFA_RMA_BW_MAX_EPS);
				return EXIT_FAILURE;
			}
			break;
		case '?':
		case 'h':
			ft_csusage(argv[0],
				   "EFA RMA bandwidth test.");
			ft_benchmark_usage();
			FT_PRINT_OPTS_USAGE("-o <op>",
				"RMA op type: write|writedata|read (default: write)");
			FT_PRINT_OPTS_USAGE("--high-pps",
				"Enable FI_EFA_WR_HIGH_PPS flag on writes");
			FT_PRINT_OPTS_USAGE("--post-list <n>",
				"Batch n posts per doorbell using FI_MORE (default: 1)");
			FT_PRINT_OPTS_USAGE("-q <n>, --num-eps <n>",
				"Number of endpoints/QPs (default: 1)");
			fprintf(stderr, "Note: read/write bw tests are bidirectional.\n"
					"      writedata bw test is unidirectional"
					" from the client side.\n");
			ft_longopts_usage();
			return EXIT_FAILURE;
		default:
			if (!ft_parse_long_opts(op, optarg))
				continue;
			ft_parse_benchmark_opts(op, optarg);
			ft_parse_api_opts(op, optarg, hints, &opts);
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			break;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->domain_attr->mr_mode = opts.mr_mode;
	hints->tx_attr->tclass = FI_TC_BULK_DATA;
	/* Using OOB sync to not mess up with the tx/rx seq cntrs in fabtests common code */
	opts.options |= FT_OPT_OOB_SYNC;

	const char *op_str = "WRITE";
	if (opts.rma_op == FT_RMA_WRITEDATA)
		op_str = "WRITEDATA";
	else if (opts.rma_op == FT_RMA_READ)
		op_str = "READ";

	if (use_high_pps)
		printf("High PPS mode: ENABLED\n");
	else
		printf("High PPS mode: DISABLED\n");

	printf("RMA op: %s\n", op_str);
	printf("Num EPs: %d\n", num_eps);

	ret = run();

	cleanup_ret = ft_free_res();
	return -(ret ? ret : cleanup_ret);
}
