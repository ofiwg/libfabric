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
 * Usage:
 *   Server: fi_efa_rma_bw
 *   Client: fi_efa_rma_bw -H <server_addr>
 *
 * Options:
 *   --high-pps        Enable FI_EFA_WR_HIGH_PPS flag on writes.
 *   -o write|writedata|read  Select RMA operation (default: write).
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

static int use_high_pps;
static int post_list = 1;

static ssize_t post_rma(char *buf, size_t size,
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
	msg.addr = remote_fi_addr;
	msg.context = context;

	if (opts.rma_op == FT_RMA_READ) {
		msg.data = 0;
		ret = fi_readmsg(ep, &msg, flags);
	} else {
		if (use_high_pps)
			flags |= FI_EFA_WR_HIGH_PPS;
		if (opts.rma_op == FT_RMA_WRITEDATA) {
			flags |= FI_REMOTE_CQ_DATA;
			msg.data = remote_cq_data;
		} else {
			msg.data = 0;
		}
		ret = fi_writemsg(ep, &msg, flags);
	}

	return ret;
}

static int bw_comp_nonblocking(struct fid_cq *cq, uint64_t *cq_cntr,
			       int *completed_cnt)
{
	int ret, cnt = 0;
	struct fi_cq_data_entry comp[EFA_RMA_BW_CQ_POLL_BATCH];

	while ((ret = fi_cq_read(cq, comp, EFA_RMA_BW_CQ_POLL_BATCH)) > 0) {
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

static int bandwidth_rma_efa(struct fi_rma_iov *remote)
{
	int ret, posted_cnt = 0, completed_cnt = 0;
	size_t offset;
	size_t rma_start_offset;
	int total_iterations = opts.iterations + opts.warmup_iterations;
	bool warmup_done = false;
	char *buf;
	uint64_t flags;

	ret = ft_sync();
	if (ret)
		return ret;

	rma_start_offset = FT_RMA_SYNC_MSG_BYTES +
			   MAX(ft_tx_prefix_size(), ft_rx_prefix_size());

	if (opts.rma_op == FT_RMA_WRITEDATA && !opts.dst_addr) {
		/* Server side for writedata: pre-post all rx buffers up to
		 * window_size before the loop starts, matching perftest behavior.
		 */
		if (fi->rx_attr->mode & FI_RX_CQ_DATA) {
			for (posted_cnt = 0; posted_cnt < opts.window_size &&
			     posted_cnt < total_iterations; posted_cnt++) {
				ret = ft_post_rx(ep, 0,
					&rx_ctx_arr[posted_cnt %
						opts.window_size].context);
				if (ret)
					return ret;
			}
		}

		/* Poll rxcq for completions, reposting as they complete. */
		while (completed_cnt < total_iterations) {
			if (!warmup_done &&
			    completed_cnt >= opts.warmup_iterations) {
				ft_start();
				warmup_done = true;
			}
			ret = bw_comp_nonblocking(rxcq, &rx_cq_cntr,
						  &completed_cnt);
			if (ret < 0)
				return ret;
			if (fi->rx_attr->mode & FI_RX_CQ_DATA) {
				int i;
				for (i = 0; i < ret &&
				     posted_cnt < total_iterations; i++) {
					int err = ft_post_rx(ep, 0,
						&rx_ctx_arr[posted_cnt %
							opts.window_size].context);
					if (err)
						return err;
					posted_cnt++;
				}
			}
		}
	} else {
		/* Initiator side: post RMA ops and poll completions */
		while (posted_cnt < total_iterations ||
		       completed_cnt < total_iterations) {
			if (!warmup_done &&
			    completed_cnt >= opts.warmup_iterations) {
				ft_start();
				warmup_done = true;
			}

			while (posted_cnt < total_iterations &&
			       (posted_cnt - completed_cnt) <
					opts.window_size) {
				offset = rma_start_offset +
					 (posted_cnt % opts.window_size) *
						opts.transfer_size;

				buf = (opts.rma_op == FT_RMA_READ) ?
				      rx_buf + offset : tx_buf + offset;

				flags = (post_list > 1 &&
					 (posted_cnt + 1) % post_list &&
					 posted_cnt + 1 < total_iterations) ?
					FI_MORE : 0;

				ret = post_rma(buf,
						opts.transfer_size, remote,
						&tx_ctx_arr[posted_cnt %
							opts.window_size].context,
						flags);
				if (ret == -FI_EAGAIN)
					break;
				if (ret)
					return ret;
				posted_cnt++;
			}

			ret = bw_comp_nonblocking(txcq, &tx_cq_cntr,
						  &completed_cnt);
			if (ret < 0)
				return ret;
		}
	}

	ft_stop();
	if (opts.machr)
		show_perf_mr(opts.transfer_size, opts.iterations, &start, &end,
			     1, opts.argc, opts.argv);
	else
		show_perf(NULL, opts.transfer_size, opts.iterations, &start,
			  &end, 1);

	return 0;
}

static int run(void)
{
	int i, ret;

	ret = ft_init_fabric();
	if (ret)
		return ret;

	ret = ft_exchange_keys(&remote);
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
	return ret;
}

enum {
	OPT_HIGH_PPS = 256,
	OPT_POST_LIST,
};

static struct option efa_extra_opts[] = {
	{"high-pps", no_argument, NULL, OPT_HIGH_PPS},
	{"post-list", required_argument, NULL, OPT_POST_LIST},
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

	while ((op = getopt_long(argc, argv, "h" CS_OPTS INFO_OPTS API_OPTS
			    BENCHMARK_OPTS, efa_long_opts,
			    &lopt_idx)) != -1) {
		switch (op) {
		case OPT_HIGH_PPS:
			use_high_pps = 1;
			break;
		case OPT_POST_LIST:
			post_list = atoi(optarg);
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

	ret = run();

	cleanup_ret = ft_free_res();
	return -(ret ? ret : cleanup_ret);
}
