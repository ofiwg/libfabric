/*
 * (C) Copyright 2022 Hewlett Packard Enterprise Development LP
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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <arpa/inet.h>
#include <rdma/fi_tagged.h>
#include <assert.h>

#include "shared.h"
#include "benchmark_shared.h"
#include "hmem.h"
#include "rpc.h"

/* TODO
 * - Client should prepost messaging/tagged response buffers to avoid unexpected
 *   messages from the server
 * - Client and server should allocate buffers from a pre-registered poll of
 *   buffers (i.e. avoid freeing data buffers)
 */

static void run_client_benchmark(size_t rpc_size, size_t rpc_limit,
				 size_t rpc_inflight_limit, enum rpc_op req_op,
				 enum rpc_op resp_op)
{
	size_t rpc_inflight;
	size_t rpc_count = 0;
	struct rpc_ctrl ctrl = {
		.op = req_op,
		.size = rpc_size,
	};
	int ret;

	for (rpc_inflight = 0;
	     rpc_inflight < MIN(rpc_inflight_limit, rpc_limit);
	     rpc_inflight++) {
		ret = rpc_op_exec(req_op, &ctrl);
		ft_assert(ret == 0);
	}

	do {
		ret = rpc_op_exec(resp_op, &ctrl);
		ft_assert(ret == 0);

		rpc_count++;
		rpc_inflight--;

		if (rpc_inflight <
		    MIN(rpc_inflight_limit, rpc_limit - rpc_count)) {
			ret = rpc_op_exec(req_op, &ctrl);
			ft_assert(ret == 0);
			rpc_inflight++;
		}
	} while (rpc_count < rpc_limit);
}

static void rpc_run_benchmarks(size_t rpc_size, size_t warmup_rpc_count,
			       size_t benchmark_rpc_count,
			       size_t rpc_inflight_count, enum rpc_op req_op,
			       enum rpc_op resp_op)
{
	struct timespec start;
	struct timespec end;

	if (warmup_rpc_count)
		run_client_benchmark(rpc_size, warmup_rpc_count,
				     rpc_inflight_count, req_op, resp_op);

	if (benchmark_rpc_count) {
		clock_gettime(CLOCK_MONOTONIC, &start);

		run_client_benchmark(rpc_size, benchmark_rpc_count,
				     rpc_inflight_count, req_op, resp_op);

		clock_gettime(CLOCK_MONOTONIC, &end);

		show_perf(rpc_op_str(req_op), rpc_size, benchmark_rpc_count,
			  &start, &end, 1);
	}
}

struct rpc_req_resp_map {
	enum rpc_op req;
	enum rpc_op resp;
};

#define RPC_BENCHMARKS 4U

static const struct rpc_req_resp_map benchmarks[RPC_BENCHMARKS] = {
	{
		.req = op_msg_req,
		.resp = op_msg_resp,
	},
	{
		.req = op_tag_req,
		.resp = op_tag_resp,
	},
	{
		.req = op_read_req,
		.resp = op_read_resp,
	},
	{
		.req = op_write_req,
		.resp = op_write_resp,
	},
};

static int run_client(void)
{
	int ret;
	int i;
	int op;

	ret = ft_init_fabric();
	if (ret) {
		FT_PRINTERR("ft_init_fabric", ret);
		return ret;
	}

	ret = fi_av_insert(av, fi->dest_addr, 1, &server_addr, 0, NULL);
	if (ret != 1) {
		ret = -FI_EINTR;
		FT_PRINTERR("fi_av_insert", ret);
		goto free;
	}

	ret = rpc_op_exec(op_hello, NULL);
	if (ret) {
		FT_PRINTERR("rpc_op_exec", ret);
		goto free;
	}

	printf("RPCs Inflight: %d\n", opts.window_size);
	printf("RPC Warmup Iterations: %d\n", opts.warmup_iterations);
	printf("HMEM Type: %s\n", fi_tostr(&opts.iface, FI_TYPE_HMEM_IFACE));
	printf("MR mode: %s\n", fi_tostr(&fi->domain_attr->mr_mode,
					 FI_TYPE_MR_MODE));
	printf("Provider: %s\n", fi->fabric_attr->prov_name);

	for (op = 0; op < RPC_BENCHMARKS; op++) {
		if (!(opts.options & FT_OPT_SIZE)) {
			for (i = 0; i < TEST_CNT; i++) {
				if (!ft_use_size(i, opts.sizes_enabled))
					continue;
				opts.transfer_size = test_size[i].size;
				rpc_run_benchmarks(opts.transfer_size,
						   opts.warmup_iterations,
						   opts.iterations,
						   opts.window_size,
						   benchmarks[op].req,
						   benchmarks[op].resp);
			}
		} else {
			rpc_run_benchmarks(opts.transfer_size,
					   opts.warmup_iterations,
					   opts.iterations, opts.window_size,
					   benchmarks[op].req,
					   benchmarks[op].resp);
		}
	}

	ret = rpc_op_exec(op_goodbye, NULL);
	if (ret) {
		FT_PRINTERR("rpc_op_exec", ret);
		goto free;
	}

free:
	ft_free_res();

	return ret;
}

/* Only a subset of the fabtest benchmark and CS options are supported. */
#define RDM_RPC_BW_BENCHMARK_OPTS "vW:"
#define RDM_RPC_BW_CS_OPTS "I:S:w:"
#define RDM_RPC_BW_OPTS ADDR_OPTS FAB_OPTS RDM_RPC_BW_BENCHMARK_OPTS \
	RDM_RPC_BW_CS_OPTS "t:"

static void rdm_rpc_bw_usage(char *name, char *desc)
{
	fprintf(stderr, "Usage:\n");
	fprintf(stderr, "  %s [OPTIONS]\t\tstart server\n", name);
	fprintf(stderr, "  %s [OPTIONS] <host>\tconnect to server\n", name);

	fprintf(stderr, "\n%s\n", desc);
	fprintf(stderr, "\nOptions:\n");

	ft_addr_usage();
	FT_PRINT_OPTS_USAGE("-f <fabric>", "fabric name");
	FT_PRINT_OPTS_USAGE("-d <domain>", "domain name");
	FT_PRINT_OPTS_USAGE("-p <provider>", "specific provider name eg sockets, verbs");
	FT_PRINT_OPTS_USAGE("-K", "fork a child process after initializing endpoint");
	FT_PRINT_OPTS_USAGE("-I <number>", "number of iterations");
	FT_PRINT_OPTS_USAGE("-w <number>", "number of warmup iterations");
	FT_PRINT_OPTS_USAGE("-S <size>", "specific transfer size or "
			    " a range of sizes (syntax r:start,inc,end) or 'all'");
	FT_PRINT_OPTS_USAGE("-v", "enables data_integrity checks");
	FT_PRINT_OPTS_USAGE("-t", "Number of threads processing requests (server only)");
	FT_PRINT_OPTS_USAGE("-W", "window size* (for bandwidth tests)\n\n"
			"* The following condition is required to have at least "
			"one window\nsize # of messsages to be sent: "
			"# of iterations > window size");
}

int main(int argc, char **argv)
{
	int op;
	bool client;
	int ret;
	int thread_count = 1;

	opts = INIT_OPTS;
	opts.options |= FT_OPT_BW | FT_OPT_SKIP_MSG_ALLOC |
		FT_OPT_SKIP_ADDR_EXCH;
	opts.mr_mode = FI_MR_ENDPOINT | FI_MR_ALLOCATED | FI_MR_VIRT_ADDR |
		FI_MR_PROV_KEY | FI_MR_LOCAL;
	opts.comp_method = FT_COMP_WAIT_FD;

	enable_rpc_output = false;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt(argc, argv, "h" RDM_RPC_BW_OPTS)) != -1) {
		switch (op) {
		default:
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			ft_parse_benchmark_opts(op, optarg);
			break;
		case 't':
			thread_count = atoi(optarg);
			break;
		case '?':
		case 'h':
			rdm_rpc_bw_usage(argv[0],
					 "RPC communication style benchmark");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc) {
		opts.dst_addr = argv[optind];
		client = true;
	} else {
		client = false;
	}

	data_verification = !!(opts.options & FT_OPT_VERIFY_DATA);

	hints->caps = FI_RMA | FI_MSG | FI_TAGGED;
	hints->ep_attr->type = FI_EP_RDM;
	hints->domain_attr->resource_mgmt = FI_RM_ENABLED;
	hints->domain_attr->mr_mode = opts.mr_mode;
	hints->addr_format = opts.address_format;
	hints->tx_attr->inject_size = sizeof(struct rpc_hello_msg);

	if (client)
		ret = run_client();
	else
		ret = rpc_run_server(thread_count);

	if (ret)
		return EXIT_FAILURE;

	return EXIT_SUCCESS;
}

