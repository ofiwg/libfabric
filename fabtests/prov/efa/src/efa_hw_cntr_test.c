/*
 * Copyright (c) 2026, Amazon.com, Inc.  All rights reserved.
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
 * EFA hardware counter test.
 *
 * Runs MSG pingpong or RMA write.
 * Use --external-mem to pass user-allocated memory for hw counters
 * in cntr_open_ext.
 */

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include <rdma/fi_errno.h>
#include <rdma/fi_ext_efa.h>
#include <shared.h>
#include "benchmarks/benchmark_shared.h"

static bool use_ext_mem;
static volatile uint64_t *tx_cntr_ptr;
static volatile uint64_t *rx_cntr_ptr;

enum {
	LONG_OPT_EXTERNAL_MEM,
};

static int open_cntr(struct fid_cntr **cntr, volatile uint64_t **cntr_ptr)
{
	struct fi_efa_ops_gda *gda_ops;
	struct fi_cntr_attr attr = {0};
	struct fi_efa_comp_cntr_init_attr efa_attr = {0};
	int ret;

	*cntr_ptr = NULL;

	ret = fi_open_ops(&domain->fid, FI_EFA_GDA_OPS, 0,
			  (void **)&gda_ops, NULL);
	if (!ret) {
		attr.events = FI_CNTR_EVENTS_COMP;
		attr.wait_obj = FI_WAIT_UNSPEC;

		if (use_ext_mem) {
			efa_attr.comp_cntr_ext_mem.type = FI_EFA_MEMORY_LOCATION_VA;
			efa_attr.comp_cntr_ext_mem.ptr = calloc(1, sizeof(uint64_t));
			if (!efa_attr.comp_cntr_ext_mem.ptr)
				return -FI_ENOMEM;

			efa_attr.err_cntr_ext_mem.type = FI_EFA_MEMORY_LOCATION_VA;
			efa_attr.err_cntr_ext_mem.ptr = calloc(1, sizeof(uint64_t));
			if (!efa_attr.err_cntr_ext_mem.ptr) {
				free(efa_attr.comp_cntr_ext_mem.ptr);
				return -FI_ENOMEM;
			}

			efa_attr.flags = FI_EFA_COMP_CNTR_INIT_WITH_COMP_EXTERNAL_MEM |
					 FI_EFA_COMP_CNTR_INIT_WITH_ERR_EXTERNAL_MEM;

			*cntr_ptr = (volatile uint64_t *)efa_attr.comp_cntr_ext_mem.ptr;
		}

		ret = gda_ops->cntr_open_ext(domain, &attr, cntr, NULL,
					     &efa_attr);
	}

	if (ret) {
		FT_WARN("hw cntr open failed (%s)\n", fi_strerror(-ret));
		free(efa_attr.comp_cntr_ext_mem.ptr);
		free(efa_attr.err_cntr_ext_mem.ptr);
		return ret;
	}

	return FI_SUCCESS;
}

/*
 * Custom init that mirrors ft_init_fabric() but opens hw counters
 * and assigns them to the global txcntr/rxcntr before fi_enable.
 * ft_get_tx_comp/ft_get_rx_comp will then wait on hw counters
 * via ft_get_cntr_comp.
 */
static int init_fabric_with_hw_cntr(void)
{
	char *node, *service;
	uint64_t flags = 0;
	int ret;

	ret = ft_init();
	if (ret)
		return ret;

	ret = ft_init_oob();
	if (ret)
		return ret;

	if (oob_sock >= 0 && opts.dst_addr) {
		ret = ft_sock_sync(oob_sock, 0);
		if (ret)
			return ret;
	}

	ret = ft_read_addr_opts(&node, &service, hints, &flags, &opts);
	if (ret)
		return ret;

	/* hw cntr require API version >= 2.5 */
	ret = fi_getinfo(FI_VERSION(2, 5), node, service, flags,
				hints, &fi);
	if (ret) {
		FT_PRINTERR("fi_getinfo", ret);
		return ret;
	}

	if (fi->domain_attr->max_cntr_value == UINT64_MAX) {
		FT_INFO("Device does not support hw counters, skipping test");
		return -FI_ENODATA;
	}

	ret = ft_open_fabric_res();
	if (ret)
		return ret;

	ret = ft_alloc_active_res(fi);
	if (ret)
		return ret;

	/* Open counters and assign to globals
	 * so ft_get_tx/rx_comp routes through ft_get_cntr_comp.
	 * Clear CQ opts so completion path uses counters instead of CQs.
	 */
	opts.options |= FT_OPT_TX_CNTR | FT_OPT_RX_CNTR;
	opts.options &= ~(FT_OPT_TX_CQ | FT_OPT_RX_CQ);

	ret = open_cntr(&txcntr, &tx_cntr_ptr);
	if (ret) {
		FT_PRINTERR("open_cntr(tx)", ret);
		return ret;
	}

	ret = open_cntr(&rxcntr, &rx_cntr_ptr);
	if (ret) {
		FT_PRINTERR("open_cntr(rx)", ret);
		return ret;
	}

	ret = ft_enable_ep_recv();
	if (ret)
		return ret;

	if (oob_sock >= 0 && !opts.dst_addr) {
		ret = ft_sock_sync(oob_sock, 0);
		if (ret)
			return ret;
	}

	ret = ft_init_av();
	if (ret)
		return ret;

	return 0;
}

/*
 * When external memory is passed, read the cntr value directly.
 * Otherwise, fi_cntr_read is called.
 */
static inline int wait_cntr(volatile uint64_t *cntr_ptr,
			    struct fid_cntr *cntr, uint64_t total)
{
	struct timespec a, b;

	if (cntr_ptr) {
		if (timeout >= 0)
			clock_gettime(CLOCK_MONOTONIC, &a);
		while (*cntr_ptr < total) {
			ft_force_progress();
			if (timeout >= 0) {
				clock_gettime(CLOCK_MONOTONIC, &b);
				if ((b.tv_sec - a.tv_sec) > timeout) {
					fprintf(stderr, "%ds timeout expired\n",
						timeout);
					return -FI_ENODATA;
				}
			}
		}
		return 0;
	}
	return ft_get_cntr_comp(cntr, total, timeout);
}

/*
 * Custom pingpong that waits for completions by reading hw counter
 * pointers (tx_cntr_ptr / rx_cntr_ptr) directly, bypassing fi_cntr_read.
 * Falls back to ft_get_cntr_comp when pointers are not available.
 */
static int msg_pingpong(void)
{
	int ret, i;

	ret = ft_sync();
	if (ret)
		return ret;

	ft_start();
	if (opts.dst_addr) {
		for (i = 0; i < opts.iterations + opts.warmup_iterations; i++) {
			if (i == opts.warmup_iterations)
				ft_start();

			ret = ft_post_tx(ep, remote_fi_addr,
					 opts.transfer_size, NO_CQ_DATA,
					 &tx_ctx);
			if (ret)
				return ret;

			ret = wait_cntr(tx_cntr_ptr, txcntr, tx_seq);
			if (ret)
				return ret;

			ret = wait_cntr(rx_cntr_ptr, rxcntr, rx_seq);
			if (ret)
				return ret;

			ret = ft_post_rx(ep, rx_size, &rx_ctx);
			if (ret)
				return ret;
		}
	} else {
		for (i = 0; i < opts.iterations + opts.warmup_iterations; i++) {
			if (i == opts.warmup_iterations)
				ft_start();

			ret = wait_cntr(rx_cntr_ptr, rxcntr, rx_seq);
			if (ret)
				return ret;

			ret = ft_post_rx(ep, rx_size, &rx_ctx);
			if (ret)
				return ret;

			ret = ft_post_tx(ep, remote_fi_addr,
					 opts.transfer_size, NO_CQ_DATA,
					 &tx_ctx);
			if (ret)
				return ret;

			ret = wait_cntr(tx_cntr_ptr, txcntr, tx_seq);
			if (ret)
				return ret;
		}
	}
	ft_stop();

	show_perf(NULL, opts.transfer_size, opts.iterations, &start, &end, 2);
	return 0;
}

static int run_msg(void)
{
	int i, ret = 0;

	ret = init_fabric_with_hw_cntr();
	if (ret)
		return ret;

	if (!(opts.options & FT_OPT_SIZE)) {
		for (i = 0; i < TEST_CNT; i++) {
			if (!ft_use_size(i, opts.sizes_enabled))
				continue;
			opts.transfer_size = test_size[i].size;
			init_test(&opts, test_name, sizeof(test_name));
			ret = msg_pingpong();
			if (ret)
				goto out;
		}
	} else {
		init_test(&opts, test_name, sizeof(test_name));
		ret = msg_pingpong();
		if (ret)
			goto out;
	}

	ft_finalize();
out:
	return ret;
}

static int rma_write(void)
{
	int i, ret;

	ret = ft_sync();
	if (ret)
		return ret;

	ft_start();
	for (i = 0; i < opts.iterations; i++) {
		ret = fi_write(ep, tx_buf, opts.transfer_size, mr_desc,
			       remote_fi_addr, remote.addr, remote.key,
			       &tx_ctx);
		if (ret) {
			FT_PRINTERR("fi_write", ret);
			return ret;
		}
		tx_seq++;

		ret = wait_cntr(tx_cntr_ptr, txcntr, tx_seq);
		if (ret)
			return ret;
	}
	ft_stop();

	show_perf(NULL, opts.transfer_size, opts.iterations, &start, &end, 1);
	return 0;
}

static int run_rma(void)
{
	int i, ret = 0;

	ret = init_fabric_with_hw_cntr();
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
			ret = rma_write();
			if (ret)
				goto out;
		}
	} else {
		init_test(&opts, test_name, sizeof(test_name));
		ret = rma_write();
		if (ret)
			goto out;
	}

	ft_finalize();
out:
	return ret;
}

int main(int argc, char **argv)
{
	int op, ret;

	opts = INIT_OPTS;
	opts.rma_op = 0;
	opts.comp_method = FT_COMP_SPIN;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	int lopt_idx = 0;
	struct option long_opts[] = {
		{"external-mem", no_argument, NULL, LONG_OPT_EXTERNAL_MEM},
		{0, 0, 0, 0}
	};
	while ((op = getopt_long(argc, argv, "h" CS_OPTS INFO_OPTS BENCHMARK_OPTS
				 API_OPTS, long_opts, &lopt_idx)) != -1) {
		switch (op) {
		case LONG_OPT_EXTERNAL_MEM:
			use_ext_mem = true;
			break;
		default:
			if (!ft_parse_long_opts(op, optarg))
				continue;
			ft_parse_benchmark_opts(op, optarg);
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			ret = ft_parse_api_opts(op, optarg, hints, &opts);
			if (ret)
				return ret;
			break;
		case '?':
		case 'h':
			ft_csusage(argv[0],
				   "Pingpong using EFA hardware counters.");
			ft_benchmark_usage();
			FT_PRINT_OPTS_USAGE("-o <op>",
				"op: msg|write (default: msg)");
			FT_PRINT_OPTS_USAGE("--external-mem",
				"use external user memory for hw counters");
			ft_longopts_usage();
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->ep_attr->type = FI_EP_RDM;
	hints->caps |= FI_MSG;
	hints->domain_attr->mr_mode = opts.mr_mode;
	hints->domain_attr->threading = FI_THREAD_DOMAIN;
	hints->tx_attr->tclass = FI_TC_LOW_LATENCY;
	hints->addr_format = opts.address_format;
	hints->mode |= FI_CONTEXT | FI_CONTEXT2;

	if (opts.rma_op) {
		hints->caps |= FI_RMA;
		ret = run_rma();
	} else {
		ret = run_msg();
	}

	ft_free_res();
	return ft_exit_code(ret);
}
