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
 * Runs MSG pingpong.
 */

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include <rdma/fi_errno.h>
#include <rdma/fi_ext_efa.h>
#include <shared.h>
#include "benchmarks/benchmark_shared.h"

static int open_cntr(struct fid_cntr **cntr)
{
	struct fi_efa_ops_gda *gda_ops;
	struct fi_cntr_attr attr = {0};
	struct fi_efa_comp_cntr_init_attr efa_attr = {0};
	int ret;

	ret = fi_open_ops(&domain->fid, FI_EFA_GDA_OPS, 0,
			  (void **)&gda_ops, NULL);
	if (!ret) {
		attr.events = FI_CNTR_EVENTS_COMP;
		attr.wait_obj = FI_WAIT_UNSPEC;

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

	ret = open_cntr(&txcntr);
	if (ret) {
		FT_PRINTERR("open_cntr(tx)", ret);
		return ret;
	}

	ret = open_cntr(&rxcntr);
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
			ret = pingpong();
			if (ret)
				goto out;
		}
	} else {
		init_test(&opts, test_name, sizeof(test_name));
		ret = pingpong();
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
	opts.comp_method = FT_COMP_SPIN;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt(argc, argv, "h" CS_OPTS INFO_OPTS BENCHMARK_OPTS)) != -1) {
		switch (op) {
		default:
			if (!ft_parse_long_opts(op, optarg))
				continue;
			ft_parse_benchmark_opts(op, optarg);
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			break;
		case '?':
		case 'h':
			ft_csusage(argv[0],
				   "MSG pingpong using EFA hardware counters.");
			ft_benchmark_usage();
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

	ret = run_msg();

	ft_free_res();
	return ft_exit_code(ret);
}
