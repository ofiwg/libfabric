/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_ext.h>
#include <shared.h>
#include "benchmarks/benchmark_shared.h"

/*
 * Mirrors ft_init_fabric() but inserts fi_setopt(HOMOGENEOUS_PEERS)
 * before fi_enable, since the option must be set before enabling the EP
 * to avoid the requirement for a handshake before using a read based
 * protocol.
 */
static int init_fabric_with_homogeneous_peers(void)
{
	int ret;
	bool homogeneous = true;

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

	ret = ft_getinfo(hints, &fi);
	if (ret)
		return ret;

	ret = ft_open_fabric_res();
	if (ret)
		return ret;

	ret = ft_alloc_active_res(fi);
	if (ret)
		return ret;

	/* Must be set before fi_enable to skip handshake requirement before
	 * using read based protocol */
	ret = fi_setopt(&ep->fid, FI_OPT_ENDPOINT,
			FI_OPT_EFA_HOMOGENEOUS_PEERS,
			&homogeneous, sizeof(homogeneous));
	if (ret) {
		FT_PRINTERR("fi_setopt(HOMOGENEOUS_PEERS)", ret);
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

static int run(void)
{
	int ret;
	char test_name[64] = "";

	ret = init_fabric_with_homogeneous_peers();
	if (ret)
		return ret;

	init_test(&opts, test_name, sizeof(test_name));
	ret = bandwidth();
	if (ret)
		return ret;

	return ft_finalize();
}

int main(int argc, char **argv)
{
	int op, ret, cleanup_ret;

	opts = INIT_OPTS;
	opts.options |= FT_OPT_BW;
	opts.iterations = 1;
	opts.warmup_iterations = 0;
	opts.transfer_size = 262144; /* 256 KB */
	opts.window_size = 1;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt_long(argc, argv, "Uh" CS_OPTS INFO_OPTS BENCHMARK_OPTS,
				 long_opts, &lopt_idx)) != -1) {
		switch (op) {
		default:
			if (!ft_parse_long_opts(op, optarg))
				continue;
			ft_parse_benchmark_opts(op, optarg);
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			break;
		case 'U':
			hints->tx_attr->op_flags |= FI_DELIVERY_COMPLETE;
			break;
		case '?':
		case 'h':
			ft_csusage(argv[0],
				   "Verify runt read protocol transfers "
				   "data with both RTM packets and RDMA read");
			ft_benchmark_usage();
			ft_longopts_usage();
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_MSG;
	hints->mode |= FI_CONTEXT | FI_CONTEXT2;
	hints->domain_attr->mr_mode = opts.mr_mode;
	hints->addr_format = opts.address_format;

	ret = run();

	cleanup_ret = ft_free_res();
	return ft_exit_code(ret ? ret : cleanup_ret);
}
