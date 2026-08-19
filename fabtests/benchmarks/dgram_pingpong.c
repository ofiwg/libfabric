/*
 * Copyright (c) 2013-2015 Intel Corporation.  All rights reserved.
 * Copyright (c) 2014-2017 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2020-2021 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>

#include "shared.h"
#include "benchmark_shared.h"

/* Per-size status exchanged with the peer over the reliable OOB socket. */
#define FT_DGRAM_STATUS_OK	0
#define FT_DGRAM_STATUS_TIMEOUT	1

/*
 * After a per-size timeout, realign completion accounting so the next size
 * starts even. CQ mode: bump the software cntr to the posted seq. Counter mode:
 * the provider counter can't be written, so lower seq to what it observed
 * (success + error); otherwise it stays behind seq and every size times out.
 */
static void ft_dgram_reconcile_counters(void)
{
	struct fi_cq_err_entry comp;
	int ret;

	/* Drain late completions so they don't leak into the next size. CQs
	 * exist even in counter mode (bound FI_SELECTIVE_COMPLETION). */
	if (txcq) {
		do {
			ret = fi_cq_read(txcq, &comp, 1);
			if (ret == -FI_EAVAIL)
				(void) fi_cq_readerr(txcq, &comp, 0);
		} while (ret > 0 || ret == -FI_EAVAIL);
	}
	if (rxcq) {
		do {
			ret = fi_cq_read(rxcq, &comp, 1);
			if (ret == -FI_EAVAIL)
				(void) fi_cq_readerr(rxcq, &comp, 0);
		} while (ret > 0 || ret == -FI_EAVAIL);
	}

	if (opts.options & FT_OPT_TX_CQ) {
		tx_cq_cntr = tx_seq;
	} else if (txcntr) {
		tx_seq = fi_cntr_read(txcntr) + fi_cntr_readerr(txcntr);
		tx_cq_cntr = tx_seq;
	}

	if (opts.options & FT_OPT_RX_CQ) {
		rx_cq_cntr = rx_seq;
	} else if (rxcntr) {
		rx_seq = fi_cntr_read(rxcntr) + fi_cntr_readerr(rxcntr);
		rx_cq_cntr = rx_seq;
	}
}

/*
 * OOB barrier between sizes: exchange local status and return TIMEOUT if either
 * peer timed out. Keeps both sides in lock-step so neither sends to a torn-down
 * QP.
 */
static int ft_dgram_size_barrier(int local_status)
{
	int peer_status;

	peer_status = ft_sock_sync(oob_sock, local_status);
	if (peer_status < 0)
		return peer_status;

	return (local_status == FT_DGRAM_STATUS_TIMEOUT ||
		peer_status == FT_DGRAM_STATUS_TIMEOUT) ?
		       FT_DGRAM_STATUS_TIMEOUT : FT_DGRAM_STATUS_OK;
}

/*
 * Run pingpong for one size. A -FI_ENODATA timeout is recoverable: reconcile
 * state, sync with the peer, and continue. *timed_out reflects either peer.
 * Any other error is fatal.
 */
static int run_one_size(bool *timed_out)
{
	int ret, status;

	ret = pingpong();
	if (ret && ret != -FI_ENODATA)
		return ret;

	if (ret == -FI_ENODATA) {
		fprintf(stderr,
			"Receive timed out for message size %zu; "
			"skipping to next size\n", opts.transfer_size);
		ft_dgram_reconcile_counters();
		status = FT_DGRAM_STATUS_TIMEOUT;
	} else {
		status = FT_DGRAM_STATUS_OK;
	}

	status = ft_dgram_size_barrier(status);
	if (status < 0)
		return status;

	*timed_out = (status == FT_DGRAM_STATUS_TIMEOUT);
	return 0;
}

static int run(void)
{
	int i, ret;
	bool timed_out, any_timed_out = false;

	ret = ft_init_fabric();
	if (ret)
		return ret;

	/* Post an extra receive to avoid lacking a posted receive in the
	 * finalize.
	 */
	ret = fi_recv(ep, rx_buf, rx_size + ft_rx_prefix_size(), mr_desc,
			0, &rx_ctx);
	if (ret)
		return ret;

	if (!(opts.options & FT_OPT_SIZE)) {
		for (i = 0; i < TEST_CNT; i++) {
			if (!ft_use_size(i, opts.sizes_enabled))
				continue;
			opts.transfer_size = test_size[i].size;
			init_test(&opts, test_name, sizeof(test_name));
			ret = run_one_size(&timed_out);
			if (ret)
				return ret;
			any_timed_out |= timed_out;
		}
	} else {
		init_test(&opts, test_name, sizeof(test_name));
		ret = run_one_size(&timed_out);
		if (ret)
			return ret;
		any_timed_out |= timed_out;
	}

	ret = ft_finalize();
	if (ret)
		return ret;

	/* Every size was attempted and both peers tore down together. Report a
	 * distinct status if any size timed out so the harness can flag it. */
	return any_timed_out ? -FI_ENODATA : 0;
}

int main(int argc, char **argv)
{
	int ret, op, cleanup_ret;

	opts = INIT_OPTS;

	timeout = 5;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt_long(argc, argv, "hT:" CS_OPTS INFO_OPTS BENCHMARK_OPTS,
				 long_opts, &lopt_idx)) != -1) {
		switch (op) {
		case 'T':
			timeout = atoi(optarg);
			break;
		default:
			if (!ft_parse_long_opts(op, optarg))
				continue;
			ft_parse_benchmark_opts(op, optarg);
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			break;
		case '?':
		case 'h':
			ft_csusage(argv[0], "Ping pong client and server using UD.");
			ft_benchmark_usage();
			FT_PRINT_OPTS_USAGE("-T <timeout>",
					"seconds before timeout on receive");
			ft_longopts_usage();
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	/*
	 * Because dgram endpoint is not reliable, we
	 * must use out-of-band sync
	 */
	opts.options |= FT_OPT_OOB_SYNC;

	hints->ep_attr->type = FI_EP_DGRAM;
	hints->caps = FI_MSG;
	hints->mode |= FI_CONTEXT | FI_CONTEXT2;
	hints->domain_attr->mr_mode = opts.mr_mode;
	hints->tx_attr->tclass = FI_TC_LOW_LATENCY;
	hints->addr_format = opts.address_format;

	ret = run();

	cleanup_ret = ft_free_res();
	return ft_exit_code(ret ? ret : cleanup_ret);
}
