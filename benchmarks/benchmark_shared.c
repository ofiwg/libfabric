/*
 * Copyright (c) 2015 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2013-2015 Intel Corporation.  All rights reserved.
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
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AWV
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>

#include <rdma/fabric.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>

#include "shared.h"
#include "benchmark_shared.h"

extern struct fi_context *ctx_arr;

void ft_parse_benchmark_opts(int op, char *optarg)
{
	switch (op) {
	case 'v':
		opts.options |= FT_OPT_VERIFY_DATA;
		break;
	case 'P':
		hints->mode |= FI_MSG_PREFIX;
		break;
	case 'j':
		hints->tx_attr->inject_size = atoi(optarg);
		break;
	case 'W':
		opts.window_size = atoi(optarg);
		break;
	default:
		break;
	}
}

void ft_benchmark_usage(void)
{
	FT_PRINT_OPTS_USAGE("-v", "enables data_integrity checks");
	FT_PRINT_OPTS_USAGE("-P", "enable prefix mode");
	FT_PRINT_OPTS_USAGE("-j", "maximum inject message size");
	FT_PRINT_OPTS_USAGE("-W", "window size (for bandwidth tests)");
}

int pingpong(void)
{
	int ret, i;

	ret = ft_sync();
	if (ret)
		return ret;

	if (opts.dst_addr) {
		for (i = 0; i < opts.iterations + opts.warmup_iterations; i++) {
			if (i == opts.warmup_iterations)
				ft_start();

			if (opts.transfer_size < fi->tx_attr->inject_size)
				ret = ft_inject(ep, opts.transfer_size);
			else
				ret = ft_tx(ep, opts.transfer_size);
			if (ret)
				return ret;

			ret = ft_rx(ep, opts.transfer_size);
			if (ret)
				return ret;
		}
	} else {
		for (i = 0; i < opts.iterations + opts.warmup_iterations; i++) {
			if (i == opts.warmup_iterations)
				ft_start();

			ret = ft_rx(ep, opts.transfer_size);
			if (ret)
				return ret;

			if (opts.transfer_size < fi->tx_attr->inject_size)
				ret = ft_inject(ep, opts.transfer_size);
			else
				ret = ft_tx(ep, opts.transfer_size);
			if (ret)
				return ret;
		}
	}
	ft_stop();

	if (opts.machr)
		show_perf_mr(opts.transfer_size, opts.iterations, &start, &end, 2, opts.argc, opts.argv);
	else
		show_perf(NULL, opts.transfer_size, opts.iterations, &start, &end, 2);

	return 0;
}

int bandwidth(void)
{
	int ret, i, j;

	ret = ft_sync();
	if (ret)
		return ret;

	/* The loop structured allows for the possibility that the sender
	 * immediately overruns the receiving side on the first transfer (or
	 * the entire window). This could result in exercising parts of the
	 * provider's implementation of FI_RM_ENABLED. For better or worse,
	 * some MPI-level benchmarks tend to use this type of loop for measuring
	 * bandwidth.  */

	if (opts.dst_addr) {
		for (i = 0; i < opts.iterations + opts.warmup_iterations; i++) {
			if (i == opts.warmup_iterations)
				ft_start();

			for(j = 0; j < opts.window_size; j++) {
				if (opts.transfer_size < fi->tx_attr->inject_size)
					ret = ft_inject(ep, opts.transfer_size);
				else
					ret = ft_post_tx(ep, opts.transfer_size,
							 &ctx_arr[j]);
				if (ret)
					return ret;
			}
			ret = ft_get_tx_comp(tx_seq);
			if (ret)
				return ret;
			ret = ft_rx(ep, 4);
			if (ret)
				return ret;
		}
	} else {
		for (i = 0; i < opts.iterations + opts.warmup_iterations; i++) {
			if (i == opts.warmup_iterations)
				ft_start();

			for(j = 0; j < opts.window_size; j++) {
				ret = ft_post_rx(ep, opts.transfer_size, &ctx_arr[j]);
				if (ret)
					return ret;
			}
			ret = ft_get_rx_comp(rx_seq-1); /* rx_seq is always one ahead */
			if (ret)
				return ret;
			ret = ft_tx(ep, 4);
			if (ret)
				return ret;
		}
	}
	ft_stop();

	if (opts.machr)
		show_perf_mr(opts.transfer_size, opts.iterations, &start, &end,
				opts.window_size, opts.argc, opts.argv);
	else
		show_perf(NULL, opts.transfer_size, opts.iterations, &start, &end,
				opts.window_size);

	return 0;
}

int bandwidth_rma(enum ft_rma_opcodes rma_op, struct fi_rma_iov *remote)
{
	int ret, i, j;

	ret = ft_sync();
	if (ret)
		return ret;

	for (i = 0; i < opts.iterations + opts.warmup_iterations; i++) {
		if (i == opts.warmup_iterations)
			ft_start();

		for(j = 0; j < opts.window_size; j++) {
			switch (rma_op) {
			case FT_RMA_WRITE:
				if (opts.transfer_size < fi->tx_attr->inject_size) {
					ret = ft_post_rma_inject(FT_RMA_WRITE, ep,
							opts.transfer_size, remote);
				} else {
					ret = ft_post_rma(rma_op, ep, opts.transfer_size,
							remote,	&ctx_arr[j]);
				}
				break;
			case FT_RMA_WRITEDATA:
				if (!opts.dst_addr) {
					ret = ft_post_rx(ep, 0, &ctx_arr[j]);
				} else {
					if (opts.transfer_size < fi->tx_attr->inject_size) {
						ret = ft_post_rma_inject(FT_RMA_WRITEDATA,
								ep,
								opts.transfer_size,
								remote);
					} else {
						ret = ft_post_rma(FT_RMA_WRITEDATA,
								ep,
								opts.transfer_size,
								remote,	&ctx_arr[j]);
					}
				}
				break;
			case FT_RMA_READ:
				ret = ft_post_rma(FT_RMA_READ, ep, opts.transfer_size,
						remote,	&ctx_arr[j]);
				break;
			default:
				FT_ERR("Unknown RMA op type\n");
				return EXIT_FAILURE;
			}
			if (ret)
				return ret;
		}

		if (rma_op == FT_RMA_WRITEDATA) {
			if (!opts.dst_addr) {
				/* rx_seq is always one ahead */
				ret = ft_get_rx_comp(rx_seq - 1);
				if (ret)
					return ret;
				ret = ft_tx(ep, 4);
			} else {
				ret = ft_rx(ep, 4);
			}
		} else {
			ret = ft_get_tx_comp(tx_seq);
		}
		if (ret)
			return ret;
	}
	ft_stop();

	if (opts.machr)
		show_perf_mr(opts.transfer_size, opts.iterations, &start, &end,
				opts.window_size, opts.argc, opts.argv);
	else
		show_perf(NULL, opts.transfer_size, opts.iterations, &start, &end,
				opts.window_size);
	return 0;
}
