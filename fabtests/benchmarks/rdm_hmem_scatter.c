/*
 * Copyright (c) Intel Corporation.  All rights reserved.
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
 */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include <rdma/fi_errno.h>

#include <shared.h>
#include "benchmark_shared.h"

static int run(void)
{
	char test_name[50] = "custom";
	bool ran = false;
	int i, ret;

	ret = ft_check_hmem_support();
	if (ret)
		return ret;

	ret = ft_init_fabric();
	if (ret)
		return ret;

	ret = ft_scatter_exchange();
	if (ret)
		return ret;

	if (opts.options & FT_OPT_SIZE) {
		if (!ft_scatter_fits(opts.transfer_size)) {
			FT_ERR("%zu buffers of %zu bytes cannot hold a %zu "
			       "byte message", ft_scatter.nsegs,
			       ft_scatter_posted_len(opts.transfer_size),
			       opts.transfer_size);
			return -FI_EINVAL;
		}

		init_test(&opts, test_name, sizeof(test_name));
		ret = bandwidth_scatter();
		if (ret)
			return ret;
	} else {
		for (i = 0; i < TEST_CNT; i++) {
			if (!ft_use_size(i, opts.sizes_enabled))
				continue;

			opts.transfer_size = test_size[i].size;
			if (!ft_scatter_fits(opts.transfer_size)) {
				FT_DEBUG("skipping %zu bytes, %zu buffers of "
					 "%zu bytes cannot hold it",
					 opts.transfer_size,
					 ft_scatter.nsegs, ft_scatter.seg_len);
				continue;
			}

			init_test(&opts, test_name, sizeof(test_name));
			ret = bandwidth_scatter();
			if (ret)
				return ret;
			ran = true;
		}

		if (!ran) {
			FT_ERR("no message size fits in %zu buffers of %zu "
			       "bytes", ft_scatter.nsegs, ft_scatter.seg_len);
			return -FI_EINVAL;
		}
	}

	return ft_finalize();
}

int main(int argc, char **argv)
{
	int op, ret, cleanup_ret;

	opts = INIT_OPTS;
	opts.options |= FT_OPT_BW | FT_OPT_ENABLE_HMEM | FT_OPT_OOB_SYNC |
			FT_OPT_OOB_ADDR_EXCH;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt_long(argc, argv,
				 "Uh" CS_OPTS INFO_OPTS BENCHMARK_OPTS
				 HMEM_SCATTER_OPTS, long_opts,
				 &lopt_idx)) != -1) {
		switch (op) {
		case 'U':
			hints->tx_attr->op_flags |= FI_DELIVERY_COMPLETE;
			break;
		case '?':
		case 'h':
			ft_csusage(argv[0], "Bandwidth test for RDM endpoints "
				   "receiving with fi_recvv into a list of "
				   "host and device buffers.\nThe client "
				   "sends from device memory if it is given "
				   "-D, on device -i (default 0),\notherwise "
				   "from host memory. The server needs -D "
				   "only if the layout has a device\nentry, "
				   "and ignores -i.");
			ft_benchmark_usage();
			ft_hmem_scatter_usage();
			ft_longopts_usage();
			ret = EXIT_FAILURE;
			goto out;
		default:
			if (!ft_parse_long_opts(op, optarg))
				continue;
			ft_parse_hmem_scatter_opts(op, optarg);
			ft_parse_benchmark_opts(op, optarg);
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			break;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	if (ft_check_hmem_scatter_opts()) {
		ret = EXIT_FAILURE;
		goto out;
	}

	if (hints->ep_attr->type != FI_EP_UNSPEC) {
		FT_DEBUG("ignoring -e, this test only runs on RDM endpoints");
	}

	if (opts.options & FT_OPT_NO_PRE_POSTED_RX) {
		FT_DEBUG("ignoring -r, this test syncs out of band");
		opts.options &= ~(FT_OPT_NO_PRE_POSTED_RX);
	}

	hints->ep_attr->type = FI_EP_RDM;
	hints->domain_attr->resource_mgmt = FI_RM_ENABLED;
	hints->caps = FI_MSG;
	hints->mode |= FI_CONTEXT | FI_CONTEXT2;
	hints->domain_attr->mr_mode = opts.mr_mode;
	hints->tx_attr->tclass = FI_TC_BULK_DATA;
	hints->addr_format = opts.address_format;

	ret = run();

out:
	cleanup_ret = ft_free_res();
	return ft_exit_code(ret ? ret : cleanup_ret);
}
