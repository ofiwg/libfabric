/*
 * Copyright (c) 2020 Intel Corporation.  All rights reserved.
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

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include <shared.h>


int init_fabric(void)
{
	int ret;

	ret = ft_init_oob();
	if (ret)
		return ret;

	ret = ft_getinfo(hints, &fi);
	if (ret)
		return ret;

	ret = ft_open_fabric_res();
	if (ret)
		return ret;

	ret = ft_alloc_active_res(fi);
	if (ret)
		return ret;

	ret = ft_enable_ep_recv();
	if (ret)
		return ret;

	return 0;
}

static int run_server(void)
{
	fi_addr_t *fi_addr;
	int i, c, ret;

	fi_addr = calloc(opts.num_connections, sizeof(*fi_addr));
	if (!fi_addr)
		return -FI_ENOMEM;

	ret = init_fabric();
	if (ret)
		return ret;

	for (i = 0; i < opts.iterations; i++) {
		for (c = 0; c < opts.num_connections; c++) {
			ret = ft_init_av_dst_addr(av, ep, &fi_addr[c]);
			if (ret)
				goto out;

			/* ft_init_av_dst_addr() assumes there is only 1 entry
			 * in the AV, and sets the return fi_addr to 0.
			 */
			if (fi->domain_attr->av_type == FI_AV_TABLE)
				fi_addr[c] = c;
		}

		fi_av_remove(av, fi_addr, opts.num_connections, 0);
	}

out:
	ft_free_res();
	free(fi_addr);
	return ret;
}

static int run_client(void)
{
	struct fi_info *saved_hints;
	int i, c, ret;

	saved_hints = hints;
	for (i = 0; i < opts.iterations; i++) {
		hints = saved_hints;
		for (c = 0; c < opts.num_connections; c++) {
			ret = ft_init_fabric();
			if (ret)
				break;
		}

		hints = NULL;
		while (c--)
			ft_free_res();
	}

	return ret;
}

int main(int argc, char **argv)
{
	int op, ret;

	opts = INIT_OPTS;
	opts.options |= FT_OPT_SIZE;
	opts.num_connections = 16;
	opts.iterations = 4;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_MSG;
	hints->mode = FI_CONTEXT;
	hints->domain_attr->mr_mode = opts.mr_mode;

	while ((op = getopt(argc, argv, "h" ADDR_OPTS INFO_OPTS)) != -1) {
		switch (op) {
		default:
			ft_parse_addr_opts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints, &opts);
			break;
		case '?':
		case 'h':
			ft_usage(argv[0], "A simple RDM client-sever example.");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	opts.av_size = opts.num_connections;

	ret = opts.dst_addr ? run_client() : run_server();

	return ft_exit_code(ret);
}
