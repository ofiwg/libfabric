/*
 * Copyright (c), Amazon.com, Inc.  All rights reserved.
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

/* This test runs the following workflow
 Client:
- send message to server x2

Server:
- post with rx any source (during ft_enable_ep_recv)
- remove the client from av to make it unknown
- get the rx comp for the first rx, which should
  succeed anyway because it was posted for any source
- insert client back into AV
- post rx with fi addr 0 (client fiaddr)
- get the rx comp for the second rx.
 */
static int run(void)
{
	int ret;
	int i;

	ret = ft_init_fabric();
	if (ret)
		return ret;

	if (opts.dst_addr) { /* client */
		ret = ft_sync();
		if (ret) {
			FT_PRINTERR("ft_sync", ret);
			return ret;
		}

		for (i = 0; i < 2; i++) {
			ret = ft_tx(ep, remote_fi_addr,
				    opts.transfer_size, &tx_ctx);
			if (ret) {
				FT_PRINTERR("ft_tx", ret);
				return ret;
			}
		}
		printf("Client: send completes\n");

		ret = ft_sync();
		if (ret) {
			FT_PRINTERR("ft_sync", ret);
			return ret;
		}

		ret = ft_init_av();
		if (ret) {
			FT_PRINTERR("ft_init_av", ret);
			return ret;
		}
	} else { /* server */
		/* First remove the peer to make it unknown */
		ret = fi_av_remove(av, &remote_fi_addr, 1, 0);
		if (ret) {
			FT_PRINTERR("fi_av_remove", ret);
			return ret;
		}

		ret = ft_sync();
		if (ret) {
			FT_PRINTERR("ft_sync", ret);
			return ret;
		}

		/*
		 * The first recv should be matched anyway
		 * because it was posted with FI_ADDR_UNSPEC
		 */
		ret = ft_get_rx_comp(rx_seq);
		if (ret)
			return ret;

		printf("Server: received the first message\n");

		ret = ft_sync();
		if (ret) {
			FT_PRINTERR("ft_sync", ret);
			return ret;
		}

		/* reinsert the peer to the av */
		ret = ft_init_av();
		if (ret) {
			FT_PRINTERR("ft_init_av", ret);
			return ret;
		}

		/* Post a directed recv*/
		ret = ft_post_rx(ep, opts.transfer_size, &rx_ctx);
		if (ret) {
			FT_PRINTERR("ft_post_rx", ret);
			return ret;
		}

		ret = ft_get_rx_comp(rx_seq);
		if (ret) {
			FT_PRINTERR("ft_get_rx_comp", ret);
			return ret;
		}

		printf("Server: received the second message\n");
	}
	return ret;
}

int main(int argc, char **argv)
{
	int op, ret;

	opts = INIT_OPTS;
	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt(argc, argv, "Uh" ADDR_OPTS INFO_OPTS)) != -1) {
		switch (op) {
		default:
			ft_parse_addr_opts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints, &opts);
			break;
		case 'U':
			hints->tx_attr->op_flags |= FI_DELIVERY_COMPLETE;
			break;
		case '?':
		case 'h':
			ft_usage(argv[0], "Test RDM endpoint with unknown peers.");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	opts.options |= FT_OPT_SIZE | FT_OPT_OOB_SYNC;
	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_MSG | FI_DIRECTED_RECV;
	hints->mode = FI_CONTEXT;
	hints->domain_attr->mr_mode = opts.mr_mode;

	ret = run();

	ft_free_res();
	return ft_exit_code(ret);
}
