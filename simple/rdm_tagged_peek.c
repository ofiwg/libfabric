/*
 * Copyright (c) 2013-2015 Intel Corporation.  All rights reserved.
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
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AWV
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <time.h>
#include <netdb.h>
#include <unistd.h>
#include <inttypes.h>

#include <rdma/fabric.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_tagged.h>
#include <shared.h>


static int init_fabric(void)
{
	uint64_t flags = 0;
	char *node, *service;
	int ret;

	ret = ft_read_addr_opts(&node, &service, hints, &flags, &opts);
	if (ret)
		return ret;

	ret = fi_getinfo(FT_FIVERSION, node, service, flags, hints, &fi);
	if (ret) {
		FT_PRINTERR("fi_getinfo", ret);
		return ret;
	}

	ret = ft_open_fabric_res();
	if (ret)
		return ret;

	ret = ft_alloc_active_res(fi);
	if (ret)
		return ret;

	ret = ft_init_ep();
	if (ret)
		return ret;

	return 0;
}

static int tagged_peek(uint64_t tag)
{
	struct fi_cq_tagged_entry comp;
	struct fi_msg_tagged msg;
	int ret;

	memset(&msg, 0, sizeof msg);
	msg.tag = tag;
	msg.context = &rx_ctx;

	ret = fi_trecvmsg(ep, &msg, FI_PEEK);
	if (ret) {
		FT_PRINTERR("FI_PEEK", ret);
		return ret;
	}

	ret = fi_cq_sread(rxcq, &comp, 1, NULL, -1);
	if (ret != 1) {
		if (ret == -FI_EAVAIL)
			ret = ft_cq_readerr(rxcq);
		else
			FT_PRINTERR("fi_cq_read", ret);
	}
	return ret;
}

static int run(void)
{
	int ret;

	ret = init_fabric();
	if (ret)
		return ret;

	ret = ft_init_av();
	if (ret)
		return ret;

	if (opts.dst_addr) {
		printf("Searching for a bad msg\n");
		ret = tagged_peek(0xbad);
		if (ret != -FI_ENOMSG) {
			FT_PRINTERR("FI_PEEK", ret);
			return ret;
		}

		printf("Synchronizing with sender..\n");
		ret = ft_sync();
		if (ret)
			return ret;

		printf("Searching for a good msg\n");
		ret = tagged_peek(0x900d);
		if (ret != 1) {
			FT_PRINTERR("FI_PEEK", ret);
			return ret;
		}

		printf("Receiving msg\n");
		ret = fi_trecv(ep, buf, rx_size, fi_mr_desc(mr), remote_fi_addr,
				0x900d, 0, &rx_ctx);
		if (ret) {
			FT_PRINTERR("fi_trecv", ret);
			return ret;
		}

		printf("Completing recv\n");
		ret = ft_get_rx_comp(++rx_seq);
		if (ret)
			return ret;

	} else {
		printf("Sending tagged message\n");
		ret = fi_tsend(ep, tx_buf, tx_size, fi_mr_desc(mr),
				remote_fi_addr, 0x900d, &tx_ctx);
		if (ret)
			return ret;

		printf("Synchronizing with receiver..\n");
		ret = ft_sync();
		if (ret)
			return ret;

		printf("Getting send completion\n");
		ret = ft_get_tx_comp(tx_seq + 1);
		if (ret)
			return ret;
	}

	ft_finalize();
	return 0;
}

int main(int argc, char **argv)
{
	int ret, op;

	opts = INIT_OPTS;
	opts.options |= FT_OPT_SIZE;

	hints = fi_allocinfo();
	if (!hints) {
		FT_PRINTERR("fi_allocinfo", -FI_ENOMEM);
		return EXIT_FAILURE;
	}

	while ((op = getopt(argc, argv, "h" ADDR_OPTS INFO_OPTS)) != -1) {
		switch (op) {
		default:
			ft_parse_addr_opts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints);
			break;
		case '?':
		case 'h':
			ft_usage(argv[0], "An RDM client-server example that uses tagged search.\n");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->rx_attr->total_buffered_recv = 1024;
	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_MSG | FI_TAGGED;
	hints->mode = FI_CONTEXT | FI_LOCAL_MR;

	ret = run();

	ft_free_res();
	return -ret;
}
