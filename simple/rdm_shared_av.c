/*
 * Copyright (c) 2013-2015 Intel Corporation.  All rights reserved.
 * Copyright (c) 2016, Cisco Systems, Inc. All rights reserved.
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
#include <getopt.h>
#include <unistd.h>
#include <sys/types.h>
#include <string.h>

#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>

#include "shared.h"

static int init_fabric(void)
{
	char *node, *service;
	uint64_t flags = 0;
	int ret;

	ret = ft_read_addr_opts(&node, &service, hints, &flags, &opts);
	if (ret)
		return ret;

	/* Get fabric info */
	ret = fi_getinfo(FT_FIVERSION, node, service, flags, hints, &fi);
	if (ret) {
		FT_PRINTERR("fi_getinfo", ret);
		return ret;
	}

	ret = ft_open_fabric_res();
	if (ret)
		return ret;

	if (opts.dst_addr && !ft_parent_proc) {
		/* child waits until parent is done creating AV */
		ret = ft_sync_pair(FI_SUCCESS);

		if (ret)
			return ret;

		/* child needs to open AV in read only mode */
		av_attr.flags = FI_READ;
	}
	ret = ft_alloc_active_res(fi);

	if (opts.dst_addr && ft_parent_proc) {
		/* parent lets the child know its status */
		ret = ft_sync_pair(ret);
	}

	/* handle the failed alloc_active_res call */
	if (ret)
		return ret;

	return ft_init_ep();
}

static int send_recv()
{
	int ret;

	if (opts.dst_addr) {
		fprintf(stdout, "Sending message...\n");
		snprintf(tx_buf, FT_MAX_CTRL_MSG, "Hello from Child Client!");

		ret = ft_tx(strlen(tx_buf));
		if (ret)
			return ret;

		fprintf(stdout, "Send completion received\n");
	} else {
		fprintf(stdout, "Waiting for message from client...\n");

		ret = ft_get_rx_comp(rx_seq);
		if (ret)
			return ret;

		fprintf(stdout, "Received data from client: %s\n", (char *)buf);
	}

	return 0;
}

static int run(void)
{
	int ret;

	ret = init_fabric();
	if (ret)
		return ret;

	if (opts.dst_addr) {
		if (ft_parent_proc) {
			/* parent inits AV and lets child proceed,
			 * and itself returns without sending a message */
			ret = ft_init_av();

			ret = ft_sync_pair(ret);

			/* parent doesn't run the send_recv loop,
			 * it waits for the child until it is done
			 * with send_recv */
			return ret;
		} else {
			ret = ft_sync_pair(FI_SUCCESS);
			if (ret)
				return ret;

			remote_fi_addr = ((fi_addr_t *)av_attr.map_addr)[0];
		}
	} else {
		ret = ft_init_av();
		if (ret)
			return ret;
	}

	return send_recv();
}

int main(int argc, char **argv)
{
	int op, ret;

	opts = INIT_OPTS;
	opts.options |= FT_OPT_SIZE;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt(argc, argv, "h" ADDR_OPTS INFO_OPTS)) != -1) {
		switch (op) {
		default:
			ft_parse_addr_opts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints);
			ft_parsecsopts(op, optarg, &opts);
			break;
		case '?':
		case 'h':
			ft_usage(argv[0], "A shared AV client-server example.");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	if (opts.dst_addr) {
		ret = ft_fork_and_pair();
		if (ret)
			return ret;

		if (!opts.av_name)
			opts.av_name = "foo";
	}

	hints->ep_attr->type	= FI_EP_RDM;
	hints->caps		= FI_MSG;
	hints->mode		= FI_CONTEXT | FI_LOCAL_MR;

	ret = run();

	if (opts.dst_addr) {
		if (ft_wait_child()) {
			FT_PRINTERR("ft_wait_child", errno);
		}
	}

	ft_free_res();

	return -ret;
}
