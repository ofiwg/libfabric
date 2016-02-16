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
#include <sys/wait.h>
#include <string.h>

#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>

#include "shared.h"

static int parent;
static int pair[2];
static int status;

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

	if (opts.dst_addr && !parent) {
		/* child waits until parent is done creating AV */
		ret = read(pair[0], &status, sizeof(int));
		if (ret < 0)
			FT_PRINTERR("read", errno);

		if (status != FI_SUCCESS)
			return status;

		/* child needs to open AV in read only mode */
		av_attr.flags = FI_READ;
	}
	ret = ft_alloc_active_res(fi);

	if (opts.dst_addr && parent) {
		/* parent lets the child know the return status of opening the
		 * AV
		 */
		if (write(pair[1], &ret, sizeof(int)) < 0)
			FT_PRINTERR("write", errno);
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
		if (parent) {
			/* parent inits AV and lets child proceed,
			 * and itself returns without sending a message */
			ret = ft_init_av();

			/* write init status to file */
			if (write(pair[1], &ret, sizeof(int)) < 0)
				FT_PRINTERR("write", errno);

			return ret;
		} else {
			/* client: child waits for parent to complete av_insert */
			ret = read(pair[0], &status, sizeof(int));
			if (ret < 0)
				FT_PRINTERR("read", errno);

			/* check status reported by parent */
			if (status != FI_SUCCESS)
				return status;

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
	pid_t child_pid = 0;

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
		ret = socketpair(AF_LOCAL, SOCK_STREAM, 0, pair);
		if (ret)
			FT_PRINTERR("socketpair", errno);
		child_pid = fork();
		if (child_pid < 0)
			FT_PRINTERR("fork", child_pid);
		if (child_pid)
			parent = 1;

		if (!opts.av_name)
			opts.av_name = "foo";
	}

	hints->ep_attr->type	= FI_EP_RDM;
	hints->caps		= FI_MSG;
	hints->mode		= FI_CONTEXT | FI_LOCAL_MR;

	ret = run();

	if (opts.dst_addr) {
		if (close(pair[0]))
			FT_PRINTERR("close", errno);
		if (close(pair[1]))
			FT_PRINTERR("close", errno);
		if (parent) {
			if (waitpid(child_pid, NULL, WCONTINUED) < 0) {
				FT_PRINTERR("waitpid", errno);
			}
		}
	}

	ft_free_res();

	return -ret;
}
