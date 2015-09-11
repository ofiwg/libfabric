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

struct fi_context fi_ctx_search;

static uint64_t tag_data = 1;
static uint64_t tag_control = 0x12345678;



static int send_msg(int size, uint64_t tag)
{
	int ret;

	ret = fi_tsend(ep, buf, (size_t) size, fi_mr_desc(mr), remote_fi_addr,
			tag, &tx_ctx);
	if (ret)
		FT_PRINTERR("fi_tsend", ret);

	ret = ft_wait_for_comp(txcq, 1);

	return ret;
}

static int recv_msg(uint64_t tag)
{
	int ret;

	ret = fi_trecv(ep, buf, rx_size, fi_mr_desc(mr), remote_fi_addr,
			tag, 0, &rx_ctx);
	if (ret)
		FT_PRINTERR("fi_trecv", ret);

	ret = ft_wait_for_comp(rxcq, 1);
	return ret;
}

static int post_recv(uint64_t tag)
{
	int ret;

	ret = fi_trecv(ep, buf, rx_size, fi_mr_desc(mr), remote_fi_addr,
			tag, 0, &rx_ctx);
	if (ret)
		FT_PRINTERR("fi_trecv", ret);

	return ret;
}

static int sync_test(void)
{
	int ret;

	ret = opts.dst_addr ? send_msg(16, tag_control) : recv_msg(tag_control);
	if (ret)
		return ret;

	ret = opts.dst_addr ? recv_msg(tag_control) : send_msg(16, tag_control);

	return ret;
}

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

	ret = fi_domain(fabric, fi, &domain, NULL);
	if (ret) {
		FT_PRINTERR("fi_domain", ret);
		return ret;
	}

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
	int ret;
	struct fi_msg_tagged msg;

	memset(&msg, 0, sizeof msg);
	msg.tag = tag;
	msg.context = &fi_ctx_search;
	ret = fi_trecvmsg(ep, &msg, FI_PEEK);
	if (ret < 0) {
		if (ret == -ENOMSG)
			fprintf(stdout,
				"No match found with tag [%" PRIu64 "]\n",
				tag);
		else
			FT_PRINTERR("fi_tsearch", ret);
	} else {
		// search was initiated asynchronously, so wait for
		// the completion event
		ret = ft_wait_for_comp(rxcq, 1);
	}

	return ret;
}

static int run(void)
{
	int ret = 0;
	ret = init_fabric();
	if (ret)
		return ret;

	ret = ft_init_av();
	if (ret)
		goto out;

	// Receiver
	if (opts.dst_addr) {
		// search for initial tag, it should fail since the sender
		// hasn't sent anything
		fprintf(stdout, "Searching msg with tag [%" PRIu64 "]\n", tag_data);
		tagged_peek(tag_data);

		fprintf(stdout, "Posting buffer for msg with tag [%" PRIu64 "]\n",
				tag_data + 1);
		ret = post_recv(tag_data + 1);
		if (ret)
			goto out;

		// synchronize with sender
		fprintf(stdout, "\nSynchronizing with sender..\n\n");
		ret = sync_test();
		if (ret)
			goto out;

		ret = ft_wait_for_comp(rxcq, 1);
		if (ret)
			goto out;
		fprintf(stdout, "Received completion event for msg with tag "
				"[%" PRIu64 "]\n", tag_data + 1);

		// search again for the initial tag, and wait for completion,  it should be successful now
		fprintf(stdout,
			"Searching msg with initial tag [%" PRIu64 "]\n",
			tag_data);
		tagged_peek(tag_data);

		fprintf(stdout, "Posted buffer and received completion event for"
			       " msg with tag [%" PRIu64 "]\n", tag_data);
	} else {
		// Sender
		// synchronize with receiver
		fprintf(stdout, "Synchronizing with receiver..\n\n");
		ret = sync_test();
		if (ret)
			goto out;

		fprintf(stdout, "Sending msg with tag [%" PRIu64 "]\n",
			tag_data);
		ret = send_msg(16, tag_data);
		if (ret)
			goto out;

		fprintf(stdout, "Sending msg with tag [%" PRIu64 "]\n",
			tag_data + 1);
		ret = send_msg(16, tag_data + 1);
		if (ret)
			goto out;
	}
	/* Finalize before closing ep */
	ft_finalize(fi, ep, txcq, rxcq, remote_fi_addr);
out:
	return ret;
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
