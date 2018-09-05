/*
 * Copyright (c) 2013-2017 Intel Corporation. All rights reserved.
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
#include <string.h>
#include <getopt.h>
#include <time.h>
#include <netdb.h>
#include <unistd.h>
#include <stdbool.h>

#include <rdma/fabric.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_cm.h>

#include "shared.h"


char *sock_sync_port = "2710";

static size_t concurrent_msgs = 5;
static size_t num_iters = 600;
struct fi_context *tx_ctxs;
struct fi_context *rx_ctxs;
static bool send_data = false;
char *tx_bufs, *rx_bufs;
static struct fid_mr *tx_mr, *rx_mr;
static void *tx_mr_desc, *rx_mr_desc;
static size_t tx_len, rx_len;

static int alloc_bufs(void)
{
	int rc;

	tx_len = opts.transfer_size + ft_tx_prefix_size();
	rx_len = opts.transfer_size + ft_rx_prefix_size();
	tx_bufs = calloc(concurrent_msgs, tx_len);
	rx_bufs = calloc(concurrent_msgs, rx_len);
	tx_ctxs = malloc(sizeof(*tx_ctxs) * concurrent_msgs);
	rx_ctxs = malloc(sizeof(*rx_ctxs) * concurrent_msgs);
	if (!tx_bufs || !rx_bufs || !tx_ctxs || !rx_ctxs)
		return -FI_ENOMEM;

	if (fi->domain_attr->mr_mode & FI_MR_LOCAL) {
		rc = fi_mr_reg(domain, tx_bufs, concurrent_msgs * tx_len,
			       FI_SEND, 0, FT_MR_KEY + 1, 0, &tx_mr, NULL);
		if (rc)
			return rc;
		tx_mr_desc = fi_mr_desc(tx_mr);

		rc = fi_mr_reg(domain, rx_bufs, concurrent_msgs * rx_len,
			       FI_RECV, 0, FT_MR_KEY + 2, 0, &rx_mr, NULL);
		if (rc)
			return rc;
		rx_mr_desc = fi_mr_desc(rx_mr);
	}

	return 0;
}

static void free_bufs(void)
{
	FT_CLOSE_FID(tx_mr);
	FT_CLOSE_FID(rx_mr);
	free(tx_bufs);
	free(rx_bufs);
	free(tx_ctxs);
	free(rx_ctxs);
}

static char *get_tx_buf(int index)
{
	return tx_bufs + tx_len * index;
}

static char *get_rx_buf(int index)
{
	return rx_bufs + rx_len * index;
}

static int wait_recvs()
{
	struct fi_cq_tagged_entry entry;
	int ret;

	if (opts.comp_method == FT_COMP_SREAD) {
		ret = fi_cq_sread(rxcq, &entry, 1, NULL, -1);
	} else {
		do {
			ret = fi_cq_read(rxcq, &entry, 1);
		} while (ret == -FI_EAGAIN);
	}

	if ((ret == 1) && send_data) {
		if (entry.data != opts.transfer_size) {
			printf("ERROR incorrect remote CQ data value. Got %lu, expected %d\n",
					(unsigned long)entry.data, opts.transfer_size);
			return -FI_EOTHER;
		}
	}

	if (ret < 1)
		printf("ERROR fi_cq_(s)read returned %d %s\n", ret, fi_strerror(-ret));
	return ret;
}

static int run_test_loop(void)
{
	int ret = 0;
	uint64_t op_data = send_data ? opts.transfer_size : NO_CQ_DATA;
	uint64_t op_tag = 0x1234;
	char *op_buf;
	int i, j;

	for (i = 0; i < num_iters; i++) {
		for (j = 0; j < concurrent_msgs; j++) {
			op_buf = get_tx_buf(j);
			if (ft_check_opts(FT_OPT_VERIFY_DATA))
				ft_fill_buf(op_buf + ft_tx_prefix_size(),
					    opts.transfer_size);

			ret = ft_post_tx_buf(ep, remote_fi_addr,
					     opts.transfer_size,
					     op_data, &tx_ctxs[j],
					     op_buf, tx_mr_desc, op_tag);
			if (ret) {
				printf("ERROR send_msg returned %d\n", ret);
				return ret;
			}
		}

		ret = ft_sock_sync(0);
		if (ret)
			return ret;

		for (j = 0; j < concurrent_msgs; j++) {
			op_buf = get_rx_buf(j);
			ret = ft_post_rx_buf(ep, opts.transfer_size,
					     &rx_ctxs[j], op_buf,
					     rx_mr_desc, op_tag);
			if (ret) {
				printf("ERROR recv_msg returned %d\n", ret);
				return ret;
			}
		}

		for (j = 0; j < concurrent_msgs; j++) {
			ret = wait_recvs();
			if (ret < 1)
				return ret;
		}

		if (ft_check_opts(FT_OPT_VERIFY_DATA)) {
			for (j = 0; j < concurrent_msgs; j++) {
				op_buf = get_rx_buf(j);
				if (ft_check_buf(op_buf + ft_rx_prefix_size(),
						 opts.transfer_size))
					return -FI_EOTHER;
			}
		}

		for (j = 0; j < concurrent_msgs; j++) {
			ret = ft_get_tx_comp(tx_seq);
			if (ret)
				return ret;
		}

		if (i % 100 == 0)
			printf("%d GOOD iter %d/%ld completed\n",
				getpid(), i, num_iters);
	}

	ft_sock_sync(0);
	printf("%d GOOD all done\n", getpid());
	return ret;
}

static int run_test(void)
{
	int ret;
	int retries = 100;

	if (hints->ep_attr->type == FI_EP_MSG)
		ret = ft_init_fabric_cm();
	else
		ret = ft_init_fabric();
	if (ret)
		return ret;

	if (opts.dst_addr) {
		while (retries > 0) {
			ret = ft_sock_connect(opts.dst_addr, sock_sync_port);
			if (ret) {
				usleep(100);
				retries--;
			} else {
				break;
			}
		}
		if (ret)
			return ret;
	} else {
		ret = ft_sock_listen(opts.src_addr, sock_sync_port);
		if (ret)
			return ret;
		ret = ft_sock_accept();
		if (ret)
			return ret;
	}

	alloc_bufs();
	ret = run_test_loop();
	free_bufs();

	return ret;
}

int main(int argc, char **argv)
{
	int op;
	int ret;

	opts = INIT_OPTS;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt(argc, argv, "m:i:c:vdSh" ADDR_OPTS INFO_OPTS)) != -1) {
		switch (op) {
		default:
			ft_parse_addr_opts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints);
			break;
		case 'c':
			concurrent_msgs = strtoul(optarg, NULL, 0);
			break;
		case 'i':
			num_iters = strtoul(optarg, NULL, 0);
			break;
		case 'S':
			opts.comp_method = FT_COMP_SREAD;
			break;
		case 'v':
			opts.options |= FT_OPT_VERIFY_DATA;
			break;
		case 'm':
			opts.transfer_size = strtoul(optarg, NULL, 0);
			break;
		case 'd':
			send_data = true;
			break;
		case '?':
		case 'h':
			ft_usage(argv[0], "Unexpected message functional test");
			FT_PRINT_OPTS_USAGE("-c <int>",
				"Concurrent messages per iteration ");
			FT_PRINT_OPTS_USAGE("-v", "Enable DataCheck testing");
			FT_PRINT_OPTS_USAGE("-i <int>", "Number of iterations");
			FT_PRINT_OPTS_USAGE("-S",
				"Use fi_cq_sread instead of polling fi_cq_read");
			FT_PRINT_OPTS_USAGE("-m <size>",
				"Size of unexpected messages");
			FT_PRINT_OPTS_USAGE("-d",
				"Send remote CQ data");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->mode = FI_CONTEXT;
	hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_ALLOCATED;
	hints->domain_attr->resource_mgmt = FI_RM_ENABLED;
	hints->rx_attr->total_buffered_recv = 0;
	hints->caps = FI_TAGGED;

	ret = run_test();

	ft_free_res();
	ft_sock_shutdown(sock);
	return ft_exit_code(ret);
}
