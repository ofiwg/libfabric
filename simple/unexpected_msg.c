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
#define FT_FIVERSION FI_VERSION(1, 5)
#include "shared.h"

char *sock_sync_port = "2710";

static size_t concurrent_msgs = 5;
static size_t num_iters = 600;
uint64_t num_send;
uint64_t num_recv;

static int wait_msg(struct fid_cq *cq, struct fi_cq_err_entry *status)
{
	int ret;

	if (opts.comp_method == FT_COMP_SREAD) {
		ret = fi_cq_sread(cq, (void *)status, 1, NULL, -1);
	} else {
		do {
			ret = fi_cq_read(cq, (void *)status, 1);
		} while (ret == -FI_EAGAIN);
	}

	if (ret < 1)
		printf("ERROR fi_cq_(s)read returned %d %s\n", ret, strerror(ret));
	return ret;
}

static int run_test_loop(void)
{
	int ret = 0;
	int i, j;
	int *sbufs[concurrent_msgs];
	int *rbufs[concurrent_msgs];
	int *tx_buffers, *rx_buffers;
	struct fi_context ctx_send[concurrent_msgs];
	struct fi_context ctx_recv[concurrent_msgs];

	tx_buffers = calloc(concurrent_msgs, opts.transfer_size);
	if (!tx_buffers)
		return -FI_ENOMEM;

	rx_buffers = calloc(concurrent_msgs, opts.transfer_size);
	if (!rx_buffers)
		return -FI_ENOMEM;

	for (j = 0; j < concurrent_msgs; j++) {
		sbufs[j] = (int *)((uintptr_t)tx_buffers + opts.transfer_size * j);
		rbufs[j] = (int *)((uintptr_t)rx_buffers + opts.transfer_size * j);
	}

	for (i = 0; i < num_iters; i++) {
		struct fi_cq_err_entry status;

		/* Init buffers and post sends */
		for (j = 0; j < concurrent_msgs; j++) {
			tx_buf = (void *)sbufs[j];
			if (ft_check_opts(FT_OPT_VERIFY_DATA))
				ft_fill_buf(tx_buf, opts.transfer_size);
			ft_tag = 0x1234;
			ret = ft_post_tx(ep, remote_fi_addr, opts.transfer_size, &ctx_send[j]);
			if (ret) {
				printf("ERROR send_msg returned %d\n", ret);
				goto cleanup_and_close;
			}
		}

		ret = ft_sock_sync(0);
		if (ret)
			goto cleanup_and_close;

		/* Post CONCURRENT_MSG recvs */
		for (j = 0; j < concurrent_msgs; j++) {
			rx_buf = (void *)rbufs[j];
			ft_tag = 0x1234;
			ret = ft_post_rx(ep, opts.transfer_size, &ctx_recv[j]);
			if (ret) {
				printf("ERROR recv_msg returned %d\n", ret);
				goto cleanup_and_close;
			}
		}

		/* Complete receives */
		for (j = 0; j < concurrent_msgs; j++) {
			ret = wait_msg(rxcq, &status);
			if (ret < 1)
				goto cleanup_and_close;

			if (ft_check_opts(FT_OPT_VERIFY_DATA)) {
				ret = ft_check_buf(status.buf,
						opts.transfer_size);
				if (ret)
					goto cleanup_and_close;
			}
		}

		/* Complete sends */
		for (j = 0; j < concurrent_msgs; j++) {
			ret = ft_get_tx_comp(tx_seq);
			if (ret)
				goto cleanup_and_close;
		}

		if (i % 100 == 0)
			printf("%d GOOD iter %d/%ld completed\n",
					getpid(), i, num_iters);
	}

	ft_sock_sync(0);
	printf("%d GOOD all done\n", getpid());

cleanup_and_close:
	free(tx_buffers);
	free(rx_buffers);

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
			}
			else
				break;
		}
		if (ret)
			return ret;
	} else {
		ret = ft_sock_listen(sock_sync_port);
		if (ret)
			return ret;
		ret = ft_sock_accept();
		if (ret)
			return ret;
	}

	ret = run_test_loop();

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

	while ((op = getopt(argc, argv, "m:i:c:vSh" ADDR_OPTS INFO_OPTS)) != -1) {
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
		case '?':
		case 'h':
			ft_usage(argv[0], "Unexpected message functional test");
			FT_PRINT_OPTS_USAGE("-c <int>", "Concurrent messages per iteration ");
			FT_PRINT_OPTS_USAGE("-v", "Enable DataCheck testing");
			FT_PRINT_OPTS_USAGE("-i <int>", "Number of iterations");
			FT_PRINT_OPTS_USAGE("-S", "Use fi_cq_sread instead of polling fi_cq_read");
			FT_PRINT_OPTS_USAGE("-m <size>", "Size of unexpected messages");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->mode = FI_CONTEXT;
	hints->domain_attr->resource_mgmt = FI_RM_ENABLED;
	hints->rx_attr->total_buffered_recv = 0;
	hints->caps = FI_TAGGED;

	ret = run_test();

	ft_free_res();
	ft_sock_shutdown(sock);
	return ft_exit_code(ret);
}
