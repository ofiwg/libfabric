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

#include <rdma/fabric.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <shared.h>

// MULTI_BUF_SIZE_FACTOR defines how large the multi recv buffer will be.
// The minimum value of the factor is 2 which will set the multi recv buffer
// size to be twice the size of the send buffer. In order to use FI_MULTI_RECV
// feature efficiently, we need to have a large recv buffer so that we don't
// to repost the buffer often to get the remaining data when the buffer is full
#define MULTI_BUF_SIZE_FACTOR 4
#define DEFAULT_MULTI_BUF_SIZE (1024 * 1024)


static char test_name[10] = "custom";
static struct timespec start, end;

static struct fid_mr *mr_multi_recv;
struct fi_context ctx_multi_recv[2];


int wait_for_recv_completion(int num_completions)
{
	int i, ret;
	struct fi_cq_data_entry comp;

	while (num_completions > 0) {
		ret = fi_cq_read(rxcq, &comp, 1);
		if (ret == -FI_EAGAIN)
			continue;

		if (ret < 0) {
			FT_PRINTERR("fi_cq_read", ret);
			return ret;
		}

		if (comp.len)
			num_completions--;

		if (comp.flags & FI_MULTI_RECV) {
			i = (comp.op_context == &ctx_multi_recv[0]) ? 0 : 1;

			ret = fi_recv(ep, rx_buf + (rx_size / 2) * i,
					rx_size / 2, fi_mr_desc(mr_multi_recv),
					0, &ctx_multi_recv[i]);
			if (ret) {
				FT_PRINTERR("fi_recv", ret);
				return ret;
			}
		}
	}
	return 0;
}

static int send_msg(size_t size)
{
	int ret;

	ret = fi_send(ep, tx_buf, size, fi_mr_desc(mr), remote_fi_addr, &tx_ctx);
	if (ret) {
		FT_PRINTERR("fi_send", ret);
		return ret;
	}

	ret = ft_wait_for_comp(txcq, 1);
	return ret;
}

static int sync_test(void)
{
	int ret;

	ret = opts.dst_addr ? send_msg(16) : wait_for_recv_completion(1);
	if (ret)
		return ret;

	return opts.dst_addr ? wait_for_recv_completion(1) : send_msg(16);
}

/*
 * Post buffer as two halves, so that we can repost one half
 * when the other half is full.
 */
static int post_multi_recv_buffer()
{
	int ret, i;

	for (i = 0; i < 2; i++) {
		ret = fi_recv(ep, rx_buf + (rx_size / 2) * i,
				rx_size / 2, fi_mr_desc(mr_multi_recv),
				0, &ctx_multi_recv[i]);
		if (ret) {
			FT_PRINTERR("fi_recv", ret);
			return ret;
		}
	}

	return 0;
}

static int run_test(void)
{
	int ret, i;

	ret = sync_test();
	if (ret) {
		fprintf(stderr, "sync_test failed!\n");
		goto out;
	}

	clock_gettime(CLOCK_MONOTONIC, &start);
	if (opts.dst_addr) {
		for (i = 0; i < opts.iterations; i++) {
			ret = send_msg(opts.transfer_size);
			if (ret)
				goto out;
		}
	} else {
		ret = wait_for_recv_completion(opts.iterations);
		if (ret)
			goto out;
	}

	clock_gettime(CLOCK_MONOTONIC, &end);

	if (opts.machr)
		show_perf_mr(opts.transfer_size, opts.iterations,
			&start, &end, 1, opts.argc, opts.argv);
	else
		show_perf(test_name, opts.transfer_size, opts.iterations,
			&start, &end, 1);

out:
	return ret;
}

static void free_res(void)
{
	FT_CLOSE_FID(mr_multi_recv);
	if (tx_buf) {
		free(tx_buf);
		tx_buf = NULL;
	}
	if (rx_buf) {
		free(rx_buf);
		rx_buf = NULL;
	}
}

static int alloc_ep_res(struct fi_info *fi)
{
	int ret;

	tx_size = MAX(FT_MAX_CTRL_MSG, opts.transfer_size);
	if (tx_size > fi->ep_attr->max_msg_size) {
		fprintf(stderr, "transfer size is larger than the maximum size "
				"of the data transfer supported by the provider\n");
		return -1;
	}

	tx_buf = malloc(tx_size);
	if (!tx_buf) {
		fprintf(stderr, "Cannot allocate tx_buf\n");
		return -1;
	}

	ret = fi_mr_reg(domain, tx_buf, tx_size, FI_SEND,
			0, FT_MR_KEY, 0, &mr, NULL);
	if (ret) {
		FT_PRINTERR("fi_mr_reg", ret);
		return ret;
	}

	// set the multi buffer size to be allocated
	rx_size = MAX(tx_size, DEFAULT_MULTI_BUF_SIZE) * MULTI_BUF_SIZE_FACTOR;
	rx_buf = malloc(rx_size);
	if (!rx_buf) {
		fprintf(stderr, "Cannot allocate rx_buf\n");
		return -1;
	}

	ret = fi_mr_reg(domain, rx_buf, rx_size, FI_RECV, 0, FT_MR_KEY + 1, 0,
			&mr_multi_recv, NULL);
	if (ret) {
		FT_PRINTERR("fi_mr_reg", ret);
		return ret;
	}

	ret = ft_alloc_active_res(fi);
	if (ret)
		return ret;

	return 0;
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

	// set FI_MULTI_RECV flag for all recv operations
	fi->rx_attr->op_flags = FI_MULTI_RECV;

	ret = ft_open_fabric_res();
	if (ret)
		return ret;

	ret = alloc_ep_res(fi);
	if (ret)
		return ret;

	ret = ft_init_ep();
	if (ret)
		return ret;

	ret = fi_setopt(&ep->fid, FI_OPT_ENDPOINT, FI_OPT_MIN_MULTI_RECV,
			&tx_size, sizeof(tx_size));
	if (ret)
		return ret;

	// post the initial receive buffers to get MULTI_RECV data
	ret = post_multi_recv_buffer();
	return ret;
}

static int init_av(void)
{
	size_t addrlen;
	int ret;

	if (opts.dst_addr) {
		ret = fi_av_insert(av, fi->dest_addr, 1, &remote_fi_addr, 0, NULL);
		if (ret < 0) {
			FT_PRINTERR("fi_av_insert", ret);
			return ret;
		} else if (ret != 1) {
			FT_ERR("fi_av_insert: number of inserted address = %d\n", ret);
			return -1;
		}

		addrlen = 64;
		ret = fi_getname(&ep->fid, tx_buf, &addrlen);
		if (ret) {
			FT_PRINTERR("fi_getname", ret);
			return ret;
		}

		ret = send_msg(addrlen);
		if (ret)
			return ret;
	} else {
		ret = wait_for_recv_completion(1);
		if (ret)
			return ret;

		ret = fi_av_insert(av, rx_buf, 1, &remote_fi_addr, 0, NULL);
		if (ret < 0) {
			FT_PRINTERR("fi_av_insert", ret);
			return ret;
		} else if (ret != 1) {
			FT_ERR("fi_av_insert: number of inserted address = %d\n", ret);
			return -1;
		}
	}

	return 0;
}

static int run(void)
{
	int ret = 0;

	ret = init_fabric();
	if (ret)
		goto out;

	ret = init_av();
	if (ret)
		goto out;

	ret = run_test();
	ft_finalize(fi, ep, txcq, rxcq, remote_fi_addr);
out:
	return ret;
}

int main(int argc, char **argv)
{
	int op, ret;

	opts = INIT_OPTS;
	opts.options |= FT_OPT_SIZE;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt(argc, argv, "h" CS_OPTS INFO_OPTS)) != -1) {
		switch (op) {
		default:
			ft_parseinfo(op, optarg, hints);
			ft_parsecsopts(op, optarg, &opts);
			break;
		case '?':
		case 'h':
			ft_csusage(argv[0], "Ping pong client and server using message endpoints.");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_MSG | FI_MULTI_RECV;
	hints->mode = FI_CONTEXT | FI_LOCAL_MR;

	cq_attr.format = FI_CQ_FORMAT_DATA;

	ret = run();

	free_res();
	ft_free_res();
	return -ret;
}
