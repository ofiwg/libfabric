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
#include <math.h>


static int ctx_cnt = 2;
static int rx_ctx_bits = 0;
static struct fid_ep *sep;
static struct fid_ep **tx_ep, **rx_ep;
static struct fid_cq **scq_array;
static struct fid_cq **rcq_array;
static void *local_addr, *remote_addr;
static size_t addrlen = 0;
static fi_addr_t remote_fi_addr;
struct fi_context fi_ctx_send;
struct fi_context fi_ctx_recv;
struct fi_context fi_ctx_av;
static fi_addr_t *remote_rx_addr;

static int send_msg(int size)
{
	int ret;

	ret = fi_send(tx_ep[0], buf, (size_t) size, fi_mr_desc(mr),
			remote_rx_addr[0], &fi_ctx_send);
	if (ret) {
		FT_PRINTERR("fi_send", ret);
		return ret;
	}

	ret = wait_for_completion(scq_array[0], 1);

	return ret;
}

static int recv_msg(void)
{
	int ret;

	/* Messages sent to scalable EP fi_addr are received in context 0 */
	ret = fi_recv(rx_ep[0], buf, rx_size, fi_mr_desc(mr), 0, &fi_ctx_recv);
	if (ret) {
		FT_PRINTERR("fi_recv", ret);
		return ret;
	}

	ret = wait_for_completion(rcq_array[0], 1);

	return ret;
}

static void free_res(void)
{
	if (rx_ep) {
		FT_CLOSEV_FID(rx_ep, ctx_cnt);
		free(rx_ep);
		rx_ep = NULL;
	}
	if (tx_ep) {
		FT_CLOSEV_FID(tx_ep, ctx_cnt);
		free(tx_ep);
		tx_ep = NULL;
	}
	if (rcq_array) {
		FT_CLOSEV_FID(rcq_array, ctx_cnt);
		free(rcq_array);
		rcq_array = NULL;
	}
	if (scq_array) {
		FT_CLOSEV_FID(scq_array, ctx_cnt);
		free(scq_array);
		scq_array = NULL;
	}
}

static int alloc_ep_res(struct fid_ep *sep)
{
	int i, ret;

	ret = ft_alloc_bufs();
	if (ret)
		return ret;

	/* Get number of bits needed to represent ctx_cnt */
	while (ctx_cnt >> ++rx_ctx_bits)
		;

	av_attr.rx_ctx_bits = rx_ctx_bits;

	ret = ft_alloc_active_res(fi);
	if (ret)
		return ret;

	FT_CLOSE_FID(ep);

	scq_array = calloc(ctx_cnt, sizeof *scq_array);
	rcq_array = calloc(ctx_cnt, sizeof *rcq_array);
	tx_ep = calloc(ctx_cnt, sizeof *tx_ep);
	rx_ep = calloc(ctx_cnt, sizeof *rx_ep);
	remote_rx_addr = calloc(ctx_cnt, sizeof *remote_rx_addr);

	if (!buf || !scq_array || !rcq_array || !tx_ep || !rx_ep || !remote_rx_addr) {
		perror("malloc");
		return -1;
	}

	for (i = 0; i < ctx_cnt; i++) {
		ret = fi_tx_context(sep, i, NULL, &tx_ep[i], NULL);
		if (ret) {
			FT_PRINTERR("fi_tx_context", ret);
			return ret;
		}

		ret = fi_cq_open(domain, &cq_attr, &scq_array[i], NULL);
		if (ret) {
			FT_PRINTERR("fi_cq_open", ret);
			return ret;
		}

		ret = fi_rx_context(sep, i, NULL, &rx_ep[i], NULL);
		if (ret) {
			FT_PRINTERR("fi_tx_context", ret);
			return ret;
		}

		ret = fi_cq_open(domain, &cq_attr, &rcq_array[i], NULL);
		if (ret) {
			FT_PRINTERR("fi_cq_open", ret);
			return ret;
		}
	}

	return 0;
}

static int bind_ep_res(void)
{
	int i, ret;

	for (i = 0; i < ctx_cnt; i++) {
		ret = fi_ep_bind(tx_ep[i], &scq_array[i]->fid, FI_SEND);
		if (ret) {
			FT_PRINTERR("fi_ep_bind", ret);
			return ret;
		}

		ret = fi_enable(tx_ep[i]);
		if (ret) {
			FT_PRINTERR("fi_enable", ret);
			return ret;
		}
	}

	for (i = 0; i < ctx_cnt; i++) {
		ret = fi_ep_bind(rx_ep[i], &rcq_array[i]->fid, FI_RECV);
		if (ret) {
			FT_PRINTERR("fi_ep_bind", ret);
			return ret;
		}

		ret = fi_enable(rx_ep[i]);
		if (ret) {
			FT_PRINTERR("fi_enable", ret);
			return ret;
		}
	}

	/* Bind scalable EP with AV */
	ret = fi_scalable_ep_bind(sep, &av->fid, 0);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		return ret;
	}

	return 0;
}

static int run_test()
{
	int ret, i;

	/* Post recvs */
	for (i = 0; i < ctx_cnt; i++) {
		fprintf(stdout, "Posting recv for ctx: %d\n", i);
		ret = fi_recv(rx_ep[i], buf, rx_size, fi_mr_desc(mr), 0, NULL);
		if (ret) {
			FT_PRINTERR("fi_recv", ret);
			return ret;
		}
	}

	if (opts.dst_addr) {
		/* Post sends directly to each of the recv contexts */
		for (i = 0; i < ctx_cnt; i++) {
			fprintf(stdout, "Posting send for ctx: %d\n", i);
			ret = fi_send(tx_ep[i], buf, tx_size, fi_mr_desc(mr),
					remote_rx_addr[i], NULL);
			if (ret) {
				FT_PRINTERR("fi_recv", ret);
				return ret;
			}

			wait_for_completion(scq_array[i], 1);
		}
	} else {
		for (i = 0; i < ctx_cnt; i++) {
			fprintf(stdout, "wait for recv completion for ctx: %d\n", i);
			wait_for_completion(rcq_array[i], 1);
		}
	}

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

	/* Check the optimal number of TX and RX contexts supported by the provider */
	ctx_cnt = MIN(ctx_cnt, fi->domain_attr->tx_ctx_cnt);
	ctx_cnt = MIN(ctx_cnt, fi->domain_attr->rx_ctx_cnt);
	if (!ctx_cnt) {
		fprintf(stderr, "Provider doesn't support contexts\n");
		return 1;
	}

	/* Get remote address */
	if (opts.dst_addr) {
		addrlen = fi->dest_addrlen;
		remote_addr = malloc(addrlen);
		memcpy(remote_addr, fi->dest_addr, addrlen);
	}

	ret = ft_open_fabric_res();
	if (ret)
		return ret;

	ret = fi_domain(fabric, fi, &domain, NULL);
	if (ret) {
		FT_PRINTERR("fi_domain", ret);
		return ret;
	}

	/* Set the required number of TX and RX context counts */
	fi->ep_attr->tx_ctx_cnt = ctx_cnt;
	fi->ep_attr->rx_ctx_cnt = ctx_cnt;

	ret = fi_scalable_ep(domain, fi, &sep, NULL);
	if (ret) {
		FT_PRINTERR("fi_scalable_ep", ret);
		return ret;
	}

	ret = alloc_ep_res(sep);
	if (ret)
		return ret;

	ret = bind_ep_res();
	return ret;
}

static int init_av(void)
{
	int ret;
	int i;

	if (opts.dst_addr) {
		/* Get local address blob. Find the addrlen first. We set addrlen
		 * as 0 and fi_getname will return the actual addrlen. */
		addrlen = 0;
		ret = fi_getname(&sep->fid, local_addr, &addrlen);
		if (ret != -FI_ETOOSMALL) {
			FT_PRINTERR("fi_getname", ret);
			return ret;
		}

		local_addr = malloc(addrlen);
		ret = fi_getname(&sep->fid, local_addr, &addrlen);
		if (ret) {
			FT_PRINTERR("fi_getname", ret);
			return ret;
		}

		ret = fi_av_insert(av, remote_addr, 1, &remote_fi_addr, 0, &fi_ctx_av);
		if (ret != 1) {
			FT_PRINTERR("fi_av_insert", ret);
			return ret;
		}

		for (i = 0; i < ctx_cnt; i++)
			remote_rx_addr[i] = fi_rx_addr(remote_fi_addr, i, rx_ctx_bits);

		/* Send local addr size and local addr */
		memcpy(buf, &addrlen, sizeof(size_t));
		memcpy(buf + sizeof(size_t), local_addr, addrlen);
		ret = send_msg(sizeof(size_t) + addrlen);
		if (ret)
			return ret;

		/* Receive ACK from server */
		ret = recv_msg();
		if (ret)
			return ret;

	} else {
		/* Post a recv to get the remote address */
		ret = recv_msg();
		if (ret)
			return ret;

		memcpy(&addrlen, buf, sizeof(size_t));
		remote_addr = malloc(addrlen);
		memcpy(remote_addr, buf + sizeof(size_t), addrlen);

		ret = fi_av_insert(av, remote_addr, 1, &remote_fi_addr, 0, &fi_ctx_av);
		if (ret != 1) {
			FT_PRINTERR("fi_av_insert", ret);
			return ret;
		}

		/* Send ACK */
		ret = send_msg(16);
		if (ret)
			return ret;
	}

	return 0;
}

static int run(void)
{
	int ret = 0;

	ret = init_fabric();
	if (ret)
		return ret;

	ret = init_av();
	if (ret)
		return ret;

	ret = run_test();

	/*TODO: Add a local finalize applicable for scalable ep */
	//ft_finalize(fi, tx_ep[0], scq_array[0], rcq_array[0], remote_rx_addr[0]);

	return ret;
}

int main(int argc, char **argv)
{
	int ret, op;

	opts = INIT_OPTS;
	opts.options = FT_OPT_SIZE;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt(argc, argv, "h" ADDR_OPTS INFO_OPTS)) != -1) {
		switch (op) {
		default:
			ft_parse_addr_opts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints);
			break;
		case '?':
		case 'h':
			ft_usage(argv[0], "An RDM client-server example with scalable endpoints.\n");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_MSG | FI_NAMED_RX_CTX;
	hints->mode = FI_CONTEXT | FI_LOCAL_MR;
	hints->addr_format = FI_SOCKADDR;

	ret = run();

	free_res();
	ft_free_res();
	return -ret;
}
