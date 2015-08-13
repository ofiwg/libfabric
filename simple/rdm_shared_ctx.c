/*
 * Copyright (c) 2013-2014 Intel Corporation.  All rights reserved.
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
#include <netdb.h>
#include <unistd.h>

#include <rdma/fabric.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <shared.h>


static size_t transfer_size = 1000;
static int rx_depth = 512;

static int ep_cnt = 2;
static struct fid_ep **ep_array, *srx_ctx;
static struct fid_stx *stx_ctx;
static void *local_addr, *remote_addr;
static size_t addrlen = 0;
static fi_addr_t *remote_fi_addr;
struct fi_context fi_ctx_send;
struct fi_context fi_ctx_recv;
struct fi_context fi_ctx_av;

static int send_msg(int size)
{
	int ret;

	ret = fi_send(ep_array[0], buf, (size_t) size, fi_mr_desc(mr),
			remote_fi_addr[0], &fi_ctx_send);
	if (ret) {
		FT_PRINTERR("fi_send", ret);
		return ret;
	}

	ret = wait_for_completion(txcq, 1);

	return ret;
}

static int recv_msg(void)
{
	int ret;

	ret = fi_recv(srx_ctx, buf, buffer_size, fi_mr_desc(mr), 0, &fi_ctx_recv);
	if (ret) {
		FT_PRINTERR("fi_recv", ret);
		return ret;
	}

	ret = wait_for_completion(rxcq, 1);

	return ret;
}

static int alloc_ep_res(struct fi_info *fi)
{
	struct fi_cq_attr cq_attr;
	struct fi_rx_attr rx_attr;
	struct fi_tx_attr tx_attr;
	struct fi_av_attr av_attr;
	int i, ret = 0;

	buffer_size = test_size[TEST_CNT - 1].size;
	buf = malloc(buffer_size);

	remote_fi_addr = (fi_addr_t *)malloc(sizeof(*remote_fi_addr) * ep_cnt);

	if (!buf || !remote_fi_addr) {
		perror("malloc");
		return -FI_ENOMEM;
	}

	memset(&cq_attr, 0, sizeof cq_attr);
	cq_attr.format = FI_CQ_FORMAT_CONTEXT;
	cq_attr.wait_obj = FI_WAIT_NONE;
	cq_attr.size = rx_depth;

	memset(&tx_attr, 0, sizeof tx_attr);
	memset(&rx_attr, 0, sizeof rx_attr);

	ret = fi_stx_context(domain, &tx_attr, &stx_ctx, NULL);
	if (ret) {
		FT_PRINTERR("fi_stx_context", ret);
		return ret;
	}

	ret = fi_cq_open(domain, &cq_attr, &txcq, NULL);
	if (ret) {
		FT_PRINTERR("fi_cq_open", ret);
		return ret;
	}

	ret = fi_srx_context(domain, &rx_attr, &srx_ctx, NULL);
	if (ret) {
		FT_PRINTERR("fi_srx_context", ret);
		return ret;
	}

	ret = fi_cq_open(domain, &cq_attr, &rxcq, NULL);
	if (ret) {
		FT_PRINTERR("fi_cq_open", ret);
		return ret;
	}

	ret = fi_mr_reg(domain, buf, buffer_size, 0, 0, 0, 0, &mr, NULL);
	if (ret) {
		FT_PRINTERR("fi_mr_reg", ret);
		return ret;
	}

	memset(&av_attr, 0, sizeof av_attr);
	av_attr.type = fi->domain_attr->av_type ?
			fi->domain_attr->av_type : FI_AV_MAP;
	av_attr.count = ep_cnt;

	ret = fi_av_open(domain, &av_attr, &av, NULL);
	if (ret) {
		FT_PRINTERR("fi_av_open", ret);
		return ret;
	}

	ep_array = calloc(ep_cnt, sizeof(*ep_array));
	if (!ep_array) {
		perror("malloc");
		return ret;
	}
	for (i = 0; i < ep_cnt; i++) {
		ret = fi_endpoint(domain, fi, &ep_array[i], NULL);
		if (ret) {
			FT_PRINTERR("fi_endpoint", ret);
			return ret;
		}
	}

	return 0;
}

static int bind_ep_res(void)
{
	int i, ret;

	for (i = 0; i < ep_cnt; i++) {
		ret = fi_ep_bind(ep_array[i], &stx_ctx->fid, 0);
		if (ret) {
			FT_PRINTERR("fi_ep_bind", ret);
			return ret;
		}

		ret = fi_ep_bind(ep_array[i], &srx_ctx->fid, 0);
		if (ret) {
			FT_PRINTERR("fi_ep_bind", ret);
			return ret;
		}

		ret = fi_ep_bind(ep_array[i], &txcq->fid, FI_SEND);
		if (ret) {
			FT_PRINTERR("fi_ep_bind", ret);
			return ret;
		}

		ret = fi_ep_bind(ep_array[i], &rxcq->fid, FI_RECV);
		if (ret) {
			FT_PRINTERR("fi_ep_bind", ret);
			return ret;
		}

		ret = fi_ep_bind(ep_array[i], &av->fid, 0);
		if (ret) {
			FT_PRINTERR("fi_ep_bind", ret);
			return ret;
		}

		ret = fi_enable(ep_array[i]);
		if (ret) {
			FT_PRINTERR("fi_enable", ret);
			return ret;
		}
	}

	return ret;
}

static int run_test()
{
	int ret, i;

	/* Post recvs */
	for (i = 0; i < ep_cnt; i++) {
		fprintf(stdout, "Posting recv for ctx: %d\n", i);
		ret = fi_recv(srx_ctx, buf, buffer_size, fi_mr_desc(mr),
				FI_ADDR_UNSPEC, NULL);
		if (ret) {
			FT_PRINTERR("fi_recv", ret);
			return ret;
		}
	}

	if (opts.dst_addr) {
		/* Post sends addressed to remote EPs */
		for (i = 0; i < ep_cnt; i++) {
			fprintf(stdout, "Posting send to remote ctx: %d\n", i);
			ret = fi_send(ep_array[i], buf, transfer_size, fi_mr_desc(mr),
					remote_fi_addr[i], NULL);
			if (ret) {
				FT_PRINTERR("fi_send", ret);
				return ret;
			}

			wait_for_completion(txcq, 1);
		}
	}

	/* Wait for recv completions */
	for (i = 0; i < ep_cnt; i++) {
		wait_for_completion(rxcq, 1);
	}

	if (!opts.dst_addr) {
		/* Post sends addressed to remote EPs */
		for (i = 0; i < ep_cnt; i++) {
			fprintf(stdout, "Posting send to remote ctx: %d\n", i);
			ret = fi_send(ep_array[i], buf, transfer_size, fi_mr_desc(mr),
					remote_fi_addr[i], NULL);
			if (ret) {
				FT_PRINTERR("fi_send", ret);
				return ret;
			}

			wait_for_completion(txcq, 1);
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

	/* Check the number of EPs supported by the provider */
	if (ep_cnt > fi->domain_attr->ep_cnt) {
		ep_cnt = fi->domain_attr->ep_cnt;
		fprintf(stderr, "Provider can support only %d of EPs\n", ep_cnt);
	}

	/* Get remote address */
	if (opts.dst_addr) {
		addrlen = fi->dest_addrlen;
		remote_addr = malloc(addrlen * ep_cnt);
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

	fi->ep_attr->tx_ctx_cnt = FI_SHARED_CONTEXT;
	fi->ep_attr->rx_ctx_cnt = FI_SHARED_CONTEXT;

	ret = alloc_ep_res(fi);
	if (ret)
		return ret;

	ret = bind_ep_res();
	if (ret)
		return ret;

	return 0;
}

static int init_av(void)
{
	int ret;
	int i;

	/* Get local address blob. Find the addrlen first. We set addrlen
	 * as 0 and fi_getname will return the actual addrlen. */
	addrlen = 0;
	ret = fi_getname(&ep_array[0]->fid, local_addr, &addrlen);
	if (ret != -FI_ETOOSMALL) {
		FT_PRINTERR("fi_getname", ret);
		return ret;
	}

	local_addr = malloc(addrlen * ep_cnt);

	/* Get local addresses for all EPs */
	for (i = 0; i < ep_cnt; i++) {
		ret = fi_getname(&ep_array[i]->fid, local_addr + addrlen * i, &addrlen);
		if (ret) {
			FT_PRINTERR("fi_getname", ret);
			return ret;
		}
	}

	if (opts.dst_addr) {
		ret = fi_av_insert(av, remote_addr, 1, &remote_fi_addr[0], 0, &fi_ctx_av);
		if (ret != 1) {
			FT_PRINTERR("fi_av_insert", ret);
			return ret;
		}

		/* Send local EP addresses to one of the remote endpoints */
		memcpy(buf, &addrlen, sizeof(size_t));
		memcpy(buf + sizeof(size_t), local_addr, addrlen * ep_cnt);
		ret = send_msg(sizeof(size_t) + addrlen * ep_cnt);
		if (ret)
			return ret;

		/* Get remote EP addresses */
		ret = recv_msg();
		if (ret)
			return ret;

		memcpy(&addrlen, buf, sizeof(size_t));
		memcpy(remote_addr, buf + sizeof(size_t), addrlen * ep_cnt);

		/* Insert remote addresses into AV
		 * Skip the first address since we already have it in AV */
		ret = fi_av_insert(av, remote_addr + addrlen, ep_cnt - 1,
				remote_fi_addr + 1, 0, &fi_ctx_av);
		if (ret != ep_cnt - 1) {
			FT_PRINTERR("fi_av_insert", ret);
			return ret;
		}

		/* Send ACK */
		ret = send_msg(16);
		if (ret)
			return ret;

	} else {
		/* Get remote EP addresses */
		ret = recv_msg();
		if (ret)
			return ret;

		memcpy(&addrlen, buf, sizeof(size_t));
		remote_addr = malloc(addrlen * ep_cnt);
		memcpy(remote_addr, buf + sizeof(size_t), addrlen * ep_cnt);

		/* Insert remote addresses into AV */
		ret = fi_av_insert(av, remote_addr, ep_cnt, remote_fi_addr, 0, &fi_ctx_av);
		if (ret != ep_cnt) {
			FT_PRINTERR("fi_av_insert", ret);
			return ret;
		}

		/* Send local EP addresses to one of the remote endpoints */
		memcpy(buf, &addrlen, sizeof(size_t));
		memcpy(buf + sizeof(size_t), local_addr, addrlen * ep_cnt);
		ret = send_msg(sizeof(size_t) + addrlen * ep_cnt);
		if (ret)
			return ret;

		/* Receive ACK from client */
		ret = recv_msg();
		if (ret)
			return ret;
	}

	free(local_addr);
	free(remote_addr);
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
		goto out;

	run_test();
	/* TODO: Add a local finalize applicable to shared ctx */
	//ft_finalize(fi, ep_array[0], txcq, rxcq, remote_fi_addr[0]);
out:
	return ret;
}

int main(int argc, char **argv)
{
	int op, ret;
	opts = INIT_OPTS;

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
			ft_usage(argv[0], "An RDM client-server example that uses shared context.\n");
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

	FT_CLOSEV_FID(ep_array, ep_cnt);
	FT_CLOSE_FID(srx_ctx);
	FT_CLOSE_FID(stx_ctx);
	ft_free_res();
	free(remote_fi_addr);
	free(ep_array);
	return -ret;
}
