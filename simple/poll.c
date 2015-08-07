/*
 * Copyright (c) 2013-2014 Intel Corporation.  All rights reserved.
 * Copyright (c) 2015 Cisco Systems, Inc.  All rights reserved.
 *
 * This software is available to you under the BSD license below:
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

#define MAX_POLL_CNT 10

enum comp_type {
	CQ_SEND = 1,
	CQ_RECV = 2
};

static struct cs_opts opts;
static void *buf;
static size_t buffer_size = 1024;
static int transfer_size = 1000;
static int rx_depth = 512;

static struct fid_poll *pollset;
static void *local_addr, *remote_addr;
static size_t addrlen = 0;
static fi_addr_t remote_fi_addr;
struct fi_context fi_ctx_send;
struct fi_context fi_ctx_recv;
struct fi_context fi_ctx_av;

static void free_ep_res(void)
{
	fi_close(&av->fid);
	fi_close(&mr->fid);
	fi_close(&pollset->fid);
	fi_close(&rxcq->fid);
	fi_close(&txcq->fid);
	free(buf);
	fi_close(&ep->fid);
}

static int alloc_ep_res(struct fi_info *fi)
{
	struct fi_cq_attr cq_attr;
	struct fi_av_attr av_attr;
	struct fi_poll_attr poll_attr;
	int ret;

	buf = malloc(buffer_size);
	if (!buf) {
		perror("malloc");
		return -1;
	}

	memset(&cq_attr, 0, sizeof cq_attr);
	cq_attr.format = FI_CQ_FORMAT_CONTEXT;
	cq_attr.wait_obj = FI_WAIT_NONE;
	cq_attr.size = rx_depth;

	/* Open completion queue for send completions */
	ret = fi_cq_open(domain, &cq_attr, &txcq, (void *)CQ_SEND);
	if (ret) {
		FT_PRINTERR("fi_cq_open", ret);
		goto err1;
	}

	/* Open completion queue for recv completions */
	ret = fi_cq_open(domain, &cq_attr, &rxcq, (void *)CQ_RECV);
	if (ret) {
		FT_PRINTERR("fi_cq_open", ret);
		goto err2;
	}

	/* Open a polling set */
	memset(&poll_attr, 0, sizeof poll_attr);
	ret = fi_poll_open(domain, &poll_attr, &pollset);
	if (ret) {
		FT_PRINTERR("fi_poll_open", ret);
		goto err2;
	}

	/* Add send CQ to the polling set */
	ret = fi_poll_add(pollset, &txcq->fid, 0);
	if (ret) {
		FT_PRINTERR("fi_poll_add", ret);
		goto err3;
	}

	/* Add recv CQ to the polling set */
	ret = fi_poll_add(pollset, &rxcq->fid, 0);
	if (ret) {
		FT_PRINTERR("fi_poll_add", ret);
		goto err3;
	}

	/* Register memory */
	ret = fi_mr_reg(domain, buf, buffer_size, 0, 0, 0, 0, &mr, NULL);
	if (ret) {
		FT_PRINTERR("fi_mr_reg", ret);
		goto err4;
	}

	memset(&av_attr, 0, sizeof av_attr);
	av_attr.type = fi->domain_attr->av_type ?
			fi->domain_attr->av_type : FI_AV_MAP;
	av_attr.count = 1;
	av_attr.name = NULL;

	/* Open Address Vector */
	ret = fi_av_open(domain, &av_attr, &av, NULL);
	if (ret) {
		FT_PRINTERR("fi_av_open", ret);
		goto err5;
	}

	ret = fi_endpoint(domain, fi, &ep, NULL);
	if (ret) {
		FT_PRINTERR("fi_endpoint", ret);
		goto err6;
	}

	return 0;

err6:
	fi_close(&av->fid);
err5:
	fi_close(&mr->fid);
err4:
	fi_close(&rxcq->fid);
err3:
	fi_close(&pollset->fid);
err2:
	fi_close(&txcq->fid);
err1:
	free(buf);
	return ret;
}

static int bind_ep_res(void)
{
	int ret;

	/* Bind AV and CQs with endpoint */
	ret = fi_ep_bind(ep, &txcq->fid, FI_SEND);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		return ret;
	}

	ret = fi_ep_bind(ep, &rxcq->fid, FI_RECV);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		return ret;
	}

	ret = fi_ep_bind(ep, &av->fid, 0);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		return ret;
	}

	ret = fi_enable(ep);
	if (ret) {
		FT_PRINTERR("fi_enable", ret);
		return ret;
	}

	return ret;
}

static int send_msg(int size)
{
	int ret;

	ret = fi_send(ep, buf, (size_t) size, fi_mr_desc(mr), remote_fi_addr,
			&fi_ctx_send);
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

	ret = fi_recv(ep, buf, buffer_size, fi_mr_desc(mr), 0, &fi_ctx_recv);
	if (ret) {
		FT_PRINTERR("fi_recv", ret);
		return ret;
	}

	ret = wait_for_completion(rxcq, 1);

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

	/* Get remote address */
	if (opts.dst_addr) {
		addrlen = fi->dest_addrlen;
		remote_addr = malloc(addrlen);
		memcpy(remote_addr, fi->dest_addr, addrlen);
	}

	ret = fi_fabric(fi->fabric_attr, &fabric, NULL);
	if (ret) {
		FT_PRINTERR("fi_fabric", ret);
		goto err0;
	}

	ret = fi_domain(fabric, fi, &domain, NULL);
	if (ret) {
		FT_PRINTERR("fi_domain", ret);
		goto err1;
	}

	ret = alloc_ep_res(fi);
	if (ret)
		goto err3;

	ret = bind_ep_res();
	if (ret)
		goto err4;

	return 0;

err4:
	free_ep_res();
err3:
	fi_close(&domain->fid);
err1:
	fi_close(&fabric->fid);
err0:
	fi_freeinfo(fi);

	return ret;
}

static int init_av(void)
{
	int ret;

	if (opts.dst_addr) {
		/* Get local address blob. Find the addrlen first. We set addrlen
		 * as 0 and fi_getname will return the actual addrlen. */
		addrlen = 0;
		ret = fi_getname(&ep->fid, local_addr, &addrlen);
		if (ret != -FI_ETOOSMALL) {
			FT_PRINTERR("fi_getname", ret);
			return ret;
		}

		local_addr = malloc(addrlen);
		ret = fi_getname(&ep->fid, local_addr, &addrlen);
		if (ret) {
			FT_PRINTERR("fi_getname", ret);
			return ret;
		}

		ret = fi_av_insert(av, remote_addr, 1, &remote_fi_addr, 0,
				&fi_ctx_av);
		if (ret != 1) {
			FT_PRINTERR("fi_av_insert", ret);
			return ret;
		}

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

		ret = fi_av_insert(av, remote_addr, 1, &remote_fi_addr, 0,
				&fi_ctx_av);
		if (ret != 1) {
			FT_PRINTERR("fi_av_insert", ret);
			return ret;
		}

		/* Send ACK */
		ret = send_msg(16);
		if (ret)
			return ret;
	}

	return ret;
}

static int send_recv()
{
	void *context[MAX_POLL_CNT];
	struct fi_cq_entry comp;
	int ret, send_pending = 0, recv_pending = 0;
	int ret_count = 0;
	int i;

	fprintf(stdout, "Posting a recv...\n");
	ret = fi_recv(ep, buf, transfer_size, fi_mr_desc(mr),
			remote_fi_addr, &fi_ctx_recv);
	if (ret) {
		FT_PRINTERR("fi_recv", ret);
		return ret;
	}
	recv_pending++;

	fprintf(stdout, "Posting a send...\n");
	ret = fi_send(ep, buf, transfer_size, fi_mr_desc(mr),
			remote_fi_addr, &fi_ctx_send);
	if (ret) {
		FT_PRINTERR("fi_send", ret);
		return ret;
	}
	send_pending++;

	while (send_pending || recv_pending) {
		struct fid_cq *cq;
		/* Poll send and recv CQs */
		do {
			ret_count = fi_poll(pollset, context, MAX_POLL_CNT);
			if (ret_count < 0) {
				FT_PRINTERR("fi_poll", ret_count);
				return ret_count;
			}
		} while (!ret_count);

		fprintf(stdout, "Retreived %d event(s)\n", ret_count);

		for (i = 0; i < ret_count; i++) {
			switch((enum comp_type)context[i]) {
			case CQ_SEND:
				printf("Send completion received\n");
				cq = txcq;
				send_pending--;
				break;
			case CQ_RECV:
				printf("Recv completion received\n");
				cq = rxcq;
				recv_pending--;
				break;
			default:
				printf("Unknown completion received\n");
				return -1;
			}

			/* Read the completion entry */
			ret = fi_cq_sread(cq, &comp, 1, NULL, -1);
			if (ret < 0) {
				if (ret == -FI_EAVAIL) {
					cq_readerr(cq, "cq");
				} else {
					FT_PRINTERR("fi_cq_sread", ret);
				}
				return ret;
			}
		}
	}

	return 0;
}

int main(int argc, char **argv)
{
	int op, ret = 0;
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
			ft_usage(argv[0], "A client-server example that uses poll.\n");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_MSG;
	hints->mode = FI_CONTEXT | FI_LOCAL_MR;

	ret = init_fabric();
	if (ret)
		return -ret;

	ret = init_av();
	if (ret)
		return ret;

	/* Exchange data */
	ret = send_recv();

	free_ep_res();
	fi_close(&domain->fid);
	fi_close(&fabric->fid);
	fi_freeinfo(hints);
	fi_freeinfo(fi);

	return ret;
}
