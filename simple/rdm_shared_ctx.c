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

#define FI_CLOSEV(FID_C, NUM)				\
	do {						\
		int i;					\
		for (i = 0; i < NUM; i++)		\
			fi_close(&FID_C[i]->fid);	\
	} while (0)

static void *buf;
static size_t buffer_size = 1024;
static size_t transfer_size = 1000;
static int rx_depth = 512;

static char *dst_addr = NULL;
static char *port = "9228";
static struct fi_info hints;
static struct fi_domain_attr domain_hints;
static struct fi_ep_attr ep_hints;
static char *dst_addr, *src_addr;

static int ep_cnt = 2;
static struct fid_fabric *fab;
static struct fid_domain *dom;
static struct fid_ep **ep, *srx_ctx;
static struct fid_stx *stx_ctx;
static struct fid_cq *scq;
static struct fid_cq *rcq;
static struct fid_mr *mr;
static struct fid_av *av;
static void *local_addr, *remote_addr;
static size_t addrlen = 0;
static fi_addr_t *remote_fi_addr;
struct fi_context fi_ctx_send;
struct fi_context fi_ctx_recv;
struct fi_context fi_ctx_av;

static int send_msg(int size)
{
	int ret;

	ret = fi_send(ep[0], buf, (size_t) size, fi_mr_desc(mr),
			remote_fi_addr[0], &fi_ctx_send);
	if (ret) {
		FI_PRINTERR("fi_send", ret);
		return ret;
	}

	ret = wait_for_completion(scq, 1);

	return ret;
}

static int recv_msg(void)
{
	int ret;

	ret = fi_recv(srx_ctx, buf, buffer_size, fi_mr_desc(mr), 0, &fi_ctx_recv);
	if (ret) {
		FI_PRINTERR("fi_recv", ret);
		return ret;
	}

	ret = wait_for_completion(rcq, 1);

	return ret;
}

static void free_ep_res(void)
{
	fi_close(&av->fid);
	fi_close(&mr->fid);
	fi_close(&rcq->fid);
	fi_close(&srx_ctx->fid);
	fi_close(&scq->fid);
	fi_close(&stx_ctx->fid);
	free(buf);
	free(remote_fi_addr);
}

static int alloc_ep_res(void)
{
	struct fi_cq_attr cq_attr;
	struct fi_rx_attr rx_attr;
	struct fi_tx_attr tx_attr;
	struct fi_av_attr av_attr;
	int ret = 0;

	buffer_size = test_size[TEST_CNT - 1].size;
	buf = malloc(buffer_size);

	remote_fi_addr = (fi_addr_t *)malloc(sizeof(*remote_fi_addr) * ep_cnt);

	if (!buf || !remote_fi_addr) {
		perror("malloc");
		goto err1;
	}

	memset(&cq_attr, 0, sizeof cq_attr);
	cq_attr.format = FI_CQ_FORMAT_CONTEXT;
	cq_attr.wait_obj = FI_WAIT_NONE;
	cq_attr.size = rx_depth;
	
	memset(&tx_attr, 0, sizeof tx_attr);
	memset(&rx_attr, 0, sizeof rx_attr);
	
	ret = fi_stx_context(dom, &tx_attr, &stx_ctx, NULL);
	if (ret) {
		FI_PRINTERR("fi_stx_context", ret);
		goto err1;
	}

	ret = fi_cq_open(dom, &cq_attr, &scq, NULL);
	if (ret) {
		FI_PRINTERR("fi_cq_open: scq", ret);
		goto err2;
	}
	
	ret = fi_srx_context(dom, &rx_attr, &srx_ctx, NULL);
	if (ret) {
		FI_PRINTERR("fi_srx_context", ret);
		goto err3;
	}

	ret = fi_cq_open(dom, &cq_attr, &rcq, NULL);
	if (ret) {
		FI_PRINTERR("fi_cq_open: rcq", ret);
		goto err4;
	}

	ret = fi_mr_reg(dom, buf, buffer_size, 0, 0, 0, 0, &mr, NULL);
	if (ret) {
		FI_PRINTERR("fi_mr_reg", ret);
		goto err5;
	}

	memset(&av_attr, 0, sizeof av_attr);
	av_attr.type = FI_AV_MAP;
	av_attr.count = ep_cnt;

	ret = fi_av_open(dom, &av_attr, &av, NULL);
	if (ret) {
		FI_PRINTERR("fi_av_open", ret);
		goto err6;
	}

	return 0;

err6:
	fi_close(&mr->fid);
err5:
	fi_close(&rcq->fid);
err4:
	fi_close(&srx_ctx->fid);
err3:
	fi_close(&scq->fid);
err2:
	fi_close(&stx_ctx->fid);
err1:
	free(buf);
	free(remote_fi_addr);
	return ret;
}

static int bind_ep_res(void)
{
	int i, ret;

	
	for (i = 0; i < ep_cnt; i++) {
		ret = fi_ep_bind(ep[i], &stx_ctx->fid, 0);
		if (ret) {
			FI_PRINTERR("fi_ep_bind: stx", ret);
			return ret;
		}

		ret = fi_ep_bind(ep[i], &srx_ctx->fid, 0);
		if (ret) {
			FI_PRINTERR("fi_ep_bind: srx", ret);
			return ret;
		}

		ret = fi_ep_bind(ep[i], &scq->fid, FI_SEND);
		if (ret) {
			FI_PRINTERR("fi_ep_bind: scq", ret);
			return ret;
		}

		ret = fi_ep_bind(ep[i], &rcq->fid, FI_RECV);
		if (ret) {
			FI_PRINTERR("fi_ep_bind: rcq", ret);
			return ret;
		}

		ret = fi_ep_bind(ep[i], &av->fid, 0);
		if (ret) {
			FI_PRINTERR("fi_ep_bind: av", ret);
			return ret;
		}

		ret = fi_enable(ep[i]);
		if (ret) {
			FI_PRINTERR("fi_enable: ep", ret);
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
			FI_PRINTERR("fi_recv", ret);
			return ret;
		}
	}

	if (dst_addr) {
		/* Post sends addressed to remote EPs */
		for (i = 0; i < ep_cnt; i++) {
			fprintf(stdout, "Posting send to remote ctx: %d\n", i);
			ret = fi_send(ep[i], buf, transfer_size, fi_mr_desc(mr),
					remote_fi_addr[i], NULL); 
			if (ret) {
				FI_PRINTERR("fi_send", ret);
				return ret;
			}

			wait_for_completion(scq, 1);
		}
	}

	/* Wait for recv completions */
	for (i = 0; i < ep_cnt; i++) {
		wait_for_completion(rcq, 1);
	}

	if (!dst_addr) {
		/* Post sends addressed to remote EPs */
		for (i = 0; i < ep_cnt; i++) {
			fprintf(stdout, "Posting send to remote ctx: %d\n", i);
			ret = fi_send(ep[i], buf, transfer_size, fi_mr_desc(mr),
					remote_fi_addr[i], NULL); 
			if (ret) {
				FI_PRINTERR("fi_send", ret);
				return ret;
			}

			wait_for_completion(scq, 1);
		}
	}

	return 0;
}

static int init_fabric(void)
{
	struct fi_info *fi;
	char *node;
	uint64_t flags = 0;
	int i, ret;

	ret = ft_getsrcaddr(src_addr, NULL, &hints);
	if (ret)
		return ret;

	if (dst_addr) {
		node = dst_addr;
	} else {
		node = src_addr;
		flags = FI_SOURCE;
	}

	ret = fi_getinfo(FI_VERSION(1, 0), node, port, flags, &hints, &fi);
	if (ret) {
		FI_PRINTERR("fi_getinfo", ret);
		return ret;
	}

	/* Check the number of EPs supported by the provider */
	if (ep_cnt > fi->domain_attr->ep_cnt) {
		ep_cnt = fi->domain_attr->ep_cnt;
		FI_DEBUG("Provider can support only %d of EPs\n", ep_cnt);
	}

	/* Get remote address */
	if (dst_addr) {
		addrlen = fi->dest_addrlen;
		remote_addr = malloc(addrlen * ep_cnt);
		memcpy(remote_addr, fi->dest_addr, addrlen);
	}

	ret = fi_fabric(fi->fabric_attr, &fab, NULL);
	if (ret) {
		FI_PRINTERR("fi_fabric", ret);
		goto err0;
	}

	ret = fi_domain(fab, fi, &dom, NULL);
	if (ret) {
		FI_PRINTERR("fi_domain", ret);
		goto err1;
	}

	fi->ep_attr->tx_ctx_cnt = FI_SHARED_CONTEXT;
	fi->ep_attr->rx_ctx_cnt = FI_SHARED_CONTEXT;

	ep = (struct fid_ep **)malloc(sizeof(*ep) * ep_cnt);
	for (i = 0; i < ep_cnt; i++) {
		if (!ep) {
			perror("malloc");
			goto err2;
		}

		ret = fi_endpoint(dom, fi, &ep[i], NULL);
		if (ret) {
			FI_PRINTERR("fi_endpoint", ret);
			goto err2;
		}
	}

	ret = alloc_ep_res();
	if (ret)
		goto err3;

	ret = bind_ep_res();
	if (ret)
		goto err4;

	return 0;

err4:
	free_ep_res();
err3:
	FI_CLOSEV(ep, ep_cnt);
err2:
	fi_close(&dom->fid);
err1:
	fi_close(&fab->fid);
err0:
	fi_freeinfo(fi);

	return ret;
}

static int init_av(void)
{
	int ret;
	int i;

	/* Get local address blob. Find the addrlen first. We set addrlen 
	 * as 0 and fi_getname will return the actual addrlen. */
	addrlen = 0;
	ret = fi_getname(&ep[0]->fid, local_addr, &addrlen);
	if (ret != -FI_ETOOSMALL) {
		FI_PRINTERR("fi_getname", ret);
		return ret;
	}

	local_addr = malloc(addrlen * ep_cnt);

	/* Get local addresses for all EPs */
	for (i = 0; i < ep_cnt; i++) {
		ret = fi_getname(&ep[i]->fid, local_addr + addrlen * i, &addrlen);
		if (ret) {
			FI_PRINTERR("fi_getname", ret);
			return ret;
		}
	}

	if (dst_addr) {
		ret = fi_av_insert(av, remote_addr, 1, &remote_fi_addr[0], 0, &fi_ctx_av);
		if (ret != 1) {
			FI_PRINTERR("fi_av_insert", ret);
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
			FI_PRINTERR("fi_av_insert", ret);
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
			FI_PRINTERR("fi_av_insert", ret);
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

out:
	free_ep_res();
	FI_CLOSEV(ep, ep_cnt);
	fi_close(&dom->fid);
	fi_close(&fab->fid);
	return ret;
}

void print_usage(char *name)
{
	fprintf(stderr, "Usage:\n");
	fprintf(stderr, "  %s [OPTIONS]\t\tstart server\n", name);
	fprintf(stderr, "  %s [OPTIONS] <host>\tconnect to given host \t\n", name);

	fprintf(stderr, "\nOptions:\n");
	fprintf(stderr, "  -n\tdomain_name\n");
	fprintf(stderr, "  -p\tport number\n");
	fprintf(stderr, "  -s\tsource address\n");

	return;
}

int main(int argc, char **argv)
{
	int op;

	while ((op = getopt(argc, argv, "n:p:s:h")) != -1) {
		switch (op) {
		case 'n':
			domain_hints.name = optarg;
			break;
		case 'p':
			port = optarg;
			break;
		case 's':
			src_addr = optarg;
			break;
		case 'h':
		default:
			print_usage(argv[0]);
			exit(1);
		}
	}

	if (optind < argc)
		dst_addr = argv[optind];

	hints.domain_attr = &domain_hints;
	hints.ep_attr = &ep_hints;
	hints.ep_type = FI_EP_RDM;
	hints.caps = FI_MSG | FI_NAMED_RX_CTX;
	hints.mode = FI_CONTEXT | FI_LOCAL_MR | FI_PROV_MR_ATTR;
	hints.addr_format = FI_SOCKADDR;

	return run();
}
