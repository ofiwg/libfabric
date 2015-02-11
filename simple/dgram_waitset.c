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

#define WAIT_TIMEOUT 10000 // 10ms

static void *buf;
static size_t buffer_size = 1024;
static int transfer_size = 1000;
static int rx_depth = 512;

static struct fi_info hints;
static char *dst_addr, *src_addr;
static char *dst_port = "5300", *src_port = "5300";

static struct fid_fabric *fab;
static struct fid_domain *dom;
static struct fid_ep *ep;
static struct fid_cq *rcq, *scq;
static struct fid_av *av;
static struct fid_mr *mr;
static struct fid_wait *waitset;
static void *local_addr, *remote_addr;
static size_t addrlen = 0;
static fi_addr_t remote_fi_addr;
struct fi_context fi_ctx_send;
struct fi_context fi_ctx_recv;
struct fi_context fi_ctx_av;


void print_usage(char *name, char *desc)
{
	fprintf(stderr, "Usage:\n");
	fprintf(stderr, "  %s [OPTIONS]\t\tstart server\n", name);
	fprintf(stderr, "  %s [OPTIONS] <host>\tconnect to server \t\n", name);
	
	if (desc)
		fprintf(stderr, "\n%s\n", desc);

	fprintf(stderr, "\nOptions:\n");
	fprintf(stderr, "  -n <domain>\tdomain name\n");
	fprintf(stderr, "  -b <src_port>\tnon default source port number\n");
	fprintf(stderr, "  -p <dst_port>\tnon default destination port number\n");
	fprintf(stderr, "  -f <provider>\tspecific provider name eg IP, verbs\n");
	fprintf(stderr, "  -s <address>\tsource address\n");
	fprintf(stderr, "  -h\t\tdisplay this help output\n");
	
	return;
}

static void free_ep_res(void)
{
	fi_close(&av->fid);
	fi_close(&mr->fid);
	fi_close(&rcq->fid);
	fi_close(&scq->fid);
	fi_close(&waitset->fid);
	free(buf);
}

static int alloc_ep_res(struct fi_info *fi)
{
	struct fi_cq_attr cq_attr;
	struct fi_av_attr av_attr;
	struct fi_wait_attr wait_attr;
	int ret;

	buf = malloc(buffer_size);
	if (!buf) {
		perror("malloc");
		return -1;
	}

	/* Open a wait set */
	memset(&wait_attr, 0, sizeof wait_attr);
	wait_attr.wait_obj = FI_WAIT_UNSPEC;
	ret = fi_wait_open(fab, &wait_attr, &waitset);
	if (ret) {
		FT_PRINTERR("fi_wait_open", ret);
		goto err1;
	}

	memset(&cq_attr, 0, sizeof cq_attr);
	cq_attr.format = FI_CQ_FORMAT_CONTEXT;
	cq_attr.wait_obj = FI_WAIT_SET;
	cq_attr.wait_cond = FI_CQ_COND_NONE;
	cq_attr.wait_set = waitset;
	cq_attr.size = rx_depth;

	/* Open completion queue for send completions */
	ret = fi_cq_open(dom, &cq_attr, &scq, NULL);
	if (ret) {
		FT_PRINTERR("fi_cq_open", ret);
		goto err2;
	}

	/* Open completion queue for recv completions */
	ret = fi_cq_open(dom, &cq_attr, &rcq, NULL);
	if (ret) {
		FT_PRINTERR("fi_cq_open", ret);
		goto err3;
	}
	
	/* Register memory */
	ret = fi_mr_reg(dom, buf, buffer_size, 0, 0, 0, 0, &mr, NULL);
	if (ret) {
		FT_PRINTERR("fi_mr_reg", ret);
		goto err4;
	}

	memset(&av_attr, 0, sizeof av_attr);
	av_attr.type = FI_AV_MAP;
	av_attr.count = 1;
	av_attr.name = NULL;

	/* Open Address Vector */
	ret = fi_av_open(dom, &av_attr, &av, NULL);
	if (ret) {
		FT_PRINTERR("fi_av_open", ret);
		goto err5;
	}

	return 0;

err5:
	fi_close(&mr->fid);
err4:
	fi_close(&rcq->fid);
err3:
	fi_close(&scq->fid);
err2:
	fi_close(&waitset->fid);
err1:
	free(buf);
	return ret;
}

static int bind_ep_res(void)
{
	int ret;

	/* Bind AV and CQs with endpoint */
	ret = fi_ep_bind(ep, &scq->fid, FI_SEND);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		return ret;
	}

	ret = fi_ep_bind(ep, &rcq->fid, FI_RECV);
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

	ret = wait_for_completion(scq, 1);

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

	ret = wait_for_completion(rcq, 1);

	return ret;
}

static int init_fabric(void)
{
	struct fi_info *fi;
	uint64_t flags = 0;
	char *node, *service;
	int ret;

	if (dst_addr) {
		ret = ft_getsrcaddr(src_addr, src_port, &hints);
		if (ret)
			return ret;
		node = dst_addr;
		service = dst_port;
	} else {
		node = src_addr;
		service = src_port;
		flags = FI_SOURCE;
	}

	ret = fi_getinfo(FT_FIVERSION, node, service, flags, &hints, &fi);
	if (ret) {
		FT_PRINTERR("fi_getinfo", ret);
		return ret;
	}

	/* Get remote address */
	if (dst_addr) {
		addrlen = fi->dest_addrlen;
		remote_addr = malloc(addrlen);
		memcpy(remote_addr, fi->dest_addr, addrlen);
	}

	ret = fi_fabric(fi->fabric_attr, &fab, NULL);
	if (ret) {
		FT_PRINTERR("fi_fabric", ret);
		goto err0;
	}

	ret = fi_domain(fab, fi, &dom, NULL);
	if (ret) {
		FT_PRINTERR("fi_domain", ret);
		goto err1;
	}

	ret = fi_endpoint(dom, fi, &ep, NULL);
	if (ret) {
		FT_PRINTERR("fi_endpoint", ret);
		goto err2;
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
	fi_close(&ep->fid);
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

	if (dst_addr) {
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
	struct fi_cq_entry comp;
	int ret, send_pending = 0, recv_pending = 0;

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
		/* Wait for completion events on CQs */
		ret = fi_wait(waitset, WAIT_TIMEOUT);
		if (ret < 0) {
			FT_PRINTERR("fi_wait", ret);
			return ret;
		}
		
		/* Read the send completion entry */
		ret = fi_cq_read(scq, &comp, 1);
		if(ret > 0) {
			send_pending--;
			fprintf(stdout, "Received send completion event!\n");
		} else if (ret < 0) {
			if (ret == -FI_EAVAIL) {
				cq_readerr(scq, "scq");
			} else {
				FT_PRINTERR("fi_cq_read", ret);
			}
			
			return ret;
		}
		
		/* Read the recv completion entry */
		ret = fi_cq_read(rcq, &comp, 1);
		if(ret > 0) {
			recv_pending--;
			fprintf(stdout, "Received recv completion event!\n");
		} else if (ret < 0) {
			if (ret == -FI_EAVAIL) {
				cq_readerr(rcq, "rcq");
			} else {
				FT_PRINTERR("fi_cq_read", ret);
			}

			return ret;
		}
	}

	return 0;
}

int main(int argc, char **argv)
{
	int op, ret = 0;

	while ((op = getopt(argc, argv, "b:p:s:h" INFO_OPTS)) != -1) {
		switch (op) {
		case 'b':
			src_port = optarg;
			break;
		case 'p':
			dst_port = optarg;
			break;
		case 's':
			src_addr = optarg;
			break;
		default:
			ft_parseinfo(op, optarg, &hints);
			break;
		case '?':
		case 'h':
			print_usage(argv[0], "A DGRAM client-server example that uses waitset.\n");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		dst_addr = argv[optind];
	
	hints.ep_type = FI_EP_DGRAM;
	hints.caps = FI_MSG;
	hints.mode = FI_CONTEXT | FI_LOCAL_MR | FI_PROV_MR_ATTR;
	hints.addr_format = FI_FORMAT_UNSPEC;

	ret = init_fabric();
	if (ret)
		return ret;

	ret = init_av();
	if (ret)
		return ret;

	/* Exchange data */
	ret = send_recv();

	/* Tear down */
	fi_close(&ep->fid);
	free_ep_res();
	fi_close(&dom->fid);
	fi_close(&fab->fid);

	return ret;
}
