/*
 * Copyright (c) 2013-2014 Intel Corporation.  All rights reserved.
 *
 * This software is available to you under the OpenIB.org BSD license
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
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_errno.h>
#include <shared.h>

static uint64_t op_type = FI_REMOTE_WRITE;
static int custom;
static int size_option;
static int iterations = 1000;
static int transfer_size = 1024;
static int max_credits = 128;
static char test_name[10] = "custom";
static struct timespec start, end;
static void *buf;
static size_t buffer_size;
struct fi_rma_iov local, remote;
static int machr, g_argc;
static char **g_argv;

static struct fi_info hints;
static struct fi_domain_attr domain_hints;
static struct fi_ep_attr ep_hints;
static char *dst_addr, *src_addr;
static char *port = "9228";

static struct fid_fabric *fab;
static struct fid_domain *dom;
static struct fid_ep *ep;
static struct fid_cq *rcq, *scq;
static struct fid_av *av;
static struct fid_mr *mr;
static void *local_addr, *remote_addr;
static size_t addrlen = 0;
static fi_addr_t remote_fi_addr;
struct fi_context fi_ctx_send;
struct fi_context fi_ctx_recv;
struct fi_context fi_ctx_write;
struct fi_context fi_ctx_read;
struct fi_context fi_ctx_av;

void usage(char *name)
{
	fprintf(stderr, "usage: %s\n", name);
	fprintf(stderr, "\t[-d destination_address]\n");
	fprintf(stderr, "\t[-n domain_name]\n");
	fprintf(stderr, "\t[-p port_number]\n");
	fprintf(stderr, "\t[-s source_address]\n");
	fprintf(stderr, "\t[-I iterations]\n");
	fprintf(stderr, "\t[-o read|write] (default: write)\n");
	fprintf(stderr, "\t[-S transfer_size or 'all']\n");
	fprintf(stderr, "\t[-m machine readable output]\n");
	exit(1);
}

static int send_msg(int size)
{
	int ret;

	ret = fi_send(ep, buf, (size_t) size, fi_mr_desc(mr), remote_fi_addr,
			&fi_ctx_send);
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

	ret = fi_recv(ep, buf, buffer_size, fi_mr_desc(mr), 0, &fi_ctx_recv);
	if (ret) {
		FI_PRINTERR("fi_recv", ret);
		return ret;
	}

	ret = wait_for_completion(rcq, 1);

	return ret;
}

static int read_data(size_t size)
{
	int ret;

	ret = fi_read(ep, buf, size, fi_mr_desc(mr), remote_fi_addr, 
		      remote.addr, remote.key, &fi_ctx_read);
	if (ret){
		FI_PRINTERR("fi_readfrom", ret);
		return ret;
	}

	return 0;
}

static int write_data(size_t size)
{
	int ret;

	ret = fi_write(ep, buf, size, fi_mr_desc(mr), remote_fi_addr, 
		       remote.addr, remote.key, &fi_ctx_write);
	if (ret){
		FI_PRINTERR("fi_writeto", ret);
		return ret;
	}
	return 0;
}

static int sync_test(void)
{
	int ret;

	ret = dst_addr ? send_msg(16) : recv_msg();
	if (ret)
		return ret;

	return dst_addr ? recv_msg() : send_msg(16);
}

static int run_test(void)
{
	int ret, i;

	ret = sync_test();
	if (ret)
		return ret;

	clock_gettime(CLOCK_MONOTONIC, &start);
	for (i = 0; i < iterations; i++) {
		if (op_type == FI_REMOTE_WRITE) {
			ret = write_data(transfer_size);
		} else {
			ret = read_data(transfer_size); 
		}
		if (ret)
			return ret;
		ret = wait_for_completion(scq, 1);
		if (ret)
			return ret;
	}
	clock_gettime(CLOCK_MONOTONIC, &end);

	if (machr)
		show_perf_mr(transfer_size, iterations, &start, &end, 1, g_argc,
			       	g_argv);
	else
		show_perf(test_name, transfer_size, iterations, &start, &end, 1);

	return 0;
}

static void free_ep_res(void)
{
	fi_close(&av->fid);
	fi_close(&mr->fid);
	fi_close(&rcq->fid);
	fi_close(&scq->fid);
	free(buf);
}

static int alloc_ep_res(struct fi_info *fi)
{
	struct fi_cq_attr cq_attr;
	struct fi_av_attr av_attr;
	int ret;

	buffer_size = !custom ? test_size[TEST_CNT - 1].size : transfer_size;
	buf = malloc(MAX(buffer_size, sizeof(uint64_t)));
	if (!buf) {
		perror("malloc");
		return -1;
	}

	memset(&cq_attr, 0, sizeof cq_attr);
	cq_attr.format = FI_CQ_FORMAT_CONTEXT;
	cq_attr.wait_obj = FI_WAIT_NONE;
	cq_attr.size = max_credits << 1;
	ret = fi_cq_open(dom, &cq_attr, &scq, NULL);
	if (ret) {
		FI_PRINTERR("fi_cq_open: scq", ret);
		goto err1;
	}

	ret = fi_cq_open(dom, &cq_attr, &rcq, NULL);
	if (ret) {
		FI_PRINTERR("fi_cq_open: rcq", ret);
		goto err2;
	}
	
	ret = fi_mr_reg(dom, buf, MAX(buffer_size, sizeof(uint64_t)), 
			op_type, 0, 0, 0, &mr, NULL);
	if (ret) {
		FI_PRINTERR("fi_mr_reg", ret);
		goto err3;
	}

	memset(&av_attr, 0, sizeof av_attr);
	av_attr.type = FI_AV_MAP;
	av_attr.count = 1;
	av_attr.name = NULL;

	ret = fi_av_open(dom, &av_attr, &av, NULL);
	if (ret) {
		FI_PRINTERR("fi_av_open", ret);
		goto err4;
	}

	return 0;

err4:
	fi_close(&mr->fid);
err3:
	fi_close(&rcq->fid);
err2:
	fi_close(&scq->fid);
err1:
	free(buf);
	return ret;
}

static int bind_ep_res(void)
{
	int ret;

	ret = fi_bind(&ep->fid, &scq->fid, FI_SEND);
	if (ret) {
		FI_PRINTERR("fi_bind: scq", ret);
		return ret;
	}

	ret = fi_bind(&ep->fid, &rcq->fid, FI_RECV);
	if (ret) {
		FI_PRINTERR("fi_bind: rcq", ret);
		return ret;
	}

	ret = fi_bind(&ep->fid, &av->fid, 0);
	if (ret) {
		FI_PRINTERR("fi_bind: av", ret);
		return ret;
	}

	ret = fi_enable(ep);
	if (ret) {
		FI_PRINTERR("fi_enable", ret);
		return ret;
	}

	return ret;
}

static int init_fabric(void)
{
	struct fi_info *fi;
	char *node;
	uint64_t flags = 0;
	int ret;

	if (src_addr) {
		ret = getaddr(src_addr, NULL, 
				(struct sockaddr **) &hints.src_addr, 
				(socklen_t *) &hints.src_addrlen);
		if (ret) {
			fprintf(stderr, "source address error %s\n", 
					gai_strerror(ret));
			return ret;
		}
	}

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

	if (!(fi->mode & FI_PROV_MR_ATTR))
		fi->mode |= FI_PROV_MR_ATTR;

	/* Get remote address */
	if (dst_addr) {
		addrlen = fi->dest_addrlen;
		remote_addr = malloc(addrlen);
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

	ret = fi_endpoint(dom, fi, &ep, NULL);
	if (ret) {
		FI_PRINTERR("fi_endpoint", ret);
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
			FI_PRINTERR("fi_getname", ret);
			return ret;
		}

		local_addr = malloc(addrlen);
		ret = fi_getname(&ep->fid, local_addr, &addrlen);
		if (ret) {
			FI_PRINTERR("fi_getname", ret);
			return ret;
		}

		ret = fi_av_insert(av, remote_addr, 1, &remote_fi_addr, 0, 
				&fi_ctx_av);
		if (ret != 1) {
			FI_PRINTERR("fi_av_insert", ret);
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
			FI_PRINTERR("fi_av_insert", ret);
			return ret;
		}

		/* Send ACK */
		ret = send_msg(16);
		if (ret)
			return ret;
	}

	return ret;
}

static int exchange_addr_key(void)
{
	local.addr = (uint64_t)buf;
	local.key = fi_mr_key(mr);

	if (dst_addr) {
		*(struct fi_rma_iov *)buf = local;
		send_msg(sizeof local);
		recv_msg();
		remote = *(struct fi_rma_iov *)buf;
	} else {
		recv_msg();
		remote = *(struct fi_rma_iov *)buf;
		*(struct fi_rma_iov *)buf = local;
		send_msg(sizeof local);
	}

	return 0;
}

static int run(void)
{
	int i, ret = 0;

	ret = init_fabric();
	if (ret)
		return ret;

	ret = init_av();
	if (ret)
		goto out;

	ret = exchange_addr_key();
	if (ret)
		goto out;

	if (!custom) {
		for (i = 0; i < TEST_CNT; i++) {
			if (test_size[i].option > size_option)
				continue;
			init_test(test_size[i].size, test_name,
					sizeof(test_name), &transfer_size,
					&iterations);
			ret = run_test();
			if(ret)
				goto out;
		}
	} else {
		ret = run_test();
	}

	sync_test();

out:
	fi_close(&ep->fid);
	free_ep_res();
	fi_close(&dom->fid);
	fi_close(&fab->fid);
	return ret;
}

int main(int argc, char **argv)
{
	int op;

	while ((op = getopt(argc, argv, "d:n:p:s:I:o:S:m")) != -1) {
		switch (op) {
		case 'd':
			dst_addr = optarg;
			break;
		case 'n':
			domain_hints.name = optarg;
			break;
		case 'p':
			port = optarg;
			break;
		case 's':
			src_addr = optarg;
			break;
		case 'I':
			custom = 1;
			iterations = atoi(optarg);
			break;
		case 'o':
			if (!strcmp(optarg, "read"))
				op_type = FI_REMOTE_READ;
			else if (!strcmp(optarg, "write"))
				op_type = FI_REMOTE_WRITE;
			else
				usage(argv[0]);
			break;
		case 'S':
			if (!strncasecmp("all", optarg, 3)) {
				size_option = 1;
			} else {
				custom = 1;
				transfer_size = atoi(optarg);
			}
			break;
		case 'm':
			machr = 1;
			g_argc = argc;
			g_argv = argv;
			break;
		default:
			usage(argv[0]);
		}
	}

	hints.domain_attr = &domain_hints;
	hints.ep_attr = &ep_hints;
	hints.ep_type = FI_EP_RDM;
	hints.caps = FI_MSG | FI_RMA;
	hints.mode = FI_CONTEXT | FI_PROV_MR_ATTR;
	hints.addr_format = FI_FORMAT_UNSPEC;

	return run();
}
