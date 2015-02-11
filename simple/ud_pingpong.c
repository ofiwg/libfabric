/*
 * Copyright (c) 2013-2014 Intel Corporation.  All rights reserved.
 * Copyright (c) 2014 Cisco Systems, Inc.  All rights reserved.
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
#include <arpa/inet.h>

#include <rdma/fabric.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>

#include "shared.h"

static struct cs_opts opts;
static int max_credits = 128;
static int credits = 128;
static char test_name[10] = "custom";
static struct timespec start, end;
static void *buf;
static void *buf_ptr;
static size_t buffer_size;
static size_t prefix_len;
static size_t max_msg_size = 0;

static struct fi_info hints;
static fi_addr_t rem_addr;

static struct fi_info *fi;
static struct fid_fabric *fab;
static struct fid_domain *dom;
static struct fid_ep *ep;
static struct fid_cq *rcq, *scq;
static struct fid_mr *mr;
static struct fid_av *av;

static int poll_all_sends(void)
{
	struct fi_cq_entry comp;
	int ret;

	do {
		ret = fi_cq_read(scq, &comp, 1);
		if (ret > 0) {
			credits++;
		} else if (ret < 0) {
			FT_PRINTERR("fi_cq_read", ret);
			return ret;
		}
	} while (ret);
	return 0;
}

static int send_xfer(int size)
{
	struct fi_cq_entry comp;
	int ret;

	while (!credits) {
		ret = fi_cq_read(scq, &comp, 1);
		if (ret > 0) {
			goto post;
		} else if (ret < 0) {
			FT_PRINTERR("fi_cq_read", ret);
			return ret;
		}
	}

	credits--;
post:
	ret = fi_send(ep, buf_ptr, (size_t) size, fi_mr_desc(mr),
			rem_addr, NULL);
	if (ret)
		FT_PRINTERR("fi_send", ret);

	return ret;
}

static int recv_xfer(int size)
{
	struct fi_cq_entry comp;
	int ret;

	do {
		ret = fi_cq_read(rcq, &comp, sizeof comp);
		if (ret < 0) {
			FT_PRINTERR("fi_cq_read", ret);
			return ret;
		}
	} while (!ret);

	ret = fi_recv(ep, buf, buffer_size, fi_mr_desc(mr), 0, buf);
	if (ret)
		FT_PRINTERR("fi_recv", ret);

	return ret;
}

static int sync_test(void)
{
	int ret;

	while (credits < max_credits)
		poll_all_sends();

	ret = opts.dst_addr ? send_xfer(16) : recv_xfer(16);
	if (ret)
		return ret;

	return opts.dst_addr ? recv_xfer(16) : send_xfer(16);
}

static int run_test(void)
{
	int ret, i;

	ret = sync_test();
	if (ret)
		return ret;

	clock_gettime(CLOCK_MONOTONIC, &start);
	for (i = 0; i < opts.iterations; i++) {
		ret = opts.dst_addr ? send_xfer(opts.transfer_size) :
				 recv_xfer(opts.transfer_size);
		if (ret)
			return ret;

		ret = opts.dst_addr ? recv_xfer(opts.transfer_size) :
				 send_xfer(opts.transfer_size);
		if (ret)
			return ret;
	}
	clock_gettime(CLOCK_MONOTONIC, &end);

	if (opts.machr)
		show_perf_mr(opts.transfer_size, opts.iterations, &start, &end, 2, opts.argc, opts.argv);
	else
		show_perf(test_name, opts.transfer_size, opts.iterations, &start, &end, 2);

	return 0;
}

static void free_ep_res(void)
{
	int ret;
	
	ret = fi_close(&mr->fid);
	if (ret != 0) {
		FT_PRINTERR("fi_close", ret);
	}
	ret = fi_close(&rcq->fid);
	if (ret != 0) {
		FT_PRINTERR("fi_close", ret);
	}
	ret = fi_close(&scq->fid);
	if (ret != 0) {
		FT_PRINTERR("fi_close", ret);
	}
	free(buf);
}

static int alloc_ep_res(struct fi_info *fi)
{
	struct fi_cq_attr cq_attr;
	struct fi_av_attr av_attr;
	int ret;

	buffer_size = !opts.custom ? test_size[TEST_CNT - 1].size : opts.transfer_size;
	if (max_msg_size > 0 && buffer_size > max_msg_size) {
		buffer_size = max_msg_size;
	}
	if (buffer_size < fi->src_addrlen) {
		buffer_size = fi->src_addrlen;
	}
	buffer_size += prefix_len;
	buf = malloc(buffer_size);
	if (!buf) {
		perror("malloc");
		return -1;
	}
	buf_ptr = (char *)buf + prefix_len;

	memset(&cq_attr, 0, sizeof cq_attr);
	cq_attr.format = FI_CQ_FORMAT_CONTEXT;
	cq_attr.wait_obj = FI_WAIT_NONE;
	cq_attr.size = max_credits << 1;
	ret = fi_cq_open(dom, &cq_attr, &scq, NULL);
	if (ret) {
		FT_PRINTERR("fi_cq_open", ret);
		goto err1;
	}

	ret = fi_cq_open(dom, &cq_attr, &rcq, NULL);
	if (ret) {
		FT_PRINTERR("fi_cq_open", ret);
		goto err2;
	}

	ret = fi_mr_reg(dom, buf, buffer_size, 0, 0, 0, 0, &mr, NULL);
	if (ret) {
		FT_PRINTERR("fi_mr_reg", ret);
		goto err3;
	}

	memset(&av_attr, 0, sizeof(av_attr));
	av_attr.type = FI_AV_MAP;
	av_attr.name = NULL;
	av_attr.flags = 0;
	ret = fi_av_open(dom, &av_attr, &av, NULL);
	if (ret) {
		FT_PRINTERR("fi_av_open", ret);
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

	ret = fi_recv(ep, buf, buffer_size, fi_mr_desc(mr), 0, buf);
	if (ret) {
		FT_PRINTERR("fi_recv", ret);
	}

	return ret;
}


static int common_setup(void)
{
	int ret;
	uint64_t flags = 0;
	char *node, *service;

	if (opts.dst_addr) {
		ret = ft_getsrcaddr(opts.src_addr, opts.src_port, &hints);
		if (ret)
			return ret;
		node = opts.dst_addr;
		service = opts.dst_port;
	} else {
		node = opts.src_addr;
		service = opts.src_port;
		flags = FI_SOURCE;
	}

	ret = fi_getinfo(FT_FIVERSION, node, service, flags, &hints, &fi);
	if (ret) {
		FT_PRINTERR("fi_getinfo", ret);
		goto err0;
	}
	if (fi->ep_attr->max_msg_size) {
		max_msg_size = fi->ep_attr->max_msg_size;
	}

	ret = fi_fabric(fi->fabric_attr, &fab, NULL);
	if (ret) {
		FT_PRINTERR("fi_fabric", ret);
		goto err1;
	}
	if (fi->mode & FI_MSG_PREFIX) {
		prefix_len = fi->ep_attr->msg_prefix_size;
	}

	ret = fi_domain(fab, fi, &dom, NULL);
	if (ret) {
		FT_PRINTERR("fi_domain", ret);
		goto err2;
	}

	ret = fi_endpoint(dom, fi, &ep, NULL);
	if (ret) {
		FT_PRINTERR("fi_endpoint", ret);
		goto err3;
	}

	ret = alloc_ep_res(fi);
	if (ret) {
		goto err4;
	}

	ret = bind_ep_res();
	if (ret) {
		goto err5;
	}

	if (hints.src_addr)
		free(hints.src_addr);
	return 0;

err5:
	free_ep_res();
err4:
	fi_close(&ep->fid);
err3:
	fi_close(&dom->fid);
err2:
	fi_close(&fab->fid);
err1:
	fi_freeinfo(fi);
err0:
	if (hints.src_addr)
		free(hints.src_addr);
	return ret;
}

static int client_connect(void)
{
	int ret;

	ret = common_setup();
	if (ret != 0)
		goto err;

	ret = ft_getdestaddr(opts.dst_addr, opts.dst_port, &hints);
	if (ret != 0)
		goto err;

	ret = fi_av_insert(av, hints.dest_addr, 1, &rem_addr, 0, NULL);
	if (ret != 1) {
		FT_PRINTERR("fi_av_insert", ret);
		goto err;
	}

	// send initial message to server with our local address
	memcpy(buf_ptr, fi->src_addr, fi->src_addrlen);
	ret = send_xfer(fi->src_addrlen);
	if (ret != 0)
		goto err;

	// wait for reply to know server is ready
	ret = recv_xfer(4);
	if (ret != 0)
		goto err;

	return 0;

err:
	free_ep_res();
	fi_close(&av->fid);
	fi_close(&ep->fid);
	fi_close(&dom->fid);
	fi_close(&fab->fid);
	return ret;
}

static int server_connect(void)
{
	int ret;
	struct fi_cq_entry comp;

	ret = common_setup();
	if (ret != 0)
		goto err;

	do {
		ret = fi_cq_read(rcq, &comp, 1);
		if (ret < 0) {
			FT_PRINTERR("fi_cq_read", ret);
			return ret;
		}
	} while (ret == 0);

	ret = fi_av_insert(av, buf_ptr, 1, &rem_addr, 0, NULL);
	if (ret != 1) {
		if (ret == 0) {
			FT_DEBUG("Unable to resolve remote address 0x%x 0x%x\n",
					((uint32_t *)buf)[0], ((uint32_t *)buf)[1]);
			ret = -FI_EINVAL;
		} else {
			FT_PRINTERR("fi_av_insert", ret);
		}
		goto err;
	}

	ret = fi_recv(ep, buf, buffer_size, fi_mr_desc(mr), 0, buf);
	if (ret != 0) {
		FT_PRINTERR("fi_recv", ret);
		goto err;
	}

	ret = send_xfer(4);
	if (ret != 0)
		goto err;

	return 0;

err:
	free_ep_res();
	fi_close(&ep->fid);
	fi_close(&dom->fid);
	fi_close(&fab->fid);
	return ret;
}

static int run(void)
{
	int i, ret = 0;

	ret = opts.dst_addr ? client_connect() : server_connect();
	if (ret)
		return ret;

	if (!opts.custom) {
		for (i = 0; i < TEST_CNT; i++) {
			if (test_size[i].option > opts.size_option ||
				(max_msg_size && test_size[i].size > max_msg_size)) {
				continue;
			}
			init_test(test_size[i].size, test_name,
					sizeof(test_name), &opts.transfer_size,
					&opts.iterations);
			run_test();
		}
	} else {

		ret = run_test();
	}

	while (credits < max_credits)
		poll_all_sends();

	ret = fi_close(&ep->fid);
	if (ret != 0) {
		FT_PRINTERR("fi_close", ret);
	}
	free_ep_res();
	ret = fi_close(&av->fid);
	if (ret != 0) {
		FT_PRINTERR("fi_close", ret);
	}
	ret = fi_close(&dom->fid);
	if (ret != 0) {
		FT_PRINTERR("fi_close", ret);
	}
	ret = fi_close(&fab->fid);
	if (ret != 0) {
		FT_PRINTERR("fi_close", ret);
	}
	fi_freeinfo(fi);
	return ret;
}

int main(int argc, char **argv)
{
	int op;
	opts = INIT_OPTS;

	while ((op = getopt(argc, argv, "h" CS_OPTS INFO_OPTS)) != -1) {
		switch (op) {
		default:
			ft_parseinfo(op, optarg, &hints);
			ft_parsecsopts(op, optarg, &opts);
			break;
		case '?':
		case 'h':
			ft_csusage(argv[0], "Ping pong client and server using UD.");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints.ep_type = FI_EP_DGRAM;
	hints.caps = FI_MSG;
	hints.mode = FI_LOCAL_MR | FI_MSG_PREFIX;
	hints.addr_format = FI_SOCKADDR;

	if (opts.prhints) {
		printf("%s", fi_tostr(&hints, FI_TYPE_INFO));
		return EXIT_SUCCESS;
	}

	return run();
}
