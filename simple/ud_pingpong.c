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
#include <shared.h>

static int custom;
static int size_option;
static int iterations = 1000;
static int transfer_size = 1000;
static int max_credits = 128;
static int credits = 128;
static char test_name[10] = "custom";
static struct timespec start, end;
static void *buf;
static void *buf_ptr;
static size_t buffer_size;
static size_t prefix_len;
static size_t max_msg_size = 0;
static int machr, g_argc;
static char **g_argv;

static struct fi_info hints;
static struct fi_domain_attr domain_hints;
static struct fi_fabric_attr fabric_hints;
static struct fi_ep_attr ep_hints;
static char *dst_addr, *src_addr;
static char *port = "3333";
static fi_addr_t client_addr;

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
			printf("scq read %d (%s)\n", ret, fi_strerror(-ret));
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
			printf("scq read %d (%s)\n", ret, fi_strerror(-ret));
			return ret;
		}
	}

	credits--;
post:
	ret = fi_send(ep, buf_ptr, (size_t) size, fi_mr_desc(mr),
			client_addr, NULL);
	if (ret)
		printf("fi_send %d (%s)\n", ret, fi_strerror(-ret));

	return ret;
}

static int recv_xfer(int size)
{
	struct fi_cq_entry comp;
	int ret;

	do {
		ret = fi_cq_read(rcq, &comp, sizeof comp);
		if (ret < 0) {
			printf("rcq read %d (%s)\n", ret, fi_strerror(-ret));
			return ret;
		}
	} while (!ret);

	ret = fi_recv(ep, buf, buffer_size, fi_mr_desc(mr), 0, buf);
	if (ret)
		printf("fi_recv %d (%s)\n", ret, fi_strerror(-ret));

	return ret;
}

static int sync_test(void)
{
	int ret;

	while (credits < max_credits)
		poll_all_sends();

	ret = dst_addr ? send_xfer(16) : recv_xfer(16);
	if (ret)
		return ret;

	return dst_addr ? recv_xfer(16) : send_xfer(16);
}

static int run_test(void)
{
	int ret, i;

	ret = sync_test();
	if (ret)
		return ret;

	clock_gettime(CLOCK_MONOTONIC, &start);
	for (i = 0; i < iterations; i++) {
		ret = dst_addr ? send_xfer(transfer_size) :
				 recv_xfer(transfer_size);
		if (ret)
			return ret;

		ret = dst_addr ? recv_xfer(transfer_size) :
				 send_xfer(transfer_size);
		if (ret)
			return ret;
	}
	clock_gettime(CLOCK_MONOTONIC, &end);

	if (machr)
		show_perf_mr(transfer_size, iterations, &start, &end, 2, g_argc, g_argv);
	else
		show_perf(test_name, transfer_size, iterations, &start, &end, 2);

	return 0;
}

static void free_ep_res(void)
{
	int ret;
	
	ret = fi_close(&mr->fid);
	if (ret != 0) {
		printf("fi_close(mr) ret=%d, %s\n", ret, fi_strerror(-ret));
	}
	ret = fi_close(&rcq->fid);
	if (ret != 0) {
		printf("fi_close(rcq) ret=%d, %s\n", ret, fi_strerror(-ret));
	}
	ret = fi_close(&scq->fid);
	if (ret != 0) {
		printf("fi_close(scq) ret=%d, %s\n", ret, fi_strerror(-ret));
	}
	free(buf);
}

static int alloc_ep_res(struct fi_info *fi)
{
	struct fi_cq_attr cq_attr;
	struct fi_av_attr av_attr;
	int ret;

	buffer_size = !custom ? test_size[TEST_CNT - 1].size : transfer_size;
	if (max_msg_size > 0 && buffer_size > max_msg_size) {
		buffer_size = max_msg_size;
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
		printf("fi_cq_open send comp %s\n", fi_strerror(-ret));
		goto err1;
	}

	ret = fi_cq_open(dom, &cq_attr, &rcq, NULL);
	if (ret) {
		printf("fi_cq_open recv comp %s\n", fi_strerror(-ret));
		goto err2;
	}

	ret = fi_mr_reg(dom, buf, buffer_size, 0, 0, 0, 0, &mr, NULL);
	if (ret) {
		printf("fi_mr_reg %s\n", fi_strerror(-ret));
		goto err3;
	}

	av_attr.type = FI_AV_MAP;
	av_attr.name = NULL;
	av_attr.flags = 0;
	ret = fi_av_open(dom, &av_attr, &av, NULL);
	if (ret) {
		printf("fi_av_open %s\n", fi_strerror(-ret));
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
		printf("fi_bind scq %d (%s)\n", ret, fi_strerror(-ret));
		return ret;
	}

	ret = fi_bind(&ep->fid, &rcq->fid, FI_RECV);
	if (ret) {
		printf("fi_bind rcq %d (%s)\n", ret, fi_strerror(-ret));
		return ret;
	}

	ret = fi_bind(&ep->fid, &av->fid, 0);
	if (ret) {
		printf("fi_bind av %d (%s)\n", ret, fi_strerror(-ret));
		return ret;
	}

	ret = fi_enable(ep);
	if (ret) {
		printf("fi_enable %d (%s)\n", ret, fi_strerror(-ret));
		return ret;
	}

	ret = fi_recv(ep, buf, buffer_size, fi_mr_desc(mr), 0, buf);
	if (ret) {
		printf("fi_recv %d (%s)\n", ret, fi_strerror(-ret));
	}

	return ret;
}


static int common_setup(void)
{
	struct fi_info *fi;
	int ret;

	ret = getaddr(src_addr, port, (struct sockaddr **) &hints.src_addr,
			  (socklen_t *) &hints.src_addrlen);
	if (ret)
		printf("source address error %s\n", gai_strerror(ret));

	ret = fi_getinfo(FI_VERSION(1, 0), dst_addr, port, 0, &hints, &fi);
	if (ret) {
		printf("fi_getinfo %s\n", strerror(-ret));
		goto err0;
	}
	if (fi->ep_attr->max_msg_size) {
		max_msg_size = fi->ep_attr->max_msg_size;
	}

	ret = fi_fabric(fi->fabric_attr, &fab, NULL);
	if (ret) {
		printf("fi_fabric %s\n", fi_strerror(-ret));
		goto err1;
	}
	if (fi->mode & FI_MSG_PREFIX) {
		prefix_len = fi->ep_attr->msg_prefix_size;
	}

	ret = fi_domain(fab, fi, &dom, NULL);
	if (ret) {
		printf("fi_domain %s %s\n", fi_strerror(-ret),
			fi->domain_attr->name);
		goto err2;
	}

	if (fi->src_addr != NULL) {
		((struct sockaddr_in *)fi->src_addr)->sin_port = 
			((struct sockaddr_in *)hints.src_addr)->sin_port;

		if (dst_addr == NULL) {
			printf("Local address %s:%d\n",
					inet_ntoa(((struct sockaddr_in *)fi->src_addr)->sin_addr),
					ntohs(((struct sockaddr_in *)fi->src_addr)->sin_port));
		}
	}

	ret = fi_endpoint(dom, fi, &ep, NULL);
	if (ret) {
		printf("fi_endpoint %s\n", fi_strerror(-ret));
		goto err3;
	}

	ret = alloc_ep_res(fi);
	if (ret) {
		printf("alloc_ep_res %s\n", fi_strerror(-ret));
		goto err4;
	}

	ret = bind_ep_res();
	if (ret) {
		printf("bind_ep_res %s\n", fi_strerror(-ret));
		goto err5;
	}

	if (hints.src_addr)
		free(hints.src_addr);
	fi_freeinfo(fi);
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
	socklen_t addrlen;
	struct sockaddr *sin;

	ret = common_setup();
	if (ret != 0)
		goto err;

	ret = getaddr(dst_addr, port, (struct sockaddr **) &sin,
			  (socklen_t *) &addrlen);
	if (ret != 0)
		goto err;

	ret = fi_connect(ep, sin, NULL, 0);
	if (ret) {
		printf("fi_connect %s\n", fi_strerror(-ret));
		goto err;
	}

	// send initial message to server
	ret = send_xfer(4);
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
		ret = fi_cq_readfrom(rcq, &comp, sizeof comp, &client_addr);
		if (ret < 0) {
			printf("fi_cq_readfrom rcq %d (%s)\n", ret, fi_strerror(-ret));
			return ret;
		}
	} while (ret == 0);

	if (client_addr == FI_ADDR_NOTAVAIL) {
		printf("Error getting address\n");
		goto err;
	}

	ret = fi_recv(ep, buf, buffer_size, fi_mr_desc(mr), 0, buf);
	if (ret != 0) {
		printf("fi_recv %d (%s)\n", ret, fi_strerror(-ret));
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

	ret = dst_addr ? client_connect() : server_connect();
	if (ret)
		return ret;

	if (!custom) {
		for (i = 0; i < TEST_CNT; i++) {
			if (test_size[i].option > size_option ||
				(max_msg_size && test_size[i].size > max_msg_size)) {
				continue;
			}
			init_test(test_size[i].size, test_name, &transfer_size, &iterations);
			run_test();
		}
	} else {

		ret = run_test();
	}

	while (credits < max_credits)
		poll_all_sends();

	ret = fi_shutdown(ep, 0);
	if (ret != 0) {
		printf("fi_shutdown ret=%d, %s\n", ret, fi_strerror(-ret));
	}
	ret = fi_close(&ep->fid);
	if (ret != 0) {
		printf("fi_close(ep) ret=%d, %s\n", ret, fi_strerror(-ret));
	}
	free_ep_res();
	ret = fi_close(&av->fid);
	if (ret != 0) {
		printf("fi_close(av) ret=%d, %s\n", ret, fi_strerror(-ret));
	}
	ret = fi_close(&dom->fid);
	if (ret != 0) {
		printf("fi_close(dom) ret=%d, %s\n", ret, fi_strerror(-ret));
	}
	ret = fi_close(&fab->fid);
	if (ret != 0) {
		printf("fi_close(fab) ret=%d, %s\n", ret, fi_strerror(-ret));
	}
	return ret;
}

int main(int argc, char **argv)
{
	int op;

	while ((op = getopt(argc, argv, "d:f:n:p:s:I:S:m")) != -1) {
		switch (op) {
		case 'd':
			dst_addr = optarg;
			break;
		case 'f':
			fabric_hints.name = optarg;
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
			printf("usage: %s\n", argv[0]);
			printf("\t[-d destination_address]\n");
			printf("\t[-f fabric_name]\n");
			printf("\t[-n domain_name]\n");
			printf("\t[-p port_number]\n");
			printf("\t[-s source_address]\n");
			printf("\t[-I iterations]\n");
			printf("\t[-S transfer_size or 'all']\n");
			printf("\t[-m machine readable output]\n");
			exit(1);
		}
	}

	hints.domain_attr = &domain_hints;
	hints.fabric_attr = &fabric_hints;
	hints.ep_attr = &ep_hints;
	hints.ep_type = FI_EP_DGRAM;
	hints.caps = FI_MSG;
	hints.mode = FI_LOCAL_MR | FI_MSG_PREFIX;
	hints.addr_format = FI_SOCKADDR;

	return run();
}
