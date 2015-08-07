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


static int max_credits = 128;
static int credits = 128;
static char test_name[10] = "custom";
static struct timespec start, end;
static void *buf;
static void *payload;
static size_t buffer_size;
static size_t prefix_len;
static size_t max_msg_size = 0;
static int timeout = 5;

static fi_addr_t remote_fi_addr;

static int poll_all_sends(void)
{
	struct fi_cq_entry comp;
	int ret;

	do {
		ret = fi_cq_read(txcq, &comp, 1);
		if (ret > 0) {
			credits++;
		} else if (ret < 0 && ret != -FI_EAGAIN) {
			FT_PRINTERR("fi_cq_read", ret);
			return ret;
		}
	} while (ret != -FI_EAGAIN);
	return 0;
}

static int send_xfer(int size)
{
	struct fi_cq_entry comp;
	int ret;

	while (!credits) {
		ret = fi_cq_read(txcq, &comp, 1);
		if (ret > 0) {
			goto post;
		} else if (ret < 0 && ret != -FI_EAGAIN) {
			FT_PRINTERR("fi_cq_read", ret);
			return ret;
		}
	}

	credits--;
post:
	ret = fi_send(ep, buf, (size_t) size + prefix_len, fi_mr_desc(mr),
			remote_fi_addr, NULL);
	if (ret)
		FT_PRINTERR("fi_send", ret);

	return ret;
}

static int recv_xfer(int size)
{
	struct timespec a, b;
	struct fi_cq_entry comp;
	int ret;

	clock_gettime(CLOCK_MONOTONIC, &a);
	do {
		ret = fi_cq_read(rxcq, &comp, sizeof comp);
		if (ret < 0) {
			if (ret != -FI_EAGAIN) {
				FT_PRINTERR("fi_cq_read", ret);
				return ret;
			} else if (timeout > 0) {
				clock_gettime(CLOCK_MONOTONIC, &b);
				if (b.tv_sec - a.tv_sec > timeout) {
					fprintf(stderr, "%ds timeout expired waiting to receive message, exiting\n", timeout);
					exit(FI_ENODATA);
				}
			}
		}
	} while (ret == -FI_EAGAIN);

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

	if (ep) {
		ret = fi_close(&ep->fid);
		if (ret != 0) {
			FT_PRINTERR("fi_close", ret);
		}
	}
	if (mr) {
		ret = fi_close(&mr->fid);
		if (ret != 0) {
			FT_PRINTERR("fi_close", ret);
		}
	}
	if (rxcq) {
		ret = fi_close(&rxcq->fid);
		if (ret != 0) {
			FT_PRINTERR("fi_close", ret);
		}
	}
	if (txcq) {
		ret = fi_close(&txcq->fid);
		if (ret != 0) {
			FT_PRINTERR("fi_close", ret);
		}
	}
	free(buf);
}

static int alloc_ep_res(struct fi_info *fi)
{
	struct fi_cq_attr cq_attr;
	struct fi_av_attr av_attr;
	int ret;

	buffer_size = opts.user_options & FT_OPT_SIZE ?
			opts.transfer_size : test_size[TEST_CNT - 1].size;
	if (max_msg_size > 0 && buffer_size > max_msg_size) {
		buffer_size = max_msg_size;
	}
	if (buffer_size < fi->src_addrlen) {
		buffer_size = fi->src_addrlen;
	}
	buffer_size += prefix_len;
	buf = calloc(1, buffer_size);
	if (!buf) {
		perror("calloc");
		return -1;
	}
	payload = (char *) buf + prefix_len;

	memset(&cq_attr, 0, sizeof cq_attr);
	cq_attr.format = FI_CQ_FORMAT_CONTEXT;
	cq_attr.wait_obj = FI_WAIT_NONE;
	cq_attr.size = max_credits << 1;
	ret = fi_cq_open(domain, &cq_attr, &txcq, NULL);
	if (ret) {
		FT_PRINTERR("fi_cq_open", ret);
		goto err1;
	}

	ret = fi_cq_open(domain, &cq_attr, &rxcq, NULL);
	if (ret) {
		FT_PRINTERR("fi_cq_open", ret);
		goto err2;
	}

	ret = fi_mr_reg(domain, buf, buffer_size, 0, 0, 0, 0, &mr, NULL);
	if (ret) {
		FT_PRINTERR("fi_mr_reg", ret);
		goto err3;
	}

	memset(&av_attr, 0, sizeof(av_attr));
	av_attr.type = fi->domain_attr->av_type ?
			fi->domain_attr->av_type : FI_AV_MAP;
	av_attr.name = NULL;
	av_attr.flags = 0;
	ret = fi_av_open(domain, &av_attr, &av, NULL);
	if (ret) {
		FT_PRINTERR("fi_av_open", ret);
		goto err4;
	}

	ret = fi_endpoint(domain, fi, &ep, NULL);
	if (ret) {
		FT_PRINTERR("fi_endpoint", ret);
		goto err5;
	}

	return 0;

err5:
	fi_close(&av->fid);
err4:
	fi_close(&mr->fid);
err3:
	fi_close(&rxcq->fid);
err2:
	fi_close(&txcq->fid);
err1:
	free(buf);
	return ret;
}

static int bind_ep_res(void)
{
	int ret;

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

	/* Post the first recv buffer */
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

	ret = ft_read_addr_opts(&node, &service, hints, &flags, &opts);
	if (ret)
		return ret;

	ret = fi_getinfo(FT_FIVERSION, node, service, flags, hints, &fi);
	if (ret) {
		FT_PRINTERR("fi_getinfo", ret);
		goto err1;
	}
	if (fi->ep_attr->max_msg_size) {
		max_msg_size = fi->ep_attr->max_msg_size;
	}

	ret = fi_fabric(fi->fabric_attr, &fabric, NULL);
	if (ret) {
		FT_PRINTERR("fi_fabric", ret);
		goto err1;
	}
	if (fi->mode & FI_MSG_PREFIX) {
		prefix_len = fi->ep_attr->msg_prefix_size;
	}

	ret = fi_domain(fabric, fi, &domain, NULL);
	if (ret) {
		FT_PRINTERR("fi_domain", ret);
		goto err2;
	}

	ret = alloc_ep_res(fi);
	if (ret) {
		goto err4;
	}

	ret = bind_ep_res();
	if (ret) {
		goto err5;
	}

	return 0;

err5:
	free_ep_res();
err4:
	fi_close(&domain->fid);
err2:
	fi_close(&fabric->fid);
err1:

	return ret;
}

static int client_connect(void)
{
	size_t addrlen;
	int ret;

	ret = common_setup();
	if (ret != 0)
		return ret;

	ret = fi_av_insert(av, fi->dest_addr, 1, &remote_fi_addr, 0, NULL);
	if (ret != 1) {
		FT_PRINTERR("fi_av_insert", ret);
		goto err;
	}

	// send initial message to server with our local address
	addrlen = buffer_size;
	ret = fi_getname(&ep->fid, payload, &addrlen);
	if (ret) {
		FT_PRINTERR("fi_getname", ret);
		return ret;
	}

	ret = send_xfer(addrlen);
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
	fi_close(&domain->fid);
	fi_close(&fabric->fid);

	return ret;
}

static int server_connect(void)
{
	int ret;
	struct fi_cq_entry comp;
	struct timespec a, b;

	ret = common_setup();
	if (ret != 0)
		return ret;

	clock_gettime(CLOCK_MONOTONIC, &a);
	do {
		ret = fi_cq_read(rxcq, &comp, 1);
		if (ret < 0) {
			if (ret != -FI_EAGAIN) {
				FT_PRINTERR("fi_cq_read", ret);
				return ret;
			} else if (timeout * 10 > 0) {
				clock_gettime(CLOCK_MONOTONIC, &b);
				if (b.tv_sec - a.tv_sec > timeout * 10) {
					fprintf(stderr, "%ds timeout expired waiting for message from fi_ud_pingpong client, exiting\n", timeout *10);
					exit(FI_ENODATA);
				}
			}
		}
	} while (ret == -FI_EAGAIN);

	ret = fi_av_insert(av, payload, 1, &remote_fi_addr, 0, NULL);
	if (ret != 1) {
		if (ret == 0) {
			fprintf(stderr, "Unable to resolve remote address 0x%x 0x%x\n",
				((uint32_t *) payload)[0],
				((uint32_t *) payload)[1]);
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
	if (domain)
		fi_close(&domain->fid);

	if (fabric)
		fi_close(&fabric->fid);

	return ret;
}

static int run(void)
{
	int i, ret = 0, ret2;

	ret = opts.dst_addr ? client_connect() : server_connect();
	if (ret)
		return ret;

	if (!(opts.user_options & FT_OPT_SIZE)) {
		for (i = 0; i < TEST_CNT; i++) {
			if (test_size[i].option > opts.size_option)
				continue;

			opts.transfer_size = test_size[i].size;
			if (opts.transfer_size > buffer_size)
				continue;

			init_test(&opts, test_name, sizeof(test_name));
			ret = run_test();
			if (ret)
				goto out;
		}
	} else {
		init_test(&opts, test_name, sizeof(test_name));
		ret = run_test();
		if (ret)
			goto out;
	}

	while (credits < max_credits)
		poll_all_sends();

	ft_finalize(fi, ep, txcq, rxcq, remote_fi_addr);
out:
	free_ep_res();
	ret2 = fi_close(&av->fid);
	if (ret2 != 0) {
		FT_PRINTERR("fi_close", ret2);
	}
	ret2 = fi_close(&domain->fid);
	if (ret2 != 0) {
		FT_PRINTERR("fi_close", ret2);
	}
	ret2 = fi_close(&fabric->fid);
	if (ret2 != 0) {
		FT_PRINTERR("fi_close", ret2);
	}

	return ret;
}

int main(int argc, char **argv)
{
	int ret, op;
	opts = INIT_OPTS;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt(argc, argv, "ht:P" CS_OPTS INFO_OPTS)) != -1) {
		switch (op) {
		case 'P':
			hints->mode |= FI_MSG_PREFIX;
			break;
		case 't':
			timeout = atoi(optarg);
			break;
		default:
			ft_parseinfo(op, optarg, hints);
			ft_parsecsopts(op, optarg, &opts);
			break;
		case '?':
		case 'h':
			ft_csusage(argv[0], "Ping pong client and server using UD.");
			FT_PRINT_OPTS_USAGE("-t <timeout>",
					"seconds before timeout on receive");
			FT_PRINT_OPTS_USAGE("-P", "enable prefix mode");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->ep_attr->type = FI_EP_DGRAM;
	if (opts.user_options & FT_OPT_SIZE)
		hints->ep_attr->max_msg_size = opts.transfer_size;
	hints->caps = FI_MSG;
	hints->mode |= FI_LOCAL_MR;

	ret = run();

	fi_freeinfo(hints);
	fi_freeinfo(fi);
	return -ret;
}
