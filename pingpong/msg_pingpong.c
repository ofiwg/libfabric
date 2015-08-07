/*
 * Copyright (c) 2013-2014 Intel Corporation.  All rights reserved.
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

#include "shared.h"

static struct cs_opts opts;
static int max_credits = 128;
static int credits = 128;
static char test_name[10] = "custom";
static struct timespec start, end;
static void *buf;
static void *recv_buf;
static void *send_buf;
static size_t buffer_size;

static int verify_data = 0;

static int send_xfer(int size)
{
	struct fi_cq_entry comp;
	int ret;

	while (!credits) {
		ret = fi_cq_read(txcq, &comp, 1);
		if (ret > 0) {
			goto post;
		} else if (ret < 0 && ret != -FI_EAGAIN) {
			if (ret == -FI_EAVAIL) {
				cq_readerr(txcq, "txcq");
			} else {
				FT_PRINTERR("fi_cq_read", ret);
			}
			return ret;
		}
	}

	credits--;
post:
	if (verify_data)
		ft_fill_buf(send_buf, size);

	ret = fi_send(ep, send_buf, (size_t) size, fi_mr_desc(mr), 0, NULL);
	if (ret)
		FT_PRINTERR("fi_send", ret);

	return ret;
}

static int recv_xfer(int size)
{
	struct fi_cq_entry comp;
	int ret;

	do {
		ret = fi_cq_read(rxcq, &comp, 1);
		if (ret < 0 && ret != -FI_EAGAIN) {
			if (ret == -FI_EAVAIL) {
				cq_readerr(rxcq, "rxcq");
			} else {
				FT_PRINTERR("fi_cq_read", ret);
			}
			return ret;
		}
	} while (ret == -FI_EAGAIN);

	if (verify_data) {
		ret = ft_check_buf(recv_buf, size);
		if (ret)
			return ret;
	}

	ret = fi_recv(ep, recv_buf, buffer_size, fi_mr_desc(mr), 0, recv_buf);
	if (ret)
		FT_PRINTERR("fi_recv", ret);

	return ret;
}

static int sync_test(void)
{
	int ret;

	ret = wait_for_completion(txcq, max_credits - credits);
	if (ret) {
		return ret;
	}
	credits = max_credits;

	ret = opts.dst_addr ? send_xfer(16) : recv_xfer(16);
	if (ret) {
		return ret;
	}

	return opts.dst_addr ? recv_xfer(16) : send_xfer(16);
}

static int run_test()
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

static void free_lres(void)
{
	fi_close(&eq->fid);
}

static int alloc_cm_res(void)
{
	struct fi_eq_attr cm_attr;
	int ret;

	memset(&cm_attr, 0, sizeof cm_attr);
	cm_attr.wait_obj = FI_WAIT_FD;
	ret = fi_eq_open(fabric, &cm_attr, &eq, NULL);
	if (ret)
		FT_PRINTERR("fi_eq_open", ret);

	return ret;
}

static void free_ep_res(void)
{
	fi_close(&ep->fid);
	fi_close(&mr->fid);
	fi_close(&rxcq->fid);
	fi_close(&txcq->fid);
	free(buf);
}

static int alloc_ep_res(struct fi_info *fi)
{
	struct fi_cq_attr cq_attr;
	int ret;

	buffer_size = opts.user_options & FT_OPT_SIZE ?
			opts.transfer_size : test_size[TEST_CNT - 1].size;

	buf = malloc(buffer_size * 2);
	if (!buf) {
		perror("malloc");
		return -1;
	}

	recv_buf = buf;
	send_buf = (char *) buf + buffer_size;

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

	ret = fi_mr_reg(domain, buf, buffer_size, FI_RECV | FI_SEND, 0, 0, 0, &mr, NULL);
	if (ret) {
		FT_PRINTERR("fi_mr_reg", ret);
		goto err3;
	}

	if (!eq) {
		ret = alloc_cm_res();
		if (ret)
			goto err4;
	}

	ret = fi_endpoint(domain, fi, &ep, NULL);
	if (ret) {
		FT_PRINTERR("fi_endpoint", ret);
		goto err4;
	}

	return 0;

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

	ret = fi_ep_bind(ep, &eq->fid, 0);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		return ret;
	}

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

	ret = fi_enable(ep);
	if (ret) {
		FT_PRINTERR("fi_enable", ret);
		return ret;
	}

	/* Post the first recv buffer */
	ret = fi_recv(ep, recv_buf, buffer_size, fi_mr_desc(mr), 0, recv_buf);
	if (ret)
		FT_PRINTERR("fi_recv", ret);

	return ret;
}

static int server_listen(void)
{
	int ret;

	ret = fi_getinfo(FT_FIVERSION, opts.src_addr, opts.src_port, FI_SOURCE,
			hints, &fi);
	if (ret) {
		FT_PRINTERR("fi_getinfo", ret);
		return ret;
	}

	ret = fi_fabric(fi->fabric_attr, &fabric, NULL);
	if (ret) {
		FT_PRINTERR("fi_fabric", ret);
		goto err0;
	}

	ret = fi_passive_ep(fabric, fi, &pep, NULL);
	if (ret) {
		FT_PRINTERR("fi_passive_ep", ret);
		goto err1;
	}

	ret = alloc_cm_res();
	if (ret)
		goto err2;

	ret = fi_pep_bind(pep, &eq->fid, 0);
	if (ret) {
		FT_PRINTERR("fi_pep_bind", ret);
		goto err3;
	}

	ret = fi_listen(pep);
	if (ret) {
		FT_PRINTERR("fi_listen", ret);
		goto err3;
	}

	fi_freeinfo(fi);
	return 0;
err3:
	free_lres();
err2:
	fi_close(&pep->fid);
err1:
	fi_close(&fabric->fid);
err0:
	fi_freeinfo(fi);
	return ret;
}

static int server_connect(void)
{
	struct fi_eq_cm_entry entry;
	uint32_t event;
	struct fi_info *info = NULL;
	ssize_t rd;
	int ret;

	rd = fi_eq_sread(eq, &event, &entry, sizeof entry, -1, 0);
	if (rd != sizeof entry) {
		FT_PROCESS_EQ_ERR(rd, eq, "fi_eq_sread", "listen");
		return (int) rd;
	}

	info = entry.info;
	if (event != FI_CONNREQ) {
		fprintf(stderr, "Unexpected CM event %d\n", event);
		ret = -FI_EOTHER;
		goto err1;
	}

	ret = fi_domain(fabric, info, &domain, NULL);
	if (ret) {
		FT_PRINTERR("fi_domain", ret);
		goto err1;
	}

	ret = alloc_ep_res(info);
	if (ret)
		 goto err1;

	ret = bind_ep_res();
	if (ret)
		goto err3;

	ret = fi_accept(ep, NULL, 0);
	if (ret) {
		FT_PRINTERR("fi_accept", ret);
		goto err3;
	}

	rd = fi_eq_sread(eq, &event, &entry, sizeof entry, -1, 0);
	if (rd != sizeof entry) {
		FT_PROCESS_EQ_ERR(rd, eq, "fi_eq_sread", "accept");
		ret = (int) rd;
		goto err3;
	}

	if (event != FI_CONNECTED || entry.fid != &ep->fid) {
		fprintf(stderr, "Unexpected CM event %d fid %p (ep %p)\n",
			event, entry.fid, ep);
		ret = -FI_EOTHER;
		goto err3;
	}

	fi_close(&pep->fid);

	fi_freeinfo(info);
	return 0;

err3:
	free_ep_res();
err1:
	fi_reject(pep, info->handle, NULL, 0);
	fi_freeinfo(info);
	return ret;
}

static int client_connect(void)
{
	struct fi_eq_cm_entry entry;
	uint32_t event;
	ssize_t rd;
	int ret;

	ret = ft_getsrcaddr(opts.src_addr, opts.src_port, hints);
	if (ret)
		return ret;

	ret = fi_getinfo(FT_FIVERSION, opts.dst_addr, opts.dst_port, 0, hints, &fi);
	if (ret) {
		FT_PRINTERR("fi_getinfo", ret);
		goto err0;
	}

	ret = fi_fabric(fi->fabric_attr, &fabric, NULL);
	if (ret) {
		FT_PRINTERR("fi_fabric", ret);
		goto err1;
	}

	ret = fi_domain(fabric, fi, &domain, NULL);
	if (ret) {
		FT_PRINTERR("fi_domain", ret);
		goto err2;
	}

	ret = alloc_ep_res(fi);
	if (ret)
		goto err3;

	ret = bind_ep_res();
	if (ret)
		goto err4;

	ret = fi_connect(ep, fi->dest_addr, NULL, 0);
	if (ret) {
		FT_PRINTERR("fi_connect", ret);
		goto err4;
	}

	rd = fi_eq_sread(eq, &event, &entry, sizeof entry, -1, 0);
	if (rd != sizeof entry) {
		FT_PROCESS_EQ_ERR(rd, eq, "fi_eq_sread", "connect");
		ret = (int) rd;
		goto err4;
	}

	if (event != FI_CONNECTED || entry.fid != &ep->fid) {
		fprintf(stderr, "Unexpected CM event %d fid %p (ep %p)\n",
			event, entry.fid, ep);
		ret = -FI_EOTHER;
		goto err4;
	}

	fi_freeinfo(fi);
	return 0;

err4:
	free_ep_res();
err3:
	fi_close(&domain->fid);
err2:
	fi_close(&fabric->fid);
err1:
	fi_freeinfo(fi);
err0:
	return ret;
}

static int run(void)
{
	int i, ret = 0;

	if (!opts.dst_addr) {
		ret = server_listen();
		if (ret)
			return ret;
	}

	ret = opts.dst_addr ? client_connect() : server_connect();
	if (ret) {
		return ret;
	}

	if (!(opts.user_options & FT_OPT_SIZE)) {
		for (i = 0; i < TEST_CNT; i++) {
			if (test_size[i].option > opts.size_option)
				continue;
			opts.transfer_size = test_size[i].size;
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

	ret = wait_for_completion(txcq, max_credits - credits);
	/* Finalize before closing ep */
	ft_finalize(fi, ep, txcq, rxcq, FI_ADDR_UNSPEC);
out:
	fi_shutdown(ep, 0);
	free_ep_res();
	if (!opts.dst_addr)
		free_lres();
	fi_close(&domain->fid);
	fi_close(&fabric->fid);
	return ret;
}

int main(int argc, char **argv)
{
	int op, ret;
	opts = INIT_OPTS;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt(argc, argv, "vh" CS_OPTS INFO_OPTS)) != -1) {
		switch (op) {
		case 'v':
			verify_data = 1;
			break;
		default:
			ft_parseinfo(op, optarg, hints);
			ft_parsecsopts(op, optarg, &opts);
			break;
		case '?':
		case 'h':
			ft_csusage(argv[0], "Ping pong client and server using message endpoints.");
			FT_PRINT_OPTS_USAGE("-v", "enables data_integrity checks");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->ep_attr->type = FI_EP_MSG;
	hints->caps = FI_MSG;
	hints->mode = FI_LOCAL_MR;
	hints->addr_format = FI_SOCKADDR;

	ret = run();

	fi_freeinfo(hints);
	return -ret;
}
