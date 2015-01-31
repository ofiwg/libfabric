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
static size_t buffer_size;

static struct fi_info hints;

static struct fid_fabric *fab;
static struct fid_pep *pep;
static struct fid_domain *dom;
static struct fid_ep *ep;
static struct fid_eq *cmeq;
static struct fid_cq *rcq, *scq;
static struct fid_mr *mr;

static int send_xfer(int size)
{
	struct fi_cq_entry comp;
	int ret;

	while (!credits) {
		ret = fi_cq_read(scq, &comp, 1);
		if (ret > 0) {
			goto post;
		} else if (ret < 0) {
			if (ret == -FI_EAVAIL) {
				cq_readerr(scq, "scq");
			} else {
				FI_PRINTERR("fi_cq_read", ret);
			}
			return ret;
		}
	}

	credits--;
post:
	ret = fi_send(ep, buf, (size_t) size, fi_mr_desc(mr), 0, NULL);
	if (ret)
		FI_PRINTERR("fi_send", ret);

	return ret;
}

static int recv_xfer(int size)
{
	struct fi_cq_entry comp;
	int ret;

	do {
		ret = fi_cq_read(rcq, &comp, 1);
		if (ret < 0) {
			if (ret == -FI_EAVAIL) {
				cq_readerr(rcq, "rcq");
			} else {
				FI_PRINTERR("fi_cq_read", ret);
			}
			return ret;
		}
	} while (!ret);

	ret = fi_recv(ep, buf, buffer_size, fi_mr_desc(mr), 0, buf);
	if (ret)
		FI_PRINTERR("fi_recv", ret);

	return ret;
}

static int sync_test(void)
{
	int ret;

	ret = wait_for_completion(scq, max_credits - credits);
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
	fi_close(&cmeq->fid);
}

static int alloc_cm_res(void)
{
	struct fi_eq_attr cm_attr;
	int ret;

	memset(&cm_attr, 0, sizeof cm_attr);
	cm_attr.wait_obj = FI_WAIT_FD;
	ret = fi_eq_open(fab, &cm_attr, &cmeq, NULL);
	if (ret)
		FI_PRINTERR("fi_eq_open", ret);

	return ret;
}

static void free_ep_res(void)
{
	fi_close(&mr->fid);
	fi_close(&rcq->fid);
	fi_close(&scq->fid);
	free(buf);
}

static int alloc_ep_res(void)
{
	struct fi_cq_attr cq_attr;
	int ret;

	buffer_size = !opts.custom ? test_size[TEST_CNT - 1].size : opts.transfer_size;
	buf = malloc(buffer_size);
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
		FI_PRINTERR("fi_cq_open", ret);
		goto err1;
	}

	ret = fi_cq_open(dom, &cq_attr, &rcq, NULL);
	if (ret) {
		FI_PRINTERR("fi_cq_open", ret);
		goto err2;
	}

	ret = fi_mr_reg(dom, buf, buffer_size, 0, 0, 0, 0, &mr, NULL);
	if (ret) {
		FI_PRINTERR("fi_mr_reg", ret);
		goto err3;
	}

	if (!cmeq) {
		ret = alloc_cm_res();
		if (ret)
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

	ret = fi_ep_bind(ep, &cmeq->fid, 0);
	if (ret) {
		FI_PRINTERR("fi_ep_bind", ret);
		return ret;
	}

	ret = fi_ep_bind(ep, &scq->fid, FI_SEND);
	if (ret) {
		FI_PRINTERR("fi_ep_bind", ret);
		return ret;
	}

	ret = fi_ep_bind(ep, &rcq->fid, FI_RECV);
	if (ret) {
		FI_PRINTERR("fi_ep_bind", ret);
		return ret;
	}

	ret = fi_enable(ep);
	if (ret)
		return ret;

	ret = fi_recv(ep, buf, buffer_size, fi_mr_desc(mr), 0, buf);
	if (ret)
		FI_PRINTERR("fi_recv", ret);

	return ret;
}

static int server_listen(void)
{
	struct fi_info *fi;
	int ret;

	ret = fi_getinfo(FT_FIVERSION, opts.src_addr, opts.port, FI_SOURCE, &hints, &fi);
	if (ret) {
		FI_PRINTERR("fi_getinfo", ret);
		return ret;
	}

	ret = fi_fabric(fi->fabric_attr, &fab, NULL);
	if (ret) {
		FI_PRINTERR("fi_fabric", ret);
		goto err0;
	}

	ret = fi_passive_ep(fab, fi, &pep, NULL);
	if (ret) {
		FI_PRINTERR("fi_passive_ep", ret);
		goto err1;
	}

	ret = alloc_cm_res();
	if (ret)
		goto err2;

	ret = fi_pep_bind(pep, &cmeq->fid, 0);
	if (ret) {
		FI_PRINTERR("fi_pep_bind", ret);
		goto err3;
	}

	ret = fi_listen(pep);
	if (ret) {
		FI_PRINTERR("fi_listen", ret);
		goto err3;
	}

	fi_freeinfo(fi);
	return 0;
err3:
	free_lres();
err2:
	fi_close(&pep->fid);
err1:
	fi_close(&fab->fid);
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

	rd = fi_eq_sread(cmeq, &event, &entry, sizeof entry, -1, 0);
	if (rd != sizeof entry) {
		FI_DEBUG("fi_eq_sread() %zd %s\n", rd, fi_strerror((int) -rd));
		return (int) rd;
	}

	if (event != FI_CONNREQ) {
		FI_DEBUG("Unexpected CM event %d\n", event);
		ret = -FI_EOTHER;
		goto err1;
	}

	info = entry.info;
	ret = fi_domain(fab, info, &dom, NULL);
	if (ret) {
		FI_PRINTERR("fi_domain", ret);
		goto err1;
	}

	ret = fi_endpoint(dom, info, &ep, NULL);
	if (ret) {
		FI_PRINTERR("fi_endpoint", ret);
		goto err1;
	}

	ret = alloc_ep_res();
	if (ret)
		 goto err2;

	ret = bind_ep_res();
	if (ret)
		goto err3;

	ret = fi_accept(ep, NULL, 0);
	if (ret) {
		FI_PRINTERR("fi_accept", ret);
		goto err3;
	}

	rd = fi_eq_sread(cmeq, &event, &entry, sizeof entry, -1, 0);
	if (rd != sizeof entry) {
		FI_DEBUG("fi_eq_sread() %zd %s\n", rd, fi_strerror((int) -rd));
		goto err3;
	}

	if (event != FI_CONNECTED || entry.fid != &ep->fid) {
		FI_DEBUG("Unexpected CM event %d fid %p (ep %p)\n",
			event, entry.fid, ep);
		ret = -FI_EOTHER;
		goto err3;
	}

	fi_close(&pep->fid);

	fi_freeinfo(info);
	return 0;

err3:
	free_ep_res();
err2:
	fi_close(&ep->fid);
err1:
	fi_reject(pep, info->connreq, NULL, 0);
	fi_freeinfo(info);
	return ret;
}

static int client_connect(void)
{
	struct fi_eq_cm_entry entry;
	struct fi_eq_err_entry err;
	uint32_t event;
	struct fi_info *fi;
	ssize_t rd;
	int ret;

	ret = fi_getinfo(FI_VERSION(1,0), opts.dst_addr, opts.port, 0, &hints, &fi);
	if (ret) {
		FI_PRINTERR("fi_getinfo", ret);
		goto err0;
	}

	ret = fi_fabric(fi->fabric_attr, &fab, NULL);
	if (ret) {
		FI_PRINTERR("fi_fabric", ret);
		goto err1;
	}

	ret = fi_domain(fab, fi, &dom, NULL);
	if (ret) {
		FI_PRINTERR("fi_domain", ret);
		goto err2;
	}

	ret = fi_endpoint(dom, fi, &ep, NULL);
	if (ret) {
		FI_PRINTERR("fi_endpoint", ret);
		goto err3;
	}

	ret = alloc_ep_res();
	if (ret)
		goto err4;

	ret = bind_ep_res();
	if (ret)
		goto err5;

	ret = fi_connect(ep, fi->dest_addr, NULL, 0);
	if (ret) {
		FI_PRINTERR("fi_connect", ret);
		goto err5;
	}

	rd = fi_eq_sread(cmeq, &event, &entry, sizeof entry, -1, 0);
	if (rd != sizeof entry) {
		if (rd == -FI_EAVAIL) {
			rd = fi_eq_readerr(cmeq, &err, 0);
			if (rd != sizeof(err)) {
				FI_DEBUG("fi_eq_readerr() %zd %s\n", rd, fi_strerror((int) -rd));
			} else {
				FI_DEBUG("EQ report error %d %s\n", err.err,
						fi_strerror(err.err));
			}
		} else {
			FI_DEBUG("fi_eq_sread() %zd %s\n", rd, fi_strerror((int) -rd));
		}
		return (int) rd;
	}

	if (event != FI_CONNECTED || entry.fid != &ep->fid) {
		FI_DEBUG("Unexpected CM event %d fid %p (ep %p)\n",
			event, entry.fid, ep);
		ret = -FI_EOTHER;
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

	if (!opts.custom) {
		for (i = 0; i < TEST_CNT; i++) {
			if (test_size[i].option > opts.size_option)
				continue;
			init_test(test_size[i].size, test_name,
					sizeof(test_name), &opts.transfer_size,
					&opts.iterations);
			run_test();
		}
	} else {
		ret = run_test();
	}

	ret = wait_for_completion(scq, max_credits - credits);
	if (ret) {
		return ret;
	}
	credits = max_credits;

	fi_shutdown(ep, 0);
	fi_close(&ep->fid);
	free_ep_res();
	if (!opts.dst_addr)
		free_lres();
	fi_close(&dom->fid);
	fi_close(&fab->fid);
	return ret;
}

int main(int argc, char **argv)
{
	int op, ret;
	opts = INIT_OPTS;

	while ((op = getopt(argc, argv, "h" CS_OPTS INFO_OPTS)) != -1) {
		switch (op) {
		default:
			ft_parseinfo(op, optarg, &hints);
			ft_parsecsopts(op, optarg, &opts);
			break;
		case '?':
		case 'h':
			ft_csusage(argv[0], "Ping pong client and server using message endpoints.");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	ret = ft_getsrcaddr(opts.src_addr, opts.port, &hints);
	if (ret)
		return EXIT_FAILURE;

	hints.ep_type = FI_EP_MSG;
	hints.caps = FI_MSG;
	hints.mode = FI_LOCAL_MR | FI_PROV_MR_ATTR;
	hints.addr_format = FI_SOCKADDR;

	if (opts.prhints) {
		printf("%s", fi_tostr(&hints, FI_TYPE_INFO));
		return EXIT_SUCCESS;
	}

	return run();
}
