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

#include <assert.h>
#include <inttypes.h>
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


static uint64_t op_type = FT_RMA_WRITE;
static int max_credits = 128;
static char test_name[10] = "custom";
static struct timespec start, end;
struct fi_rma_iov remote;
static uint64_t cq_data = 1;

static void *local_addr, *remote_addr;
static size_t addrlen = 0;
static fi_addr_t remote_fi_addr;
struct fi_context fi_ctx_send;
struct fi_context fi_ctx_recv;
struct fi_context fi_ctx_write;
struct fi_context fi_ctx_writedata;
struct fi_context fi_ctx_read;
struct fi_context fi_ctx_av;

static int send_msg(int size)
{
	int ret;

	ret = fi_send(ep, buf, (size_t) size, fi_mr_desc(mr), remote_fi_addr,
			&fi_ctx_send);
	if (ret) {
		FT_PRINTERR("fi_send", ret);
		return ret;
	}

	ret = wait_for_data_completion(txcq, 1);

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

	ret = wait_for_data_completion(rxcq, 1);

	return ret;
}

static int read_data(size_t size)
{
	int ret;

	ret = fi_read(ep, buf, size, fi_mr_desc(mr), remote_fi_addr,
		      remote.addr, remote.key, &fi_ctx_read);
	if (ret){
		FT_PRINTERR("fi_read", ret);
		return ret;
	}

	return 0;
}

static int write_data_with_cq_data(size_t size)
{
	int ret;

	ret = fi_writedata(ep, buf, size, fi_mr_desc(mr), cq_data,
			remote_fi_addr, remote.addr, remote.key,
			&fi_ctx_writedata);
	if (ret) {
		FT_PRINTERR("fi_writedata", ret);
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
		FT_PRINTERR("fi_write", ret);
		return ret;
	}
	return 0;
}

static int wait_remote_writedata_completion(void)
{
	struct fi_cq_data_entry comp;
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

	ret = 0;
	if (comp.data != cq_data) {
		fprintf(stderr, "Got unexpected completion data %" PRIu64 "\n",
			comp.data);
	}
	assert(comp.op_context == &fi_ctx_recv || comp.op_context == NULL);
	if (comp.op_context == &fi_ctx_recv) {
		/* We need to repost the receive */
		ret = fi_recv(ep, buf, buffer_size, fi_mr_desc(mr),
				remote_fi_addr, &fi_ctx_recv);
		if (ret)
			FT_PRINTERR("fi_recv", ret);
	}

	return ret;
}

static int sync_test(void)
{
	int ret;

	ret = opts.dst_addr ? send_msg(16) : recv_msg();
	if (ret)
		return ret;

	return opts.dst_addr ? recv_msg() : send_msg(16);
}

static int run_test(void)
{
	int ret, i;

	ret = sync_test();
	if (ret)
		return ret;

	clock_gettime(CLOCK_MONOTONIC, &start);
	for (i = 0; i < opts.iterations; i++) {
		switch (op_type) {
		case FT_RMA_WRITE:
			ret = write_data(opts.transfer_size);
			break;
		case FT_RMA_WRITEDATA:
			ret = write_data_with_cq_data(opts.transfer_size);
			if (ret)
				return ret;
			ret = wait_remote_writedata_completion();
			break;
		case FT_RMA_READ:
			ret = read_data(opts.transfer_size);
			break;
		}
		if (ret)
			return ret;
		ret = wait_for_data_completion(txcq, 1);
		if (ret)
			return ret;
	}
	clock_gettime(CLOCK_MONOTONIC, &end);

	if (opts.machr)
		show_perf_mr(opts.transfer_size, opts.iterations, &start, &end,
				1, opts.argc, opts.argv);
	else
		show_perf(test_name, opts.transfer_size, opts.iterations,
				&start, &end, 1);

	return 0;
}

static void free_ep_res(void)
{
	fi_close(&ep->fid);
	fi_close(&av->fid);
	fi_close(&mr->fid);
	fi_close(&rxcq->fid);
	fi_close(&txcq->fid);
	free(buf);
}

static int alloc_ep_res(struct fi_info *fi)
{
	struct fi_cq_attr cq_attr;
	struct fi_av_attr av_attr;
	uint64_t access_mode;
	int ret;

	buffer_size = opts.user_options & FT_OPT_SIZE ?
			opts.transfer_size : test_size[TEST_CNT - 1].size;
	buf = malloc(MAX(buffer_size, sizeof(uint64_t)));
	if (!buf) {
		perror("malloc");
		return -1;
	}

	memset(&cq_attr, 0, sizeof cq_attr);
	cq_attr.format = FI_CQ_FORMAT_DATA;
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

	switch (op_type) {
	case FT_RMA_READ:
		access_mode = FI_REMOTE_READ;
		break;
	case FT_RMA_WRITE:
	case FT_RMA_WRITEDATA:
		access_mode = FI_REMOTE_WRITE;
		break;
	default:
		/* Impossible to reach here */
		FT_PRINTERR("invalid op_type", ret);
		exit(1);
	}
	ret = fi_mr_reg(domain, buf, MAX(buffer_size, sizeof(uint64_t)),
			access_mode, 0, 0, 0, &mr, NULL);
	if (ret) {
		FT_PRINTERR("fi_mr_reg", ret);
		goto err3;
	}

	memset(&av_attr, 0, sizeof av_attr);
	av_attr.type = fi->domain_attr->av_type ?
			fi->domain_attr->av_type : FI_AV_MAP;
	av_attr.count = 1;
	av_attr.name = NULL;

	ret = fi_av_open(domain, &av_attr, &av, NULL);
	if (ret) {
		FT_PRINTERR("fi_av_open", ret);
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

	ret = fi_endpoint(domain, fi, &ep, NULL);
	if (ret) {
		FT_PRINTERR("fi_endpoint", ret);
		goto err2;
	}

	ret = alloc_ep_res(fi);
	if (ret)
		goto err2;

	ret = bind_ep_res();
	if (ret)
		goto err4;

	return 0;

err4:
	free_ep_res();
err2:
	fi_close(&domain->fid);
err1:
	fi_close(&fabric->fid);
err0:
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

	/* Post the first recv buffer */
	ret = fi_recv(ep, buf, buffer_size, fi_mr_desc(mr), 0, &fi_ctx_recv);
	if (ret) {
		FT_PRINTERR("fi_recv", ret);
		return ret;
	}

	return ret;
}

static int exchange_addr_key(void)
{
	struct fi_rma_iov *rma_iov;
	int ret;

	rma_iov = buf;

	if (opts.dst_addr) {
		rma_iov->addr = fi->domain_attr->mr_mode == FI_MR_SCALABLE ?
				0 : (uintptr_t) buf;
		rma_iov->key = fi_mr_key(mr);
		ret = send_msg(sizeof *rma_iov);
		if (ret)
			return ret;

		ret = recv_msg();
		if (ret)
			return ret;
		remote = *rma_iov;
	} else {
		ret = recv_msg();
		if (ret)
			return ret;
		remote = *rma_iov;

		rma_iov->addr = fi->domain_attr->mr_mode == FI_MR_SCALABLE ?
				0 : (uintptr_t) buf;
		rma_iov->key = fi_mr_key(mr);
		ret = send_msg(sizeof *rma_iov);
		if (ret)
			return ret;
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
	/* Finalize before closing ep */
	ft_finalize(fi, ep, txcq, rxcq, remote_fi_addr);
out:
	free_ep_res();
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

	while ((op = getopt(argc, argv, "ho:" CS_OPTS INFO_OPTS)) != -1) {
		switch (op) {
		case 'o':
			if (!strcmp(optarg, "read")) {
				op_type = FT_RMA_READ;
			} else if (!strcmp(optarg, "writedata")) {
				op_type = FT_RMA_WRITEDATA;
			} else if (!strcmp(optarg, "write")) {
				op_type = FT_RMA_WRITE;
			} else {
				ft_csusage(argv[0], "Ping pong client and server using rma.");
				fprintf(stderr, "  -o <op>\trma op type: read|write (default: write)]\n");
				return EXIT_FAILURE;
			}
			break;
		default:
			ft_parseinfo(op, optarg, hints);
			ft_parsecsopts(op, optarg, &opts);
			break;
		case '?':
		case 'h':
			ft_csusage(argv[0], "Ping pong client and server using rma.");
			fprintf(stderr, "  -o <op>\trma op type: read|write|writedata (default: write)]\n");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_MSG | FI_RMA;
	hints->mode = FI_CONTEXT | FI_LOCAL_MR | FI_RX_CQ_DATA;

	ret =run();

	fi_freeinfo(hints);
	fi_freeinfo(fi);
	return -ret;
}
