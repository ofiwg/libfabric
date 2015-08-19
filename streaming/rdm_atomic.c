/*
 * Copyright (c) 2013-2015 Intel Corporation.  All rights reserved.
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
#include <strings.h>
#include <errno.h>
#include <getopt.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <netdb.h>
#include <fcntl.h>
#include <unistd.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <time.h>

#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_atomic.h>

#include "shared.h"


static enum fi_op op_type = FI_MIN;
static char test_name[10] = "custom";
static struct timespec start, end;
static void *result;
static void *compare;
struct fi_rma_iov remote;

static struct fid_mr *mr_result;
static struct fid_mr *mr_compare;
static struct fi_context fi_ctx_atomic;

// performing aotmics operation on UINT_64 as an example
static enum fi_datatype datatype = FI_UINT64;
static size_t *count;
static int run_all_ops = 1;


static const char* get_fi_op_name(enum fi_op op)
{
	switch (op) {
	case FI_MIN: return "min";
	case FI_MAX: return "max";
	case FI_ATOMIC_READ: return "read";
	case FI_ATOMIC_WRITE: return "write";
	case FI_CSWAP: return "cswap";
	default: return "";
	}
}

static enum fi_op get_fi_op(char *op) {
	if (!strcmp(op, "min"))
		return FI_MIN;
	else if (!strcmp(op, "max"))
		return FI_MAX;
	else if (!strcmp(op, "read"))
		return FI_ATOMIC_READ;
	else if (!strcmp(op, "write"))
		return FI_ATOMIC_WRITE;
	else if (!strcmp(op, "cswap"))
		return FI_CSWAP;
	else {
		fprintf(stderr, "Not supported by the example\n");
		return FI_ATOMIC_OP_LAST;
	}
}

static int post_recv(void)
{
	int ret;

	ret = fi_recv(ep, buf, rx_size, fi_mr_desc(mr), 0, &rx_ctx);
	if (ret){
		FT_PRINTERR("fi_recv", ret);
		return ret;
	}

	return ft_wait_for_comp(rxcq, 1);
}

static int send_msg(int size)
{
	int ret;

	ret = fi_send(ep, buf, (size_t) size, fi_mr_desc(mr), remote_fi_addr,
			&tx_ctx);
	if (ret) {
		FT_PRINTERR("fi_send", ret);
		return ret;
	}

	return ft_wait_for_comp(txcq, 1);
}

static int sync_test(void)
{
	int ret;

	ret = opts.dst_addr ? send_msg(16) : post_recv();
	if (ret)
		return ret;

	return opts.dst_addr ? post_recv() : send_msg(16);
}

static int is_valid_base_atomic_op(enum fi_op op)
{
	int ret;

	ret = fi_atomicvalid(ep, datatype, op, count);
	if (ret) {
		fprintf(stderr, "Provider doesn't support %s"
				" base atomic operation\n", get_fi_op_name(op));
		return 0;
	}

	return 1;
}

static int is_valid_fetch_atomic_op(enum fi_op op)
{
	int ret;

	ret = fi_fetch_atomicvalid(ep, datatype, op, count);
	if (ret) {
		fprintf(stderr, "Provider doesn't support %s"
				" fetch atomic operation\n", get_fi_op_name(op));
		return 0;
	}

	return 1;
}

static int is_valid_compare_atomic_op(enum fi_op op)
{
	int ret;

	ret = fi_compare_atomicvalid(ep, datatype, op, count);
	if (ret) {
		fprintf(stderr, "Provider doesn't support %s"
				" compare atomic operation\n", get_fi_op_name(op));
		return 0;
	}

	return 1;
}


static int execute_base_atomic_op(enum fi_op op)
{
	int ret;

	ret = fi_atomic(ep, buf, 1, fi_mr_desc(mr), remote_fi_addr, remote.addr,
		       	remote.key, datatype, op, &fi_ctx_atomic);
	if (ret) {
		FT_PRINTERR("fi_atomic", ret);
	} else {
		ret = ft_wait_for_comp(txcq, 1);
	}

	return ret;
}

static int execute_fetch_atomic_op(enum fi_op op)
{
	int ret;

	ret = fi_fetch_atomic(ep, buf, 1, fi_mr_desc(mr), result,
			fi_mr_desc(mr_result), remote_fi_addr, remote.addr,
			remote.key, datatype, op, &fi_ctx_atomic);
	if (ret) {
		FT_PRINTERR("fi_fetch_atomic", ret);
	} else {
		ret = ft_wait_for_comp(txcq, 1);
	}

	return ret;
}

static int execute_compare_atomic_op(enum fi_op op)
{
	int ret;

	ret = fi_compare_atomic(ep, buf, 1, fi_mr_desc(mr), compare,
			fi_mr_desc(mr_compare), result, fi_mr_desc(mr_result),
			remote_fi_addr, remote.addr, remote.key, datatype, op,
			&fi_ctx_atomic);
	if (ret) {
		FT_PRINTERR("fi_compare_atomic", ret);
	} else {
		ret = ft_wait_for_comp(txcq, 1);
	}

	return ret;
}

static int run_op(void)
{
	int ret, i;

	count = (size_t *) malloc(sizeof(size_t));
	sync_test();
	clock_gettime(CLOCK_MONOTONIC, &start);

	switch (op_type) {
	case FI_MIN:
	case FI_MAX:
	case FI_ATOMIC_WRITE:
		ret = is_valid_base_atomic_op(op_type);
		if (ret > 0) {
			for (i = 0; i < opts.iterations; i++) {
				ret = execute_base_atomic_op(op_type);
				if (ret)
					break;
			}
		}
	case FI_ATOMIC_READ:
		ret = is_valid_fetch_atomic_op(op_type);
		if (ret > 0) {
			for (i = 0; i < opts.iterations; i++) {
				ret = execute_fetch_atomic_op(op_type);
				if (ret)
					break;
			}
		}
		break;
	case FI_CSWAP:
		ret = is_valid_compare_atomic_op(op_type);
		if (ret > 0) {
			for (i = 0; i < opts.iterations; i++) {
				ret = execute_compare_atomic_op(op_type);
				if(ret)
					break;
			}
		}
		break;
	default:
		ret = -EINVAL;
		goto out;
	}

	clock_gettime(CLOCK_MONOTONIC, &end);

	if (ret)
		goto out;

	if (opts.machr)
		show_perf_mr(opts.transfer_size, opts.iterations, &start, &end,
				(op_type == FI_CSWAP || op_type == FI_ATOMIC_READ) ? 1 : 2, opts.argc, opts.argv);
	else
		show_perf(test_name, opts.transfer_size, opts.iterations,
				&start, &end, (op_type == FI_CSWAP || op_type == FI_ATOMIC_READ) ? 1 : 2);

	ret = 0;
out:
	free(count);
	return ret;
}

static int run_ops(void)
{
	int ret;

	op_type = FI_MIN;
	ret = run_op();
	if (ret)
		return ret;

	op_type = FI_MAX;
	ret = run_op();
	if (ret)
		return ret;

	op_type = FI_ATOMIC_READ;
	ret = run_op();
	if (ret)
		return ret;

	op_type = FI_ATOMIC_WRITE;
	ret = run_op();
	if (ret)
		return ret;

	op_type = FI_CSWAP;
	ret = run_op();
	if (ret)
		return ret;

	return 0;
}

static int run_test(void)
{
	return run_all_ops ? run_ops() : run_op();
}

static void free_res(void)
{
	FT_CLOSE_FID(mr_result);
	FT_CLOSE_FID(mr_compare);
	if (result) {
		free(result);
		result = NULL;
	}
	if (compare) {
		free(compare);
		compare = NULL;
	}
}

static uint64_t get_mr_key()
{
	static uint64_t user_key = FT_MR_KEY;

	return fi->domain_attr->mr_mode == FI_MR_SCALABLE ?
		user_key++ : 0;
}

static int alloc_ep_res(struct fi_info *fi)
{
	int ret;

	ret = ft_alloc_active_res(fi);
	if (ret)
		return ret;

	result = malloc(buf_size);
	if (!result) {
		perror("malloc");
		return -1;
	}

	compare = malloc(buf_size);
	if (!compare) {
		perror("malloc");
		return -1;
	}

	// registers local data buffer buff that specifies
	// the first operand of the atomic operation
	ret = fi_mr_reg(domain, buf, buf_size,
		FI_REMOTE_READ | FI_REMOTE_WRITE, 0,
		get_mr_key(), 0, &mr, NULL);
	if (ret) {
		FT_PRINTERR("fi_mr_reg", ret);
		return ret;
	}

	// registers local data buffer that stores initial value of
	// the remote buffer
	ret = fi_mr_reg(domain, result, buf_size,
		FI_REMOTE_READ | FI_REMOTE_WRITE, 0,
		get_mr_key(), 0, &mr_result, NULL);
	if (ret) {
		FT_PRINTERR("fi_mr_reg", -ret);
		return ret;
	}

	// registers local data buffer that contains comparison data
	ret = fi_mr_reg(domain, compare, buf_size,
		FI_REMOTE_READ | FI_REMOTE_WRITE, 0,
		get_mr_key(), 0, &mr_compare, NULL);
	if (ret) {
		FT_PRINTERR("fi_mr_reg", ret);
		return ret;
	}

	return 0;
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

	ret = ft_open_fabric_res();
	if (ret)
		return ret;

	ret = fi_domain(fabric, fi, &domain, NULL);
	if (ret) {
		FT_PRINTERR("fi_domain", ret);
		return ret;
	}

	ret = alloc_ep_res(fi);
	if (ret)
		return ret;

	ret = ft_init_ep();
	if (ret)
		return ret;

	return 0;
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

		ret = post_recv();
		if (ret)
			return ret;
		remote = *rma_iov;
	} else {
		ret = post_recv();
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

	ret = ft_init_av();
	if (ret)
			goto out;

	ret = exchange_addr_key();
	if (ret)
		goto out;

	if (!(opts.options & FT_OPT_SIZE)) {
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
			if (!strncasecmp("all", optarg, 3)) {
				run_all_ops = 1;
			} else {
				run_all_ops = 0;
				op_type = get_fi_op(optarg);
				if (op_type == FI_ATOMIC_OP_LAST) {
					ft_csusage(argv[0], NULL);
					fprintf(stderr, "  -o <op>\tatomic op type: all|min|max|read|write|cswap (default: all)]\n");
					return EXIT_FAILURE;
				}
			}
			break;
		default:
			ft_parseinfo(op, optarg, hints);
			ft_parsecsopts(op, optarg, &opts);
			break;
		case '?':
		case 'h':
			ft_csusage(argv[0], "Ping pong client and server using atomic ops.");
			fprintf(stderr, "  -o <op>\tatomic op type: all|min|max|read|write|cswap (default: all)]\n");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_MSG | FI_ATOMICS;
	hints->mode = FI_CONTEXT | FI_LOCAL_MR;

	ret = run();

	free_res();
	ft_free_res();
	return -ret;
}
