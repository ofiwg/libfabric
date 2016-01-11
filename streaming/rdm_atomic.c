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
#include <complex.h>

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
static void *result;
static void *compare;
struct fi_rma_iov remote;

static struct fid_mr *mr_result;
static struct fid_mr *mr_compare;
static struct fi_context fi_ctx_atomic;

static enum fi_datatype datatype;
static size_t *count;
static int run_all_ops = 1;

static enum fi_op get_fi_op(char *op) {
	if (!strcmp(op, "min"))
		return FI_MIN;
	else if (!strcmp(op, "max"))
		return FI_MAX;
	else if (!strcmp(op, "sum"))
		return FI_SUM;
	else if (!strcmp(op, "prod"))
		return FI_PROD;
	else if (!strcmp(op, "lor"))
		return FI_LOR;
	else if (!strcmp(op, "land"))
		return FI_LAND;
	else if (!strcmp(op, "bor"))
		return FI_BOR;
	else if (!strcmp(op, "band"))
		return FI_BAND;
	else if (!strcmp(op, "lxor"))
		return FI_LXOR;
	else if (!strcmp(op, "bxor"))
		return FI_BXOR;
	else if (!strcmp(op, "read"))
		return FI_ATOMIC_READ;
	else if (!strcmp(op, "write"))
		return FI_ATOMIC_WRITE;
	else if (!strcmp(op, "cswap"))
		return FI_CSWAP;
	else if (!strcmp(op, "cswap_ne"))
		return FI_CSWAP_NE;
	else if (!strcmp(op, "cswap_le"))
		return FI_CSWAP_LE;
	else if (!strcmp(op, "cswap_lt"))
		return FI_CSWAP_LT;
	else if (!strcmp(op, "cswap_ge"))
		return FI_CSWAP_GE;
	else if (!strcmp(op, "cswap_gt"))
		return FI_CSWAP_GT;
	else if (!strcmp(op, "mswap"))
		return FI_MSWAP;
	else {
		fprintf(stderr, "Not a valid atomic operation\n");
		return FI_ATOMIC_OP_LAST;
	}
}

static inline size_t datatype_to_size(enum fi_datatype datatype)
{
        switch (datatype) {
	case FI_INT8:   return sizeof(int8_t);
	case FI_UINT8:  return sizeof(uint8_t);
	case FI_INT16:  return sizeof(int16_t);
	case FI_UINT16: return sizeof(uint16_t);
	case FI_INT32:  return sizeof(int32_t);
	case FI_UINT32: return sizeof(uint32_t);
	case FI_FLOAT:  return sizeof(float);
	case FI_INT64:  return sizeof(int64_t);
	case FI_UINT64: return sizeof(uint64_t);
	case FI_DOUBLE: return sizeof(double);
	case FI_FLOAT_COMPLEX: return sizeof(float complex);
	case FI_DOUBLE_COMPLEX: return sizeof(double complex);
	case FI_LONG_DOUBLE: return sizeof(long double);
	case FI_LONG_DOUBLE_COMPLEX: return sizeof(long double complex);
	default:        return 0;
	}
}

static int is_valid_base_atomic_op(enum fi_op op)
{
	int ret;

	ret = fi_atomicvalid(ep, datatype, op, count);
	if (ret) {
		fprintf(stderr, "Provider doesn't support %s base atomic operation ",
			fi_tostr(&op, FI_TYPE_ATOMIC_OP));
		fprintf(stderr, "on %s\n", fi_tostr(&datatype, FI_TYPE_ATOMIC_TYPE));
		return 0;
	}

	return 1;
}

static int is_valid_fetch_atomic_op(enum fi_op op)
{
	int ret;

	ret = fi_fetch_atomicvalid(ep, datatype, op, count);
	if (ret) {
		fprintf(stderr, "Provider doesn't support %s fetch atomic operation ",
			fi_tostr(&op, FI_TYPE_ATOMIC_OP));
		fprintf(stderr, "on %s\n", fi_tostr(&datatype, FI_TYPE_ATOMIC_TYPE));
		return 0;
	}

	return 1;
}

static int is_valid_compare_atomic_op(enum fi_op op)
{
	int ret;

	ret = fi_compare_atomicvalid(ep, datatype, op, count);
	if (ret) {
		fprintf(stderr, "Provider doesn't support %s compare atomic operation ",
			fi_tostr(&op, FI_TYPE_ATOMIC_OP));
		fprintf(stderr, "on %s\n", fi_tostr(&datatype, FI_TYPE_ATOMIC_TYPE));
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
		ret = ft_get_tx_comp(++tx_seq);
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
		ret = ft_get_tx_comp(++tx_seq);
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
		ret = ft_get_tx_comp(++tx_seq);
	}

	return ret;
}

static void report_perf()
{
	if (opts.machr)
		show_perf_mr(opts.transfer_size, opts.iterations, &start, &end, 1, opts.argc,
			opts.argv);
	else
		show_perf(test_name, opts.transfer_size, opts.iterations, &start, &end, 1);
}

static int run_op(void)
{
	int ret, i, len;

	count = (size_t *) malloc(sizeof(size_t));
	ft_sync();

	switch (op_type) {
	case FI_MIN:
	case FI_MAX:
	case FI_SUM:
	case FI_PROD:
	case FI_LOR:
	case FI_LAND:
	case FI_BOR:
	case FI_BAND:
	case FI_LXOR:
	case FI_BXOR:
	case FI_ATOMIC_WRITE:
		for (datatype = 0; datatype <= FI_LONG_DOUBLE_COMPLEX; datatype++) {
			ret = is_valid_base_atomic_op(op_type);
			if (ret <= 0)
				continue;

			len = snprintf(test_name, sizeof(test_name), "%s_",
				fi_tostr(&datatype, FI_TYPE_ATOMIC_TYPE));
			snprintf(test_name + len, sizeof(test_name), "%s_base_lat",
				fi_tostr(&op_type, FI_TYPE_ATOMIC_OP));
			opts.transfer_size = datatype_to_size(datatype);
			init_test(&opts, test_name, sizeof(test_name));

			ft_start();
			for (i = 0; i < opts.iterations; i++) {
				ret = execute_base_atomic_op(op_type);
				if (ret)
					break;
			}
			ft_stop();
			report_perf();
		}
	case FI_ATOMIC_READ:
		for (datatype = 0; datatype <= FI_LONG_DOUBLE_COMPLEX; datatype++) {
			ret = is_valid_fetch_atomic_op(op_type);
			if (ret <= 0)
				continue;

			len = snprintf(test_name, sizeof(test_name), "%s_",
				fi_tostr(&datatype, FI_TYPE_ATOMIC_TYPE));
			snprintf(test_name + len, sizeof(test_name), "%s_fetch_lat",
				fi_tostr(&op_type, FI_TYPE_ATOMIC_OP));
			opts.transfer_size = datatype_to_size(datatype);
			init_test(&opts, test_name, sizeof(test_name));

			ft_start();
			for (i = 0; i < opts.iterations; i++) {
				ret = execute_fetch_atomic_op(op_type);
				if (ret)
					break;
			}
			ft_stop();
			report_perf();
		}
		break;
	case FI_CSWAP:
	case FI_CSWAP_NE:
	case FI_CSWAP_LE:
	case FI_CSWAP_LT:
	case FI_CSWAP_GE:
	case FI_CSWAP_GT:
	case FI_MSWAP:
		for (datatype = 0; datatype <= FI_LONG_DOUBLE_COMPLEX; datatype++) {
			ret = is_valid_compare_atomic_op(op_type);
			if (ret <= 0)
				continue;

			len = snprintf(test_name, sizeof(test_name), "%s_",
				fi_tostr(&datatype, FI_TYPE_ATOMIC_TYPE));
			snprintf(test_name + len, sizeof(test_name), "%s_lat",
				fi_tostr(&op_type, FI_TYPE_ATOMIC_OP));
			opts.transfer_size = datatype_to_size(datatype);
			init_test(&opts, test_name, sizeof(test_name));

			ft_start();
			for (i = 0; i < opts.iterations; i++) {
				ret = execute_compare_atomic_op(op_type);
				if(ret)
					break;
			}
			ft_stop();
			report_perf();
		}
		break;
	default:
		ret = -EINVAL;
		goto out;
	}

	if (ret)
		goto out;

	ret = 0;
out:
	free(count);
	return ret;
}

static int run_ops(void)
{
	int ret;

	for (op_type = FI_MIN; op_type < FI_ATOMIC_OP_LAST; op_type++) {
		ret = run_op();
		if (ret)
			return ret;
	}

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

	ret = alloc_ep_res(fi);
	if (ret)
		return ret;

	ret = ft_init_ep();
	if (ret)
		return ret;

	return 0;
}

static int run(void)
{
	int ret;

	ret = init_fabric();
	if (ret)
		return ret;

	ret = ft_init_av();
	if (ret)
		goto out;

	ret = ft_exchange_keys(&remote);
	if (ret)
		goto out;

	ret = run_test();
	if (ret)
		goto out;

	ft_sync();
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
					FT_PRINT_OPTS_USAGE("-o <op>", "atomic op type: all|min|max|sum|prod|lor|");
					FT_PRINT_OPTS_USAGE("", "land|bor|band|lxor|bxor|read|write|cswap|cswap_ne|"
						"cswap_le|cswap_lt|");
					FT_PRINT_OPTS_USAGE("", "cswap_ge|cswap_gt|mswap (default: all)]");
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
			FT_PRINT_OPTS_USAGE("-o <op>", "atomic op type: all|min|max|sum|prod|lor|");
			FT_PRINT_OPTS_USAGE("", "land|bor|band|lxor|bxor|read|write|cswap|cswap_ne|"
					"cswap_le|cswap_lt|");
			FT_PRINT_OPTS_USAGE("", "cswap_ge|cswap_gt|mswap (default: all)]");
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
