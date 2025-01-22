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
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
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

#include <rdma/fi_errno.h>
#include <rdma/fi_atomic.h>

#include "shared.h"
#include <hmem.h>

static enum fi_op op_type = FI_MIN;
static void *result;
static void *compare;
static void *cpy_dst;

static struct fid_mr *mr_result;
static struct fid_mr *mr_compare;
static struct fi_context2 fi_ctx_atomic;

static enum fi_datatype datatype;
static int run_all_ops = 1, run_all_datatypes = 1;

static enum fi_op get_fi_op(char *op)
{
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
		return OFI_ATOMIC_OP_CNT;
	}
}

static enum fi_datatype get_fi_datatype(char *op)
{
	if (!strcmp(op, "int8"))
		return FI_INT8;
	else if (!strcmp(op, "uint8"))
		return FI_UINT8;
	else if (!strcmp(op, "int16"))
		return FI_INT16;
	else if (!strcmp(op, "uint16"))
		return FI_UINT16;
	else if (!strcmp(op, "int32"))
		return FI_INT32;
	else if (!strcmp(op, "uint32"))
		return FI_UINT32;
	else if (!strcmp(op, "int64"))
		return FI_INT64;
	else if (!strcmp(op, "uint64"))
		return FI_UINT64;
	else if (!strcmp(op, "int128"))
		return FI_INT128;
	else if (!strcmp(op, "uint128"))
		return FI_UINT128;
	else if (!strcmp(op, "float"))
		return FI_FLOAT;
	else if (!strcmp(op, "double"))
		return FI_DOUBLE;
	else if (!strcmp(op, "float_complex"))
		return FI_FLOAT_COMPLEX;
	else if (!strcmp(op, "double_complex"))
		return FI_DOUBLE_COMPLEX;
	else if (!strcmp(op, "long_double"))
		return FI_LONG_DOUBLE;
	else if (!strcmp(op, "long_double_complex"))
		return FI_LONG_DOUBLE_COMPLEX;
	else {
		fprintf(stderr, "Not a valid atomic operation\n");
		return OFI_DATATYPE_CNT;
	}
}

static void print_opts_usage(char *name)
{
	ft_csusage(name, NULL);
	/* Atomic op type */
	FT_PRINT_OPTS_USAGE("-o <op>", "atomic op type: all|min|max|sum|prod|lor|");
	FT_PRINT_OPTS_USAGE("", "land|bor|band|lxor|bxor|read|write|cswap|cswap_ne|"
				"cswap_le|cswap_lt|");
	FT_PRINT_OPTS_USAGE("", "cswap_ge|cswap_gt|mswap (default: all)");
	/* Atomic datatype */
	FT_PRINT_OPTS_USAGE("-z <datatype>", "atomic datatype: int8|uint8|int16|uint16|");
	FT_PRINT_OPTS_USAGE("", "int32|uint32|int64|uint64|int128|uint128|"
			    "float|double|float_complex|double_complex|");
	FT_PRINT_OPTS_USAGE("", "long_double|long_double_complex (default: all)");
	FT_PRINT_OPTS_USAGE("-v", "enables data_integrity checks");
}

static inline int execute_base_atomic_op(void)
{
	int ret;

	ret = ft_post_atomic(FT_ATOMIC_BASE, ep, NULL, NULL, NULL, NULL,
			     &remote, datatype, op_type, &fi_ctx_atomic);
	if (ret)
		return ret;

	ret = ft_get_tx_comp(tx_seq);

	return ret;
}

static inline int execute_fetch_atomic_op(void)
{
	int ret;

	ret = ft_post_atomic(FT_ATOMIC_FETCH, ep, NULL, NULL, result,
			     fi_mr_desc(mr_result), &remote, datatype,
			     op_type, &fi_ctx_atomic);
	if (ret)
		return ret;

	ret = ft_get_tx_comp(tx_seq);

	return ret;
}

static inline int execute_compare_atomic_op(void)
{
	int ret;

	ret = ft_post_atomic(FT_ATOMIC_COMPARE, ep, compare, fi_mr_desc(mr_compare),
			     result, fi_mr_desc(mr_result), &remote, datatype,
			     op_type, &fi_ctx_atomic);
	if (ret)
		return ret;

	ret = ft_get_tx_comp(tx_seq);

	return ret;
}

static int fill_data(enum ft_atomic_opcodes opcode)
{
	int ret;

	switch (opcode) {
	case FT_ATOMIC_COMPARE:
		ft_fill_atomic(compare, 1, datatype);
		/* fall through */
	case FT_ATOMIC_FETCH:
		ft_hmem_memset(opts.iface, opts.device, result, 0,
			       datatype_to_size(datatype));
		/* fall through */
	case FT_ATOMIC_BASE:
		ft_fill_atomic(tx_buf, 1, datatype);
		ft_fill_atomic(rx_buf, 1, datatype);
		break;
	default:
		break;
	}

	ret = ft_hmem_copy_from(opts.iface, opts.device, cpy_dst,
				rx_buf, datatype_to_size(datatype));
	if (ret)
		return ret;

	ft_sync();
	return ret;
}

static void report_perf(void)
{
	int len;

	len = snprintf((test_name), sizeof(test_name), "%s_",
		       fi_tostr(&(datatype), FI_TYPE_ATOMIC_TYPE));
	snprintf((test_name) + len, sizeof(test_name) - len, "%s_lat",
		 fi_tostr(&op_type, FI_TYPE_ATOMIC_OP));

	if (opts.machr)
		show_perf_mr(opts.transfer_size, opts.iterations, &start, &end, 1, opts.argc,
			opts.argv);
	else
		show_perf(test_name, opts.transfer_size, opts.iterations, &start, &end, 1);
}

static int handle_atomic_base_op(void)
{
	int ret = FI_SUCCESS, i;
	size_t count = 0;

	ret = check_base_atomic_op(ep, op_type, datatype, &count);
	if (ret)
		return ret;

	opts.transfer_size = datatype_to_size(datatype);
	ft_start();
	for (i = 0; i < opts.iterations; i++) {
		if (ft_check_opts(FT_OPT_VERIFY_DATA)) {
			ret = fill_data(FT_ATOMIC_BASE);
			if (ret)
				return ret;
		}

		ret = execute_base_atomic_op();
		if (ret)
			break;

		if (ft_check_opts(FT_OPT_VERIFY_DATA)) {
			ft_sync();
			ret = ft_check_atomic(FT_ATOMIC_BASE, op_type, datatype,
					      tx_buf, cpy_dst, rx_buf, compare,
					      result, 1);
			if (ret)
				return ret;
		}
	}
	ft_stop();
	report_perf();
	return FI_SUCCESS;
}

static int handle_atomic_fetch_op(void)
{
	int ret = FI_SUCCESS, i;
	size_t count = 0;

	ret = check_fetch_atomic_op(ep, op_type, datatype, &count);
	if (ret)
		return ret;

	opts.transfer_size = datatype_to_size(datatype);
	ft_start();
	for (i = 0; i < opts.iterations; i++) {
		if (ft_check_opts(FT_OPT_VERIFY_DATA)) {
			ret = fill_data(FT_ATOMIC_FETCH);
			if (ret)
				return ret;
		}

		ret = execute_fetch_atomic_op();
		if (ret)
			break;

		if (ft_check_opts(FT_OPT_VERIFY_DATA)) {
			ft_sync();
			ret = ft_check_atomic(FT_ATOMIC_FETCH, op_type, datatype,
					      tx_buf, cpy_dst, rx_buf, compare,
					      result, 1);
			if (ret)
				return ret;
		}
	}
	ft_stop();
	report_perf();
	return FI_SUCCESS;
}

static int handle_atomic_compare_op(void)
{
	int ret = FI_SUCCESS, i;
	size_t count = 0;

	ret = check_compare_atomic_op(ep, op_type, datatype, &count);
	if (ret)
		return ret;

	opts.transfer_size = datatype_to_size(datatype);
	ft_start();
	for (i = 0; i < opts.iterations; i++) {
		if (ft_check_opts(FT_OPT_VERIFY_DATA)) {
			ret = fill_data(FT_ATOMIC_COMPARE);
			if (ret)
				return ret;
		}

		ret = execute_compare_atomic_op();
		if (ret)
			break;

		if (ft_check_opts(FT_OPT_VERIFY_DATA)) {
			ft_sync();
			ret = ft_check_atomic(FT_ATOMIC_COMPARE, op_type, datatype,
					      tx_buf, cpy_dst, rx_buf, compare,
					      result, 1);
			if (ret)
				return ret;
		}
	}
	ft_stop();
	report_perf();
	return FI_SUCCESS;
}

static int run_dt(void)
{
	int ret = -FI_EINVAL;

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
		ret = handle_atomic_base_op();
		break;
	case FI_ATOMIC_READ:
		ret = handle_atomic_fetch_op();
		break;
	case FI_CSWAP:
	case FI_CSWAP_NE:
	case FI_CSWAP_LE:
	case FI_CSWAP_LT:
	case FI_CSWAP_GE:
	case FI_CSWAP_GT:
	case FI_MSWAP:
		ret = handle_atomic_compare_op();
		break;
	default:
		FT_WARN("Invalid atomic operation type %d\n", op_type);
		break;
	}

	ft_sync();

	if (ret == -FI_ENOSYS || ret == -FI_EOPNOTSUPP) {
		fprintf(stderr, "Provider doesn't support %s ",
			fi_tostr(&op_type, FI_TYPE_ATOMIC_OP));
		fprintf(stderr, "atomic operation on %s\n",
			fi_tostr(&datatype, FI_TYPE_ATOMIC_TYPE));
		return FI_SUCCESS;
	}
	if (ret) {
		fprintf(stderr, "Failed atomic op %s ",
			fi_tostr(&op_type, FI_TYPE_ATOMIC_OP));
		fprintf(stderr, "with datatype %s\n",
			fi_tostr(&datatype, FI_TYPE_ATOMIC_TYPE));
	}
	return ret;
}

static int run_op(void)
{
	int ret;

	if (!run_all_datatypes)
		return run_dt();

	for (datatype = 0; datatype < OFI_DATATYPE_CNT; datatype++) {
		ret = run_dt();
		if (ret && ret != -FI_ENOSYS && ret != -FI_EOPNOTSUPP) {
			FT_PRINTERR("run_op", ret);
			return ret;
		}
	}
	return FI_SUCCESS;
}

static int run_test(void)
{
	int ret;

	if (!run_all_ops)
		return run_op();

	for (op_type = FI_MIN; op_type < OFI_ATOMIC_OP_CNT; op_type++) {
		ret = run_op();
		if (ret && ret != -FI_ENOSYS && ret != -FI_EOPNOTSUPP) {
			FT_PRINTERR("run_op", ret);
			return ret;
		}
	}

	return FI_SUCCESS;
}

static void free_res(void)
{
	FT_CLOSE_FID(mr_result);
	FT_CLOSE_FID(mr_compare);
	if (result) {
		ft_hmem_free(opts.iface, result);
		result = NULL;
	}
	if (compare) {
		ft_hmem_free(opts.iface, compare);
		compare = NULL;
	}
	if (cpy_dst) {
		ft_hmem_free_host(opts.iface, cpy_dst);
		cpy_dst = NULL;
	}
}

static uint64_t get_mr_key()
{
	static uint64_t user_key = FT_MR_KEY + 1;

	return fi->domain_attr->mr_mode & FI_MR_PROV_KEY ? 0 : user_key++;
}

static int alloc_ep_res(struct fi_info *fi)
{
	int ret;
	int mr_local = !!(fi->domain_attr->mr_mode & FI_MR_LOCAL);

	ret = ft_hmem_alloc(opts.iface, opts.device, &result, buf_size);
	if (ret) {
		perror("hmem allocation error");
		return -1;
	}

	ret = ft_hmem_alloc(opts.iface, opts.device, &compare, buf_size);
	if (ret) {
		perror("hmem allocation error");
		return -1;
	}

	ret = ft_hmem_alloc_host(opts.iface, &cpy_dst, opts.transfer_size);
	if (ret)
		return ret;

	// registers local data buffer that stores results
	ret = ft_reg_mr(fi, result, buf_size,
			(mr_local ? FI_READ : 0) | FI_REMOTE_WRITE,
			 get_mr_key(), opts.iface, opts.device, &mr_result, NULL);
	if (ret) {
		FT_PRINTERR("fi_mr_reg", -ret);
		return ret;
	}

	// registers local data buffer that contains comparison data
	ret = ft_reg_mr(fi, compare, buf_size,
			(mr_local ? FI_WRITE : 0) | FI_REMOTE_READ,
			 get_mr_key(), opts.iface, opts.device, &mr_compare, NULL);
	if (ret) {
		FT_PRINTERR("fi_mr_reg", ret);
		return ret;
	}

	return 0;
}

static int run(void)
{
	int ret;

	ret = ft_init_fabric();
	if (ret)
		return ret;

	ret = alloc_ep_res(fi);
	if (ret)
		return ret;

	ret = ft_exchange_keys(&remote);
	if (ret)
		goto out;

	ret = run_test();
	if (ret)
		goto out;

	ft_sync();
	ft_finalize();
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

	while ((op = getopt_long(argc, argv, "ho:Uz:v" CS_OPTS INFO_OPTS,
				 long_opts, &lopt_idx)) != -1) {
		switch (op) {
		case 'o':
			if (!strncasecmp("all", optarg, 3)) {
				run_all_ops = 1;
			} else {
				run_all_ops = 0;
				op_type = get_fi_op(optarg);
				if (op_type == OFI_ATOMIC_OP_CNT) {
					print_opts_usage(argv[0]);
					return EXIT_FAILURE;
				}
			}
			break;
		case 'U':
			hints->tx_attr->op_flags |= FI_DELIVERY_COMPLETE;
			break;
		case 'z':
			if (!strncasecmp("all", optarg, 3)) {
				run_all_datatypes = 1;
			} else {
				run_all_datatypes = 0;
				datatype = get_fi_datatype(optarg);
				if (datatype == OFI_DATATYPE_CNT) {
					print_opts_usage(argv[0]);
					return EXIT_FAILURE;
				}
			}
			break;
		case 'v':
			opts.options |= FT_OPT_VERIFY_DATA;
			break;
		default:
			if (!ft_parse_long_opts(op, optarg))
				continue;
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			break;
		case '?':
		case 'h':
			print_opts_usage(argv[0]);
			ft_longopts_usage();
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_MSG | FI_ATOMICS;
	hints->mode = FI_CONTEXT | FI_CONTEXT2;
	hints->domain_attr->mr_mode = opts.mr_mode;

	ret = run();

	free_res();
	ft_free_res();
	return ft_exit_code(ret);
}
