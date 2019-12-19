/*
 * Copyright (c) 2019 Amazon.com, Inc. or its affiliates. All rights reserved.
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
#include <getopt.h>
#include <string.h>
#include <time.h>

#include <rdma/fi_errno.h>

#include "unit_common.h"
#include "shared.h"

#define EP_TEST_SIZES_STRIDE 101

static unsigned int num_eps = 128;
static unsigned int max_eps;
static unsigned int seed = -1;

struct ft_res {
	struct fid_ep *ep;
	struct fid_av *av;
	struct fid_cq *txcq;
	struct fid_cq *rxcq;
} *res;

static char err_buf[512];

static int create_av(struct fid_av **av, enum fi_av_type type, int rx_ctx_bits,
		     size_t count, size_t ep_per_node, const char *name,
		     void *map_addr, uint64_t flags)
{
	struct fi_av_attr av_attr;

	memset(&av_attr, 0, sizeof(av_attr));
	av_attr.type = type;
	av_attr.rx_ctx_bits = rx_ctx_bits;
	av_attr.count = count;
	av_attr.ep_per_node = ep_per_node;
	av_attr.name = name;
	av_attr.map_addr = map_addr;
	av_attr.flags = flags;

	return fi_av_open(domain, &av_attr, av, NULL);
}

static int create_cq(struct fid_cq **cq, size_t size, uint64_t flags,
		     enum fi_cq_format format, enum fi_wait_obj wait_obj)
{
	struct fi_cq_attr cq_attr;

	memset(&cq_attr, 0, sizeof(cq_attr));
	cq_attr.size = size;
	cq_attr.flags = flags;
	cq_attr.format = format;
	cq_attr.wait_obj = wait_obj;

	return fi_cq_open(domain, &cq_attr, cq, NULL);
}

static void close_ep(int idx)
{
	FT_CLOSE_FID(res[idx].ep);
	FT_CLOSE_FID(res[idx].av);
	FT_CLOSE_FID(res[idx].txcq);
	FT_CLOSE_FID(res[idx].rxcq);
}

static int open_ep(int idx)
{
	int ret;

	struct ft_res *tres = &res[idx];

	fi->tx_attr->op_flags = 0;
	ret = fi_endpoint(domain, fi, &tres->ep, NULL);
	if (ret) {
		FT_UNIT_STRERR(err_buf, "fi_endpoint failed when opening new EP", ret);
		return ret;
	}

	ret = create_av(&tres->av, FI_AV_UNSPEC, 0, fi->domain_attr->ep_cnt,
			fi->domain_attr->ep_cnt, NULL, NULL, 0);
	if (ret)
		goto out_close_ep;
	ret = create_cq(&tres->txcq, fi->tx_attr->size, 0, FI_CQ_FORMAT_UNSPEC, FI_WAIT_NONE);
	if (ret)
		goto out_close_ep;
	ret = create_cq(&tres->rxcq, fi->rx_attr->size, 0, FI_CQ_FORMAT_UNSPEC, FI_WAIT_NONE);
	if (ret)
		goto out_close_ep;

	ret = ft_enable_ep(tres->ep, NULL, tres->av, tres->txcq, tres->rxcq, NULL, NULL);
	if (ret) {
		FT_UNIT_STRERR(err_buf, "fi_enable_ep failed when opening new EP", ret);
		goto out_close_ep;
	}

	return 0;

out_close_ep:
	close_ep(idx);
	return ret;
}

struct ep_sizes_pair {
	size_t tx_size;
	size_t rx_size;
};

static int ep_open_sizes(void)
{
	size_t size, max_size;
	size_t max_tx_size = fi->tx_attr->size, max_rx_size = fi->rx_attr->size;
	int testret = FAIL;
	int i, ret, ep_idx = 0;
	struct ep_sizes_pair expected_success[] = {
			/* zero sized QP it not yet supported
			 * {0, max_rx_size},
			 * {max_tx_size, 0},
			 */
			{max_tx_size, max_rx_size}
	};
	struct ep_sizes_pair expected_failure[] = {
			{max_tx_size + 1, max_rx_size},
			{max_tx_size, max_rx_size + 1}
	};

	max_size = MAX(max_tx_size, max_rx_size);
	for (size = 1; size <= max_size; size += EP_TEST_SIZES_STRIDE) {
		fi->tx_attr->size = MIN(size, max_tx_size);
		fi->rx_attr->size = MIN(size, max_rx_size);
		ret = open_ep(ep_idx);
		if (ret) {
			printf("open_ep failed on size tx[%zu]rx[%zu]",
			       fi->tx_attr->size, fi->rx_attr->size);
			goto out;
		}
		close_ep(ep_idx);
	}

	for (i = 0; i < sizeof(expected_success) / sizeof(struct ep_sizes_pair); i++) {
		fi->tx_attr->size = expected_success[i].tx_size;
		fi->rx_attr->size = expected_success[i].rx_size;
		ret = open_ep(ep_idx);
		if (ret) {
			printf("open_ep failed on sizes tx[%zu]rx[%zu]",
			       fi->tx_attr->size, fi->rx_attr->size);
			goto out;
		}
		close_ep(ep_idx);
	}

	for (i = 0; i < sizeof(expected_failure) / sizeof(struct ep_sizes_pair); i++) {
		fi->tx_attr->size = expected_failure[i].tx_size;
		fi->rx_attr->size = expected_failure[i].rx_size;
		ret = open_ep(ep_idx);
		if (!ret) {
			close_ep(ep_idx);
			printf("Expected open_ep to fail on sizes tx[%zu]rx[%zu]. max size: tx[%zu]rx[%zu]\n",
			       fi->tx_attr->size, fi->rx_attr->size,
			       max_tx_size, max_rx_size);
			goto out;
		}
	}

	ret = 0;
	testret = PASS;
out:
	fi->tx_attr->size = max_tx_size;
	fi->rx_attr->size = max_rx_size;
	return TEST_RET_VAL(ret, testret);
}

static int ep_open_close(void)
{
	int i, ret = 0;
	int testret = FAIL;
	unsigned int to_open;

	to_open = num_eps;
	if (to_open > max_eps) {
		printf("\nToo many EPs to open [%u]. Dropped to %u\n", to_open, max_eps);
		to_open = max_eps;
	}

	for (i = 0; i < to_open; ++i) {
		ret = open_ep(i);
		if (ret) {
			printf(" Failed at %d\n", i);
			goto out;
		}
		close_ep(i);
	}
	testret = PASS;
out:
	return TEST_RET_VAL(ret, testret);
}

static int ep_open_max_simultaneously(void)
{
	int i, ret = 0;
	int testret = PASS;

	for (i = 0; i < max_eps; ++i) {
		ret = open_ep(i);
		if (ret)
			break;
	}
	if (i == 0 || i > max_eps) {
		testret = FAIL;
		printf("Expected[%d] Opened[%d]\n", max_eps, i);
	}

	for (i = i - 1; i >= 0; --i)
		close_ep(i);

	return TEST_RET_VAL(ret, testret);
}

static int ep_rand_access(void)
{
	int ret = 0;
	int testret = FAIL;
	unsigned int i, opened_count, closed_count;
	unsigned int num_actions = num_eps;
	int *open_res_arr;

	if (seed == -1)
		seed = time(NULL);
	srand(seed);

	open_res_arr = calloc(max_eps, sizeof(*open_res_arr));
	if (!open_res_arr) {
		sprintf(err_buf, "Out of memory");
		return FAIL;
	}

	opened_count = closed_count = 0;
	for (i = 0; i < num_actions; ++i) {
		int idx = rand() % max_eps;

		if (!open_res_arr[idx]) {
			++opened_count;
			ret = open_ep(idx);
			if (ret) {
				printf("seed = %d\n", seed);
				goto out;
			}
			open_res_arr[idx] = 1;
		} else {
			++closed_count;
			close_ep(idx);
			open_res_arr[idx] = 0;
		}
	}
	testret = PASS;
out:
	for (i = 0; i < max_eps; ++i) {
		close_ep(i);
		open_res_arr[i] = 0;
	}
	free(open_res_arr);
	return TEST_RET_VAL(ret, testret);
}

static int ep_check_flags(void)
{
	int ret, testret = FAIL;
	int ep_idx = 1;
	uint64_t flags;
	struct fid_ep *ep;

	ret = open_ep(ep_idx);
	if (ret)
		goto out;

	ep = res[ep_idx].ep;
	flags = FI_TRANSMIT | FI_TRANSMIT_COMPLETE;
	ret = fi_control(&ep->fid, FI_SETOPSFLAG, &flags);
	if (ret) {
		printf("set tx flag FI_TRANSMIT | FI_TRANSMIT_COMPLETE failed!\n");
		goto out;
	}

	ret = fi_control(&ep->fid, FI_GETOPSFLAG, &flags);
	if (ret || flags != FI_TRANSMIT_COMPLETE) {
		printf("get result of set tx flag FI_TRANSMIT_COMPLETE failed!\n");
		goto out;
	}

	flags = FI_RECV | FI_COMMIT_COMPLETE;
	ret = fi_control(&ep->fid, FI_SETOPSFLAG, &flags);
	if (ret) {
		printf("set rx flag FI_RECV | FI_COMMIT_COMPLETE failed!\n");
		goto out;
	}

	ret = fi_control(&ep->fid, FI_GETOPSFLAG, &flags);
	if (ret || flags != FI_COMMIT_COMPLETE) {
		printf("get result of rx flag FI_COMMIT_COMPLETE failed!\n");
		goto out;
	}

	testret = PASS;
out:
	close_ep(ep_idx);
	return TEST_RET_VAL(ret, testret);
}

struct test_entry test_array[] = {
	TEST_ENTRY(ep_open_close,
		   "Test open and close EPs"),
	TEST_ENTRY(ep_open_max_simultaneously,
		   "Test opening maximum number of EPs at the same time"),
	TEST_ENTRY(ep_rand_access,
		   "Test opening and closing EPs randomly"),
	TEST_ENTRY(ep_open_sizes,
		   "Test opening and closing EPs with different context sizes"),
	TEST_ENTRY(ep_check_flags,
		   "Test get and set EP operation flags"),
	{ NULL, "" }
};

static void usage(void)
{
	ft_unit_usage("ep_test", "Unit test for end point (EP)");
	FT_PRINT_OPTS_USAGE("-n <num>", "number of EPs to open");
	FT_PRINT_OPTS_USAGE("-z <seed>", "seeds the random number generator");
}

int main(int argc, char **argv)
{
	int op, ret;
	int failed;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	opts = INIT_OPTS;
	opts.options |= FT_OPT_SIZE;
	opts.options |= FT_OPT_SKIP_REG_MR;

	while ((op = getopt(argc, argv, INFO_OPTS "hn:z:")) != -1) {
		switch (op) {
		default:
			ft_parseinfo(op, optarg, hints, &opts);
			break;
		case 'n':
			num_eps = atoi(optarg);
			break;
		case 'z':
			seed = atoi(optarg);
			break;
		case '?':
		case 'h':
			usage();
			return EXIT_FAILURE;

		}
	}

	hints->domain_attr->mr_mode = opts.mr_mode;

	ret = fi_getinfo(FT_FIVERSION, opts.src_addr, 0, 0, hints, &fi);
	if (ret) {
		FT_PRINTERR("fi_getinfo", ret);
		goto err;
	}

	ret = ft_open_fabric_res();
	if (ret)
		goto err;

	max_eps = fi->domain_attr->ep_cnt;
	res = calloc(max_eps + 1, sizeof(struct ft_res));
	if (!res) {
		ret = -FI_ENOMEM;
		goto err;
	}

	printf("Testing EPs on fabric %s\n", fi->fabric_attr->name);

	failed = run_tests(test_array, err_buf);
	if (failed > 0)
		printf("\nSummary: %d tests failed\n", failed);
	else
		printf("\nSummary: all tests passed\n");

err:
	ft_free_res();
	return ret ? ft_exit_code(ret) : (failed > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
