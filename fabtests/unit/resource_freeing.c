/*
 * Copyright (c) 2017 Intel Corporation. All rights reserved.
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
#include "shared.h"

#define lengthof(arr) (sizeof(arr) / sizeof(*arr))

enum test_depth {
	DEPTH_FABRIC,
	DEPTH_DOMAIN,
	DEPTH_ENABLE_ENDPOINT
};

int test_resource_freeing(enum test_depth test_depth,
		const char *fabric_service)
{
	int our_ret = FI_SUCCESS;
	int ret;
	uint64_t flags;
	struct fi_info *info;

	/* Setup fabric */

	hints = fi_allocinfo();
	if (!hints) {
		our_ret = -FI_ENOMEM;
		goto error_return;
	}

	flags = FI_SOURCE;
	hints->caps = FI_RMA;
	hints->ep_attr->type = FI_EP_RDM;

	ret = fi_getinfo(FT_FIVERSION, NULL, fabric_service, flags,
			 hints, &info);
	if (ret) {
		FT_PRINTERR("fi_getinfo", ret);
		our_ret = ret;
		goto free_hints;
	}

	ret = fi_fabric(info->fabric_attr, &fabric, NULL);
	if (ret) {
		FT_PRINTERR("fi_fabric", ret);
		our_ret = ret;
		goto free_info;
	}

	if (test_depth == DEPTH_FABRIC) {
		goto close_fabric;
	}

	ret = fi_domain(fabric, info, &domain, NULL);
	if (ret) {
		FT_PRINTERR("fi_domain", ret);
		our_ret = ret;
		goto close_fabric;
	}

	if (test_depth == DEPTH_DOMAIN) {
		goto close_domain;
	}

	/* Create pre-endpoint resources */

	av_attr.type = info->domain_attr->av_type;
	av_attr.count = 0;
	av_attr.name = NULL;
	ret = fi_av_open(domain, &av_attr, &av, NULL);
	if (ret) {
		FT_PRINTERR("fi_av_open", ret);
		our_ret = ret;
		goto close_domain;
	}

	cntr_attr.events = FI_CNTR_EVENTS_COMP;
	cntr_attr.wait_obj = FI_WAIT_UNSPEC;
	ret = fi_cntr_open(domain, &cntr_attr, &txcntr, NULL);
	if (ret) {
		FT_PRINTERR("fi_cntr_open", ret);
		our_ret = ret;
		goto close_av;
	}

	ret = fi_cq_open(domain, &cq_attr, &txcq, NULL);
	if (ret) {
		FT_PRINTERR("fi_cq_open", ret);
		our_ret = ret;
		goto close_txcntr;
	}

	ret = fi_endpoint(domain, info, &ep, NULL);
	if (ret) {
		FT_PRINTERR("fi_endpoint", ret);
		our_ret = ret;
		goto close_txcq;
	}

	/* Bind pre-endpoint resources to ep */

	ret = fi_ep_bind(ep, &txcntr->fid, FI_WRITE);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		our_ret = ret;
		goto close_ep;
	}

	ret = fi_ep_bind(ep, &av->fid, 0);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		our_ret = ret;
		goto close_ep;
	}

	ret = fi_ep_bind(ep, &txcq->fid, FI_TRANSMIT);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		our_ret = ret;
		goto close_ep;
	}

	/* Enable ep */

	ret = fi_enable(ep);
	if (ret) {
		FT_PRINTERR("fi_enable", ret);
		our_ret = ret;
		goto close_ep;
	}

	if (test_depth == DEPTH_ENABLE_ENDPOINT) {
		goto close_ep;
	}

close_ep:
	ret = fi_close(&ep->fid);
	if (ret) {
		FT_PRINTERR("fi_close", ret);
		our_ret = our_ret ? our_ret : ret;
	}

close_txcq:
	ret = fi_close(&txcq->fid);
	if (ret) {
		FT_PRINTERR("fi_close", ret);
		our_ret = our_ret ? our_ret : ret;
	}

close_txcntr:
	ret = fi_close(&txcntr->fid);
	if (ret) {
		FT_PRINTERR("fi_close", ret);
		our_ret = our_ret ? our_ret : ret;
	}

close_av:
	ret = fi_close(&av->fid);
	if (ret) {
		FT_PRINTERR("fi_close", ret);
		our_ret = our_ret ? our_ret : ret;
	}

close_domain:
	ret = fi_close(&domain->fid);
	if (ret) {
		FT_PRINTERR("fi_close", ret);
		our_ret = our_ret ? our_ret : ret;
	}

close_fabric:
	ret = fi_close(&fabric->fid);
	if (ret) {
		FT_PRINTERR("fi_close", ret);
		our_ret = our_ret ? our_ret : ret;
	}

free_info:
	fi_freeinfo(info);

free_hints:
	fi_freeinfo(hints);

error_return:
	return our_ret;
}

void print_test_resource_freeing_call(enum test_depth test_depth, int iter)
{
	fprintf(stdout,
		"Running test_resource_freeing with "
		"[%s] for %d iterations\n",
		(test_depth == DEPTH_FABRIC) ? "DEPTH_FABRIC"
		: (test_depth == DEPTH_DOMAIN) ? "DEPTH_DOMAIN"
		: (test_depth == DEPTH_ENABLE_ENDPOINT) ? "DEPTH_ENABLE_ENDPOINT"
		: "(unknown test depth)",
		iter
	);

	fflush(stderr);
	fflush(stdout);
}

void print_test_resource_freeing_result_call(int success,
		enum test_depth test_depth,
		int iter)
{
	fprintf(success ? stdout : stderr,
		"%s: test_resource_freeing %s with "
		"[%s]\n",
		success ? "GOOD" : "ERROR",
		success ? "succeeded" : "failed",
		(test_depth == DEPTH_FABRIC) ? "DEPTH_FABRIC"
		: (test_depth == DEPTH_DOMAIN) ? "DEPTH_DOMAIN"
		: (test_depth == DEPTH_ENABLE_ENDPOINT) ? "DEPTH_ENABLE_ENDPOINT"
		: "(unknown test depth)"
	);

	fflush(stderr);
	fflush(stdout);
}

int main(int argc, char **argv)
{
	int op, i, td_idx, ret = 0, iters = 2, exit_code = 0;

	opts = INIT_OPTS;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt(argc, argv, "i:h" ADDR_OPTS INFO_OPTS)) != -1) {
		switch (op) {
		default:
			ft_parse_addr_opts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints, &opts);
			break;
		case 'i':
			iters = atoi(optarg);
			break;
		case '?':
		case 'h':
			ft_usage(argv[0], "Test which exercises resource freeing in a provider\n");
			FT_PRINT_OPTS_USAGE("-i <int>", "number of iterations to test");
			return EXIT_FAILURE;
		}
	}

	enum test_depth test_depth[] = {
		DEPTH_FABRIC, DEPTH_DOMAIN, DEPTH_ENABLE_ENDPOINT};

	for (td_idx = 0; td_idx < lengthof(test_depth); td_idx += 1) {
		print_test_resource_freeing_call(
			test_depth[td_idx], iters);
		for (i = 0; i < iters; i += 1) {
			ret = test_resource_freeing(
				test_depth[td_idx], default_port);
			if (ret) {
				exit_code = EXIT_FAILURE;
				break;
			}
		}
		print_test_resource_freeing_result_call(
			!ret, /* int success */
			test_depth[td_idx],
			i);
	}

	return ft_exit_code(exit_code);
}
