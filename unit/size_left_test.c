/*
 * Copyright (c) 2015-2016 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2015 Intel Corporation.  All rights reserved.
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
#include <errno.h>
#include <getopt.h>
#include <string.h>
#include <sys/types.h>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>

#include "shared.h"
#include "unit_common.h"

static char err_buf[512];

/* Need to define a separate setup function from the ft_* functions defined in
 * common/shared.c, this is because extra flexibility is needed in this test.
 * The ft_* functions have a couple of incompatibility issues with this test:
 *
 * - ft_free_res always frees the hints
 * - ft_init_ep posts a recv
 */
static void setup_ep_fixture(void)
{
	int ret;

	ret = fi_fabric(fi->fabric_attr, &fabric, NULL);
	if (ret) {
		FT_PRINTERR("fi_fabric", ret);
		exit(EXIT_FAILURE);
	}

	ret = fi_domain(fabric, fi, &domain, NULL);
	if (ret) {
		FT_PRINTERR("fi_domain", ret);
		exit(EXIT_FAILURE);
	}

	/* Queues are opened for providers that need a queue bound before
	 * calling fi_enable. This avoids getting back -FI_ENOCQ.
	 */
	cq_attr.format = FI_CQ_FORMAT_CONTEXT;
	cq_attr.wait_obj = FI_WAIT_NONE;

	ret = fi_cq_open(domain, &cq_attr, &txcq, &txcq);
	if (ret) {
		FT_PRINTERR("fi_cq_open", ret);
		exit(EXIT_FAILURE);
	}

	ret = fi_cq_open(domain, &cq_attr, &rxcq, &rxcq);
	if (ret) {
		FT_PRINTERR("fi_cq_open", ret);
		exit(EXIT_FAILURE);
	}

	ret = fi_endpoint(domain, fi, &ep, NULL);
	if (ret) {
		FT_PRINTERR("fi_endpoint", ret);
		exit(EXIT_FAILURE);
	}

	/* Some providers require that an endpoint be bound to an AV before
	 * calling fi_enable on unconnected endpoints.
	 */
	if (fi->ep_attr->type == FI_EP_RDM ||
	    fi->ep_attr->type == FI_EP_DGRAM) {
		ret = fi_av_open(domain, &av_attr, &av, NULL);
		if (ret) {
			FT_PRINTERR("fi_av_open", ret);;
			exit(EXIT_FAILURE);
		}

		ret = fi_ep_bind(ep, &av->fid, 0);
		if (ret) {
			FT_PRINTERR("fi_ep_bind", ret);
			exit(EXIT_FAILURE);
		}
	}

	/* MSG endpoints require EQ to be bound to an endpoint */
	if (fi->ep_attr->type == FI_EP_MSG) {
		ret = fi_eq_open(fabric, &eq_attr, &eq, NULL);
		if (ret) {
			FT_PRINTERR("fi_eq_open", ret);
			exit(EXIT_FAILURE);
		}
		ret = fi_ep_bind(ep, &eq->fid, 0);
		if (ret) {
			FT_PRINTERR("fi_ep_bind", ret);
			exit(EXIT_FAILURE);
		}
	}

	ret = fi_ep_bind(ep, &txcq->fid, FI_TRANSMIT);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		exit(EXIT_FAILURE);
	}

	ret = fi_ep_bind(ep, &rxcq->fid, FI_RECV);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		exit(EXIT_FAILURE);
	}
}

/* Teardown after every test. */
static void teardown_ep_fixture(void)
{
	FT_CLOSE_FID(ep);
	FT_CLOSE_FID(av);
	FT_CLOSE_FID(rxcq);
	FT_CLOSE_FID(txcq);
	FT_CLOSE_FID(domain);
	FT_CLOSE_FID(eq);
	FT_CLOSE_FID(fabric);
}

/* Test that a provider returns -FI_EOPBADSTATE before an endpoint has been
 * enabled.
 */
static int test_size_left_bad(void)
{
	ssize_t ret;
	int testret;

	FT_DEBUG("testing fi_rx_size_left for -FI_EOPBADSTATE");
	ret = fi_rx_size_left(ep);
	if (ret != -FI_EOPBADSTATE)
		goto fail;

	FT_DEBUG("testing fi_tx_size_left for -FI_EOPBADSTATE");
	ret = fi_tx_size_left(ep);
	if (ret != -FI_EOPBADSTATE)
		goto fail;

	return PASS;

fail:
	testret = TEST_RET_VAL(ret, FAIL);
	if (testret == SKIPPED)
		snprintf(err_buf, sizeof(err_buf),
			 "provider returned: [%zd]: <%s>", ret,
			 fi_strerror(-ret));
	else
		snprintf(err_buf, sizeof(err_buf),
			 "%zd not equal to -FI_EOPBADSTATE", ret);

	return testret;
}

/* Test that the initial sizes are equal to the size attribute for the tx and rx
 * context.
 */
static int test_size_left_good(void)
{
	ssize_t expected;
	ssize_t ret;
	int testret;

	FT_DEBUG("testing fi_rx_size_left for size equal to or greater than fi->rx_attr->size");
	expected = fi->rx_attr->size;

	ret = fi_rx_size_left(ep);
	if (ret < expected)
		goto fail;

	FT_DEBUG("testing fi_tx_size_left for size equal to or greater than fi->tx_attr->size");
	expected = fi->tx_attr->size;

	ret = fi_tx_size_left(ep);
	if (ret < expected)
		goto fail;

	return PASS;

fail:
	testret = TEST_RET_VAL(ret, FAIL);
	if (testret == SKIPPED)
		snprintf(err_buf, sizeof(err_buf),
			 "provider returned: [%zd]: <%s>", ret,
			 fi_strerror(-ret));
	else
		snprintf(err_buf, sizeof(err_buf),
			 "%zd not greater than or equal to %zd", ret, expected);

	return testret;
}

/* Separate these into separate entries since one of them needs to get run
 * before the endpoint is enabled, and one needs to be run after the endpoint is
 * enabled.
 */
struct test_entry bad_tests[] = {
	TEST_ENTRY(test_size_left_bad),
	{ NULL, "" }
};

struct test_entry good_tests[] = {
	TEST_ENTRY(test_size_left_good),
	{ NULL, "" }
};

int run_test_set(struct fi_info *hints)
{
	struct fi_info *info;
	int failed = 0;
	int ep_type;
	int ret;

	ret = fi_getinfo(FT_FIVERSION, NULL, 0, 0, hints, &info);
	if (ret) {
		FT_PRINTERR("fi_getinfo", ret);
		exit(EXIT_FAILURE);
	}

	for (ep_type = FI_EP_MSG; ep_type <= FI_EP_RDM; ep_type++) {
		for (fi = info; fi; fi = fi->next) {
			if (fi->ep_attr->type == ep_type)
				break;
		}

		if (!fi)
			continue;

		setup_ep_fixture();

		printf("Testing provider %s on fabric %s with EP type %s\n",
				fi->fabric_attr->prov_name,
				fi->fabric_attr->name,
				fi_tostr(&ep_type, FI_TYPE_EP_TYPE));

		failed += run_tests(bad_tests, err_buf);

		ret = fi_enable(ep);
		if (ret) {
			FT_PRINTERR("fi_enable", ret);
			exit(EXIT_FAILURE);
		}

		failed += run_tests(good_tests, err_buf);

		teardown_ep_fixture();
	}

	fi_freeinfo(info);

	return failed;
}

static void usage(void)
{
	ft_unit_usage("size_left_test", "Unit test for checking TX and RX context sizes");
}

int main(int argc, char **argv)
{
	int op;
	int failed;

	opts = INIT_OPTS;
	opts.options |= FT_OPT_SIZE;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt(argc, argv, "f:a:h")) != -1) {
		switch (op) {
		case 'a':
			free(hints->fabric_attr->name);
			hints->fabric_attr->name = strdup(optarg);
			break;
		case 'f':
			free(hints->fabric_attr->prov_name);
			hints->fabric_attr->prov_name = strdup(optarg);
			break;
		case 'h':
			usage();
			return EXIT_SUCCESS;
		default:
			usage();
			return EXIT_FAILURE;
		}
	}

	hints->mode = ~0;
	hints->caps = FI_MSG;

	failed = run_test_set(hints);

	if (failed)
		printf("Summary: %d tests failed\n", failed);
	else
		printf("Summary: all tests passed!\n");

	fi_freeinfo(hints);

	return (failed > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
