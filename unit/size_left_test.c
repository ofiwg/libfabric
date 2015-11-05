/*
 * Copyright (c) 2015 Cisco Systems, Inc.  All rights reserved.
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

#include <assert.h>
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


#define DEBUG(...) \
	if (fabtests_debug) { \
		fprintf(stderr, __VA_ARGS__); \
	}

int fabtests_debug = 0;

static char err_buf[512];


static void teardown_ep_fixture(void)
{
	FT_CLOSE_FID(mr);
	FT_CLOSE_FID(ep);
	FT_CLOSE_FID(txcq);
	FT_CLOSE_FID(rxcq);
	FT_CLOSE_FID(av);
	if (buf) {
		free(buf);
		buf = rx_buf = tx_buf = NULL;
		buf_size = rx_size = tx_size = 0;
	}
}

/* returns 0 on success or a negative value that can be stringified with
 * fi_strerror on error */
static int setup_ep_fixture(void)
{
	int ret;

	ret = ft_alloc_active_res(fi);
	if (ret)
		return ret;

	ret = ft_init_ep();
	if (ret)
		return ret;

	return 0;
}

static int
rx_size_left(void)
{
	int ret;
	int testret;

	testret = FAIL;

	ret = setup_ep_fixture();
	if (ret != 0) {
		printf("failed to setup test fixture: %s\n", fi_strerror(-ret));
		goto fail;
	}

	ret = fi_rx_size_left(ep);
	if (ret < 0) {
		printf("fi_rx_size_left returned %d (-%s)\n", ret,
			fi_strerror(-ret));
		goto fail;
	}

	/* TODO: add basic sanity checking here */

	testret = PASS;
fail:
	teardown_ep_fixture();
	return testret;
}

static int
rx_size_left_err(void)
{
	int ret;
	int testret;

	testret = FAIL;

	/* datapath operation, not expected to be caught by libfabric */
#if 0
	ret = fi_rx_size_left(NULL);
	if (ret != -FI_EINVAL) {
		goto fail;
	}
#endif

	ret = fi_endpoint(domain, fi, &ep, NULL);
	if (ret != 0) {
		printf("fi_endpoint %s\n", fi_strerror(-ret));
		goto fail;
	}

	/* ep starts in a non-enabled state, may fail, should not SEGV */
	fi_rx_size_left(ep);

	testret = PASS;
fail:
	FT_CLOSE_FID(ep);
	return testret;
}

static int
tx_size_left(void)
{
	int ret;
	int testret;

	testret = FAIL;

	ret = setup_ep_fixture();
	if (ret != 0) {
		printf("failed to setup test fixture: %s\n", fi_strerror(-ret));
		goto fail;
	}

	ret = fi_tx_size_left(ep);
	if (ret < 0) {
		printf("fi_rx_size_left returned %d (-%s)\n", ret,
			fi_strerror(-ret));
		goto fail;
	}

	/* TODO: once fi_tx_attr's size field meaning has been fixed to refer to
	 * queue depth instead of number of bytes, we can do a little basic
	 * sanity checking here */

	testret = PASS;
fail:
	teardown_ep_fixture();
	return testret;
}

struct test_entry test_rx_size_left[] = {
	TEST_ENTRY(rx_size_left_err),
	TEST_ENTRY(rx_size_left),
	{ NULL, "" }
};

struct test_entry test_tx_size_left[] = {
	TEST_ENTRY(tx_size_left),
	{ NULL, "" }
};

/* TODO: Rewrite test to use size_left() during data transfers and check
 * that posted sends complete when indicated and check for proper failures
 * when size_left() returns 0.
 */
int run_test_set(void)
{
	int failed;

	failed = 0;
	failed += run_tests(test_rx_size_left, err_buf);
	failed += run_tests(test_tx_size_left, err_buf);

	return failed;
}

int main(int argc, char **argv)
{
	int op, ret;
	int failed;
	char *debug_str;

	opts = INIT_OPTS;
	opts.options |= FT_OPT_SIZE;

	debug_str = getenv("FABTESTS_DEBUG");
	if (debug_str) {
		fabtests_debug = atoi(debug_str);
	}

	hints = fi_allocinfo();
	if (!hints)
		exit(1);

	while ((op = getopt(argc, argv, "f:a:")) != -1) {
		switch (op) {
		case 'a':
			free(hints->fabric_attr->name);
			hints->fabric_attr->name = strdup(optarg);
			break;
		case 'f':
			free(hints->fabric_attr->prov_name);
			hints->fabric_attr->prov_name = strdup(optarg);
			break;
		default:
			printf("usage: %s\n", argv[0]);
			printf("\t[-a fabric_name]\n");
			printf("\t[-f provider_name]\n");
			exit(1);
		}
	}

	hints->mode = ~0;
	hints->ep_attr->type = FI_EP_RDM;

	ret = fi_getinfo(FT_FIVERSION, NULL, 0, 0, hints, &fi);
	if (ret != 0) {
		printf("fi_getinfo %s\n", fi_strerror(-ret));
		exit(-ret);
	}

	DEBUG("using provider \"%s\" and fabric \"%s\"\n",
		fi->fabric_attr->prov_name,
		fi->fabric_attr->name);

	ret = ft_open_fabric_res();
	if (ret)
		return ret;

	failed = run_test_set();

	if (failed > 0) {
		printf("Summary: %d tests failed\n", failed);
	} else {
		printf("Summary: all tests passed\n");
	}

	ft_free_res();
	return (failed > 0);
}
