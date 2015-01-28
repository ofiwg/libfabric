/*
 * Copyright (c) 2015 Cisco Systems, Inc.  All rights reserved.
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

#define RX_CQ_DEPTH (128)
#define TX_CQ_DEPTH (128)

int fabtests_debug = 0;

struct fi_info hints;
struct fi_tx_attr tx_attr;
struct fi_rx_attr rx_attr;
static struct fi_fabric_attr fabric_hints;
static struct fi_eq_attr eq_attr;

static struct fi_info *fi;
static struct fid_fabric *fabric;
static struct fid_domain *domain;
static struct fid_eq *eq;

static char err_buf[512];

/* per-test fixture variables */
static struct fid_cq *wcq = NULL;
static struct fid_cq *rcq = NULL;
static struct fid_av *av = NULL;

/* returns 0 on success or a negative value that can be stringified with
 * fi_strerror on error.
 *
 * Idempotent. */
static int teardown_ep_fixture(struct fid_ep *ep)
{
	int teardown_ret;
	int ret;

	teardown_ret = 0;

	if (ep != NULL) {
		ret = fi_close(&ep->fid);
		if (ret != 0) {
			printf("fi_close(ep) %s\n", fi_strerror(-ret));
			teardown_ret = ret;
		}
	}
	if (wcq != NULL) {
		ret = fi_close(&rcq->fid);
		if (ret != 0) {
			printf("fi_close(rcq) %s\n", fi_strerror(-ret));
			teardown_ret = ret;
		}
	}
	if (rcq != NULL) {
		ret = fi_close(&wcq->fid);
		if (ret != 0) {
			printf("fi_close(wcq) %s\n", fi_strerror(-ret));
			teardown_ret = ret;
		}
	}
	if (av != NULL) {
		ret = fi_close(&av->fid);
		if (ret != 0) {
			printf("fi_close(av) %s\n", fi_strerror(-ret));
			teardown_ret = ret;
		}
	}

	return teardown_ret;
}

/* returns 0 on success or a negative value that can be stringified with
 * fi_strerror on error */
static int setup_ep_fixture(struct fid_ep **ep_o)
{
	int ret;
	struct fi_info *myfi;
	struct fi_av_attr av_attr;
	struct fi_cq_attr cq_attr;

	assert(ep_o != NULL);
	ret = 0;

	myfi = fi_dupinfo(fi);
	if (myfi == NULL) {
		printf("fi_dupinfo returned NULL\n");
		goto fail;
	}

	ret = fi_endpoint(domain, myfi, ep_o, NULL);
	if (ret != 0) {
		printf("fi_endpoint %s\n", fi_strerror(-ret));
		goto fail;
	}

	memset(&cq_attr, 0, sizeof cq_attr);
	cq_attr.format = FI_CQ_FORMAT_CONTEXT;
	cq_attr.wait_obj = FI_WAIT_NONE;
	cq_attr.size = TX_CQ_DEPTH;

	ret = fi_cq_open(domain, &cq_attr, &wcq, /*context=*/NULL);
	if (ret != 0) {
		printf("fi_cq_open %s\n", fi_strerror(-ret));
		goto fail;
	}

	memset(&cq_attr, 0, sizeof cq_attr);
	cq_attr.format = FI_CQ_FORMAT_CONTEXT;
	cq_attr.wait_obj = FI_WAIT_NONE;
	cq_attr.size = RX_CQ_DEPTH;

	ret = fi_cq_open(domain, &cq_attr, &rcq, /*context=*/NULL);
	if (ret != 0) {
		printf("fi_cq_open %s\n", fi_strerror(-ret));
		goto fail;
	}

	memset(&av_attr, 0, sizeof av_attr);
	av_attr.type = FI_AV_MAP;
	av_attr.count = 1;
	av_attr.name = NULL;

	ret = fi_av_open(domain, &av_attr, &av, NULL);
	if (ret != 0) {
		printf("fi_av_open %s\n", fi_strerror(-ret));
		goto fail;
	}

	ret = fi_ep_bind(*ep_o, &wcq->fid, FI_SEND);
	if (ret != 0) {
		printf("fi_ep_bind(wcq) %s\n", fi_strerror(-ret));
		goto fail;
	}

	ret = fi_ep_bind(*ep_o, &rcq->fid, FI_RECV);
	if (ret != 0) {
		printf("fi_ep_bind(rcq) %s\n", fi_strerror(-ret));
		goto fail;
	}

	ret = fi_ep_bind(*ep_o, &av->fid, 0);
	if (ret != 0) {
		printf("fi_ep_bind(av) %s\n", fi_strerror(-ret));
		goto fail;
	}

	ret = fi_enable(*ep_o);
	if (ret != 0) {
		printf("fi_enable %s\n", fi_strerror(-ret));
		goto fail;
	}

	if (myfi != NULL) {
		fi_freeinfo(myfi);
	}
	return ret;

fail:
	if (myfi != NULL) {
		fi_freeinfo(myfi);
	}
	return teardown_ep_fixture(*ep_o);
}

static int
rx_size_left(void)
{
	int ret;
	int testret;
	struct fid_ep *ep;

	testret = FAIL;
	ep = NULL;

	ret = setup_ep_fixture(&ep);
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

	/* TODO: once fi_rx_attr's size field meaning has been fixed to refer to
	 * queue depth instead of number of bytes, we can do a little basic
	 * sanity checking here */

	testret = PASS;
fail:
	ret = teardown_ep_fixture(ep);
	if (ret != 0)
		testret = FAIL;
	return testret;
}

static int
rx_size_left_err(void)
{
	int ret;
	int testret;
	struct fid_ep *ep;
	struct fi_info *myfi;

	testret = FAIL;
	ep = NULL;

	myfi = fi_dupinfo(fi);

	/* datapath operation, not expected to be caught by libfabric */
#if 0
	ret = fi_rx_size_left(NULL);
	if (ret != -FI_EINVAL) {
		goto fail;
	}
#endif

	ret = fi_endpoint(domain, myfi, &ep, NULL);
	if (ret != 0) {
		printf("fi_endpoint %s\n", fi_strerror(-ret));
		goto fail;
	}

	/* ep starts in a non-enabled state, may fail, should not SEGV */
	fi_rx_size_left(ep);

	testret = PASS;
fail:
	if (ep != NULL) {
		ret = fi_close(&ep->fid);
		if (ret != 0)
			printf("fi_close %s\n", fi_strerror(-ret));
		ep = NULL;
	}
	if (myfi != NULL) {
		fi_freeinfo(myfi);
	}
	return testret;
}

static int
tx_size_left(void)
{
	int ret;
	int testret;
	struct fid_ep *ep;

	testret = FAIL;
	ep = NULL;

	ret = setup_ep_fixture(&ep);
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
	ret = teardown_ep_fixture(ep);
	if (ret != 0)
		testret = FAIL;
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

	if (getenv("FABTESTS_DEBUG")) {
		fabtests_debug = atoi(getenv("FABTESTS_DEBUG"));
	}

	memset(&hints, 0x00, sizeof(hints));
	memset(&fabric_hints, 0x00, sizeof(fabric_hints));

	while ((op = getopt(argc, argv, "f:p:")) != -1) {
		switch (op) {
		case 'f':
			fabric_hints.name = optarg;
			hints.fabric_attr = &fabric_hints;
			break;
		case 'p':
			fabric_hints.prov_name = optarg;
			hints.fabric_attr = &fabric_hints;
			break;
		default:
			printf("usage: %s\n", argv[0]);
			printf("\t[-f fabric_name]\n");
			printf("\t[-p provider_name]\n");
			exit(1);
		}
	}

	hints.mode = ~0;

	ret = fi_getinfo(FI_VERSION(1, 0), NULL, 0, 0, &hints, &fi);
	if (ret != 0) {
		printf("fi_getinfo %s\n", fi_strerror(-ret));
		exit(1);
	}

	DEBUG("using provider \"%s\" and fabric \"%s\"\n",
		fi->fabric_attr->prov_name,
		fi->fabric_attr->name);

	ret = fi_fabric(fi->fabric_attr, &fabric, NULL);
	if (ret != 0) {
		printf("fi_fabric %s\n", fi_strerror(-ret));
		exit(1);
	}
	ret = fi_domain(fabric, fi, &domain, NULL);
	if (ret != 0) {
		printf("fi_domain %s\n", fi_strerror(-ret));
		exit(1);
	}

	eq_attr.size = 1024;
	eq_attr.wait_obj = FI_WAIT_UNSPEC;
	ret = fi_eq_open(fabric, &eq_attr, &eq, NULL);
	if (ret != 0) {
		printf("fi_eq_open %s\n", fi_strerror(-ret));
		exit(1);
	}

	failed =0;

	failed += run_test_set();

	if (failed > 0) {
		printf("Summary: %d tests failed\n", failed);
	} else {
		printf("Summary: all tests passed\n");
	}

	ret = fi_close(&eq->fid);
	if (ret != 0) {
		printf("Error %d closing EQ: %s\n", ret, fi_strerror(-ret));
		exit(1);
	}
	ret = fi_close(&domain->fid);
	if (ret != 0) {
		printf("Error %d closing domain: %s\n", ret, fi_strerror(-ret));
		exit(1);
	}
	ret = fi_close(&fabric->fid);
	if (ret != 0) {
		printf("Error %d closing fabric: %s\n", ret, fi_strerror(-ret));
		exit(1);
	}
	fi_freeinfo(fi);
	if (ret != 0) {
		printf("Error %d freeing info: %s\n", ret, fi_strerror(-ret));
		exit(1);
	}

	return (failed > 0);
}
