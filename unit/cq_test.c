/*
 * Copyright (c) 2013-2016 Intel Corporation.  All rights reserved.
 * Copyright (c) 2014-2016 Cisco Systems, Inc.  All rights reserved.
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
#include <poll.h>
#include <time.h>
#include <string.h>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>

#include "unit_common.h"
#include "shared.h"


static char err_buf[512];

static int
create_cq(struct fid_cq **cq, size_t size, uint64_t flags,
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

/* Try opening fi->domain_attr->cq_cnt number of completion queues
 * simultaneously using a size hint of 0 (indicating the provider should choose
 * the size)
 */
static int cq_open_close_simultaneous(void)
{
	int ret;
	int opened;
	size_t count;
	int testret = FAIL;
	struct fid_cq **cq_array;

	count = fi->domain_attr->cq_cnt;
	FT_DEBUG("testing creation of up to %zu simultaneous CQs\n", count);

	cq_array = calloc(count, sizeof(*cq_array));
	if (!cq_array)
		return -FI_ENOMEM;

	ret = 0;
	for (opened = 0; opened < count && !ret; opened++) {
		ret = create_cq(&cq_array[opened], 0, 0, FI_CQ_FORMAT_UNSPEC,
				FI_WAIT_UNSPEC);
	}
	if (ret) {
		FT_WARN("fi_cq_open failed after %d (cq_cnt: %d): %s",
			opened, count, fi_strerror(-ret));
	}

	testret = PASS;

	FT_CLOSEV_FID(cq_array, opened);
	free(cq_array);

	return TEST_RET_VAL(ret, testret);
}

/*
 * Tests:
 * - test open and close of CQ over a range of sizes
 */
static int
cq_open_close_sizes()
{
	int i;
	int ret;
	int size;
	int testret;
	struct fid_cq *cq;

	testret = FAIL;

	for (i = -1; i < 17; ++i) {
		size = (i < 0) ? 0 : 1 << i;

		ret = create_cq(&cq, size, 0, FI_CQ_FORMAT_UNSPEC, FI_WAIT_UNSPEC);
		if (ret == -FI_EINVAL) {
			FT_WARN("\nSuccessfully completed %d iterations up to "
				"size %d before the provider returned "
				"EINVAL...",
				i + 1, size >> 1);
			ret = 0;
			goto pass;
		}
		if (ret != 0) {
			sprintf(err_buf, "fi_cq_open(%d, 0, FI_CQ_FORMAT_UNSPEC, "
					"FI_WAIT_UNSPEC) = %d, %s",
					size, ret, fi_strerror(-ret));
			goto fail;
		}

		ret = fi_close(&cq->fid);
		if (ret != 0) {
			sprintf(err_buf, "close(cq) = %d, %s", ret, fi_strerror(-ret));
			goto fail;
		}
		cq = NULL;
	}

pass:
	testret = PASS;
fail:
	cq = NULL;
	return TEST_RET_VAL(ret, testret);
}

struct test_entry test_array[] = {
	TEST_ENTRY(cq_open_close_sizes),
	TEST_ENTRY(cq_open_close_simultaneous),
	{ NULL, "" }
};

static void usage(void)
{
	ft_unit_usage("cq_test", "Unit test for Completion Queue (CQ)");
}

int main(int argc, char **argv)
{
	int op, ret;
	int failed;

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

	ret = fi_getinfo(FT_FIVERSION, NULL, 0, 0, hints, &fi);
	if (ret) {
		FT_PRINTERR("fi_getinfo", ret);
		goto err;
	}

	ret = ft_open_fabric_res();
	if (ret)
		goto err;

	printf("Testing CQs on fabric %s\n", fi->fabric_attr->name);

	failed = run_tests(test_array, err_buf);
	if (failed > 0) {
		printf("Summary: %d tests failed\n", failed);
	} else {
		printf("Summary: all tests passed\n");
	}

err:
	ft_free_res();
	return ret ? -ret : (failed > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
