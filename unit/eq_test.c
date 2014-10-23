/*
 * Copyright (c) 2013-2014 Intel Corporation.  All rights reserved.
 * Copyright (c) 2014 Cisco Systems, Inc.  All rights reserved.
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

#include "shared.h"

struct fi_info hints;
static struct fi_fabric_attr fabric_hints;

static struct fid_fabric *fabric;
static struct fid_eq *eq;

static char err_buf[512];

enum { PASS, FAIL, NOTSUPP };

static int
create_eq(size_t size, uint64_t flags, enum fi_wait_obj wait_obj)
{
	struct fi_eq_attr eq_attr;

	memset(&eq_attr, 0, sizeof(eq_attr));
	eq_attr.size = size;
	eq_attr.flags = flags;
	eq_attr.wait_obj = wait_obj;

	return fi_eq_open(fabric, &eq_attr, &eq, NULL);
}

/*
 * Tests:
 * - test open and close of EQ over a range of sizes
 */
static int
eq_open_close()
{
	int i;
	int ret;
	int size;
	int testret;

	testret = FAIL;

	for (i = 0; i < 17; ++i) {
		size = 1 << i;
		ret = create_eq(size, 0, FI_WAIT_UNSPEC);
		if (ret != 0) {
			sprintf(err_buf, "fi_eq_open(%d, 0, FI_WAIT_UNSPEC) = %d, %s",
					size, ret, fi_strerror(-ret));
			goto fail;
		}

		ret = fi_close(&eq->fid);
		if (ret != 0) {
			sprintf(err_buf, "close(eq) = %d, %s", ret, fi_strerror(-ret));
			goto fail;
		}
		eq = NULL;
	}
	testret = PASS;

fail:
	eq = NULL;
	return testret;
}

/*
 * Tests:
 * - writing to EQ
 * - reading from EQ with and without FI_PEEK
 * - underflow read
 */
static int
eq_write_read_self()
{
	struct fi_eq_entry entry;
	uint32_t event;
	int testret;
	int ret;
	int i;

	testret = FAIL;

	ret = create_eq(32, FI_WRITE, FI_WAIT_NONE);
	if (ret != 0) {
		sprintf(err_buf, "fi_eq_open ret=%d, %s", ret, fi_strerror(-ret));
		goto fail;
	}

	/* Insert some events */
	for (i = 0; i < 5; ++i) {
		if (i & 1) {
			entry.fid = &fabric->fid;
		} else {
			entry.fid = &eq->fid;
		}
		entry.context = (void *)(uintptr_t)i;
		ret = fi_eq_write(eq, FI_COMPLETE, &entry, sizeof(entry), 0);
		if (ret != sizeof(entry)) {
			sprintf(err_buf, "fi_eq_write ret=%d, %s", ret, fi_strerror(-ret));
			goto fail;
		}
	}

	/* Now read them back, peeking first at each one */
	for (i = 0; i < 10; ++i) {
		event = ~0;
		memset(&entry, 0, sizeof(entry));
		ret = fi_eq_read(eq, &event, &entry, sizeof(entry),
				(i & 1) ? 0 : FI_PEEK); 
		if (ret != sizeof(entry)) {
			sprintf(err_buf, "fi_eq_read ret=%d, %s", ret, fi_strerror(-ret));
			goto fail;
		}

		if (event != FI_COMPLETE) {
			sprintf(err_buf, "iter %d: event = %d, should be %d\n", i, event,
					FI_COMPLETE);
			goto fail;
		}

		if ((int)(uintptr_t)entry.context != i / 2) {
			sprintf(err_buf, "iter %d: context mismatch %d != %d", i,
					(int)(uintptr_t)entry.context, i / 2);
			goto fail;
		}

		if (entry.fid != ((i & 2) ? &fabric->fid : &eq->fid)) {
			sprintf(err_buf, "iter %d: fid mismatch %p != %p", i,
					entry.fid, ((i & 2) ? &fabric->fid : &eq->fid));
			goto fail;
		}
	}

	/* queue is now empty */
	ret = fi_eq_read(eq, &event, &entry, sizeof(entry), 0);
	if (ret != -FI_EAGAIN) {
		sprintf(err_buf, "fi_eq_read of empty EQ returned %d", ret);
		goto fail;
	}
	testret = PASS;

fail:
	if (eq != NULL) {
		fi_close(&eq->fid);
		eq = NULL;
	}
	return testret;
}

/*
 * Tests:
 * - write overflow
 */
static int
eq_write_overflow()
{
	struct fi_eq_entry entry;
	int testret;
	int ret;
	int i;

	testret = FAIL;

	ret = create_eq(32, FI_WRITE, FI_WAIT_NONE);
	if (ret != 0) {
		sprintf(err_buf, "fi_eq_open ret=%d, %s", ret, fi_strerror(-ret));
		goto fail;
	}

	/* Insert some events */
	for (i = 0; i < 32; ++i) {
		entry.fid = &fabric->fid;
		entry.context = (void *)(uintptr_t)i;
		ret = fi_eq_write(eq, FI_COMPLETE, &entry, sizeof(entry), 0);
		if (ret != sizeof(entry)) {
			sprintf(err_buf, "fi_eq_write ret=%d, %s", ret, fi_strerror(-ret));
			goto fail;
		}
	}

	ret = fi_eq_write(eq, FI_COMPLETE, &entry, sizeof(entry), 0);
	if (ret != -FI_EAGAIN) {
		sprintf(err_buf, "fi_eq_write of full EQ returned %d", ret);
		goto fail;
	}

	testret = PASS;

fail:
	if (eq != NULL) {
		fi_close(&eq->fid);
		eq = NULL;
	}
	return testret;
}

/*
 * Tests:
 * - extracting FD from EQ with FI_WAIT_FD
 * - wait on fd with nothing pending
 * - wait on fd with event pending
 */
static int
eq_wait_fd_poll()
{
	int fd;
	struct fi_eq_entry entry;
	struct pollfd pfd;
	int testret;
	int ret;

	testret = FAIL;

	ret = create_eq(32, FI_WRITE, FI_WAIT_FD);
	if (ret != 0) {
		sprintf(err_buf, "fi_eq_open ret=%d, %s", ret, fi_strerror(-ret));
		goto fail;
	}

	ret = fi_control(&eq->fid, FI_GETWAIT, &fd);
	if (ret != 0) {
		sprintf(err_buf, "fi_control ret=%d, %s", ret, fi_strerror(-ret));
		goto fail;
	}

	/* poll on empty EQ */
	pfd.fd = fd;
	pfd.events = POLLIN;
	ret = poll(&pfd, 1, 0);
	if (ret < 0) {
		sprintf(err_buf, "poll errno=%d, %s", errno, fi_strerror(-errno));
		goto fail;
	}
	if (ret > 0) {
		sprintf(err_buf, "poll returned %d, should be 0", ret);
		goto fail;
	}

	/* write an event */
	entry.fid = &eq->fid;
	entry.context = eq;
	ret = fi_eq_write(eq, FI_COMPLETE, &entry, sizeof(entry), 0);
	if (ret != sizeof(entry)) {
		sprintf(err_buf, "fi_eq_write ret=%d, %s", ret, fi_strerror(-ret));
		goto fail;
	}

	/* poll on EQ with event */
	pfd.fd = fd;
	pfd.events = POLLIN;
	ret = poll(&pfd, 1, 0);
	if (ret < 0) {
		sprintf(err_buf, "poll errno=%d, %s", errno, fi_strerror(-errno));
		goto fail;
	}
	if (ret != 1) {
		sprintf(err_buf, "poll returned %d, should be 1", ret);
		goto fail;
	}

	testret = PASS;
fail:
	if (eq != NULL) {
		fi_close(&eq->fid);
		eq = NULL;
	}
	return testret;
}

/*
 * Tests:
 * - sread with event pending
 * - sread with no event pending
 */
static int
eq_wait_fd_sread()
{
	struct fi_eq_entry entry;
	uint32_t event;
	struct timespec before, after;
	uint64_t elapsed;
	int testret;
	int ret;

	testret = FAIL;

	ret = create_eq(32, FI_WRITE, FI_WAIT_FD);
	if (ret != 0) {
		sprintf(err_buf, "fi_eq_open ret=%d, %s", ret, fi_strerror(-ret));
		goto fail;
	}

	/* timed sread on empty EQ, 2s timeout */
	clock_gettime(CLOCK_MONOTONIC_RAW, &before);
	ret = fi_eq_sread(eq, &event, &entry, sizeof(entry), 2000, 0);
	if (ret != -FI_EAGAIN) {
		sprintf(err_buf, "fi_eq_read of empty EQ returned %d", ret);
		goto fail;
	}

	/* check timeout accuracy */
	clock_gettime(CLOCK_MONOTONIC_RAW, &after);
	elapsed = get_elapsed_ms(&before, &after);
	if (elapsed < 1500 || elapsed > 2500) {
		sprintf(err_buf, "fi_eq_sread slept %d ms, expected 2000",
				(int)elapsed);
		goto fail;
	}

	/* write an event */
	entry.fid = &eq->fid;
	entry.context = eq;
	ret = fi_eq_write(eq, FI_COMPLETE, &entry, sizeof(entry), 0);
	if (ret != sizeof(entry)) {
		sprintf(err_buf, "fi_eq_write ret=%d, %s", ret, fi_strerror(-ret));
		goto fail;
	}

	/* timed sread on EQ with event, 2s timeout */
	clock_gettime(CLOCK_MONOTONIC_RAW, &before);
	event = ~0;
	memset(&entry, 0, sizeof(entry));
	ret = fi_eq_sread(eq, &event, &entry, sizeof(entry), 2000, 0);
	if (ret != sizeof(entry)) {
		sprintf(err_buf, "fi_eq_read ret=%d, %s", ret, fi_strerror(-ret));
		goto fail;
	}

	/* check that no undue waiting occurred */
	clock_gettime(CLOCK_MONOTONIC_RAW, &after);
	elapsed = get_elapsed_ms(&before, &after);
	if (elapsed > 5) {
		sprintf(err_buf, "fi_eq_sread slept %d ms, expected immediate return",
				(int)elapsed);
		goto fail;
	}

	if (event != FI_COMPLETE) {
		sprintf(err_buf, "fi_eq_sread: event = %d, should be %d\n", event,
				FI_COMPLETE);
		goto fail;
	}
	if (entry.fid != &eq->fid) {
		sprintf(err_buf, "fi_eq_sread: fid mismatch: %p should be %p\n",
				entry.fid, &eq->fid);
		goto fail;
	}
	if (entry.context != eq) {
		sprintf(err_buf, "fi_eq_sread: context mismatch: %p should be %p\n",
				entry.context, eq);
		goto fail;
	}

	testret = PASS;
fail:
	if (eq != NULL) {
		fi_close(&eq->fid);
		eq = NULL;
	}
	return testret;
}

#define TEST_ENTRY(NAME) { NAME, #NAME }

struct test_entry {
	int (*test)();
	char *name;
};

struct test_entry test_array[] = {
	TEST_ENTRY(eq_open_close),
	TEST_ENTRY(eq_write_read_self),
	TEST_ENTRY(eq_write_overflow),
	TEST_ENTRY(eq_wait_fd_poll),
	TEST_ENTRY(eq_wait_fd_sread),
	{ NULL, "" }
};

static int
run_tests()
{
	int ret;
	struct test_entry *tep;
	int failed;

	failed = 0;

	tep = test_array;
	while (tep->test != NULL) {
		printf("Running %s...", tep->name);
		fflush(stdout);
		ret = tep->test();
		switch (ret) {
		case PASS:
			printf("PASS!\n");
			break;
		case FAIL:
			printf("FAIL\n");
			printf("  %s\n", err_buf);
			failed++;
			break;
		case NOTSUPP:
			printf("requires unsupported feature: %s\n", err_buf);
			break;
		default:
			printf("FATAL: unexpected code: %d\n", ret);
			exit(1);
			break;
		}

		++tep;
	}

	return failed;
}

int main(int argc, char **argv)
{
	int op, ret;
	struct fi_info *fi;
	int failed;

	while ((op = getopt(argc, argv, "f:p:")) != -1) {
		switch (op) {
		case 'f':
			fabric_hints.name = optarg;
			break;
		case 'p':
			fabric_hints.prov_name = optarg;
			break;
		default:
			printf("usage: %s\n", argv[0]);
			printf("\t[-f fabric_name]\n");
			printf("\t[-p provider_name]\n");
			exit(1);
		}
	}

	hints.fabric_attr = &fabric_hints;
	hints.mode = ~0;

	ret = fi_getinfo(FI_VERSION(1, 0), NULL, 0, 0, &hints, &fi);
	if (ret != 0) {
		printf("fi_getinfo %s\n", fi_strerror(-ret));
		exit(1);
	}

	ret = fi_fabric(fi->fabric_attr, &fabric, NULL);
	if (ret != 0) {
		printf("fi_fabric %s\n", fi_strerror(-ret));
		exit(1);
	}

	printf("Testing EQs on fabric %s\n", fi->fabric_attr->name);

	failed = run_tests();
	if (failed > 0) {
		printf("Summary: %d tests failed\n", failed);
	} else {
		printf("Summary: all tests passed\n");
	}

	ret = fi_close(&fabric->fid);
	if (ret != 0) {
		printf("Error %d closing fabric: %s\n", ret, fi_strerror(-ret));
		exit(1);
	}

	exit(failed > 0);
}
