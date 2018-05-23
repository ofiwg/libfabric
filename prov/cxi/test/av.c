/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2015-2018 Cray Inc. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>

#include <criterion/criterion.h>

#include "cxi_prov.h"
#include "cxi_test_common.h"

static struct cxi_addr *test_addrs;
fi_addr_t *test_fi_addrs;
#define AV_COUNT 1024
int naddrs = AV_COUNT * 10;

static void test_addrs_init(void)
{
	int i;

	test_addrs = malloc(naddrs * sizeof(struct cxi_addr));
	cr_assert(test_addrs != NULL);

	test_fi_addrs = calloc(naddrs, sizeof(fi_addr_t));
	cr_assert(test_fi_addrs != NULL);

	for (i = 0; i < naddrs; i++) {
		test_addrs[i].nic = i;
		test_addrs[i].domain = i+1;
		test_addrs[i].port = i+2;
	}
}

static void test_addrs_fini(void)
{
	free(test_fi_addrs);
	free(test_addrs);
}

static void test_create(void)
{
	cxit_create_av();
	cr_assert(cxit_av != NULL);

	cxit_destroy_av();
}

static void __test_insert(int count, int iters)
{
	int j, i, ret;
	struct cxi_addr addr;
	size_t addrlen = sizeof(struct cxi_addr);

	cr_assert(naddrs >= count);

	cxit_create_av();
	test_addrs_init();

	for (j = 0; j < iters; j++) {
		for (i = 0; i < count; i++) {
			ret = fi_av_insert(cxit_av, &test_addrs[i], 1,
					   &test_fi_addrs[i], 0, NULL);
			cr_assert(ret == 1);
		}


		for (i = 0; i < count; i++) {
			ret = fi_av_lookup(cxit_av, test_fi_addrs[i], &addr,
					   &addrlen);
			cr_assert(ret == FI_SUCCESS);
			cr_assert(addr.nic == test_addrs[i].nic);
			cr_assert(addr.domain == test_addrs[i].domain);
			cr_assert(addr.port == test_addrs[i].port);
		}

		for (i = 0; i < count; i++) {
			ret = fi_av_remove(cxit_av, &test_fi_addrs[i], 1, 0);
			cr_assert(ret == 0);
		}
	}

	test_addrs_fini();
	cxit_destroy_av();
}

static void test_insert(void)
{
	int iters = 1;

	__test_insert(AV_COUNT/2, iters);
	__test_insert(AV_COUNT, iters);
	__test_insert(naddrs, iters);

	iters = 3;

	__test_insert(AV_COUNT/2, iters);
	__test_insert(AV_COUNT, iters);
	__test_insert(naddrs, iters);
}

TestSuite(av, .init = cxit_setup_av, .fini = cxit_teardown_av);

/* Test basic AV table creation */
Test(av, table_simple)
{
	cxit_av_attr.type = FI_AV_TABLE;
	test_create();
}

/* Test basic AV map creation */
Test(av, map_simple)
{
	cxit_av_attr.type = FI_AV_MAP;
	test_create();
}

/* Test basic AV table insert */
Test(av, table_insert)
{
	cxit_av_attr.count = AV_COUNT;
	cxit_av_attr.type = FI_AV_TABLE;
	naddrs = cxit_av_attr.count * 10;

	test_insert();
}

/* Test basic AV map insert */
Test(av, map_insert)
{
	cxit_av_attr.count = AV_COUNT;
	cxit_av_attr.type = FI_AV_MAP;
	naddrs = cxit_av_attr.count * 10;

	test_insert();
}

Test(av, straddr)
{
	uint64_t addr = 0xabcd1234abcd1234;
	size_t len = 0;
	char *buf = NULL;
	const char *tmp_buf;

	cxit_create_av();

	tmp_buf = fi_av_straddr(cxit_av, &addr, buf, &len);
	cr_assert(len == 33); /* "fi_addr_cxi://0x0000000000000000\0" */
	cr_assert(tmp_buf == buf);

	buf = malloc(len);
	cr_assert(buf != NULL);

	tmp_buf = fi_av_straddr(cxit_av, &addr, buf, &len);
	cr_assert(len == 33);
	cr_assert(tmp_buf == buf);

	printf("straddr output: %s\n", buf);

	free(buf);

	cxit_destroy_av();
}

