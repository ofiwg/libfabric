/*
 * Copyright (c) 2020 Intel Corporation. All rights reserved.
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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHWARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. const NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER const AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS const THE
 * SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <complex.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>

#include <ofi.h>

#include "cxip.h"
#include "cxip_test_common.h"

TestSuite(avset, .init = cxit_setup_rma, .fini = cxit_teardown_rma,
	  .disabled = false, .timeout = CXIT_DEFAULT_TIMEOUT);

/*
 * Simple test to ensure that any attempt to close the AV before closing any AV
 * Set will fail with -FI_EBUSY.
 */
Test(avset, av_set_refcnt)
{
	// Make sure open AV sets preclude closing AV
	struct fi_av_set_attr attr = {.flags=FI_UNIVERSE};
	struct fid_av_set *set;
	int ret;

	ret = fi_av_set(cxit_av, &attr, &set, NULL);
	cr_expect_eq(ret, 0, "fi_av_set failed, ret=%d", ret);

	ret = fi_close(&cxit_av->fid);
	cr_expect_eq(ret, -FI_EBUSY, "premature AV close failed, ret=%d", ret);

	ret = fi_close(&set->fid);
	cr_expect_eq(ret, 0, "fi_close(set) failed, ret=%d", ret);
}

/*
 * Test of AVSet operations
 *
 * We choose by-two and by-three spans to explore union, intersection, diff
 */
static bool is_div_2(fi_addr_t addr)
{
	return (addr & 1) == 0;
}

static bool is_div_3(fi_addr_t addr)
{
	return ((addr / 3) * 3) == addr;
}

static bool is_not2_and_3(fi_addr_t addr)
{
	return !is_div_2(addr) && is_div_3(addr);
}

static bool is_2_and_3(fi_addr_t addr)
{
	return is_div_2(addr) && is_div_3(addr);
}

static bool is_2_or_3(fi_addr_t addr)
{
	return is_div_2(addr) || is_div_3(addr);
}

static int _comp_fi_addr(const void *a, const void *b)
{
	// for sorting unsigned
	if (*(fi_addr_t *)a < *(fi_addr_t *)b) return -1;
	if (*(fi_addr_t *)a > *(fi_addr_t *)b) return  1;
	return 0;
}

static void check_av_set(const char *name, struct fid_av_set *set, int max,
			 bool (*func)(fi_addr_t), bool is_ordered)
{
	// ensure all elements of set satisfy expectations
	struct cxip_av_set *cxi_set;
	int errors = 0;
	int i, j;

	cxi_set = container_of(set, struct cxip_av_set, av_set_fid);

	// If set is not ordered, sort into order to test
	if (! is_ordered)
		qsort(cxi_set->fi_addr_ary, cxi_set->fi_addr_cnt,
		      sizeof(fi_addr_t), _comp_fi_addr);

	// Traverse maximum span, ensuring that allowed addr is the next addr
	for (i = j = 0; i < cxi_set->cxi_av->table_hdr->stored; i++) {
		if ((*func)(i)) {
			// This should be next in the set
			if (cxi_set->fi_addr_ary[j] != i) {
				printf("%s: set[%2d]: %2ld bad value\n",
				       name, j, cxi_set->fi_addr_ary[j]);
				errors++;
			}
			j++;
		}
	}

	cr_assert_eq(errors, 0, "%s: check_av_set failure", name);
}

Test(avset, basics)
{
	// Test basic set operations
	struct fi_av_set_attr attr2 = {
		.count = 20, .start_addr = 0, .end_addr = 20, .stride = 2
	};
	struct fi_av_set_attr attr3 = {
		.count = 20, .start_addr = 0, .end_addr = 20, .stride = 3
	};
	struct fid_av_set *set2;
	struct fid_av_set *setX;
	struct cxip_av *cxi_av;
	int first;
	int i, ret;

	// Expand the AV, so we have enough addresses to test
	cxi_av = container_of(cxit_av, struct cxip_av, av_fid);
	first = cxi_av->table_hdr->stored;
	for (i = first; i < 20; i++) {
		struct cxip_addr fake_addr = { .nic = i, .pid = 0xff };
		int inserted;

		inserted = fi_av_insert(cxit_av, (void *)&fake_addr,
					1, NULL, 0, NULL);
		cr_expect_eq(inserted, 1,
			     "fi_av_insert[%2d] failed, inserted=%d",
			     i, inserted);
	}
	cr_expect_eq(cxi_av->table_hdr->stored, 20, "av insert failure");

	// Create a stride of every second element
	ret = fi_av_set(cxit_av, &attr2, &set2, NULL);
	cr_expect_eq(ret, 0, "1 fi_av_set set2 failed, ret=%d", ret);
	check_av_set("1 two", set2, 20, is_div_2, true);

	// Create a stride of every third element
	ret = fi_av_set(cxit_av, &attr3, &setX, NULL);
	cr_expect_eq(ret, 0, "1 fi_av_set setX failed, ret=%d", ret);
	check_av_set("1 three", setX, 20, is_div_3, true);

	ret = fi_close(&setX->fid);
	cr_expect_eq(ret, 0, "1 fi_close(setX) failed, ret=%d", ret);

	// 3 union 2
	ret = fi_av_set(cxit_av, &attr3, &setX, NULL);
	cr_expect_eq(ret, 0, "2 fi_av_set setX failed, ret=%d", ret);
	check_av_set("2 dst", setX, 20, is_div_3, true);

	ret = fi_av_set_union(setX, set2);
	cr_expect_eq(ret, 0, "2 fi_av_set set_union failed, ret=%d", ret);
	check_av_set("2 union", setX, 20, is_2_or_3, false);

	ret = fi_close(&setX->fid);
	cr_expect_eq(ret, 0, "2 fi_close(setX) failed, ret=%d", ret);

	// 3 diff 2
	ret = fi_av_set(cxit_av, &attr3, &setX, NULL);
	cr_expect_eq(ret, 0, "3 fi_av_set setX failed, ret=%d", ret);
	check_av_set("3 dst", setX, 20, is_div_3, true);

	ret = fi_av_set_diff(setX, set2);
	cr_expect_eq(ret, 0, "3 fi_av_set set_diff failed, ret=%d", ret);
	check_av_set("3 diff", setX, 20, is_not2_and_3, true);

	ret = fi_close(&setX->fid);
	cr_expect_eq(ret, 0, "3 fi_close(setX) failed, ret=%d", ret);

	// 3 intersect 2
	ret = fi_av_set(cxit_av, &attr3, &setX, NULL);
	cr_expect_eq(ret, 0, "4 fi_av_set setX failed, ret=%d", ret);
	check_av_set("4 dst", setX, 20, is_div_3, true);

	ret = fi_av_set_intersect(setX, set2);
	cr_expect_eq(ret, 0, "4 fi_av_set set_intersect failed, ret=%d", ret);
	check_av_set("4 intersect", setX, 20, is_2_and_3, true);

	ret = fi_close(&setX->fid);
	cr_expect_eq(ret, 0, "4 fi_close(setX) failed, ret=%d", ret);

	// clean up
	ret = fi_close(&set2->fid);
	cr_expect_eq(ret, 0, "fi_close(set2) failed, ret=%d", ret);
}


