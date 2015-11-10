/*
 * Copyright (c) 2015 Cray Inc. All rights reserved.
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
#include <stddef.h>
#include <sys/time.h>

#include "gnix_freelist.h"

#include <criterion/criterion.h>

static void setup(void)
{
	srand(time(NULL));
}

static void teardown(void)
{
}

static void generate_perm(int *perm, int len)
{
	int i;
	/* good 'nuff */
	for (i = 0; i < len; i++) {
		int t = perm[i];
		int j = rand() % len;

		perm[i] = perm[j];
		perm[j] = t;
	}
}

TestSuite(gnix_freelist, .init = setup, .fini = teardown);

Test(gnix_freelist, freelist_init_destroy)
{
	const int n = 13;
	struct gnix_s_freelist fls[n];
	int i, ret;

	/* non-optimized code may not zero structures */
	memset(fls, 0x0, n * sizeof(struct gnix_s_freelist));

	for (i = 0; i < n; i++) {
		ret = _gnix_sfl_init(sizeof(struct slist_entry), 0,
				     2*n, n, n, 3*n, &fls[i]);
		cr_assert_eq(ret, FI_SUCCESS, "Failed to initialize freelist");
	}

	for (i = n-1; i >= 0; i--)
		_gnix_sfl_destroy(&fls[i]);
}

Test(gnix_freelist, freelist_refill_test)
{
	struct gnix_s_freelist fl;
	int i, ret;
	const int num_elems = 71;
	struct slist_entry *elems[num_elems];
	const int refill_size = 47;
	struct slist_entry *refill_elems[refill_size];

	/* non-optimized code may not zero structures */
	memset(&fl, 0x0, sizeof(struct gnix_s_freelist));

	ret = _gnix_sfl_init(sizeof(struct slist_entry), 0,
			     num_elems, refill_size, 0, 0, &fl);
	cr_assert_eq(ret, FI_SUCCESS, "Failed to initialize freelist");

	for (i = 0; i < num_elems; i++) {
		ret = _gnix_sfe_alloc(&elems[i], &fl);
		cr_assert_eq(ret, FI_SUCCESS, "Failed to obtain slist_entry");
	}
	cr_assert(_gnix_sfl_empty(&fl), "Freelist not empty");

	for (i = 0; i < refill_size; i++) {
		ret = _gnix_sfe_alloc(&refill_elems[i], &fl);
		cr_assert_eq(ret, FI_SUCCESS, "Failed to obtain slist_entry");
		if (i != refill_size-1) {
			/* Not the last one, so must not be empty */
			cr_assert(!_gnix_sfl_empty(&fl), "Freelist empty");
		}
	}
	cr_assert(_gnix_sfl_empty(&fl), "Freelist not empty");

	for (i = num_elems-1; i >= 0 ; i--)
		_gnix_sfe_free(elems[i], &fl);

	for (i = refill_size-1; i >= 0 ; i--)
		_gnix_sfe_free(refill_elems[i], &fl);

	_gnix_sfl_destroy(&fl);
}

struct slist_ts {
	char dummy[7];
	struct slist_entry e;
	int n;
};

Test(gnix_freelist, freelist_random_alloc_free)
{
	struct gnix_s_freelist fl;
	int i, ret;
	const int n = 719;
	int perm[n];
	struct slist_entry *se;
	struct slist_ts *ts[n];

	for (i = 0; i < n; i++)
		perm[i] = i;

	generate_perm(perm, n);

	/* non-optimized code may not zero structures */
	memset(&fl, 0x0, sizeof(struct gnix_s_freelist));

	ret = _gnix_sfl_init(sizeof(struct slist_ts),
			     offsetof(struct slist_ts, e),
			     0, 0, 0, 0, &fl);
	cr_assert_eq(ret, FI_SUCCESS, "Failed to initialize freelist");

	for (i = 0; i < n; i++) {
		ret = _gnix_sfe_alloc(&se, &fl);
		cr_assert_eq(ret, FI_SUCCESS,
			     "Failed to obtain valid slist_entry");
		ts[i] = container_of(se, struct slist_ts, e);
		ts[i]->n = perm[i];
	}

	for (i = 0; i < n; i++) {
		int j = perm[i];

		cr_assert(ts[j]->n == perm[j], "Incorrect value");
		_gnix_sfe_free(&ts[j]->e, &fl);
		ts[j] = NULL;
	}

	_gnix_sfl_destroy(&fl);
}

