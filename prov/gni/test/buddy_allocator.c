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

#include "gnix_buddy_allocator.h"
#include <criterion/criterion.h>

#define LEN 1024 * 1024 	/* buddy_handle->len */
#define MAX_LEN LEN / 1024	/* buddy_handle->max */
#define MIN_LEN 16		/* buddy_handle->min */

long *buf = NULL;		/* buddy_handle->base */
gnix_buddy_alloc_handle_t *buddy_handle;
void **ptr = NULL;		/* ptrs alloc'd by _gnix_buddy_alloc */


void buddy_allocator_setup(void)
{
	int ret;

	ptr = calloc(LEN / MIN_LEN, sizeof(void *));
	cr_assert(ptr, "buddy_allocator_setup");

	buf = calloc(LEN, sizeof(long));
	cr_assert(buf, "buddy_allocator_setup");

	ret = _gnix_buddy_allocator_create(buf, LEN, MAX_LEN, &buddy_handle);
	cr_assert(!ret, "_gnix_buddy_allocator_create");
}

void buddy_allocator_teardown(void)
{
	int ret;

	ret = _gnix_buddy_allocator_destroy(buddy_handle);
	cr_assert(!ret, "_gnix_buddy_allocator_destroy");

	free(ptr);
	free(buf);
}

/* Test invalid parameters for setup */
void buddy_allocator_setup_error(void)
{
	int ret;

	ret = _gnix_buddy_allocator_create(NULL, LEN, MAX_LEN, &buddy_handle);
	cr_assert_eq(ret, -FI_EINVAL);

	ret = _gnix_buddy_allocator_create(buf, 0, MAX_LEN, &buddy_handle);
	cr_assert_eq(ret, -FI_EINVAL);

	ret = _gnix_buddy_allocator_create(buf, LEN, LEN + 1, &buddy_handle);
	cr_assert_eq(ret, -FI_EINVAL);

	ret = _gnix_buddy_allocator_create(buf, LEN, 0, &buddy_handle);
	cr_assert_eq(ret, -FI_EINVAL);

	ret = _gnix_buddy_allocator_create(buf, LEN, MAX_LEN, NULL);
	cr_assert_eq(ret, -FI_EINVAL);
}

/* Test invalid parameters for teardown */
void buddy_allocator_teardown_error(void)
{
	int ret;

	ret = _gnix_buddy_allocator_destroy(NULL);
	cr_assert_eq(ret, -FI_EINVAL);
}

void do_alloc(int len)
{
	int i = 0, ret;

	/* Allocate all the memory and write to each block */
	for (; i < LEN / len; i++) {
		ret = _gnix_buddy_alloc(buddy_handle, ptr + i, len);
		cr_assert(!ret, "_gnix_buddy_alloc");
		memset(ptr[i], 0xffffffff, len);
	}

	/* Ensure that all free lists are empty */
	for (i = 0; i < buddy_handle->nlists; i++) {
		ret = dlist_empty(buddy_handle->lists + i);
		cr_assert_eq(ret, 1);
	}
}

void do_free(int len)
{
	int i = 0, ret;

	/* Free all allocated blocks */
	for (; i < LEN / len; i++) {
		ret = _gnix_buddy_free(buddy_handle, ptr[i], len);
		cr_assert(!ret, "_gnix_buddy_free");
	}

	/* Ensure that every free list except the last is empty */
	for (i = 0; i < buddy_handle->nlists - 1; i++) {
		ret = dlist_empty(buddy_handle->lists + i);
		cr_assert_eq(ret, 1);
	}
	ret = dlist_empty(buddy_handle->lists + i);
	cr_assert_eq(ret, 0);
}

TestSuite(buddy_allocator, .init = buddy_allocator_setup,
	  .fini = buddy_allocator_teardown, .disabled = false);

Test(buddy_allocator, alloc_free)
{
	int i = MIN_LEN;

	/* Sequential alloc and frees */
	for (i = MIN_LEN; i <= MAX_LEN; i *= 2) {
		do_alloc(i);
		do_free(i);
	}

	/* TODO: Random allocs and frees */
}

Test(buddy_allocator, alloc_free_error)
{
	int ret;
	void *tmp;

	do_alloc(MIN_LEN);

	/* Request one additional block */
	ret = _gnix_buddy_alloc(buddy_handle, &tmp, MIN_LEN);
	cr_assert_eq(ret, -FI_ENOMEM);

	do_free(MIN_LEN);
}

/* Test invalid buddy alloc and free parameters */
Test(buddy_allocator, parameter_error)
{
	int ret;

	buddy_allocator_setup_error();
	buddy_allocator_teardown_error();

	/* BEGIN: Alloc, invalid parameters */
	ret = _gnix_buddy_alloc(NULL, ptr, MAX_LEN);
	cr_assert_eq(ret, -FI_EINVAL);

	ret = _gnix_buddy_alloc(buddy_handle, ptr, MAX_LEN + 1);
	cr_assert_eq(ret, -FI_EINVAL);

	ret = _gnix_buddy_alloc(buddy_handle, ptr, 0);
	cr_assert_eq(ret, -FI_EINVAL);

	ret = _gnix_buddy_alloc(buddy_handle, NULL, MAX_LEN);
	cr_assert_eq(ret, -FI_EINVAL);
	/* END: Alloc, invalid parameters */

	/* BEGIN: Free, invalid parameters */
	ret = _gnix_buddy_free(NULL, ptr, MAX_LEN);
	cr_assert_eq(ret, -FI_EINVAL);

	ret = _gnix_buddy_free(buddy_handle, NULL, MAX_LEN);
	cr_assert_eq(ret, -FI_EINVAL);

	ret = _gnix_buddy_free(buddy_handle, buf - 1, MAX_LEN);
	cr_assert_eq(ret, -FI_EINVAL);

	ret = _gnix_buddy_free(buddy_handle, buf + LEN, MAX_LEN);
	cr_assert_eq(ret, -FI_EINVAL);

	ret = _gnix_buddy_free(buddy_handle, buf, MAX_LEN + 1);
	cr_assert_eq(ret, -FI_EINVAL);

	ret = _gnix_buddy_free(buddy_handle, buf - 1, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	/* END: Free, invalid parameters */
}
