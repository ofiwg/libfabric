/*
 * Copyright (c) 2015-2016 Los Alamos National Security, LLC. All
 * rights reserved.
 * Copyright (c) 2015-2016 Cray Inc. All rights reserved.
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

#include "gnix_vector.h"
#include <unistd.h>
#include <criterion/criterion.h>
#include <stdlib.h>

gnix_vector_t vec;
gnix_vec_attr_t attr;

#define VEC_MAX  (1024)
#define VEC_INIT 128

void vector_setup_lockless(void)
{
	int ret;

	attr.vec_increase_step = 2;
	attr.vec_increase_type = GNIX_VEC_INCREASE_MULT;
	attr.vec_initial_size = VEC_INIT;
	attr.vec_internal_locking = GNIX_VEC_UNLOCKED;
	attr.vec_maximum_size = VEC_MAX;

	ret = _gnix_vec_init(&vec, &attr);
	cr_assert(!ret, "_gnix_vec_init");
}

void vector_setup_locked()
{
	int ret;

	attr.vec_increase_step = 2;
	attr.vec_increase_type = GNIX_VEC_INCREASE_ADD;
	attr.vec_initial_size = VEC_INIT;
	attr.vec_internal_locking = GNIX_VEC_LOCKED;
	attr.vec_maximum_size = VEC_MAX;

	ret = _gnix_vec_init(&vec, &attr);
	cr_assert(!ret, "_gnix_vec_init");
}

void do_invalid_ops_params()
{
	int ret;
	void *tmp;

	/* ret = _gnix_vec_insert_first(&vec, NULL); */
	/* cr_assert_eq(ret, -FI_EINVAL); */

	/* ret = _gnix_vec_remove_first(&vec); */
	/* cr_assert_eq(ret, -FI_EINVAL); */

	/* ret = _gnix_vec_first(&vec, &tmp); */
	/* cr_assert_eq(ret, -FI_EINVAL); */

	ret = _gnix_vec_insert_first(NULL, NULL);
	cr_assert_eq(ret, -FI_EINVAL);

	ret = _gnix_vec_remove_first(NULL);
	cr_assert_eq(ret, -FI_EINVAL);

	ret = _gnix_vec_first(NULL, &tmp);
	cr_assert_eq(ret, -FI_EINVAL);
}

void vector_teardown(void)
{
	int ret;

	ret = _gnix_vec_close(&vec);
	cr_assert(!ret, "_gnix_vec_close");

	do_invalid_ops_params();

	/* ret = _gnix_vec_close(&vec); */
	/* cr_assert_eq(ret, -FI_EINVAL); */

	ret = _gnix_vec_close(NULL);
	cr_assert_eq(ret, -FI_EINVAL);
}

/* Test invalid parameters for setup */
void vector_setup_error(void)
{
	int ret;

	attr.vec_increase_step = 2;
	attr.vec_increase_type = GNIX_VEC_INCREASE_MULT;
	attr.vec_initial_size = VEC_INIT;
	attr.vec_internal_locking = GNIX_VEC_UNLOCKED;
	attr.vec_maximum_size = VEC_MAX;

	ret = _gnix_vec_init(NULL, NULL);
	cr_assert_eq(ret, -FI_EINVAL);

	ret = _gnix_vec_init(NULL, &attr);
	cr_assert_eq(ret, -FI_EINVAL);

	ret = _gnix_vec_init(&vec, NULL);
	cr_assert_eq(ret, -FI_EINVAL);

	attr.vec_initial_size = 256;
	attr.vec_maximum_size = 128;

	ret = _gnix_vec_init(&vec, &attr);
	cr_assert_eq(ret, -FI_EINVAL);

	/* attr.vec_initial_size = 64; */
	/* attr.vec_maximum_size = 128; */
	/* vec.state = GNIX_VEC_STATE_READY; */
	/* ret = _gnix_vec_init(&vec, &attr); */

	vector_setup_lockless();
}

void do_insert_first()
{
	int ret;
	void *tmp = malloc(sizeof(gnix_vec_entry_t));
	cr_assert(tmp, "do_insert_first");

	ret = vec.ops->insert_first(&vec, tmp);
	cr_assert(!ret, "_gnix_vec_insert_first");

	ret = vec.ops->insert_first(&vec, tmp);
	cr_assert_eq(ret, -FI_ECANCELED);
}

void do_insert_last()
{
	int ret;
	void *tmp = malloc(sizeof(gnix_vec_entry_t));
	cr_assert(tmp, "do_insert_last");

	ret = vec.ops->insert_last(&vec, tmp);
	cr_assert(!ret, "_gnix_vec_insert_last");

	ret = vec.ops->insert_last(&vec, tmp);
	cr_assert_eq(ret, -FI_ECANCELED);
}

void do_fill_insert_at()
{
	int i, ret;
	void *tmp;

	for (i = 0; i < vec.attr.cur_size; i++) {
		tmp = malloc(sizeof(gnix_vec_entry_t));
		cr_assert(tmp, "do_insert_at");

		ret = vec.ops->insert_at(&vec, tmp, i);
		cr_assert(!ret, "_gnix_vec_insert_at");
	}

	/* Test grow. */
	tmp = malloc(sizeof(gnix_vec_entry_t));
	cr_assert(tmp, "do_insert_at");

	ret = vec.ops->insert_at(&vec, tmp, VEC_MAX-1);
	cr_assert(!ret, "_gnix_vec_insert_at");
	cr_assert_eq(vec.attr.cur_size, VEC_MAX);

	ret = vec.ops->insert_at(&vec, tmp, VEC_MAX-1);
	cr_assert_eq(ret, -FI_ECANCELED);

	for (; i < vec.attr.cur_size - 1; i++) {
		tmp = malloc(sizeof(gnix_vec_entry_t));
		cr_assert(tmp, "do_insert_at");

		ret = vec.ops->insert_at(&vec, tmp, i);
		cr_assert(!ret, "_gnix_vec_insert_at");
	}

	ret = vec.ops->insert_at(&vec, tmp, VEC_MAX);
	cr_assert_eq(ret, -FI_EINVAL);
}

void do_remove_first()
{
	int ret;

	ret = vec.ops->remove_first(&vec);
	cr_assert(!ret, "_gnix_vec_remove_first");

	ret = vec.ops->remove_first(&vec);
	cr_assert_eq(ret, -FI_ECANCELED);
}

void do_remove_last()
{
	int ret;

	ret = vec.ops->remove_last(&vec);
	cr_assert(!ret, "_gnix_vec_remove_last");

	ret = vec.ops->remove_last(&vec);
	cr_assert_eq(ret, -FI_ECANCELED);
}

void do_unfill_remove_at()
{
	int i, ret;

	for (i = 0; i < vec.attr.cur_size; i++) {
		ret = vec.ops->remove_at(&vec, i);
		cr_assert(!ret, "_gnix_vec_remove_at");
	}

	ret = vec.ops->remove_at(&vec, vec.attr.cur_size / 2);
	cr_assert_eq(ret, -FI_ECANCELED);
}

void do_first()
{
	int ret;
	void *tmp;

	ret = vec.ops->first(&vec, &tmp);
	cr_assert(tmp, "_gnix_vec_first");
	cr_assert(!ret, "_gnix_vec_first");

	tmp = NULL;

	do_remove_first();

	ret = vec.ops->first(&vec, &tmp);
	cr_assert_eq(tmp, NULL);
	cr_assert_eq(ret, -FI_ECANCELED);
}

void do_last()
{
	int ret;
	void *tmp;

	ret = vec.ops->last(&vec, &tmp);
	cr_assert(tmp, "_gnix_vec_last");
	cr_assert(!ret, "_gnix_vec_last");

	tmp = NULL;

	do_remove_last();

	ret = vec.ops->last(&vec, &tmp);
	cr_assert_eq(tmp, NULL);
	cr_assert_eq(ret, -FI_ECANCELED);
}

void do_at()
{
	int i ,ret;
	void *tmp;

	for (i = 0; i < vec.attr.cur_size; i++) {
		ret = vec.ops->at(&vec, &tmp, i);
		cr_assert(!ret, "_gnix_vec_at");

		cr_assert(!!tmp, "_gnix_vec_at");
		tmp = NULL;
	}
}

void do_tests()
{
	do_insert_first();
	do_first();

	do_insert_last();
	do_last();

	do_fill_insert_at();
	do_at();
	do_unfill_remove_at();
}

TestSuite(vector_lockless, .init = vector_setup_lockless,
	  .fini = vector_teardown, .disabled = false);

Test(vector_lockless, do_first)
{
	do_insert_first();
	do_first();
}

Test(vector_lockless, do_last)
{
	do_insert_last();
	do_last();
}

Test(vector_lockless, do_at)
{
	do_fill_insert_at();
	do_at();
	do_unfill_remove_at();
}

TestSuite(vector_locked, .init = vector_setup_locked,
	  .fini = vector_teardown, .disabled = false);

Test(vector_locked, do_first)
{
	do_insert_first();
	do_first();
}

Test(vector_locked, do_last)
{
	do_insert_last();
	do_last();
}

Test(vector_locked, do_at)
{
	do_fill_insert_at();
	do_at();
	do_unfill_remove_at();
}

TestSuite(vector_error_lockless, .init = vector_setup_error,
	  .fini = vector_teardown, .disabled = false);

Test(vector_error_lockless, setup_teardown_error)
{

}
