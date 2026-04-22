/*
 * Copyright (c) 2024 Intel Corporation. All rights reserved.
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

#include "rxd_unit_tests.h"

/*
 * rxd_info_to_core: When hints include FI_HMEM, core_info->caps must
 * include both FI_MSG and FI_HMEM.
 */
void test_rxd_info_to_core_hmem_passthrough(void **state)
{
	struct fi_info *hints;
	struct fi_info *core_info;
	int ret;

	(void) state;

	hints = fi_allocinfo();
	assert_non_null(hints);
	hints->caps = FI_MSG | FI_HMEM;

	core_info = fi_allocinfo();
	assert_non_null(core_info);

	ret = rxd_info_to_core(FI_VERSION(1, 9), hints, NULL, core_info);
	assert_int_equal(ret, 0);
	assert_true(core_info->caps & FI_MSG);
	assert_true(core_info->caps & FI_HMEM);
	assert_int_equal(core_info->ep_attr->type, FI_EP_DGRAM);

	fi_freeinfo(core_info);
	fi_freeinfo(hints);
}

/*
 * rxd_info_to_core: When hints do not include FI_HMEM, core_info->caps
 * must be FI_MSG only.
 */
void test_rxd_info_to_core_no_hmem(void **state)
{
	struct fi_info *hints;
	struct fi_info *core_info;
	int ret;

	(void) state;

	hints = fi_allocinfo();
	assert_non_null(hints);
	hints->caps = FI_MSG;

	core_info = fi_allocinfo();
	assert_non_null(core_info);

	ret = rxd_info_to_core(FI_VERSION(1, 9), hints, NULL, core_info);
	assert_int_equal(ret, 0);
	assert_true(core_info->caps & FI_MSG);
	assert_false(core_info->caps & FI_HMEM);

	fi_freeinfo(core_info);
	fi_freeinfo(hints);
}

/*
 * rxd_info_to_core: When hints is NULL, core_info->caps must be FI_MSG
 * only (no FI_HMEM).
 */
void test_rxd_info_to_core_null_hints(void **state)
{
	struct fi_info *core_info;
	int ret;

	(void) state;

	core_info = fi_allocinfo();
	assert_non_null(core_info);

	ret = rxd_info_to_core(FI_VERSION(1, 9), NULL, NULL, core_info);
	assert_int_equal(ret, 0);
	assert_true(core_info->caps & FI_MSG);
	assert_false(core_info->caps & FI_HMEM);

	fi_freeinfo(core_info);
}

/*
 * rxd_info_to_core_mr_modes: With version >= 1.5 and hints requesting
 * FI_HMEM, mr_mode must include FI_MR_HMEM.
 */
void test_rxd_info_to_core_mr_modes_hmem(void **state)
{
	struct fi_info *hints;
	struct fi_info *core_info;

	(void) state;

	hints = fi_allocinfo();
	assert_non_null(hints);
	hints->caps = FI_MSG | FI_HMEM;

	core_info = fi_allocinfo();
	assert_non_null(core_info);

	rxd_info_to_core_mr_modes(FI_VERSION(1, 9), hints, core_info);
	assert_true(core_info->domain_attr->mr_mode & FI_MR_LOCAL);
	assert_true(core_info->domain_attr->mr_mode & FI_MR_HMEM);

	fi_freeinfo(core_info);
	fi_freeinfo(hints);
}

/*
 * rxd_info_to_core_mr_modes: With version >= 1.5 and hints NOT requesting
 * FI_HMEM, mr_mode must include FI_MR_LOCAL but NOT FI_MR_HMEM.
 */
void test_rxd_info_to_core_mr_modes_no_hmem(void **state)
{
	struct fi_info *hints;
	struct fi_info *core_info;

	(void) state;

	hints = fi_allocinfo();
	assert_non_null(hints);
	hints->caps = FI_MSG;

	core_info = fi_allocinfo();
	assert_non_null(core_info);

	rxd_info_to_core_mr_modes(FI_VERSION(1, 9), hints, core_info);
	assert_true(core_info->domain_attr->mr_mode & FI_MR_LOCAL);
	assert_false(core_info->domain_attr->mr_mode & FI_MR_HMEM);

	fi_freeinfo(core_info);
	fi_freeinfo(hints);
}

/*
 * rxd_info_to_core_mr_modes: With version < 1.5 the function must take
 * the legacy path (FI_MR_UNSPEC), regardless of whether hints include
 * FI_HMEM.
 */
void test_rxd_info_to_core_mr_modes_old_version(void **state)
{
	struct fi_info *hints;
	struct fi_info *core_info;

	(void) state;

	hints = fi_allocinfo();
	assert_non_null(hints);
	hints->caps = FI_MSG | FI_HMEM;

	core_info = fi_allocinfo();
	assert_non_null(core_info);

	rxd_info_to_core_mr_modes(FI_VERSION(1, 4), hints, core_info);
	assert_true(core_info->mode & FI_LOCAL_MR);
	assert_false(core_info->domain_attr->mr_mode & FI_MR_HMEM);

	fi_freeinfo(core_info);
	fi_freeinfo(hints);
}

/*
 * rxd_info_to_rxd: When core_info reports FI_HMEM in its caps, the
 * resulting rxd info must retain FI_HMEM in caps, tx_attr->caps, and
 * rx_attr->caps.
 */
void test_rxd_info_to_rxd_core_has_hmem(void **state)
{
	struct fi_info *core_info;
	struct fi_info *info;
	int ret;

	(void) state;

	core_info = fi_allocinfo();
	assert_non_null(core_info);
	core_info->caps = FI_MSG | FI_HMEM;
	core_info->domain_attr->caps = FI_LOCAL_COMM | FI_REMOTE_COMM;
	core_info->ep_attr->max_msg_size = 4096;
	core_info->ep_attr->msg_prefix_size = 0;

	info = fi_allocinfo();
	assert_non_null(info);

	ret = rxd_info_to_rxd(FI_VERSION(1, 9), core_info, NULL, info);
	assert_int_equal(ret, 0);
	assert_true(info->caps & FI_HMEM);
	assert_true(info->tx_attr->caps & FI_HMEM);
	assert_true(info->rx_attr->caps & FI_HMEM);

	fi_freeinfo(info);
	fi_freeinfo(core_info);
}

/*
 * rxd_info_to_rxd: When core_info does NOT report FI_HMEM, the
 * resulting rxd info must have FI_HMEM stripped from caps, tx_attr->caps,
 * and rx_attr->caps.
 */
void test_rxd_info_to_rxd_core_lacks_hmem(void **state)
{
	struct fi_info *core_info;
	struct fi_info *info;
	int ret;

	(void) state;

	core_info = fi_allocinfo();
	assert_non_null(core_info);
	core_info->caps = FI_MSG;  /* no FI_HMEM */
	core_info->domain_attr->caps = FI_LOCAL_COMM | FI_REMOTE_COMM;
	core_info->ep_attr->max_msg_size = 4096;
	core_info->ep_attr->msg_prefix_size = 0;

	info = fi_allocinfo();
	assert_non_null(info);

	ret = rxd_info_to_rxd(FI_VERSION(1, 9), core_info, NULL, info);
	assert_int_equal(ret, 0);
	assert_false(info->caps & FI_HMEM);
	assert_false(info->tx_attr->caps & FI_HMEM);
	assert_false(info->rx_attr->caps & FI_HMEM);

	fi_freeinfo(info);
	fi_freeinfo(core_info);
}

int main(void)
{
	const struct CMUnitTest tests[] = {
		/* rxd_info_to_core */
		cmocka_unit_test(test_rxd_info_to_core_hmem_passthrough),
		cmocka_unit_test(test_rxd_info_to_core_no_hmem),
		cmocka_unit_test(test_rxd_info_to_core_null_hints),
		/* rxd_info_to_core_mr_modes */
		cmocka_unit_test(test_rxd_info_to_core_mr_modes_hmem),
		cmocka_unit_test(test_rxd_info_to_core_mr_modes_no_hmem),
		cmocka_unit_test(test_rxd_info_to_core_mr_modes_old_version),
		/* rxd_info_to_rxd */
		cmocka_unit_test(test_rxd_info_to_rxd_core_has_hmem),
		cmocka_unit_test(test_rxd_info_to_rxd_core_lacks_hmem),
	};

	return cmocka_run_group_tests(tests, NULL, NULL);
}
