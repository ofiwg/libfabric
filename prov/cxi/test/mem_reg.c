/*
 * (C) Copyright 2023 Hewlett Packard Enterprise Development LP
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
#include <sys/wait.h>
#include <ctype.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#include <pthread.h>

#include "libcxi/libcxi.h"
#include "cxip.h"
#include "cxip_test_common.h"

TestSuite(memReg, .timeout = CXIT_DEFAULT_TIMEOUT);

static void hmem_dev_reg_test_runner(bool dev_reg, bool cache_enable)
{
	int ret;
	void *buf;
	size_t buf_size = 1234;
	struct fid_mr *mr;
	struct cxip_mr *cxi_mr;

	if (dev_reg)
		ret = setenv("FI_CXI_DISABLE_HMEM_DEV_REGISTER", "0", 1);
	else
		ret = setenv("FI_CXI_DISABLE_HMEM_DEV_REGISTER", "1", 1);
	cr_assert_eq(ret, 0,
		     "Failed to set FI_CXI_DISABLE_HMEM_DEV_REGISTER %d",
		     -errno);

	if (cache_enable)
		ret = setenv("FI_MR_CACHE_MONITOR", "memhooks", 1);
	else
		ret = setenv("FI_MR_CACHE_MONITOR", "disabled", 1);
	cr_assert_eq(ret, 0,
		     "Failed to set FI_MR_CACHE_MONITOR %d",
		     -errno);

	buf = malloc(buf_size);
	cr_assert_neq(buf, NULL, "Failed to alloc mem");

	cxit_setup_msg();

	ret = fi_mr_reg(cxit_domain, buf, buf_size, FI_READ | FI_WRITE, 0, 0, 0,
			&mr, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_reg failed: %d", ret);

	ret = fi_mr_bind(mr, &cxit_ep->fid, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_bind failed: %d", ret);

	ret = fi_mr_enable(mr);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_enable failed: %d", ret);

	/* Have to examine the struct to determine if correct behavior is
	 * happening.
	 */
	cxi_mr = container_of(mr, struct cxip_mr, mr_fid);
	if (dev_reg)
		cr_assert_neq(cxi_mr->md->host_addr, NULL,
			      "Bad cxip_md host_addr");
	else
		cr_assert_eq(cxi_mr->md->host_addr, NULL,
			     "Bad cxip_md host_addr");
	cr_assert_eq(cxi_mr->md->cached, cache_enable, "Bad cxip_md cached");

	ret = fi_close(&mr->fid);
	cr_assert_eq(ret, FI_SUCCESS, "fi_close failed: %d", ret);

	cxit_teardown_msg();
	free(buf);
}

Test(memReg, disableHmemDevRegisterEnabled_mrCacheEnabled)
{
	hmem_dev_reg_test_runner(true, true);
}

Test(memReg, disableHmemDevRegisterEnabled_mrCacheDisabled)
{
	hmem_dev_reg_test_runner(true, false);
}

Test(memReg, disableHmemDevRegisterDisabled_mrCacheEnabled)
{
	hmem_dev_reg_test_runner(false, true);
}

Test(memReg, disableHmemDevRegisterDisabled_mrCacheDisabled)
{
	hmem_dev_reg_test_runner(false, false);
}
