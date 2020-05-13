/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2020 Cray Inc. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>

#include <criterion/criterion.h>

#include "cxip.h"
#include "cxip_test_common.h"

TestSuite(mr, .init = cxit_setup_rma, .fini = cxit_teardown_rma,
	  .timeout = CXIT_DEFAULT_TIMEOUT);

Test(mr, opt_mrs, .timeout = 60)
{
	int opt_mr_cnt = 200;
	struct mem_region opt_mrs[opt_mr_cnt];
	int i;

	for (i = 0; i < opt_mr_cnt; i++)
		mr_create(0x1000, FI_REMOTE_WRITE, 0, i, &opt_mrs[i]);

	for (i = 0; i < opt_mr_cnt; i++)
		mr_destroy(&opt_mrs[i]);
}

Test(mr, std_mrs, .timeout = 600, .disabled = true)
{
	int std_mr_cnt = 16*1024;
	int mrs = 0;
	struct mem_region std_mrs[std_mr_cnt];
	int i;
	int ret;

	for (i = 0; i < std_mr_cnt; i++) {
		mrs++;
		ret = mr_create(8, FI_REMOTE_WRITE, 0, i+200, &std_mrs[i]);
		if (ret) {
			printf("Standard MR limit: %d\n", mrs);
			break;
		}
	}

	/* It's difficult to predict available resources. An idle system
	 * currently supports at least 13955 total standard MRs. This is
	 * roughly:
	 * 16k total LEs -
	 * 1000 (reserved for services) -
	 * 1400 (reserved for other pools) =
	 * 13984
	 *
	 * An EP requires a few other LEs to implement messaging and other
	 * APIs.
	 */
	cr_assert(mrs >= 13955);

	/* Note: MR close is very slow in emulation due to
	 * cxil_invalidate_pte_le().
	 */
	for (i = 0; i < mrs; i++)
		mr_destroy(&std_mrs[i]);
}

/* Perform zero-byte Puts to zero-byte standard and optimized MRs. Validate
 * remote counting events.
 */
Test(mr, mr_zero_len)
{
	struct mem_region mr;
	struct fi_cq_tagged_entry cqe;
	int ret;

	/* Optimized MR */

	ret = mr_create(0, FI_REMOTE_WRITE, 0, 0, &mr);
	cr_assert(ret == FI_SUCCESS);

	ret = fi_write(cxit_ep, NULL, 0, NULL,
		       cxit_ep_fi_addr, 0, 0, NULL);
	cr_assert(ret == FI_SUCCESS);

	ret = cxit_await_completion(cxit_tx_cq, &cqe);
	cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);

	while (fi_cntr_read(cxit_rem_cntr) != 1)
		sched_yield();

	mr_destroy(&mr);

	/* Standard MR */

	ret = mr_create(0, FI_REMOTE_WRITE, 0, 200, &mr);
	cr_assert(ret == FI_SUCCESS);

	ret = fi_write(cxit_ep, NULL, 0, NULL,
		       cxit_ep_fi_addr, 0, 200, NULL);
	cr_assert(ret == FI_SUCCESS, "ret: %d\n", ret);

	ret = cxit_await_completion(cxit_tx_cq, &cqe);
	cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);

	while (fi_cntr_read(cxit_rem_cntr) != 2)
		sched_yield();

	mr_destroy(&mr);
}

/* Validate that unique keys are enforced. */
Test(mr, mr_unique_key)
{
	struct mem_region mr;
	struct mem_region mr2;
	int ret;

	/* Optimized MR */

	ret = mr_create(0, FI_REMOTE_WRITE, 0, 0, &mr);
	cr_assert(ret == FI_SUCCESS);

	ret = mr_create(0, FI_REMOTE_WRITE, 0, 0, &mr2);
	cr_assert(ret == -FI_ENOKEY);

	mr_destroy(&mr);
	mr_destroy(&mr2);

	/* Standard MR */

	ret = mr_create(0, FI_REMOTE_WRITE, 0, 200, &mr);
	cr_assert(ret == FI_SUCCESS);

	ret = mr_create(0, FI_REMOTE_WRITE, 0, 200, &mr2);
	cr_assert(ret == -FI_ENOKEY);

	mr_destroy(&mr);
	mr_destroy(&mr2);
}

/* Test creating and destroying an MR that is never bound to an EP. */
Test(mr, no_bind)
{
	int ret;
	size_t buf_len = 0x1000;
	void *buf;
	struct fid_mr *mr;

	buf = malloc(buf_len);
	cr_assert(buf);

	/* Optimized MR */

	ret = fi_mr_reg(cxit_domain, buf, buf_len, FI_REMOTE_WRITE,
			0, 0, 0, &mr, NULL);
	cr_assert_eq(ret, FI_SUCCESS);

	fi_close(&mr->fid);

	/* Standard MR */

	ret = fi_mr_reg(cxit_domain, buf, buf_len, FI_REMOTE_WRITE,
			0, 200, 0, &mr, NULL);
	cr_assert_eq(ret, FI_SUCCESS);

	fi_close(&mr->fid);

	free(buf);
}
