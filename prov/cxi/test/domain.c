/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>

#include <criterion/criterion.h>

#include "cxip.h"
#include "cxip_test_common.h"

TestSuite(domain, .init = cxit_setup_domain, .fini = cxit_teardown_domain,
	  .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test basic domain creation */
Test(domain, simple)
{
	cxit_create_domain();
	cr_assert(cxit_domain != NULL);

	cxit_destroy_domain();
}

/* Test use of topology ops */
Test(domain, topology)
{
	unsigned int group_num, switch_num, port_num;
	int ret;

	cxit_create_domain();
	cr_assert(cxit_domain != NULL);
	ret = dom_ops->topology(&cxit_domain->fid, &group_num, &switch_num,
				&port_num);
	cr_assert_eq(ret, FI_SUCCESS, "topology failed: %d\n", ret);

	ret = dom_ops->topology(&cxit_domain->fid, NULL, &switch_num,
				&port_num);
	cr_assert_eq(ret, FI_SUCCESS, "null group topology failed: %d\n", ret);

	ret = dom_ops->topology(&cxit_domain->fid, &group_num, NULL,
				&port_num);
	cr_assert_eq(ret, FI_SUCCESS, "null switch topology failed: %d\n", ret);

	ret = dom_ops->topology(&cxit_domain->fid, &group_num, &switch_num,
				NULL);
	cr_assert_eq(ret, FI_SUCCESS, "null port topology failed: %d\n", ret);

	cxit_destroy_domain();
}

Test(domain, enable_hybrid_mr_desc)
{
	int ret;

	cxit_create_domain();
	cr_assert(cxit_domain != NULL);

	ret = dom_ops->enable_hybrid_mr_desc(&cxit_domain->fid, true);
	cr_assert_eq(ret, FI_SUCCESS, "enable_hybrid_mr_desc failed: %d\n",
		     ret);

	cxit_destroy_domain();
}

TestSuite(domain_cntrs, .init = cxit_setup_rma, .fini = cxit_teardown_rma,
	  .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test basic counter read */
Test(domain_cntrs, cntr_read)
{
	int ret;
	uint64_t value;
	struct timespec ts;

	ret = dom_ops->cntr_read(&cxit_domain->fid, C_CNTR_LPE_SUCCESS_CNTR,
				 &value, &ts);
	cr_assert_eq(ret, FI_SUCCESS, "cntr_read failed: %d\n", ret);

	printf("LPE_SUCCESS_CNTR: %lu\n", value);
}
