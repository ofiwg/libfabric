/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2020 Hewlett Packard Enterprise Development LP
 */

/* Notes:
 *
 * This test is perfunctory at present. A fuller set of tests is available:
 *
 * virtualize.sh fabtests/unit/fi_eq_test
 *
 * TODO: current implementation does not support wait states.
 */

#include <stdio.h>
#include <stdlib.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>

#include <ofi.h>

#include "cxip.h"
#include "cxip_test_common.h"

TestSuite(eq, .init = cxit_setup_eq, .fini = cxit_teardown_eq,
	  .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test basic EQ creation */
Test(eq, simple)
{
	cxit_create_eq();
	cr_assert(cxit_eq != NULL);
	cxit_destroy_eq();
}

void eq_bad_wait_obj(enum fi_wait_obj wait_obj)

{
	struct fi_eq_attr attr = {
		.size = 32,
		.flags = FI_WRITE,
		.wait_obj = wait_obj,
	};
	int ret;

	ret = fi_eq_open(cxit_fabric, &attr, &cxit_eq, NULL);
	cr_assert(ret == -FI_ENOSYS, "fi_eq_open unexpected success");
	cr_assert(cxit_eq == NULL, "cxit_eq not NULL on bad wait_obj");
}

Test(eq, bad_wait_obj_unspec)
{
	eq_bad_wait_obj(FI_WAIT_UNSPEC);
}

Test(eq, bad_wait_obj_wait_fd)
{
	eq_bad_wait_obj(FI_WAIT_UNSPEC);
}

