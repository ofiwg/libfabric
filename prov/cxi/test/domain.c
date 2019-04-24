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

