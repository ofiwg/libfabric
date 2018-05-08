/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>

#include <criterion/criterion.h>

#include "cxi_prov.h"
#include "cxi_test_common.h"

TestSuite(domain, .init = cxit_setup_domain, .fini = cxit_teardown_domain);

/* Test basic domain creation */
Test(domain, simple)
{
	cxit_create_domain();
	cr_assert(cxit_domain != NULL);

	cxit_destroy_domain();
}

