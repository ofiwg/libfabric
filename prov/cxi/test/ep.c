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

TestSuite(ep, .init = cxit_setup_ep, .fini = cxit_teardown_ep);

/* Test basic EP creation */
Test(ep, simple)
{
	cxit_create_ep();
	cr_assert(cxit_ep != NULL);

	cxit_destroy_ep();
}

