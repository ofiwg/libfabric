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

TestSuite(cq, .init = cxit_setup_cq, .fini = cxit_teardown_cq);

/* Test basic CQ creation */
Test(cq, simple)
{
	cxit_create_cqs();
	cr_assert(cxit_tx_cq != NULL);
	cr_assert(cxit_rx_cq != NULL);

	cxit_destroy_cqs();
}
