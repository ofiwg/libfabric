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

#ifndef RXD_UNIT_TESTS_H
#define RXD_UNIT_TESTS_H

#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <rdma/fabric.h>
#include "rxd.h"

/* rxd_info_to_core tests */
void test_rxd_info_to_core_hmem_passthrough(void **state);
void test_rxd_info_to_core_no_hmem(void **state);
void test_rxd_info_to_core_null_hints(void **state);

/* rxd_info_to_core_mr_modes tests */
void test_rxd_info_to_core_mr_modes_hmem(void **state);
void test_rxd_info_to_core_mr_modes_no_hmem(void **state);
void test_rxd_info_to_core_mr_modes_old_version(void **state);

/* rxd_info_to_rxd tests */
void test_rxd_info_to_rxd_core_has_hmem(void **state);
void test_rxd_info_to_rxd_core_lacks_hmem(void **state);

#endif /* RXD_UNIT_TESTS_H */
