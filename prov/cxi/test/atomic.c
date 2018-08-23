/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>

#include <ofi.h>

#include "cxip.h"
#include "cxip_test_common.h"

struct mem_region {
	uint8_t *mem;
	struct fid_mr *mr;
};

static void _mr_create(size_t len, uint64_t access, uint64_t key,
		       struct mem_region *mr)
{
	int ret;

	mr->mem = calloc(1, len);
	cr_assert_not_null(mr->mem);

	ret = fi_mr_reg(cxit_domain, mr->mem, len, access, 0, key, 0,
			&mr->mr, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_reg failed %d", ret);

	ret = fi_mr_bind(mr->mr, &cxit_ep->fid, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_bind failed %d", ret);

	ret = fi_mr_enable(mr->mr);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_enable failed %d", ret);
}

static void _mr_destroy(struct mem_region *mr)
{
	fi_close(&mr->mr->fid);
	free(mr->mem);
}

static void _await_completion(struct fi_cq_tagged_entry *cqe)
{
	int ret;

	do {
		ret = fi_cq_read(cxit_tx_cq, cqe, 1);
	} while (ret == -FI_EAGAIN);
	cr_assert(ret == 1);
}

TestSuite(atomic, .init = cxit_setup_rma, .fini = cxit_teardown_rma);

Test(atomic, simple_amo, .timeout = 10)
{
	int ret;
	struct mem_region mr;	/* Target memory region for RMA */
	uint64_t *rma_win;	/* Target buffer for RMA */
	uint64_t operand1;	/* Operand 1 buffer */
	int win_len = 0x1000;
	int key_val = 0;
	struct fi_cq_tagged_entry cqe;
	uint64_t expected = 0;

	_mr_create(win_len, FI_REMOTE_WRITE | FI_REMOTE_READ, key_val, &mr);
	rma_win = (uint64_t *)mr.mem;

	cr_assert_eq(*rma_win, expected,
		     "Result = %ld, expected = %ld",
		     *rma_win, expected);

	operand1 = 1;
	expected += operand1;
	ret = fi_atomic(cxit_ep, &operand1, 1, 0,
			0, 0, key_val,
			FI_UINT64, FI_SUM, NULL);
	cr_assert(ret == FI_SUCCESS, "Return code = %d", ret);
	_await_completion(&cqe);
	cr_assert_eq(*rma_win, expected,
		     "Result = %ld, expected = %ld",
		     *rma_win, expected);

	operand1 = 3;
	expected += operand1;
	ret = fi_atomic(cxit_ep, &operand1, 1, 0,
			0, 0, key_val,
			FI_UINT64, FI_SUM, NULL);
	cr_assert(ret == FI_SUCCESS, "Return code = %d", ret);
	_await_completion(&cqe);
	cr_assert_eq(*rma_win, expected,
		     "Result = %ld, expected = %ld",
		     *rma_win, expected);

	operand1 = 9;
	expected += operand1;
	ret = fi_atomic(cxit_ep, &operand1, 1, 0,
			0, 0, key_val,
			FI_UINT64, FI_SUM, NULL);
	cr_assert(ret == FI_SUCCESS, "Return code = %d", ret);
	_await_completion(&cqe);
	cr_assert_eq(*rma_win, expected,
		     "Result = %ld, expected = %ld",
		     *rma_win, expected);

	_mr_destroy(&mr);
}

Test(atomic, simple_fetch, .timeout = 10)
{
	int ret;
	struct mem_region mr;	/* Target memory region for RMA */
	uint64_t *rma_win;	/* Target buffer for RMA */
	uint64_t *result;	/* Local buffer for result */
	uint64_t operand1;
	int win_len = 0x1000;
	int key_val = 1;
	struct fi_cq_tagged_entry cqe;
	uint64_t expected = 0;
	uint64_t previous = 0;

	_mr_create(win_len, FI_REMOTE_WRITE | FI_REMOTE_READ, key_val, &mr);
	rma_win = (uint64_t *)mr.mem;

	result = calloc(1, win_len);
	cr_assert_not_null(result);

	cr_assert_eq(*rma_win, expected, "Result = %ld, expected = %ld",
		     *rma_win, expected);

	operand1 = 1;
	*result = -1;
	previous = expected;
	expected += operand1;
	ret = fi_fetch_atomic(cxit_ep, &operand1, 1, 0,
			      result, 0,
			      0, 0, key_val,
			      FI_UINT64, FI_SUM, NULL);
	cr_assert(ret == FI_SUCCESS, "Return code = %d", ret);
	_await_completion(&cqe);
	cr_assert_eq(*rma_win, expected,
		     "Add Result = %ld, expected = %ld",
		     *rma_win, expected);
	cr_assert_eq(*result, previous,
		     "Fetch Result = %016lx, expected = %016lx",
		     *result, previous);

	operand1 = 3;
	*result = -1;
	previous = expected;
	expected += operand1;
	ret = fi_fetch_atomic(cxit_ep, &operand1, 1, 0,
			      result, 0,
			      0, 0, key_val,
			      FI_UINT64, FI_SUM, NULL);
	cr_assert(ret == FI_SUCCESS, "Return code = %d", ret);
	_await_completion(&cqe);
	cr_assert_eq(*rma_win, expected,
		     "Add Result = %ld, expected = %ld",
		     *rma_win, expected);
	cr_assert_eq(*result, previous,
		     "Fetch Result = %016lx, expected = %016lx",
		     *result, previous);

	operand1 = 9;
	*result = -1;
	previous = expected;
	expected += operand1;
	ret = fi_fetch_atomic(cxit_ep, &operand1, 1, 0,
			      result, 0,
			      0, 0, key_val,
			      FI_UINT64, FI_SUM, NULL);
	cr_assert(ret == FI_SUCCESS, "Return code = %d", ret);
	_await_completion(&cqe);
	cr_assert_eq(*rma_win, expected,
		     "Add Result = %ld, expected = %ld",
		     *rma_win, expected);
	cr_assert_eq(*result, previous,
		     "Fetch Result = %016lx, expected = %016lx",
		     *result, previous);

	free(result);
	_mr_destroy(&mr);
}

Test(atomic, simple_swap, .timeout = 10)
{
	int ret;
	struct mem_region mr;	/* Target memory region for RMA */
	uint64_t *rma_win;	/* Target buffer for RMA */
	uint64_t *result;
	uint64_t operand1;
	uint64_t compare;
	int win_len = 0x1000;
	int key_val = 2;
	struct fi_cq_tagged_entry cqe;
	uint64_t exp_remote = 0;
	uint64_t exp_result = 0;

	_mr_create(win_len, FI_REMOTE_WRITE | FI_REMOTE_READ, key_val, &mr);
	rma_win = (uint64_t *)mr.mem;

	result = calloc(1, win_len);
	cr_assert_not_null(result);

	cr_assert_eq(*rma_win, exp_remote, "Result = %ld, expected = %ld",
		     *rma_win, exp_remote);

	*rma_win = 0;	/* remote == 0 */
	operand1 = 1;	/* change remote to 1 */
	compare = 2;	/* if remote != 2 (true) */
	*result = -1;	/* initialize result */
	exp_remote = 1;	/* expect remote == 1 */
	exp_result = 0;	/* expect result == 0 */
	ret = fi_compare_atomic(cxit_ep,
				&operand1, 1, 0,
				&compare, 0,
				result, 0,
				0, 0, key_val,
				FI_UINT64, FI_CSWAP_NE, NULL);
	cr_assert(ret == FI_SUCCESS, "Return code = %d", ret);
	_await_completion(&cqe);
	cr_assert_eq(*rma_win, exp_remote,
		     "Add Result = %ld, expected = %ld",
		     *rma_win, exp_remote);
	cr_assert_eq(*result, exp_result,
		     "Fetch Result = %016lx, expected = %016lx",
		     *result, exp_result);

	*rma_win = 2;	/* remote == 2 */
	operand1 = 1;	/* change remote to 1 */
	compare = 2;	/* if remote != 2 (false) */
	*result = -1;	/* initialize result */
	exp_remote = 2;	/* expect remote == 2 (no op) */
	exp_result = 2;	/* expect result == 2 (does return value) */
	ret = fi_compare_atomic(cxit_ep,
				&operand1, 1, 0,
				&compare, 0,
				result, 0,
				0, 0, key_val,
				FI_UINT64, FI_CSWAP_NE, NULL);
	cr_assert(ret == FI_SUCCESS, "Return code = %d", ret);
	_await_completion(&cqe);
	cr_assert_eq(*rma_win, exp_remote,
		     "Add Result = %ld, expected = %ld",
		     *rma_win, exp_remote);
	cr_assert_eq(*result, exp_result,
		     "Fetch Result = %016lx, expected = %016lx",
		     *result, exp_result);

	free(result);
	_mr_destroy(&mr);
}

