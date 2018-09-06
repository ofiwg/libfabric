/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <complex.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>

#include <ofi.h>

#include "cxip.h"
#include "cxip_test_common.h"

#define	AMO_DISABLED	false

struct mem_region {
	uint8_t *mem;
	struct fid_mr *mr;
};

#define RMA_WIN_LEN	64
#define RMA_WIN_KEY	2
#define RMA_WIN_ACCESS	(FI_REMOTE_READ | FI_REMOTE_WRITE)

/* Create MR -- works like a "remote_calloc()" */
static void *_cxit_create_mr(struct mem_region *mr)
{
	int ret;

	mr->mem = calloc(1, RMA_WIN_LEN);
	cr_assert_not_null(mr->mem);

	ret = fi_mr_reg(cxit_domain, mr->mem, RMA_WIN_LEN, RMA_WIN_ACCESS, 0,
			RMA_WIN_KEY, 0, &mr->mr, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_reg failed %d", ret);

	ret = fi_mr_bind(mr->mr, &cxit_ep->fid, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_bind failed %d", ret);

	ret = fi_mr_enable(mr->mr);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_enable failed %d", ret);

	return mr->mem;
}

/* Destroy MR -- works like a "remote_free()" */
static void _cxit_destroy_mr(struct mem_region *mr)
{
	fi_close(&mr->mr->fid);

	free(mr->mem);
}

/* Everyone needs to wait sometime */
static void _await_completion(struct fi_cq_tagged_entry *cqe)
{
	int ret;

	do {
		ret = fi_cq_read(cxit_tx_cq, cqe, 1);
	} while (ret == -FI_EAGAIN);
	cr_assert(ret == 1);
}

/* Test failures associated with bad call parameters.
 */
TestSuite(atomic_invalid, .init = cxit_setup_rma, .fini = cxit_teardown_rma,
	  .disabled = AMO_DISABLED);

Test(atomic_invalid, invalid_amo)
{
	uint64_t operand1 = 0;
	struct fi_ioc iov = {
		.addr = &operand1,
		.count = 1
	};
	int ret;

	ret = fi_atomic(cxit_ep, &operand1, 1, 0, 0, 0, 0,
			FI_UINT64, FI_ATOMIC_OP_LAST, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_atomic(cxit_ep, &operand1, 1, 0, 0, 0, 0,
			FI_UINT64, -1, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_atomic(cxit_ep, &operand1, 1, 0, 0, 0, 0,
			FI_DATATYPE_LAST, FI_SUM, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_atomic(cxit_ep, &operand1, 1, 0, 0, 0, 0,
			-1, FI_SUM, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_atomic(cxit_ep, &operand1, 0, 0, 0, 0, 0,
			FI_UINT64, FI_SUM, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_atomic(cxit_ep, &operand1, 2, 0, 0, 0, 0,
			FI_UINT64, FI_SUM, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_atomic(cxit_ep, 0, 1, 0, 0, 0, 0,
			FI_UINT64, FI_SUM, 0);
	cr_assert_eq(ret, -FI_EINVAL);

	ret = fi_atomicv(cxit_ep, &iov, 0, 0, 0, 0, 0,
			 FI_UINT64, FI_SUM, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_atomicv(cxit_ep, &iov, 0, 2, 0, 0, 0,
			 FI_UINT64, FI_SUM, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	iov.count = 0;
	ret = fi_atomicv(cxit_ep, &iov, 0, 1, 0, 0, 0,
			 FI_UINT64, FI_SUM, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	iov.count = 2;
	ret = fi_atomicv(cxit_ep, &iov, 0, 1, 0, 0, 0,
			 FI_UINT64, FI_SUM, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_atomicv(cxit_ep, 0, 0, 1, 0, 0, 0,
			 FI_UINT64, FI_SUM, 0);
	cr_assert_eq(ret, -FI_EINVAL);
}

Test(atomic_invalid, invalid_fetch)
{
	uint64_t operand1 = 0;
	uint64_t result = 0;
	struct fi_ioc iov = {
		.addr = &operand1,
		.count = 1
	};
	struct fi_ioc riov = {
		.addr = &result,
		.count = 1
	};
	int ret;

	ret = fi_fetch_atomic(cxit_ep, &operand1, 1, 0, &result, 0,
			      0, 0, 0, FI_UINT64, FI_ATOMIC_OP_LAST, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_fetch_atomic(cxit_ep, &operand1, 1, 0, &result, 0,
			      0, 0, 0, FI_UINT64, -1, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_fetch_atomic(cxit_ep, &operand1, 1, 0, &result, 0,
			      0, 0, 0, FI_DATATYPE_LAST, FI_SUM, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_fetch_atomic(cxit_ep, &operand1, 1, 0, &result, 0,
			      0, 0, 0, -1, FI_SUM, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_fetch_atomic(cxit_ep, &operand1, 1, 0, 0, 0,
			      0, 0, 0, FI_UINT64, FI_SUM, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_fetch_atomic(cxit_ep, &operand1, 0, 0, &result, 0,
			      0, 0, 0, FI_UINT64, FI_SUM, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_fetch_atomic(cxit_ep, &operand1, 2, 0, &result, 0,
			      0, 0, 0, FI_UINT64, FI_SUM, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_fetch_atomic(cxit_ep, 0, 1, 0, &result, 0,
			      0, 0, 0, FI_UINT64, FI_SUM, 0);
	cr_assert_eq(ret, -FI_EINVAL);


	ret = fi_fetch_atomicv(cxit_ep, &iov, 0, 1, &riov, 0, 0,
			       0, 0, 0, FI_UINT64, FI_SUM, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_fetch_atomicv(cxit_ep, &iov, 0, 1, &riov, 0, 2,
			       0, 0, 0, FI_UINT64, FI_SUM, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_fetch_atomicv(cxit_ep, &iov, 0, 1, 0, 0, 1,
			       0, 0, 0, FI_UINT64, FI_SUM, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_fetch_atomicv(cxit_ep, &iov, 0, 0, &riov, 0, 1,
			       0, 0, 0, FI_UINT64, FI_SUM, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_fetch_atomicv(cxit_ep, &iov, 0, 2, &riov, 0, 1,
			       0, 0, 0, FI_UINT64, FI_SUM, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_fetch_atomicv(cxit_ep, 0, 0, 1, &riov, 0, 1,
			       0, 0, 0, FI_UINT64, FI_SUM, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	riov.count = 0;
	ret = fi_fetch_atomicv(cxit_ep, &iov, 0, 1, &riov, 0, 1,
			       0, 0, 0, FI_UINT64, FI_SUM, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	riov.count = 2;
	ret = fi_fetch_atomicv(cxit_ep, &iov, 0, 1, &riov, 0, 1,
			       0, 0, 0, FI_UINT64, FI_SUM, 0);
	riov.count = 1;
	iov.count = 0;
	ret = fi_fetch_atomicv(cxit_ep, &iov, 0, 1, &riov, 0, 1,
			       0, 0, 0, FI_UINT64, FI_SUM, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	iov.count = 2;
	ret = fi_fetch_atomicv(cxit_ep, &iov, 0, 1, &riov, 0, 1,
			       0, 0, 0, FI_UINT64, FI_SUM, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	iov.count = 1;
	cr_assert_eq(ret, -FI_EINVAL);
}

Test(atomic_invalid, invalid_swap)
{
	uint64_t operand1 = 0;
	uint64_t compare = 0;
	uint64_t result = 0;
	struct fi_ioc iov = {
		.addr = &operand1,
		.count = 1
	};
	struct fi_ioc ciov = {
		.addr = &compare,
		.count = 1
	};
	struct fi_ioc riov = {
		.addr = &result,
		.count = 1
	};
	int ret;

	ret = fi_compare_atomic(cxit_ep,
				&operand1, 1, 0,
				&compare, 0,
				&result, 0,
				0, 0, 0,
				FI_UINT64, FI_ATOMIC_OP_LAST, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_compare_atomic(cxit_ep,
				&operand1, 1, 0,
				&compare, 0,
				&result, 0,
				0, 0, 0,
				FI_UINT64, -1, 0);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_compare_atomic(cxit_ep,
				&operand1, 1, 0,
				&compare, 0,
				&result, 0,
				0, 0, 0,
				FI_DATATYPE_LAST, FI_CSWAP_NE, NULL);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_compare_atomic(cxit_ep,
				&operand1, 1, 0,
				&compare, 0,
				&result, 0,
				0, 0, 0,
				-1, FI_CSWAP_NE, NULL);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_compare_atomic(cxit_ep,
				&operand1, 1, 0,
				&compare, 0,
				0, 0,
				0, 0, 0,
				FI_UINT64, FI_CSWAP_NE, NULL);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_compare_atomic(cxit_ep,
				&operand1, 1, 0,
				0, 0,
				&result, 0,
				0, 0, 0,
				FI_UINT64, FI_CSWAP_NE, NULL);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_compare_atomic(cxit_ep,
				&operand1, 2, 0,
				&compare, 0,
				&result, 0,
				0, 0, 0,
				FI_UINT64, FI_CSWAP_NE, NULL);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_compare_atomic(cxit_ep,
				&operand1, 0, 0,
				&compare, 0,
				&result, 0,
				0, 0, 0,
				FI_UINT64, FI_CSWAP_NE, NULL);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_compare_atomic(cxit_ep,
				0, 1, 0,
				&compare, 0,
				&result, 0,
				0, 0, 0,
				FI_UINT64, FI_CSWAP_NE, NULL);

	ret = fi_compare_atomicv(cxit_ep,
				&iov, 0, 1,
				&ciov, 0, 1,
				&riov, 0, 2,
				0, 0, 0,
				FI_UINT64, FI_CSWAP_NE, NULL);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_compare_atomicv(cxit_ep,
				&iov, 0, 1,
				&ciov, 0, 1,
				&riov, 0, 0,
				0, 0, 0,
				FI_UINT64, FI_CSWAP_NE, NULL);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_compare_atomicv(cxit_ep,
				&iov, 0, 1,
				&ciov, 0, 2,
				&riov, 0, 1,
				0, 0, 0,
				FI_UINT64, FI_CSWAP_NE, NULL);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_compare_atomicv(cxit_ep,
				&iov, 0, 1,
				&ciov, 0, 0,
				&riov, 0, 1,
				0, 0, 0,
				FI_UINT64, FI_CSWAP_NE, NULL);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_compare_atomicv(cxit_ep,
				&iov, 0, 2,
				&ciov, 0, 1,
				&riov, 0, 1,
				0, 0, 0,
				FI_UINT64, FI_CSWAP_NE, NULL);
	cr_assert_eq(ret, -FI_EINVAL);
	ret = fi_compare_atomicv(cxit_ep,
				&iov, 0, 0,
				&ciov, 0, 1,
				&riov, 0, 1,
				0, 0, 0,
				FI_UINT64, FI_CSWAP_NE, NULL);
	cr_assert_eq(ret, -FI_EINVAL);
	riov.count = 2;
	ret = fi_compare_atomicv(cxit_ep,
				&iov, 0, 1,
				&ciov, 0, 1,
				&riov, 0, 1,
				0, 0, 0,
				FI_UINT64, FI_CSWAP_NE, NULL);
	cr_assert_eq(ret, -FI_EINVAL);
	riov.count = 0;
	ret = fi_compare_atomicv(cxit_ep,
				&iov, 0, 1,
				&ciov, 0, 1,
				&riov, 0, 1,
				0, 0, 0,
				FI_UINT64, FI_CSWAP_NE, NULL);
	cr_assert_eq(ret, -FI_EINVAL);
	riov.count = 1;
	ciov.count = 2;
	ret = fi_compare_atomicv(cxit_ep,
				&iov, 0, 1,
				&ciov, 0, 1,
				&riov, 0, 1,
				0, 0, 0,
				FI_UINT64, FI_CSWAP_NE, NULL);
	cr_assert_eq(ret, -FI_EINVAL);
	ciov.count = 0;
	ret = fi_compare_atomicv(cxit_ep,
				&iov, 0, 1,
				&ciov, 0, 1,
				&riov, 0, 1,
				0, 0, 0,
				FI_UINT64, FI_CSWAP_NE, NULL);
	cr_assert_eq(ret, -FI_EINVAL);
	ciov.count = 1;
	iov.count = 2;
	ret = fi_compare_atomicv(cxit_ep,
				&iov, 0, 1,
				&ciov, 0, 1,
				&riov, 0, 1,
				0, 0, 0,
				FI_UINT64, FI_CSWAP_NE, NULL);
	cr_assert_eq(ret, -FI_EINVAL);
	iov.count = 0;
	ret = fi_compare_atomicv(cxit_ep,
				&iov, 0, 1,
				&ciov, 0, 1,
				&riov, 0, 1,
				0, 0, 0,
				FI_UINT64, FI_CSWAP_NE, NULL);
	cr_assert_eq(ret, -FI_EINVAL);
	iov.count = 1;
}

/* Test simple operations: AMO SUM UINT64_T, FAMO SUM UINT64_T, and CAMO SWAP_NE
 * UINT64_T. If this doesn't work, nothing else will.
 */
TestSuite(atomic, .init = cxit_setup_rma, .fini = cxit_teardown_rma,
	  .disabled = AMO_DISABLED);

Test(atomic, simple_amo, .timeout = 10)
{
	struct mem_region mr;
	struct fi_cq_tagged_entry cqe;
	uint64_t operand1;
	uint64_t exp_remote = 0;
	uint64_t *rma;
	int ret;

	rma = _cxit_create_mr(&mr);
	cr_assert_eq(*rma, exp_remote,
		     "Result = %ld, expected = %ld",
		     *rma, exp_remote);

	operand1 = 1;
	exp_remote += operand1;
	ret = fi_atomic(cxit_ep, &operand1, 1, 0,
			0, 0, RMA_WIN_KEY,
			FI_UINT64, FI_SUM, NULL);
	cr_assert(ret == FI_SUCCESS, "Return code  = %d", ret);
	_await_completion(&cqe);
	cr_assert_eq(*rma, exp_remote,
		     "Result = %ld, expected = %ld",
		     *rma, exp_remote);

	operand1 = 3;
	exp_remote += operand1;
	ret = fi_atomic(cxit_ep, &operand1, 1, 0,
			0, 0, RMA_WIN_KEY,
			FI_UINT64, FI_SUM, NULL);
	cr_assert(ret == FI_SUCCESS, "Return code = %d", ret);
	_await_completion(&cqe);
	cr_assert_eq(*rma, exp_remote,
		     "Result = %ld, expected = %ld",
		     *rma, exp_remote);

	operand1 = 9;
	exp_remote += operand1;
	ret = fi_atomic(cxit_ep, &operand1, 1, 0,
			0, 0, RMA_WIN_KEY,
			FI_UINT64, FI_SUM, NULL);
	cr_assert(ret == FI_SUCCESS, "Return code = %d", ret);
	_await_completion(&cqe);
	cr_assert_eq(*rma, exp_remote,
		     "Result = %ld, expected = %ld",
		     *rma, exp_remote);

	_cxit_destroy_mr(&mr);
}

Test(atomic, simple_fetch, .timeout = 10)
{
	struct mem_region mr;
	struct fi_cq_tagged_entry cqe;
	uint64_t operand1;
	uint64_t exp_remote = 0;
	uint64_t exp_result = 0;
	uint64_t *rma;
	uint64_t *loc;
	int ret;

	rma = _cxit_create_mr(&mr);
	cr_assert_eq(*rma, exp_remote,
		     "Result = %ld, expected = %ld",
		     *rma, exp_remote);

	loc = calloc(1, RMA_WIN_LEN);
	cr_assert_not_null(loc);

	operand1 = 1;
	*loc = -1;
	exp_result = exp_remote;
	exp_remote += operand1;
	ret = fi_fetch_atomic(cxit_ep, &operand1, 1, 0,
			      loc, 0,
			      0, 0, RMA_WIN_KEY,
			      FI_UINT64, FI_SUM, NULL);
	cr_assert(ret == FI_SUCCESS, "Return code = %d", ret);
	_await_completion(&cqe);
	cr_assert_eq(*rma, exp_remote,
		     "Add Result = %ld, expected = %ld",
		     *rma, exp_remote);
	cr_assert_eq(*loc, exp_result,
		     "Fetch Result = %016lx, expected = %016lx",
		     *loc, exp_result);

	operand1 = 3;
	*loc = -1;
	exp_result = exp_remote;
	exp_remote += operand1;
	ret = fi_fetch_atomic(cxit_ep, &operand1, 1, 0,
			      loc, 0,
			      0, 0, RMA_WIN_KEY,
			      FI_UINT64, FI_SUM, NULL);
	cr_assert(ret == FI_SUCCESS, "Return code = %d", ret);
	_await_completion(&cqe);
	cr_assert_eq(*rma, exp_remote,
		     "Add Result = %ld, expected = %ld",
		     *rma, exp_remote);
	cr_assert_eq(*loc, exp_result,
		     "Fetch Result = %016lx, expected = %016lx",
		     *loc, exp_result);

	operand1 = 9;
	*loc = -1;
	exp_result = exp_remote;
	exp_remote += operand1;
	ret = fi_fetch_atomic(cxit_ep, &operand1, 1, 0,
			      loc, 0,
			      0, 0, RMA_WIN_KEY,
			      FI_UINT64, FI_SUM, NULL);
	cr_assert(ret == FI_SUCCESS, "Return code = %d", ret);
	_await_completion(&cqe);
	cr_assert_eq(*rma, exp_remote,
		     "Add Result = %ld, expected = %ld",
		     *rma, exp_remote);
	cr_assert_eq(*loc, exp_result,
		     "Fetch Result = %016lx, expected = %016lx",
		     *loc, exp_result);

	free(loc);
	_cxit_destroy_mr(&mr);
}

Test(atomic, simple_swap, .timeout = 10)
{
	struct mem_region mr;
	struct fi_cq_tagged_entry cqe;
	uint64_t operand1;
	uint64_t compare;
	uint64_t exp_remote = 0;
	uint64_t exp_result = 0;
	uint64_t *rma;
	uint64_t *loc;
	int ret;

	rma = _cxit_create_mr(&mr);
	cr_assert_eq(*rma, exp_remote,
		     "Result = %ld, expected = %ld",
		     *rma, exp_remote);

	loc = calloc(1, RMA_WIN_LEN);
	cr_assert_not_null(loc);

	*rma = 0;	/* remote == 0 */
	operand1 = 1;	/* change remote to 1 */
	compare = 2;	/* if remote != 2 (true) */
	*loc = -1;	/* initialize result */
	exp_remote = 1;	/* expect remote == 1 */
	exp_result = 0;	/* expect result == 0 */
	ret = fi_compare_atomic(cxit_ep,
				&operand1, 1, 0,
				&compare, 0,
				loc, 0,
				0, 0, RMA_WIN_KEY,
				FI_UINT64, FI_CSWAP_NE, NULL);
	cr_assert(ret == FI_SUCCESS, "Return code = %d", ret);
	_await_completion(&cqe);
	cr_assert_eq(*rma, exp_remote,
		     "Add Result = %ld, expected = %ld",
		     *rma, exp_remote);
	cr_assert_eq(*loc, exp_result,
		     "Fetch Result = %016lx, expected = %016lx",
		     *loc, exp_result);

	*rma = 2;	/* remote == 2 */
	operand1 = 1;	/* change remote to 1 */
	compare = 2;	/* if remote != 2 (false) */
	*loc = -1;	/* initialize result */
	exp_remote = 2;	/* expect remote == 2 (no op) */
	exp_result = 2;	/* expect result == 2 (does return value) */
	ret = fi_compare_atomic(cxit_ep,
				&operand1, 1, 0,
				&compare, 0,
				loc, 0,
				0, 0, RMA_WIN_KEY,
				FI_UINT64, FI_CSWAP_NE, NULL);
	cr_assert(ret == FI_SUCCESS, "Return code = %d", ret);
	_await_completion(&cqe);
	cr_assert_eq(*rma, exp_remote,
		     "Add Result = %ld, expected = %ld",
		     *rma, exp_remote);
	cr_assert_eq(*loc, exp_result,
		     "Fetch Result = %016lx, expected = %016lx",
		     *loc, exp_result);

	free(loc);
	_cxit_destroy_mr(&mr);
}

/* Perform a full combinatorial test suite.
 */
#define	MAX_TEST_SIZE	16

/**
 * Compare a seen value with an expected value, with 'len' valid bytes. This
 * checks the seen buffer all the way to MAX_TEST_SIZE, and looks for a
 * predefined value in every byte, to ensure that there is no overflow.
 * The seen buffer will always be either the rma or the loc buffer, which have
 * 64 bytes of space in them.
 *
 * Summation of real and complex types is trickier. Every decimal constant is
 * internally represented by a binary approximation, and summation can
 * accumulate errors. With only a single sum with two arguments, the error could
 * be +1 or -1 in the LSBit.
 *
 * @param saw 'seen' buffer
 * @param exp 'expected' value
 * @param len number of valid bytes
 *
 * @return bool true if successful, false if comparison fails
 */
static bool _compare(void *saw, void *exp, int len,
		     enum fi_op op, enum fi_datatype dt)
{
	uint8_t *bval = saw;
	uint8_t *bexp = exp;
	uint64_t uval = 0;
	uint64_t uexp = 0;
	int i;

	/* Test MS pad bits */
	for (i = MAX_TEST_SIZE-1; i >= len; i--) {
		if (bval[i] != bexp[i])
			return false;
	}
	if (op == FI_SUM) {
		switch (dt) {
		case FI_FLOAT:
		case FI_DOUBLE:
			/* Copy to UINT64, adjust diff (-1,1) to (0,2) */
			memcpy(&uval, bval, len);
			memcpy(&uexp, bexp, len);
			if ((uval - uexp) + 1 > 2)
				return false;
			return true;
		case FI_FLOAT_COMPLEX:
		case FI_DOUBLE_COMPLEX:
			/* Do real and imag parts separately */
			memcpy(&uval, bval, len/2);
			memcpy(&uexp, bexp, len/2);
			if (uval - uexp + 1 > 2)
				return false;
			memcpy(&uval, bval+len/2, len/2);
			memcpy(&uexp, bexp+len/2, len/2);
			if (uval - uexp + 1 > 2)
				return false;
			return true;
		default:
			break;
		}
	}
	/* Test LS value bits */
	for (i = len-1; i >= 0; i--) {
		if (bval[i] != bexp[i])
			return false;
	}
	return true;
}

/**
 * Generates a useful error message.
 *
 * @param op opcode
 * @param dt dtcode
 * @param saw 'seen' buffer
 * @param exp 'expected' value
 * @param len number of valid bytes
 * @param buf buffer to fill with message
 * @param siz buffer size
 *
 * @return const char* returns the buf pointer
 */
static const char *_errmsg(enum fi_op op, enum fi_datatype dt,
			   void *saw, void *exp, int len,
			   char *buf, size_t siz)
{
	char *p = &buf[0];
	char *e = &buf[siz];
	uint8_t *bsaw = saw;
	uint8_t *bexp = exp;
	int i;

	p += snprintf(p, e-p, "%d:%d: saw=", op, dt);
	for (i = MAX_TEST_SIZE-1; i >= 0; i--)
		p += snprintf(p, e-p, "%02x%s", bsaw[i], i == len ? "/" : "");
	p += snprintf(p, e-p, " exp=");
	for (i = MAX_TEST_SIZE-1; i >= 0; i--)
		p += snprintf(p, e-p, "%02x%s", bexp[i], i == len ? "/" : "");
	return buf;
}

/**
 * The general AMO test.
 *
 * @param index value used to help identify the test if error
 * @param dt FI datatype
 * @param op FI operation
 * @param err 0 if success expected, 1 if failure expected
 * @param operand1 operation data value pointer
 * @param compare operation compare value pointer
 * @param loc operation result (local) buffer pointer
 * @param loc_init operation result initialization value pointer
 * @param rma operation rma (remote) buffer pointer
 * @param rma_init operation rma initialization value pointer
 * @param rma_expect operation rma (remote) expected value pointer
 */
static void _test_amo(int index, enum fi_datatype dt, enum fi_op op, int err,
		      void *operand1,
		      void *compare,
		      void *loc, void *loc_init,
		      void *rma, void *rma_init, void *rma_expect)
{
	struct fi_cq_tagged_entry cqe;
	char msgbuf[128];
	char opstr[64];
	char dtstr[64];
	uint8_t rexp[MAX_TEST_SIZE];
	uint8_t lexp[MAX_TEST_SIZE];
	void *rma_exp = rexp;
	void *loc_exp = lexp;
	int len = ofi_datatype_size(dt);
	int ret;

	strcpy(opstr, fi_tostr(&op, FI_TYPE_ATOMIC_OP));
	strcpy(dtstr, fi_tostr(&dt, FI_TYPE_ATOMIC_TYPE));

	cr_log_info("Testing %s %s (%d)\n", opstr, dtstr, len);

	memset(rma, -1, MAX_TEST_SIZE);
	memset(rma_exp, -1, MAX_TEST_SIZE);
	memcpy(rma, rma_init, len);
	memcpy(rma_exp, rma_expect, len);

	if (loc && loc_init) {
		memset(loc, -1, MAX_TEST_SIZE);
		memset(loc_exp, -1, MAX_TEST_SIZE);
		memcpy(loc, loc_init, len);
		memcpy(loc_exp, rma_init, len);
	}
	if (compare && loc) {
		/* This is a compare command */
		ret = fi_compare_atomic(cxit_ep, operand1, 1, 0,
					compare, 0, loc, 0,
					0, 0, RMA_WIN_KEY, dt, op, NULL);
	} else if (loc) {
		/* This is a fetch command */
		ret = fi_fetch_atomic(cxit_ep, operand1, 1, 0, loc, 0,
				      0, 0, RMA_WIN_KEY, dt, op, NULL);
	} else {
		/* This is a simple command */
		ret = fi_atomic(cxit_ep, operand1, 1, 0,
				0, 0, RMA_WIN_KEY, dt, op, NULL);
	}

	if (err) {
		/* Expected an error. Tests only invoke "unsupported" failures,
		 * so any other error is fatal. Success is also fatal if we
		 * expect a failure.
		 */
		cr_assert_eq(ret, -FI_EOPNOTSUPP,
			     "rtn #%d:%d:%d saw=%d exp=%d\n",
			     index, op, dt, ret, -FI_EOPNOTSUPP);
		return;
	}


	/* If we weren't expecting an error, any error is fatal  */
	cr_assert_eq(ret, 0,
		     "rtn #%d:%d:%d saw=%d exp=%d\n",
		     index, op, dt, ret, err);

	_await_completion(&cqe);

	/* We expect the RMA effect to be as predicted */
	cr_expect(_compare(rma, rma_exp, len, op, dt),
		  "rma #%d:%s\n", index,
		  _errmsg(op, dt, rma, rma_exp, len, msgbuf,
			  sizeof(msgbuf)));

	/* We expect the local result to be as predicted, if there is one */
	if (loc && loc_init) {
		cr_expect(_compare(loc, loc_exp, len, op, dt),
			  "loc #%d:%s\n", index,
			  _errmsg(op, dt, loc, loc_exp, len, msgbuf,
				  sizeof(msgbuf)));
	}
}

/* Every parameter list can create an OR of the following values, to indicate
 * what forms should be attempted.
 */
#define	_AMO	1
#define	_FAMO	2
#define	_CAMO	4

/* The INT tests test 8, 16, 32, and 64 bits for each line item.
 */
struct test_int_parms {
	int opmask;
	int index;
	enum fi_op op;
	int err;
	uint64_t comp;
	uint64_t o1;
	uint64_t rini;
	uint64_t rexp;
};

static struct test_int_parms int_parms[] = {
	{ _AMO|_FAMO, 11, FI_MIN,  0, 0, 123, 120, 120 },
	{ _AMO|_FAMO, 12, FI_MIN,  0, 0, 120, 123, 120 },
	{ _AMO|_FAMO, 21, FI_MAX,  0, 0, 123, 120, 123 },
	{ _AMO|_FAMO, 22, FI_MAX,  0, 0, 120, 123, 123 },
	{ _AMO|_FAMO, 31, FI_SUM,  0, 0,   1,   0,   1 },
	{ _AMO|_FAMO, 32, FI_SUM,  0, 0,   1,  10,  11 },
	{ _AMO|_FAMO, 33, FI_SUM,  0, 0,   2,  -1,   1 },
	{ _AMO|_FAMO, 41, FI_LOR,  0, 0,   0,   0,   0 },
	{ _AMO|_FAMO, 42, FI_LOR,  0, 0, 128,   0,   1 },
	{ _AMO|_FAMO, 43, FI_LOR,  0, 0,   0, 128,   1 },
	{ _AMO|_FAMO, 44, FI_LOR,  0, 0,  64, 128,   1 },
	{ _AMO|_FAMO, 51, FI_LAND, 0, 0,   0,   0,   0 },
	{ _AMO|_FAMO, 52, FI_LAND, 0, 0, 128,   0,   0 },
	{ _AMO|_FAMO, 53, FI_LAND, 0, 0,   0, 128,   0 },
	{ _AMO|_FAMO, 54, FI_LAND, 0, 0,  64, 128,   1 },
	{ _AMO|_FAMO, 61, FI_LXOR, 0, 0,   0,   0,   0 },
	{ _AMO|_FAMO, 62, FI_LXOR, 0, 0, 128,   0,   1 },
	{ _AMO|_FAMO, 63, FI_LXOR, 0, 0,   0, 128,   1 },
	{ _AMO|_FAMO, 64, FI_LXOR, 0, 0,  64, 128,   0 },
	{ _AMO|_FAMO, 71, FI_BOR,  0, 0,
		0xf0e1f2e3f4e5f6e7,
		0x1818181818181818,
		0xf8f9fafbfcfdfeff },
	{ _AMO|_FAMO, 81, FI_BAND, 0, 0,
		0xf0e1f2e3f4e5f6e7,
		0x1818181818181818,
		0x1000100010001000 },
	{ _AMO|_FAMO, 91, FI_BXOR, 0, 0,
		0xf0e1f2e3f4e5f6e7,
		0x1818181818181818,
		0xe8f9eafbecfdeeff },
	{ _AMO|_FAMO, 101, FI_ATOMIC_WRITE, 0, 0,
		0x1234123412341234,
		0xabcdabcdabcdabcd,
		0x1234123412341234 },
	{ _FAMO, 111, FI_ATOMIC_READ, 0, 0,
		0x1010101010101010,
		0x4321432143214321,
		0x4321432143214321 },
	{ _AMO, 112, FI_ATOMIC_READ, 1 },
	{ _CAMO, 121, FI_CSWAP,     0, 120, 123, 100, 100 },
	{ _CAMO, 122, FI_CSWAP,     0, 100, 123, 100, 123 },
	{ _CAMO, 131, FI_CSWAP_NE,  0, 120, 123, 100, 123 },
	{ _CAMO, 132, FI_CSWAP_NE,  0, 100, 123, 100, 100 },
	{ _CAMO, 141, FI_CSWAP_LE,  0, 101, 123, 100, 100 },
	{ _CAMO, 142, FI_CSWAP_LE,  0, 100, 123, 100, 123 },
	{ _CAMO, 143, FI_CSWAP_LE,  0,  99, 123, 100, 123 },
	{ _CAMO, 151, FI_CSWAP_LT,  0, 101, 123, 100, 100 },
	{ _CAMO, 152, FI_CSWAP_LT,  0, 100, 123, 100, 100 },
	{ _CAMO, 153, FI_CSWAP_LT,  0,  99, 123, 100, 123 },
	{ _CAMO, 161, FI_CSWAP_GE,  0, 101, 123, 100, 123 },
	{ _CAMO, 162, FI_CSWAP_GE,  0, 100, 123, 100, 123 },
	{ _CAMO, 163, FI_CSWAP_GE,  0,  99, 123, 100, 100 },
	{ _CAMO, 171, FI_CSWAP_GT,  0, 101, 123, 100, 123 },
	{ _CAMO, 173, FI_CSWAP_GT,  0, 100, 123, 100, 100 },
	{ _CAMO, 173, FI_CSWAP_GT,  0,  99, 123, 100, 100 },
	{ _CAMO, 181, FI_MSWAP,     0,
		0xf0f0f0f0f0f0f0f0,
		0xaaaaaaaaaaaaaaaa,
		0x1111111111111111,
		0xa1a1a1a1a1a1a1a1
	},
};

ParameterizedTestParameters(atomic, test_int)
{
	return cr_make_param_array(struct test_int_parms, int_parms,
				   ARRAY_SIZE(int_parms));
}

ParameterizedTest(struct test_int_parms *p, atomic, test_int)
{
	struct mem_region mr;
	enum fi_datatype dt;
	uint64_t *rma;
	uint64_t *loc;
	uint64_t lini = -1;

	rma = _cxit_create_mr(&mr);

	loc = calloc(1, RMA_WIN_LEN);
	cr_assert_not_null(loc);

	if (p->opmask & _AMO) {
		for (dt = FI_INT8; dt <= FI_UINT64; dt++) {
			_test_amo(p->index, dt, p->op, p->err, &p->o1,
				  0, 0, 0,
				  rma, &p->rini, &p->rexp);
		}
	}

	if (p->opmask & _FAMO) {
		for (dt = FI_INT8; dt <= FI_UINT64; dt++) {
			_test_amo(p->index, dt, p->op, p->err, &p->o1,
				  0, loc, &lini, rma, &p->rini, &p->rexp);
		}
	}

	if (p->opmask & _CAMO) {
		for (dt = FI_INT8; dt <= FI_UINT64; dt++) {
			_test_amo(p->index, dt, p->op, p->err, &p->o1,
				  &p->comp, loc, &lini, rma, &p->rini,
				  &p->rexp);
		}
	}

	free(loc);
	_cxit_destroy_mr(&mr);
}

/* The FLT tests only test the float type.
 */
struct test_flt_parms {
	int opmask;
	int index;
	enum fi_op op;
	int err;
	float comp;
	float o1;
	float rini;
	float rexp;
};

static struct test_flt_parms flt_parms[] = {
	{ _AMO|_FAMO, 11, FI_MIN,  0, 0.0f, 12.3f, 12.0f, 12.0f },
	{ _AMO|_FAMO, 12, FI_MIN,  0, 0.0f, 12.0f, 12.3f, 12.0f },
	{ _AMO|_FAMO, 21, FI_MAX,  0, 0.0f, 12.3f, 12.0f, 12.3f },
	{ _AMO|_FAMO, 22, FI_MAX,  0, 0.0f, 12.0f, 12.3f, 12.3f },
	{ _AMO|_FAMO, 31, FI_SUM,  0, 0.0f,  1.1f,  1.2f, (1.1f + 1.2f) },
	{ _AMO|_FAMO, 32, FI_SUM,  0, 0.0f,  0.4f,  1.7f, (0.4f + 1.7f) },
	{ _AMO|_FAMO, 41, FI_LOR,  1 },
	{ _AMO|_FAMO, 51, FI_LAND, 1 },
	{ _AMO|_FAMO, 61, FI_LXOR, 1 },
	{ _AMO|_FAMO, 71, FI_BOR,  1 },
	{ _AMO|_FAMO, 81, FI_BAND, 1 },
	{ _AMO|_FAMO, 91, FI_BXOR, 1 },
	{ _AMO|_FAMO, 101, FI_ATOMIC_WRITE, 0, 0.0f, 10.2f, 96.6f, 10.2f },
	{ _FAMO, 111, FI_ATOMIC_READ, 0, 0.0f, 1.1f, 10.2f, 10.2f },
	{ _AMO,  112, FI_ATOMIC_READ, 1 },
	{ _CAMO, 121, FI_CSWAP,     0, 12.0f, 12.3f, 10.0f, 10.0f },
	{ _CAMO, 122, FI_CSWAP,     0, 10.0f, 12.3f, 10.0f, 12.3f },
	{ _CAMO, 131, FI_CSWAP_NE,  0, 12.0f, 12.3f, 10.0f, 12.3f },
	{ _CAMO, 132, FI_CSWAP_NE,  0, 10.0f, 12.3f, 10.0f, 10.0f },
	{ _CAMO, 141, FI_CSWAP_LE,  0, 10.1f, 12.3f, 10.0f, 10.0f },
	{ _CAMO, 142, FI_CSWAP_LE,  0, 10.0f, 12.3f, 10.0f, 12.3f },
	{ _CAMO, 143, FI_CSWAP_LE,  0,  9.9f, 12.3f, 10.0f, 12.3f },
	{ _CAMO, 151, FI_CSWAP_LT,  0, 10.1f, 12.3f, 10.0f, 10.0f },
	{ _CAMO, 152, FI_CSWAP_LT,  0, 10.0f, 12.3f, 10.0f, 10.0f },
	{ _CAMO, 153, FI_CSWAP_LT,  0,  9.9f, 12.3f, 10.0f, 12.3f },
	{ _CAMO, 161, FI_CSWAP_GE,  0, 10.1f, 12.3f, 10.0f, 12.3f },
	{ _CAMO, 162, FI_CSWAP_GE,  0, 10.0f, 12.3f, 10.0f, 12.3f },
	{ _CAMO, 163, FI_CSWAP_GE,  0,  9.9f, 12.3f, 10.0f, 10.0f },
	{ _CAMO, 171, FI_CSWAP_GT,  0, 10.1f, 12.3f, 10.0f, 12.3f },
	{ _CAMO, 172, FI_CSWAP_GT,  0, 10.0f, 12.3f, 10.0f, 10.0f },
	{ _CAMO, 173, FI_CSWAP_GT,  0,  9.9f, 12.3f, 10.0f, 10.0f },
	{ _CAMO, 181, FI_MSWAP,     1 },
};

ParameterizedTestParameters(atomic, test_flt)
{
	return cr_make_param_array(struct test_flt_parms, flt_parms,
				   ARRAY_SIZE(flt_parms));
}

ParameterizedTest(struct test_flt_parms *p, atomic, test_flt)
{
	struct mem_region mr;
	enum fi_datatype dt = FI_FLOAT;
	uint64_t *rma;
	uint64_t *loc;
	uint64_t lini = -1;

	rma = _cxit_create_mr(&mr);

	loc = calloc(1, RMA_WIN_LEN);
	cr_assert_not_null(loc);

	if (p->opmask & _AMO) {
		_test_amo(p->index, dt, p->op, p->err, &p->o1,
			  0, 0, 0,
			  rma, &p->rini, &p->rexp);
	}

	if (p->opmask & _FAMO) {
		_test_amo(p->index, dt, p->op, p->err, &p->o1,
			  0, loc, &lini, rma, &p->rini, &p->rexp);
	}

	if (p->opmask & _CAMO) {
		_test_amo(p->index, dt, p->op, p->err, &p->o1,
			  &p->comp, loc, &lini, rma, &p->rini,
			  &p->rexp);
	}

	free(loc);
	_cxit_destroy_mr(&mr);
}

/* The DBL tests only test the double type.
 */
struct test_dbl_parms {
	int opmask;
	int index;
	enum fi_op op;
	int err;
	double comp;
	double o1;
	double rini;
	double rexp;
};

static struct test_dbl_parms dbl_parms[] = {
	{ _AMO|_FAMO, 11, FI_MIN,  0, 0.0, 12.3, 12.0, 12.0 },
	{ _AMO|_FAMO, 12, FI_MIN,  0, 0.0, 12.0, 12.3, 12.0 },
	{ _AMO|_FAMO, 21, FI_MAX,  0, 0.0, 12.3, 12.0, 12.3 },
	{ _AMO|_FAMO, 22, FI_MAX,  0, 0.0, 12.0, 12.3, 12.3 },
	{ _AMO|_FAMO, 31, FI_SUM,  0, 0.0,  1.1,  1.2, (1.1 + 1.2) },
	{ _AMO|_FAMO, 32, FI_SUM,  0, 0.0,  0.4,  1.7, (0.4 + 1.7) },
	{ _AMO|_FAMO, 41, FI_LOR,  1 },
	{ _AMO|_FAMO, 51, FI_LAND, 1 },
	{ _AMO|_FAMO, 61, FI_LXOR, 1 },
	{ _AMO|_FAMO, 71, FI_BOR,  1 },
	{ _AMO|_FAMO, 81, FI_BAND, 1 },
	{ _AMO|_FAMO, 91, FI_BXOR, 1 },
	{ _AMO|_FAMO, 101, FI_ATOMIC_WRITE, 0, 0.0, 10.2, 123.4, 10.2 },
	{ _FAMO, 111, FI_ATOMIC_READ, 0, 0.0, 1.1, 10.2, 10.2 },
	{ _AMO,  112, FI_ATOMIC_READ, 1 },
	{ _CAMO, 121, FI_CSWAP,     0, 12.0, 12.3, 10.0, 10.0 },
	{ _CAMO, 122, FI_CSWAP,     0, 10.0, 12.3, 10.0, 12.3 },
	{ _CAMO, 131, FI_CSWAP_NE,  0, 12.0, 12.3, 10.0, 12.3 },
	{ _CAMO, 132, FI_CSWAP_NE,  0, 10.0, 12.3, 10.0, 10.0 },
	{ _CAMO, 141, FI_CSWAP_LE,  0, 10.1, 12.3, 10.0, 10.0 },
	{ _CAMO, 142, FI_CSWAP_LE,  0, 10.0, 12.3, 10.0, 12.3 },
	{ _CAMO, 143, FI_CSWAP_LE,  0,  9.9, 12.3, 10.0, 12.3 },
	{ _CAMO, 151, FI_CSWAP_LT,  0, 10.1, 12.3, 10.0, 10.0 },
	{ _CAMO, 152, FI_CSWAP_LT,  0, 10.0, 12.3, 10.0, 10.0 },
	{ _CAMO, 153, FI_CSWAP_LT,  0,  9.9, 12.3, 10.0, 12.3 },
	{ _CAMO, 161, FI_CSWAP_GE,  0, 10.1, 12.3, 10.0, 12.3 },
	{ _CAMO, 162, FI_CSWAP_GE,  0, 10.0, 12.3, 10.0, 12.3 },
	{ _CAMO, 163, FI_CSWAP_GE,  0,  9.9, 12.3, 10.0, 10.0 },
	{ _CAMO, 171, FI_CSWAP_GT,  0, 10.1, 12.3, 10.0, 12.3 },
	{ _CAMO, 172, FI_CSWAP_GT,  0, 10.0, 12.3, 10.0, 10.0 },
	{ _CAMO, 173, FI_CSWAP_GT,  0,  9.9, 12.3, 10.0, 10.0 },
	{ _CAMO, 181, FI_MSWAP,     1 },
};

ParameterizedTestParameters(atomic, test_dbl)
{
	return cr_make_param_array(struct test_dbl_parms, dbl_parms,
				   ARRAY_SIZE(dbl_parms));
}

ParameterizedTest(struct test_dbl_parms *p, atomic, test_dbl)
{
	struct mem_region mr;
	enum fi_datatype dt = FI_DOUBLE;
	uint64_t *rma;
	uint64_t *loc;
	uint64_t lini = -1;

	rma = _cxit_create_mr(&mr);

	loc = calloc(1, RMA_WIN_LEN);
	cr_assert_not_null(loc);

	if (p->opmask & _AMO) {
		_test_amo(p->index, dt, p->op, p->err, &p->o1,
			  0, 0, 0,
			  rma, &p->rini, &p->rexp);
	}

	if (p->opmask & _FAMO) {
		_test_amo(p->index, dt, p->op, p->err, &p->o1,
			  0, loc, &lini, rma, &p->rini, &p->rexp);
	}

	if (p->opmask & _CAMO) {
		_test_amo(p->index, dt, p->op, p->err, &p->o1,
			  &p->comp, loc, &lini, rma, &p->rini,
			  &p->rexp);
	}

	free(loc);
	_cxit_destroy_mr(&mr);
}

/* The CMPLX tests only test the float complex type.
 */
struct test_cplx_parms {
	int opmask;
	int index;
	enum fi_op op;
	int err;

	float complex comp;
	float complex o1;
	float complex rini;
	float complex rexp;

};

static struct test_cplx_parms cplx_parms[] = {
	{ _AMO|_FAMO, 11, FI_MIN,  1 },
	{ _AMO|_FAMO, 21, FI_MAX,  1 },
	{ _AMO|_FAMO, 31, FI_SUM,  0, 0.0,  1.1,  1.2, (1.1 + 1.2) },
	{ _AMO|_FAMO, 32, FI_SUM,  0, 0.0,  0.4,  1.7, (0.4 + 1.7) },
	{ _AMO|_FAMO, 31, FI_SUM,  0,
		0.0f, 1.1f+I*0.4f, 1.2f+I*1.7f, (1.1f+I*0.4f + 1.2f+I*1.7f) },
	{ _AMO|_FAMO, 32, FI_SUM,  0,
		0.0f, 1.1f+I*1.7f, 1.2f+I*0.4f, (1.1f+I*1.7f + 1.2f+I*0.4f) },
	{ _AMO|_FAMO, 41, FI_LOR,  1 },
	{ _AMO|_FAMO, 51, FI_LAND, 1 },
	{ _AMO|_FAMO, 61, FI_LXOR, 1 },
	{ _AMO|_FAMO, 71, FI_BOR,  1 },
	{ _AMO|_FAMO, 81, FI_BAND, 1 },
	{ _AMO|_FAMO, 91, FI_BXOR, 1 },
	{ _AMO|_FAMO, 101, FI_ATOMIC_WRITE, 0,
		0.0f, 10.2f+I*1.1f, 0.3f+I*2.2f, 10.2f+I*1.1f },
	{ _FAMO, 111, FI_ATOMIC_READ, 0,
		0.0f, 1.1f+I*1.1f, 10.2f+I*1.1f, 10.2f+I*1.1f },
	{ _AMO,  112, FI_ATOMIC_READ, 1 },
	{ _CAMO, 121, FI_CSWAP,     0,
		12.0f+I*1.1f, 12.3f+I*1.1f, 10.0f+I*1.1f, 10.0f+I*1.1f },
	{ _CAMO, 122, FI_CSWAP,     0,
		10.0f+I*1.1f, 12.3f+I*1.1f, 10.0f+I*1.1f, 12.3f+I*1.1f },
	{ _CAMO, 131, FI_CSWAP_NE,  0,
		12.0f+I*1.1f, 12.3f+I*1.1f, 10.0f+I*1.1f, 12.3f+I*1.1f },
	{ _CAMO, 132, FI_CSWAP_NE,  0,
		10.0f+I*1.1f, 12.3f+I*1.1f, 10.0f+I*1.1f, 10.0f+I*1.1f },
	{ _CAMO, 141, FI_CSWAP_LE,  1 },
	{ _CAMO, 151, FI_CSWAP_LT,  1 },
	{ _CAMO, 161, FI_CSWAP_GE,  1 },
	{ _CAMO, 171, FI_CSWAP_GT,  1 },
	{ _CAMO, 181, FI_MSWAP,     1 },
};

ParameterizedTestParameters(atomic, test_cplx)
{
	return cr_make_param_array(struct test_cplx_parms, cplx_parms,
				   ARRAY_SIZE(cplx_parms));
}

ParameterizedTest(struct test_cplx_parms *p, atomic, test_cplx)
{
	struct mem_region mr;
	enum fi_datatype dt = FI_FLOAT_COMPLEX;
	uint64_t *rma;
	uint64_t *loc;
	uint64_t lini = -1;

	rma = _cxit_create_mr(&mr);

	loc = calloc(1, RMA_WIN_LEN);
	cr_assert_not_null(loc);

	if (p->opmask & _AMO) {
		_test_amo(p->index, dt, p->op, p->err, &p->o1,
			  0, 0, 0,
			  rma, &p->rini, &p->rexp);
	}

	if (p->opmask & _FAMO) {
		_test_amo(p->index, dt, p->op, p->err, &p->o1,
			  0, loc, &lini, rma, &p->rini, &p->rexp);
	}

	if (p->opmask & _CAMO) {
		_test_amo(p->index, dt, p->op, p->err, &p->o1,
			  &p->comp, loc, &lini, rma, &p->rini,
			  &p->rexp);
	}

	free(loc);
	_cxit_destroy_mr(&mr);
}

/* The DCMPLX tests only test the double complex type.
 */

struct test_dcplx_parms {
	int opmask;
	int index;
	enum fi_op op;
	int err;

	double complex comp;
	double complex o1;
	double complex rini;
	double complex rexp;

};

static struct test_dcplx_parms dcplx_parms[] = {
	{ _AMO|_FAMO, 11, FI_MIN,  1 },
	{ _AMO|_FAMO, 21, FI_MAX,  1 },
	{ _AMO|_FAMO, 31, FI_SUM,  0,
		0.0, 1.1+I*0.4, 1.2+I*1.7, (1.1+I*0.4 + 1.2+I*1.7) },
	{ _AMO|_FAMO, 32, FI_SUM,  0,
		0.0, 1.1+I*1.7, 1.2+I*0.4, (1.1+I*1.7 + 1.2+I*0.4) },
	{ _AMO|_FAMO, 41, FI_LOR,  1 },
	{ _AMO|_FAMO, 51, FI_LAND, 1 },
	{ _AMO|_FAMO, 61, FI_LXOR, 1 },
	{ _AMO|_FAMO, 71, FI_BOR,  1 },
	{ _AMO|_FAMO, 81, FI_BAND, 1 },
	{ _AMO|_FAMO, 91, FI_BXOR, 1 },
	{ _AMO|_FAMO, 101, FI_ATOMIC_WRITE, 0,
		0.0, 10.2+I*1.1, 0.3+I*2.2, 10.2+I*1.1 },
	{ _FAMO, 111, FI_ATOMIC_READ, 0,
		0.0, 1.1+I*1.1, 10.2+I*1.1, 10.2+I*1.1 },
	{ _AMO,  112, FI_ATOMIC_READ, 1 },
	{ _CAMO, 121, FI_CSWAP,     0,
		12.0+I*1.1, 12.3+I*1.1, 10.0+I*1.1, 10.0+I*1.1 },
	{ _CAMO, 122, FI_CSWAP,     0,
		10.0+I*1.1, 12.3+I*1.1, 10.0+I*1.1, 12.3+I*1.1 },
	{ _CAMO, 131, FI_CSWAP_NE,  0,
		12.0+I*1.1, 12.3+I*1.1, 10.0+I*1.1, 12.3+I*1.1 },
	{ _CAMO, 132, FI_CSWAP_NE,  0,
		10.0+I*1.1, 12.3+I*1.1, 10.0+I*1.1, 10.0+I*1.1 },
	{ _CAMO, 141, FI_CSWAP_LE,  1 },
	{ _CAMO, 151, FI_CSWAP_LT,  1 },
	{ _CAMO, 161, FI_CSWAP_GE,  1 },
	{ _CAMO, 171, FI_CSWAP_GT,  1 },
	{ _CAMO, 181, FI_MSWAP,     1 },
};

ParameterizedTestParameters(atomic, test_dcplx)
{
	return cr_make_param_array(struct test_dcplx_parms, dcplx_parms,
				   ARRAY_SIZE(dcplx_parms));
}

ParameterizedTest(struct test_dcplx_parms *p, atomic, test_dcplx)
{
	struct mem_region mr;
	enum fi_datatype dt = FI_DOUBLE_COMPLEX;
	uint64_t *rma;
	uint64_t *loc;
	uint64_t lini = -1;

	rma = _cxit_create_mr(&mr);

	loc = calloc(1, RMA_WIN_LEN);
	cr_assert_not_null(loc);

	if (p->opmask & _AMO) {
		_test_amo(p->index, dt, p->op, p->err, &p->o1,
			  0, 0, 0,
			  rma, &p->rini, &p->rexp);
	}

	if (p->opmask & _FAMO) {
		_test_amo(p->index, dt, p->op, p->err, &p->o1,
			  0, loc, &lini, rma, &p->rini, &p->rexp);
	}

	if (p->opmask & _CAMO) {
		_test_amo(p->index, dt, p->op, p->err, &p->o1,
			  &p->comp, loc, &lini, rma, &p->rini,
			  &p->rexp);
	}

	free(loc);
	_cxit_destroy_mr(&mr);
}

