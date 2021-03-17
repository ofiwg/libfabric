/*
 * Copyright (c) 2017-2019 Intel Corporation. All rights reserved.
 * Copyright (c) 2020 Cray Inc. All rights reserved.
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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHWARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. const NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER const AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS const THE
 * SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <complex.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#include <ltu_utils_pm.h>
#include <fenv.h>

#include <ofi.h>

#include "cxip.h"
#include "cxip_test_common.h"
#include "cxip_test_pmi.h"

/* Simulated user context, specifically to return error codes */
struct user_context {
	struct dlist_entry entry;
	int node;		// reduction simulated node (MC object)
	int seqno;		// reduction sequence number
	int red_id;		// reduction ID
	int errcode;		// reduction error code
	int hw_rc;		// reduction hardware failure code
	uint64_t expval;	// expected reduction value
};

/* Simulated user data type */
struct int_data {
	uint64_t ival[4];
};

/* Poll rx and tx CQs until user context seen.
 *
 * If context == NULL, this polls to advance the state and returns.
 */
static void _reduce_wait(struct fid_cq *rx_cq, struct fid_cq *tx_cq,
			 struct user_context *context)
{
	static struct dlist_entry done_list, *done;
	static int initialized = 0;

	struct fi_cq_data_entry entry;
	struct fi_cq_err_entry err_entry;
	struct user_context *ctx;
	int ret;

	/* initialize the static locals */
	if (! initialized) {
		dlist_init(&done_list);
		initialized = 1;
	}

	/* search for prior detection of context */
	dlist_foreach(&done_list, done) {
		if ((void *)context == (void *)done) {
			dlist_remove(done);
			return;
		}
	}

	do {
		/* Wait for a tx CQ completion event */
		do {
			sched_yield();
			/* read rx CQ and progress state until nothing to do */
			ret = fi_cq_read(rx_cq, &entry, 1);
			if (ret == -FI_EAGAIN && context)
				continue;
			/* read tx CQ to see a completion event */
			ret = fi_cq_read(tx_cq, &entry, 1);
			if (ret != -FI_EAGAIN || !context)
				break;
		} while (true);
		ctx = NULL;
		if (ret == -FI_EAVAIL) {
			/* tx CQ posted an error, copy to user context */
			ret = fi_cq_readerr(tx_cq, &err_entry, 1);
			cr_assert(ret == 1, "fi_cq_readerr failed: %d\n", ret);
			ctx = err_entry.op_context;
			ctx->errcode = err_entry.err;
			ctx->hw_rc = err_entry.prov_errno;
			cr_assert(err_entry.err != 0,
				  "Failure with good return\n");
		} else if (ret == 1) {
			/* tx CQ posted a normal completion */
			ctx = entry.op_context;
			ctx->errcode = 0;
			ctx->hw_rc = 0;
		}
		if (ctx == context)
			return;
		if (ctx)
			dlist_insert_tail(&ctx->entry, &done_list);
	} while (context);
}

TestSuite(coll_dist, .init = cxit_setup_distributed,
	  .fini = cxit_teardown_distributed,
	  .disabled = false, .timeout = CXIT_DEFAULT_TIMEOUT);

Test(coll_dist, basic1)
{
	printf("Basic Test 1\n");
	printf("  ranks      = %d\n", cxit_ranks);
	printf("  rank       = %d\n", cxit_rank);
}

TestSuite(coll_mcast, .init = cxit_setup_multicast,
	  .fini = cxit_teardown_multicast,
	  .disabled = false, .timeout = 300);

Test(coll_mcast, barrier)
{
	struct timespec ts0, ts1;
	struct user_context context;
	ssize_t ret;
	double delay;

	if (cxit_mcast_id <= 0)
		cr_skip_test("No multicast address, skipping test\n");

	clock_gettime(CLOCK_REALTIME, &ts0);
	if (cxit_rank == 0) {
		sleep(1);
	}
	ret = fi_barrier(cxit_ep, (fi_addr_t)cxit_mc, &context);
	cr_assert_eq(ret, 0, "fi_barrier() initiated=%ld\n", ret);

	_reduce_wait(cxit_rx_cq, cxit_tx_cq, &context);
	clock_gettime(CLOCK_REALTIME, &ts1);
	printf("=============================\n");
	printf("completion rank %d of %d\n", cxit_rank, cxit_ranks);
	delay = _print_delay(&ts0, &ts1, __func__, "barrier 1");
	cr_assert(delay + 0.01 > 1.0);

	clock_gettime(CLOCK_REALTIME, &ts0);
	if (cxit_rank == 1) {
		sleep(1);
	}
	ret = fi_barrier(cxit_ep, (fi_addr_t)cxit_mc, &context);
	cr_assert_eq(ret, 0, "fi_barrier() initiated=%ld\n", ret);

	_reduce_wait(cxit_rx_cq, cxit_tx_cq, &context);
	clock_gettime(CLOCK_REALTIME, &ts1);
	printf("=============================\n");
	printf("completion rank %d of %d\n", cxit_rank, cxit_ranks);
	delay = _print_delay(&ts0, &ts1, __func__, "barrier 2");
	cr_assert(delay + 0.01 > 1.0);
	printf("barrier success\n");
}

static void _bcast(int root)
{
	struct user_context context;
	ssize_t ret;
	uint64_t buf[4];
	uint64_t idata[4] = {
		0x8100000000000001LL,
		0x8200000000000002LL,
		0x8300000000000003LL,
		0x8400000000000004LL
	};
	int i;

	if (cxit_mcast_id <= 0)
		cr_skip_test("No multicast address, skipping test\n");

	if (cxit_rank == root) {
		memcpy(buf, idata, sizeof(idata));
	} else {
		memset(buf, 1, sizeof(buf));
	}
	ret = fi_broadcast(cxit_ep, (void *)buf, 4, NULL,
			   (fi_addr_t)cxit_mc, root,
			   FI_UINT64, 0, &context);
	cr_assert_eq(ret, 0, "fi_bcast() initiated=%ld\n", ret);

	_reduce_wait(cxit_rx_cq, cxit_tx_cq, &context);

	ret = memcmp(buf, idata, sizeof(idata));
	if (ret != 0) {
		printf("%-18s | %s\n", "expected", "actual");
		for (i = 0; i < 4; i++)
			printf("0x%016lx | 0x%016lx\n", idata[i], buf[i]);
		cr_assert_fail();
	}
	printf("bcast success\n");
}

Test(coll_mcast, bcast)
{
	_bcast(0);
	_bcast(1);
}

#define	RANDOM_UINT64	0
#define	RANDOM_DOUBLE	1
#define	RANDOM_SUBNORM	2
#define	RANDOM_SZERO	3
#define	RANDOM_INF	4
#define	RANDOM_SNAN	5
#define	RANDOM_QNAN	6
#define	RANDOM_LAST	7

/* 64-bit structure that decomposes numbers */
union randombits {
	uint64_t ival;
	double fval;
	struct {
		uint64_t mantissa:52;
		uint64_t exponent:11;
		uint64_t sign:1;
	};
};

/* Set this to a previous seed to reproduce results */
static unsigned int random_seed = 0;

__attribute__((unused))
static inline void _random_init(void)
{
	static int init;
	if (!init) {
		init = 1;
		if (!random_seed)
			random_seed = time(NULL) + cxit_rank + 1;
		srandom(random_seed);
		printf("RANDOM_SEED = %d\n", random_seed);
	}
}

/**
 * Generate a random 64-bit quantity
 *
 * See https://en.wikipedia.org/wiki/Double-precision_floating-point_format
 * See https://en.wikipedia.org/wiki/NaN
 *
 * @param data : pointer to 64-bit data storage
 * @param special : specify the type of random value
 * @param sign : specify -1 for negative, +1 for positive, 0 for random
 */
__attribute__((unused))
static void _random64bit(void *data, int special, int sign)
{
	union randombits *val = data;

	_random_init();

	/* random() only produces 31 bits, we need 64 */
	val->ival = random() | (random() << 31) | (random() & 0x3);

	/* allow sign to be specified (for +/- zero and inf) */
	if (sign)
		val->sign = (sign > 0) ? 0 : 1;

	switch (special) {
	case RANDOM_UINT64:
		/* return value as-is */
		break;
	case RANDOM_DOUBLE:
		/* exponent 0x000 and 0x7ff reserved */
		if (val->exponent == 0x000)
			val->exponent = 0x001;
		else if (val->exponent == 0x7ff)
			val->exponent = 0x7fe;
		break;
	case RANDOM_SUBNORM:
		/* non-zero mantissa is a subnormal number */
		val->mantissa |= 1;
		val->exponent = 0x000;
		break;
	case RANDOM_SZERO:
		/* zero mantissa is a signed zero */
		val->mantissa = 0;
		val->exponent = 0x000;
		break;
	case RANDOM_SNAN:
		/* non-zero mantissa is NaN, bit 51 clear is signalling */
		val->mantissa = (1LL << 50);
		val->exponent = 0x7ff;
		break;
	case RANDOM_QNAN:
		/* non-zero mantissa is NaN, bit 51 set is quiet */
		val->mantissa = (1LL << 51);
		val->exponent = 0x7ff;
		break;
	case RANDOM_INF:
		/* zero mantissa is a signed infinity */
		val->mantissa = 0;
		val->exponent = 0x7ff;
		break;
	}
}

__attribute__((unused))
static const char *_decode_value(void *a)
{
	union randombits *av = (union randombits *)a;

	if (av->exponent == 0x7ff && (av->mantissa & (1LL << 51))) {
		return (av->sign) ? "-NaN(Q)" : "NaN(Q)";
	} else if (av->exponent == 0x7ff && av->mantissa) {
		return (av->sign) ? "-NaN(S)" : "NaN(S)";
	} else if (av->exponent == 0x7ff) {
		return (av->sign) ? "-inf" : "inf";
	} else if (av->exponent == 0 && av->mantissa) {
		return (av->sign) ? "-sub" : "sub";
	} else if (av->exponent == 0 && av->sign) {
		return (av->sign) ? "-0" : "+0";
	} else {
		return "double";
	}
}

__attribute__((unused))
static const char *_display_special(int special)
{
	switch (special) {
	case RANDOM_UINT64:
		return "UINT64";
	case RANDOM_DOUBLE:
		return "DOUBLE";
	case RANDOM_SUBNORM:
		return "SUBNORM";
	case RANDOM_SZERO:
		return "SZERO";
	case RANDOM_INF:
		return "INF";
	case RANDOM_SNAN:
		return "SNaN";
	case RANDOM_QNAN:
		return "QNaN";
	}
	return "invalid";
}

static void test_op(int dtyp, int op, union cxip_coll_data *data)
{
	struct user_context context;
	union cxip_coll_data rslt[2];
	int i, trial, r, ret;

	memset(&rslt[0], 0, sizeof(rslt[0]));
	memset(&rslt[1], 0, sizeof(rslt[1]));

	/* This Barrier operation disables Rosetta reduction for the
	 * subsequent operation ONLY. The operation after that will use
	 * Rosetta normally.
	 */
	cxip_coll_arm_disable_once();
	fi_barrier(cxit_ep, (fi_addr_t)cxit_mc, &context);
	_reduce_wait(cxit_rx_cq, cxit_tx_cq, &context);

	/* Two trials of the same reduction. The first does not do Rosetta
	 * reduction, so all reduction is done in software on the hwroot node.
	 * The second does Rosetta reduction. Our definition of "correct"
	 * behavior is that the results match. If the don't match, further
	 * analysis is required to decide which is wrong, if not both.
	 *
	 * The ltu_pm_Barrier() is intended to avoid a scrum at the Rosetta RE
	 * for non-associative operations. It ensures the order in which sends
	 * are executed, and -- if there is only one Rosetta -- the order in
	 * which the Rosetta operations are performed. This ordering will not
	 * work if the multicast tree spans more than one Rosetta.
	 */
	for (trial = 0; trial < 2; trial++) {
		for (r = 0; r < cxit_ranks; r++) {
			ltu_pm_Barrier();
			if (r == cxit_rank) {
				ret = fi_allreduce(cxit_ep,
						   (void *)data, 4, NULL,
						   (void *)&rslt[trial], NULL,
						   (fi_addr_t)cxit_mc,
						   dtyp, op, 0, &context);
				cr_assert_eq(ret, FI_SUCCESS, "fi_allreduce() = %d", ret);
			}
		}
		_reduce_wait(cxit_rx_cq, cxit_tx_cq, &context);
	}
	for (i = 0; i < 4; i++) {
		if (rslt[0].ival[i] != rslt[1].ival[i]) {
			printf("Miscompare on #%d\n", i);
			printf("  a = %016lx\n", rslt[0].ival[i]);
			printf("  b = %016lx\n", rslt[1].ival[i]);
			cr_assert_fail();
		}
	}
}

Test(coll_mcast, reductions)
{
	unsigned int round[4] = {
		FE_TONEAREST,
		FE_UPWARD,
		FE_DOWNWARD,
		FE_TOWARDZERO
	};
	int rnd0, rnd;
	union cxip_coll_data data;
	int i, j;

	if (cxit_mcast_id <= 0)
		cr_skip_test("No multicast address, skipping test\n");

	/* Perform randomized tests on UINT64 */
	printf("Test: INT64 ops\n");
	for (i = 0; i < 100; i++) {
		continue;
		for (j = 0; j < 4; j++)
			_random64bit(&data.ival[j], RANDOM_UINT64, 0);
		test_op(FI_UINT64, FI_SUM, &data);
		test_op(FI_UINT64, FI_MIN, &data);
		test_op(FI_UINT64, FI_MAX, &data);
		test_op(FI_UINT64, FI_BOR, &data);
		test_op(FI_UINT64, FI_BAND, &data);
		test_op(FI_UINT64, FI_BXOR, &data);
		test_op(FI_UINT64, CXI_FI_MINMAXLOC, &data);
	}

	/* Perform randomized tests on normal DOUBLES  */
	rnd0 = fegetround();
	for (rnd = 0; rnd < 4; rnd++) {
		printf("Test: DOUBLE ops, rounding=%d\n", rnd);
		fesetround(round[rnd]);
		cxip_coll_populate_opcodes();
		for (i = 0; i < 100; i++) {
			for (j = 0; j < 4; j++)
				_random64bit(&data.ival[j], RANDOM_DOUBLE, 0);
			test_op(FI_DOUBLE, FI_SUM, &data);
			test_op(FI_DOUBLE, FI_MIN, &data);
			test_op(FI_DOUBLE, FI_MAX, &data);
			test_op(FI_DOUBLE, CXI_FI_MINMAXLOC, &data);
			test_op(FI_DOUBLE, CXI_FI_MINNUM, &data);
			test_op(FI_DOUBLE, CXI_FI_MAXNUM, &data);
			test_op(FI_DOUBLE, CXI_FI_MINMAXNUMLOC, &data);
		}
	}
	fesetround(rnd0);

	/* Perform permutations on special DOUBLES */
	printf("Testing special doubles\n");
	for (i = RANDOM_DOUBLE; i < RANDOM_LAST; i++) {
		for (j = RANDOM_DOUBLE; j < RANDOM_LAST; j++) {
			if (cxit_rank == 1) {
				/* one leaf rank based on outer-loop i */
				_random64bit(&data.fval[0], i, -1);
				_random64bit(&data.fval[1], i, -1);
				_random64bit(&data.fval[2], i, 1);
				_random64bit(&data.fval[3], i, 1);
			} else {
				/* other ranks based on inner-loop j */
				_random64bit(&data.fval[0], j, -1);
				_random64bit(&data.fval[1], j, 1);
				_random64bit(&data.fval[2], j, -1);
				_random64bit(&data.fval[3], j, 1);
			}
			printf("Test: SUM %s vs %s\n",
			       _display_special(i),
			       _display_special(j));
			test_op(FI_DOUBLE, FI_SUM, &data);
			printf("Test: MIN %s vs %s\n",
			       _display_special(i),
			       _display_special(j));
			test_op(FI_DOUBLE, FI_MIN, &data);
			printf("Test: MAX %s vs %s\n",
			       _display_special(i),
			       _display_special(j));
			test_op(FI_DOUBLE, FI_MAX, &data);
			printf("Test: MINMAXLOC %s vs %s\n",
			       _display_special(i),
			       _display_special(j));
			test_op(FI_DOUBLE, CXI_FI_MINMAXLOC, &data);
			printf("Test: MINNUM %s vs %s\n",
			       _display_special(i),
			       _display_special(j));
			test_op(FI_DOUBLE, CXI_FI_MINNUM, &data);
			printf("Test: MAXNUM %s vs %s\n",
			       _display_special(i),
			       _display_special(j));
			test_op(FI_DOUBLE, CXI_FI_MAXNUM, &data);
			printf("Test: MINMAXNUMLOC %s vs %s\n",
			       _display_special(i),
			       _display_special(j));
			test_op(FI_DOUBLE, CXI_FI_MINMAXNUMLOC, &data);
		}
	}
	printf("reduce success\n");
}

TestSuite(coll_demo, .disabled = true, .timeout = 300);

Test(coll_demo, demo)
{
	const char *redbar = "\033]6;1;bg;green;brightness;0\a"
			     "\033]6;1;bg;red;brightness;255\a";
	const char *grnbar = "\033]6;1;bg;green;brightness;255\a"
			     "\033]6;1;bg;red;brightness;0\a";
	const char *dflbar = "\033]6;1;bg;*;default\a";
	const char *fifo_path = "/tmp/cxit_fifo";
	struct user_context context;
	struct timespec delay;
	int i, j, max, ret;
	uint64_t data, rslt;
	FILE *fifo;

	printf("Starting demo\n");
	printf("Creating fifo output file\n");
	mkfifo(fifo_path, 0666);
	printf("Opening fifo output file\n");
	fifo = fopen(fifo_path, "w");
	printf("Open file pointer = %p\n", fifo);
	setlinebuf(fifo);

	fprintf(fifo, "\n\n\n\n======================\n");
	fprintf(fifo, "Setting up multicast tree...\n");
	cxit_setup_multicast();
	fprintf(fifo, "%d ranks, myrank = %d\n", cxit_ranks, cxit_rank);
	fprintf(fifo, "Ready\n");

	srandom(cxit_rank + 3);
	delay.tv_sec = 0;
	delay.tv_nsec = 1000000;
	for (i = 0; i < 10; i++) {
		fprintf(fifo, "%s", grnbar); fflush(fifo);
		max = random() % 5000;
		for (j = 0; j < max; j++) {
			fprintf(fifo, "%9d\r", j);
			fflush(fifo);
			nanosleep(&delay, NULL);
		}
		data = max;
		ret = fi_allreduce(cxit_ep, (void *)&data, 1, NULL,
				(void *)&rslt, NULL,
				(fi_addr_t)cxit_mc,
				FI_UINT64, FI_MAX, 0, &context);
		cr_assert_eq(ret, 0, "fi_allreduce() initiated=%d\n", ret);
		fprintf(fifo, "%s", redbar); fflush(fifo);
		_reduce_wait(cxit_rx_cq, cxit_tx_cq, &context);
		fprintf(fifo, "\nresult = %ld\n", rslt);
	}
	fprintf(fifo, "%s", dflbar); fflush(fifo);
	fprintf(fifo, "done\n");
	fprintf(fifo, "===================\n");

	fprintf(fifo, "Destroying multicast tree...\n");
	cxit_teardown_multicast();
	fprintf(fifo, "Finished\n\n\n\n");
	fflush(fifo);
	fclose(fifo);
}

