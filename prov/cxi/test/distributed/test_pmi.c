/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2021 Cray Inc. All rights reserved.
 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <malloc.h>
#include <time.h>
#include <pmi_utils.h>

#define	ASSERT(c,fmt,args...) \
	do{if (!(c)) pmi_Abort(1, fmt, ## args);} while(0)

/**
 * See if requirements for this test are met, before starting test.
 *
 * @param numranks    job size
 * @param rank        rank of this process
 * @return int 0 if successful, -1 on failure
 */
static int _check_requirements(int numranks, int rank)
{
	char *kvs_env, *dbg_env;
	int kvs_num;
	int required;

	dbg_env = getenv("PMI_DEBUG_LEVEL");
	pmi_set_debug((dbg_env) ? atoi(dbg_env) : 0);

	kvs_env = getenv("PMI_MAX_KVS_ENTRIES");
	kvs_num = (kvs_env) ? atoi(kvs_env) : 0;
	required = 4*numranks + 3;

	if (numranks < 4) {
		/* only one rank makes noise */
		if (!rank)
			fprintf(stderr, "Requires >= 4 ranks\n");
		return -1;
	}
	if (kvs_num < required) {
		/* only one rank makes noise */
		if (!rank)
			fprintf(stderr,
				"export PMI_MAX_KVS_ENTRIES=%d\n",
				required);
		return -1;
	}
	return 0;
}

/**
 * Measure time difference (usec) between now and t0.
 *
 * @param t0 timestamp
 * @return uint64_t usecs between t0 and this call
 */
uint64_t measure(struct timespec *t0)
{
	struct timespec t1;

	clock_gettime(CLOCK_MONOTONIC, &t1);
	if (t1.tv_nsec < t0->tv_nsec) {
		t1.tv_nsec += 1000000000;
		t1.tv_sec -= 1;
	}
	t1.tv_sec -= t0->tv_sec;
	t1.tv_nsec -= t0->tv_nsec;

	return (1000000 * t1.tv_sec) + (t1.tv_nsec / 1000);
}

/**
 * Exercise the pmi_utils package.
 */
int main(int argc, char **argv)
{
	int numranks, rank, appnum;
	int numranks1, rank1, appnum1;
	struct timespec t0;
	uint64_t random_delay;
	uint64_t barrier_delay;
	uint64_t bcst_delay;
	uint64_t random_diff;
	uint64_t barrier_diff;
	uint64_t ratio;
	uint64_t dmin, dmax;
	uint64_t *allgather;
	uint64_t *allsend;
	uint64_t *allrecv;
	int offset;
	int bytes;
	int i;

	pmi_Init(&numranks, &rank, &appnum);
	if (_check_requirements(numranks, rank))
		return -1;

	printf("[%3d] numranks=%d appnum=%d\n", rank, numranks, appnum);

	/* Ensure second Init works properly */
	pmi_Init(&numranks1, &rank1, &appnum1);
	ASSERT(numranks == numranks1, "numranks %d != %d\n",
		numranks, numranks1);
	ASSERT(rank == rank1, "rank %d != %d\n",
		rank, rank1);
	ASSERT(appnum == appnum1, "appnum %d != %d\n",
		appnum, appnum1);

	/* reference timestamp */
	clock_gettime(CLOCK_MONOTONIC, &t0);

	/* ensure different seeds, delay random usec */
	srandom(rank + 1);
	usleep(random()%1000000);
	random_delay = measure(&t0);
	printf("[%3d] random_delay = %ld\n", rank, random_delay);

	/* wait for other processes */
	pmi_Barrier();
	barrier_delay = measure(&t0);
	printf("[%3d] barrier_delay= %ld\n", rank, barrier_delay);

	/* broadcast the random delay from rank 0 */
	/* requires one KVS entry */
	bcst_delay = random_delay;
	pmi_Bcast(0, &bcst_delay, sizeof(bcst_delay));

	/* gather all the random delays, and determine spread */
	/* requires numranks KVS entries */
	allgather = calloc(numranks, sizeof(uint64_t));
	pmi_Allgather(&random_delay, sizeof(uint64_t), allgather);
	dmin = dmax = allgather[0];
	for (i = 1; i < numranks; i++) {
		if (dmin > allgather[i])
			dmin = allgather[i];
		if (dmax < allgather[i])
			dmax = allgather[i];
	}
	random_diff = dmax - dmin;
	printf("[%3d] random_diff  = %ld\n", rank, random_diff);

	/* All ranks should have rank 0 bcast_delay */
	ASSERT(bcst_delay == allgather[0], "bcst=%ld != allg[0]=%d",
		bcst_delay, allgather[0]);

	/* gather the barrier delays, and determine spread */
	/* requires numranks KVS entries */
	pmi_Allgather(&barrier_delay, sizeof(uint64_t), allgather);
	dmin = dmax = allgather[0];
	for (i = 1; i < numranks; i++) {
		if (dmin > allgather[i])
			dmin = allgather[i];
		if (dmax < allgather[i])
			dmax = allgather[i];
	}
	barrier_diff = dmax - dmin;
	printf("[%3d] barrier_diff = %ld\n", rank, barrier_diff);

	/* barrier_diff should be small, avoid divide-by-zero */
	if (barrier_diff == 0)
		barrier_diff = 1;
	ratio = random_diff/barrier_diff;
	printf("[%3d] ratio        = %ld\n", rank, ratio);
	if (ratio < 500) {
		printf("[%3d] WARNING: small ratio\n", rank);
		printf("[%3d]   random_diff  = %ld\n", rank, random_diff);
		printf("[%3d]   barrier_diff = %ld\n", rank, barrier_diff);
		printf("[%3d]   ratio        = %ld\n", rank, ratio);
		ASSERT(0, "Barrier may not have worked\n");
	}
	free(allgather);

	/* Attempt a larger allgather block */
	/* Requires 2*numranks KVS entries */
	bytes = 0;
	while (pmi_GetKVSCount(bytes) < 3)
		bytes += sizeof(uint64_t);
	bytes -= sizeof(uint64_t);
	offset = (rank * bytes)/sizeof(uint64_t);
	allsend = calloc(1, bytes);
	allrecv = calloc(numranks, bytes);
	pmi_Allgather(allsend, bytes, allrecv);
	if (memcmp(allsend, &allrecv[offset], bytes)) {
		printf("[%3d] memcmp failed at offset %d\n", rank, offset);
		ASSERT(0, "multiblock allgather failed\n");
	}
	free(allrecv);
	free(allsend);

	/* Ensure double-finalize works properly */
	pmi_Finalize();
	pmi_Finalize();
	return 0;
}