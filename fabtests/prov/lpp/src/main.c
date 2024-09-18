/*
 * Copyright (c) 2018-2024 GigaIO, Inc. All Rights Reserved.
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

#include <sys/uio.h>
#include <sys/param.h>
#include <sys/user.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <fcntl.h>
#include <stdbool.h>
#include <pthread.h>
#include <assert.h>

#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_atomic.h>
#include <rdma/fabric.h>

#include "error.h"
#include "ipc.h"
#include "test.h"
#include "test_util.h"

#define NUM_TESTS	(sizeof(testlist) / sizeof(testlist[0]))
#define NUM_CUDA_TESTS	(sizeof(cuda_testlist) / sizeof(cuda_testlist[0]))
#define NUM_ROCM_TESTS	(sizeof(rocm_testlist) / sizeof(rocm_testlist[0]))
#define TOTAL_TESTS	(NUM_TESTS + NUM_CUDA_TESTS + NUM_ROCM_TESTS)

static char myhostname[128];
static const char *peerhostname;
static int peer_node;
static int n_exclude_tests = 0;
static int iteration = 0;
static int iterations = 1;
static const char *filter = NULL;
static pthread_barrier_t _barrier;
bool run_cuda_tests = false;
bool run_rocm_tests = false;

enum node_id my_node;

// Note: the two large RMA tests are intentionally far apart to reduce the
// chances they run simultaneously. On configs with small IOVAs spaces, this
// can be a problem. This only matters when running with -p > 1, of course.
static const struct test testlist[] = {
	{ run_simple_rma_write,                    "simple_rma_write" },
	{ run_offset_rma_write,                    "offset_rma_write" },
	{ run_inject_rma_write,                    "inject_rma_write" },
	{ run_large_rma_write,                     "large_rma_write" },
	{ run_simple_rma_read,                     "simple_rma_read" },
	{ run_simple_msg,                          "simple_msg" },
	{ run_simple_small_msg,                    "simple_small_msg" },
	{ run_inject_msg,                          "inject_msg" },
	{ run_tagged_msg,                          "tagged_msg" },
	{ run_directed_recv_msg,                   "directed_recv_msg" },
	{ run_multi_recv_msg,                      "multi_recv_msg" },
	{ run_multi_recv_small_msg,                "multi_recv_small_msg" },
	{ run_unexpected_msg,                      "unexpected_msg" },
	{ run_unexpected_multi_recv_msg,           "unexpected_multi_recv_msg" },
	{ run_os_bypass_rma,                       "os_bypass_rma" },
	{ run_os_bypass_offset_rma,                "os_bypass_offset_rma" },
	{ run_os_bypass_outofbounds_rma,           "os_bypass_outofbounds_rma" },
	{ run_selective_completion,                "selective_completion" },
	{ run_selective_completion2,               "selective_completion2" },
	{ run_selective_completion_error,          "selective_completion_error" },
	{ run_selective_completion_osbypass_error, "selective_completion_osbypass_error" },
	{ run_rsrc_mgmt_cq_overrun,                "rsrc_mgmt_cq_overrun" },
	{ run_rma_write_auto_reg_mr,               "rma_write_auto_reg_mr" },
	{ run_msg_auto_reg_mr,                     "msg_auto_reg_mr" },
	{ run_small_msg_auto_reg_mr,               "small_msg_auto_reg_mr" },
	{ run_rma_read_auto_reg_mr,                "rma_read_auto_reg_mr" },
	{ run_zero_length,                         "zero_length" },
	{ run_large_rma_read,                      "large_rma_read" },
	{ run_loopback_msg,                        "loopback_msg" },
	{ run_loopback_small_msg,                  "loopback_small_msg" },
	{ run_loopback_write,                      "loopback_write" },
	{ run_loopback_small_write,                "loopback_small_write" },
	{ run_loopback_read,                       "loopback_read" },
	{ run_loopback_small_read,                 "loopback_small_read" },
	{ run_cq_sread,                            "cq_sread" },
	{ run_simple_atomic_write,                 "simple_atomic_write" },
	{ run_simple_atomic_write2,                "simple_atomic_write2" },
	{ run_simple_atomic_fetch_write,           "simple_atomic_fetch_write" },
	{ run_simple_atomic_fetch_write2,          "simple_atomic_fetch_write2" },
	{ run_simple_atomic_fetch_read,            "simple_atomic_fetch_read" },
	{ run_simple_atomic_fetch_read2,           "simple_atomic_fetch_read2" },
	{ run_simple_atomic_cswap,                 "simple_atomic_cswap" },
	{ run_simple_atomic_cswap2,                "simple_atomic_cswap2" },
	{ run_fi_tsenddata,                        "fi_tsenddata" },
	{ run_fi_tinjectdata,                      "fi_tinjectdata" },
};

static const struct test cuda_testlist[] = {
#ifdef USE_CUDA
	{ run_fi_hmem_cuda_tag_d2d, "fi_hmem_cuda_tag_d2d"},
	{ run_fi_hmem_cuda_sendrecv_d2d, "fi_hmem_cuda_sendrecv_d2d" },
#endif
};

static const struct test rocm_testlist[] = {
#ifdef USE_ROCM
	{ run_fi_hmem_rocm_tag_d2d, "fi_hmem_rocm_tag_d2d"},
#endif
};

static const struct test *filtered_testlist[TOTAL_TESTS];
static int n_include_tests;
static atomic_int next_test;

static void sigusr1_handler(int sig)
{
	printf("iteration %d\n", iteration);
}

static void sigint_handler(int sig)
{
	printf("interrupted at iteration %d\n", iteration);
	exit(1);
}

static void setup(const char *configured_peerhostname, int configured_node,
		  int server_port)
{
	util_global_init();

	signal(SIGUSR1, sigusr1_handler);
	signal(SIGINT, sigint_handler);

	if (gethostname(myhostname, sizeof(myhostname)) != 0) {
		errorx("gethostname");
	}

	peerhostname = configured_peerhostname;
	if (configured_node == 0) {
		my_node = NODE_A;
		peer_node = NODE_B;
	} else {
		my_node = NODE_B;
		peer_node = NODE_A;
	}

	if (my_node == NODE_A) {
		debug("I am NODE_A (%s)\n", myhostname);
		debug("NODE_B is %s\n", peerhostname);
	} else {
		debug("I am NODE_B (%s)\n", myhostname);
		debug("NODE_A is %s\n", peerhostname);
	}

	server_init(peerhostname, server_port);
}

static void test_post_checks()
{
	const char *fatal_events = "/sys/class/klpp/stat/fatal_events";
	FILE *f;

	if ((f = fopen(fatal_events, "r")) != NULL) {
		char buf[64];
		if (fgets(buf, sizeof(buf), f) == NULL) {
			errorx("failed to read %s\n", fatal_events);
		}
		if (strcmp(buf, "0\n") != 0) {
			printf("WARNING: %s != 0 (%s)\n", fatal_events, buf);
		}
		fclose(f);
	}
}

static int test_filtered(const char *test_name)
{
	const char *filt = filter;
	int invert = 0;

	if (filter == NULL) {
		return 0;
	}

	if (filter[0] == '!') {
		invert = 1;
		filt = &filter[1];
	}

	if (strstr(filt, test_name) != NULL) {
		return invert;
	} else {
		return !invert;
	}
}

static void *worker_thread(void *arg)
{
	uint64_t rank = (uint64_t)arg;

	while (1) {
		// First barrier: begin testing for this iteration.
		pthread_barrier_wait(&_barrier);

		if (iteration >= iterations) {
			break;
		}

		while (1) {
			struct rank_info *ri = get_rank_info(rank, iteration);

			// NODE_A grabs the index for the next test to run in
			// this iteration and provides it to the passive
			// NODE_B, which just waits to be told what's happening
			// next.
			if (my_node == NODE_A) {
				ri->cur_test_num = atomic_fetch_add(&next_test, 1);
				announce_peer_up(ri, iteration);
			} else {
				wait_for_peer_up(ri);
			}
			if (ri->cur_test_num >= n_include_tests) {
				// All tests done (or at least claimed; other
				// threads might still be working) for this
				// iteration.
				put_rank_info(ri);
				break;
			}
			const struct test *test =
				filtered_testlist[ri->cur_test_num];
			ri->cur_test_name = test->name;
			debugt(ri, "BEGIN TEST\n");
			int result = test->run(ri);
			INSIST_EQ(ri, result, 0, "%d");
			debugt(ri, "END TEST\n");
			put_rank_info(ri);
		}

		// Second barrier: testing done for this iteration.
		pthread_barrier_wait(&_barrier);
	}

	return (void*)1;
}

static inline void populate_filtered_testlist(const struct test* tlist,
							size_t num_tests)
{
	for (int i = 0; i < num_tests; i++) {
		if (test_filtered(tlist[i].name)) {
			if (verbose) {
				debug("skipping %s (filtered out)\n",
				      tlist[i].name);
			}
			n_exclude_tests++;
		} else {
			filtered_testlist[n_include_tests] = &tlist[i];
			n_include_tests++;
		}
	}
}

static void run_tests(int parallel)
{
	int ret = 0;
	n_include_tests = 0;
	n_exclude_tests = 0;

	populate_filtered_testlist(testlist, NUM_TESTS);

	if(run_cuda_tests)
		populate_filtered_testlist(cuda_testlist, NUM_CUDA_TESTS);
	else if (NUM_CUDA_TESTS > 0)
		debug("skipping Cuda tests\n");

	if(run_rocm_tests)
		populate_filtered_testlist(rocm_testlist, NUM_ROCM_TESTS);
	else if (NUM_ROCM_TESTS > 0)
		debug("skipping Cuda tests\n");

	if (n_include_tests == 0) {
		errorx("all tests filtered out");
	}
	debug("%d tests selected\n", n_include_tests);

	int nthreads = MIN(parallel, n_include_tests);
	debug("using %d threads\n", nthreads);

	// Barrier is used to sync main thread with worker threads each
	// iteration.
	ret = pthread_barrier_init(&_barrier, NULL, nthreads + 1);
	assert(ret == 0);
	(void) ret; /* suppress compiler warning for non-debug build */

	pthread_t *threads = calloc(nthreads, sizeof(pthread_t));
	assert(threads);
	for (uint64_t rank = 0; rank < nthreads; rank++) {
		if (pthread_create(&threads[rank], NULL, worker_thread,
				   (void *)rank) != 0) {
			errorx("pthread_create() failed");
		}
	}

	for (iteration = 0; iteration < iterations; iteration++) {
		// First barrier: begin testing for this iteration.
		pthread_barrier_wait(&_barrier);

		// Second barrier: testing done for this iteration.
		pthread_barrier_wait(&_barrier);
		atomic_store(&next_test, 0);
	}
	// Release the workers from their final barrier wait.
	pthread_barrier_wait(&_barrier);

	for (int i = 0; i < nthreads; i++) {
		int64_t retval;
		pthread_join(threads[i], (void**)&retval);
		if (retval != 1) {
			errorx("thread %d did not return normally", i);
		}
	}
	free(threads);

	printf("============================================\n");
	printf("            S U C C E S S\n");
	printf("============================================\n");
	printf("%d/%ld tests done, %d filtered, %d iterations each, parallelism of %d at a time \n",
	       n_include_tests, TOTAL_TESTS, n_exclude_tests, iterations,
	       nthreads);
	if (!run_cuda_tests)
		printf("skipped %lu cuda tests\n", NUM_CUDA_TESTS);
	if (!run_rocm_tests)
		printf("skipped %lu rocm tests\n", NUM_ROCM_TESTS);
}

void usage()
{
	printf("lpp_regression -P PEER_HOSTNAME -N NODE_NUM [-f FILTER] [-p THREAD_COUNT] [-i ITERATION] [-q] [-S SERVER_PORT]\n");
	printf("\t-f: Filter which tests are run. Tests containing the substring FILTER are filtered. Precede with a ! to exclude rather than include.\n");
	printf("\t-N: Specify which node this is. Must be either 0 or 1. The peer node must use the other value\n");
	printf("\t-p: Run tests in parallel, THREAD_COUNT threads at once\n");
	printf("\t-P: Set the peer hostname\n");
	printf("\t-i: Run ITERATION iterations\n");
	printf("\t-q: Quiet debug messages\n");
	printf("\t-S: Set the server port (necessary when running multiple instances)\n");
	printf("\t-v: Print verbose test info (for debugging)\n");
#ifdef USE_CUDA
	printf("\t-c: Run FI_HMEM tests with CUDA (excluded by default) \n");
#endif
#ifdef USE_ROCM
	printf("\t-r: Run FI_HMEM tests with ROCM (excluded by default) \n");
#endif
	exit(1);
}

int main(int argc, char *argv[])
{
	char c;
	int parallel = 1;
	char *endptr;
	char configured_peerhostname[128] = { 0 };
	int configured_node = -1;
	int server_port = -1;

	while ((c = getopt(argc, argv, "f:hi:N:p:P:qS:vcr")) != -1) {
		switch(c) {
		case 'c':
			run_cuda_tests = true;
			break;
		case 'r':
			run_rocm_tests = true;
			break;
		case 'f':
			filter = strdup(optarg);
			break;
		case 'h':
			usage();
			break;
		case 'i':
			iterations = strtoul(optarg, &endptr, 0);
			if (*endptr != '\0') {
				usage();
			}
			if (iterations < 1) {
				printf("iterations must be > 0\n\n");
				usage();
			}
			break;
		case 'N':
			configured_node = strtol(optarg, &endptr, 0);
			if (*endptr != '\0') {
				usage();
			}
			break;
		case 'p':
			parallel = strtoul(optarg, &endptr, 0);
			if (*endptr !='\0') {
				usage();
			}
			break;
		case 'P':
			strncpy(configured_peerhostname, optarg, sizeof(configured_peerhostname) - 1);
			break;
		case 'q':
			debug_quiet = true;
			break;
		case 'S':
			server_port = strtol(optarg, &endptr, 0);
			if (*endptr != '\0') {
				usage();
			}
			break;
		case 'v':
			verbose = true;
			break;
		default:
			usage();
			break;
		}
	}

	if (configured_peerhostname[0] == '\0' || configured_node < 0) {
		errorx("Both -N and -P must be specified");
	}

	argc -= optind;
	argv += optind;

	setup(configured_peerhostname, configured_node, server_port);
	run_tests(parallel);
	hmem_cleanup();
	test_post_checks();

	return 0;
}
