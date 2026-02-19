/*
 * Copyright (c) 2025, Amazon.com, Inc.  All rights reserved.
 *
 * This software is available to you under the BSD license
 * below:
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
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <sys/mman.h>
#include <string.h>
#include <errno.h>
#include <shared.h>
#include <rdma/fi_ext.h>

static void *mmap_buf;
static struct fid_mr *mmap_mr;
static bool use_emulated_read;

struct test_case {
	const char *name;
	int mmap_prot;
	uint64_t mr_access;
	bool should_pass;
};

static struct test_case test_cases[] = {
	{"PROT_READ for FI_WRITE", PROT_READ, FI_WRITE, true},
	{"PROT_READ for FI_REMOTE_WRITE", PROT_READ, FI_REMOTE_WRITE, false},
	{"PROT_READ|PROT_WRITE for FI_REMOTE_WRITE", PROT_READ | PROT_WRITE, FI_REMOTE_WRITE, true},
	{"PROT_READ for FI_RECV", PROT_READ, FI_RECV, false},
	{"PROT_READ|PROT_WRITE for FI_RECV", PROT_READ | PROT_WRITE, FI_RECV, true},
	{"PROT_READ for FI_READ", PROT_READ, FI_READ, false},
	{"PROT_READ|PROT_WRITE for FI_READ", PROT_READ | PROT_WRITE, FI_READ, true},
	{"PROT_READ for FI_REMOTE_READ", PROT_READ, FI_REMOTE_READ, true},
};

#define NUM_TEST_CASES (sizeof(test_cases) / sizeof(test_cases[0]))

static int setup_mmap_buffer(int prot, uint64_t access)
{
	int ret;
	void *mmap_desc;

	/* Create writable mapping first to initialize data */
	mmap_buf = mmap(NULL, opts.transfer_size, PROT_READ | PROT_WRITE,
			MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
	if (mmap_buf == MAP_FAILED) {
		FT_PRINTERR("mmap", -errno);
		return errno;
	}

	/* Initialize buffer with test data */
	ret = ft_fill_buf(mmap_buf, opts.transfer_size);
	if (ret != FI_SUCCESS) {
		FT_PRINTERR("ft_fill_buf", ret);
		return ret;
	}

	/* Set desired protection */
	if (mprotect(mmap_buf, opts.transfer_size, prot) < 0) {
		FT_PRINTERR("mprotect", -errno);
		munmap(mmap_buf, opts.transfer_size);
		return errno;
	}

	/* Register memory region */
	ret = ft_reg_mr(fi, mmap_buf, opts.transfer_size, access,
			FT_MR_KEY, FI_HMEM_SYSTEM, 0, &mmap_mr, &mmap_desc);
	if (ret) {
		FT_PRINTERR("ft_reg_mr", ret);
		munmap(mmap_buf, opts.transfer_size);
		return ret;
	}

	return 0;
}

static void cleanup_mmap_buffer(void)
{
	int ret;
	if (mmap_mr) {
		ret = fi_close(&mmap_mr->fid);
		assert(ret == FI_SUCCESS);
		mmap_mr = NULL;
	}
	if (mmap_buf) {
		ret = munmap(mmap_buf, opts.transfer_size);
		assert(ret == 0);
		mmap_buf = NULL;
	}
}


static int run_single_test(struct test_case *test)
{
	int ret;

	printf("Testing: %s\n", test->name);

	ret = setup_mmap_buffer(test->mmap_prot, test->mr_access);

	/* FI_REMOTE_WRITE/FI_READ do not need IBV_ACCESS_LOCAL_WRITE without RDMA read */
	if (use_emulated_read &&
	    (test->mr_access == FI_REMOTE_WRITE || test->mr_access == FI_READ))
		test->should_pass = true;

	if (ret) {
		if (ret > 0) {
			printf("FAIL: %s - test failed unexpectedly: %s\n\n",
			       test->name, fi_strerror(-ret));
			return ret;
		}
		if (test->should_pass) {
			printf("FAIL: %s - mr reg failed unexpectedly: %s\n\n", 
			       test->name, fi_strerror(-ret));
			cleanup_mmap_buffer();
			return -1;
		} else {
			printf("PASS: %s - mr reg failed as expected\n\n", test->name);
			cleanup_mmap_buffer();
			return 0;
		}
	} else {
		if (test->should_pass) {
			printf("PASS: %s - mr reg succeeded as expected\n\n", test->name);
			cleanup_mmap_buffer();
			return 0;
		} else {
			printf("FAIL: %s - mr reg succeeded unexpectedly\n\n", test->name);
			cleanup_mmap_buffer();
			return -1;
		}
	}
}

static int run_test(void)
{
	int ret, i;
	int failed_tests = 0;

	for (i = 0; i < NUM_TEST_CASES; i++) {
		ret = run_single_test(&test_cases[i]);
		if (ret)
			failed_tests++;
	}

	printf("\nTest Summary: %ld/%ld tests passed\n", 
	       NUM_TEST_CASES - failed_tests, NUM_TEST_CASES);

	return failed_tests > 0 ? -1 : 0;
}

static int run(void)
{
	int ret;

	ret = ft_init();
	if (ret)
		return ret;

	ret = ft_getinfo(hints, &fi);
	if (ret) {
		FT_PRINTERR("ft_getinfo", -ret);
		return ret;
	}

	ret = ft_open_fabric_res();
	if (ret) {
		FT_PRINTERR("ft_open_fabric_res", -ret);
		return ret;
	}

	ret = ft_alloc_active_res(fi);
	if (ret) {
		FT_PRINTERR("ft_alloc_active_res", -ret);
		return ret;
	}

	ret = fi_getopt(&ep->fid, FI_OPT_ENDPOINT, FI_OPT_EFA_EMULATED_READ,
			&use_emulated_read,
			&(size_t) {sizeof use_emulated_read});
	if (ret) {
		FT_PRINTERR("fi_getopt(FI_OPT_EFA_EMULATED_READ)", ret);
		return ret;
	}

	ret = run_test();

	ft_free_res();
	return ret;
}

int main(int argc, char **argv)
{
	int op, ret;

	opts = INIT_OPTS;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt(argc, argv, "h" CS_OPTS INFO_OPTS API_OPTS)) != -1) {
		switch (op) {
		default:
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			ft_parse_api_opts(op, optarg, hints, &opts);
			break;
		case '?':
		case 'h':
			ft_usage(argv[0],
				"Test mmap buffer memory registration with different protection flags.\n");
			return EXIT_FAILURE;
		}
	}

	hints->ep_attr->type = FI_EP_RDM;
	hints->caps |= FI_MSG | FI_RMA;
	hints->domain_attr->mr_mode = opts.mr_mode;

	ret = run();

	return ft_exit_code(ret);
}
