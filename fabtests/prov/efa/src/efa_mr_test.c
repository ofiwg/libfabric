/*
 * Copyright (c) 2026, Amazon.com, Inc.  All rights reserved.
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
 * Test HMEM dmabuf memory registration with aligned and unaligned VAs.
 */

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <shared.h>
#include <hmem.h>

#define DEFAULT_BUF_SIZE 4096
#define MIN_BUF_SIZE 1024
#define MAX_BUF_SIZE (1024 * 1024)
#define NUM_UNALIGNED_TESTS 8

static size_t buf_size;

static int test_aligned_mr(void)
{
	struct fid_mr *mr;
	void *desc;
	int ret;

	ret = ft_reg_mr(fi, buf, buf_size, FI_SEND | FI_RECV,
			FT_MR_KEY, opts.iface, opts.device, &mr, &desc);
	if (ret) {
		printf("FAIL: aligned VA - ft_reg_mr failed: %s\n",
		       fi_strerror(-ret));
		return ret;
	}

	printf("PASS: aligned VA\n");

	ret = fi_close(&mr->fid);
	if (ret) {
		printf("FAIL: aligned VA - fi_close failed: %s\n",
		       fi_strerror(-ret));
		return ret;
	}

	return 0;
}

static int test_unaligned_mr(uint64_t offset)
{
	struct fid_mr *mr;
	void *desc;
	int ret;
	int version;
	bool should_fail = false;

	/* Only nrt_get_dmabuf_fd v1 should fail on unaligned */
	if (opts.iface == FI_HMEM_NEURON) {
		version = ft_nrt_get_op_version(NRT_GET_DMABUF_FD);
		if (version < 2)
			should_fail = true;
	}

	ret = ft_reg_mr(fi, (char *)buf + offset, buf_size - offset,
			FI_SEND | FI_RECV, FT_MR_KEY, opts.iface, opts.device,
			&mr, &desc);

	if (should_fail && ret) {
		printf("PASS: unaligned VA offset=%lu - failed as expected (nrt_get_dmabuf_fd() v1)\n",
		       offset);
		return 0;
	} else if (should_fail && !ret) {
		printf("FAIL: unaligned VA offset=%lu - passed unexpectedly (nrt_get_dmabuf_fd() v1)\n",
		       offset);
		fi_close(&mr->fid);
		return -FI_EINVAL;
	} else if (!should_fail && ret) {
		printf("FAIL: unaligned VA offset=%lu - ft_reg_mr failed: %s\n",
		       offset, fi_strerror(-ret));
		return ret;
	} else {
		/* !should_fail && !ret */
		printf("PASS: unaligned VA offset=%lu\n", offset);
	}

	ret = fi_close(&mr->fid);
	if (ret) {
		printf("FAIL: unaligned VA offset=%lu - fi_close failed: %s\n",
		       offset, fi_strerror(-ret));
		return ret;
	}

	return 0;
}

static int run_tests(void)
{
	int ret, failed = 0, i;
	uint64_t test_offsets[NUM_UNALIGNED_TESTS];
	uint64_t step;
	const char *iface_name;

	switch (opts.iface) {
	case FI_HMEM_NEURON:
		iface_name = "Neuron";
		break;
	case FI_HMEM_CUDA:
		iface_name = "CUDA";
		break;
	default:
		iface_name = "Unknown";
	}

	printf("Testing %s dmabuf MR with libfabric\n", iface_name);
	printf("Memory size: %zu bytes\n", buf_size);
	if (opts.iface == FI_HMEM_NEURON) {
		printf("NRT_GET_DMABUF_FD version: %d\n",
		       ft_nrt_get_op_version(NRT_GET_DMABUF_FD));
	}

	ret = test_aligned_mr();
	if (ret)
		failed++;

	/* Generate test offsets from 1 to buf_size-1 */
	step = (buf_size - 2) / (NUM_UNALIGNED_TESTS - 1);
	for (i = 0; i < NUM_UNALIGNED_TESTS; i++) {
		test_offsets[i] = 1 + (i * step);
	}

	for (i = 0; i < NUM_UNALIGNED_TESTS; i++) {
		ret = test_unaligned_mr(test_offsets[i]);
		if (ret)
			failed++;
	}

	printf("\nTest Summary: %d/%d tests passed\n",
	       NUM_UNALIGNED_TESTS + 1 - failed, NUM_UNALIGNED_TESTS + 1);

	return failed > 0 ? -1 : 0;
}

static int setup(void)
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

	ret = ft_hmem_alloc(opts.iface, opts.device, (void **)&buf, buf_size);
	if (ret) {
		FT_PRINTERR("ft_hmem_alloc", ret);
		return ret;
	}

	return 0;
}

static void usage(char *name)
{
	ft_usage(name, "Test HMEM dmabuf MR with aligned and unaligned VAs");
	ft_hmem_usage();
	ft_longopts_usage();
}

int main(int argc, char **argv)
{
	int op, ret;

	opts = INIT_OPTS;
	opts.transfer_size = DEFAULT_BUF_SIZE;
	opts.options |= FT_OPT_REG_DMABUF_MR;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

#define EFA_MR_OPTS "hS:f:D:"
	while ((op = getopt(argc, argv, EFA_MR_OPTS)) != -1) {
		switch (op) {
		default:
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			break;
		case '?':
		case 'h':
			usage(argv[0]);
			return EXIT_FAILURE;
		}
	}

	buf_size = opts.transfer_size;

	if (buf_size < MIN_BUF_SIZE || buf_size > MAX_BUF_SIZE) {
		fprintf(stderr, "Buffer size must be between %d and %d bytes\n",
			MIN_BUF_SIZE, MAX_BUF_SIZE);
		return EXIT_FAILURE;
	}

	ret = ft_init();
	if (ret) {
		FT_PRINTERR("ft_init", ret);
		goto out;
	}

	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_MSG;
	if (opts.options & FT_OPT_ENABLE_HMEM)
		hints->caps |= FI_HMEM;
	hints->domain_attr->mr_mode = opts.mr_mode;

	ret = setup();
	if (ret)
		goto out;

	ret = run_tests();

out:
	ft_free_res();
	fi_freeinfo(hints);
	return ft_exit_code(ret);
}
