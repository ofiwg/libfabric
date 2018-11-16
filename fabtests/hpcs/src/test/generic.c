/*
 * Copyright (c) 2017-2018 Intel Corporation. All rights reserved.
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

#include <limits.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <unistd.h>

#include <rdma/fi_errno.h>
#include <rdma/fi_eq.h>

#include <test/user.h>

struct test_arguments {
};

void test_generic_free_arguments(struct test_arguments *arguments)
{
	free (arguments);
}

void test_generic_tx_init_buffer(const struct test_arguments *arguments,
		uint8_t *buffer, size_t len)
{
	memset(buffer, UINT8_MAX, len);
}

void test_generic_rx_init_buffer(const struct test_arguments *arguments,
		uint8_t *buffer, size_t len)
{
	memset(buffer, 0, len);
}

static struct fid_mr *test_generic_create_mr(const struct test_arguments *arguments,
		struct fid_domain *domain, const uint64_t key,
		uint8_t *buffer, size_t len, uint64_t access, uint64_t flags)
{
	int ret;
	struct fid_mr *mr;

	ret = fi_mr_reg(domain, buffer, len, access, 0, key, flags, &mr, NULL);
	if (ret)
		goto err_mr_reg;

	return mr;

err_mr_reg:
	fprintf(stderr, "fi_mr_reg failed\n");
	return NULL;
}

struct fid_mr *test_generic_tx_create_mr(const struct test_arguments *arguments,
		struct fid_domain *domain, const uint64_t key,
		uint8_t *buffer, size_t len, uint64_t access, uint64_t flags)
{
	return test_generic_create_mr(arguments, domain, key,
			buffer, len, access, flags);
}

struct fid_mr *test_generic_rx_create_mr(const struct test_arguments *arguments,
		struct fid_domain *domain, const uint64_t key,
		uint8_t *buffer, size_t len, uint64_t access, uint64_t flags)
{
	return test_generic_create_mr(arguments, domain, key,
			buffer, len, access, flags);
}

int test_generic_tx_destroy_mr(const struct test_arguments *arguments, struct fid_mr *mr)
{
	return fi_close(&mr->fid);
}

int test_generic_rx_destroy_mr(const struct test_arguments *arguments, struct fid_mr *mr)
{
	return fi_close(&mr->fid);
}

size_t test_generic_tx_window_usage(const struct test_arguments *arguments,
		const size_t transfer_id, const size_t transfer_count)
{
	return 1;
}

size_t test_generic_rx_window_usage(const struct test_arguments *arguments,
		const size_t transfer_id, const size_t transfer_count)
{
	return 1;
}

int test_generic_tx_transfer(const struct test_arguments *arguments,
		const size_t transfer_id, const size_t transfer_count,
		const fi_addr_t rx_address, struct fid_ep *endpoint,
		struct op_context *op_context, uint8_t *buffer,
		void *desc, uint64_t key, int rank, uint64_t flags)
{
	return 0;
}

int test_generic_rx_transfer(const struct test_arguments *arguments,
		const size_t transfer_id, const size_t transfer_count,
		const fi_addr_t tx_address, struct fid_ep *endpoint,
		struct op_context *op_context, uint8_t *buffer,
		void *desc, uint64_t flags)
{
	return 0;
}

static int generic_cntr_completion(const struct test_arguments *arguments,
		const size_t completion_count, struct fid_cntr *cntr)
{
	return fi_cntr_wait(cntr, completion_count, -1);
}


int test_generic_tx_cntr_completion(const struct test_arguments *arguments,
		const size_t completion_count, struct fid_cntr *cntr)
{
	return generic_cntr_completion(arguments, completion_count, cntr);
}

int test_generic_rx_cntr_completion(const struct test_arguments *arguments,
		const size_t completion_count, struct fid_cntr *cntr)
{
	return generic_cntr_completion(arguments, completion_count, cntr);
}

/*
 * This function and core assumes that there is only one cq completion
 * per tx or rx transfer.  Tests that need more than one completion per
 * transfer should implement their own completion logic (possibly calling
 * this in a loop) or use counters.
 */
static int test_generic_cq_completion(const struct test_arguments *args,
		struct op_context **op_contextp, struct fid_cq *cq)
{
	ssize_t ret;
	struct fi_cq_tagged_entry cq_entry;
	struct context_info *ctxinfo;
	struct op_context *op_context;

	ret = fi_cq_read(cq, (void *) &cq_entry, 1);

	if (ret == 1) {
		ctxinfo = (struct context_info *)(cq_entry.op_context);
		op_context = ctxinfo->op_context;
		op_context->test_state++;
		*op_contextp = op_context;
		return 0;
	}

	if (ret == -FI_EAVAIL) {
		struct fi_cq_err_entry err;

		ret = fi_cq_readerr(cq, &err, 0);
		if (ret < 0) {
			fprintf(stderr, "unable to read error completion\n");
		} else {
			fprintf(stderr,
					"cq_read error ctx %p len %ld tag %ld err %d (%s) prov_errno %d err_data %p err_data_size %ld\n",
					err.op_context, err.len, err.tag,
					err.err,
					fi_cq_strerror(cq, err.prov_errno, err.err_data, NULL, 0),
					err.prov_errno, err.err_data,
					err.err_data_size);
		}

		return -FI_EFAULT;
	}

	return ret;
}

int test_generic_tx_cq_completion(const struct test_arguments *args,
		struct op_context **context, struct fid_cq *cq)
{
	return test_generic_cq_completion(args, context, cq);
}

int test_generic_rx_cq_completion(const struct test_arguments *args,
		struct op_context **context, struct fid_cq *cq)
{
	return test_generic_cq_completion(args, context, cq);
}

static int test_generic_datacheck(const struct test_arguments *arguments,
		const uint8_t *buffer, size_t len)
{
	size_t i;

	for (i = 0; i < len; i++) {
		if (((uint8_t *) buffer)[i] != UINT8_MAX) {
			fprintf(stderr, "datacheck failed at byte %zu (expected %u, found %u)\n",
					i, UINT8_MAX, buffer[i]);
			return -FI_EIO;
		}
	}

	return 0;
}

int test_generic_tx_datacheck(const struct test_arguments *arguments,
		const uint8_t *buffer, size_t len)
{
	return test_generic_datacheck(arguments, buffer, len);
}

int test_generic_rx_datacheck(const struct test_arguments *args,
		const uint8_t *buffer, size_t len, size_t rx_peers)
{
	return test_generic_datacheck(args, buffer, len);
}

int test_generic_tx_fini_buffer(const struct test_arguments *arguments,
		uint8_t *buffer)
{
	return 0;
}

int test_generic_rx_fini_buffer(const struct test_arguments *arguments,
		uint8_t *buffer)
{
	return 0;
}

