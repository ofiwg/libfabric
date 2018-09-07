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
#include <getopt.h>

#include <rdma/fi_errno.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_tagged.h>

#include <test/user.h>


_Static_assert((TEST_API_VERSION_MAJOR == 0), "bad version");

#define DEFAULT_TRANSFER_SIZE 4

struct test_arguments {
	size_t transfer_size;
};

static int parse_arguments(
		const int argc,
		char * const *argv,
		struct test_arguments **arguments,
		size_t *buffer_size)
{
	int longopt_idx=0, op;
	static struct option longopt[] = {
		{"size", required_argument, 0, 's'},
		{"help", no_argument, 0, 'h'},
		{0}
	};

	struct test_arguments *args = calloc(sizeof(struct test_arguments), 1);
	if (args == NULL)
		return -ENOMEM;

	*args = (struct test_arguments) {
		.transfer_size = DEFAULT_TRANSFER_SIZE
	};

	if (argc > 0 && argv != NULL) {
		while ((op = getopt_long(argc, argv, "s:h", longopt, &longopt_idx)) != -1) {
			switch (op) {
			case 's':
				if (sscanf(optarg, "%zu", &args->transfer_size) != 1)
					return -EINVAL;
				break;
			case 'h':
			default:
				fprintf(stderr, "<test arguments> :=\n"
						"\t[-s | --size=<size>]\n"
						"\t[-h | --help]\n");
				return -EINVAL;
				break;
			}
		}
	}

	*buffer_size = args->transfer_size;
	*arguments = args;

	return 0;
}

void free_arguments (struct test_arguments *arguments)
{
	free (arguments);
	return;
}


static struct test_config config(const struct test_arguments *arguments)
{
	struct test_config config = {

		.minimum_caps = FI_TAGGED | FI_SEND | FI_RECV,

		.tx_use_cntr = false,
		.rx_use_cntr = false,
		.tx_use_cq = true,
		.rx_use_cq = true,

		.rx_use_mr = false,

		.tx_buffer_size = arguments->transfer_size,
		.rx_buffer_size = arguments->transfer_size,
		.tx_buffer_alignment = 0,
		.rx_buffer_alignment = 0,

		.tx_data_object_sharing = DATA_OBJECT_PER_DOMAIN,
		.rx_data_object_sharing = DATA_OBJECT_PER_TRANSFER,

		.tx_context_count = 1,
		.rx_context_count = 1,
	};

	return config;
}


static void tx_init_buffer(
		const struct test_arguments *arguments,
		uint8_t *buffer)
{
	memset(buffer, UCHAR_MAX, arguments->transfer_size);
}

static void rx_init_buffer(
		const struct test_arguments *arguments,
		uint8_t *buffer)
{
	memset (buffer, 0, arguments->transfer_size);
}

static struct fid_mr *tx_create_mr(
		const struct test_arguments *arguments,
		struct fid_domain *domain,
		const uint64_t key,
		uint8_t *buffer,
		size_t len,
		uint64_t access)
{
	int ret;

	struct fid_mr *mr;

	const uint64_t offset = 0;
	const uint64_t flags = 0;
	void *context = NULL;
	ret = fi_mr_reg (
			domain,
			buffer,
			arguments->transfer_size,
			access,
			offset,
			key,
			flags,
			&mr,
			context);
	if (ret)
		goto err_mr_reg;

	return mr;

err_mr_reg:
	fprintf(stderr, "fi_mr_reg failed (ret %d)\n", ret);

	return NULL;
}

static int destroy_mr(const struct test_arguments *arguments, struct fid_mr *mr)
{
        return fi_close(&mr->fid);
}

static size_t tx_window_usage(
		const struct test_arguments *arguments,
		const size_t transfer_id,
		const size_t transfer_count)
{
	return 1;
}

static size_t rx_window_usage(
		const struct test_arguments *arguments,
		const size_t transfer_id,
		const size_t transfer_count)
{
	return 1;
}

static int tx_transfer(
		const struct test_arguments *arguments,
		const size_t transfer_id,
		const size_t transfer_count,
		const fi_addr_t rx_address,
		struct fid_ep *endpoint,
		struct op_context *op_context,
		uint8_t *buffer,
		void *desc,
		uint64_t key,
		int rank,
		struct fid_cntr *tx_cntr,
		struct fid_cntr *rx_cntr)
{
	const uint64_t tag = transfer_id;
	return fi_tsend (
			endpoint,
			buffer,
			arguments->transfer_size,
			desc,
			rx_address,
			tag,
			&op_context->ctxinfo[0].fi_context);
}

static int rx_transfer(
		const struct test_arguments *arguments,
		const size_t transfer_id,
		const size_t transfer_count,
		const fi_addr_t tx_address,
		struct fid_ep *endpoint,
		struct op_context *op_context,
		uint8_t *buffer,
		void *desc,
		struct fid_cntr *tx_cntr,
		struct fid_cntr *rx_cntr)
{
	const uint64_t tag = transfer_id;
	const uint64_t ignore = 0;
	return fi_trecv(
			endpoint,
			buffer,
			arguments->transfer_size,
			desc,
			tx_address,
			tag,
			ignore,
			&op_context->ctxinfo[0].fi_context);
}


static int tx_cntr_completion(
		const struct test_arguments *arguments,
		const size_t completion_count,
		struct fid_cntr *cntr)
{
	return fi_cntr_wait (cntr, completion_count, -1);
}

static int rx_cntr_completion (
	const struct test_arguments *arguments,
	const size_t completion_count,
	struct fid_cntr *cntr)
{
	return fi_cntr_wait(cntr, completion_count, -1);
}

static int cq_completion (
	struct op_context **context,
	struct fid_cq *cq)
{
	ssize_t ret;
	struct fi_cq_tagged_entry cq_entry;

	ret = fi_cq_read (cq, (void *) &cq_entry, 1);
	if (ret == 1) {
		struct context_info *ctxinfo = (struct context_info *)(cq_entry.op_context);
		*context = ctxinfo->op_context;
		return 0;
	}

	return ret;
}

static int tx_cq_completion(
		const struct test_arguments *arguments,
		struct op_context **context,
		struct fid_cq *cq)
{
	return cq_completion(context, cq);
}

static int rx_cq_completion(
		const struct test_arguments *arguments,
		struct op_context **context,
		struct fid_cq *cq)
{
	return cq_completion(context, cq);
}


static int datacheck(
		const struct test_arguments *arguments,
		const uint8_t *buffer)
{
	int our_ret = FI_SUCCESS;
	size_t b;

	for (b = 0; b < arguments->transfer_size; b += 1) {
		if (((uint8_t *) buffer)[b] != UINT8_MAX) {
			fprintf(stderr, "datacheck failed at byte %zu\n", b);
			our_ret = -EIO;
			goto err_data_check;
		}
	}

err_data_check:
	return our_ret;
}

static int tx_datacheck(
		const struct test_arguments *arguments,
		const uint8_t *buffer)
{
	return datacheck(arguments, buffer);
}

static int rx_datacheck(
		const struct test_arguments *arguments,
		const uint8_t *buffer,
		size_t rx_peers)
{
	return datacheck(arguments, buffer);
}

static int tx_fini_buffer(
		const struct test_arguments *arguments,
		uint8_t *buffer)
{
	return 0;
}

static int rx_fini_buffer(
		const struct test_arguments *arguments,
		uint8_t *buffer)
{
	return 0;
}


struct test_api test_api(void)
{
	struct test_api api = {
		.parse_arguments = &parse_arguments,
		.free_arguments = &free_arguments,

		.config = &config,

		.tx_init_buffer = &tx_init_buffer,
		.rx_init_buffer = &rx_init_buffer,
		.tx_create_mr = &tx_create_mr,
		.rx_create_mr = NULL,

		.tx_window_usage = &tx_window_usage,
		.rx_window_usage = &rx_window_usage,

		.tx_transfer = &tx_transfer,
		.rx_transfer = &rx_transfer,
		.tx_cntr_completion = &tx_cntr_completion,
		.rx_cntr_completion = &rx_cntr_completion,
		.tx_cq_completion = &tx_cq_completion,
		.rx_cq_completion = &rx_cq_completion,

		.tx_datacheck = &tx_datacheck,
		.rx_datacheck = &rx_datacheck,

		.tx_fini_buffer = &tx_fini_buffer,
		.rx_fini_buffer = &rx_fini_buffer,
		.tx_destroy_mr = &destroy_mr,
		.rx_destroy_mr = NULL
	};

	return api;
}
