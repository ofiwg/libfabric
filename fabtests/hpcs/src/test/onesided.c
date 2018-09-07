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
#include <rdma/fi_rma.h>
#include <rdma/fi_atomic.h>

#include <test/user.h>


_Static_assert((TEST_API_VERSION_MAJOR == 0), "bad version");

enum dest_mode {
	SHARED_DEST,	/* senders all write to same offset */
	SEPARATE_DEST	/* senders write to different offset */
};

enum test_op {
	READ,
	WRITE,
	ADD
};

struct test_arguments {
	size_t transfer_size;
	enum dest_mode dest_mode;
	enum test_op test_op;
	size_t repeat;
};

#define DEFAULT_TRANSFER_SIZE 4
#define DEFAULT_DEST_MODE SEPARATE_DEST
#define DEFAULT_TEST_OP WRITE

static int parse_arguments(
		const int argc,
		char * const *argv,
		struct test_arguments **arguments,
		size_t *buffer_size)
{
	int longopt_idx=0, op;
	static struct option longopt[] = {
		{"size", required_argument, 0, 's'},
		{"dest-mode", required_argument, 0, 'd'},
		{"op", required_argument, 0, 'o'},
		{"repeat", required_argument, 0, 'r'},
		{"help", no_argument, 0, 'h'},
		{0}
	};

	struct test_arguments *args = calloc(sizeof(struct test_arguments), 1);
	if (!args)
		return -ENOMEM;

	*args = (struct test_arguments) {
		.transfer_size = DEFAULT_TRANSFER_SIZE,
		.dest_mode = DEFAULT_DEST_MODE,
		.test_op = DEFAULT_TEST_OP,
		.repeat = 1
	};

	if (argc > 0 && argv != NULL) {
		while ((op = getopt_long(argc, argv, "s:d:o:r:h", longopt, &longopt_idx)) != -1) {
			switch (op) {
			case 's':
				if (sscanf(optarg, "%zu", &args->transfer_size) != 1) {
					fprintf(stderr, "failed to parse transfer size\n");
					return -EINVAL;
				}
				break;
			case 'd':
				if (strcmp(optarg, "shared-dest") == 0) {
					args->dest_mode = SHARED_DEST;
				} else if (strcmp(optarg, "separate-dest") == 0) {
					args->dest_mode = SEPARATE_DEST;
				} else {
					fprintf(stderr, "unable to parse one-sided destination buffer mode\n");
					return -EINVAL;
				}
				break;
			case 'o':
				if (strcmp(optarg, "write") == 0) {
					args->test_op = WRITE;
				} else if (strcmp(optarg, "read") == 0) {
					args->test_op = READ;
				} else if (strcmp(optarg, "add") == 0) {
					args->test_op = ADD;
				} else {
					fprintf(stderr, "unable to parse one-sided test op\n");
					return -EINVAL;
				}
				break;
			case 'r':
				if (sscanf(optarg, "%zu", &args->repeat) != 1)
				 	return -EINVAL;
				break;
			case 'h':
			default:
				fprintf(stderr, "<test arguments> := \n"
						"\t[-s | --size=<transfer size>]\n"
						"\t[-d | --dest_mode=<shared_dest|separate_dest>]\n"
						"\t[-o | --op=<write|read|add>]\n"
						"\t[-r | --repeat<N>]\n"
						"\t[-h | --help]\n");
				return -EINVAL;
			}
		}
	}

	*buffer_size = args->transfer_size;
	*arguments = args;

	return 0;
}

static
void free_arguments(struct test_arguments *arguments)
{
	free (arguments);
	return;
}


static
struct test_config config(const struct test_arguments *arguments)
{
	struct test_config config = {

		.minimum_caps = FI_RMA | FI_ATOMIC | FI_RMA_EVENT,

		.tx_use_cntr = false,
		.rx_use_cntr = true,
		.tx_use_cq = true,
		.rx_use_cq = false,

		.rx_use_mr = true,

		.tx_buffer_size = arguments->transfer_size,
		.rx_buffer_size = arguments->transfer_size,
		.tx_buffer_alignment = 0,
		.rx_buffer_alignment = 0,

		.tx_data_object_sharing = DATA_OBJECT_PER_DOMAIN,
		.rx_data_object_sharing = DATA_OBJECT_PER_TRANSFER,

		.tx_context_count = arguments->repeat,
		.rx_context_count = arguments->repeat,

		.mr_rx_flags =
				arguments->test_op == READ
					? FI_REMOTE_READ
					: FI_REMOTE_WRITE
	};

	return config;
}


static void tx_init_buffer(
		const struct test_arguments *arguments,
		uint8_t *buffer)
{
	switch (arguments->test_op) {
	case WRITE:
		memset (buffer, UCHAR_MAX, arguments->transfer_size);
		break;
	case READ:
		memset (buffer, 0, arguments->transfer_size);
		break;
	case ADD:
		memset (buffer, 1, arguments->transfer_size);
		break;
	}
}

static void rx_init_buffer(
		const struct test_arguments *arguments,
		uint8_t *buffer)
{
	switch (arguments->test_op) {
	case WRITE:
	case ADD:
		memset (buffer, 0, arguments->transfer_size);
		break;
	case READ:
		memset (buffer, UCHAR_MAX, arguments->transfer_size);
		break;
	}
}

struct fid_mr *create_mr(
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
	const uint64_t flags = FI_RMA_EVENT;
	void *context = NULL;

	ret = fi_mr_reg(
			domain,
			buffer,
			len,
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
	fprintf(stderr, "fi_mr_reg failed\n");
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

static size_t rx_window_usage (
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
		struct fid_cntr *rx_cntr
){
	size_t addr = 0;
	int ret = 0;
	int i;
	struct context_info *ctxinfo = op_context->ctxinfo;

	switch (arguments->dest_mode) {
	case SHARED_DEST:
		break;
	case SEPARATE_DEST:
		addr = rank * arguments->transfer_size;
		break;
	default:
		return -FI_EINVAL;
		break;
	}

	for (i=0; i<arguments->repeat; i++) {
		switch (arguments->test_op) {
		case WRITE:
			ret = fi_write(endpoint,
					buffer,
					arguments->transfer_size,
					desc,
					rx_address,
					addr,
					key,
					&ctxinfo[i].fi_context);
			break;
		case READ:
			ret = fi_read(endpoint,
					buffer,
					arguments->transfer_size,
					desc,
					rx_address,
					addr,
					key,
					&ctxinfo[i].fi_context);
			break;

		case ADD:
			ret = fi_atomic(endpoint,
					buffer,
					arguments->transfer_size,
					desc,
					rx_address,
					addr,
					key,
					FI_UINT8,
					FI_SUM,
					&ctxinfo[i]);
			break;

		default:
			return -FI_ENOSYS;
			break;
		}
	}

	return ret;
}

/* RMA memory region for receive is set up in advance; nothing to do. */
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
	return 0;
}


static int tx_cntr_completion(
		const struct test_arguments *arguments,
		const size_t completion_count,
		struct fid_cntr *cntr)
{
	return fi_cntr_wait(cntr, completion_count*arguments->repeat, -1);
}

static int rx_cntr_completion(
		const struct test_arguments *arguments,
		const size_t completion_count,
		struct fid_cntr *cntr)
{
	return fi_cntr_wait(cntr, completion_count*arguments->repeat, -1);
}

static int cq_completion(
		const struct test_arguments *args,
		struct op_context **op_contextp,
		struct fid_cq *cq)
{
	ssize_t ret;
	struct fi_cq_tagged_entry cq_entry;
	struct context_info *ctxinfo;
	struct op_context *op_context;

	while (1) {
		ret = fi_cq_read(cq, (void *) &cq_entry, 1);

		if (ret == 1) {
			ctxinfo = (struct context_info *)(cq_entry.op_context);
			op_context = ctxinfo->op_context;
			op_context->test_state++;

			if (op_context->test_state == args->repeat) {
				*op_contextp = op_context;
				break;
			}

			continue;
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
		}

		return ret;
	}

	return 0;
}

static int tx_cq_completion(
		const struct test_arguments *args,
		struct op_context **context,
		struct fid_cq *cq)
{
	return cq_completion(args, context, cq);
}


static int datacheck(
		const struct test_arguments *arguments,
		const uint8_t *buffer,
		uint8_t expected,
		size_t len)
{
	int our_ret = FI_SUCCESS;
	size_t b;

	for (b = 0; b < len; b += 1) {
		if (((uint8_t *) buffer)[b] != expected) {
			fprintf (stderr, "datacheck failed at byte %zu (expected %u, got %u)\n",
					b, expected, ((uint8_t *) buffer)[b]);
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
	uint8_t expected;

	switch (arguments->test_op) {
	case READ:
	case WRITE:
		expected = UINT8_MAX; break;
	case ADD:
		expected = 1; break;
	default:
		return -FI_EINVAL;
	}

	return datacheck(
		arguments,
		buffer,
		expected,
		arguments->transfer_size);
}

static int rx_datacheck(
		const struct test_arguments *args,
		const uint8_t *buffer,
		size_t rx_peers)
{
	uint8_t expected;
	size_t len;

	switch (args->test_op) {
	case READ:
	case WRITE:
		expected = UINT8_MAX;
		break;
	case ADD:
		expected =
			args->test_op == ADD
				? args->dest_mode == SHARED_DEST
					? rx_peers * args->repeat
					: args->repeat
				: UINT8_MAX;
		break;
	default:
		return -FI_EINVAL;
	}

	len = args->dest_mode == SHARED_DEST
		? args->transfer_size
		: args->transfer_size * rx_peers;

	return datacheck(
			args,
			buffer,
			expected,
			len);
}


static int fini_buffer(
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
		.tx_create_mr = &create_mr,
		.rx_create_mr = &create_mr,

		.tx_window_usage = &tx_window_usage,
		.rx_window_usage = &rx_window_usage,

		.tx_transfer = &tx_transfer,
		.rx_transfer = &rx_transfer,
		.tx_cntr_completion = &tx_cntr_completion,
		.rx_cntr_completion = &rx_cntr_completion,
		.tx_cq_completion = &tx_cq_completion,
		.rx_cq_completion = NULL,

		.tx_datacheck = &tx_datacheck,
		.rx_datacheck = &rx_datacheck,

		.tx_fini_buffer = &fini_buffer,
		.rx_fini_buffer = &fini_buffer,
		.tx_destroy_mr = &destroy_mr,
		.rx_destroy_mr = &destroy_mr
	};

	return api;
}
