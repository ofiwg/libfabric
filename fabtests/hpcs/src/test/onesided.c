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
	bool use_workqueue;
};

#define DEFAULT_TRANSFER_SIZE 4
#define DEFAULT_DEST_MODE SEPARATE_DEST
#define DEFAULT_TEST_OP WRITE
#define DEFAULT_USE_WORKQUEUE 0

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
		{"offset-mode", required_argument, 0, 'm'},
		{"op", required_argument, 0, 'o'},
		{"repeat", required_argument, 0, 'r'},
		{"workqueue", no_argument, 0, 'w'},
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

	while ((op = getopt_long(argc, argv, "s:d:m:o:r:wh", longopt, &longopt_idx)) != -1) {
		switch (op) {
		case 's':
			if (sscanf(optarg, "%zu", &args->transfer_size) != 1) {
				fprintf(stderr, "failed to parse transfer size\n");
				return -EINVAL;
			}
			break;
		case 'd':
		case 'm':
			if (strcmp(optarg, "shared-dest") == 0 ||
					strcmp(optarg, "same-offset") == 0)
			{
				args->dest_mode = SHARED_DEST;
			} else if (strcmp(optarg, "separate-dest") == 0 ||
					strcmp(optarg, "different-offset") == 0)
			{
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
		case 'w':
			args->use_workqueue = 1;
			break;
		case 'h':
		default:
			fprintf(stderr, "<test arguments> := \n"
					"\t[-s | --size=<transfer size>]\n"
					"\t[-m | --offset-mode=<same-offset|different-offset>]\n"
					"\t[-o | --op=<write|read|add>]\n"
					"\t[-r | --repeat<n>]\n"
					"\t[-w | --workqueue]\n"
					"\t[-h | --help]\n");
			return -EINVAL;
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

		.tx_use_cntr = true,
		.rx_use_cntr = true,
		.tx_use_cq = false,
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
		uint8_t *buffer,
		size_t len)
{
	switch (arguments->test_op) {
	case WRITE:
		memset(buffer, UCHAR_MAX, arguments->transfer_size);
		break;
	case READ:
		memset(buffer, 0, arguments->transfer_size);
		break;
	case ADD:
		memset(buffer, 1, arguments->transfer_size);
		break;
	}
}

static struct fid_mr *rx_create_mr(const struct test_arguments *arguments,
                struct fid_domain *domain, const uint64_t key,
                uint8_t *buffer, size_t len, uint64_t access, uint64_t flags)
{
	return test_generic_rx_create_mr(arguments, domain, key,
			buffer, len, access, flags | FI_RMA_EVENT);
}

static void rx_init_buffer(const struct test_arguments *arguments,
		uint8_t *buffer, size_t len)
{
	switch (arguments->test_op) {
	case WRITE:
	case ADD:
		memset(buffer, 0, arguments->transfer_size);
		break;
	case READ:
		memset(buffer, UCHAR_MAX, arguments->transfer_size);
		break;
	}
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
		uint64_t flags
){
	size_t addr = 0;
	int ret = 0;
	int i;

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
		struct context_info *ctxinfo = &op_context->ctxinfo[i];

		switch (arguments->test_op) {
		case WRITE:
		case READ:
			ctxinfo->iov = (struct iovec) {
				.iov_base = buffer,
				.iov_len = arguments->transfer_size
			};

			ctxinfo->rma_remote_iov = (struct fi_rma_iov) {
				.addr = addr,
				.len = arguments->transfer_size,
				.key = key
			};

			ctxinfo->rma_msg = (struct fi_msg_rma) {
				.msg_iov = &ctxinfo->iov,
				.desc = desc,
				.iov_count = 1,
				.addr = rx_address,
				.rma_iov = &ctxinfo->rma_remote_iov,
				.rma_iov_count = 1,
				.context = &ctxinfo->fi_context
			};
			break;
		case ADD:
			ctxinfo->ioc = (struct fi_ioc) {
				.addr = buffer,
				.count = arguments->transfer_size
			};

			ctxinfo->rma_remote_ioc = (struct fi_rma_ioc) {
				.addr = addr,
				.count = arguments->transfer_size,
				.key = key
			};

			ctxinfo->atomic_msg = (struct fi_msg_atomic) {
				.msg_iov = &ctxinfo->ioc,
				.desc = desc,
				.iov_count = 1,
				.addr = rx_address,
				.rma_iov = &ctxinfo->rma_remote_ioc,
				.rma_iov_count = 1,
				.datatype = FI_UINT8,
				.op = FI_SUM,
				.context = &ctxinfo->fi_context
			};
			break;
		}

		if (arguments->use_workqueue && flags & FI_TRIGGER) {
			struct fi_deferred_work *work = &ctxinfo->def_work;

			work->triggering_cntr =
					ctxinfo->fi_trig_context.trigger.threshold.cntr;
			work->threshold =
					ctxinfo->fi_trig_context.trigger.threshold.threshold;
			work->completion_cntr =
					op_context->tx_cntr;

			switch (arguments->test_op) {
			case WRITE:
				ctxinfo->rma_op = (struct fi_op_rma) {
					.ep = endpoint,
					.msg = ctxinfo->rma_msg,
					.flags = flags
				};
				work->op_type = FI_OP_WRITE;
				work->op.rma = &ctxinfo->rma_op;
				break;
			case READ:
				ctxinfo->rma_op = (struct fi_op_rma) {
					.ep = endpoint,
					.msg = ctxinfo->rma_msg,
					.flags = flags
				};
				work->op_type = FI_OP_READ;
				work->op.rma = &ctxinfo->rma_op;
				break;
			case ADD:
				ctxinfo->atomic_op = (struct fi_op_atomic) {
					.ep = endpoint,
					.msg = ctxinfo->atomic_msg,
					.flags = flags
				};
				work->op_type = FI_OP_ATOMIC;
				work->op.atomic = &ctxinfo->atomic_op;
				break;
			default:
				return -FI_ENOSYS;
				break;
			}

			ret = fi_control(&op_context->domain->fid, FI_QUEUE_WORK, work);
		} else {
			switch (arguments->test_op) {
			case WRITE:
				ret = fi_writemsg(endpoint, &ctxinfo->rma_msg, flags);
				break;
			case READ:
				ret = fi_readmsg(endpoint, &ctxinfo->rma_msg, flags);
				break;
			case ADD:
				ret = fi_atomicmsg(endpoint, &ctxinfo->atomic_msg, flags);
				break;
			default:
				return -FI_ENOSYS;
				break;
			}
		};
	}

	return ret;
}

/* RMA memory region for receive is set up in advance; nothing to do. */
static int rx_transfer(const struct test_arguments *arguments,
		const size_t transfer_id, const size_t transfer_count,
		const fi_addr_t tx_address, struct fid_ep *endpoint,
		struct op_context *op_context,
		uint8_t *buffer, void *desc, uint64_t flags)
{
	return 0;
}

static int datacheck(const struct test_arguments *arguments,
		const uint8_t *buffer, uint8_t expected, size_t len)
{
	int our_ret = FI_SUCCESS;
	size_t b;

	for (b = 0; b < len; b += 1) {
		if (((uint8_t *) buffer)[b] != expected) {
			fprintf (stderr, "datacheck failed at byte %zu (expected %u, got %u)\n",
					b, expected, ((uint8_t *) buffer)[b]);
			our_ret = -FI_EIO;
			goto err_data_check;
		}
	}

err_data_check:
	return our_ret;
}

static int tx_datacheck(const struct test_arguments *arguments,
		const uint8_t *buffer, size_t len)
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

	return datacheck(arguments, buffer, expected, arguments->transfer_size);
}

static int rx_datacheck(const struct test_arguments *args,
		const uint8_t *buffer, size_t len, size_t rx_peers)
{
	uint8_t expected;

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

	return datacheck(args, buffer, expected, len);
}

struct test_api test_api(void)
{
	struct test_api api = {
		.parse_arguments = &parse_arguments,
		.free_arguments = &free_arguments,

		.config = &config,

		.tx_init_buffer = &tx_init_buffer,
		.rx_init_buffer = &rx_init_buffer,
		.tx_create_mr = &test_generic_tx_create_mr,
		.rx_create_mr = &rx_create_mr,

		.tx_window_usage = &test_generic_tx_window_usage,
		.rx_window_usage = &test_generic_rx_window_usage,

		.tx_transfer = &tx_transfer,
		.rx_transfer = &rx_transfer,
		.tx_cntr_completion = &test_generic_tx_cntr_completion,
		.rx_cntr_completion = &test_generic_rx_cntr_completion,
		.tx_cq_completion = NULL,
		.rx_cq_completion = NULL,

		.tx_datacheck = &tx_datacheck,
		.rx_datacheck = &rx_datacheck,

		.tx_fini_buffer = &test_generic_tx_fini_buffer,
		.rx_fini_buffer = &test_generic_rx_fini_buffer,
		.tx_destroy_mr = &test_generic_tx_destroy_mr,
		.rx_destroy_mr = &test_generic_rx_destroy_mr
	};

	return api;
}
