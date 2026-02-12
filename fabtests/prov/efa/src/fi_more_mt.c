/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

/**
 * FI_MORE multi-threaded test
 *
 * Validates FI_MORE batching correctness under concurrent posting and
 * completion polling threads. Operations are posted in batches of 64
 * with FI_MORE set on all but the last in each batch.
 *
 * Client threads:
 * - Send thread: Posts batches of operations with FI_MORE flag
 * - TX CQ thread: Polls TX completions and validates that each
 *   completion returns a unique, non-NULL context pointer
 *
 * Server threads:
 * - RX post thread: Posts receive buffers (for send/senddata/writedata)
 * - RX CQ thread: Polls RX completions and validates CQ data if applicable
 *
 * Supported operations (-o flag):
 * - send: fi_sendmsg
 * - senddata: fi_sendmsg with FI_REMOTE_CQ_DATA
 * - write: fi_writemsg (no server-side completions)
 * - writedata: fi_writemsg with FI_REMOTE_CQ_DATA
 *
 * Requires FI_THREAD_SAFE. Uses OOB sync for teardown coordination.
 */

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_errno.h>

#include "shared.h"

#define BATCH_SIZE 64
#define CQ_BATCH_SIZE 64

static pthread_t send_thread, cq_thread;
static struct fi_context2 *send_contexts;
static void **received_contexts;
static int total_sends;
static volatile int sends_posted = 0;
static volatile int sends_completed = 0;

static inline bool op_is_rma(void)
{
	return opts.rma_op == FT_RMA_WRITE || opts.rma_op == FT_RMA_WRITEDATA;
}

static void *send_thread_func(void *arg)
{
	int ret, i, batch_idx;
	struct fi_msg msg;
	struct fi_msg_rma rma_msg;
	struct iovec iov;
	struct fi_rma_iov rma_iov;
	uint64_t flags;

	printf("Send thread: Starting to post %d ops in batches of %d\n",
	       total_sends, BATCH_SIZE);

	iov.iov_base = tx_buf;
	iov.iov_len = opts.transfer_size;

	if (op_is_rma()) {
		rma_iov.addr = remote.addr;
		rma_iov.len = opts.transfer_size;
		rma_iov.key = remote.key;

		rma_msg.msg_iov = &iov;
		rma_msg.desc = &mr_desc;
		rma_msg.iov_count = 1;
		rma_msg.addr = remote_fi_addr;
		rma_msg.rma_iov = &rma_iov;
		rma_msg.rma_iov_count = 1;
		rma_msg.data = (opts.rma_op == FT_RMA_WRITEDATA) ? remote_cq_data : 0;
	} else {
		msg.msg_iov = &iov;
		msg.desc = &mr_desc;
		msg.iov_count = 1;
		msg.addr = remote_fi_addr;
		msg.data = (opts.cqdata_op == FT_CQDATA_SENDDATA) ? remote_cq_data : 0;
	}

	for (i = 0; i < total_sends; i++) {
		batch_idx = i % BATCH_SIZE;
		flags = (batch_idx < BATCH_SIZE - 1) ? FI_MORE : 0;

		if (opts.cqdata_op == FT_CQDATA_SENDDATA ||
		    opts.rma_op == FT_RMA_WRITEDATA)
			flags |= FI_REMOTE_CQ_DATA;

		if (op_is_rma()) {
			rma_msg.context = &send_contexts[i];
			ret = fi_writemsg(ep, &rma_msg, flags);
		} else {
			msg.context = &send_contexts[i];
			ret = fi_sendmsg(ep, &msg, flags);
		}

		if (ret) {
			FT_PRINTERR(op_is_rma() ? "fi_writemsg" : "fi_sendmsg", ret);
			return NULL;
		}

		sends_posted++;
	}

	printf("Send thread: Posted all %d ops\n", total_sends);
	return NULL;
}

static void *cq_thread_func(void *arg)
{
	int ret, i, j, num_comps;
	struct fi_cq_data_entry comps[CQ_BATCH_SIZE];
	bool error_found = false;

	printf("CQ thread: Starting to poll completions in batches of %d\n", CQ_BATCH_SIZE);

	while (sends_completed < total_sends) {
		ret = fi_cq_read(txcq, comps, CQ_BATCH_SIZE);
		if (ret > 0) {
			num_comps = ret;
			for (j = 0; j < num_comps; j++) {
				if (comps[j].op_context == NULL) {
					printf("ERROR: NULL context at completion %d!\n",
					       sends_completed + 1);
					error_found = true;
				}

				for (i = 0; i < sends_completed; i++) {
					if (received_contexts[i] == comps[j].op_context) {
						printf("ERROR: Duplicate context %p at completion %d!\n",
						       comps[j].op_context, sends_completed + 1);
						error_found = true;
						break;
					}
				}

				received_contexts[sends_completed] = comps[j].op_context;
				sends_completed++;
			}
		} else if (ret == -FI_EAVAIL) {
			ft_cq_readerr(txcq);
			return NULL;
		} else if (ret < 0 && ret != -FI_EAGAIN) {
			FT_PRINTERR("fi_cq_read", ret);
			return NULL;
		}
	}

	if (error_found)
		printf("CQ thread: FAILED - NULL or duplicate contexts detected!\n");
	else
		printf("CQ thread: PASSED - got all %d unique non-NULL completions\n", total_sends);
	return NULL;
}

static int run_client(void)
{
	int ret;

	if (op_is_rma()) {
		ret = ft_exchange_keys(&remote);
		if (ret)
			return ret;
	}

	total_sends = opts.iterations;

	send_contexts = calloc(total_sends, sizeof(*send_contexts));
	if (!send_contexts) {
		FT_ERR("Failed to allocate send contexts");
		return -FI_ENOMEM;
	}

	received_contexts = calloc(total_sends, sizeof(*received_contexts));
	if (!received_contexts) {
		FT_ERR("Failed to allocate received contexts array");
		free(send_contexts);
		return -FI_ENOMEM;
	}

	ret = pthread_create(&send_thread, NULL, send_thread_func, NULL);
	if (ret) {
		FT_ERR("Failed to create send thread");
		goto cleanup;
	}

	ret = pthread_create(&cq_thread, NULL, cq_thread_func, NULL);
	if (ret) {
		FT_ERR("Failed to create CQ thread");
		goto cleanup;
	}

	pthread_join(send_thread, NULL);
	pthread_join(cq_thread, NULL);

	printf("Client: %s FI_MORE test - posted %d, completed %d\n",
	       (sends_completed == total_sends) ? "PASSED" : "FAILED",
	       sends_posted, sends_completed);

cleanup:
	free(send_contexts);
	free(received_contexts);
	return ret;
}

static pthread_t rx_post_thread, rx_cq_thread;
static volatile int recvs_completed = 0;

static void *rx_post_thread_func(void *arg)
{
	int ret, i;
	int expected = opts.iterations;

	printf("RX post thread: Posting %d receive buffers\n", expected);
	for (i = 0; i < expected; i++) {
		ret = ft_post_rx(ep, opts.transfer_size, &rx_ctx);
		if (ret) {
			FT_PRINTERR("ft_post_rx", ret);
			return NULL;
		}
	}
	printf("RX post thread: Posted all %d receives\n", expected);
	return NULL;
}

static void *rx_cq_thread_func(void *arg)
{
	int ret;
	struct fi_cq_data_entry comp;
	int expected = opts.iterations;
	bool check_cq_data = (opts.rma_op == FT_RMA_WRITEDATA ||
			      opts.cqdata_op == FT_CQDATA_SENDDATA);

	printf("RX CQ thread: Waiting for %d completions\n", expected);
	while (recvs_completed < expected) {
		ret = fi_cq_read(rxcq, &comp, 1);
		if (ret > 0) {
			if (check_cq_data && comp.data != remote_cq_data) {
				printf("ERROR: Expected CQ data 0x%lx, got 0x%lx at completion %d\n",
				       (unsigned long)remote_cq_data,
				       (unsigned long)comp.data, recvs_completed + 1);
				return NULL;
			}
			recvs_completed++;
		} else if (ret == -FI_EAVAIL) {
			ft_cq_readerr(rxcq);
			return NULL;
		} else if (ret < 0 && ret != -FI_EAGAIN) {
			FT_PRINTERR("fi_cq_read", ret);
			return NULL;
		}
	}

	printf("RX CQ thread: PASSED - received all %d completions\n", expected);
	return NULL;
}

static int run_server(void)
{
	int ret;

	if (op_is_rma()) {
		ret = ft_exchange_keys(&remote);
		if (ret)
			return ret;

		if (opts.rma_op == FT_RMA_WRITE) {
			printf("Server: PASSED - write completed\n");
			return 0;
		}
	}

	if (!op_is_rma() || (fi->mode & FI_RX_CQ_DATA)) {
		ret = pthread_create(&rx_post_thread, NULL, rx_post_thread_func, NULL);
		if (ret) {
			FT_ERR("Failed to create RX post thread");
			return ret;
		}
	}

	ret = pthread_create(&rx_cq_thread, NULL, rx_cq_thread_func, NULL);
	if (ret) {
		FT_ERR("Failed to create RX CQ thread");
		return ret;
	}

	if (!op_is_rma() || (fi->mode & FI_RX_CQ_DATA))
		pthread_join(rx_post_thread, NULL);
	pthread_join(rx_cq_thread, NULL);

	printf("Server: %s - received %d of %d completions\n",
	       (recvs_completed == opts.iterations) ? "PASSED" : "FAILED",
	       recvs_completed, opts.iterations);
	return 0;
}

int main(int argc, char **argv)
{
	int op, ret;

	opts = INIT_OPTS;
	opts.iterations = 1024;
	opts.transfer_size = 64;
	opts.rma_op = 0;
	opts.cqdata_op = 0;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	hints->caps = FI_MSG | FI_RMA;
	hints->mode = FI_CONTEXT | FI_CONTEXT2;
	hints->ep_attr->type = FI_EP_RDM;
	hints->domain_attr->mr_mode = FI_MR_VIRT_ADDR | FI_MR_ALLOCATED |
				      FI_MR_PROV_KEY | FI_MR_LOCAL;

	while ((op = getopt_long(argc, argv, "h" ADDR_OPTS INFO_OPTS CS_OPTS API_OPTS,
				 long_opts, &lopt_idx)) != -1) {
		switch (op) {
		default:
			if (!ft_parse_long_opts(op, optarg))
				continue;
			ft_parse_addr_opts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			ret = ft_parse_api_opts(op, optarg, hints, &opts);
			if (ret)
				return ret;
			break;
		case '?':
		case 'h':
			ft_usage(argv[0], "FI_MORE multi-threaded test");
			FT_PRINT_OPTS_USAGE("-o <op>", "op: send|senddata|write|writedata");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	opts.threading = FI_THREAD_SAFE;
	cq_attr.format = FI_CQ_FORMAT_DATA;

	ret = ft_init_fabric();
	if (ret)
		return ret;

	if (opts.dst_addr)
		ret = run_client();
	else
		ret = run_server();

	ft_sync_oob();
	ft_free_res();
	return ft_exit_code(ret);
}
