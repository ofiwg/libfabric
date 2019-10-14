/*
 * Copyright (c) 2017-2019 Intel Corporation. All rights reserved.
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
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <limits.h>
#include <stdarg.h>

#include <rdma/fi_errno.h>
#include <rdma/fi_domain.h>
#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_trigger.h>

#include <core.h>
#include <pattern.h>
#include <shared.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <assert.h>

struct pattern_ops *pattern;
struct multinode_xfer_state state;

static inline void multinode_init_state()
{
	state.cur_source = PATTERN_NO_CURRENT;
	state.cur_target = PATTERN_NO_CURRENT;

	state.all_completions_done = false;
	state.all_recvs_posted = false;
	state.all_sends_posted = false;

	state.rx_window = opts.window_size;
	state.tx_window = opts.window_size;
}

static int multinode_setup_fabric(int argc, char **argv)
{
	char my_name[FT_MAX_CTRL_MSG];
	size_t len;
	int ret, i;
	struct fi_rma_iov *remote = malloc(sizeof(struct fi_rma_iov));

	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_MSG | FI_RMA;
	hints->mode = FI_CONTEXT;
	hints->domain_attr->mr_mode = opts.mr_mode & ~FI_MR_VIRT_ADDR;

	tx_seq = 0;
	rx_seq = 0;
	tx_cq_cntr = 0;
	rx_cq_cntr = 0;

	if (pm_job.my_rank != 0)
		pm_barrier();

	ret = ft_getinfo(hints, &fi);
	if (ret)
		return ret;

	ret = ft_open_fabric_res();
	if (ret)
		return ret;

	opts.av_size = pm_job.num_ranks;
	ret = ft_alloc_active_res(fi);
	if (ret)
		return ret;

	ret = ft_enable_ep(ep, eq, av, txcq, rxcq, txcntr, rxcntr);
	if (ret)
		return ret;

	len = FT_MAX_CTRL_MSG;
	ret = fi_getname(&ep->fid, (void *) my_name, &len);
	if (ret) {
		FT_PRINTERR("error determining local endpoint name\n", ret);
		goto err;
	}

	pm_job.name_len = len;
	pm_job.names = malloc(len * pm_job.num_ranks);
	if (!pm_job.names) {
		FT_ERR("error allocating memory for address exchange\n");
		ret = -FI_ENOMEM;
		goto err;
	}

	if (pm_job.my_rank == 0)
		pm_barrier();

	ret = pm_allgather(my_name, pm_job.names, pm_job.name_len);
	if (ret) {
		FT_PRINTERR("error exchanging addresses\n", ret);
		goto err;
	}

	pm_job.fi_addrs = calloc(pm_job.num_ranks, sizeof(*pm_job.fi_addrs));
	if (!pm_job.fi_addrs) {
		FT_ERR("error allocating memory for av fi addrs\n");
		ret = -FI_ENOMEM;
		goto err;
	}

	ret = fi_av_insert(av, pm_job.names, pm_job.num_ranks,
			   pm_job.fi_addrs, 0, NULL);
	if (ret != pm_job.num_ranks) {
		FT_ERR("unable to insert all addresses into AV table\n");
		ret = -1;
		goto err;
	}
	
	pm_job.fi_iovs = malloc(sizeof(struct fi_rma_iov) * pm_job.num_ranks);
	if (!pm_job.fi_iovs) {
		FT_ERR("error allocation memory for rma_iovs\n");
		goto err;
	}
	
	remote->addr = 0; 	
	remote->key = fi_mr_key(mr);	
	remote->len = rx_size;
	
	ret = pm_allgather(remote, pm_job.fi_iovs, sizeof(*remote));
	if (ret) {	
		FT_ERR("error exchanging rma_iovs\n");
		goto err;
	}
	for (i = 0; i < pm_job.num_ranks; i++) {
		pm_job.fi_iovs[i].addr = (tx_size * pm_job.my_rank);
	}

	return 0;
err:
	ft_free_res();
	return ft_exit_code(ret);
}

static int multinode_rma_setup(int argc, char** argv) 
{
	opts.options &= (~FT_OPT_ALLOC_MULT_MR);
	return multinode_setup_fabric(argc, argv);
}

static int socket_setup_fabric(int argc, char** argv)
{
	return multinode_setup_fabric(argc, argv);
}

static int multinode_post_rx()
{
	int ret, offset;

	/* post receives */
	while (!state.all_recvs_posted) {

		if (state.rx_window == 0)
			break;

		ret = pattern->next_source(&state.cur_source);
		if (ret == -FI_ENODATA) {
			state.all_recvs_posted = true;
			break;
		} else if (ret < 0) {
			return ret;
		}

		offset = state.recvs_posted % opts.window_size ;
		assert(rx_ctx_arr[offset].state == OP_DONE);

		ret = ft_post_rx_buf(ep, opts.transfer_size,
				     &rx_ctx_arr[offset].context,
				     rx_ctx_arr[offset].buf,
				     rx_ctx_arr[offset].desc, 0);
		if (ret)
			return ret;

		rx_ctx_arr[offset].state = OP_PENDING;
		state.recvs_posted++;
		state.rx_window--;
	};
	return 0;
}

static int multinode_post_tx()
{
	int ret, offset;
	fi_addr_t dest;

	while (!state.all_sends_posted) {

		if (state.tx_window == 0)
			break;

		ret = pattern->next_target(&state.cur_target);
		if (ret == -FI_ENODATA) {
			state.all_sends_posted = true;
			break;
		} else if (ret < 0) {
			return ret;
		}

		offset = state.sends_posted % opts.window_size;
		assert(tx_ctx_arr[offset].state == OP_DONE);

		tx_ctx_arr[offset].buf[0] = offset;
		dest = pm_job.fi_addrs[state.cur_target];
		ret = ft_post_tx_buf(ep, dest, opts.transfer_size,
				     NO_CQ_DATA,
				     &tx_ctx_arr[offset].context,
				     tx_ctx_arr[offset].buf,
				     tx_ctx_arr[offset].desc, 0);
		if (ret)
			return ret;

		tx_ctx_arr[offset].state = OP_PENDING;
		state.sends_posted++;
		state.tx_window--;
	}
	return 0;
}

static int multinode_wait_for_comp()
{
	int ret, i;

	ret = ft_get_tx_comp(tx_seq);
	if (ret)
		return ret;

	ret = ft_get_rx_comp(rx_seq);
	if (ret)
		return ret;

	for (i = 0; i < opts.window_size; i++) {
		rx_ctx_arr[i].state = OP_DONE;
		tx_ctx_arr[i].state = OP_DONE;
	}

	state.rx_window = opts.window_size;
	state.tx_window = opts.window_size;

	if (state.all_recvs_posted && state.all_sends_posted)
		state.all_completions_done = true;

	return 0;
}

int send_recv_barrier() 
{
	int ret;

	multinode_init_state();
	pattern = &patterns[1];
	
	ret = multinode_post_tx();
	if (ret)
		return ret;

	ret = multinode_post_rx();
	if (ret)
		return ret;

	ret = multinode_wait_for_comp();

	return ret;
}

static int multinode_rma_write() 
{
	int ret;
	struct fi_msg_rma *message = (struct fi_msg_rma *) malloc(sizeof(struct fi_msg_rma));
	struct iovec *loc_iov = (struct iovec *) malloc(sizeof(struct iovec));

	message->desc = &mr_desc;
	message->context = NULL;
	message->rma_iov_count = 1;
	message->iov_count = 1;
	message->msg_iov = loc_iov;
	message->data = 0;

	while (!state.all_sends_posted) {

		if (state.tx_window == 0)
			break;

		ret = pattern->next_target(&state.cur_target);
		if (ret == -FI_ENODATA) {
			state.all_sends_posted = true;
			break;
		} else if (ret < 0) {
			return ret;
		}
		
		loc_iov->iov_base = (void *) (tx_buf + tx_size * state.cur_target);
		loc_iov->iov_len = tx_size;
		
		snprintf((char*) loc_iov->iov_base, tx_size,
				"Hello World! from %zu to %i on the %zuth iteration, %s test", 
				pm_job.my_rank, state.cur_target, (size_t) tx_seq, pattern->name);

		message->rma_iov = &pm_job.fi_iovs[state.cur_target];
		message->addr = pm_job.fi_addrs[state.cur_target];
		message->context = &tx_ctx_arr[state.tx_window].context;
	
		do {
			ret = fi_writemsg(ep, message, FI_DELIVERY_COMPLETE);
		} while (ret == -FI_EAGAIN);

		if (ret) { 
			printf("rma post failed: %i\n", ret);
			return ret;
		}
		tx_seq++;
		state.sends_posted++;
		state.tx_window--;
	}
	return 0;
}

static int multinode_rma_recv() 
{
	state.all_recvs_posted = true;
	return 0;
}

static int multinode_rma_wait() 
{
	int ret;
	
	ret = ft_get_tx_comp(tx_seq);	
	if (ret)
		return ret;
	
	ret = ft_get_rx_comp(rx_seq);
	if (ret)
		return ret;

	state.rx_window = opts.window_size;
	state.tx_window = opts.window_size;

	if (state.all_recvs_posted && state.all_sends_posted)
		state.all_completions_done = true;	
	
	ret = send_recv_barrier();
	if (ret)
		return ret;
	
	return 0;
}


static int multinode_run_test(struct multinode_xfer_method method)
{
	int ret;
	int iter;

	for (iter = 0; iter < opts.iterations; iter++) {
		multinode_init_state();
		while (!state.all_completions_done ||
				!state.all_recvs_posted ||
				!state.all_sends_posted) {
			
			ret = method.send();
			if (ret)
				return ret;
			
			ret = method.recv();
			if (ret)
				return ret;

			ret = method.wait();
			if (ret)
				return ret;
		}
	}
	return 0;
}

static void pm_job_free_res()
{
	free(pm_job.names);

	free(pm_job.fi_addrs);
	
	free(pm_job.fi_iovs);
}

int multinode_run_tests(int argc, char **argv)
{
	int ret = FI_SUCCESS;
	int i, j;
	struct fi_info *cp_hints;
	
	cp_hints = fi_dupinfo(hints);

	struct multinode_xfer_method method[] = {
		{
			.name = "rma",
			.send = multinode_rma_write,
			.recv = multinode_rma_recv,
			.wait = multinode_rma_wait,
			.setup = multinode_rma_setup,
		},
		{
			.name = "send/recv",
			.send = multinode_post_tx,
			.recv = multinode_post_rx,
			.wait = multinode_wait_for_comp,
			.setup = socket_setup_fabric,
		}
	};
		
	for (i = 0; i < ARRAY_SIZE(method); i++) {
		ret = method[i].setup(argc, argv);
		if (ret)
			return ret;
		printf("Transfer Method: %s\n", method[i].name);	
	
		for (j = 0; j < NUM_TESTS; j++) {
			printf("starting %s... ", patterns[j].name);
			pattern = &patterns[j];
			ret = multinode_run_test(method[i]);
			if (ret) {
				printf("failed: %i\n", ret);
				return ret;
			} else {
				printf("passed\n");
			}
			ret = send_recv_barrier();
			if (ret)
				return ret;
		}

		printf("\n\n");		
		pm_job_free_res();
		ft_free_res();
		hints = fi_dupinfo(cp_hints);
	}

	return ft_exit_code(ret);
}
