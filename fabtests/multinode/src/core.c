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
struct multi_xfer_method method;
struct multi_xfer_method multi_xfer_methods[] = {
	{
		.name = "send/recv",
		.send = multi_msg_send,
		.recv = multi_msg_recv,
		.wait = multi_msg_wait,
	}
};

static int multi_setup_fabric(int argc, char **argv)
{
	char my_name[FT_MAX_CTRL_MSG];
	size_t len;
	int ret;

	if (pm_job.transfer_method == multi_msg) {
		hints->caps = FI_MSG;
	} else {
		printf("Not a valid cabability\n");
		return -FI_ENODATA;
	}

	method = multi_xfer_methods[pm_job.transfer_method];

	hints->ep_attr->type = FI_EP_RDM;
	hints->mode = FI_CONTEXT;
	hints->domain_attr->mr_mode = opts.mr_mode;

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
	return 0;
err:
	ft_free_res();
	return ft_exit_code(ret);
}

int multi_msg_recv()
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

int multi_msg_send()
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

int multi_msg_wait()
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

static inline void multi_init_state()
{
	state.cur_source = PATTERN_NO_CURRENT;
	state.cur_target = PATTERN_NO_CURRENT;

	state.all_completions_done = false;
	state.all_recvs_posted = false;
	state.all_sends_posted = false;

	state.rx_window = opts.window_size;
	state.tx_window = opts.window_size;
}

static int multi_run_test()
{
	int ret;
	int iter;

	for (iter = 0; iter < opts.iterations; iter++) {

		multi_init_state();
		while (!state.all_completions_done ||
				!state.all_recvs_posted ||
				!state.all_sends_posted) {
			ret = method.recv();
			if (ret)
				return ret;

			ret = method.send();
			if (ret)
				return ret;

			ret = method.wait();
			if (ret)
				return ret;

			pm_barrier();
		}
	}
	return 0;
}

static void pm_job_free_res()
{

	free(pm_job.names);

	free(pm_job.fi_addrs);
}

int multinode_run_tests(int argc, char **argv)
{
	int ret = FI_SUCCESS;
	int i;

	ret = multi_setup_fabric(argc, argv);
	if (ret)
		return ret;

	for (i = 0; i < NUM_TESTS && !ret; i++) {
		printf("starting %s... ", patterns[i].name);
		pattern = &patterns[i];
		ret = multi_run_test();
		if (ret)
			printf("failed\n");
		else
			printf("passed\n");
	}

	pm_job_free_res();
	ft_free_res();
	return ft_exit_code(ret);
}
