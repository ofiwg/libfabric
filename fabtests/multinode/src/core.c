/*
<<<<<<< HEAD
 * Copyright (c) 2017-2019 Intel Corporation. All rights reserved.
=======
 * Copyright (c) 2017-2018 Intel Corporation. All rights reserved.
>>>>>>> fabtests/multinode: Initial version of multinode sendrecv test
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

struct pattern_ops *pattern;
<<<<<<< HEAD
struct multinode_xfer_state state;

static int multinode_setup_fabric(int argc, char **argv)
{
	char my_name[FT_MAX_CTRL_MSG];
=======

struct multinode_xfer_state state =

	(struct multinode_xfer_state) {
	.recvs_posted = 0,
	.sends_posted = 0,
	.recvs_done = 0,
	.sends_done = 0,

	.tx_window = 0,
	.rx_window = 0,

	.cur_sender = PATTERN_NO_CURRENT,
	.cur_receiver = PATTERN_NO_CURRENT,

	.all_sends_done = false,
	.all_recvs_done = false,
	.all_completions_done =	false,

	.tx_flags = 0,
	.rx_flags = 0,
};

static int multinode_init_fabric(int argc, char **argv)
{
	void *my_name;
>>>>>>> fabtests/multinode: Initial version of multinode sendrecv test
	size_t len;
	int ret;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_MSG;
	hints->mode = FI_CONTEXT;
	hints->domain_attr->mr_mode = opts.mr_mode;

	tx_seq = 0;
	rx_seq = 0;
	tx_cq_cntr = 0;
	rx_cq_cntr = 0;

	ret = ft_getinfo(hints, &fi);
	if (ret)
		return ret;

	ret = ft_open_fabric_res();
	if (ret)
		return ret;

<<<<<<< HEAD
	opts.av_size = pm_job.num_ranks;
=======
	opts.av_size = pm_job.ranks;
>>>>>>> fabtests/multinode: Initial version of multinode sendrecv test
	ret = ft_alloc_active_res(fi);
	if (ret)
		return ret;

<<<<<<< HEAD
=======
	len = FT_MAX_CTRL_MSG;
	my_name = calloc(len, 1);
	if (!my_name) {
		FT_ERR("allocating memory failed\n");
		ret = -FI_ENOMEM;
		goto err1;
	}

>>>>>>> fabtests/multinode: Initial version of multinode sendrecv test
	ret = ft_enable_ep(ep, eq, av, txcq, rxcq, txcntr, rxcntr);
	if (ret)
		return ret;

<<<<<<< HEAD
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
=======
	ret = fi_getname(&ep->fid, (void *) my_name, &len);
	if (ret) {
		FT_PRINTERR("error determining local endpoint name\n", ret);
		goto err2;
	}

	pm_job.names = malloc(len * pm_job.ranks);
	pm_job.name_len = len;

	if (pm_job.names == NULL) {
		FT_ERR("error allocating memory for address exchange\n");
		ret = -FI_ENOMEM;
		goto err2;
	}

	ret = pm_job.allgather(my_name, pm_job.names, pm_job.name_len);
	if (ret) {
		FT_PRINTERR("error exchanging addresses\n", ret);
		goto err2;
	}

	pm_job.fi_addrs = calloc(pm_job.ranks, sizeof(*pm_job.fi_addrs));
	ret = fi_av_insert(av, pm_job.names, pm_job.ranks,
			   pm_job.fi_addrs, 0, NULL);
	if (ret != pm_job.ranks) {
		FT_ERR("unable to insert all addresses into AV table\n");
		ret = -1;
		goto err2;
	} else {
		ret = 0;
	}


	return 0;
err2:
	free(my_name);
err1:
	ft_free_res();
	return ft_exit_code(ret);
}

static int multinode_close_fabric(int ret)
{
>>>>>>> fabtests/multinode: Initial version of multinode sendrecv test
	ft_free_res();
	return ft_exit_code(ret);
}

static int multinode_post_rx()
{
	int ret, prev, offset;

	/* post receives */
	while (!state.all_recvs_done) {
<<<<<<< HEAD
		prev = state.cur_source;

		ret = pattern->next_source(&state.cur_source);
=======
		prev = state.cur_sender;
		
		ret = pattern->next_sender(&state.cur_sender);
>>>>>>> fabtests/multinode: Initial version of multinode sendrecv test
		if (ret == -ENODATA) {
			state.all_recvs_done = true;
			break;
		} else if (ret < 0) {
			return ret;
		}

		if (state.rx_window == 0) {
<<<<<<< HEAD
			state.cur_source = prev;
=======
			state.cur_sender = prev;
>>>>>>> fabtests/multinode: Initial version of multinode sendrecv test
			break;
		}

		offset = state.recvs_posted % opts.window_size ;
		/* find context and buff */
		if (rx_ctx_arr[offset].state != OP_DONE) {
<<<<<<< HEAD
			state.cur_source = prev;
=======
			state.cur_sender = prev;
>>>>>>> fabtests/multinode: Initial version of multinode sendrecv test
			break;
		}

		ret = ft_post_rx_buf(ep, opts.transfer_size,
				     &rx_ctx_arr[offset],
				     rx_ctx_arr[offset].buf,
				     rx_ctx_arr[offset].desc, 0);
		if (ret)
			return ret;

<<<<<<< HEAD
		rx_ctx_arr[offset].state = OP_PENDING;
=======
		rx_ctx_arr[offset].state = 20 + offset;
>>>>>>> fabtests/multinode: Initial version of multinode sendrecv test
		state.recvs_posted++;
		state.rx_window--;
	};
	return 0;
}

static int multinode_post_tx()
{
	int ret, prev, offset;
	fi_addr_t dest;

	while (!state.all_sends_done) {

<<<<<<< HEAD
		prev = state.cur_target;

		ret = pattern->next_target(&state.cur_target);
=======
		prev = state.cur_receiver;

		ret = pattern->next_receiver(&state.cur_receiver);
>>>>>>> fabtests/multinode: Initial version of multinode sendrecv test
		if (ret == -ENODATA) {
			state.all_sends_done = true;
			break;
		} else if (ret < 0) {
			return ret;
		}

		if (state.tx_window == 0) {
<<<<<<< HEAD
			state.cur_target = prev;
=======
			state.cur_receiver = prev;
>>>>>>> fabtests/multinode: Initial version of multinode sendrecv test
			break;
		}

		offset = state.sends_posted % opts.window_size;

		if (tx_ctx_arr[offset].state != OP_DONE) {
<<<<<<< HEAD
			state.cur_target = prev;
			break;
		}
		tx_ctx_arr[offset].buf[0] = offset;
		dest = pm_job.fi_addrs[state.cur_target];
=======
			state.cur_receiver = prev;
			break;
		}

		tx_ctx_arr[offset].buf[0] = offset;
		dest = pm_job.fi_addrs[state.cur_receiver];
>>>>>>> fabtests/multinode: Initial version of multinode sendrecv test
		ret = ft_post_tx_buf(ep, dest, opts.transfer_size,
				     NO_CQ_DATA,
				     &tx_ctx_arr[offset],
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

	state.all_completions_done = true;
	return 0;
}

<<<<<<< HEAD
static inline void multinode_init_state()
{
	state.cur_source = PATTERN_NO_CURRENT;
	state.cur_target = PATTERN_NO_CURRENT;

	state.all_completions_done = false;
	state.all_recvs_done = false;
	state.all_sends_done = false;

	state.rx_window = opts.window_size;
	state.tx_window = opts.window_size;
}

static int multinode_run_test()
{
	int ret;
	int iter;

	for (iter = 0; iter < opts.iterations; iter++) {

		multinode_init_state();
=======
static int multinode_run_test()
{
	int ret;

	for (state.iteration = 0; state.iteration < opts.iterations; state.iteration++) {
		state.cur_sender = PATTERN_NO_CURRENT;
		state.cur_receiver = PATTERN_NO_CURRENT;

		state.all_completions_done = false;
		state.all_recvs_done = false;
		state.all_sends_done = false;

		state.rx_window = opts.window_size;
		state.tx_window = opts.window_size;
>>>>>>> fabtests/multinode: Initial version of multinode sendrecv test

		while (!state.all_completions_done || !state.all_recvs_done || !state.all_sends_done) {
			ret = multinode_post_rx();
			if (ret)
				return ret;

			ret = multinode_post_tx();
			if (ret)
				return ret;

			ret = multinode_wait_for_comp();
			if (ret)
				return ret;
		}
	}
<<<<<<< HEAD
	pm_barrier();
	return 0;
}

static void pm_job_free_res()
{
	if (pm_job.names)
		free(pm_job.names);

	if (pm_job.fi_addrs)
	free(pm_job.fi_addrs);
}

=======
	pm_job.barrier();
	return 0;
}

>>>>>>> fabtests/multinode: Initial version of multinode sendrecv test
int multinode_run_tests(int argc, char **argv)
{
	int ret = FI_SUCCESS;

<<<<<<< HEAD
	ret = multinode_setup_fabric(argc, argv);
	if (ret)
		return ret;

	pattern = &full_mesh_ops;

	ret = multinode_run_test();

	pm_job_free_res();
	ft_free_res();
	return ft_exit_code(ret);
=======
	ret = multinode_init_fabric(argc, argv);
	if (ret)
		return ret;

	FT_DEBUG("all to all test\n");
	pattern = &all2all_ops;
	ret = multinode_run_test();
	if (ret){
		FT_ERR("all to all failed");
		return ret;
	}

	FT_DEBUG("gather test\n");
	pattern = &gather_ops;
	ret = multinode_run_test();
	if (ret){
		FT_ERR("all to one failed");
		return ret;
	}

	FT_DEBUG("broadcast test\n");
	pattern = &broadcast_ops;
	ret = multinode_run_test();
	if (ret){
		FT_ERR("broadcast failed");
		return ret;
	}

	FT_DEBUG("ring test\n");
	pattern = &ring_ops;
	ret = multinode_run_test();
	if (ret){
		FT_ERR("ring test failed");
		return ret;
	}

	multinode_close_fabric(ret);
	return ret;
>>>>>>> fabtests/multinode: Initial version of multinode sendrecv test
}
