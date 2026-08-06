/*
 * Copyright (c) 2013-2015 Intel Corporation.  All rights reserved.
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
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <unistd.h>

#include <rdma/fi_errno.h>
#include <rdma/fi_rpc.h>

#include <shared.h>


#define RPC_CNT		10
#define RPC_REQ_SIZE	256
#define RPC_RESP_SIZE	4096
#define RPC_TIMEOUT	(-1)

struct {
	uint64_t	rpc_id;
	int		timeout;
	bool		done;
} rpc_reqs[RPC_CNT];

static int handle_one_rpc_req(int i)
{
	int ret;

	do {
		ret = fi_rpc_resp(ep, tx_buf + RPC_RESP_SIZE * i, RPC_RESP_SIZE,
				  mr_desc, remote_fi_addr, rpc_reqs[i].rpc_id,
				  &tx_ctx_arr[i].context);
	} while (ret == -FI_EAGAIN);

	if (ret) {
		fprintf(stderr, "fi_rpc_resp failed: %d\n", ret);
		return ret;
	}

	rpc_reqs[i].done = true;
	return 0;
}

static int process_rpc_reqs(int count)
{
	int i, ret;
	int completions = 0;

	while (completions < count) {
		i = rand() % count;
		if (rpc_reqs[i].done)
			continue;

		ret = handle_one_rpc_req(i);
		if (ret)
			continue;
		completions++;
	}

	return ret;
}

static int wait_for_rpc_reqs(int count)
{
	int ret, completions = 0;
	struct fi_cq_rpc_entry comp;
	struct fi_cq_err_entry cq_err;

	do {
		ret = fi_cq_read(rxcq, &comp, 1);
		if (ret == -FI_EAGAIN)
			continue;
		if (ret == -FI_EAVAIL) {
			ret = fi_cq_readerr(rxcq, &cq_err, 0);
			if (ret < 0)
				FT_PRINTERR("fi_cq_readerr", ret);
			else
				ret = -cq_err.err;
			return ret;
		}
		if (ret < 0) {
			FT_PRINTERR("fi_cq_read", ret);
			return ret;
		}
		if (ret == 0) {
			continue;
		}
		if ((comp.flags & FI_RPC) && (comp.flags & FI_RECV)) {
			printf("Get RPC request: rpc_id 0x%lx timeout %d size %lu\n",
				comp.rpc_id, comp.timeout, comp.len);
			rpc_reqs[completions].rpc_id = comp.rpc_id;
			rpc_reqs[completions].timeout = comp.timeout;
			rpc_reqs[completions].done = false;
		} else {
			fprintf(stderr, "Unexpected CQ entry\n");
			return -FI_EIO;
		}
		completions++;
	} while (completions < count);

	return 0;
}

static int wait_for_rpc_resps(int count)
{
	int ret, completions = 0;
	struct fi_cq_tagged_entry comp;
	struct fi_cq_err_entry cq_err;

	do {
		ret = fi_cq_read(rxcq, &comp, 1);
		if (ret == -FI_EAGAIN)
			continue;
		if (ret == -FI_EAVAIL) {
			ret = fi_cq_readerr(rxcq, &cq_err, 0);
			if (ret < 0)
				FT_PRINTERR("fi_cq_readerr", ret);
			else
				ret = -cq_err.err;
			return ret;
		}
		if (ret < 0) {
			FT_PRINTERR("fi_cq_read", ret);
			return ret;
		}
		if ((comp.flags & FI_RPC) && (comp.flags & FI_RECV)) {
			printf("Get RPC response: rpc_id 0x%lx size %lu\n",
			       comp.tag, comp.len);
		} else {
			fprintf(stderr, "Unexpected CQ entry\n");
			return -FI_EIO;
		}
		completions++;
	} while (completions < count);

	return 0;
}

static int do_server(void)
{
	int i, ret;

	printf("Posting %d buffers for RPC requests\n", RPC_CNT);
	for (i = 0; i < RPC_CNT; i++) {
		ret = fi_recv(ep, rx_buf + RPC_REQ_SIZE * i, RPC_REQ_SIZE,
			      mr_desc, remote_fi_addr, &rx_ctx_arr[i].context);
		if (ret)
			return ret;
	}

	printf("Waiting for RPC requests\n");
	wait_for_rpc_reqs(RPC_CNT);

	printf("Processing RPC requests in random order\n");
	process_rpc_reqs(RPC_CNT);

	return 0;
}

static int do_client(void)
{
	int i, ret;

	printf("Sending %d RPC requests\n", RPC_CNT);

	(void) ft_fill_buf(tx_buf, tx_size);

	for (i = 0; i < RPC_CNT; i++) {
		do {
			ret = fi_rpc(ep, tx_buf + RPC_REQ_SIZE * i,
				     RPC_REQ_SIZE, mr_desc,
				     rx_buf + RPC_RESP_SIZE * i, RPC_RESP_SIZE,
				     mr_desc, remote_fi_addr, RPC_TIMEOUT,
				     &tx_ctx_arr[i].context);
		} while (ret == -FI_EAGAIN);
		if (ret)
			return ret;
	}

	printf("Waiting for RPC responses\n");
	ret = wait_for_rpc_resps(RPC_CNT);

	return ret;
}

static int run(void)
{
	int ret;

	ret = ft_init_fabric();
	if (ret)
		return ret;

	if (tx_size < RPC_REQ_SIZE * RPC_CNT) {
		FT_PRINTERR("Tx buffer too small", -FI_EINVAL);
		return -FI_EINVAL;
	}

	if (rx_size < RPC_RESP_SIZE * RPC_CNT) {
		FT_PRINTERR("Rx buffer too small", -FI_EINVAL);
		return -FI_EINVAL;
	}

	if (opts.dst_addr) {
		ret = do_client();
		if (ret)
			return ret;
	} else {
		ret = do_server();
		if (ret)
			return ret;
	}

	ft_sync();
	return 0;
}

int main(int argc, char **argv)
{
	int ret, op;

	opts = INIT_OPTS;
	opts.options |= FT_OPT_OOB_SYNC;

	hints = fi_allocinfo();
	if (!hints) {
		FT_PRINTERR("fi_allocinfo", -FI_ENOMEM);
		return EXIT_FAILURE;
	}

	while ((op = getopt(argc, argv, "h" CS_OPTS INFO_OPTS)) != -1) {
		switch (op) {
		default:
			ft_parsecsopts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints, &opts);
			break;
		case '?':
		case 'h':
			ft_csusage(argv[0], "An RDM client-server example for RPC.\n");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->domain_attr->resource_mgmt = FI_RM_ENABLED;
	hints->tx_attr->msg_order = FI_ORDER_SAS;
	hints->rx_attr->msg_order = FI_ORDER_SAS;
	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_MSG | FI_RPC;
	hints->mode = FI_CONTEXT | FI_CONTEXT2;
	hints->domain_attr->mr_mode = opts.mr_mode;
	hints->addr_format = opts.address_format;

	cq_attr.format = FI_CQ_FORMAT_RPC;

	ret = run();

	ft_free_res();
	return ft_exit_code(ret);
}
