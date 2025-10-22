/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <unistd.h>

#include <rdma/fabric.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>

#include "shared.h"
#include "hmem.h"

static struct fid_ep **server_eps;
static char **send_bufs, **recv_bufs;
static struct fid_mr **send_mrs, **recv_mrs;
static void **send_descs, **recv_descs;
static struct fi_context2 *recv_ctx, *send_ctx;
static struct fid_cq **server_txcqs, **server_rxcqs;
static struct fid_av **server_avs;
static fi_addr_t *remote_fiaddr;
static int num_server_eps = 3;
static bool directed_recv = false;
static bool unexpected_path = false;
static bool implicit_av = false;

int get_one_comp(struct fid_cq *cq)
{
	struct fi_cq_err_entry comp;
	struct fi_cq_err_entry cq_err;

	memset(&cq_err, 0, sizeof(cq_err));
	int ret;

	do {
		ret = fi_cq_read(cq, &comp, 1);
		if (ret > 0)
			break;

		if (ret < 0) {
			if (ret != -FI_EAGAIN) {
				printf("fi_cq_read returns error %d\n", ret);
				(void) fi_cq_readerr(cq, &cq_err, 0);
				return ret;
			}
		}
	} while (1);

	return FI_SUCCESS;
}

static void free_res(void)
{
	int i;

	for (i = 0; i < num_server_eps; i++) {
		FT_CLOSE_FID(send_mrs[i]);
		FT_CLOSE_FID(recv_mrs[i]);

		if (send_bufs[i])
			(void) ft_hmem_free(opts.iface, (void *) send_bufs[i]);
		if (recv_bufs[i])
			(void) ft_hmem_free(opts.iface, (void *) recv_bufs[i]);
	}

	free(send_bufs);
	free(recv_bufs);
	free(send_mrs);
	free(recv_mrs);
	free(send_descs);
	free(recv_descs);
	free(send_ctx);
	free(recv_ctx);
	free(remote_fiaddr);
}

static void free_server_res(void)
{
	int i;

	free_res();

	for (i = 0; i < num_server_eps; i++) {
		FT_CLOSE_FID(server_eps[i]);
		FT_CLOSE_FID(server_txcqs[i]);
		FT_CLOSE_FID(server_rxcqs[i]);
		FT_CLOSE_FID(server_avs[i]);
	}

	free(server_txcqs);
	free(server_rxcqs);
	free(server_eps);
	free(server_avs);
}

static int alloc_bufs(void)
{
	int i, ret;
	size_t alloc_size;

	remote_fiaddr = calloc(num_server_eps, sizeof(*remote_fiaddr));
	send_ctx = calloc(num_server_eps, sizeof(*send_ctx));
	recv_ctx = calloc(num_server_eps, sizeof(*recv_ctx));

	send_bufs = calloc(num_server_eps, sizeof(*send_bufs));
	recv_bufs = calloc(num_server_eps, sizeof(*recv_bufs));

	if (!send_bufs || !recv_bufs || !remote_fiaddr || !send_ctx ||
	    !recv_ctx)
		return -FI_ENOMEM;

	alloc_size = opts.transfer_size < FT_MAX_CTRL_MSG ? FT_MAX_CTRL_MSG : opts.transfer_size;
	for (i = 0; i < num_server_eps; i++) {
		ret = ft_hmem_alloc(opts.iface, opts.device,
				    (void **) &send_bufs[i],
				    alloc_size);
		if (ret)
			return ret;

		ret = ft_hmem_alloc(opts.iface, opts.device,
				    (void **) &recv_bufs[i],
				    alloc_size);
		if (ret)
			return ret;

		if (ft_check_opts(FT_OPT_VERIFY_DATA)) {
			ret = ft_fill_buf(send_bufs[i], opts.transfer_size);
			if (ret)
				return ret;
		}
	}

	return 0;
}

static int alloc_server_res(void)
{
	int ret;

	server_eps = calloc(num_server_eps, sizeof(*server_eps));
	server_txcqs = calloc(num_server_eps, sizeof(*server_txcqs));
	server_rxcqs = calloc(num_server_eps, sizeof(*server_rxcqs));
	server_avs = calloc(num_server_eps, sizeof(*server_avs));

	if (!server_eps || !server_txcqs || !server_rxcqs || !server_avs)
		return -FI_ENOMEM;

	ret = alloc_bufs();
	if (ret)
		return ret;

	return 0;
}

static int setup_server_ep(int idx)
{
	int ret;

	ret = fi_endpoint(domain, fi, &server_eps[idx], NULL);
	if (ret) {
		FT_PRINTERR("fi_endpoint", ret);
		return ret;
	}

	ret = ft_alloc_ep_res(fi, &server_txcqs[idx], &server_rxcqs[idx], 
			      NULL, NULL, NULL, &server_avs[idx]);
	if (ret)
		return ret;

	ret = ft_enable_ep(server_eps[idx], eq, server_avs[idx], 
			   server_txcqs[idx], server_rxcqs[idx], 
			   NULL, NULL, NULL);
	if (ret)
		return ret;

	return 0;
}

static int reg_mrs(void)
{
	int i, ret;

	send_mrs = calloc(num_server_eps, sizeof(*send_mrs));
	recv_mrs = calloc(num_server_eps, sizeof(*recv_mrs));
	send_descs = calloc(num_server_eps, sizeof(*send_descs));
	recv_descs = calloc(num_server_eps, sizeof(*recv_descs));

	if (!send_mrs || !recv_mrs || !send_descs ||
	    !recv_descs)
		return -FI_ENOMEM;

	for (i = 0; i < num_server_eps; i++) {
		ret = ft_reg_mr(fi, send_bufs[i], opts.transfer_size,
				ft_info_to_mr_access(fi),
				(FT_MR_KEY + 1) * (i + 1), opts.iface,
				opts.device, &send_mrs[i], 
				&send_descs[i]);
		if (ret)
			return ret;

		ret = ft_reg_mr(fi, recv_bufs[i], opts.transfer_size,
				ft_info_to_mr_access(fi),
				(FT_MR_KEY + 2) * (i + 2), opts.iface,
				opts.device, &recv_mrs[i], 
				&recv_descs[i]);
		if (ret)
			return ret;
	}

	return 0;
}

static int server_post_send(int idx)
{
	return ft_post_tx_buf(server_eps[idx], remote_fiaddr[idx],
			      opts.transfer_size, NO_CQ_DATA, &send_ctx[idx],
			      send_bufs[idx], send_descs[idx], ft_tag);
}

static int client_post_recv(void)
{
	int i, ret = 0;
	fi_addr_t addr;

	for (i = 0; i < num_server_eps; i++) {
		if (directed_recv)
			addr = remote_fiaddr[i];
		else
			addr = FI_ADDR_UNSPEC;

		ret = ft_post_rx_buf(ep, addr, opts.transfer_size,
				     &recv_ctx[i], recv_bufs[i], recv_descs[i],
				     ft_tag);
		if (ret) {
			FT_PRINTERR("client_post_recv_directed", ret);
			return ret;
		}
	}

	return ret;
}

static int run_test(void)
{
	int i, ret;

	if (opts.dst_addr) {
		/* Client side - single endpoint */
		FT_INFO("Client: Step 1 - Post receive buffers\n");

		ret = alloc_bufs();
		if (ret)
			goto cleanup_client;

		ret = reg_mrs();
		if (ret)
			goto cleanup_client;

		if (!unexpected_path) {
			ret = client_post_recv();
			if (ret)
				goto cleanup_client;
		}

		FT_INFO("Client: Initial sync\n");
		ft_sync_oob();

		if (implicit_av) {
			FT_INFO("Implicit AV. Only server inserts client's address\n");
			/* Send client endpoint address to server */
			for (i = 0; i < num_server_eps; i++) {
				ret = ft_send_addr_oob(ep);
				if (ret) {
					FT_PRINTERR("ft_server_insert_addr_oob", ret);
					goto cleanup_client;
				}
			}
		} else {
			FT_INFO("Not using implicit AV. Full address exchange \n");
			for (i = 0; i < num_server_eps; i++) {
				ft_init_av_dst_addr(av, ep, &remote_fiaddr[i]);
			}
		}

		/* TODO: poll CQ while waiting for the OOB sync from the sender.
		 * Doing so will allow the test to run with large message sizes
		 * that use emulated RMA protocols and with delivery complete */
		FT_INFO("Client: Sync after send complete\n");
		ft_sync_oob();

		if (unexpected_path) {
			ret = client_post_recv();
			if (ret)
				goto cleanup_client;
		}

		FT_INFO("Client: Waiting for messages from %d server endpoints\n", num_server_eps);
		/* Wait for all receive completions */
		for (i = 0; i < num_server_eps; i++) {
			ret = get_one_comp(rxcq);
			if (ret) {
				FT_PRINTERR("get_client_comp", ret);
				goto cleanup_client;
			}
		}

		if (ft_check_opts(FT_OPT_VERIFY_DATA)) {
			for (i = 0; i < num_server_eps; i++) {
				ret = ft_check_buf(recv_bufs[i],
						   opts.transfer_size);
				if (ret) {
					FT_PRINTERR("ft_check_buf", ret);
					goto cleanup_client;
				}
			}
		}

		FT_INFO("Client: Final sync before end of test\n");
		ft_sync_oob();

cleanup_client:
		free_res();

	} else {
		/* Server side - multiple endpoints */
		FT_INFO("Server: Creating %d endpoints\n", num_server_eps);

		/* Create server endpoints */
		ret = alloc_server_res();
		if (ret)
			return ret;

		for (i = 0; i < num_server_eps; i++) {
			ret = setup_server_ep(i);
			if (ret)
				goto cleanup_server;
		}

		ret = reg_mrs();
		if (ret)
			goto cleanup_server;

		FT_INFO("Server: Initial sync\n");
		ft_sync_oob();

		if (implicit_av) {
			/* Initialize AV for each endpoint OOB */
			FT_INFO("Implicit AV. Only sender inserts receiver's address\n");
			for (i = 0; i < num_server_eps; i++) {
				ret = ft_recv_addr_oob(server_avs[i], &remote_fiaddr[i]);
				if (ret) {
					FT_PRINTERR("ft_server_insert_addr_oob", ret);
					goto cleanup_server;
				}
			}
		} else {
			FT_INFO("Not using implicit AV. Full address exchange \n");
			for (i = 0; i < num_server_eps; i++) {
				ft_init_av_dst_addr(server_avs[i], server_eps[i], &remote_fiaddr[i]);
				FT_INFO("fi_addr %ld\n", remote_fiaddr[i]);
			}
		}

		FT_INFO("Server: Step 1 - Send messages from all endpoints\n");
		/* Send from all endpoints */
		for (i = 0; i < num_server_eps; i++) {
			ret = server_post_send(i);
			if (ret) {
				FT_PRINTERR("server_post_send", ret);
				goto cleanup_server;
			}
		}

		/* Wait for all send completions */
		for (i = 0; i < num_server_eps; i++) {
			ret = get_one_comp(server_txcqs[i]);
			if (ret) {
				FT_PRINTERR("get_server_comp", ret);
				goto cleanup_server;
			}
		}

		FT_INFO("Server: Sync after send complete\n");
		ft_sync_oob();

		FT_INFO("Server: Final sync before end of test\n");
		ft_sync_oob();

cleanup_server:
		free_server_res();
	}

	if (ret == FI_SUCCESS)
		FT_INFO("Test completed successfully\n");
	return ret;
}

int main(int argc, char **argv)
{
	int i, op, ret;

	opts = INIT_OPTS;
	opts.transfer_size = 64;
	opts.options |= FT_OPT_OOB_ADDR_EXCH;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt(argc, argv, "c:XLrvUh" ADDR_OPTS INFO_OPTS CS_OPTS)) != -1) {
		switch (op) {
		default:
			ft_parse_addr_opts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			break;
		case 'c':
			num_server_eps = atoi(optarg);
			break;
		case 'X':
			unexpected_path = true;
			break;
		case 'L':
			implicit_av = true;
			break;
		case 'r':
			directed_recv = true;
			break;
		case 'v':
			opts.options |= FT_OPT_VERIFY_DATA;
			break;
		case 'U':
			FT_ERR("Test does not support delivery complete\n");
			return -FI_EINVAL;
		case '?':
		case 'h':
			ft_usage(argv[0], "AV message order test");
			FT_PRINT_OPTS_USAGE("-c <int>",
				"number of server endpoints (default 3)");
			return EXIT_FAILURE;
		}
	}

	if (directed_recv && implicit_av) {
		FT_ERR("Directed receive cannot be used with implicit AV\n");
		return -FI_EINVAL;
	}

	/* Exchange addresses OOB and avoid posting initial receive to avoid
	 * conflict with receives posted in the test */
	opts.options |= FT_OPT_OOB_ADDR_EXCH | FT_OPT_SKIP_MSG_ALLOC;

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->caps = FI_MSG;
	hints->mode = FI_CONTEXT | FI_CONTEXT2;
	hints->ep_attr->type = FI_EP_RDM;
	hints->domain_attr->mr_mode = opts.mr_mode;
	hints->addr_format = opts.address_format;

	ret = ft_init_fabric();
	if (ret)
		return ret;

	if (!(opts.options & FT_OPT_SIZE)) {
		for (i = 0; i < TEST_CNT; i++) {
			if (!ft_use_size(i, opts.sizes_enabled))
				continue;
			opts.transfer_size = test_size[i].size;
			FT_INFO("Running test for message size: %ld\n", opts.transfer_size);
			ret = run_test();
			if (ret)
				return ret;
		}
	} else {
		FT_INFO("Running test for message size: %ld\n",
		       opts.transfer_size);
		ret = run_test();
		if (ret)
			return ret;
	}

	ft_free_res();
	return ft_exit_code(ret);
}
