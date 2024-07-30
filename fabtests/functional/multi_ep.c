/*
 * Copyright (c) 2013-2017 Intel Corporation. All rights reserved.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <time.h>
#include <netdb.h>
#include <unistd.h>

#include <rdma/fabric.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_cm.h>

#include "shared.h"
#include "hmem.h"

static struct fid_ep **eps;
static char **send_bufs, **recv_bufs;
static struct fid_mr **send_mrs, **recv_mrs;
static void **send_descs, **recv_descs;
static struct fi_rma_iov *peer_iovs;
static struct fi_context *recv_ctx;
static struct fi_context *send_ctx;
static struct fid_cq **txcqs, **rxcqs;
static struct fid_av **avs;
static fi_addr_t *remote_fiaddr;
static bool shared_cq = false;
static bool shared_av = false;
int num_eps = 3;

enum {
	LONG_OPT_SHARED_AV,
	LONG_OPT_SHARED_CQ,
};

static void free_ep_res()
{
	int i;

	for (i = 0; i < num_eps; i++) {
		if (fi->domain_attr->mr_mode & FI_MR_RAW)
			(void) fi_mr_unmap_key(domain, peer_iovs[i].key);

		FT_CLOSE_FID(send_mrs[i]);
		FT_CLOSE_FID(recv_mrs[i]);
		FT_CLOSE_FID(eps[i]);

		(void) ft_hmem_free(opts.iface, (void *) send_bufs[i]);
		(void) ft_hmem_free(opts.iface, (void *) recv_bufs[i]);
	}

	for (i = 0; i < num_eps; i++) {
		FT_CLOSE_FID(txcqs[i]);
		FT_CLOSE_FID(rxcqs[i]);
		FT_CLOSE_FID(avs[i]);
	}

	free(txcqs);
	free(rxcqs);
	free(send_bufs);
	free(recv_bufs);
	free(send_mrs);
	free(recv_mrs);
	free(peer_iovs);
	free(send_descs);
	free(recv_descs);
	free(send_ctx);
	free(recv_ctx);
	free(remote_fiaddr);
	free(eps);
	free(avs);
}

static int reg_mrs(void)
{
	int i, ret;

	for (i = 0; i < num_eps; i++) {
		ret = ft_reg_mr(fi, send_bufs[i], opts.transfer_size,
				ft_info_to_mr_access(fi),
				(FT_MR_KEY + 1) * (i + 1), opts.iface,
				opts.device, &send_mrs[i], &send_descs[i]);
		if (ret)
			return ret;

		ret = ft_reg_mr(fi, recv_bufs[i], opts.transfer_size,
				ft_info_to_mr_access(fi),
				(FT_MR_KEY + 2) * (i + 2), opts.iface,
				opts.device, &recv_mrs[i], &recv_descs[i]);
		if (ret)
			return ret;
	}

	return FI_SUCCESS;
}

static int alloc_multi_ep_res()
{
	int i, ret;

	eps = calloc(num_eps, sizeof(*eps));
	remote_fiaddr = calloc(num_eps, sizeof(*remote_fiaddr));
	send_mrs = calloc(num_eps, sizeof(*send_mrs));
	recv_mrs = calloc(num_eps, sizeof(*recv_mrs));
	send_descs = calloc(num_eps, sizeof(*send_descs));
	recv_descs = calloc(num_eps, sizeof(*recv_descs));
	peer_iovs = calloc(num_eps, sizeof(*peer_iovs));
	send_ctx = calloc(num_eps, sizeof(*send_ctx));
	recv_ctx = calloc(num_eps, sizeof(*recv_ctx));
	send_bufs = calloc(num_eps, opts.transfer_size);
	recv_bufs = calloc(num_eps, opts.transfer_size);

	txcqs = calloc(num_eps, sizeof(*txcqs));
	rxcqs = calloc(num_eps, sizeof(*rxcqs));
	avs = calloc(num_eps, sizeof(*avs));

	if (!eps || !remote_fiaddr || !send_bufs || !recv_bufs ||
	    !send_ctx || !recv_ctx || !send_bufs || !recv_bufs ||
	    !send_mrs || !recv_mrs || !send_descs || !recv_descs ||
	    !txcqs || !rxcqs || !peer_iovs)
		return -FI_ENOMEM;

	for (i = 0; i < num_eps; i++) {
		ret = ft_hmem_alloc(opts.iface, opts.device,
				    (void **) &send_bufs[i], opts.transfer_size);
		if (ret)
			return ret;

		ret = ft_hmem_alloc(opts.iface, opts.device,
				    (void **) &recv_bufs[i], opts.transfer_size);
		if (ret)
			return ret;
	}

	return 0;
}

static int ep_post_rx(int idx)
{
	int ret, cq_read_idx = idx;

	if (shared_cq)
		cq_read_idx = 0;

	do {
		ret = fi_recv(eps[idx], recv_bufs[idx], opts.transfer_size,
			      recv_descs[idx], FI_ADDR_UNSPEC,
			      &recv_ctx[idx]);
		if (ret == -FI_EAGAIN)
			(void) fi_cq_read(rxcqs[cq_read_idx], NULL, 0);

	} while (ret == -FI_EAGAIN);

	return ret;
}

static int ep_post_tx(int idx, size_t len)
{
	int ret, cq_read_idx = idx;

	if (shared_cq)
		cq_read_idx = 0;

	do {
		ret = fi_send(eps[idx], send_bufs[idx], len,
			      send_descs[idx], remote_fiaddr[idx],
			      &send_ctx[idx]);
		if (ret == -FI_EAGAIN)
			(void) fi_cq_read(txcqs[cq_read_idx], NULL, 0);

	} while (ret == -FI_EAGAIN);

	return ret;
}

static int ep_post_write(int idx)
{
	int ret, cq_read_idx = idx;

	if (shared_cq)
		cq_read_idx = 0;

	do {
		ret = fi_write(eps[idx], send_bufs[idx], opts.transfer_size,
			       send_descs[idx], remote_fiaddr[idx],
			       peer_iovs[idx].addr, peer_iovs[idx].key,
			       &send_ctx[idx]);
		if (ret == -FI_EAGAIN)
			(void) fi_cq_read(txcqs[cq_read_idx], NULL, 0);

	} while (ret == -FI_EAGAIN);

	return ret;
}

static int get_one_comp(struct fid_cq *cq)
{
	struct fi_cq_err_entry comp;
	int ret, i;

	do {
		ret = fi_cq_read(cq, &comp, 1);
		if (ret > 0)
			break;

		if (ret < 0 && ret != -FI_EAGAIN)
			return ret;

		if (!shared_cq) {
			/* Drive progress on all EPs in case peer is waiting on
			 * different EP pair
			 */
			for (i = 0; i < num_eps; i++)
				(void) fi_cq_read(rxcqs[i], NULL, 0);
		}
	} while (1);

	return FI_SUCCESS;
}

static int sync_all(void)
{
	int i, ret, cq_read_idx;

	for (i = 0; i < num_eps; i++) {
		ret = ep_post_rx(i);
		if (ret) {
			FT_PRINTERR("fi_recv", ret);
			return ret;
		}

		ret = ep_post_tx(i, 0);
		if (ret) {
			FT_PRINTERR("fi_send", ret);
			return ret;
		}

		cq_read_idx = shared_cq ? 0 : i;

		ret = get_one_comp(txcqs[cq_read_idx]);
		if (ret)
			return ret;

		ret = get_one_comp(rxcqs[cq_read_idx]);
		if (ret)
			return ret;
	}
	return FI_SUCCESS;
}

static int do_sends(void)
{
	char temp[FT_MAX_CTRL_MSG];
	struct fi_rma_iov *rma_iov = (struct fi_rma_iov *) temp;
	int i, ret, cq_read_idx;
	size_t key_size, len;

	for (i = 0; i < num_eps; i++) {
		ret = ep_post_rx(i);
		if (ret) {
			FT_PRINTERR("fi_recv", ret);
			return ret;
		}
	}

	memset(peer_iovs, 0, sizeof(*peer_iovs) * num_eps);

	printf("Send RMA info to all %d remote EPs\n", num_eps);
	for (i = 0; i < num_eps; i++) {
		len = opts.transfer_size;
		ret = ft_fill_rma_info(recv_mrs[i], recv_bufs[i], rma_iov,
				       &key_size, &len);
		if (ret)
			return ret;

		ret = ft_hmem_copy_to(opts.iface, opts.device, send_bufs[i],
				      rma_iov, len);
		if (ret)
			return ret;

		ret = ep_post_tx(i, len);
		if (ret) {
			FT_PRINTERR("fi_send", ret);
			return ret;
		}

		cq_read_idx = shared_cq ? 0 : i;

		ret = get_one_comp(rxcqs[cq_read_idx]);
		if (ret)
			return ret;

		ret = get_one_comp(txcqs[cq_read_idx]);
		if (ret)
			return ret;
	}

	printf("Wait for all messages from peer\n");
	for (i = 0; i < num_eps; i++) {
		ret = ft_hmem_copy_from(opts.iface, opts.device, rma_iov,
				        recv_bufs[i], len);
		if (ret)
			return ret;

		ret = ft_get_rma_info(rma_iov, &peer_iovs[i], key_size);
		if (ret)
			return ret;
	}

	ret = sync_all();
	if (ret)
		return ret;

	printf("PASSED multi ep sends\n");
	return 0;
}

static int do_rma(void)
{
	int i, ret, cq_read_idx;

	for (i = 0; i < num_eps; i++) {
		if (ft_check_opts(FT_OPT_VERIFY_DATA)) {
			ret = ft_fill_buf(send_bufs[i], opts.transfer_size);
			if (ret)
				return ret;
		}
		ret = ep_post_write(i);
		if (ret)
			return ret;
	}

	printf("Wait for all writes from peer\n");
	for (i = 0; i < num_eps; i++) {
		cq_read_idx = shared_cq ? 0 : i;
		ret = get_one_comp(txcqs[cq_read_idx]);
		if (ret)
			return ret;
	}

	ret = sync_all();
	if (ret)
		return ret;

	if (ft_check_opts(FT_OPT_VERIFY_DATA)) {
		for (i = 0; i < num_eps; i++) {
			ret = ft_check_buf(recv_bufs[i], opts.transfer_size);
			if (ret)
				return ret;
		}
		printf("Data check OK\n");
	}

	printf("PASSED multi ep writes\n");
	return 0;
}

static int setup_client_ep(int idx)
{
	int ret, av_bind_idx = idx, cq_bind_idx = idx;

	if (shared_cq)
		cq_bind_idx = 0;

	if (shared_av)
		av_bind_idx = 0;

	ret = fi_endpoint(domain, fi, &eps[idx], NULL);
	if (ret) {
		FT_PRINTERR("fi_endpoint", ret);
		return ret;
	}

	ret = ft_alloc_ep_res(fi, &txcqs[idx], &rxcqs[idx], NULL, NULL, NULL, &avs[idx]);
	if (ret)
		return ret;

	ret = ft_enable_ep(eps[idx], eq, avs[av_bind_idx], txcqs[cq_bind_idx], rxcqs[cq_bind_idx],
			   NULL, NULL, NULL);
	if (ret)
		return ret;

	ret = ft_connect_ep(eps[idx], eq, fi->dest_addr);
	if (ret)
		return ret;

	return 0;
}

static int setup_server_ep(int idx)
{
	int ret, av_bind_idx = idx, cq_bind_idx = idx;

	if (shared_cq)
		cq_bind_idx = 0;

	if (shared_av)
		av_bind_idx = 0;

	ret = ft_retrieve_conn_req(eq, &fi);
	if (ret)
		goto failed_accept;

	ret = fi_endpoint(domain, fi, &eps[idx], NULL);
	if (ret) {
		FT_PRINTERR("fi_endpoint", ret);
		goto failed_accept;
	}

	ret = ft_alloc_ep_res(fi, &txcqs[idx], &rxcqs[idx], NULL, NULL, NULL, &avs[idx]);
	if (ret)
		return ret;

	ret = ft_enable_ep(eps[idx], eq, avs[av_bind_idx], txcqs[cq_bind_idx], rxcqs[cq_bind_idx],
			   NULL, NULL, NULL);
	if (ret)
		goto failed_accept;

	ret = ft_accept_connection(eps[idx], eq);
	if (ret)
		goto failed_accept;

	return 0;

failed_accept:
	fi_reject(pep, fi->handle, NULL, 0);
	return ret;
}

static int setup_av_ep(int idx)
{
	int ret;

	fi_freeinfo(hints);
	hints = fi_dupinfo(fi);
	fi_freeinfo(fi);

	free(hints->src_addr);
	hints->src_addr = NULL;
	hints->src_addrlen = 0;

	ret = fi_getinfo(FT_FIVERSION, opts.src_addr, NULL, 0, hints, &fi);
	if (ret) {
		FT_PRINTERR("fi_getinfo", ret);
		return ret;
	}

	ret = fi_endpoint(domain, fi, &eps[idx], NULL);
	if (ret) {
		FT_PRINTERR("fi_endpoint", ret);
		return ret;
	}

	ret = ft_alloc_ep_res(fi, &txcqs[idx], &rxcqs[idx], NULL, NULL, NULL, &avs[idx]);
	if (ret)
		return ret;

	return 0;
}

static int enable_ep(int idx)
{
	int ret, av_bind_idx = idx, cq_bind_idx = idx;

	if (shared_cq)
		cq_bind_idx = 0;

	if (shared_av)
		av_bind_idx = 0;

	ret = ft_enable_ep(eps[idx], eq, avs[av_bind_idx], txcqs[cq_bind_idx], rxcqs[cq_bind_idx],
			   NULL, NULL, NULL);
	if (ret)
		return ret;

	ret = ft_init_av_addr(avs[av_bind_idx], eps[idx], &remote_fiaddr[idx]);
	if (ret)
		return ret;

	return 0;
}

static int run_test(void)
{
	int i, ret;

	if (hints->ep_attr->type == FI_EP_MSG) {
		ret = ft_init_fabric_cm();
		if (ret)
			return ret;
	} else {
		opts.av_size = num_eps + 1;
		ret = ft_init_fabric();
		if (ret)
			return ret;
	}

	ret = alloc_multi_ep_res();
	if (ret)
		return ret;

	/* Create additional endpoints. */
	printf("Creating %d EPs\n", num_eps);
	for (i = 0; i < num_eps; i++) {
		if (hints->ep_attr->type == FI_EP_MSG) {
			if (opts.dst_addr) {
				ret = setup_client_ep(i);
				if (ret)
					goto out;
			} else {
				ret = setup_server_ep(i);
				if (ret)
					goto out;
			}
		} else {
			ret = setup_av_ep(i);
			if (ret)
				goto out;
		}
	}

	ret = reg_mrs();
	if (ret)
		goto out;

	for (i = 0; i < num_eps; i++) {
		if (hints->ep_attr->type != FI_EP_MSG) {
			ret = enable_ep(i);
			if (ret)
				goto out;
		}
	}

	ret = do_sends();
	if (ret)
		goto out;

	ret = do_rma();
	if (ret)
		goto out;

	printf("Testing closing and re-registering all MRs and retesting\n");
	for (i = 0; i < num_eps; i++) {
		if (fi->domain_attr->mr_mode & FI_MR_RAW) {
			ret = fi_mr_unmap_key(domain, peer_iovs[i].key);
			if (ret)
				goto out;
		}

		FT_CLOSE_FID(send_mrs[i]);
		FT_CLOSE_FID(recv_mrs[i]);
	}

	ret = reg_mrs();
	if (ret)
		goto out;

	ret = do_sends();
	if (ret)
		goto out;

	ret = do_rma();
	if (ret)
		goto out;

	ret = ft_finalize_ep(ep);
out:
	free_ep_res();
	return ret;
}

int main(int argc, char **argv)
{
	int op;
	int ret = 0;

	opts = INIT_OPTS;
	opts.transfer_size = 256;
	opts.options |= FT_OPT_OOB_ADDR_EXCH;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt_long(argc, argv, "c:vhAQ" ADDR_OPTS INFO_OPTS,
				 long_opts, &lopt_idx)) != -1) {
		switch (op) {
		default:
			if (!ft_parse_long_opts(op, optarg))
				continue;
			ft_parse_addr_opts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints, &opts);
			break;
		case 'c':
			num_eps = atoi(optarg);
			break;
		case 'v':
			opts.options |= FT_OPT_VERIFY_DATA;
			break;
		case 'A':
			shared_av = true;
			break;
		case 'Q':
			shared_cq = true;
			break;
		case '?':
		case 'h':
			ft_usage(argv[0], "Multi endpoint test");
			FT_PRINT_OPTS_USAGE("-c <int>",
				"number of endpoints to create and test (def 3)");
			FT_PRINT_OPTS_USAGE("-v", "Enable data verification");
			FT_PRINT_OPTS_USAGE("-Q",
				"Share tx/rx cq among endpoints. \n"
				"By default each ep has its own tx/rx cq");
			FT_PRINT_OPTS_USAGE("-A",
				"Share the av among endpoints. \n"
				"By default each ep has its own av");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->caps = FI_MSG | FI_RMA;
	hints->mode = FI_CONTEXT;
	hints->domain_attr->mr_mode = opts.mr_mode;
	hints->addr_format = opts.address_format;

	ret = run_test();

	ft_free_res();
	return ft_exit_code(ret);
}
