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
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AWV
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
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_errno.h>
#include <shared.h>


struct fi_rma_iov local, remote;

struct fi_context fi_ctx_write;
struct fi_context fi_ctx_read;

static char * welcome_text = "Hello from Client!";


static int alloc_ep_res(struct fi_info *fi)
{
	int ret;

	ret = ft_alloc_active_res(fi);
	if (ret)
		return ret;

	ret = fi_mr_reg(domain, buf, buf_size, FI_WRITE | FI_REMOTE_WRITE, 0,
			FT_MR_KEY, 0, &mr, NULL);
	if (ret) {
		FT_PRINTERR("fi_mr_reg", ret);
		return ret;
	}

	return 0;
}

static int init_fabric(void)
{
	char *node, *service;
	uint64_t flags = 0;
	int ret;

	ret = ft_read_addr_opts(&node, &service, hints, &flags, &opts);
	if (ret)
		return ret;

	ret = fi_getinfo(FT_FIVERSION, node, service, flags, hints, &fi);
	if (ret) {
		FT_PRINTERR("fi_getinfo", ret);
		return ret;
	}

	ret = ft_open_fabric_res();
	if (ret)
		return ret;

	ret = alloc_ep_res(fi);
	if (ret)
		return ret;

	ret = ft_init_ep();
	if (ret)
		return ret;

	return 0;
}

static int run_test(void)
{
	int ret = 0;

	ret = init_fabric();
	if (ret)
		return ret;

	ret = ft_init_av();
	if (ret)
		return ret;

	if (opts.dst_addr) {
		fprintf(stdout, "RMA write to server\n");
		sprintf(buf, "%s", welcome_text);
		ret = fi_write(ep, buf, sizeof(char *) * strlen(buf), fi_mr_desc(mr),
				remote_fi_addr, 0, FT_MR_KEY, &fi_ctx_write);
		if (ret)
			return ret;

		ret = ft_get_tx_comp(++tx_seq);
		if (ret)
			return ret;

		fprintf(stdout, "Received a completion event for RMA write\n");
	} else {
		ret = ft_get_rx_comp(rx_seq);
		if (ret)
			return ret;

		fprintf(stdout, "Received data from Client: %s\n", (char *) rx_buf);
	}

	/* TODO: need support for finalize operation to sync test */
	return 0;
}

int main(int argc, char **argv)
{
	int op, ret;

	opts = INIT_OPTS;
	opts.options = FT_OPT_SIZE | FT_OPT_RX_CNTR | FT_OPT_TX_CNTR;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt(argc, argv, "h" ADDR_OPTS INFO_OPTS)) != -1) {
		switch (op) {
		default:
			ft_parse_addr_opts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints);
			break;
		case '?':
		case 'h':
			ft_usage(argv[0], "A simple RDM client-sever RMA example.");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->domain_attr->mr_mode = FI_MR_SCALABLE;
	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_MSG | FI_RMA | FI_RMA_EVENT;
	hints->mode = FI_CONTEXT | FI_LOCAL_MR;

	ret = run_test();

	ft_free_res();
	return -ret;
}
