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
#include <rdma/fi_trigger.h>
#include <shared.h>


struct fi_triggered_context triggered_ctx;

static char *welcome_text1 = "Hello1 from Client!";
static char *welcome_text2 = "Hello2 from Client!";

static int rma_write(void *src, size_t size,
		     void *context, uint64_t flags)
{
	int ret;
	struct fi_msg_rma msg;
	struct iovec msg_iov;
	struct fi_rma_iov rma_iov;
	void *desc = fi_mr_desc(mr);

	msg_iov.iov_base = src;
	msg_iov.iov_len = size;

	rma_iov.addr = 0;
	rma_iov.len = size;
	rma_iov.key = FT_MR_KEY;

	msg.msg_iov = &msg_iov;
	msg.desc = &desc;
	msg.iov_count = 1;
	msg.rma_iov_count = 1;
	msg.addr = remote_fi_addr;
	msg.rma_iov = &rma_iov;
	msg.context = context;

	ret = fi_writemsg(ep, &msg, flags);
 	if (ret){
 		FT_PRINTERR("fi_write", ret);
 		return ret;
	}
	return 0;
 }

static int rma_write_trigger(void *src, size_t size,
			     struct fid_cntr *cntr, size_t threshold)
{
	triggered_ctx.event_type = FI_TRIGGER_THRESHOLD;
	triggered_ctx.trigger.threshold.cntr = cntr;
	triggered_ctx.trigger.threshold.threshold = threshold;
	return rma_write(src, size, &triggered_ctx, FI_TRIGGER);
}

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

	ret = fi_domain(fabric, fi, &domain, NULL);
	if (ret) {
		FT_PRINTERR("fi_domain", ret);
		return ret;
	}

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
		sprintf(tx_buf, "%s%s", welcome_text1, welcome_text2);

		fprintf(stdout, "Triggered RMA write to server\n");
		ret = rma_write_trigger((char *) tx_buf + strlen(welcome_text1),
					strlen(welcome_text2), txcntr, 2);
		if (ret)
			goto out;

		fprintf(stdout, "RMA write to server\n");
		ret = rma_write(tx_buf, strlen(welcome_text1), &tx_ctx, 0);
		if (ret)
			goto out;

		ret = fi_cntr_wait(txcntr, 3, -1);
		if (ret < 0) {
			FT_PRINTERR("fi_cntr_wait", ret);
			goto out;
		}

		fprintf(stdout, "Received completion events for RMA write operations\n");
	} else {
		ret = fi_cntr_wait(rxcntr, 3, -1);
		if (ret < 0) {
			FT_PRINTERR("fi_cntr_wait", ret);
			goto out;
		}

		fprintf(stdout, "Received data from Client: %s\n", (char *) rx_buf);
		if (strncmp(rx_buf, welcome_text2, strlen(welcome_text2))) {
			fprintf(stderr, "*** Data corruption\n");
			ret = -1;
			goto out;
		} else {
			fprintf(stderr, "Data check OK\n");
			ret = 0;
		}
	}

out:
	return ret;
}

int main(int argc, char **argv)
{
	int op, ret;

	opts = INIT_OPTS;
	opts.options = FT_OPT_SIZE | FT_OPT_RX_CNTR | FT_OPT_TX_CNTR;
	opts.transfer_size = strlen(welcome_text1) + strlen(welcome_text2);

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
			ft_usage(argv[0], "A simple RDM client-sever Triggered RMA example.");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->domain_attr->mr_mode = FI_MR_SCALABLE;
	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_MSG | FI_RMA | FI_RMA_EVENT | FI_TRIGGER;
	hints->mode = FI_CONTEXT | FI_LOCAL_MR;

	ret = run_test();

	ft_free_res();
	return -ret;
}
