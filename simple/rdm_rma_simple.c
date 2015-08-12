/*
 * Copyright (c) 2013-2014 Intel Corporation.  All rights reserved.
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

static void *remote_addr;
static size_t addrlen = 0;
static fi_addr_t remote_fi_addr;
struct fi_context fi_ctx_send;
struct fi_context fi_ctx_recv;
struct fi_context fi_ctx_write;
struct fi_context fi_ctx_read;
struct fi_context fi_ctx_av;

static uint64_t user_defined_key = 45678;
static char * welcome_text = "Hello from Client!";

static int rma_write(size_t size)
{
	int ret;

	/* Using specified base address and MR key for RMA write */
	ret = fi_write(ep, buf, size, fi_mr_desc(mr), remote_fi_addr, 0,
			user_defined_key, &fi_ctx_write);
	if (ret){
		FT_PRINTERR("fi_write", ret);
		return ret;
	}
	return 0;
}

static int alloc_ep_res(struct fi_info *fi)
{
	struct fi_cntr_attr cntr_attr;
	struct fi_av_attr av_attr;
	int ret;

	buffer_size = MAX(sizeof(char *) * strlen(welcome_text),
			sizeof(uint64_t));
	buf = malloc(buffer_size);
	if (!buf) {
		perror("malloc");
		return -1;
	}

	memset(&cntr_attr, 0, sizeof cntr_attr);
	cntr_attr.events = FI_CNTR_EVENTS_COMP;

	ret = fi_cntr_open(domain, &cntr_attr, &txcntr, NULL);
	if (ret) {
		FT_PRINTERR("fi_cntr_open", ret);
		return ret;
	}

	ret = fi_cntr_open(domain, &cntr_attr, &rxcntr, NULL);
	if (ret) {
		FT_PRINTERR("fi_cntr_open", ret);
		return ret;
	}

	ret = fi_mr_reg(domain, buf, buffer_size, FI_WRITE | FI_REMOTE_WRITE, 0,
			user_defined_key, 0, &mr, NULL);
	if (ret) {
		FT_PRINTERR("fi_mr_reg", ret);
		return ret;
	}

	memset(&av_attr, 0, sizeof av_attr);
	av_attr.type = fi->domain_attr->av_type ?
			fi->domain_attr->av_type : FI_AV_MAP;
	av_attr.count = 1;
	av_attr.name = NULL;

	ret = fi_av_open(domain, &av_attr, &av, NULL);
	if (ret) {
		FT_PRINTERR("fi_av_open", ret);
		return ret;
	}

	ret = fi_endpoint(domain, fi, &ep, NULL);
	if (ret) {
		FT_PRINTERR("fi_endpoint", ret);
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

	if (opts.dst_addr) {
		addrlen = fi->dest_addrlen;
		remote_addr = malloc(addrlen);
		memcpy(remote_addr, fi->dest_addr, addrlen);
	}

	ret = fi_fabric(fi->fabric_attr, &fabric, NULL);
	if (ret) {
		FT_PRINTERR("fi_fabric", ret);
		return ret;
	}

	ret = fi_domain(fabric, fi, &domain, NULL);
	if (ret) {
		FT_PRINTERR("fi_domain", ret);
		return ret;
	}

	ret = alloc_ep_res(fi);
	if (ret)
		return ret;

	ret = ft_init_ep(NULL);
	if (ret)
		return ret;

	if(opts.dst_addr) {
		ret = fi_av_insert(av, remote_addr, 1, &remote_fi_addr, 0,
				&fi_ctx_av);
		if (ret != 1) {
			FT_PRINTERR("fi_av_insert", ret);
			return ret;
		}
	}

	return 0;
}

static int run_test(void)
{
	int ret = 0;

	ret = init_fabric();
	if (ret)
		return ret;

	if (opts.dst_addr) {
		/* Execute RMA write operation from Client */
		fprintf(stdout, "RMA write to server\n");
		sprintf(buf, "%s", welcome_text);
		ret = rma_write(sizeof(char *) * strlen(buf));
		if (ret)
			return ret;

		ret = fi_cntr_wait(txcntr, 1, -1);
		if (ret < 0) {
			FT_PRINTERR("fi_cntr_wait", ret);
			return ret;
		}

		fprintf(stdout, "Received a completion event for RMA write\n");
	} else {
		/* Server waits for message from Client */
		ret = fi_cntr_wait(rxcntr, 1, -1);
		if (ret < 0) {
			FT_PRINTERR("fi_cntr_wait", ret);
			return ret;
		}

		fprintf(stdout, "Received data from Client: %s\n", (char *)buf);
	}

	/* TODO: need support for finalize operation to sync test */
	return 0;
}

int main(int argc, char **argv)
{
	int op, ret;

	opts = INIT_OPTS;
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
