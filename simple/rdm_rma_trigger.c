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

static struct cs_opts opts;
static void *buf;
static size_t buffer_size;

static struct fid_cntr *rcntr, *scntr;
static void *remote_addr;
static size_t addrlen = 0;
static fi_addr_t remote_fi_addr;
struct fi_context fi_ctx_write;

static uint64_t user_defined_key = 45678;
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
	rma_iov.key = user_defined_key;

	msg.msg_iov = &msg_iov;
	msg.desc = &desc;
	msg.iov_count = 1;
	msg.rma_iov_count = 1;
	msg.addr = remote_fi_addr;
	msg.rma_iov = &rma_iov;
	msg.context = context;

 	/* Using specified base address and MR key for RMA write */
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
	struct fi_triggered_context triggered_ctx;
	triggered_ctx.event_type = FI_TRIGGER_THRESHOLD;
	triggered_ctx.trigger.threshold.cntr = cntr;
	triggered_ctx.trigger.threshold.threshold = threshold;
	return rma_write(src, size, &triggered_ctx, FI_TRIGGER);
}


static void free_ep_res(void)
{
	fi_close(&ep->fid);
	fi_close(&av->fid);
	fi_close(&mr->fid);
	fi_close(&rcntr->fid);
	fi_close(&scntr->fid);
	free(buf);
}

static int alloc_ep_res(struct fi_info *fi)
{
	struct fi_cntr_attr cntr_attr;
	struct fi_av_attr av_attr;
	int ret;

	buffer_size = strlen(welcome_text1) + strlen(welcome_text2);
	buf = malloc(buffer_size);
	if (!buf) {
		perror("malloc");
		return -1;
	}

	memset(&cntr_attr, 0, sizeof cntr_attr);
	cntr_attr.events = FI_CNTR_EVENTS_COMP;

	ret = fi_cntr_open(domain, &cntr_attr, &scntr, NULL);
	if (ret) {
		FT_PRINTERR("fi_cntr_open", ret);
		goto err1;
	}

	ret = fi_cntr_open(domain, &cntr_attr, &rcntr, NULL);
	if (ret) {
		FT_PRINTERR("fi_cntr_open", ret);
		goto err2;
	}

	ret = fi_mr_reg(domain, buf, buffer_size, FI_WRITE | FI_REMOTE_WRITE, 0,
			user_defined_key, 0, &mr, NULL);
	if (ret) {
		FT_PRINTERR("fi_mr_reg", ret);
		goto err3;
	}

	memset(&av_attr, 0, sizeof av_attr);
	av_attr.type = fi->domain_attr->av_type ?
			fi->domain_attr->av_type : FI_AV_MAP;
	av_attr.count = 1;
	av_attr.name = NULL;

	ret = fi_av_open(domain, &av_attr, &av, NULL);
	if (ret) {
		FT_PRINTERR("fi_av_open", ret);
		goto err4;
	}

	ret = fi_endpoint(domain, fi, &ep, NULL);
	if (ret) {
		FT_PRINTERR("fi_endpoint", ret);
		goto err5;
	}

	return 0;

err5:
	fi_close(&av->fid);
err4:
	fi_close(&mr->fid);
err3:
	fi_close(&rcntr->fid);
err2:
	fi_close(&scntr->fid);
err1:
	free(buf);
	return ret;
}

static int bind_ep_res(void)
{
	int ret;

	ret = fi_ep_bind(ep, &scntr->fid, FI_WRITE);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		return ret;
	}

	/* Use FI_REMOTE_WRITE flag so that remote side can get completion event
	 *  for RMA write operation */
	ret = fi_ep_bind(ep, &rcntr->fid, FI_REMOTE_WRITE);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		return ret;
	}

	ret = fi_ep_bind(ep, &av->fid, 0);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		return ret;
	}

	ret = fi_enable(ep);
	if (ret) {
		FT_PRINTERR("fi_enable", ret);
		return ret;
	}

	return ret;
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
		goto err0;
	}

	ret = fi_domain(fabric, fi, &domain, NULL);
	if (ret) {
		FT_PRINTERR("fi_domain", ret);
		goto err1;
	}

	ret = alloc_ep_res(fi);
	if (ret)
		goto err3;

	ret = bind_ep_res();
	if (ret)
		goto err4;

	if(opts.dst_addr) {
		ret = fi_av_insert(av, remote_addr, 1, &remote_fi_addr, 0, NULL);
		if (ret != 1) {
			FT_PRINTERR("fi_av_insert", ret);
			return ret;
		}
	}

	return 0;

err4:
	free_ep_res();
err3:
	fi_close(&domain->fid);
err1:
	fi_close(&fabric->fid);
err0:
	return ret;
}

static int run_test(void)
{
	int ret = 0;
	char *ptr1, *ptr2;

	ret = init_fabric();
	if (ret)
		return ret;

	if (opts.dst_addr) {
		ptr1 = buf;
		sprintf(ptr1, "%s", welcome_text1);
		ptr2 = ptr1 + strlen(welcome_text1);
		sprintf(ptr2, "%s", welcome_text2);

		fprintf(stdout, "Triggered RMA write to server\n");
		/* Post triggered RMA write operation */
		ret = rma_write_trigger(ptr2, strlen(welcome_text2), scntr, 1);
		if (ret)
			goto out;

		/* Execute RMA write operation from Client */
		fprintf(stdout, "RMA write to server\n");
		ret = rma_write(ptr1, strlen(welcome_text1), NULL, 0);
		if (ret)
			goto out;

		ret = fi_cntr_wait(scntr, 2, -1);
		if (ret < 0) {
			FT_PRINTERR("fi_cntr_wait", ret);
			goto out;
		}

		fprintf(stdout, "Received completion events for RMA write operations\n");
	} else {
		/* Server waits for message from Client */
		ret = fi_cntr_wait(rcntr, 2, -1);
		if (ret < 0) {
			FT_PRINTERR("fi_cntr_wait", ret);
			goto out;
		}

		fprintf(stdout, "Received data from Client: %s\n", (char *)buf);
		if (strncmp(buf, welcome_text2, strlen(welcome_text2))) {
			fprintf(stderr, "*** Data corruption\n");
			ret = -1;
			goto out;
		} else {
			fprintf(stderr, "Data check OK\n");
			ret = 0;
		}
	}

out:
	free_ep_res();
	fi_close(&domain->fid);
	fi_close(&fabric->fid);
	return ret;
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
	fi_freeinfo(hints);
	fi_freeinfo(fi);
	return -ret;
}
