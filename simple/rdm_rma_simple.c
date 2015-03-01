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

static void *buf;
static size_t buffer_size;
struct fi_rma_iov local, remote;

static struct fi_info *fi, *hints;
static char *dst_addr;
static char *port = "9228";

static struct fid_fabric *fab;
static struct fid_domain *dom;
static struct fid_ep *ep;
static struct fid_cntr *rcntr, *scntr;
static struct fid_av *av;
static struct fid_mr *mr;
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

void print_usage(char *name, char *desc)
{
	fprintf(stderr, "Usage:\n");
	fprintf(stderr, "  %s [OPTIONS]\t\tstart server\n", name);
	fprintf(stderr, "  %s [OPTIONS] <host>\tconnect to server\n", name);

	if (desc)			
		fprintf(stderr, "\n%s\n", desc);

	fprintf(stderr, "\nOptions:\n");
	fprintf(stderr, "  -f <provider>\tspecific provider name eg IP, verbs\n");
	fprintf(stderr, "  -h\t\tdisplay this help output\n");
									
	return;
}

static int write_data(size_t size)
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

static void free_ep_res(void)
{
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
	uint64_t flags = 0;
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

	ret = fi_cntr_open(dom, &cntr_attr, &scntr, NULL);
	if (ret) {
		FT_PRINTERR("fi_cntr_open", ret);
		goto err1;
	}

	ret = fi_cntr_open(dom, &cntr_attr, &rcntr, NULL);
	if (ret) {
		FT_PRINTERR("fi_cntr_open", ret);
		goto err2;
	}
	
	/* Set FI_MR_KEY to associate the memory region with the specified key
	 * Set FI_MR_OFFSET to use specified offset as the base address */
	flags = FI_MR_KEY | FI_MR_OFFSET;
	ret = fi_mr_reg(dom, buf, buffer_size, FI_REMOTE_WRITE, 0, 
			user_defined_key, flags, &mr, NULL);
	if (ret) {
		FT_PRINTERR("fi_mr_reg", ret);
		goto err3;
	}

	memset(&av_attr, 0, sizeof av_attr);
	av_attr.type = fi->domain_attr->av_type ?
			fi->domain_attr->av_type : FI_AV_MAP;
	av_attr.count = 1;
	av_attr.name = NULL;

	ret = fi_av_open(dom, &av_attr, &av, NULL);
	if (ret) {
		FT_PRINTERR("fi_av_open", ret);
		goto err4;
	}

	return 0;

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
	char *node;
	uint64_t flags = 0;
	int ret;

	if (dst_addr) {
		node = dst_addr;
	} else {
		node = NULL;
		flags = FI_SOURCE;
	}

	ret = fi_getinfo(FT_FIVERSION, node, port, flags, hints, &fi);
	if (ret) {
		FT_PRINTERR("fi_getinfo", ret);
		return ret;
	}

	if (dst_addr) {
		addrlen = fi->dest_addrlen;
		remote_addr = malloc(addrlen);
		memcpy(remote_addr, fi->dest_addr, addrlen);
	}

	ret = fi_fabric(fi->fabric_attr, &fab, NULL);
	if (ret) {
		FT_PRINTERR("fi_fabric", ret);
		goto err0;
	}

	ret = fi_domain(fab, fi, &dom, NULL);
	if (ret) {
		FT_PRINTERR("fi_domain", ret);
		goto err1;
	}

	ret = fi_endpoint(dom, fi, &ep, NULL);
	if (ret) {
		FT_PRINTERR("fi_endpoint", ret);
		goto err2;
	}

	ret = alloc_ep_res(fi);
	if (ret)
		goto err3;

	ret = bind_ep_res();
	if (ret)
		goto err4;
	
	if(dst_addr) {
		ret = fi_av_insert(av, remote_addr, 1, &remote_fi_addr, 0, 
				&fi_ctx_av);
		if (ret != 1) {
			FT_PRINTERR("fi_av_insert", ret);
			return ret;
		}
	}

	return 0;

err4:
	free_ep_res();
err3:
	fi_close(&ep->fid);
err2:
	fi_close(&dom->fid);
err1:
	fi_close(&fab->fid);
err0:
	return ret;
}

static int run_test(void)
{
	int ret = 0;

	ret = init_fabric();
	if (ret)
		return ret;

	if (dst_addr) {	
		/* Execute RMA write operation from Client */
		fprintf(stdout, "RMA write to server\n");
		sprintf(buf, "%s", welcome_text);
		ret = write_data(sizeof(char *) * strlen(buf));
		if (ret)
			return ret;
	
		ret = fi_cntr_wait(scntr, 1, -1);
		if (ret < 0) {
			FT_PRINTERR("fi_cntr_wait", ret);
			return ret;
		}

		fprintf(stdout, "Received a completion event for RMA write\n");
	} else {	
		/* Server waits for message from Client */
		ret = fi_cntr_wait(rcntr, 1, -1);
		if (ret < 0) {
			FT_PRINTERR("fi_cntr_wait", ret);
			return ret;
		}

		fprintf(stdout, "Received data from Client: %s\n", (char *)buf);
	}

	fi_close(&ep->fid);
	free_ep_res();
	fi_close(&dom->fid);
	fi_close(&fab->fid);

	return 0;
}

int main(int argc, char **argv)
{
	int op, ret;
		
	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt(argc, argv, "f:h")) != -1) {
		switch (op) {
		case 'f':
			hints->fabric_attr->prov_name = optarg;
			break;
		case '?':
		case 'h':
			print_usage(argv[0], "A simple RDM client-sever RMA example.");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		dst_addr = argv[optind];

	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_MSG | FI_RMA | FI_REMOTE_COMPLETE;
	// FI_PROV_MR_ATTR flag is not set
	hints->mode = FI_CONTEXT;

	ret = run_test();
	fi_freeinfo(hints);
	fi_freeinfo(fi);
	return ret;
}
