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
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <shared.h>

static void *buf;
static size_t buffer_size = 1024;
static int rx_depth = 512;

static char *dst_addr;
static char *port = "9228";
static void *remote_addr;
static size_t addrlen = 0;
static fi_addr_t remote_fi_addr;

static struct fi_info hints;
static struct fid_fabric *fab;
static struct fid_domain *dom;
static struct fid_ep *ep;
static struct fid_av *av;
static struct fid_cq *rcq, *scq;
static struct fid_mr *mr;
struct fi_context fi_ctx_send;
struct fi_context fi_ctx_recv;
struct fi_context fi_ctx_av;

static enum node_type {
	SERVER = 1,
	CLIENT
} type;

static void usage(char *name)
{
	printf("usage: %s\n", name);
	printf("\t-d destination_address\n");
	printf("\t-s or -c (server or client)\n");
	exit(1);
}

static void free_ep_res(void)
{
	fi_close(&av->fid);
	fi_close(&mr->fid);
	fi_close(&rcq->fid);
	fi_close(&scq->fid);
	free(buf);
}

static int alloc_ep_res(void)
{
	struct fi_cq_attr cq_attr;
	struct fi_av_attr av_attr;
	int ret;

	buf = malloc(buffer_size);
	if (!buf) {
		perror("malloc");
		return -1;
	}

	memset(&cq_attr, 0, sizeof cq_attr);
	cq_attr.format = FI_CQ_FORMAT_CONTEXT;
	cq_attr.wait_obj = FI_WAIT_NONE;
	cq_attr.size = rx_depth;

	/* Open completion queue for send completions */
	ret = fi_cq_open(dom, &cq_attr, &scq, NULL);
	if (ret) {
		FI_PRINTERR("fi_cq_open: scq", ret);
		goto err1;
	}

	/* Open completion queue for recv completions */
	ret = fi_cq_open(dom, &cq_attr, &rcq, NULL);
	if (ret) {
		FI_PRINTERR("fi_cq_open: rcq", ret);
		goto err2;
	}

	/* Register memory */
	ret = fi_mr_reg(dom, buf, buffer_size, 0, 0, 0, 0, &mr, NULL);
	if (ret) {
		FI_PRINTERR("fi_mr_reg", ret);
		goto err3;
	}

	memset(&av_attr, 0, sizeof av_attr);
	av_attr.type = FI_AV_MAP;
	av_attr.count = 1;
	av_attr.name = NULL;

	/* Open address vector (AV) for mapping address */
	ret = fi_av_open(dom, &av_attr, &av, NULL);
	if (ret) {
		FI_PRINTERR("fi_av_open", ret);
		 goto err4;
	 }

	return 0;

err4:
	fi_close(&mr->fid);
err3:
	fi_close(&rcq->fid);
err2:
	fi_close(&scq->fid);
err1:
	free(buf);
	return ret;
}

static int bind_ep_res(void)
{
	int ret;

	/* Bind Send CQ with endpoint to collect send completions */
	ret = fi_bind(&ep->fid, &scq->fid, FI_SEND);
	if (ret) {
		FI_PRINTERR("fi_bind: scq", ret);
		return ret;
	}

	/* Bind Recv CQ with endpoint to collect recv completions */
	ret = fi_bind(&ep->fid, &rcq->fid, FI_RECV);
	if (ret) {
		FI_PRINTERR("fi_bind: rcq", ret);
		return ret;
	}
	
	/* Bind AV with the endpoint to map addresses */
	ret = fi_bind(&ep->fid, &av->fid, 0);
	if (ret) {
		FI_PRINTERR("fi_bind: av", ret);
		return ret;
	}
	ret = fi_enable(ep);
	if (ret) {
			FI_PRINTERR("fi_enable", ret);
			return ret;
	 }

	return ret;
}

static int init_fabric(void)
{
	struct fi_info *fi;
	int ret;

	/* Get fabric info */
	ret = fi_getinfo(FI_VERSION(1, 0), dst_addr, port, 0, &hints, &fi);
	if (ret) {
		FI_PRINTERR("fi_getinfo", ret);
		goto err0;
	}
	
	addrlen = fi->dest_addrlen;
	remote_addr = malloc(addrlen);
	memcpy(remote_addr, fi->dest_addr, addrlen);

	/* Open fabric */
	ret = fi_fabric(fi->fabric_attr, &fab, NULL);
	if (ret) {
		FI_PRINTERR("fi_fabric", ret);
		goto err1;
	}

	/* Open domain */
	ret = fi_domain(fab, fi, &dom, NULL);
	if (ret) {
		FI_PRINTERR("fi_domain", ret);
		goto err2;
	}

	/* Open endpoint */
	ret = fi_endpoint(dom, fi, &ep, NULL);
	if (ret) {
		FI_PRINTERR("fi_endpoint", ret);
		goto err3;
	}

	/* Allocate endpoint resources */
	ret = alloc_ep_res();
	if (ret)
		goto err4;

	/* Bind EQs and AVs with endpoint */
	ret = bind_ep_res();
	if (ret)
		goto err5;
	
	/* Insert address to the AV and get the fabric address back */ 	
	ret = fi_av_insert(av, remote_addr, 1, &remote_fi_addr, 0, &fi_ctx_av);
	if (ret != 1) {
		FI_PRINTERR("fi_av_insert", ret);
		return ret;
	}

	fi_freeinfo(fi);
	return 0;

err5:
	free_ep_res();
err4:
	fi_close(&ep->fid);
err3:
	fi_close(&dom->fid);
err2:
	fi_close(&fab->fid);
err1:
	fi_freeinfo(fi);
err0:
	return ret;
}

static int send_recv()
{
	struct fi_cq_entry comp;
	int ret;

	if (type == CLIENT) {
		/* Client */
		fprintf(stdout, "Posting a send...\n");
		sprintf(buf, "Hello from Client!"); 
		ret = fi_send(ep, buf, sizeof("Hello from Client!"), fi_mr_desc(mr), 
				remote_fi_addr, &fi_ctx_send);
		if (ret) {
			FI_PRINTERR("fi_send", ret);
			return ret;
		}

		/* Read send queue */
		do {
			ret = fi_cq_read(scq, &comp, 1);
			if (ret < 0) {
				FI_PRINTERR("fi_cq_read: scq", ret);
				return ret;
			}
		} while (!ret);

		fprintf(stdout, "Send completion received\n");
	} else {
		/* Server */
		fprintf(stdout, "Posting a recv...\n");
		ret = fi_recv(ep, buf, buffer_size, fi_mr_desc(mr), 0, &fi_ctx_recv);
		if (ret) {
			FI_PRINTERR("fi_recv", ret);
			return ret;
		}

		/* Read recv queue */
		fprintf(stdout, "Waiting for client...\n");
		do {
			ret = fi_cq_read(rcq, &comp, 1);
			if (ret < 0) {
				FI_PRINTERR("fi_cq_read: rcq", ret);
				return ret;
			}
		} while (!ret);

		fprintf(stdout, "Received data from client: %s\n", (char *)buf);
	}

	return 0;
}

int main(int argc, char **argv)
{
	int op, ret;
	struct fi_domain_attr domain_hints;
	struct fi_ep_attr ep_hints;
	
	memset(&domain_hints, 0, sizeof(struct fi_domain_attr));
	memset(&ep_hints, 0, sizeof(struct fi_ep_attr));

	while ((op = getopt(argc, argv, "d:sc")) != -1) {
		switch (op) {
		case 'd':
			dst_addr = optarg;
			break;
		case 's':
			type = SERVER;
			break;
		case 'c':
			type = CLIENT;
			break;
		default:
			usage(argv[0]);
		}
	}
	/* Check if we got required args */
	if (optind != 4)
		usage(argv[0]);

	hints.domain_attr	= &domain_hints;
	hints.ep_attr		= &ep_hints;
	hints.ep_type		= FI_EP_RDM;
	hints.caps		= FI_MSG | FI_BUFFERED_RECV;
	hints.mode		= FI_CONTEXT;
	hints.addr_format	= FI_FORMAT_UNSPEC;

	/* Fabric initialization */
	ret = init_fabric();
	if(ret) {
		fprintf(stderr, "Problem in fabric initialization\n");
		return ret;
	}

	/* Exchange data */
	ret = send_recv();

	/* Tear down */
	fi_shutdown(ep, 0);
	fi_close(&ep->fid);
	free_ep_res();
	fi_close(&dom->fid);
	fi_close(&fab->fid);

	return ret;
}
