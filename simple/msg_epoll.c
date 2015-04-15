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
#include <sys/epoll.h>

#include <rdma/fabric.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <shared.h>

static struct cs_opts opts;
static void *buf;
static size_t buffer_size = 1024;
static int rx_depth = 512;

static struct fi_info *hints;
static struct fid_fabric *fab;
static struct fid_pep *pep;
static struct fid_domain *dom;
static struct fid_ep *ep;
static struct fid_eq *cmeq;
static struct fid_cq *rcq, *scq;
static int epfd;
static struct fid_mr *mr;

static int alloc_cm_res(void)
{
	struct fi_eq_attr cm_attr;
	int ret;

	memset(&cm_attr, 0, sizeof cm_attr);
	cm_attr.wait_obj = FI_WAIT_FD;

	/* Open EQ to receive CM events */
	ret = fi_eq_open(fab, &cm_attr, &cmeq, NULL);
	if (ret)
		FT_PRINTERR("fi_eq_open", ret);

	return ret;
}

static void free_ep_res(void)
{
	fi_close(&ep->fid);
	fi_close(&mr->fid);
	close(epfd);
	fi_close(&rcq->fid);
	fi_close(&scq->fid);
	free(buf);
}

static int alloc_ep_res(struct fi_info *fi)
{
	struct fi_cq_attr cq_attr;
	struct epoll_event event;
	int ret, fd;

	buf = malloc(buffer_size);
	if (!buf) {
		perror("malloc");
		return -1;
	}

	memset(&cq_attr, 0, sizeof cq_attr);
	cq_attr.format = FI_CQ_FORMAT_CONTEXT;
	cq_attr.wait_obj = FI_WAIT_FD;
	cq_attr.size = rx_depth;

	/* Open completion queue for send completions */
	ret = fi_cq_open(dom, &cq_attr, &scq, NULL);
	if (ret) {
		FT_PRINTERR("fi_cq_open", ret);
		goto err1;
	}

	/* Open completion queue for recv completions */
	ret = fi_cq_open(dom, &cq_attr, &rcq, NULL);
	if (ret) {
		FT_PRINTERR("fi_cq_open", ret);
		goto err2;
	}

	/* Create epoll set */
	epfd = epoll_create1(0);
	if (epfd < 0) {
		ret = -errno;
		FT_PRINTERR("epoll_create1", ret);
		goto err3;
	}

	/* Retrieve receive queue wait object */
	ret = fi_control (&rcq->fid, FI_GETWAIT, (void *) &fd);
	if (ret) {
		FT_PRINTERR("fi_control(FI_GETWAIT)", ret);
		goto err4;
	}

	/* Add receive queue wait object to epoll set */
	memset((void *)&event, 0, sizeof event);
	event.events = EPOLLIN;
	event.data.ptr = (void *)&rcq->fid;
	ret = epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &event);
	if (ret) {
		ret = -errno;
		FT_PRINTERR("epoll_ctl", ret);
		goto err4;
	}

	/* Retrieve send queue wait object */
	ret = fi_control (&scq->fid, FI_GETWAIT, (void *) &fd);
	if (ret) {
		FT_PRINTERR("fi_control(FI_GETWAIT)", ret);
		goto err4;
	}

	/* Add send queue wait object to epoll set */
	memset((void *)&event, 0, sizeof event);
	event.events = EPOLLIN;
	event.data.ptr = (void *)&scq->fid;
	ret = epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &event);
	if (ret) {
		ret = -errno;
		FT_PRINTERR("epoll_ctl", ret);
		goto err4;
	}

	/* Register memory */
	ret = fi_mr_reg(dom, buf, buffer_size, 0, 0, 0, 0, &mr, NULL);
	if (ret) {
		FT_PRINTERR("fi_mr_reg", ret);
		goto err4;
	}

	ret = fi_endpoint(dom, fi, &ep, NULL);
	if (ret) {
		FT_PRINTERR("fi_endpoint", ret);
		goto err5;
	}

	return 0;

err5:
	fi_close(&mr->fid);
err4:
	close(epfd);
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

	/* Bind EQ with endpoint */
	ret = fi_ep_bind(ep, &cmeq->fid, 0);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		return ret;
	}

	/* Bind Send CQ with endpoint to collect send completions */
	ret = fi_ep_bind(ep, &scq->fid, FI_SEND);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		return ret;
	}

	/* Bind Recv CQ with endpoint to collect recv completions */
	ret = fi_ep_bind(ep, &rcq->fid, FI_RECV);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		return ret;
	}

	return ret;
}

static int server_listen(void)
{
	struct fi_info *fi;
	int ret;

	/* Get fabric info */
	ret = fi_getinfo(FT_FIVERSION, NULL, opts.src_port, FI_SOURCE, hints, &fi);
	if (ret) {
		FT_PRINTERR("fi_getinfo", ret);
		return ret;
	}

	/* Open the fabric */
	ret = fi_fabric(fi->fabric_attr, &fab, NULL);
	if (ret) {
		FT_PRINTERR("fi_fabric", ret);
		goto err0;
	}

	/* Open a passive endpoint */
	ret = fi_passive_ep(fab, fi, &pep, NULL);
	if (ret) {
		FT_PRINTERR("fi_passive_ep", ret);
		goto err1;
	}

	/* Allocate connection management resources */
	ret = alloc_cm_res();
	if (ret)
		goto err2;

	/* Bind EQ to passive endpoint */
	ret = fi_pep_bind(pep, &cmeq->fid, 0);
	if (ret) {
		FT_PRINTERR("fi_pep_bind", ret);
		goto err3;
	}

	/* Listen for incoming connections */
	ret = fi_listen(pep);
	if (ret) {
		FT_PRINTERR("fi_listen", ret);
		goto err3;
	}

	fi_freeinfo(fi);
	return 0;
err3:
	fi_close(&cmeq->fid);
err2:
	fi_close(&pep->fid);
err1:
	fi_close(&fab->fid);
err0:
	fi_freeinfo(fi);
	return ret;
}

static int server_connect(void)
{
	struct fi_eq_cm_entry entry;
	uint32_t event;
	struct fi_info *info = NULL;
	ssize_t rd;
	int ret;

	/* Wait for connection request from client */
	rd = fi_eq_sread(cmeq, &event, &entry, sizeof entry, -1, 0);
	if (rd != sizeof entry) {
		FT_PRINTERR("fi_eq_sread", rd);
		return (int) rd;
	}

	info = entry.info;
	if (event != FI_CONNREQ) {
		fprintf(stderr, "Unexpected CM event %d\n", event);
		ret = -FI_EOTHER;
		goto err1;
	}

	ret = fi_domain(fab, info, &dom, NULL);
	if (ret) {
		FT_PRINTERR("fi_domain", ret);
		goto err1;
	}

	ret = alloc_ep_res(info);
	if (ret)
		 goto err1;

	ret = bind_ep_res();
	if (ret)
		goto err3;

	/* Accept the incoming connection. Also transitions endpoint to active state */
	ret = fi_accept(ep, NULL, 0);
	if (ret) {
		FT_PRINTERR("fi_accept", ret);
		goto err3;
	}

	/* Wait for the connection to be established */
	rd = fi_eq_sread(cmeq, &event, &entry, sizeof entry, -1, 0);
	if (rd != sizeof entry) {
		FT_PRINTERR("fi_eq_sread", rd);
		goto err3;
	}

	if (event != FI_CONNECTED || entry.fid != &ep->fid) {
		fprintf(stderr, "Unexpected CM event %d fid %p (ep %p)\n",
			event, entry.fid, ep);
		ret = -FI_EOTHER;
		goto err3;
	}

	fi_freeinfo(info);
	return 0;

err3:
	free_ep_res();
err1:
	fi_reject(pep, info->handle, NULL, 0);
	fi_freeinfo(info);
	return ret;
}

static int client_connect(void)
{
	struct fi_eq_cm_entry entry;
	uint32_t event;
	struct fi_info *fi;
	ssize_t rd;
	int ret;

	/* Get fabric info */
	ret = fi_getinfo(FT_FIVERSION, opts.dst_addr, opts.dst_port, 0, hints, &fi);
	if (ret) {
		FT_PRINTERR("fi_getinfo", ret);
		goto err0;
	}

	/* Open fabric */
	ret = fi_fabric(fi->fabric_attr, &fab, NULL);
	if (ret) {
		FT_PRINTERR("fi_fabric", ret);
		goto err1;
	}

	/* Open domain */
	ret = fi_domain(fab, fi, &dom, NULL);
	if (ret) {
		FT_PRINTERR("fi_domain", ret);
		goto err2;
	}

	ret = alloc_cm_res();
	if (ret)
		goto err4;

	ret = alloc_ep_res(fi);
	if (ret)
		goto err5;

	ret = bind_ep_res();
	if (ret)
		goto err6;

	/* Connect to server */
	ret = fi_connect(ep, fi->dest_addr, NULL, 0);
	if (ret) {
		FT_PRINTERR("fi_connect", ret);
		goto err6;
	}

	/* Wait for the connection to be established */
	rd = fi_eq_sread(cmeq, &event, &entry, sizeof entry, -1, 0);
	if (rd != sizeof entry) {
		FT_PRINTERR("fi_eq_sread", rd);
		return (int) rd;
	}

	if (event != FI_CONNECTED || entry.fid != &ep->fid) {
		fprintf(stderr, "Unexpected CM event %d fid %p (ep %p)\n",
			event, entry.fid, ep);
		ret = -FI_EOTHER;
		goto err6;
	}

	fi_freeinfo(fi);
	return 0;

err6:
	free_ep_res();
err5:
	fi_close(&cmeq->fid);
err4:
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
	struct epoll_event event;
	int ret;

	if (opts.dst_addr) {
		/* Client */
		fprintf(stdout, "Posting a send...\n");
		sprintf(buf, "Hello World!");
		ret = fi_send(ep, buf, sizeof("Hello World!"), fi_mr_desc(mr), 0, buf);
		if (ret) {
			FT_PRINTERR("fi_send", ret);
			return ret;
		}

		memset((void *)&event, 0, sizeof event);
		ret = TEMP_FAILURE_RETRY(epoll_wait(epfd, &event, 1, -1));
		if (ret < 0) {
			ret = -errno;
			FT_PRINTERR("epoll_wait", ret);
			return ret;
		}

		if (event.data.ptr != &scq->fid) {
			fprintf(stdout, "unexpected event!\n");
		}

		/* Read send queue */
		ret = fi_cq_sread(scq, &comp, 1, NULL, 0);
		if (ret < 0) {
			FT_PRINTERR("fi_cq_read", ret);
			return ret;
		}

		fprintf(stdout, "Send completion received\n");
	} else {
		/* Server */
		fprintf(stdout, "Posting a recv...\n");
		ret = fi_recv(ep, buf, buffer_size, fi_mr_desc(mr), 0, buf);
		if (ret) {
			FT_PRINTERR("fi_recv", ret);
			return ret;
		}

		fprintf(stdout, "Waiting for client...\n");

		memset((void *)&event, 0, sizeof event);
		ret = TEMP_FAILURE_RETRY(epoll_wait(epfd, &event, 1, -1));
		if (ret < 0) {
			ret = -errno;
			FT_PRINTERR("epoll_wait", ret);
			return ret;
		}

		if (event.data.ptr != &rcq->fid) {
			fprintf(stdout, "unexpected event!\n");
		}

		/* Read recv queue */
		ret = fi_cq_sread(rcq, &comp, 1, NULL, 0);
		if (ret < 0) {
			FT_PRINTERR("fi_cq_read", ret);
			return ret;
		}

		fprintf(stdout, "Received data from client: %s\n", (char *)buf);
	}

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
			ft_usage(argv[0], "A simple MSG client-sever example that demonstrates one possible usage of the underlying cq wait objects.");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->ep_attr->type	= FI_EP_MSG;
	hints->caps		= FI_MSG;
	hints->mode		= FI_LOCAL_MR | FI_PROV_MR_ATTR;
	hints->addr_format	= FI_SOCKADDR;

	/* Fabric and connection setup */
	if (!opts.dst_addr) {
		ret = server_listen();
		if (ret)
			return -ret;
	}

	ret = opts.dst_addr ? client_connect() : server_connect();
	if (ret) {
		return -ret;
	}

	/* Exchange data */
	ret = send_recv();

	fi_shutdown(ep, 0);
	free_ep_res();
	fi_close(&cmeq->fid);
	fi_close(&dom->fid);
	fi_close(&fab->fid);

	return ret;
}
