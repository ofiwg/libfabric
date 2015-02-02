/*
 * Copyright (c) 2013-2014 Intel Corporation.  All rights reserved.
 *
 * This software is available to you under the BSD license below:
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
#include <inttypes.h>

#include <rdma/fabric.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <shared.h>

static void *buf;
static size_t buffer_size;
static int rx_depth = 500;
static size_t cq_data_size;

static struct fi_info hints;
static char *dst_addr, *src_addr;
static char *port = "9228";

static struct fid_fabric *fab;
static struct fid_pep *pep;
static struct fid_domain *dom;
static struct fid_ep *ep;
static struct fid_eq *cmeq;
static struct fid_cq *rcq, *scq;
static struct fid_mr *mr;

void print_usage(char *name, char *desc)
{
	fprintf(stderr, "Usage:\n");
	fprintf(stderr, "  %s [OPTIONS]\t\tstart server\n", name);
	fprintf(stderr, "  %s [OPTIONS] <host>\tconnect to server \t\n", name);
	if (desc)
		fprintf(stderr, "\n%s\n", desc);
	
	fprintf(stderr, "\nOptions:\n");
	fprintf(stderr, "  -n <domain>\tdomain name\n");
	fprintf(stderr, "  -p <port>\tnon default port number\n");
	fprintf(stderr, "  -f <provider>\tspecific provider name eg IP, verbs\n");
	fprintf(stderr, "  -s <address>\tsource address\n");
	fprintf(stderr, "  -h\t\tdisplay this help output\n");

	return;
}

static void free_lres(void)
{
	fi_close(&cmeq->fid);
}

static int alloc_cm_res(void)
{
	struct fi_eq_attr cm_attr;
	int ret;

	memset(&cm_attr, 0, sizeof cm_attr);
	cm_attr.wait_obj = FI_WAIT_FD;
	ret = fi_eq_open(fab, &cm_attr, &cmeq, NULL);
	if (ret)
		FI_PRINTERR("fi_eq_open: cm", ret);

	return ret;
}

static void free_ep_res(void)
{
	fi_close(&mr->fid);
	fi_close(&rcq->fid);
	fi_close(&scq->fid);
	free(buf);
}

static int alloc_ep_res(struct fi_info *fi)
{
	struct fi_cq_attr cq_attr;
	int ret;

	buffer_size = test_size[TEST_CNT - 1].size;
	buf = malloc(buffer_size);
	if (!buf) {
		perror("malloc");
		return -1;
	}

	memset(&cq_attr, 0, sizeof cq_attr);
	cq_attr.format = FI_CQ_FORMAT_DATA;
	cq_attr.wait_obj = FI_WAIT_NONE;
	cq_attr.size = rx_depth;
	ret = fi_cq_open(dom, &cq_attr, &rcq, NULL);
	if (ret) {
		FI_PRINTERR("fi_cq_open: recv completion", ret);
		goto err1;
	}

	cq_attr.format = FI_CQ_FORMAT_CONTEXT;
	ret = fi_cq_open(dom, &cq_attr, &scq, NULL);
	if (ret) {
		FI_PRINTERR("fi_cq_open: send completion", ret);
		goto err2;
	}

	ret = fi_mr_reg(dom, buf, buffer_size, 0, 0, 0, 0, &mr, NULL);
	if (ret) {
		FI_PRINTERR("fi_mr_reg", ret);
		goto err3;
	}

	if (!cmeq) {
		ret = alloc_cm_res();
		if (ret)
			goto err4;
	}

	return 0;

err4:
	fi_close(&mr->fid);
err3:
	fi_close(&scq->fid);
err2:
	fi_close(&rcq->fid);
err1:
	free(buf);
	return ret;
}

static int bind_ep_res(void)
{
	int ret;

	ret = fi_ep_bind(ep, &cmeq->fid, 0);
	if (ret) {
		FI_PRINTERR("fi_ep_bind: cmeq", ret);
		return ret;
	}

	ret = fi_ep_bind(ep, &scq->fid, FI_SEND);
	if (ret) {
		FI_PRINTERR("fi_ep_bind: scq", ret);
		return ret;
	}

	ret = fi_ep_bind(ep, &rcq->fid, FI_RECV);
	if (ret) {
		FI_PRINTERR("fi_ep_bind: rcq", ret);
		return ret;
	}

	ret = fi_enable(ep);
	if (ret)
		return ret;

	ret = fi_recv(ep, buf, buffer_size, fi_mr_desc(mr), 0, buf);
	if (ret)
		FI_PRINTERR("fi_recv", ret);

	return ret;
}

static int server_listen(void)
{
	struct fi_info *fi;
	int ret;

	ret = fi_getinfo(FT_FIVERSION, src_addr, port, FI_SOURCE, &hints,
			&fi);
	if (ret) {
		FI_PRINTERR("fi_getinfo", ret);
		return ret;
	}

	cq_data_size = fi->domain_attr->cq_data_size;

	ret = fi_fabric(fi->fabric_attr, &fab, NULL);
	if (ret) {
		FI_PRINTERR("fi_fabric", ret);
		goto err0;
	}

	ret = fi_passive_ep(fab, fi, &pep, NULL);
	if (ret) {
		FI_PRINTERR("fi_passive_ep", ret);
		goto err1;
	}

	ret = alloc_cm_res();
	if (ret)
		goto err2;

	ret = fi_pep_bind(pep, &cmeq->fid, 0);
	if (ret) {
		FI_PRINTERR("fi_pep_bind", ret);
		goto err3;
	}

	ret = fi_listen(pep);
	if (ret) {
		FI_PRINTERR("fi_listen", ret);
		goto err3;
	}

	fi_freeinfo(fi);
	return 0;
err3:
	free_lres();
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

	rd = fi_eq_sread(cmeq, &event, &entry, sizeof entry, -1, 0);
	if (rd != sizeof entry) {
		FI_DEBUG("fi_eq_sread() %zd %s\n", rd, fi_strerror((int) -rd));
		return (int) rd;
	}

	if (event != FI_CONNREQ) {
		FI_DEBUG("Unexpected CM event %d\n", event);
		ret = -FI_EOTHER;
		goto err1;
	}

	info = entry.info;
	ret = fi_domain(fab, info, &dom, NULL);
	if (ret) {
		FI_PRINTERR("fi_domain", ret);
		goto err1;
	}

	ret = fi_endpoint(dom, info, &ep, NULL);
	if (ret) {
		FI_PRINTERR("fi_endpoint", ret);
		goto err1;
	}

	ret = alloc_ep_res(info);
	if (ret)
		 goto err2;

	ret = bind_ep_res();
	if (ret)
		goto err3;

	ret = fi_accept(ep, NULL, 0);
	if (ret) {
		FI_PRINTERR("fi_accept", ret);
		goto err3;
	}

	rd = fi_eq_sread(cmeq, &event, &entry, sizeof entry, -1, 0);
	if (rd != sizeof entry) {
		FI_DEBUG("fi_eq_sread() %zd %s\n", rd, fi_strerror((int) -rd));
		goto err3;
	}

	if (event != FI_CONNECTED || entry.fid != &ep->fid) {
		FI_DEBUG("Unexpected CM event %d fid %p (ep %p)\n",
			event, entry.fid, ep);
		ret = -FI_EOTHER;
		goto err3;
	}

	fi_freeinfo(info);
	return 0;

err3:
	free_ep_res();
err2:
	fi_close(&ep->fid);
err1:
	fi_reject(pep, info->connreq, NULL, 0);
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

	ret = ft_getsrcaddr(src_addr, port, &hints);
	if (ret)
		return ret;

	ret = fi_getinfo(FT_FIVERSION, dst_addr, port, 0, &hints, &fi);
	if (ret) {
		FI_PRINTERR("fi_getinfo", ret);
		goto err0;
	}

	cq_data_size = fi->domain_attr->cq_data_size;

	ret = fi_fabric(fi->fabric_attr, &fab, NULL);
	if (ret) {
		FI_PRINTERR("fi_fabric", ret);
		goto err1;
	}

	ret = fi_domain(fab, fi, &dom, NULL);
	if (ret) {
		FI_PRINTERR("fi_domain", ret);
		goto err2;
	}

	ret = fi_endpoint(dom, fi, &ep, NULL);
	if (ret) {
		FI_PRINTERR("fi_endpoint", ret);
		goto err3;
	}

	ret = alloc_ep_res(fi);
	if (ret)
		goto err4;

	ret = bind_ep_res();
	if (ret)
		goto err5;

	ret = fi_connect(ep, fi->dest_addr, NULL, 0);
	if (ret) {
		FI_PRINTERR("fi_connect", ret);
		goto err5;
	}

	rd = fi_eq_sread(cmeq, &event, &entry, sizeof entry, -1, 0);
	if (rd != sizeof entry) {
		FI_DEBUG("fi_eq_sread() %zd %s\n", rd, fi_strerror((int) -rd));
		return (int) rd;
	}

	if (event != FI_CONNECTED || entry.fid != &ep->fid) {
		FI_DEBUG("Unexpected CM event %d fid %p (ep %p)\n",
			event, entry.fid, ep);
		ret = -FI_EOTHER;
		goto err5;
	}

	if (hints.src_addr)
		free(hints.src_addr);
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
	if (hints.src_addr)
		free(hints.src_addr);
	return ret;
}

static int run_test()
{
	int ret;
	size_t size = 1000;
	uint64_t remote_cq_data;
	struct fi_cq_data_entry comp;
	
	/* Set remote_cq_data based on the cq_data_size we got from fi_getinfo */
	remote_cq_data = 0x0123456789abcdef & ((0x1ULL << (cq_data_size * 8)) - 1);

	if (dst_addr) {
		fprintf(stdout,
			"Posting send with immediate data: 0x%" PRIx64 "\n",
			remote_cq_data);
		ret = fi_senddata(ep, buf, size, fi_mr_desc(mr), remote_cq_data,
				0, buf);
		if (ret) {
			FI_PRINTERR("fi_send", ret);
			return ret;
		}

		wait_for_completion(scq, 1);
		fprintf(stdout, "Done\n");
	} else {
		ret = fi_recv(ep, buf, size, fi_mr_desc(mr), 0, buf);
		if (ret) {
			FI_PRINTERR("fi_recv", ret);
			return ret;
		}

		fprintf(stdout, "Waiting for immediate data from client\n");
		ret = fi_cq_sread(rcq, &comp, 1, NULL, -1);
		if (ret < 0) {
			if (ret == -FI_EAVAIL) {
				cq_readerr(rcq, "rcq");
			} else {
				FI_PRINTERR("fi_cq_read: rcq", ret);
			}
			return ret;
		}

		/* Verify completion data */
		if (comp.flags & FI_REMOTE_CQ_DATA) {
			if (comp.data == remote_cq_data)
				fprintf(stdout, "remote_cq_data: success\n");
			else
				fprintf(stdout, "remote_cq_data: failure\n");

			fprintf(stdout, "Expected data:0x%" PRIx64
				", Received data:0x%" PRIx64 "\n",
				remote_cq_data, comp.data);
		}
	}
	
	return 0;
}

static int run(void)
{
	int ret = 0;

	if (!dst_addr) {
		ret = server_listen();
		if (ret)
			return ret;
	}

	ret = dst_addr ? client_connect() : server_connect();
	if (ret) {
		return ret;
	}

	run_test();

	fi_shutdown(ep, 0);
	fi_close(&ep->fid);
	free_ep_res();
	if (!dst_addr)
		free_lres();
	fi_close(&dom->fid);
	fi_close(&fab->fid);
	return ret;
}

int main(int argc, char **argv)
{
	int op;
	while ((op = getopt(argc, argv, "p:s:h" INFO_OPTS)) != -1) {
		switch (op) {
		case 'p':
			port = optarg;
			break;
		case 's':
			src_addr = optarg;
			break;
		default:
			ft_parseinfo(op, optarg, &hints);
			break;
		case '?':
		case 'h':
			print_usage(argv[0], "A client-server example that tranfers immediate data.\n");
			return EXIT_FAILURE;
		}
	}
	
	if (optind < argc)
		dst_addr = argv[optind];

	hints.ep_type = FI_EP_MSG;
	hints.caps = FI_MSG | FI_REMOTE_CQ_DATA;
	hints.mode = FI_LOCAL_MR | FI_PROV_MR_ATTR;
	hints.addr_format = FI_SOCKADDR;

	return run();
}
