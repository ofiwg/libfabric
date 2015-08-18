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


static int rx_depth = 500;


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
	cq_attr.wait_obj = FI_WAIT_UNSPEC;
	cq_attr.size = rx_depth;
	ret = fi_cq_open(domain, &cq_attr, &rxcq, NULL);
	if (ret) {
		FT_PRINTERR("fi_cq_open", ret);
		return ret;
	}

	cq_attr.format = FI_CQ_FORMAT_CONTEXT;
	ret = fi_cq_open(domain, &cq_attr, &txcq, NULL);
	if (ret) {
		FT_PRINTERR("fi_cq_open", ret);
		return ret;
	}

	ret = fi_mr_reg(domain, buf, buffer_size, 0, 0, 0, 0, &mr, NULL);
	if (ret) {
		FT_PRINTERR("fi_mr_reg", ret);
		return ret;
	}

	ret = fi_endpoint(domain, fi, &ep, NULL);
	if (ret) {
		FT_PRINTERR("fi_endpoint", ret);
		return ret;
	}

	return 0;
}

static int server_connect(void)
{
	struct fi_eq_cm_entry entry;
	uint32_t event;
	struct fi_info *info = NULL;
	ssize_t rd;
	int ret;

	rd = fi_eq_sread(eq, &event, &entry, sizeof entry, -1, 0);
	if (rd != sizeof entry) {
		FT_PROCESS_EQ_ERR(rd, eq, "fi_eq_sread", "listen");
		return (int) rd;
	}

	info = entry.info;
	if (event != FI_CONNREQ) {
		fprintf(stderr, "Unexpected CM event %d\n", event);
		ret = -FI_EOTHER;
		goto err;
	}

	ret = fi_domain(fabric, info, &domain, NULL);
	if (ret) {
		FT_PRINTERR("fi_domain", ret);
		goto err;
	}

	ret = alloc_ep_res(info);
	if (ret)
		 goto err;

	ret = ft_init_ep(buf);
	if (ret)
		goto err;

	ret = fi_accept(ep, NULL, 0);
	if (ret) {
		FT_PRINTERR("fi_accept", ret);
		goto err;
	}

	rd = fi_eq_sread(eq, &event, &entry, sizeof entry, -1, 0);
	if (rd != sizeof entry) {
		FT_PROCESS_EQ_ERR(rd, eq, "fi_eq_sread", "accept");
		ret = (int) rd;
		goto err;
	}

	if (event != FI_CONNECTED || entry.fid != &ep->fid) {
		fprintf(stderr, "Unexpected CM event %d fid %p (ep %p)\n",
			event, entry.fid, ep);
		ret = -FI_EOTHER;
		goto err;
	}

	fi_freeinfo(info);
	return 0;

err:
	fi_reject(pep, info->handle, NULL, 0);
	fi_freeinfo(info);
	return ret;
}

static int client_connect(void)
{
	struct fi_eq_cm_entry entry;
	uint32_t event;
	ssize_t rd;
	int ret;

	ret = ft_getsrcaddr(opts.src_addr, opts.src_port, hints);
	if (ret)
		return ret;

	ret = fi_getinfo(FT_FIVERSION, opts.dst_addr, opts.dst_port, 0, hints, &fi);
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

	ret = ft_init_ep(buf);
	if (ret)
		return ret;

	ret = fi_connect(ep, fi->dest_addr, NULL, 0);
	if (ret) {
		FT_PRINTERR("fi_connect", ret);
		return ret;
	}

	rd = fi_eq_sread(eq, &event, &entry, sizeof entry, -1, 0);
	if (rd != sizeof entry) {
		FT_PROCESS_EQ_ERR(rd, eq, "fi_eq_sread", "connect");
		return (int) rd;
	}

	if (event != FI_CONNECTED || entry.fid != &ep->fid) {
		fprintf(stderr, "Unexpected CM event %d fid %p (ep %p)\n",
			event, entry.fid, ep);
		return -FI_EOTHER;
	}

	return 0;
}

static int run_test()
{
	int ret;
	size_t size = 1000;
	uint64_t remote_cq_data;
	struct fi_cq_data_entry comp;

	/* Set remote_cq_data based on the cq_data_size we got from fi_getinfo */
	remote_cq_data = 0x0123456789abcdef &
			((0x1ULL << (fi->domain_attr->cq_data_size * 8)) - 1);

	if (opts.dst_addr) {
		fprintf(stdout,
			"Posting send with immediate data: 0x%" PRIx64 "\n",
			remote_cq_data);
		ret = fi_senddata(ep, buf, size, fi_mr_desc(mr), remote_cq_data,
				0, buf);
		if (ret) {
			FT_PRINTERR("fi_send", ret);
			return ret;
		}

		wait_for_completion(txcq, 1);
		fprintf(stdout, "Done\n");
	} else {
		fprintf(stdout, "Waiting for immediate data from client\n");
		ret = fi_cq_sread(rxcq, &comp, 1, NULL, -1);
		if (ret < 0) {
			if (ret == -FI_EAVAIL) {
				cq_readerr(rxcq, "rxcq");
			} else {
				FT_PRINTERR("fi_cq_sread", ret);
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

	if (!opts.dst_addr) {
		ret = ft_start_server();
		if (ret)
			return ret;
	}

	ret = opts.dst_addr ? client_connect() : server_connect();
	if (ret) {
		return ret;
	}

	run_test();

	fi_shutdown(ep, 0);
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
			ft_usage(argv[0], "A client-server example that tranfers CQ data.\n");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->domain_attr->cq_data_size = 4;  /* required minimum */

	hints->ep_attr->type = FI_EP_MSG;
	hints->caps = FI_MSG;
	hints->mode = FI_LOCAL_MR;
	hints->addr_format = FI_SOCKADDR;

	ret = run();

	ft_free_res();
	return -ret;
}
