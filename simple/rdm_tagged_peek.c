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
#include <inttypes.h>

#include <rdma/fabric.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_tagged.h>
#include <shared.h>

static struct cs_opts opts;
static void *buf;
static size_t buffer_size = 1024;
static int rx_depth = 512;

static struct fi_info *fi, *hints;

static struct fid_fabric *fab;
static struct fid_domain *dom;
static struct fid_ep *ep;
static struct fid_cq *rcq, *scq;
static struct fid_av *av;
static struct fid_mr *mr;
static void *local_addr, *remote_addr;
static size_t addrlen = 0;
static fi_addr_t remote_fi_addr;
struct fi_context fi_ctx_send;
struct fi_context fi_ctx_recv;
struct fi_context fi_ctx_av;
struct fi_context fi_ctx_search;

static uint64_t tag_data = 1;
static uint64_t tag_control = 0x12345678;
static uint64_t tag_param = 0x87654321;

int wait_for_tagged_completion(struct fid_cq *cq, int num_completions)
{
	int ret;
	struct fi_cq_tagged_entry comp;

	while (num_completions > 0) {
		ret = fi_cq_read(cq, &comp, 1);
		if (ret > 0) {
			num_completions--;
		} else if (ret < 0 && ret != -FI_EAGAIN) {
			FT_PRINTERR("fi_cq_read", ret);
			return ret;
		}
	}
	return 0;
}

static int send_msg(int size, uint64_t tag)
{
	int ret;

	ret = fi_tsend(ep, buf, (size_t) size, fi_mr_desc(mr), remote_fi_addr,
			tag, &fi_ctx_send);
	if (ret)
		FT_PRINTERR("fi_tsend", ret);

	ret = wait_for_tagged_completion(scq, 1);

	return ret;
}

static int recv_msg(uint64_t tag)
{
	int ret;

	ret = fi_trecv(ep, buf, buffer_size, fi_mr_desc(mr), remote_fi_addr,
			tag, 0, &fi_ctx_recv);
	if (ret)
		FT_PRINTERR("fi_trecv", ret);

	ret = wait_for_tagged_completion(rcq, 1);
	return ret;
}

static int post_recv(uint64_t tag)
{
	int ret;

	ret = fi_trecv(ep, buf, buffer_size, fi_mr_desc(mr), remote_fi_addr,
			tag, 0, &fi_ctx_recv);
	if (ret)
		FT_PRINTERR("fi_trecv", ret);

	return ret;
}

static int sync_test(void)
{
	int ret;

	ret = opts.dst_addr ? send_msg(16, tag_control) : recv_msg(tag_control);
	if (ret)
		return ret;

	ret = opts.dst_addr ? recv_msg(tag_control) : send_msg(16, tag_control);

	return ret;
}

static void free_ep_res(void)
{
	fi_close(&ep->fid);
	fi_close(&av->fid);
	fi_close(&mr->fid);
	fi_close(&rcq->fid);
	fi_close(&scq->fid);
	free(buf);
}

static int alloc_ep_res(struct fi_info *fi)
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
	ret = fi_cq_open(dom, &cq_attr, &scq, NULL);
	if (ret) {
		FT_PRINTERR("fi_cq_open", ret);
		goto err1;
	}

	ret = fi_cq_open(dom, &cq_attr, &rcq, NULL);
	if (ret) {
		FT_PRINTERR("fi_cq_open", ret);
		goto err2;
	}

	ret = fi_mr_reg(dom, buf, buffer_size, 0, 0, 0, 0, &mr, NULL);
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

	ret = fi_endpoint(dom, fi, &ep, NULL);
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

	ret = fi_ep_bind(ep, &scq->fid, FI_SEND);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		return ret;
	}

	ret = fi_ep_bind(ep, &rcq->fid, FI_RECV);
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

	/* Post a recv buffer for synchronization */
	ret = fi_recv(ep, buf, buffer_size, fi_mr_desc(mr), 0, &fi_ctx_recv);
	if (ret) {
		FT_PRINTERR("fi_recv", ret);
		return ret;
	}

	return ret;
}

static int init_fabric(void)
{
	uint64_t flags = 0;
	char *node, *service;
	int ret;

	ret = ft_read_addr_opts(&node, &service, hints, &flags, &opts);
	if (ret)
		return ret;

	ret = fi_getinfo(FT_FIVERSION, node, service, flags, hints, &fi);
	if (ret) {
		FT_PRINTERR("fi_getinfo", ret);
		return ret;
	}

	// we use provider MR attributes and direct address (no offsets)
	// for RMA calls
	if (!(fi->mode & FI_PROV_MR_ATTR))
		fi->mode |= FI_PROV_MR_ATTR;

	// get remote address
	if (opts.dst_addr) {
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

	ret = alloc_ep_res(fi);
	if (ret)
		goto err3;

	ret = bind_ep_res();
	if (ret)
		goto err4;

	return 0;

err4:
	free_ep_res();
err3:
	fi_close(&dom->fid);
err1:
	fi_close(&fab->fid);
err0:
	return ret;
}

static int init_av(void)
{
	int ret;

	if (opts.dst_addr) {
		// get local address blob. Find the addrlen first. We set
		// addrlen as 0 and fi_getname will return the actual addrlen.
		addrlen = 0;
		ret = fi_getname(&ep->fid, local_addr, &addrlen);
		if (ret != -FI_ETOOSMALL) {
			FT_PRINTERR("fi_getname", ret);
			return ret;
		}

		local_addr = malloc(addrlen);
		ret = fi_getname(&ep->fid, local_addr, &addrlen);
		if (ret) {
			FT_PRINTERR("fi_getname", ret);
			return ret;
		}

		ret = fi_av_insert(av, remote_addr, 1, &remote_fi_addr, 0,
				&fi_ctx_av);
		if (ret != 1) {
			FT_PRINTERR("fi_av_insert", ret);
			return ret;
		}

		// send local addr size and local addr
		memcpy(buf, &addrlen, sizeof(size_t));
		memcpy(buf + sizeof(size_t), local_addr, addrlen);
		ret = send_msg(sizeof(size_t) + addrlen, tag_param);
		if (ret)
			return ret;

		// receive ACK from server
		ret = recv_msg(tag_param + 1);
		if (ret)
			return ret;

	} else {
		// post a recv to get the remote address
		ret = recv_msg(tag_param);
		if (ret)
			return ret;

		memcpy(&addrlen, buf, sizeof(size_t));
		remote_addr = malloc(addrlen);
		memcpy(remote_addr, buf + sizeof(size_t), addrlen);

		ret = fi_av_insert(av, remote_addr, 1, &remote_fi_addr, 0,
				&fi_ctx_av);
		if (ret != 1) {
			FT_PRINTERR("fi_av_insert", ret);
			return ret;
		}

		// send ACK
		ret = send_msg(16, tag_param + 1);
		if (ret)
			return ret;
	}

	return ret;
}

static int tagged_peek(uint64_t tag)
{
	int ret;
	struct fi_msg_tagged msg;

	memset(&msg, 0, sizeof msg);
	msg.tag = tag;
	msg.context = &fi_ctx_search;
	ret = fi_trecvmsg(ep, &msg, FI_PEEK);
	if (ret < 0) {
		if (ret == -ENOMSG)
			fprintf(stdout,
				"No match found with tag [%" PRIu64 "]\n",
				tag);
		else
			FT_PRINTERR("fi_tsearch", ret);
	} else {
		// search was initiated asynchronously, so wait for
		// the completion event
		ret = wait_for_tagged_completion(scq, 1);
	}

	return ret;
}

static int run(void)
{
	int ret = 0;
	ret = init_fabric();
	if (ret)
		return ret;

	ret = init_av();
	if (ret)
		goto out;

	// Receiver
	if (opts.dst_addr) {
		// search for initial tag, it should fail since the sender
		// hasn't sent anyting
		fprintf(stdout, "Searching msg with tag [%" PRIu64 "]\n", tag_data);
		tagged_peek(tag_data);

		fprintf(stdout, "Posting buffer for msg with tag [%" PRIu64 "]\n", 
				tag_data + 1);
		ret = post_recv(tag_data + 1);
		if (ret)
			goto out;

		// synchronize with sender
		fprintf(stdout, "\nSynchronizing with sender..\n\n");
		ret = sync_test();
		if (ret)
			goto out;

		// wait for the completion event of the next tag
		ret = wait_for_tagged_completion(rcq, 1);
		if (ret)
			goto out;
		fprintf(stdout, "Received completion event for msg with tag "
				"[%" PRIu64 "]\n", tag_data + 1);

		// search again for the initial tag, it should be successful now
		fprintf(stdout,
			"Searching msg with initial tag [%" PRIu64 "]\n",
			tag_data);
		tagged_peek(tag_data);

		// wait for the completion event of the initial tag
		ret = recv_msg(tag_data);
		if (ret)
			goto out;
		fprintf(stdout, "Posted buffer and received completion event for"
			       " msg with tag [%" PRIu64 "]\n", tag_data);
	} else {
		// Sender
		// synchronize with receiver
		fprintf(stdout, "Synchronizing with receiver..\n\n");
		ret = sync_test();
		if (ret)
			goto out;

		fprintf(stdout, "Sending msg with tag [%" PRIu64 "]\n",
			tag_data);
		ret = send_msg(16, tag_data);
		if (ret)
			goto out;

		fprintf(stdout, "Sending msg with tag [%" PRIu64 "]\n",
			tag_data + 1);
		ret = send_msg(16, tag_data + 1);
		if (ret)
			goto out;
	}
	/* Finalize before closing ep */
	ft_finalize(ep, scq, rcq, remote_fi_addr);
out:
	free_ep_res();
	fi_close(&dom->fid);
	fi_close(&fab->fid);
	return ret;
}

int main(int argc, char **argv)
{
	int ret, op;
	opts = INIT_OPTS;

	hints = fi_allocinfo();
	if (!hints) {
		FT_PRINTERR("fi_allocinfo", -FI_ENOMEM);
		return EXIT_FAILURE;
	}

	while ((op = getopt(argc, argv, "h" ADDR_OPTS INFO_OPTS)) != -1) {
		switch (op) {
		default:
			ft_parse_addr_opts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints);
			break;
		case '?':
		case 'h':
			ft_usage(argv[0], "An RDM client-server example that uses tagged search.\n");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->rx_attr->total_buffered_recv = buffer_size;
	hints->ep_attr->type = FI_EP_RDM;
	hints->caps = FI_MSG | FI_TAGGED;
	hints->mode = FI_CONTEXT;

	ret = run();
	fi_freeinfo(hints);
	fi_freeinfo(fi);
	return -ret;
}
