/*
 * Copyright (c) 2005-2006,2011-2012,2015 Intel Corporation.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
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
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <netdb.h>
#include <getopt.h>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>
#include <shared.h>


struct cma_node {
	int			id;
	int			connected;
	struct fid_domain	*domain;
	struct fid_cq		*cq[2];
	struct fid_ep		*ep;
	struct fid_mr		*mr;
	void			*mrdesc;
	void			*mem;
};

enum CQ_INDEX {
	SEND_CQ_INDEX,
	RECV_CQ_INDEX
};

static struct cs_opts 		opts;
static struct fid_fabric	*fabric;
static struct fid_eq		*eq;
static struct fid_pep		*pep;
static struct cma_node		*nodes;
static int			conn_index;
static int			connects_left;
static int			disconnects_left;
static int			connections = 1;

static struct fi_info		*hints, *info;


static int post_recvs(struct cma_node *node)
{
	int i, ret = 0;

	for (i = 0; i < hints->tx_attr->size && !ret; i++ ) {
		ret = fi_recv(node->ep, node->mem, hints->ep_attr->max_msg_size,
				node->mrdesc, 0, node);
		if (ret) {
			FT_PRINTERR("fi_recv", ret);
			break;
		}
	}
	return ret;
}

static int create_messages(struct cma_node *node)
{
	int ret;

	if (!hints->ep_attr->max_msg_size)
		hints->tx_attr->size = 0;

	if (!hints->tx_attr->size)
		return 0;

	node->mem = malloc(hints->ep_attr->max_msg_size);
	if (!node->mem) {
		printf("failed message allocation\n");
		return -1;
	}

	if (info->mode & FI_LOCAL_MR) {
		ret = fi_mr_reg(node->domain, node->mem, hints->ep_attr->max_msg_size,
				FI_SEND | FI_RECV, 0, 0, 0, &node->mr, NULL);
		if (ret) {
			FT_PRINTERR("fi_reg_mr", ret);
			goto err;
		}
		node->mrdesc = fi_mr_desc(node->mr);
	}

	ret = post_recvs(node);
	return ret;

err:
	free(node->mem);
	return -1;
}

static int init_node(struct cma_node *node, struct fi_info *info)
{
	struct fi_cq_attr cq_attr;
	int ret;

	ret = fi_domain(fabric, info, &node->domain, NULL);
	if (ret) {
		FT_PRINTERR("fi_domain", ret);
		goto out;
	}

	memset(&cq_attr, 0, sizeof cq_attr);
	cq_attr.size = hints->tx_attr->size ? hints->tx_attr->size : 1;
	cq_attr.format = FI_CQ_FORMAT_CONTEXT;
	ret = fi_cq_open(node->domain, &cq_attr, &node->cq[SEND_CQ_INDEX], NULL);
	if (ret) {
		FT_PRINTERR("fi_cq_open", ret);
		goto out;
	}

	ret = fi_cq_open(node->domain, &cq_attr, &node->cq[RECV_CQ_INDEX], NULL);
	if (ret) {
		FT_PRINTERR("fi_cq_open", ret);
		goto out;
	}

	ret = fi_endpoint(node->domain, info, &node->ep, node);
	if (ret) {
		FT_PRINTERR("fi_endpoint", ret);
		goto out;
	}

	ret = fi_ep_bind(node->ep, &node->cq[SEND_CQ_INDEX]->fid, FI_SEND);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		goto out;
	}

	ret = fi_ep_bind(node->ep, &node->cq[RECV_CQ_INDEX]->fid, FI_RECV);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		goto out;
	}

	ret = fi_ep_bind(node->ep, &eq->fid, 0);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		goto out;
	}

	ret = fi_enable(node->ep);
	if (ret) {
		FT_PRINTERR("fi_enable", ret);
		goto out;
	}

	ret = create_messages(node);
	if (ret) {
		printf("cmatose: failed to create messages: %d\n", ret);
		goto out;
	}
out:
	return ret;
}

static int post_sends(struct cma_node *node)
{
	int i, ret = 0;

	if (!node->connected || !hints->tx_attr->size)
		return 0;

	for (i = 0; i < hints->tx_attr->size && !ret; i++) {
		ret = fi_send(node->ep, node->mem, hints->ep_attr->max_msg_size,
				node->mrdesc, 0, node);
		if (ret) {
			FT_PRINTERR("fi_send", ret);
			break;
		}
	}
	return ret;
}

static void destroy_node(struct cma_node *node)
{
	if (node->ep)
		fi_close(&node->ep->fid);

	if (node->cq[SEND_CQ_INDEX])
		fi_close(&node->cq[SEND_CQ_INDEX]->fid);

	if (node->cq[RECV_CQ_INDEX])
		fi_close(&node->cq[RECV_CQ_INDEX]->fid);

	if (node->mr)
		fi_close(&node->mr->fid);

	if (node->mem)
		free(node->mem);

	if (node->domain)
		fi_close(&node->domain->fid);
}

static int alloc_nodes(void)
{
	int ret, i;

	nodes = calloc(connections, sizeof *nodes);
	if (!nodes) {
		printf("cmatose: unable to allocate memory for test nodes\n");
		return -ENOMEM;
	}

	for (i = 0; i < connections; i++) {
		nodes[i].id = i;
		if (opts.dst_addr) {
			ret = init_node(nodes + i, info);
			if (ret)
				goto err;
		}
	}
	return 0;
err:
	while (--i >= 0)
		destroy_node(nodes + i);
	free(nodes);
	return ret;
}

static void destroy_nodes(void)
{
	int i;

	for (i = 0; i < connections; i++)
		destroy_node(nodes + i);
	free(nodes);
}

static int poll_cqs(enum CQ_INDEX index)
{
	struct fi_cq_entry entry[8];
	int done, i, ret;

	for (i = 0; i < connections; i++) {
		if (!nodes[i].connected)
			continue;

		for (done = 0; done < hints->tx_attr->size; done += ret) {
			ret = fi_cq_read(nodes[i].cq[index], entry, 8);
			if (ret == -FI_EAGAIN) {
				ret = 0;
				continue;
			} else if (ret < 0) {
				printf("cmatose: failed polling CQ: %d\n", ret);
				return ret;
			}
		}
	}
	return 0;
}

static int connreq_handler(struct fi_info *info)
{
	struct cma_node *node;
	int ret;

	if (conn_index == connections) {
		ret = -ENOMEM;
		goto err1;
	}

	node = &nodes[conn_index++];
	ret = init_node(node, info);
	if (ret)
		goto err2;

	ret = fi_accept(node->ep, NULL, 0);
	if (ret) {
		FT_PRINTERR("fi_accept", ret);
		goto err2;
	}

	return 0;

err2:
	connects_left--;
err1:
	printf("cmatose: failing connection request\n");
	fi_reject(pep, info->handle, NULL, 0);
	return ret;
}

static int cma_handler(uint32_t event, struct fi_eq_cm_entry *entry)
{
	struct cma_node *node;
	int ret = 0;

	switch (event) {
	case FI_CONNREQ:
		ret = connreq_handler(entry->info);
		fi_freeinfo(entry->info);
		break;
	case FI_CONNECTED:
		node = entry->fid->context;
		node->connected = 1;
		connects_left--;
		disconnects_left++;
		break;
	case FI_SHUTDOWN:
		node = entry->fid->context;
		fi_shutdown(node->ep, 0);
		disconnects_left--;
		break;
	default:
		printf("unexpected event %d\n", event);
		break;
	}

	return ret;
}

static int connect_events(void)
{
	struct fi_eq_cm_entry entry;
	uint32_t event;
	int ret = 0;
	ssize_t rd;

	while (connects_left && !ret) {
		rd = fi_eq_sread(eq, &event, &entry, sizeof entry, -1, 0);
		if (rd != sizeof entry) {
			printf("unexpected event during connect\n");
			FT_PROCESS_EQ_ERR(rd, eq, "fi_eq_sread", "connect");
			ret = -FI_EIO;
			break;
		}

		ret = cma_handler(event, &entry);
	}

	return ret;
}

static int shutdown_events(void)
{
	struct fi_eq_cm_entry entry;
	uint32_t event;
	int ret = 0;
	ssize_t rd;

	while (disconnects_left && !ret) {
		rd = fi_eq_sread(eq, &event, &entry, sizeof entry, -1, 0);
		if (rd != sizeof entry) {
			printf("unexpected event during shutdown\n");
			FT_PROCESS_EQ_ERR(rd, eq, "fi_eq_sread", "shutdown");
			ret = -FI_EIO;
			break;
		}

		ret = cma_handler(event, &entry);
	}

	return ret;
}

static int run_server(void)
{
	int i, ret;

	printf("cmatose: starting server\n");
	ret = fi_passive_ep(fabric, info, &pep, NULL);
	if (ret) {
		FT_PRINTERR("fi_passive_ep", ret);
		return ret;
	}

	ret = fi_pep_bind(pep, &eq->fid, 0);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		goto out;
	}

	ret = fi_listen(pep);
	if (ret) {
		FT_PRINTERR("fi_listen", ret);
		goto out;
	}

	ret = connect_events();
	if (ret)
		goto out;

	if (hints->tx_attr->size) {
		printf("initiating data transfers\n");
		for (i = 0; i < connections; i++) {
			ret = post_sends(&nodes[i]);
			if (ret)
				goto out;
		}

		printf("completing sends\n");
		ret = poll_cqs(SEND_CQ_INDEX);
		if (ret)
			goto out;

		printf("receiving data transfers\n");
		ret = poll_cqs(RECV_CQ_INDEX);
		if (ret)
			goto out;
		printf("data transfers complete\n");

	}

	printf("cmatose: disconnecting\n");
	for (i = 0; i < connections; i++) {
		if (!nodes[i].connected)
			continue;

		nodes[i].connected = 0;
		fi_shutdown(nodes[i].ep, 0);
	}

	ret = shutdown_events();
 	printf("disconnected\n");

out:
	fi_close(&pep->fid);
	return ret;
}

static int run_client(void)
{
	int i, ret, ret2;

	printf("cmatose: starting client\n");

	printf("cmatose: connecting\n");
	for (i = 0; i < connections; i++) {
		ret = fi_connect(nodes[i].ep, info->dest_addr, NULL, 0);
		if (ret) {
			FT_PRINTERR("fi_connect", ret);
			connects_left--;
			return ret;
		}
	}

	ret = connect_events();
	if (ret)
		goto disc;

	if (hints->tx_attr->size) {
		printf("receiving data transfers\n");
		ret = poll_cqs(RECV_CQ_INDEX);
		if (ret)
			goto disc;

		printf("sending replies\n");
		for (i = 0; i < connections; i++) {
			ret = post_sends(nodes + i);
			if (ret)
				goto disc;
		}

		printf("completing sends\n");
		ret = poll_cqs(SEND_CQ_INDEX);
		if (ret)
			goto disc;

		printf("data transfers complete\n");
	}

	ret = 0;
disc:
	ret2 = shutdown_events();
 	printf("disconnected\n");
	if (ret2)
		ret = ret2;
	return ret;
}

static void usage(char *progname)
{
	ft_usage(progname, "Connection establishment test");
	FT_PRINT_OPTS_USAGE("-c <connections>", "# of connections");
	FT_PRINT_OPTS_USAGE("-C <message_count>", "Message count");
	FT_PRINT_OPTS_USAGE("-S <message_size>", "Message size");
	exit(1);
}

int main(int argc, char **argv)
{
	char *node, *service;
	uint64_t flags = 0;
	struct fi_eq_attr eq_attr;
	int op, ret;

	opts = INIT_OPTS;

	hints = fi_allocinfo();
	if (!hints)
		exit(1);

	hints->caps = FI_MSG;
	hints->ep_attr->type = FI_EP_MSG;
	hints->mode = FI_LOCAL_MR;
	hints->tx_attr->mode = hints->mode;
	hints->rx_attr->mode = hints->mode;

	hints->ep_attr->max_msg_size = 100;
	hints->tx_attr->size = 10;
	hints->rx_attr->size = 10;

	while ((op = getopt(argc, argv, "c:C:S:" ADDR_OPTS INFO_OPTS)) != -1) {
		switch (op) {
		case 'c':
			connections = atoi(optarg);
			break;
		case 'C':
			hints->tx_attr->size = atoi(optarg);
			hints->rx_attr->size = hints->tx_attr->size;
			break;
		case 'S':
			hints->ep_attr->max_msg_size = atoi(optarg);
			break;
		default:
			ft_parse_addr_opts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints);
			break;
		case '?':
		case 'h':
			usage(argv[0]);
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];
	
	connects_left = connections;

	ret = ft_read_addr_opts(&node, &service, hints, &flags, &opts);
	if (ret)
		return ret;
	
	ret = fi_getinfo(FT_FIVERSION, node, service, flags, hints, &info);
	if (ret) {
		FT_PRINTERR("fi_getinfo", ret);
		goto exit0;
	}

	printf("using provider: %s\n", info->fabric_attr->prov_name);
	ret = fi_fabric(info->fabric_attr, &fabric, NULL);
	if (ret) {
		FT_PRINTERR("fi_fabric", ret);
		goto exit1;
	}

	memset(&eq_attr, 0, sizeof eq_attr);
	eq_attr.wait_obj = FI_WAIT_UNSPEC;
	ret = fi_eq_open(fabric, &eq_attr, &eq, NULL);
	if (ret) {
		FT_PRINTERR("fi_eq_open", ret);
		goto exit2;
	}

	if (alloc_nodes())
		goto exit3;

	ret = opts.dst_addr ? run_client() : run_server();

	printf("test complete\n");
	destroy_nodes();


exit3:
	fi_close(&eq->fid);
exit2:
	fi_close(&fabric->fid);
exit1:
	fi_freeinfo(info);
exit0:
	fi_freeinfo(hints);
	return -ret;
}
