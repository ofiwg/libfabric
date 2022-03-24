/*
 * Copyright (c) 2022 Intel Corporation.  All rights reserved.
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
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <wait.h>
#include <unistd.h>

#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_tagged.h>

#include "shared.h"
#include "jsmn.h"


/* Input test control */
enum {
	op_noop,
	op_hello,
	op_goodbye,
	op_msg_req,
	op_msg_resp,
	op_tag_req,
	op_tag_resp,
	op_read_req,
	op_read_resp,
	op_write_req,
	op_write_resp,
	op_sleep,
	op_exit,
	op_last,
};

struct rpc_ctrl {
	uint32_t op;
	uint64_t size;
	union {
		uint64_t offset;
		uint64_t tag;
	};
	char *buf;
	struct fid_mr *mr;
};

enum {
#if ENABLE_DEBUG
	rpc_timeout = 300000, /* ms */
#else
	rpc_timeout = 10000, /* ms */
#endif
	rpc_write_key = 189,
	rpc_read_key = 724,
};

/* Wire protocol */
enum {
	cmd_hello,
	cmd_goodbye,
	cmd_msg,
	cmd_tag,
	cmd_read,
	cmd_write,
	cmd_last,
};

struct rpc_hdr {
	uint32_t client_id;
	uint32_t cmd;
	uint64_t size;
	uint64_t offset;
	uint64_t data;
};

struct rpc_hello_msg {
	struct rpc_hdr hdr;
	char addr[32];
};

enum {
	rpc_flag_ack = (1 << 0),
};

struct rpc_resp {
	struct fid_mr *mr;
	int status;
	int flags;
	struct rpc_hdr hdr;
};

struct rpc_client {
	pid_t pid;
};

struct rpc_ctrl *ctrl;
struct rpc_ctrl *pending_req;
int ctrl_cnt;
struct rpc_client clients[128];

static uint32_t myid;
static uint32_t id_at_server;
static fi_addr_t server_addr;
static char *ctrlfile = NULL;


static char *rpc_cmd_str(uint32_t cmd)
{
	static char *cmd_str[cmd_last] = {
		"hello",
		"goodbye",
		"msg",
		"tag",
		"read",
		"write",
	};

	if (cmd >= cmd_last)
		return "unknown";

	return cmd_str[cmd];
}

static char *rpc_op_str(uint32_t op)
{
	static char *op_str[op_last] = {
		"noop",
		"hello",
		"goodbye",
		"msg_req",
		"msg_resp",
		"tag_req",
		"tag_resp",
		"read_req",
		"read_resp",
		"write_req",
		"write_resp",
		"sleep",
		"exit",
	};

	if (op >= op_last)
		return "unknown";

	return op_str[op];
}

static int rpc_inject(struct rpc_hdr *hdr, fi_addr_t addr)
{
	uint64_t start;
	int ret;

	start = ft_gettime_ms();
	do {
		fi_cq_read(txcq, NULL, 0);
		ret = (int) fi_inject(ep, hdr, sizeof(*hdr), addr);
	} while ((ret == -FI_EAGAIN) && (ft_gettime_ms() - start < rpc_timeout));

	if (ret)
		FT_PRINTERR("fi_inject", ret);

	return ret;
}

static int rpc_send(struct rpc_hdr *hdr, size_t size, fi_addr_t addr)
{
	struct fi_cq_tagged_entry comp;
	uint64_t start;
	int ret;

	start = ft_gettime_ms();
	do {
		fi_cq_read(txcq, NULL, 0);
		ret = (int) fi_send(ep, hdr, size, NULL, addr, hdr);
	} while ((ret == -FI_EAGAIN) && (ft_gettime_ms() - start < rpc_timeout));

	if (ret) {
		FT_PRINTERR("fi_send", ret);
		return ret;
	}

	ret = fi_cq_sread(txcq, &comp, 1, NULL, rpc_timeout);
	return ret == 1 ? 0 : ret;
}

static int rpc_deliver(struct rpc_hdr *hdr, size_t size, fi_addr_t addr)
{
	struct fi_msg msg = {0};
	struct iovec iov;
	struct fi_cq_tagged_entry comp;
	uint64_t start;
	int ret;

	iov.iov_base = hdr;
	iov.iov_len = size;

	msg.msg_iov = &iov;
	msg.iov_count = 1;
	msg.addr = addr;
	msg.context = hdr;

	start = ft_gettime_ms();
	do {
		fi_cq_read(txcq, NULL, 0);
		ret = (int) fi_sendmsg(ep, &msg, FI_DELIVERY_COMPLETE);
	} while ((ret == -FI_EAGAIN) && (ft_gettime_ms() - start < rpc_timeout));

	if (ret) {
		FT_PRINTERR("fi_sendmsg (delivery_complete)", ret);
		return ret;
	}

	ret = fi_cq_sread(txcq, &comp, 1, NULL, rpc_timeout);
	return ret == 1 ? 0 : ret;
}

static int rpc_recv(struct rpc_hdr *hdr, size_t size, fi_addr_t addr)
{
	struct fi_cq_tagged_entry comp;
	int ret;

	ret = (int) fi_recv(ep, hdr, size, NULL, addr, hdr);
	if (ret) {
		FT_PRINTERR("fi_recv", ret);
		return ret;
	}

	ret = fi_cq_sread(rxcq, &comp, 1, NULL, rpc_timeout);
	return ret == 1 ? 0 : ret;
}

static int
rpc_trecv(struct rpc_hdr *hdr, size_t size, uint64_t tag, fi_addr_t addr)
{
	struct fi_cq_tagged_entry comp;
	int ret;

	ret = (int) fi_trecv(ep, hdr, size, NULL, addr, tag, 0, hdr);
	if (ret) {
		FT_PRINTERR("fi_trecv", ret);
		return ret;
	}

	ret = fi_cq_sread(rxcq, &comp, 1, NULL, rpc_timeout);
	return ret == 1 ? 0 : ret;
}


static int rpc_send_req(struct rpc_ctrl *req, struct rpc_hdr *hdr)
{
	int ret;

	ret = rpc_inject(hdr, server_addr);
	if (!ret)
		pending_req = req;
	return ret;
}

/* Only support 1 outstanding request at a time for now */
static struct rpc_ctrl *rcp_get_req(struct rpc_ctrl *resp)
{
	return pending_req;
}

static int rpc_noop(struct rpc_ctrl *ctrl)
{
	return 0;
}

/* Send server our address.  This call is synchronous since we need
 * the response before we can send any other requests.
 */
static int rpc_hello(struct rpc_ctrl *ctrl)
{
	struct rpc_hello_msg msg = {0};
	struct rpc_hdr resp;
	size_t addrlen;
	int ret;

	printf("(%d-?) saying hello\n", myid);
	msg.hdr.client_id = myid;
	msg.hdr.cmd = cmd_hello;

	addrlen = sizeof(msg.addr);
	ret = fi_getname(&ep->fid, &msg.addr, &addrlen);
	if (ret) {
		FT_PRINTERR("fi_getname", ret);
		return ret;
	}

	msg.hdr.size = addrlen;
	ret = rpc_send(&msg.hdr, sizeof(msg.hdr) + addrlen, server_addr);
	if (ret)
		return ret;

	ret = rpc_recv(&resp, sizeof(resp), FI_ADDR_UNSPEC);
	if (ret)
		return ret;

	assert(resp.cmd == cmd_hello);
	id_at_server = resp.client_id;
	printf("(%d-%d) we're friends now\n", myid, id_at_server);
	return (int) resp.data;
}

/* Let server know we're leaving gracefully - no response expected. */
static int rpc_goodbye(struct rpc_ctrl *ctrl)
{
	struct rpc_hdr hdr = {0};

	hdr.client_id = id_at_server;
	hdr.cmd = cmd_goodbye;
	return rpc_deliver(&hdr, sizeof hdr, server_addr);
}

static int rpc_msg_req(struct rpc_ctrl *ctrl)
{
	struct rpc_hdr req = {0};

	req.client_id = id_at_server;
	req.cmd = cmd_msg;
	req.size = ctrl->size;
	return rpc_send_req(ctrl, &req);
}

static int rpc_msg_resp(struct rpc_ctrl *ctrl)
{
	struct rpc_ctrl *req;
	struct rpc_hdr *resp;
	size_t size;
	int ret;

	req = rcp_get_req(ctrl);
	size = sizeof(*resp) + req->size;
	resp = calloc(1, size);
	if (!resp)
		return -FI_ENOMEM;

	ret = rpc_recv(resp, size, FI_ADDR_UNSPEC);
	if (ret)
		goto free;

	assert(resp->cmd == cmd_msg);
	ret = ft_check_buf(resp + 1, req->size);

free:
	free(resp);
	return ret;
}

static int rpc_tag_req(struct rpc_ctrl *ctrl)
{
	struct rpc_hdr req = {0};

	req.client_id = id_at_server;
	req.cmd = cmd_tag;
	req.size = ctrl->size;
	req.data = ctrl->tag;
	return rpc_send_req(ctrl, &req);
}

static int rpc_tag_resp(struct rpc_ctrl *ctrl)
{
	struct rpc_ctrl *req;
	struct rpc_hdr *resp;
	size_t size;
	int ret;

	req = rcp_get_req(ctrl);
	size = sizeof(*resp) + req->size;
	resp = calloc(1, size);
	if (!resp)
		return -FI_ENOMEM;

	ret = rpc_trecv(resp, size, req->tag, FI_ADDR_UNSPEC);
	if (ret)
		goto free;

	assert(resp->cmd == cmd_tag);
	ret = ft_check_buf(resp + 1, req->size);

free:
	free(resp);
	return ret;
}

static int rpc_read_req(struct rpc_ctrl *ctrl)
{
	struct rpc_hdr req = {0};
	size_t size;
	int ret;

	size = ctrl->offset + ctrl->size;
	ctrl->buf = calloc(1, size);
	if (!ctrl->buf)
		return -FI_ENOMEM;

	ft_fill_buf(&ctrl->buf[ctrl->offset], ctrl->size);
	ret = fi_mr_reg(domain, ctrl->buf, size, FI_REMOTE_READ, 0,
			rpc_read_key, 0, &ctrl->mr, NULL);
	if (ret) {
		FT_PRINTERR("fi_mr_reg", ret);
		goto free;
	}

	req.client_id = id_at_server;
	req.cmd = cmd_read;
	req.size = ctrl->size;

	req.offset = ctrl->offset;
	if (fi->domain_attr->mr_mode & FI_MR_VIRT_ADDR)
		req.offset += (uintptr_t) ctrl->buf;
	req.data = fi_mr_key(ctrl->mr);

	ret = rpc_send_req(ctrl, &req);
	if (ret)
		goto close;

	return 0;

close:
	fi_close(&ctrl->mr->fid);
free:
	free(ctrl->buf);
	return ret;
}

static int rpc_read_resp(struct rpc_ctrl *ctrl)
{
	struct rpc_hdr resp = {0};
	struct rpc_ctrl *req;
	int ret;

	req = rcp_get_req(ctrl);
	ret = rpc_recv(&resp, sizeof(resp), FI_ADDR_UNSPEC);
	if (ret)
		goto close;

	assert(resp.cmd == cmd_read);
	ret = ft_check_buf(&req->buf[req->offset], req->size);

close:
	fi_close(&req->mr->fid);
	free(req->buf);
	return ret;
}

static int rpc_write_req(struct rpc_ctrl *ctrl)
{
	struct rpc_hdr req = {0};
	size_t size;
	int ret;

	size = ctrl->offset + ctrl->size;
	ctrl->buf = calloc(1, size);
	if (!ctrl->buf)
		return -FI_ENOMEM;

	ret = fi_mr_reg(domain, ctrl->buf, size, FI_REMOTE_WRITE, 0,
			rpc_write_key, 0, &ctrl->mr, NULL);
	if (ret) {
		FT_PRINTERR("fi_mr_reg", ret);
		goto free;
	}

	req.client_id = id_at_server;
	req.cmd = cmd_write;
	req.size = ctrl->size;

	req.offset = ctrl->offset;
	if (fi->domain_attr->mr_mode & FI_MR_VIRT_ADDR)
		req.offset += (uintptr_t) ctrl->buf;
	req.data = fi_mr_key(ctrl->mr);

	ret = rpc_send_req(ctrl, &req);
	if (ret)
		goto close;

	return 0;

close:
	fi_close(&ctrl->mr->fid);
free:
	free(ctrl->buf);
	return ret;
}

static int rpc_write_resp(struct rpc_ctrl *ctrl)
{
	struct rpc_hdr resp = {0};
	struct rpc_ctrl *req;
	int ret;

	req = rcp_get_req(ctrl);
	ret = rpc_recv(&resp, sizeof(resp), FI_ADDR_UNSPEC);
	if (ret)
		goto close;

	assert(resp.cmd == cmd_write);
	ret = ft_check_buf(&req->buf[req->offset], req->size);

close:
	fi_close(&req->mr->fid);
	free(req->buf);
	return ret;
}

/* Used to delay client, which can force server into a flow control
 * state or into the middle of a transfer when the client exits.
 */
static int rpc_sleep(struct rpc_ctrl *ctrl)
{
	int ret;

	ret = usleep((useconds_t) ctrl->size * 1000);
	return ret ? -errno : 0;
}

static int rpc_exit(struct rpc_ctrl *ctrl)
{
	exit(0);
}


int (*ctrl_op[op_last])(struct rpc_ctrl *ctrl) = {
	rpc_noop,
	rpc_hello,
	rpc_goodbye,
	rpc_msg_req,
	rpc_msg_resp,
	rpc_tag_req,
	rpc_tag_resp,
	rpc_read_req,
	rpc_read_resp,
	rpc_write_req,
	rpc_write_resp,
	rpc_sleep,
	rpc_exit,
};

static int run_child(void)
{
	int i, ret;

	printf("(%d-?) running\n", myid);
	ret = ft_init_fabric();
	if (ret) {
		FT_PRINTERR("ft_init_fabric", ret);
		return ret;
	}

	ret = fi_av_insert(av, fi->dest_addr, 1, &server_addr, 0, NULL);
	if (ret != 1) {
		ret = -FI_EINTR;
		FT_PRINTERR("fi_av_insert", ret);
		goto free;
	}

	ret = rpc_hello(NULL);
	if (ret)
		goto free;

	for (i = 0; i < ctrl_cnt && !ret; i++) {
		printf("(%d-%d) rpc op %s\n", myid, id_at_server,
		       rpc_op_str(ctrl[i].op));
		ret = ctrl_op[ctrl[i].op](&ctrl[i]);
	}

free:
	ft_free_res();
	return ret;
}


static struct rpc_ctrl ctrl_array[] = {
	{op_goodbye},
	{op_hello},
	{op_msg_req, 1000},
	{op_msg_resp},
	{op_tag_req, 2000},
	{op_tag_resp},
	{op_msg_req, 1000000},
	{op_msg_resp},
	{op_write_req, 5600000, {12000}},
	{op_write_resp},
	{op_read_req, 64000},
	{op_read_resp},
	{op_tag_req, 2000000},
	{op_tag_resp},
	{op_read_req, 32000},
	{op_read_resp},
	{op_write_req, 86000, {6000}},
	{op_write_resp},
	{op_read_req, 1000000},
	{op_read_resp},
	{op_write_req, 56000, {12000}},
	{op_write_resp},
	{op_sleep, 100},
	{op_exit},
};

/* TODO: read and parse control file */
static int init_rpc(void)
{
	ctrl = ctrl_array;
	ctrl_cnt = ARRAY_SIZE(ctrl_array);
	return 0;
}

static int run_parent(void)
{
	pid_t pid;
	int i, ret;

	printf("Starting rpc client(s)\n");
	ret = init_rpc();
	if (ret)
		return ret;

	for (i = 0; i < opts.iterations; i++) {
		/* If there's only 1 client, run it from this process.  This
		 * greatly helps with debugging.
		 */
		if (opts.num_connections == 1) {
			ret = run_child();
			if (ret)
				return ret;
			continue;
		}

		for (myid = 0; myid < (uint32_t) opts.num_connections; myid++) {
			if (clients[myid].pid) {
				pid = waitpid(clients[myid].pid, NULL, 0);
				if (pid < 0)
					FT_PRINTERR("waitpid", -errno);
				clients[myid].pid = 0;
			}

			ret = fork();
			if (!ret)
				return run_child();
			if (ret < 0)
				return -errno;

			clients[myid].pid = ret;
		}
	}

	for (myid = 0; myid < (uint32_t) opts.num_connections; myid++) {
		if (clients[myid].pid) {
			pid = waitpid(clients[myid].pid, NULL, 0);
			if (pid < 0)
				FT_PRINTERR("waitpid", -errno);
			clients[myid].pid = 0;
		}
	}

	return ret;
}


/* If we fail to send the response (e.g. EAGAIN), we need to remove the
 * address from the AV to avoid double insertions.  We could loop on
 * EAGAIN in this call, but by replaying the entire handle_hello sequence
 * we end up stressing the AV insert/remove path more.
 */
static int handle_hello(struct rpc_hdr *req, struct rpc_resp *resp)
{
	struct rpc_hello_msg *msg;
	fi_addr_t addr;
	int ret;

	if (!req->size || req->size > sizeof(msg->addr))
		return -FI_EINVAL;

	msg = (struct rpc_hello_msg *) req;
	ret = fi_av_insert(av, &msg->addr, 1, &addr, 0, NULL);
	if (ret <= 0)
		return -FI_EADDRNOTAVAIL;

	resp->hdr.client_id = (uint32_t) addr;
	resp->hdr.size = 0;
	ret = fi_send(ep, &resp->hdr, sizeof(resp->hdr), NULL, addr, resp);
	if (ret)
		(void) fi_av_remove(av, &addr, 1, 0);
	return ret;
}

/* How do we know that the client didn't restart with the same address
 * and send a hello message immediately before we could handle this
 * goodbye message?  This is a race that the test has to handle, rather
 * than libfabric, but also very unlikely to happen unless the client
 * intentionally re-uses addresses.
 */
static int handle_goodbye(struct rpc_hdr *req, struct rpc_resp *resp)
{
	fi_addr_t addr;
	int ret;

	addr = req->client_id;
	ret = fi_av_remove(av, &addr, 1, 0);
	assert(ret == FI_SUCCESS);

	/* No response generated */
	printf("(%d) complete rpc %s (%s)\n", resp->hdr.client_id,
	       rpc_cmd_str(resp->hdr.cmd), fi_strerror(resp->status));
	free(resp);
	return 0;
}

static int handle_msg(struct rpc_hdr *req, struct rpc_resp *resp)
{
	return fi_send(ep, &resp->hdr, sizeof(resp->hdr) + resp->hdr.size,
		       NULL, req->client_id, resp);
}

static int handle_tag(struct rpc_hdr *req, struct rpc_resp *resp)
{
	return fi_tsend(ep, &resp->hdr, sizeof(resp->hdr) + resp->hdr.size,
			NULL, req->client_id, req->data, resp);
}

static int handle_read(struct rpc_hdr *req, struct rpc_resp *resp)
{
	resp->flags = rpc_flag_ack;
	return fi_read(ep, resp + 1, resp->hdr.size, NULL, req->client_id,
		       req->offset, req->data, resp);
}

static int handle_write(struct rpc_hdr *req, struct rpc_resp *resp)
{
	resp->flags = rpc_flag_ack;
	return fi_write(ep, resp + 1, resp->hdr.size, NULL, req->client_id,
			req->offset, req->data, resp);
}

int (*handle_rpc[cmd_last])(struct rpc_hdr *req, struct rpc_resp *resp) = {
	handle_hello,
	handle_goodbye,
	handle_msg,
	handle_tag,
	handle_read,
	handle_write,
};

static void *complete_rpc(void *arg)
{
	struct rpc_resp *resp = arg;

	printf("(%d) complete rpc %s (%s)\n", resp->hdr.client_id,
	       rpc_cmd_str(resp->hdr.cmd), fi_strerror(resp->status));

	if (resp->flags & rpc_flag_ack)
		(void) rpc_inject(&resp->hdr, resp->hdr.client_id);

	if (resp->mr)
		fi_close(&resp->mr->fid);
	(void) ft_check_buf(resp + 1, resp->hdr.size);
	free(resp);
	return NULL;
}

static void *start_rpc(void *arg)
{
	struct rpc_resp *resp;
	struct rpc_hdr *req = arg;
	uint64_t start;
	int ret;

	printf("(%d) start rpc %s\n", req->client_id, rpc_cmd_str(req->cmd));
	if (req->cmd >= cmd_last)
		goto free;

	if (!req->size)
		goto free;

	resp = calloc(1, sizeof(*resp) + req->size);
	if (!resp)
		goto free;

	resp->hdr = *req;
	ft_fill_buf(resp + 1, resp->hdr.size);

	start = ft_gettime_ms();
	do {
		(void) fi_cq_read(txcq, NULL, 0);
		ret = handle_rpc[req->cmd](req, resp);
	} while ((ret == -FI_EAGAIN) && (ft_gettime_ms() - start < rpc_timeout));

	if (ret) {
		resp->status = ret;
		(void) complete_rpc(resp);
	}

free:
	free(req);
	return NULL;
}

/* Completion errors are expected as clients are misbehaving */
static int handle_cq_error(void)
{
	struct fi_cq_err_entry cq_err = {0};
	struct rpc_resp *resp;
	int ret;

	ret = fi_cq_readerr(txcq, &cq_err, 0);
	if (ret < 0) {
		if (ret == -FI_EAGAIN)
			return 0;

		FT_PRINTERR("fi_cq_readerr", ret);
		return ret;
	}

	resp = cq_err.op_context;
	resp->status = -cq_err.err;
	FT_CQ_ERR(txcq, cq_err, NULL, 0);
	complete_rpc(resp);
	return 0;
}

static int run_server(void)
{
	struct fi_cq_tagged_entry comp = {0};
	struct rpc_hello_msg *req;
	pthread_t thread;
	int ret;

	printf("Starting rpc stress server\n");
	opts.options |= FT_OPT_CQ_SHARED;
	ret = ft_init_fabric();
	if (ret)
		return ret;

	do {
		req = calloc(1, sizeof(*req));
		if (!req) {
			ret = -FI_ENOMEM;
			break;
		}

		ret = (int) fi_recv(ep, req, sizeof(*req), NULL,
				    FI_ADDR_UNSPEC, req);
		if (ret) {
			FT_PRINTERR("fi_read", ret);
			break;
		}

		do {
			/* The rx and tx cq's are the same */
			ret = fi_cq_sread(rxcq, &comp, 1, NULL, -1);
			if (ret < 0) {
				comp.flags = FI_SEND;
				ret = handle_cq_error();
			} else if (ret > 0) {
				if (comp.flags & FI_RECV) {
					ret = pthread_create(&thread, NULL,
							     start_rpc, req);
				} else {
					ret = pthread_create(&thread, NULL,
							     complete_rpc,
							     comp.op_context);
				}
				if (ret) {
					ret = -ret;
					FT_PRINTERR("pthread_create", -ret);
				}
			}
		} while (!ret && !(comp.flags & FI_RECV));
	} while (!ret);

	ft_free_res();
	return ret;
}

int main(int argc, char **argv)
{
	int op, ret;

	opts = INIT_OPTS;
	opts.options |= FT_OPT_SKIP_MSG_ALLOC | FT_OPT_SKIP_ADDR_EXCH;
	opts.mr_mode = 0;
	opts.iterations = 1; // remove
	opts.num_connections = 1; // 16
	opts.comp_method = FT_COMP_WAIT_FD;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt(argc, argv, "u:h" CS_OPTS INFO_OPTS)) != -1) {
		switch (op) {
		default:
			ft_parsecsopts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints, &opts);
			break;
		case 'u':
			ctrlfile = strdup(optarg);
			break;
		case '?':
		case 'h':
			ft_csusage(argv[0], "An RDM endpoint error stress test.");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->caps = FI_MSG | FI_TAGGED | FI_RMA;
	hints->domain_attr->mr_mode = opts.mr_mode;
	hints->domain_attr->av_type = FI_AV_TABLE;
	hints->ep_attr->type = FI_EP_RDM;
	hints->tx_attr->inject_size = sizeof(struct rpc_hello_msg);

	if (opts.dst_addr)
		ret = run_parent();
	else
		ret = run_server();

//	free(ctrl);
	return ft_exit_code(ret);
}
