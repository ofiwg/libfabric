/*
 * Copyright (c) 2022 Intel Corporation.  All rights reserved.
 * (C) Copyright 2022 Hewlett Packard Enterprise Development LP
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

#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_tagged.h>

#include "shared.h"
#include "rpc.h"

int rpc_timeout = 2000; /* ms */
const uint32_t invalid_id = ~0;
fi_addr_t server_addr = 0;
uint32_t myid = 0;
uint32_t id_at_server = 0;

/* Ring buffer of outstanding requests.*/
static struct rpc_ctrl pending_reqs[MAX_RPCS_INFLIGHT];
static int pending_reqs_read_ptr;
static int pending_reqs_write_ptr;

static bool pending_reqs_avail(void)
{
	if (pending_reqs_read_ptr == 0 &&
	    pending_reqs_write_ptr == (MAX_RPCS_INFLIGHT - 1))
		return false;

	if ((pending_reqs_read_ptr - 1) == pending_reqs_write_ptr)
		return false;

	return true;
}

/* Consume all non-pending entries by advancing the read pointer. This path is
 * used for RMA where RPCs at the target can complete out-of-order with respect
 * to the client RPC request pending queue.
 */
static void advance_pending_req(void)
{
	while (pending_reqs_read_ptr != pending_reqs_write_ptr) {
		if (pending_reqs[pending_reqs_read_ptr].pending)
			break;

		if (++pending_reqs_read_ptr == MAX_RPCS_INFLIGHT)
			pending_reqs_read_ptr = 0;
	}
}

/* Copy a request to the internal queue are return index into the queue. The
 * index into the queue is used as a cookie echoed back by the server in an RPC
 * response message. This is need for RMA RPC requests to lookup the correct
 * pending request RPC.
 */
static int push_pending_req(struct rpc_ctrl *req)
{
	int index;

	advance_pending_req();

	if (!pending_reqs_avail())
		return -ENOSPC;

	pending_reqs[pending_reqs_write_ptr] = *req;
	pending_reqs[pending_reqs_write_ptr].pending = true;

	index = pending_reqs_write_ptr;

	if (++pending_reqs_write_ptr == MAX_RPCS_INFLIGHT)
		pending_reqs_write_ptr = 0;

	return index;
}

/* Pop a pending request RPC. This is reused for messaging RPC responses. */
static int pop_pending_req(struct rpc_ctrl *req)
{
	if (pending_reqs_read_ptr == pending_reqs_write_ptr)
		return -EAGAIN;

	ft_assert(pending_reqs[pending_reqs_read_ptr].pending);

	pending_reqs[pending_reqs_read_ptr].pending = false;
	*req = pending_reqs[pending_reqs_read_ptr];

	if (++pending_reqs_read_ptr == MAX_RPCS_INFLIGHT)
		pending_reqs_read_ptr = 0;

	return 0;
}

static uint64_t rpc_gen_rkey(void)
{
	static uint64_t rkey = 0;
	uint64_t mr_rkey;

	if (fi->domain_attr->mr_mode & FI_MR_PROV_KEY)
		return 0;

	mr_rkey = rkey++;

	if (fi->domain_attr->mr_key_size != 8 &&
	    rkey == (1ULL << (fi->domain_attr->mr_key_size * 8)))
		rkey = 0;

	return mr_rkey;
}

/* Require 32-bits of non-structured tag format. If this proves too limiting to
 * providers, this can change.
 */
#define RDM_RPC_TAG_FORMAT 0xaaaaaaaaULL
#define RDM_RPC_TAG_BIT 31ULL

static int64_t rpc_gen_tag(void)
{
	static uint64_t tag = 0;
	uint64_t rpc_tag;

	if ((fi->ep_attr->mem_tag_format & RDM_RPC_TAG_FORMAT) !=
	    RDM_RPC_TAG_FORMAT) {
		FT_ERR("Unsupported mem_tag_format: %#lx\n",
		       fi->ep_attr->mem_tag_format);
		return -EINVAL;
	}

	rpc_tag = (1ULL << RDM_RPC_TAG_BIT) | tag++;
	if (tag == (1ULL << RDM_RPC_TAG_BIT))
		tag = 0;

	return rpc_tag;
}

char *rpc_cmd_str(uint32_t cmd)
{
	static char *cmd_str[cmd_last] = {
		"hello",
		"goodbye",
		"msg",
		"msg_inject",
		"tag",
		"read",
		"write",
	};

	if (cmd >= cmd_last)
		return "unknown";

	return cmd_str[cmd];
}

char *rpc_op_str(uint32_t op)
{
	static char *op_str[op_last] = {
		"noop",
		"hello",
		"goodbye",
		"msg_req",
		"msg_inject_req",
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

	/* Grab an index into the pend_requests queue. */
	ret = push_pending_req(req);
	if (ret < 0)
		return ret;

	hdr->cookie = ret;

	ret = rpc_inject(hdr, server_addr);
	if (ret)
		pending_reqs[hdr->cookie].pending = false;

	return ret;
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

	ft_assert(resp.cmd == cmd_hello);
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

static int rpc_msg_inject_req(struct rpc_ctrl *ctrl)
{
	struct rpc_hdr req = {0};

	req.client_id = id_at_server;
	req.cmd = cmd_msg_inject;
	req.size = ctrl->size;
	return rpc_send_req(ctrl, &req);
}

static int rpc_msg_resp(struct rpc_ctrl *ctrl)
{
	struct rpc_ctrl req;
	struct rpc_hdr *resp;
	size_t size;
	int ret;

	ret = pop_pending_req(&req);
	if (ret)
		return ret;

	size = sizeof(*resp) + req.size;
	resp = calloc(1, size);
	if (!resp)
		return -FI_ENOMEM;

	ret = rpc_recv(resp, size, FI_ADDR_UNSPEC);
	if (ret)
		goto free;

	ft_assert(resp->cmd == cmd_msg || resp->cmd == cmd_msg_inject);
	ret = ft_check_buf(resp + 1, req.size);

free:
	free(resp);
	return ret;
}

static int rpc_tag_req(struct rpc_ctrl *ctrl)
{
	struct rpc_hdr req = {0};
	int64_t tag;

	req.client_id = id_at_server;
	req.cmd = cmd_tag;
	req.size = ctrl->size;

	tag = rpc_gen_tag();
	if (tag < 0)
		return tag;

	req.data = ctrl->tag = tag;
	return rpc_send_req(ctrl, &req);
}

static int rpc_tag_resp(struct rpc_ctrl *ctrl)
{
	struct rpc_ctrl req;
	struct rpc_hdr *resp;
	size_t size;
	int ret;

	ret = pop_pending_req(&req);
	if (ret)
		return ret;

	size = sizeof(*resp) + req.size;
	resp = calloc(1, size);
	if (!resp)
		return -FI_ENOMEM;

	ret = rpc_trecv(resp, size, req.tag, FI_ADDR_UNSPEC);
	if (ret)
		goto free;

	ft_assert(resp->cmd == cmd_tag);
	ret = ft_check_buf(resp + 1, req.size);

free:
	free(resp);
	return ret;
}

static int rpc_reg_buf(struct rpc_ctrl *ctrl, size_t size, uint64_t access)
{
	int ret;
	uint64_t rkey;

	rkey = rpc_gen_rkey();
	ret = fi_mr_reg(domain, ctrl->buf, size, access, 0, rkey, 0, &ctrl->mr,
			NULL);
	if (ret) {
		FT_PRINTERR("fi_mr_reg", ret);
		return ret;
	}

	if (fi->domain_attr->mr_mode & FI_MR_ENDPOINT) {
		ret = fi_mr_bind(ctrl->mr, &ep->fid, 0);
		if (ret) {
			FT_PRINTERR("fi_mr_bind", ret);
			goto close;
		}
		ret = fi_mr_enable(ctrl->mr);
		if (ret) {
			FT_PRINTERR("fi_mr_enable", ret);
			goto close;
		}
	}
	return FI_SUCCESS;

close:
	fi_close(&ctrl->mr->fid);
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

	ret = rpc_reg_buf(ctrl, size, FI_REMOTE_READ);
	if (ret)
		goto free;

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

	ret = rpc_recv(&resp, sizeof(resp), FI_ADDR_UNSPEC);
	if (ret)
		return ret;

	ft_assert(resp.cmd == cmd_read);

	/* Due to RMA ordering, the client request queue may not align with the
	 * order the RMA operations are completing at the target. This can
	 * result in the client receiving RMA responses in a different order.
	 *
	 * Since the multiple operations inflight requires everything to be
	 * flushed before the operation and/or size changes, all of the RPC
	 * control data except for the MR buffer values should be the same. The
	 * response cookie is need to find the exact RPC request.
	 *
	 * Marking an entry as pending effectively frees it.
	 */
	req = (struct rpc_ctrl *) &pending_reqs[resp.cookie];

	ret = ft_check_buf(&req->buf[req->offset], req->size);

	fi_close(&req->mr->fid);
	free(req->buf);
	req->pending = false;

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

	ret = rpc_reg_buf(ctrl, size, FI_REMOTE_WRITE);
	if (ret)
		goto free;

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

	ret = rpc_recv(&resp, sizeof(resp), FI_ADDR_UNSPEC);
	if (ret)
		return ret;

	ft_assert(resp.cmd == cmd_write);

	/* Due to RMA ordering, the client request queue may not align with the
	 * order the RMA operations are completing at the target. This can
	 * result in the client receiving RMA responses in a different order.
	 *
	 * Since the multiple operations inflight requires everything to be
	 * flushed before the operation and/or size changes, all of the RPC
	 * control data except for the MR buffer values should be the same. The
	 * response cookie is need to find the exact RPC request.
	 *
	 * Marking an entry as pending effectively frees it.
	 */
	req = (struct rpc_ctrl *) &pending_reqs[resp.cookie];

	ret = ft_check_buf(&req->buf[req->offset], req->size);

	fi_close(&req->mr->fid);
	free(req->buf);
	req->pending = false;

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

static int (*ctrl_op[op_last])(struct rpc_ctrl *ctrl) = {
	rpc_noop,
	rpc_hello,
	rpc_goodbye,
	rpc_msg_req,
	rpc_msg_inject_req,
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

int rpc_op_exec(enum rpc_op op, struct rpc_ctrl *ctrl)
{
	return ctrl_op[op](ctrl);
}

void init_rpc_ctrl(struct rpc_ctrl *ctrl)
{
	ctrl->op = op_last;
	ctrl->size = 0;
	ctrl->offset = 0;
	ctrl->buf = 0;
	ctrl->mr = 0;
}

static void complete_rpc(struct rpc_resp *resp)
{
	fi_addr_t addr;
	int ret;

	printf("(%d) complete rpc %s (%s)\n", resp->hdr.client_id,
	       rpc_cmd_str(resp->hdr.cmd), fi_strerror(resp->status));

	if (!resp->status && (resp->flags & rpc_flag_ack))
		ret = rpc_inject(&resp->hdr, resp->hdr.client_id);
	else
		ret = resp->status;

	if (ret) {
		if (resp->hdr.client_id != invalid_id) {
			addr = resp->hdr.client_id;
			printf("(%d) unreachable, removing\n", resp->hdr.client_id);
			ret = fi_av_remove(av, &addr, 1, 0);
			if (ret)
				FT_PRINTERR("fi_av_remove", ret);
		}
	}

	if (resp->mr)
		fi_close(&resp->mr->fid);
	(void) ft_check_buf(resp + 1, resp->hdr.size);
	free(resp);
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
	if (ret) {
		(void) fi_av_remove(av, &addr, 1, 0);
		resp->hdr.client_id = invalid_id;
	}
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
	if (ret)
		FT_PRINTERR("fi_av_remove", ret);

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

static int handle_msg_inject(struct rpc_hdr *req, struct rpc_resp *resp)
{
	int ret;

	ret = fi_inject(ep, &resp->hdr, sizeof(resp->hdr) + resp->hdr.size,
		        req->client_id);
	if (!ret)
		complete_rpc(resp);
	return ret;
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
	handle_msg_inject,
	handle_tag,
	handle_read,
	handle_write,
};

static void start_rpc(struct rpc_hdr *req)
{
	struct rpc_resp *resp;
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

static int wait_on_fd(struct fid_cq *cq, struct fi_cq_tagged_entry *comp)
{
	struct fid *fids[1];
	int fd, ret;

	fd = (cq == txcq) ? tx_fd : rx_fd;
	fids[0] = &cq->fid;

	do {
		ret = fi_trywait(fabric, fids, 1);
		if (ret == FI_SUCCESS) {
			ret = ft_poll_fd(fd, -1);
			if (ret && ret != -FI_EAGAIN)
				break;
		}

		ret = fi_cq_read(cq, comp, 1);
	} while (ret == -FI_EAGAIN);

	return ret;
}

static int wait_for_comp(struct fid_cq *cq, struct fi_cq_tagged_entry *comp)
{
	int ret;

	if (opts.comp_method == FT_COMP_SREAD)
		ret = fi_cq_sread(cq, comp, 1, NULL, -1);
	else
		ret = wait_on_fd(cq, comp);

	return ret;
}

static void *process_rpcs(void *context)
{
	struct fi_cq_tagged_entry comp = {0};
	struct rpc_hello_msg *req;
	int ret;

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
			ret = wait_for_comp(rxcq, &comp);
			if (ret < 0) {
				comp.flags = FI_SEND;
				ret = handle_cq_error();
			} else if (ret > 0) {
				ret = 0;
				if (comp.flags & FI_RECV) {
					req = comp.op_context;
					start_rpc(&req->hdr);
				} else {
					complete_rpc(comp.op_context);
				}
			}
		} while (!ret && !(comp.flags & FI_RECV));
	} while (!ret);

	return NULL;
}

int rpc_run_server(void)
{
	pthread_t thread[rpc_threads];
	int i, ret;

	printf("Starting rpc stress server\n");
	opts.options |= FT_OPT_CQ_SHARED;
	ret = ft_init_fabric();
	if (ret)
		return ret;

	for (i = 0; i < rpc_threads; i++) {
		ret = pthread_create(&thread[i], NULL, process_rpcs,
				     (void *) (uintptr_t) i);
		if (ret) {
			ret = -ret;
			FT_PRINTERR("pthread_create", ret);
			break;
		}
	}

	while (i-- > 0)
		pthread_join(thread[i], NULL);

	ft_free_res();
	return ret;
}
