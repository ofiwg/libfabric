/*
 * (C) Copyright 2022 Hewlett Packard Enterprise Development LP
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

#include <rdma/fi_tagged.h>
#include <assert.h>

#include "shared.h"
#include "benchmark_shared.h"
#include "hmem.h"

/* Require 32-bits of non-structured tag format. If this proves too limiting to
 * providers, this can change.
 */
#define RDM_RPC_TAG_FORMAT 0xaaaaaaaaULL
#define RDM_RPC_TAG_BIT 31ULL

enum rpc_op_type {
	/* Client only RPC operation values. */
	RPC_READ,
	RPC_WRITE,

	/* Server only RPC operation values. */
	RPC_RESP,
};

struct rpc_req {
	uint64_t op;
	uint64_t rkey;
	uint64_t addr;
	uint64_t len;
	uint64_t tag;
};

struct rpc_resp {
	uint64_t op;
	uint32_t status;
};

struct rpc_desc {
	uint64_t tag;

	/* Message MR is only allocate if provider requires FI_MR_LOCAL. */
	struct fid_mr *msg_mr;
	void *msg_mr_desc;

	/* Payload MR is only allocated if RPC descriptor is used for client
	 * (i.e. remote MR) or server with FI_MR_LOCAL | FI_MR_HMEM.
	 */
	void *payload;
	struct fid_mr *payload_mr;
	void *payload_mr_desc;

	/* RPC messages used for sending/receiving requests/responses. */
	struct {
		struct rpc_req req;
		struct rpc_resp resp;
	} msg;
};

/* Array of RPC descriptors. Size is based on the number of RPCs inflight. */
static struct rpc_desc *descs;

static size_t rpc_count;
static size_t rpc_inflight;
static size_t rpc_byte_count;
static enum rpc_op_type rpc_op;
static bool rpc_client;

static int rpc_reg_mr(const void *buf, size_t len, uint64_t access,
		      enum fi_hmem_iface iface, uint64_t device,
		      struct fid_mr **mr)
{
	static uint64_t rkey = 0;
	uint64_t mr_rkey;

	if (!(fi->domain_attr->mr_mode & FI_MR_PROV_KEY)) {
		mr_rkey = rkey;
		rkey++;

		if (fi->domain_attr->mr_key_size != 8 &&
		    rkey == (1ULL << (fi->domain_attr->mr_key_size * 8)))
			rkey = 0;
	} else {
		mr_rkey = 0;
	}

	return ft_reg_mr_iface(fi, (void *) buf, len, access, mr_rkey, iface,
			       device, mr, NULL);
}

static int rpc_desc_reg_payload_mr(struct rpc_desc *desc)
{
	int ret;
	uint64_t access;

	if (rpc_op == RPC_READ || rpc_op == RPC_WRITE ||
	    fi->domain_attr->mr_mode & FI_MR_LOCAL ||
	    fi->domain_attr->mr_mode & FI_MR_HMEM) {

		/* RPC_READ/RPC_WRITE assumes the payload buffer is a client
		 * buffer (i.e. buffer server will do RMA against).
		 */
		if (rpc_op == RPC_READ)
			access = FI_REMOTE_WRITE;
		else if (rpc_op == RPC_WRITE)
			access = FI_REMOTE_READ;
		else
			access = FI_WRITE | FI_READ;

		ret = rpc_reg_mr(desc->payload, opts.transfer_size, access,
				 opts.iface, opts.device, &desc->payload_mr);
		if (ret)
			goto out;

		desc->payload_mr_desc = fi_mr_desc(desc->payload_mr);
	} else {
		desc->payload_mr = NULL;
		desc->payload_mr_desc = NULL;
		ret = 0;
	}

out:
	return ret;
}

static void rpc_desc_cleanup(struct rpc_desc *desc)
{
	FT_CLOSE_FID(desc->payload_mr);
	FT_CLOSE_FID(desc->msg_mr);
	ft_hmem_free(opts.iface, desc->payload);
}

static void rpc_desc_client_init_req(struct rpc_desc *desc)
{
	desc->msg.req.op = rpc_op;
	desc->msg.req.rkey = fi_mr_key(desc->payload_mr);

	if (fi->domain_attr->mr_mode & FI_MR_VIRT_ADDR)
		desc->msg.req.addr = (uint64_t) desc->payload;
	else
		desc->msg.req.addr = 0;

	desc->msg.req.len = opts.transfer_size;
	desc->msg.req.tag = desc->tag;
}

static int rpc_desc_init(struct rpc_desc *desc)
{
	int ret;
	static uint64_t tag = 0;

	ret = ft_hmem_alloc(opts.iface, opts.device, &desc->payload,
			    opts.transfer_size);
	if (!desc->payload) {
		ret = -ENOMEM;
		FT_ERR("ft_hmem_alloc failed: %d", ret);
		goto err;
	}

	/* Register messaging buffer. */
	if (fi->domain_attr->mr_mode & FI_MR_LOCAL) {
		ret = rpc_reg_mr(&desc->msg, sizeof(desc->msg),
				 FI_WRITE | FI_READ, FI_HMEM_SYSTEM, 0,
				 &desc->msg_mr);
		if (ret) {
			FT_ERR("rpc_context_reg_mr failed: %d", ret);
			goto err_free_payload;
		}

		desc->msg_mr_desc = fi_mr_desc(desc->msg_mr);
	} else {
		desc->msg_mr = NULL;
		desc->msg_mr_desc = NULL;
	}

	if (rpc_op == RPC_READ || rpc_op == RPC_WRITE) {
		desc->tag = (1ULL << RDM_RPC_TAG_BIT) | tag++;
		if (tag == (1ULL << RDM_RPC_TAG_BIT))
			tag = 0;
	}

	return 0;

err_free_payload:
	free(desc->payload);
err:
	return ret;
}

/* Process a clients request. Issue RMA operation to read/write data. */
static int rpc_desc_server_process_req(struct rpc_desc *desc)
{
	int ret;

	if (desc->msg.req.len != opts.transfer_size) {
		FT_ERR("Bad client RPC size: expected=%lu got=%lu",
		       opts.transfer_size, desc->msg.req.len);
		return -EINVAL;
	}

	do {
		if (desc->msg.req.op == RPC_READ)
			ret = fi_write(ep, desc->payload, desc->msg.req.len,
				       desc->payload_mr_desc, remote_fi_addr,
				       desc->msg.req.addr, desc->msg.req.rkey,
				       desc);
		else
			ret = fi_read(ep, desc->payload, desc->msg.req.len,
				      desc->payload_mr_desc, remote_fi_addr,
				      desc->msg.req.addr, desc->msg.req.rkey,
				      desc);

		if (ret == -FI_EAGAIN)
			fi_cq_read(txcq, NULL, 0);
	} while (ret == -FI_EAGAIN);

	if (ret)
		FT_ERR("%s failed: %d",
		       desc->msg.req.op == RPC_READ ? "fi_write" : "fi_read",
		       ret);

	return ret;
}

static int rpc_desc_server_process_rma_response(struct rpc_desc *desc)
{
	int ret;

	/* Echo the cookie back to the client. */
	desc->msg.resp.op = RPC_RESP;
	desc->msg.resp.status = 0;

	do {
		ret = fi_tsend(ep, &desc->msg.resp, sizeof(desc->msg.resp),
			       desc->msg_mr_desc, remote_fi_addr,
			       desc->msg.req.tag, desc);
		if (ret == -FI_EAGAIN)
			fi_cq_read(txcq, NULL, 0);
	} while (ret == -FI_EAGAIN);

	if (ret)
		FT_ERR("fi_send failed: %d", ret);

	return ret;
}

static int rpc_desc_server_process_send_response(struct rpc_desc *desc)
{
	/* Server has completed the RPC. Metrics can be updated. */
	rpc_byte_count += desc->msg.req.len;
	rpc_count++;
	rpc_inflight--;

	return 0;
}

static int rpc_desc_server_post_recv(struct rpc_desc *desc)
{
	int ret;

	/* Reuse the RPC descriptor for the next RPC request. Unlike the client,
	 *no cleanup and re-init of the RPC descriptor happens.
	 */
	do {
		ret = fi_recv(ep, &desc->msg.req, sizeof(desc->msg.req),
			      desc->msg_mr_desc, FI_ADDR_UNSPEC, desc);
		if (ret == -FI_EAGAIN)
			fi_cq_read(rxcq, NULL, 0);
	} while (ret == -FI_EAGAIN);

	if (ret)
		FT_ERR("fi_recv failed: %d", ret);
	else
		rpc_inflight++;

	return ret;
}

static int rpc_desc_client_send_req(struct rpc_desc *desc)
{
	int ret;

	/* Register the payload buffer and prepare the RPC request. The payload
	 * buffer should have already been allocated during RPC descriptor init.
	 */
	ret = rpc_desc_reg_payload_mr(desc);
	if (ret) {
		FT_ERR("fi_trecv failed: %d", ret);
		goto err;
	}

	/* Prepost tagged receive buffer to handle response. */
	do {
		ret = fi_trecv(ep, &desc->msg.resp, sizeof(desc->msg.resp),
			       desc->msg_mr_desc, FI_ADDR_UNSPEC, desc->tag, 0,
			       desc);
		if (ret == -FI_EAGAIN)
			fi_cq_read(rxcq, NULL, 0);
	} while (ret == -FI_EAGAIN);

	if (ret) {
		FT_ERR("fi_trecv failed: %d", ret);
		goto err_dereg_mr;
	}

	/* Initialize the RPC request message and send it to the server.
	 * Tag is defined during RPC descriptor initialization.
	 * rpc_desc_reg_payload_mr() fills in the remaining fields.
	 */
	rpc_desc_client_init_req(desc);

	do {
		ret = fi_send(ep, &desc->msg.req, sizeof(desc->msg.req),
			      desc->msg_mr_desc, remote_fi_addr, desc);
		if (ret == -FI_EAGAIN)
			fi_cq_read(txcq, NULL, 0);
	} while (ret == -FI_EAGAIN);

	if (ret) {
		FT_ERR("fi_send failed: %d", ret);
		goto err_cancel_trecv;
	} else {
		rpc_inflight++;
	}

	return 0;

err_cancel_trecv:
	fi_cancel(&ep->fid, desc);
err_dereg_mr:
	FT_CLOSE_FID(desc->payload_mr);
err:
	return ret;
}

static int rpc_desc_client_process_recv(struct rpc_desc *desc)
{
	if (desc->msg.resp.status != 0) {
		FT_ERR("Bad server response status");
		return -FI_EIO;
	}

	/* Server has completed the RPC. Metrics can be updated. */
	rpc_byte_count += desc->msg.req.len;
	rpc_count++;
	rpc_inflight--;

	FT_CLOSE_FID(desc->payload_mr);

	return 0;
}

static void rpc_process_tx_cq(size_t rpc_limit)
{
	struct fi_cq_tagged_entry msg_event;
	int ret;

	ret = fi_cq_read(txcq, &msg_event, 1);
	if (ret == -FI_EAGAIN)
		return;

	ft_assert(ret == 1);

	if (rpc_client) {
		/* Send events are ignored. */
		ft_assert(msg_event.flags == (FI_MSG | FI_SEND));
	} else {
		ft_assert((msg_event.flags == (FI_TAGGED | FI_SEND)) ||
			  (msg_event.flags == (FI_RMA | FI_READ)) ||
			  (msg_event.flags == (FI_RMA | FI_WRITE)));

		if (msg_event.flags == (FI_TAGGED | FI_SEND)) {
			ret = rpc_desc_server_process_send_response(msg_event.op_context);
			ft_assert(ret == FI_SUCCESS);

			/* Repost the RPC descriptor if the limit has not been
			 * reached.
			 */
			if (rpc_inflight <
			    MIN(opts.window_size, rpc_limit - rpc_count)) {
				ret = rpc_desc_server_post_recv(msg_event.op_context);
				ft_assert(ret == FI_SUCCESS);
			}
		} else {
			ret = rpc_desc_server_process_rma_response(msg_event.op_context);
			ft_assert(ret == FI_SUCCESS);
		}
	}
}

static void rpc_process_rx_cq(size_t rpc_limit)
{
	struct fi_cq_tagged_entry msg_event;
	int ret;

	ret = fi_cq_read(rxcq, &msg_event, 1);
	if (ret == -FI_EAGAIN)
		return;

	ft_assert(ret == 1);

	if (rpc_client) {
		ft_assert(msg_event.flags == (FI_TAGGED | FI_RECV));

		ret = rpc_desc_client_process_recv(msg_event.op_context);
		ft_assert(ret == FI_SUCCESS);

		/* Repost the RPC request if the limit has not been reached. */
		if (rpc_inflight <
		    MIN(opts.window_size, rpc_limit - rpc_count)) {
			ret = rpc_desc_client_send_req(msg_event.op_context);
			ft_assert(ret == FI_SUCCESS);
		}
	} else {
		ft_assert(msg_event.flags == (FI_MSG | FI_RECV));

		ret = rpc_desc_server_process_req(msg_event.op_context);
		ft_assert(ret == FI_SUCCESS);
	}
}

static void rpc_server_run_benchmark(size_t rpc_limit)
{
	int ret;
	size_t i;
	struct rpc_desc *desc;

	rpc_count = 0;
	rpc_byte_count = 0;

	ft_start();

	/* Prepost receive buffers. */
	for (i = 0; i < MIN(opts.window_size, rpc_limit); i++) {
		desc = descs + i;

		ret = rpc_desc_server_post_recv(desc);
		ft_assert(ret == FI_SUCCESS);
	}

	do {
		rpc_process_tx_cq(rpc_limit);
		rpc_process_rx_cq(rpc_limit);
	} while (rpc_count < rpc_limit);

	ft_stop();
}

static void rpc_server_run(void)
{
	size_t i;
	int ret;
	struct rpc_desc *desc;

	/* Register payload buffer for the duration of the test. */
	for (i = 0; i < opts.window_size; i++) {
		desc = descs + i;

		ret = rpc_desc_reg_payload_mr(desc);
		ft_assert(ret == FI_SUCCESS);
	}

	ft_sync();
	rpc_server_run_benchmark(opts.warmup_iterations);

	ft_sync();
	rpc_server_run_benchmark(opts.iterations);

	/* Deregister payload buffers with test completed of the test. */
	for (i = 0; i < opts.window_size; i++) {
		desc = descs + i;
		FT_CLOSE_FID(desc->payload_mr);
	}
}

static void rpc_client_run_benchmark(size_t rpc_limit)
{
	int ret;
	size_t i;
	struct rpc_desc *desc;

	rpc_count = 0;
	rpc_byte_count = 0;

	ft_start();

	/* Post RPC requests. */
	for (i = 0; i < MIN(opts.window_size, rpc_limit); i++) {
		desc = descs + i;

		ret = rpc_desc_client_send_req(desc);
		ft_assert(ret == FI_SUCCESS);
	}

	do {
		rpc_process_tx_cq(rpc_limit);
		rpc_process_rx_cq(rpc_limit);
	} while (rpc_count < rpc_limit);

	ft_stop();
}

static void rpc_client_run(void)
{
	ft_sync();
	rpc_client_run_benchmark(opts.warmup_iterations);

	ft_sync();
	rpc_client_run_benchmark(opts.iterations);
}

static char *test_names[2][2] = {
	{
		"Server RPC Read Response",
		"Server RPC Write Response",
	},
	{
		"Client RPC Read Request",
		"Client RPC Write Request",
	},
};

static void rpc_run(void)
{
	struct rpc_desc *desc;
	size_t i;
	int ret;

	descs = calloc(opts.window_size, sizeof(*descs));
	ft_assert(descs);

	for (i = 0; i < opts.window_size; i++) {
		desc = descs + i;

		ret = rpc_desc_init(desc);
		ft_assert(ret == FI_SUCCESS);
	}

	if (rpc_client)
		rpc_client_run();
	else
		rpc_server_run();

	show_perf(test_names[rpc_client][rpc_op], opts.transfer_size,
		  opts.iterations, &start, &end, 1);

	for (i = 0; i < opts.window_size; i++) {
		desc = descs + i;
		rpc_desc_cleanup(desc);
	}

	free(descs);
}

int main(int argc, char **argv)
{
	int op;
	int ret;
	int i;

	opts = INIT_OPTS;
	opts.options |= FT_OPT_BW;
	opts.mr_mode = FI_MR_ENDPOINT | FI_MR_ALLOCATED | FI_MR_VIRT_ADDR |
		FI_MR_PROV_KEY | FI_MR_LOCAL | FI_MR_HMEM;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt(argc, argv, "h" CS_OPTS INFO_OPTS BENCHMARK_OPTS)) != -1) {
		switch (op) {
		default:
			ft_parse_benchmark_opts(op, optarg);
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			break;
		case '?':
		case 'h':
			ft_csusage(argv[0],
				   "RPC communication style benchmark");
			ft_benchmark_usage();
			return EXIT_FAILURE;
		}
	}

	if (optind < argc) {
		opts.dst_addr = argv[optind];
		rpc_client = true;
	} else {
		rpc_client = false;
	}

	hints->caps = FI_RMA | FI_MSG | FI_TAGGED;
	hints->ep_attr->type = FI_EP_RDM;
	hints->domain_attr->resource_mgmt = FI_RM_ENABLED;
	hints->domain_attr->mr_mode = opts.mr_mode;
	hints->addr_format = opts.address_format;
	hints->ep_attr->mem_tag_format = RDM_RPC_TAG_FORMAT;

	ret = ft_init_fabric();
	if (ret) {
		FT_ERR("ft_init_fabric failed: %d", ret);
		return EXIT_FAILURE;
	}

	printf("RPCs Inflight: %d\n", opts.window_size);
	printf("RPC Warmup Iterations: %d\n", opts.warmup_iterations);
	printf("HMEM Type: %s\n", fi_tostr(&opts.iface, FI_TYPE_HMEM_IFACE));
	printf("MR mode: %s\n", fi_tostr(&fi->domain_attr->mr_mode,
					 FI_TYPE_MR_MODE));
	printf("Provider: %s\n", fi->fabric_attr->prov_name);

	for (rpc_op = 0; rpc_op < RPC_RESP; rpc_op++) {
		if (!(opts.options & FT_OPT_SIZE)) {
			for (i = 0; i < TEST_CNT; i++) {
				if (!ft_use_size(i, opts.sizes_enabled))
					continue;
				opts.transfer_size = test_size[i].size;
				rpc_run();
			}
		} else {
			rpc_run();
		}
	}

	ft_free_res();

	return EXIT_SUCCESS;
}
