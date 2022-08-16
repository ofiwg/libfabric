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

#ifndef _RPC_H_
#define _RPC_H_

#include <rdma/fabric.h>

#define MAX_RPCS_INFLIGHT 256

/* Input test control */
enum rpc_op {
	op_noop,
	op_hello,
	op_goodbye,
	op_msg_req,
	op_msg_inject_req,
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
	bool pending;
};

extern int rpc_timeout;
extern const uint32_t invalid_id;
extern fi_addr_t server_addr;
extern uint32_t myid;
extern uint32_t id_at_server;
extern bool enable_rpc_output;
extern bool data_verification;

#define RPC_PRINTF(fmt, ...)						\
	do {								\
		if (enable_rpc_output)					\
			printf(fmt, ##__VA_ARGS__);			\
	} while (0);

/* Wire protocol */
enum {
	cmd_hello,
	cmd_goodbye,
	cmd_msg,
	cmd_msg_inject,
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
	uint64_t cookie;
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

char *rpc_cmd_str(uint32_t cmd);
char *rpc_op_str(uint32_t op);
int rpc_op_exec(enum rpc_op op, struct rpc_ctrl *ctrl);
void init_rpc_ctrl(struct rpc_ctrl *ctrl);
int rpc_run_server(int threads);

#endif /* _RPC_H_ */
