/*
 * Copyright (c) 2017 Intel Corporation, Inc.  All rights reserved.
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
#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <sys/types.h>
#include <netdb.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/ip.h>

#include <rdma/fabric.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_trigger.h>

#include <ofi.h>
#include <ofi_enosys.h>
#include <ofi_rbuf.h>
#include <ofi_list.h>
#include <ofi_signal.h>
#include <ofi_util.h>
#include <ofi_proto.h>

#ifndef _TCP_H_
#define _TCP_H_

#define TCPX_MAJOR_VERSION 0
#define TCPX_MINOR_VERSION 1

#define TCPX_MAX_CM_DATA_SIZE	(1<<8)
#define TCPX_IOV_LIMIT		(4)
#define TCPX_MAX_INJECT_SZ	(64)

#define MAX_EPOLL_EVENTS	100
#define STAGE_BUF_SIZE		512

extern struct fi_provider	tcpx_prov;
extern struct util_prov		tcpx_util_prov;
extern struct fi_info		tcpx_info;
struct tcpx_xfer_entry;
struct tcpx_ep;

enum tcpx_xfer_op_codes {
	TCPX_OP_MSG_SEND,
	TCPX_OP_MSG_RECV,
	TCPX_OP_MSG_RESP,
	TCPX_OP_WRITE,
	TCPX_OP_REMOTE_WRITE,
	TCPX_OP_READ_REQ,
	TCPX_OP_READ_RSP,
	TCPX_OP_REMOTE_READ,
	TCPX_OP_CODE_MAX,
};

enum tcpx_cm_event_type {
	SERVER_SOCK_ACCEPT,
	CLIENT_SEND_CONNREQ,
	SERVER_RECV_CONNREQ,
	SERVER_SEND_CM_ACCEPT,
	CLIENT_RECV_CONNRESP,
};

struct tcpx_cm_context {
	fid_t			fid;
	enum tcpx_cm_event_type	type;
	size_t			cm_data_sz;
	char			cm_data[TCPX_MAX_CM_DATA_SIZE];
};

struct tcpx_conn_handle {
	struct fid		handle;
	struct tcpx_pep		*pep;
	SOCKET			conn_fd;
};

struct tcpx_pep {
	struct util_pep 	util_pep;
	struct fi_info		info;
	SOCKET			sock;
	struct tcpx_cm_context	cm_ctx;
};

enum tcpx_cm_state {
	TCPX_EP_CONNECTING,
	TCPX_EP_CONNECTED,
	TCPX_EP_SHUTDOWN,
	TCPX_EP_ERROR,
};

struct tcpx_msg_hdr {
	struct ofi_op_hdr	hdr;
	size_t			rma_iov_cnt;
	union {
		struct fi_rma_iov	rma_iov[TCPX_IOV_LIMIT];
		struct fi_rma_ioc	rma_ioc[TCPX_IOV_LIMIT];
	};
};

struct tcpx_rx_detect {
	struct tcpx_msg_hdr	hdr;
	uint64_t		done_len;
};

struct tcpx_rx_ctx {
	struct fid_ep		rx_fid;
	struct slist		rx_queue;
	struct util_buf_pool	*buf_pool;
	fastlock_t		lock;
};

typedef int (*tcpx_rx_process_fn_t)(struct tcpx_xfer_entry *rx_entry);
typedef void (*tcpx_ep_progress_func_t)(struct tcpx_ep *ep);
typedef int (*tcpx_get_rx_func_t)(struct tcpx_ep *ep);

struct stage_buf {
	uint8_t			buf[STAGE_BUF_SIZE];
	size_t			size;
	size_t			len;
	size_t			off;
};

struct tcpx_ep {
	struct util_ep		util_ep;
	SOCKET			conn_fd;
	struct tcpx_rx_detect	rx_detect;
	struct tcpx_xfer_entry	*cur_rx_entry;
	tcpx_rx_process_fn_t 	cur_rx_proc_fn;
	struct dlist_entry	ep_entry;
	struct slist		rx_queue;
	struct slist		tx_queue;
	struct slist		tx_rsp_pend_queue;
	struct slist		rma_read_queue;
	struct tcpx_rx_ctx	*srx_ctx;
	enum tcpx_cm_state	cm_state;
	/* lock for protecting tx/rx queues,rma list,cm_state*/
	fastlock_t		lock;
	tcpx_ep_progress_func_t progress_func;
	tcpx_get_rx_func_t	get_rx_entry[ofi_op_write + 1];
	struct stage_buf	stage_buf;
	bool			send_ready_monitor;
};

struct tcpx_fabric {
	struct util_fabric	util_fabric;
};

struct tcpx_msg_data {
	size_t			iov_cnt;
	struct iovec		iov[TCPX_IOV_LIMIT+1];
	uint8_t			inject[TCPX_MAX_INJECT_SZ];
};

struct tcpx_xfer_entry {
	struct slist_entry	entry;
	struct tcpx_msg_hdr	msg_hdr;
	struct tcpx_msg_data	msg_data;
	struct tcpx_ep		*ep;
	uint64_t		flags;
	void			*context;
	uint64_t		done_len;
};

struct tcpx_domain {
	struct util_domain	util_domain;
};

struct tcpx_buf_pool {
	struct util_buf_pool	*pool;
	enum tcpx_xfer_op_codes	op_type;
};

struct tcpx_cq {
	struct util_cq		util_cq;
	/* buf_pools protected by util.cq_lock */
	struct tcpx_buf_pool	buf_pools[TCPX_OP_CODE_MAX];
};

int tcpx_create_fabric(struct fi_fabric_attr *attr,
		       struct fid_fabric **fabric,
		       void *context);

int tcpx_passive_ep(struct fid_fabric *fabric, struct fi_info *info,
		    struct fid_pep **pep, void *context);

int tcpx_domain_open(struct fid_fabric *fabric, struct fi_info *info,
		     struct fid_domain **domain, void *context);


int tcpx_endpoint(struct fid_domain *domain, struct fi_info *info,
		  struct fid_ep **ep_fid, void *context);


int tcpx_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		 struct fid_cq **cq_fid, void *context);
void tcpx_cq_report_completion(struct util_cq *cq,
			       struct tcpx_xfer_entry *xfer_entry,
			       int err);

int tcpx_recv_msg_data(struct tcpx_xfer_entry *recv_entry);
int tcpx_send_msg(struct tcpx_xfer_entry *tx_entry);
int tcpx_recv_hdr(SOCKET sock, struct stage_buf *sbuf,
		  struct tcpx_rx_detect *rx_detect);
int tcpx_read_to_buffer(SOCKET sock, struct stage_buf *stage_buf);

struct tcpx_xfer_entry *tcpx_xfer_entry_alloc(struct tcpx_cq *cq,
					      enum tcpx_xfer_op_codes type);
void tcpx_xfer_entry_release(struct tcpx_cq *tcpx_cq,
			     struct tcpx_xfer_entry *xfer_entry);

void tcpx_progress(struct util_ep *util_ep);
void tcpx_ep_progress(struct tcpx_ep *ep);
int tcpx_ep_shutdown_report(struct tcpx_ep *ep, fid_t fid);
int tcpx_cq_wait_ep_add(struct tcpx_ep *ep);
void tcpx_cq_wait_ep_del(struct tcpx_ep *ep);
void tcpx_tx_queue_insert(struct tcpx_ep *tcpx_ep,
			  struct tcpx_xfer_entry *tx_entry);

void tcpx_conn_mgr_run(struct util_eq *eq);
int tcpx_eq_wait_try_func(void *arg);
int tcpx_eq_create(struct fid_fabric *fabric_fid, struct fi_eq_attr *attr,
		   struct fid_eq **eq_fid, void *context);

int tcpx_get_rx_entry_op_invalid(struct tcpx_ep *tcpx_ep);
int tcpx_get_rx_entry_op_msg(struct tcpx_ep *tcpx_ep);
int tcpx_get_rx_entry_op_read_req(struct tcpx_ep *tcpx_ep);
int tcpx_get_rx_entry_op_write(struct tcpx_ep *tcpx_ep);
int tcpx_get_rx_entry_op_read_rsp(struct tcpx_ep *tcpx_ep);

#endif //_TCP_H_
