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


extern struct fi_provider	tcpx_prov;
extern struct util_prov		tcpx_util_prov;
extern struct fi_info		tcpx_info;
struct tcpx_fabric;
struct tcpx_domain;
struct tcpx_pe_entry;
struct tcpx_progress;
struct tcpx_ep;
struct tcpx_op_send;

#define TCPX_NO_COMPLETION	(1ULL << 63)

#define POLL_MGR_FREE		(1 << 0)
#define POLL_MGR_DEL		(1 << 1)
#define POLL_MGR_ACK		(1 << 2)

#define TCPX_MAX_CM_DATA_SIZE	(1<<8)
#define TCPX_PE_COMM_BUFF_SZ	(1<<10)
#define TCPX_MAX_SOCK_REQS	(1<<10)
#define TCPX_PE_MAX_ENTRIES	(128)
#define TCPX_IOV_LIMIT		(4)
#define TCPX_MAX_INJECT_SZ	(64)
#define TCPX_MAX_EPOLL_EVENTS	(100)
#define TCPX_MAX_EP_RB_SIZE     (1024*sizeof(struct tcpx_op_send))

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

int tcpx_conn_mgr_init(struct tcpx_fabric *tcpx_fabric);
void tcpx_conn_mgr_close(struct tcpx_fabric *tcpx_fabric);
int tcpx_recv_msg(struct tcpx_pe_entry *pe_entry);
int tcpx_send_msg(struct tcpx_pe_entry *pe_entry);
void posted_rx_find(struct tcpx_pe_entry *pe_entry);
int tcpx_progress_init(struct tcpx_progress *progress);
int tcpx_progress_close(struct tcpx_progress *progress);
struct tcpx_pe_entry *pe_entry_alloc(struct tcpx_progress *progress);
void pe_entry_release(struct tcpx_pe_entry *pe_entry);
void tcpx_progress(struct util_ep *util_ep);

enum tcpx_xfer_states {
	TCPX_XFER_IDLE,
	TCPX_XFER_STARTED,
	TCPX_XFER_HDR_SENT,
	TCPX_XFER_FLUSH_COMM_BUF,
	TCPX_XFER_HDR_RECVD,
	TCPX_XFER_COMPLETE,
};

enum tcpx_xfer_op_codes {
	TCPX_OP_MSG_SEND,
	TCPX_OP_MSG_RECV,
};

enum tcpx_xfer_field {
	TCPX_MSG_HDR_FIELD,
	TCPX_DATA_FIELD,
};

enum poll_fd_type {
	CONNECT_SOCK,
	PASSIVE_SOCK,
	ACCEPT_SOCK,
};

enum poll_fd_state {
	ESTABLISH_CONN,
	RCV_RESP,
	CONNECT_DONE,
};

struct poll_fd_info {
	fid_t			fid;
	struct dlist_entry	entry;
	int			flags;
	enum poll_fd_type	type;
	enum poll_fd_state	state;
	size_t			cm_data_sz;
	char			cm_data[TCPX_MAX_CM_DATA_SIZE];
};

struct poll_fd_mgr {
	struct fd_signal	signal;
	struct dlist_entry	list;
	fastlock_t		lock;
	int			run;

	struct pollfd		*poll_fds;
	struct poll_fd_info	*poll_info;
	int			nfds;
	int			max_nfds;
};

struct tcpx_conn_handle {
	struct fid		handle;
	SOCKET			conn_fd;
};

struct tcpx_pep {
	struct util_pep 	util_pep;
	struct fi_info		info;
	SOCKET			sock;
	struct poll_fd_info	poll_info;
};

struct tcpx_ep {
	struct util_ep		util_ep;
	SOCKET			conn_fd;
	struct dlist_entry	ep_entry;
	struct dlist_entry	rx_queue;
	struct dlist_entry	tx_queue;
};

struct tcpx_fabric {
	struct util_fabric	util_fabric;
	struct poll_fd_mgr	poll_mgr;
	pthread_t		conn_mgr_thread;
};

struct tcpx_msg_data {
	size_t		iov_cnt;
	union {
		struct iovec		iov[TCPX_IOV_LIMIT+1];
		struct fi_rma_iov	rma_iov[TCPX_IOV_LIMIT+1];
		struct fi_rma_ioc	ram_ioc[TCPX_IOV_LIMIT+1];
	};
	uint8_t			inject[TCPX_MAX_INJECT_SZ];
};

struct tcpx_pe_entry {
	struct ofi_op_hdr	msg_hdr;
	struct tcpx_msg_data	msg_data;
	struct dlist_entry	entry;
	struct tcpx_ep		*ep;
	uint64_t		flags;
	void			*context;
	uint64_t		done_len;
};

struct tcpx_progress {
	struct util_buf_pool	*pe_entry_pool;
};

struct tcpx_domain {
	struct util_domain	util_domain;
	struct tcpx_progress	progress;
};

#endif //_TCP_H_
