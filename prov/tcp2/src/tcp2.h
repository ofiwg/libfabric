/*
 * Copyright (c) 2017-2022 Intel Corporation, Inc.  All rights reserved.
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
#include <ofi_net.h>

#include "tcp2_proto.h"

#ifndef _TCP2_H_
#define _TCP2_H_


#define TCP2_RDM_VERSION	0
#define TCP2_MAX_INJECT		128
#define TCP2_MAX_EVENTS		1024
#define TCP2_MIN_MULTI_RECV	16384
#define TCP2_PORT_MAX_RANGE	(USHRT_MAX)

extern struct fi_provider	tcp2_prov;
extern struct util_prov		tcp2_util_prov;
extern struct fi_info		tcp2_info;
extern struct tcp2_port_range	tcp2_ports;

extern int tcp2_nodelay;
extern int tcp2_staging_sbuf_size;
extern int tcp2_prefetch_rbuf_size;
extern size_t tcp2_default_tx_size;
extern size_t tcp2_default_rx_size;
extern size_t tcp2_zerocopy_size;

struct tcp2_xfer_entry;
struct tcp2_ep;
struct tcp2_progress;
struct tcp2_domain;


/* Lock ordering:
 * progress->list_lock - protects against rdm destruction
 * rdm->lock - protects rdm_conn lookup and access
 * progress->lock - serializes ep connection, transfers, destruction
 * cq->lock or eq->lock - protects event queues
 * TODO: simplify locking now that progress locks are available
 */

enum tcp2_state {
	TCP2_IDLE,
	TCP2_CONNECTING,
	TCP2_ACCEPTING,
	TCP2_REQ_SENT,
	TCP2_CONNECTED,
	TCP2_DISCONNECTED,
	TCP2_LISTENING,
};

#define OFI_PROV_SPECIFIC_TCP (0x7cb << 16)
enum {
	TCP2_CLASS_CM = OFI_PROV_SPECIFIC_TCP,
	TCP2_CLASS_PROGRESS,
};

struct tcp2_port_range {
	int high;
	int low;
};

struct tcp2_conn_handle {
	struct fid		fid;
	struct tcp2_pep		*pep;
	SOCKET			sock;
	bool			endian_match;
};

struct tcp2_pep {
	struct util_pep 	util_pep;
	struct fi_info		*info;
	struct tcp2_progress	*progress;
	SOCKET			sock;
	enum tcp2_state		state;
};

int tcp2_listen(struct tcp2_pep *pep, struct tcp2_progress *progress);
void tcp2_accept_sock(struct tcp2_pep *pep);
void tcp2_connect_done(struct tcp2_ep *ep);
void tcp2_req_done(struct tcp2_ep *ep);
int tcp2_send_cm_msg(struct tcp2_ep *ep);

struct tcp2_cur_rx {
	union {
		struct tcp2_base_hdr	base_hdr;
		struct tcp2_cq_data_hdr cq_data_hdr;
		struct tcp2_tag_data_hdr tag_data_hdr;
		struct tcp2_tag_hdr	tag_hdr;
		uint8_t			max_hdr[TCP2_MAX_HDR];
	} hdr;
	size_t			hdr_len;
	size_t			hdr_done;
	size_t			data_left;
	struct tcp2_xfer_entry	*entry;
	ssize_t			(*handler)(struct tcp2_ep *ep);
};

struct tcp2_cur_tx {
	size_t			data_left;
	struct tcp2_xfer_entry	*entry;
};

struct tcp2_srx {
	struct fid_ep		rx_fid;
	struct tcp2_domain	*domain;
	struct tcp2_cq		*cq;
	struct slist		rx_queue;
	struct slist		tag_queue;
	struct tcp2_xfer_entry	*(*match_tag_rx)(struct tcp2_srx *srx,
						 struct tcp2_ep *ep,
						 uint64_t tag);

	struct ofi_bufpool	*buf_pool;
	uint64_t		op_flags;
};

int tcp2_srx_context(struct fid_domain *domain, struct fi_rx_attr *attr,
		     struct fid_ep **rx_ep, void *context);

struct tcp2_ep {
	struct util_ep		util_ep;
	struct ofi_bsock	bsock;
	struct tcp2_cur_rx	cur_rx;
	struct tcp2_cur_tx	cur_tx;
	OFI_DBG_VAR(uint8_t, tx_id)
	OFI_DBG_VAR(uint8_t, rx_id)

	struct dlist_entry	progress_entry; /* protected by progress->lock */
	struct slist		rx_queue;
	struct slist		tx_queue;
	struct slist		priority_queue;
	struct slist		need_ack_queue;
	struct slist		async_queue;
	struct slist		rma_read_queue;
	int			rx_avail;
	struct tcp2_srx		*srx;

	enum tcp2_state		state;
	fi_addr_t		src_addr;
	struct tcp2_conn_handle *conn;
	struct tcp2_cm_msg	*cm_msg;

	void (*hdr_bswap)(struct tcp2_base_hdr *hdr);
	void (*report_success)(struct tcp2_ep *ep, struct util_cq *cq,
			       struct tcp2_xfer_entry *xfer_entry);
	size_t			min_multi_recv_size;
	bool			pollout_set;
};

struct tcp2_event {
	struct slist_entry list_entry;
	uint32_t event;
	struct fi_eq_cm_entry cm_entry;
};

enum {
	TCP2_CONN_INDEXED = BIT(0),
};

struct tcp2_conn {
	struct tcp2_ep		*ep;
	struct tcp2_rdm		*rdm;
	struct util_peer_addr	*peer;
	uint32_t		remote_pid;
	int			flags;
	struct dlist_entry	loopback_entry;
};

struct tcp2_rdm {
	struct util_ep		util_ep;

	struct tcp2_pep		*pep;
	struct tcp2_srx		*srx;

	struct index_map	conn_idx_map;
	struct dlist_entry	loopback_list;
	union ofi_sock_ip	addr;

	struct slist		event_list; /* protected by progress lock */
	struct dlist_entry	progress_entry;
};

int tcp2_rdm_ep(struct fid_domain *domain, struct fi_info *info,
		struct fid_ep **ep_fid, void *context);
ssize_t tcp2_get_conn(struct tcp2_rdm *rdm, fi_addr_t dest_addr,
		      struct tcp2_conn **conn);
void tcp2_freeall_conns(struct tcp2_rdm *rdm);

struct tcp2_progress {
	struct fid		fid;
	struct ofi_genlock	lock;
	struct ofi_genlock	rdm_lock;
	struct ofi_genlock	*active_lock;

	struct dlist_entry	active_wait_list;
	struct fd_signal	signal;

	struct dlist_entry	event_list;
	uint32_t		event_cnt;
	struct ofi_bufpool	*xfer_pool;

	/* epoll works better for apps that wait on the fd,
	 * but tests show that poll performs better
	 */
	bool			use_epoll;
	union {
		struct ofi_pollfds *pollfds;
		ofi_epoll_t	epoll;
	};

	int (*poll_wait)(struct tcp2_progress *progress,
			struct ofi_epollfds_event *events, int max_events,
			int timeout);
	int (*poll_add)(struct tcp2_progress *progress, int fd, uint32_t events,
			void *context);
	void (*poll_mod)(struct tcp2_progress *progress, int fd, uint32_t events,
			void *context);
	int (*poll_del)(struct tcp2_progress *progress, int fd);
	void (*poll_close)(struct tcp2_progress *progress);

	pthread_t		thread;
	bool			auto_progress;
};

int tcp2_init_progress(struct tcp2_progress *progress, struct fi_info *info);
void tcp2_close_progress(struct tcp2_progress *progress);
int tcp2_start_progress(struct tcp2_progress *progress);
void tcp2_stop_progress(struct tcp2_progress *progress);

void tcp2_run_progress(struct tcp2_progress *progress, bool internal);
void tcp2_run_conn(struct tcp2_conn_handle *conn, bool pin, bool pout, bool perr);
void tcp2_handle_events(struct tcp2_progress *progress);

int tcp2_trywait(struct fid_fabric *fid_fabric, struct fid **fids, int count);
void tcp2_update_poll(struct tcp2_ep *ep);
int tcp2_monitor_sock(struct tcp2_progress *progress, SOCKET sock,
		      uint32_t events, struct fid *fid);
void tcp2_halt_sock(struct tcp2_progress *progress, SOCKET sock);

static inline int tcp2_progress_locked(struct tcp2_progress *progress)
{
	return ofi_genlock_held(progress->active_lock);
}


struct tcp2_fabric {
	struct util_fabric	util_fabric;
	struct tcp2_progress	progress;
};

int tcp2_start_all(struct tcp2_fabric *fabric);
void tcp2_progress_all(struct tcp2_fabric *fabric);

static inline void tcp2_signal_progress(struct tcp2_progress *progress)
{
	if (progress->auto_progress)
		fd_signal_set(&progress->signal);
}


/* tcp2_xfer_entry::ctrl_flags */
#define TCP2_NEED_RESP		BIT(1)
#define TCP2_NEED_ACK		BIT(2)
#define TCP2_INTERNAL_XFER	BIT(3)
#define TCP2_NEED_DYN_RBUF 	BIT(4)
#define TCP2_ASYNC		BIT(5)
#define TCP2_INJECT_OP		BIT(6)

struct tcp2_xfer_entry {
	struct slist_entry	entry;
	union {
		struct tcp2_base_hdr	base_hdr;
		struct tcp2_cq_data_hdr cq_data_hdr;
		struct tcp2_tag_data_hdr tag_data_hdr;
		struct tcp2_tag_hdr	tag_hdr;
		uint8_t		       	max_hdr[TCP2_MAX_HDR + TCP2_MAX_INJECT];
	} hdr;
	size_t			iov_cnt;
	struct iovec		iov[TCP2_IOV_LIMIT+1];
	struct tcp2_ep		*ep;
	uint64_t		tag;
	uint64_t		ignore;
	fi_addr_t		src_addr;
	uint64_t		cq_flags;
	uint32_t		ctrl_flags;
	uint32_t		async_index;
	void			*context;
	void			*mrecv_msg_start;
	// for RMA read requests, we need a way to track the request response
	// so that we don't propagate multiple completions for the same operation
	struct tcp2_xfer_entry  *resp_entry;
};

struct tcp2_domain {
	struct util_domain		util_domain;
	struct tcp2_progress		progress;
};

static inline struct tcp2_progress *tcp2_ep2_progress(struct tcp2_ep *ep)
{
	struct tcp2_domain *domain;
	domain = container_of(ep->util_ep.domain, struct tcp2_domain,
			      util_domain);
	return &domain->progress;
}

static inline struct tcp2_progress *tcp2_rdm2_progress(struct tcp2_rdm *rdm)
{
	struct tcp2_domain *domain;
	domain = container_of(rdm->util_ep.domain, struct tcp2_domain,
			      util_domain);
	return &domain->progress;
}

static inline struct tcp2_progress *tcp2_srx2_progress(struct tcp2_srx *srx)
{
	return &srx->domain->progress;
}

struct tcp2_cq {
	struct util_cq		util_cq;
	struct ofi_bufpool	*xfer_pool;
};

static inline struct tcp2_progress *tcp2_cq2_progress(struct tcp2_cq *cq)
{
	struct tcp2_domain *domain;
	domain = container_of(cq->util_cq.domain, struct tcp2_domain,
			      util_domain);
	return &domain->progress;
}

/* tcp2_cntr maps directly to util_cntr */

static inline struct tcp2_progress *tcp2_cntr2_progress(struct util_cntr *cntr)
{
	struct tcp2_domain *domain;
	domain = container_of(cntr->domain, struct tcp2_domain, util_domain);
	return &domain->progress;
}

struct tcp2_eq {
	struct util_eq		util_eq;
	/*
	  The following lock avoids race between ep close
	  and connection management code.
	 */
	ofi_mutex_t		close_lock;
};

static inline struct tcp2_progress *tcp2_eq2_progress(struct tcp2_eq *eq)
{
	struct tcp2_fabric *fabric ;
	fabric = container_of(eq->util_eq.fabric, struct tcp2_fabric,
			      util_fabric);
	return &fabric->progress;
}

int tcp2_eq_write(struct util_eq *eq, uint32_t event,
		  const void *buf, size_t len, uint64_t flags);

int tcp2_create_fabric(struct fi_fabric_attr *attr,
		       struct fid_fabric **fabric,
		       void *context);

int tcp2_passive_ep(struct fid_fabric *fabric, struct fi_info *info,
		    struct fid_pep **pep, void *context);

int tcp2_set_port_range(void);

int tcp2_domain_open(struct fid_fabric *fabric, struct fi_info *info,
		     struct fid_domain **domain, void *context);


int tcp2_setup_socket(SOCKET sock, struct fi_info *info);
void tcp2_set_zerocopy(SOCKET sock);

int tcp2_endpoint(struct fid_domain *domain, struct fi_info *info,
		  struct fid_ep **ep_fid, void *context);
void tcp2_ep_disable(struct tcp2_ep *ep, int cm_err, void* err_data,
		     size_t err_data_size);


int tcp2_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		 struct fid_cq **cq_fid, void *context);
void tcp2_report_success(struct tcp2_ep *ep, struct util_cq *cq,
			 struct tcp2_xfer_entry *xfer_entry);
void tcp2_cq_report_error(struct util_cq *cq,
			  struct tcp2_xfer_entry *xfer_entry,
			  int err);
void tcp2_get_cq_info(struct tcp2_xfer_entry *entry, uint64_t *flags,
		      uint64_t *data, uint64_t *tag);
int tcp2_cntr_open(struct fid_domain *fid_domain, struct fi_cntr_attr *attr,
		   struct fid_cntr **cntr_fid, void *context);
void tcp2_report_cntr_success(struct tcp2_ep *ep, struct util_cq *cq,
			      struct tcp2_xfer_entry *xfer_entry);
void tcp2_cntr_incerr(struct tcp2_ep *ep, struct tcp2_xfer_entry *xfer_entry);

void tcp2_reset_rx(struct tcp2_ep *ep);

void tcp2_progress_rx(struct tcp2_ep *ep);
void tcp2_progress_async(struct tcp2_ep *ep);

void tcp2_hdr_none(struct tcp2_base_hdr *hdr);
void tcp2_hdr_bswap(struct tcp2_base_hdr *hdr);

void tcp2_tx_queue_insert(struct tcp2_ep *ep,
			  struct tcp2_xfer_entry *tx_entry);

int tcp2_eq_create(struct fid_fabric *fabric_fid, struct fi_eq_attr *attr,
		   struct fid_eq **eq_fid, void *context);


static inline void
tcp2_set_ack_flags(struct tcp2_xfer_entry *xfer, uint64_t flags)
{
	if (flags & (FI_TRANSMIT_COMPLETE | FI_DELIVERY_COMPLETE)) {
		xfer->hdr.base_hdr.flags |= TCP2_DELIVERY_COMPLETE;
		xfer->ctrl_flags |= TCP2_NEED_ACK;
	}
}

static inline void
tcp2_set_commit_flags(struct tcp2_xfer_entry *xfer, uint64_t flags)
{
	tcp2_set_ack_flags(xfer, flags);
	if (flags & FI_COMMIT_COMPLETE) {
		xfer->hdr.base_hdr.flags |= TCP2_COMMIT_COMPLETE;
		xfer->ctrl_flags |= TCP2_NEED_ACK;
	}
}

static inline uint64_t
tcp2_tx_completion_flag(struct tcp2_ep *ep, uint64_t op_flags)
{
	/* Generate a completion if op flags indicate or we generate
	 * completions by default
	 */
	return (ep->util_ep.tx_op_flags | op_flags) & FI_COMPLETION;
}

static inline uint64_t
tcp2_rx_completion_flag(struct tcp2_ep *ep, uint64_t op_flags)
{
	/* Generate a completion if op flags indicate or we generate
	 * completions by default
	 */
	return (ep->util_ep.rx_op_flags | op_flags) & FI_COMPLETION;
}

static inline struct tcp2_xfer_entry *
tcp2_alloc_xfer(struct tcp2_progress *progress)
{
	assert(tcp2_progress_locked(progress));
	return ofi_buf_alloc(progress->xfer_pool);
}

static inline void
tcp2_free_xfer(struct tcp2_ep *ep, struct tcp2_xfer_entry *xfer)
{
	assert(tcp2_progress_locked(tcp2_ep2_progress(ep)));
	xfer->hdr.base_hdr.flags = 0;
	xfer->cq_flags = 0;
	xfer->ctrl_flags = 0;
	xfer->context = 0;
	ofi_buf_free(xfer);
}

static inline struct tcp2_xfer_entry *
tcp2_alloc_rx(struct tcp2_ep *ep)
{
	struct tcp2_xfer_entry *xfer;

	assert(tcp2_progress_locked(tcp2_ep2_progress(ep)));
	xfer = tcp2_alloc_xfer(tcp2_ep2_progress(ep));
	if (xfer)
		xfer->ep = ep;

	return xfer;
}

static inline struct tcp2_xfer_entry *
tcp2_alloc_tx(struct tcp2_ep *ep)
{
	struct tcp2_xfer_entry *xfer;

	assert(tcp2_progress_locked(tcp2_ep2_progress(ep)));
	xfer = tcp2_alloc_xfer(tcp2_ep2_progress(ep));
	if (xfer) {
		xfer->hdr.base_hdr.version = TCP2_HDR_VERSION;
		xfer->hdr.base_hdr.op_data = 0;
		xfer->ep = ep;
	}

	return xfer;
}

/* If we've buffered receive data, it counts the same as if a POLLIN
 * event were set, and we need to process the data.
 * We also need to progress receives in the case where we're waiting
 * on the application to post a buffer to consume a receive
 * that we've already read from the kernel.  If the message is
 * of length 0, there's no additional data to read, so calling
 * poll without forcing progress can result in application hangs.
 */
static inline bool tcp2_active_wait(struct tcp2_ep *ep)
{
	assert(tcp2_progress_locked(tcp2_ep2_progress(ep)));
	return ofi_bsock_readable(&ep->bsock) ||
	       (ep->cur_rx.handler && !ep->cur_rx.entry);
}

#define TCP2_WARN_ERR(subsystem, log_str, err) \
	FI_WARN(&tcp2_prov, subsystem, log_str "%s (%d)\n", \
		fi_strerror((int) -(err)), (int) err)

#endif //_TCP2_H_
