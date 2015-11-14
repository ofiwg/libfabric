/*
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
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

#ifndef _FI_PSM2_H
#define _FI_PSM2_H

#ifdef __cplusplus
extern "C" {
#endif

#include "config.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <complex.h>
#include <rdma/fabric.h>
#include <rdma/fi_prov.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_trigger.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_log.h>
#include "fi.h"
#include "fi_enosys.h"
#include "fi_list.h"
#include "fi_indexer.h"
#include "version.h"

extern struct fi_provider psmx2_prov;

#define PSMX2_OP_FLAGS	(FI_INJECT | FI_MULTI_RECV | FI_COMPLETION | \
			 FI_TRIGGER | FI_INJECT_COMPLETE | \
			 FI_TRANSMIT_COMPLETE | FI_DELIVERY_COMPLETE)

#define PSMX2_CAPS	(FI_TAGGED | FI_MSG | FI_ATOMICS | \
			 FI_RMA | FI_MULTI_RECV | \
                         FI_READ | FI_WRITE | FI_SEND | FI_RECV | \
                         FI_REMOTE_READ | FI_REMOTE_WRITE | \
			 FI_TRIGGER | FI_RMA_EVENT | \
			 FI_REMOTE_CQ_DATA | FI_SOURCE | FI_DIRECTED_RECV)

#define PSMX2_SUB_CAPS	(FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE | \
			 FI_SEND | FI_RECV)

#define PSMX2_MAX_MSG_SIZE	((0x1ULL << 32) - 1)
#define PSMX2_INJECT_SIZE	(64)
#define PSMX2_MSG_ORDER		FI_ORDER_SAS
#define PSMX2_COMP_ORDER	FI_ORDER_NONE

#define PSMX2_MSG_BIT	(0x1ULL << 63)
#define PSMX2_RMA_BIT	(0x1ULL << 62)

/* Bits 60 .. 63 of the flag are provider specific */
#define PSMX2_NO_COMPLETION	(1ULL << 60)

enum psmx2_context_type {
	PSMX2_NOCOMP_SEND_CONTEXT = 1,
	PSMX2_NOCOMP_RECV_CONTEXT,
	PSMX2_NOCOMP_WRITE_CONTEXT,
	PSMX2_NOCOMP_READ_CONTEXT,
	PSMX2_SEND_CONTEXT,
	PSMX2_RECV_CONTEXT,
	PSMX2_MULTI_RECV_CONTEXT,
	PSMX2_TSEND_CONTEXT,
	PSMX2_TRECV_CONTEXT,
	PSMX2_WRITE_CONTEXT,
	PSMX2_READ_CONTEXT,
	PSMX2_REMOTE_WRITE_CONTEXT,
	PSMX2_REMOTE_READ_CONTEXT,
};

union psmx2_pi {
	void	*p;
	int	i;
};

#define PSMX2_CTXT_REQ(fi_context)	((fi_context)->internal[0])
#define PSMX2_CTXT_TYPE(fi_context)	(((union psmx2_pi *)&(fi_context)->internal[1])->i)
#define PSMX2_CTXT_USER(fi_context)	((fi_context)->internal[2])
#define PSMX2_CTXT_EP(fi_context)	((fi_context)->internal[3])

#define PSMX2_SET_TAG(tag96,tag64,tag32)	do { \
						tag96.tag0 = (uint32_t)tag64; \
						tag96.tag1 = (uint32_t)(tag64>>32); \
						tag96.tag2 = tag32; \
					} while (0)

#define PSMX2_AM_RMA_HANDLER	0
#define PSMX2_AM_MSG_HANDLER	1
#define PSMX2_AM_ATOMIC_HANDLER	2
#define PSMX2_AM_CHUNK_SIZE	2032	/* The maximum that's actually working:
					 * 2032 for inter-node, 2072 for intra-node.
					 */

#define PSMX2_AM_OP_MASK		0x0000FFFF
#define PSMX2_AM_FLAG_MASK	0xFFFF0000
#define PSMX2_AM_EOM		0x40000000
#define PSMX2_AM_DATA		0x20000000
#define PSMX2_AM_FORCE_ACK	0x10000000

enum {
	PSMX2_AM_REQ_WRITE = 1,
	PSMX2_AM_REQ_WRITE_LONG,
	PSMX2_AM_REP_WRITE,
	PSMX2_AM_REQ_READ,
	PSMX2_AM_REQ_READ_LONG,
	PSMX2_AM_REP_READ,
	PSMX2_AM_REQ_SEND,
	PSMX2_AM_REP_SEND,
	PSMX2_AM_REQ_ATOMIC_WRITE,
	PSMX2_AM_REP_ATOMIC_WRITE,
	PSMX2_AM_REQ_ATOMIC_READWRITE,
	PSMX2_AM_REP_ATOMIC_READWRITE,
	PSMX2_AM_REQ_ATOMIC_COMPWRITE,
	PSMX2_AM_REP_ATOMIC_COMPWRITE,
};

struct psmx2_am_request {
	int op;
	union {
		struct {
			void	*buf;
			size_t	len;
			uint64_t addr;
			uint64_t key;
			void	*context;
			void	*peer_context;
			void	*peer_addr;
			uint64_t data;
		} write;
		struct {
			void	*buf;
			size_t	len;
			uint64_t addr;
			uint64_t key;
			void	*context;
			void	*peer_addr;
			size_t	len_read;
		} read;
		struct {
			void	*buf;
			size_t	len;
			void	*context;
			void	*peer_context;
			volatile int peer_ready;
			void	*dest_addr;
			size_t	len_sent;
		} send;
		struct {
			void	*buf;
			size_t	len;
			void	*context;
			void	*src_addr;
			size_t  len_received;
		} recv;
		struct {
			void	*buf;
			size_t	len;
			uint64_t addr;
			uint64_t key;
			void	*context;
			void 	*result;
		} atomic;
	};
	uint64_t cq_flags;
	struct fi_context fi_context;
	struct psmx2_fid_ep *ep;
	int no_event;
	int error;
	struct slist_entry list_entry;
};

struct psmx2_unexp {
	psm2_epaddr_t		sender_addr;
	uint64_t		sender_context;
	uint32_t		len_received;
	uint32_t		done;
	struct slist_entry	list_entry;
	char			buf[0];
};

struct psmx2_req_queue {
	fastlock_t	lock;
	struct slist	list;
};

struct psmx2_multi_recv {
	uint64_t	tag;
	uint64_t	tagsel;
	void		*buf;
	size_t		len;
	size_t		offset;
	int		min_buf_size;
	int		flag;
	void		*context;
};

struct psmx2_fid_fabric {
	struct fid_fabric	fabric;
	int			refcnt;
	struct psmx2_fid_domain	*active_domain;
	psm2_uuid_t		uuid;
	pthread_t		name_server_thread;
};

struct psmx2_fid_domain {
	struct fid_domain	domain;
	struct psmx2_fid_fabric	*fabric;
	int			refcnt;
	psm2_ep_t		psm2_ep;
	psm2_epid_t		psm2_epid;
	psm2_mq_t		psm2_mq;
	struct psmx2_fid_ep	*tagged_ep;
	struct psmx2_fid_ep	*msg_ep;
	struct psmx2_fid_ep	*rma_ep;
	struct psmx2_fid_ep	*atomics_ep;
	uint64_t		mode;
	uint64_t		caps;

	enum fi_mr_mode		mr_mode;
	fastlock_t		mr_lock;
	uint64_t		mr_reserved_key;
	struct index_map	mr_map;

	int			am_initialized;

	/* incoming req queue for AM based RMA request. */
	struct psmx2_req_queue	rma_queue;

	/* send queue for AM based messages. */
	struct psmx2_req_queue	send_queue;

	/* recv queue for AM based messages. */
	struct psmx2_req_queue	recv_queue;
	struct psmx2_req_queue	unexp_queue;

	/* triggered operations that are ready to be processed */
	struct psmx2_req_queue	trigger_queue;

	/* certain bits in the tag space can be reserved for non tag-matching
	 * purpose. The tag-matching functions automatically treat these bits
	 * as 0. This field is a bit mask, with reserved bits valued as "1".
	 */
	uint64_t		reserved_tag_bits; 

	/* lock to prevent the sequence of psm2_mq_ipeek and psm2_mq_test be
	 * interleaved in a multithreaded environment.
	 */
	fastlock_t		poll_lock;

	int			progress_thread_enabled;
	pthread_t		progress_thread;
};

struct psmx2_cq_event {
	union {
		struct fi_cq_entry		context;
		struct fi_cq_msg_entry		msg;
		struct fi_cq_data_entry		data;
		struct fi_cq_tagged_entry	tagged;
		struct fi_cq_err_entry		err;
	} cqe;
	int error;
	uint64_t source;
	struct slist_entry list_entry;
};

struct psmx2_eq_event {
	int event;
	union {
		struct fi_eq_entry		data;
		struct fi_eq_cm_entry		cm;
		struct fi_eq_err_entry		err;
	} eqe;
	int error;
	size_t entry_size;
	struct slist_entry list_entry;
};

struct psmx2_fid_wait {
	struct fid_wait			wait;
	struct psmx2_fid_fabric		*fabric;
	int				type;
	union {
		int			fd[2];
		struct {
			pthread_mutex_t	mutex;
			pthread_cond_t	cond;
		};
	};
};

struct psmx2_poll_list {
	struct dlist_entry		entry;
	struct fid			*fid;
};

struct psmx2_fid_poll {
	struct fid_poll			poll;
	struct psmx2_fid_domain		*domain;
	struct dlist_entry		poll_list_head;
};

struct psmx2_fid_cq {
	struct fid_cq			cq;
	struct psmx2_fid_domain		*domain;
	int 				format;
	int				entry_size;
	size_t				event_count;
	struct slist			event_queue;
	struct slist			free_list;
	fastlock_t			lock;
	struct psmx2_cq_event		*pending_error;
	struct psmx2_fid_wait		*wait;
	int				wait_cond;
	int				wait_is_local;
};

struct psmx2_fid_eq {
	struct fid_eq			eq;
	struct psmx2_fid_fabric		*fabric;
	struct slist			event_queue;
	struct slist			error_queue;
	struct slist			free_list;
	fastlock_t			lock;
	struct psmx2_fid_wait		*wait;
	int				wait_is_local;
};

enum psmx2_triggered_op {
	PSMX2_TRIGGERED_SEND,
	PSMX2_TRIGGERED_RECV,
	PSMX2_TRIGGERED_TSEND,
	PSMX2_TRIGGERED_TRECV,
	PSMX2_TRIGGERED_WRITE,
	PSMX2_TRIGGERED_READ,
	PSMX2_TRIGGERED_ATOMIC_WRITE,
	PSMX2_TRIGGERED_ATOMIC_READWRITE,
	PSMX2_TRIGGERED_ATOMIC_COMPWRITE,
};

struct psmx2_trigger {
	enum psmx2_triggered_op	op;
	struct psmx2_fid_cntr	*cntr;
	size_t			threshold;
	union {
		struct {
			struct fid_ep	*ep;
			const void	*buf;
			size_t		len;
			void		*desc;
			fi_addr_t	dest_addr;
			void		*context;
			uint64_t	flags;
			uint32_t	data;
		} send;
		struct {
			struct fid_ep	*ep;
			void		*buf;
			size_t		len;
			void		*desc;
			fi_addr_t	src_addr;
			void		*context;
			uint64_t	flags;
		} recv;
		struct {
			struct fid_ep	*ep;
			const void	*buf;
			size_t		len;
			void		*desc;
			fi_addr_t	dest_addr;
			uint64_t	tag;
			void		*context;
			uint64_t	flags;
			uint32_t	data;
		} tsend;
		struct {
			struct fid_ep	*ep;
			void		*buf;
			size_t		len;
			void		*desc;
			fi_addr_t	src_addr;
			uint64_t	tag;
			uint64_t	ignore;
			void		*context;
			uint64_t	flags;
		} trecv;
		struct {
			struct fid_ep	*ep;
			const void	*buf;
			size_t		len;
			void		*desc;
			fi_addr_t	dest_addr;
			uint64_t	addr;
			uint64_t	key;
			void		*context;
			uint64_t	flags;
			uint64_t	data;
		} write;
		struct {
			struct fid_ep	*ep;
			void		*buf;
			size_t		len;
			void		*desc;
			fi_addr_t	src_addr;
			uint64_t	addr;
			uint64_t	key;
			void		*context;
			uint64_t	flags;
		} read;
		struct {
			struct fid_ep	*ep;
			const void	*buf;
			size_t		count;
			void		*desc;
			fi_addr_t	dest_addr;
			uint64_t	addr;
			uint64_t	key;
			enum fi_datatype datatype;
			enum fi_op	atomic_op;
			void		*context;
			uint64_t	flags;
		} atomic_write;
		struct {
			struct fid_ep	*ep;
			const void	*buf;
			size_t		count;
			void		*desc;
			void		*result;
			void		*result_desc;
			fi_addr_t	dest_addr;
			uint64_t	addr;
			uint64_t	key;
			enum fi_datatype datatype;
			enum fi_op	atomic_op;
			void		*context;
			uint64_t	flags;
		} atomic_readwrite;
		struct {
			struct fid_ep	*ep;
			const void	*buf;
			size_t		count;
			void		*desc;
			const void	*compare;
			void		*compare_desc;
			void		*result;
			void		*result_desc;
			fi_addr_t	dest_addr;
			uint64_t	addr;
			uint64_t	key;
			enum fi_datatype datatype;
			enum fi_op	atomic_op;
			void		*context;
			uint64_t	flags;
		} atomic_compwrite;
	};
	struct psmx2_trigger *next;	/* used for randomly accessed trigger list */
	struct slist_entry list_entry;	/* used for ready-to-fire trigger queue */
};

struct psmx2_fid_cntr {
	struct fid_cntr		cntr;
	struct psmx2_fid_domain	*domain;
	int			events;
	uint64_t		flags;
	volatile uint64_t	counter;
	volatile uint64_t	error_counter;
	uint64_t		counter_last_read;
	uint64_t		error_counter_last_read;
	struct psmx2_fid_wait	*wait;
	int			wait_is_local;
	struct psmx2_trigger	*trigger;
	pthread_mutex_t		trigger_lock;
};

struct psmx2_fid_av {
	struct fid_av		av;
	struct psmx2_fid_domain	*domain;
	struct psmx2_fid_eq	*eq;
	int			type;
	uint64_t		flags;
	size_t			addrlen;
	size_t			count;
	size_t			last;
	psm2_epid_t		*psm2_epids;
	psm2_epaddr_t		*psm2_epaddrs;
};

struct psmx2_fid_ep {
	struct fid_ep		ep;
	struct psmx2_fid_domain	*domain;
	struct psmx2_fid_av	*av;
	struct psmx2_fid_cq	*send_cq;
	struct psmx2_fid_cq	*recv_cq;
	struct psmx2_fid_cntr	*send_cntr;
	struct psmx2_fid_cntr	*recv_cntr;
	struct psmx2_fid_cntr	*write_cntr;
	struct psmx2_fid_cntr	*read_cntr;
	struct psmx2_fid_cntr	*remote_write_cntr;
	struct psmx2_fid_cntr	*remote_read_cntr;
	unsigned		send_selective_completion:1;
	unsigned		recv_selective_completion:1;
	uint64_t		flags;
	uint64_t		caps;
	struct fi_context	nocomp_send_context;
	struct fi_context	nocomp_recv_context;
	size_t			min_multi_recv;
};

struct psmx2_fid_stx {
	struct fid_stx		stx;
	struct psmx2_fid_domain	*domain;
};

struct psmx2_fid_mr {
	struct fid_mr		mr;
	struct psmx2_fid_domain	*domain;
	struct psmx2_fid_cntr	*cntr;
	uint64_t		access;
	uint64_t		flags;
	uint64_t		offset;
	size_t			iov_count;
	struct iovec		iov[0];	/* must be the last field */
};

struct psmx2_epaddr_context {
	struct psmx2_fid_domain	*domain;
	psm2_epid_t		epid;
};

struct psmx2_env {
	int name_server;
	int am_msg;
	int tagged_rma;
	char *uuid;
	int delay;
	int timeout;
	int prog_interval;
	char *prog_affinity;
};

extern struct fi_ops_mr		psmx2_mr_ops;
extern struct fi_ops_cm		psmx2_cm_ops;
extern struct fi_ops_tagged	psmx2_tagged_ops;
extern struct fi_ops_tagged	psmx2_tagged_ops_no_flag_av_map;
extern struct fi_ops_tagged	psmx2_tagged_ops_no_flag_av_table;
extern struct fi_ops_tagged	psmx2_tagged_ops_no_event_av_map;
extern struct fi_ops_tagged	psmx2_tagged_ops_no_event_av_table;
extern struct fi_ops_tagged	psmx2_tagged_ops_no_send_event_av_map;
extern struct fi_ops_tagged	psmx2_tagged_ops_no_send_event_av_table;
extern struct fi_ops_tagged	psmx2_tagged_ops_no_recv_event_av_map;
extern struct fi_ops_tagged	psmx2_tagged_ops_no_recv_event_av_table;
extern struct fi_ops_msg	psmx2_msg_ops;
extern struct fi_ops_msg	psmx2_msg2_ops;
extern struct fi_ops_rma	psmx2_rma_ops;
extern struct fi_ops_atomic	psmx2_atomic_ops;
extern struct psm2_am_parameters psmx2_am_param;
extern struct psmx2_env		psmx2_env;
extern struct psmx2_fid_fabric	*psmx2_active_fabric;

int	psmx2_fabric(struct fi_fabric_attr *attr,
		    struct fid_fabric **fabric, void *context);
int	psmx2_domain_open(struct fid_fabric *fabric, struct fi_info *info,
			 struct fid_domain **domain, void *context);
int	psmx2_wait_open(struct fid_fabric *fabric, struct fi_wait_attr *attr,
		       struct fid_wait **waitset);
int	psmx2_ep_open(struct fid_domain *domain, struct fi_info *info,
		     struct fid_ep **ep, void *context);
int	psmx2_stx_ctx(struct fid_domain *domain, struct fi_tx_attr *attr,
		     struct fid_stx **stx, void *context);
int	psmx2_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		     struct fid_cq **cq, void *context);
int	psmx2_eq_open(struct fid_fabric *fabric, struct fi_eq_attr *attr,
		     struct fid_eq **eq, void *context);
int	psmx2_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		     struct fid_av **av, void *context);
int	psmx2_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		       struct fid_cntr **cntr, void *context);
int	psmx2_poll_open(struct fid_domain *domain, struct fi_poll_attr *attr,
		       struct fid_poll **pollset);

static inline void psmx2_fabric_acquire(struct psmx2_fid_fabric *fabric)
{
	++fabric->refcnt;
}

void	psmx2_fabric_release(struct psmx2_fid_fabric *fabric);

static inline void psmx2_domain_acquire(struct psmx2_fid_domain *domain)
{
	++domain->refcnt;
}

void	psmx2_domain_release(struct psmx2_fid_domain *domain);
int	psmx2_domain_check_features(struct psmx2_fid_domain *domain, int ep_cap);
int	psmx2_domain_enable_ep(struct psmx2_fid_domain *domain, struct psmx2_fid_ep *ep);
void	psmx2_domain_disable_ep(struct psmx2_fid_domain *domain, struct psmx2_fid_ep *ep);
void 	*psmx2_name_server(void *args);
void	*psmx2_resolve_name(const char *servername, int port);
void	psmx2_get_uuid(psm2_uuid_t uuid);
int	psmx2_uuid_to_port(psm2_uuid_t uuid);
char	*psmx2_uuid_to_string(psm2_uuid_t uuid);
int	psmx2_errno(int err);
int	psmx2_epid_to_epaddr(struct psmx2_fid_domain *domain,
			    psm2_epid_t epid, psm2_epaddr_t *epaddr);
void	psmx2_query_mpi(void);

void	psmx2_eq_enqueue_event(struct psmx2_fid_eq *eq, struct psmx2_eq_event *event);
struct	psmx2_eq_event *psmx2_eq_create_event(struct psmx2_fid_eq *eq,
					uint32_t event_num,
					void *context, uint64_t data,
					int err, int prov_errno,
					void *err_data, size_t err_data_size);
void	psmx2_cq_enqueue_event(struct psmx2_fid_cq *cq, struct psmx2_cq_event *event);
struct	psmx2_cq_event *psmx2_cq_create_event(struct psmx2_fid_cq *cq,
					void *op_context, void *buf,
					uint64_t flags, size_t len,
					uint64_t data, uint64_t tag,
					size_t olen, int err);
int	psmx2_cq_poll_mq(struct psmx2_fid_cq *cq, struct psmx2_fid_domain *domain,
			struct psmx2_cq_event *event, int count, fi_addr_t *src_addr);
int	psmx2_wait_get_obj(struct psmx2_fid_wait *wait, void *arg);
int	psmx2_wait_wait(struct fid_wait *wait, int timeout);
void	psmx2_wait_signal(struct fid_wait *wait);

int	psmx2_am_init(struct psmx2_fid_domain *domain);
int	psmx2_am_fini(struct psmx2_fid_domain *domain);
int	psmx2_am_progress(struct psmx2_fid_domain *domain);
int	psmx2_am_process_send(struct psmx2_fid_domain *domain,
				struct psmx2_am_request *req);
int	psmx2_am_process_rma(struct psmx2_fid_domain *domain,
				struct psmx2_am_request *req);
int	psmx2_process_trigger(struct psmx2_fid_domain *domain,
				struct psmx2_trigger *trigger);
int	psmx2_am_msg_handler(psm2_am_token_t token,
				psm2_amarg_t *args, int nargs, void *src, uint32_t len);
int	psmx2_am_rma_handler(psm2_am_token_t token,
				psm2_amarg_t *args, int nargs, void *src, uint32_t len);
int	psmx2_am_atomic_handler(psm2_am_token_t token,
				psm2_amarg_t *args, int nargs, void *src, uint32_t len);
void	psmx2_atomic_init(void);
void	psmx2_atomic_fini(void);

void	psmx2_am_ack_rma(struct psmx2_am_request *req);

struct	psmx2_fid_mr *psmx2_mr_get(struct psmx2_fid_domain *domain, uint64_t key);
int	psmx2_mr_validate(struct psmx2_fid_mr *mr, uint64_t addr, size_t len, uint64_t access);
void	psmx2_cntr_check_trigger(struct psmx2_fid_cntr *cntr);
void	psmx2_cntr_add_trigger(struct psmx2_fid_cntr *cntr, struct psmx2_trigger *trigger);

static inline void psmx2_cntr_inc(struct psmx2_fid_cntr *cntr)
{
	cntr->counter++;
	psmx2_cntr_check_trigger(cntr);
	if (cntr->wait)
		psmx2_wait_signal((struct fid_wait *)cntr->wait);
}

static inline void psmx2_progress(struct psmx2_fid_domain *domain)
{
	if (domain) {
		psmx2_cq_poll_mq(NULL, domain, NULL, 0, NULL);
		if (domain->am_initialized)
			psmx2_am_progress(domain);
	}
}

ssize_t _psmx2_send(struct fid_ep *ep, const void *buf, size_t len,
		   void *desc, fi_addr_t dest_addr, void *context,
		   uint64_t flags, uint32_t data);
ssize_t _psmx2_recv(struct fid_ep *ep, void *buf, size_t len,
		   void *desc, fi_addr_t src_addr, void *context,
		   uint64_t flags);
ssize_t _psmx2_tagged_send(struct fid_ep *ep, const void *buf, size_t len,
			  void *desc, fi_addr_t dest_addr, uint64_t tag,
			  void *context, uint64_t flags, uint32_t data);
ssize_t _psmx2_tagged_recv(struct fid_ep *ep, void *buf, size_t len,
			  void *desc, fi_addr_t src_addr, uint64_t tag,
			  uint64_t ignore, void *context, uint64_t flags);
ssize_t _psmx2_write(struct fid_ep *ep, const void *buf, size_t len,
		    void *desc, fi_addr_t dest_addr,
		    uint64_t addr, uint64_t key, void *context,
		    uint64_t flags, uint64_t data);
ssize_t _psmx2_read(struct fid_ep *ep, void *buf, size_t len,
		   void *desc, fi_addr_t src_addr,
		   uint64_t addr, uint64_t key, void *context,
		   uint64_t flags);
ssize_t _psmx2_atomic_write(struct fid_ep *ep,
			   const void *buf,
			   size_t count, void *desc,
			   fi_addr_t dest_addr,
			   uint64_t addr, uint64_t key,
			   enum fi_datatype datatype,
			   enum fi_op op, void *context,
			   uint64_t flags);
ssize_t _psmx2_atomic_readwrite(struct fid_ep *ep,
				const void *buf,
				size_t count, void *desc,
				void *result, void *result_desc,
				fi_addr_t dest_addr,
				uint64_t addr, uint64_t key,
				enum fi_datatype datatype,
				enum fi_op op, void *context,
				uint64_t flags);
ssize_t _psmx2_atomic_compwrite(struct fid_ep *ep,
				const void *buf,
				size_t count, void *desc,
				const void *compare, void *compare_desc,
				void *result, void *result_desc,
				fi_addr_t dest_addr,
				uint64_t addr, uint64_t key,
				enum fi_datatype datatype,
				enum fi_op op, void *context,
				uint64_t flags);

#ifdef __cplusplus
}
#endif

#endif

