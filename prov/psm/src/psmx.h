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

#ifndef _FI_PSM_H
#define _FI_PSM_H

#ifdef __cplusplus
extern "C" {
#endif

#if HAVE_CONFIG_H
#include <config.h>
#endif

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

#include "version.h"
#if (PSMX_VERSION >= 2)
#include <psm2.h>
#include <psm2_mq.h>
#include <psm2_am.h>
#else
#include <psm.h>
#include <psm_mq.h>
#include "psm_am.h"
#if (PSM_VERNO_MAJOR >= 2)
#error "building PSM provider against PSM2 is not supported"
#endif
#endif

#include "fi.h"
#include "fi_enosys.h"
#include "fi_list.h"

#ifndef PSMX_DL
#define PSMX_DL			1
#endif

#if PSMX_DL
#include <dlfcn.h>
#define PSMX_CALL(func)	(*psmx_lib.func)
struct psmx_lib {
	psm_error_t	(*psm_init)(int *major, int *minor);
	psm_error_t	(*psm_finalize)(void);
	psm_error_t 	(*psm_error_register_handler)(psm_ep_t ep,
				const psm_ep_errhandler_t errhandler);
	psm_error_t	(*psm_error_defer)(psm_error_token_t err_token);
	const char *	(*psm_error_get_string)(psm_error_t error);
	uint64_t	(*psm_epid_nid)(psm_epid_t epid);
	uint64_t	(*psm_epid_context)(psm_epid_t epid);
	uint64_t	(*psm_epid_port)(psm_epid_t epid);
	psm_error_t	(*psm_ep_num_devunits)(uint32_t *num_units);
	void		(*psm_uuid_generate)(psm_uuid_t uuid);
	psm_error_t	(*psm_ep_open)(const psm_uuid_t uuid,
				       const struct psm_ep_open_opts *opts,
				       psm_ep_t *ep, psm_epid_t *epid);
	psm_error_t	(*psm_ep_open_opts_get_defaults)(struct psm_ep_open_opts *opts);
	psm_error_t	(*psm_ep_epid_share_memory)(psm_ep_t ep, psm_epid_t epid,
						    int *result);
	psm_error_t	(*psm_ep_close)(psm_ep_t ep, int mode, int64_t timeout);
	psm_error_t	(*psm_map_nid_hostname)(int num, const uint64_t *nids,
						const char **hostnames);
	psm_error_t	(*psm_ep_connect)(psm_ep_t ep, int num,
					  const psm_epid_t *epids, const int *masks,
					  psm_error_t *errors, psm_epaddr_t *epaddrs,
					  int64_t timeout);
	psm_error_t	(*psm_poll)(psm_ep_t ep);
	void		(*psm_epaddr_setlabel)(psm_epaddr_t epaddr, const char *label);
	void		(*psm_epaddr_setctxt)(psm_epaddr_t epaddr, void *ctxt);
	void *		(*psm_epaddr_getctxt)(psm_epaddr_t epaddr);
	psm_error_t	(*psm_setopt)(psm_component_t component,
				      const void *obj, int optname,
				      const void *optval, uint64_t optlen);
	psm_error_t	(*psm_getopt)(psm_component_t component,
				      const void *obj, int optname, void *optval,
				      uint64_t *optlen);
	psm_error_t	(*psm_ep_query)(int *num, psm_epinfo_t *epinfos);
	psm_error_t	(*psm_ep_epid_lookup)(psm_epid_t epid, psm_epconn_t *epconn);
	psm_error_t	(*psm_mq_init)(psm_ep_t ep, uint64_t tag_order_mask,
				       const struct psm_optkey *opts, int numopts,
				       psm_mq_t *mq);
	psm_error_t	(*psm_mq_finalize)(psm_mq_t mq);
	psm_error_t	(*psm_mq_getopt)(psm_mq_t mq, int opt, void *val);
	psm_error_t	(*psm_mq_setopt)(psm_mq_t mq, int opt, const void *val);
	psm_error_t	(*psm_mq_irecv)(psm_mq_t mq, uint64_t rtag,
				        uint64_t rtagsel, uint32_t flags, void *buf,
				        uint32_t len, void *ctxt, psm_mq_req_t *req);
	psm_error_t	(*psm_mq_send)(psm_mq_t mq, psm_epaddr_t dest,
				       uint32_t flags, uint64_t stag, const void *buf,
				       uint32_t len);
	psm_error_t	(*psm_mq_isend)(psm_mq_t mq, psm_epaddr_t dest,
					uint32_t flags, uint64_t stag, const void *buf,
					uint32_t len, void *ctxt, psm_mq_req_t *req);
	psm_error_t	(*psm_mq_iprobe)(psm_mq_t mq, uint64_t rtag,
					 uint64_t rtagsel, psm_mq_status_t *status);
	psm_error_t	(*psm_mq_ipeek)(psm_mq_t mq, psm_mq_req_t *req,
					psm_mq_status_t *status);
	psm_error_t	(*psm_mq_wait)(psm_mq_req_t *req, psm_mq_status_t *status);
	psm_error_t	(*psm_mq_test)(psm_mq_req_t *req, psm_mq_status_t *status);
	psm_error_t	(*psm_mq_cancel)(psm_mq_req_t *req);
	void		(*psm_mq_get_stats)(psm_mq_t mq, psm_mq_stats_t *stats);
	psm_error_t	(*psm_am_register_handlers)(psm_ep_t ep,
					const psm_am_handler_fn_t *handlers,
					int num_handlers, int *handlers_idx);
	psm_error_t	(*psm_am_request_short)(psm_epaddr_t epaddr, psm_handler_t handler,
						psm_amarg_t *args, int nargs, void *src,
						size_t len, int flags,
						psm_am_completion_fn_t completion_fn,
						void *completion_ctxt);
	psm_error_t	(*psm_am_reply_short)(psm_am_token_t token, psm_handler_t handler,
					      psm_amarg_t *args, int nargs, void *src,
					      size_t len, int flags,
					      psm_am_completion_fn_t completion_fn,
					      void *completion_ctxt);
	psm_error_t	(*psm_am_get_parameters)(psm_ep_t ep, struct psm_am_parameters *params,
						 size_t sizeof_parameters_in,
						 size_t *sizeof_parameters_out);
#if (PSM_VERNO_MAJOR >= 2)
	psm_error_t	(*psm_mq_irecv2)(psm_mq_t mq, psm_epaddr_t src,
					 psm_mq_tag_t *rtag, psm_mq_tag_t *rtagsel,
					 uint32_t flags, void *buf, uint32_t len,
					 void *ctxt, psm_mq_req_t *req);
	psm_error_t	(*psm_mq_imrecv)(psm_mq_t mq, uint32_t flags, void *buf,
					 uint32_t len, void *ctxt, psm_mq_req_t *req);
	psm_error_t	(*psm_mq_send2)(psm_mq_t mq, psm_epaddr_t dest,
					uint32_t flags, psm_mq_tag_t *stag,
					const void *buf, uint32_t len);
	psm_error_t	(*psm_mq_isend2)(psm_mq_t mq, psm_epaddr_t dest,
					 uint32_t flags, psm_mq_tag_t *stag,
					 const void *buf, uint32_t len, void *ctxt,
					 psm_mq_req_t *req);
	psm_error_t	(*psm_mq_iprobe2)(psm_mq_t mq, psm_epaddr_t src,
					  psm_mq_tag_t *rtag, psm_mq_tag_t *rtagsel,
					  psm_mq_status2_t *status);
	psm_error_t	(*psm_mq_improbe)(psm_mq_t mq, uint64_t rtag,
					  uint64_t rtagsel, psm_mq_req_t *req,
					  psm_mq_status_t *status);
	psm_error_t	(*psm_mq_improbe2)(psm_mq_t mq, psm_epaddr_t src,
					   psm_mq_tag_t *rtag, psm_mq_tag_t *rtagsel,
					   psm_mq_req_t *req, psm_mq_status2_t *status);
	psm_error_t	(*psm_mq_ipeek2)(psm_mq_t mq, psm_mq_req_t *req,
					 psm_mq_status2_t *status);
	psm_error_t	(*psm_mq_wait2)(psm_mq_req_t *req, psm_mq_status2_t *status);
	psm_error_t	(*psm_mq_test2)(psm_mq_req_t *req, psm_mq_status2_t *status);
	psm_error_t	(*psm_am_get_source)(psm_am_token_t token, psm_epaddr_t *epaddr_out);
#endif
};
extern struct psmx_lib	psmx_lib;
int	psmx_dl_open(void);
void	psmx_dl_close(void);
#else
#define PSMX_CALL(func)	func
#endif

#if (PSM_VERNO_MAJOR >= 2)
#define PSMX_LIB_NAME		"libpsm2.so"
#define PSMX_PROV_NAME		"psm2"
#define PSMX_PROV_NAME_LEN	4
#define PSMX_DOMAIN_NAME	"psm2"
#define PSMX_DOMAIN_NAME_LEN	4
#define PSMX_FABRIC_NAME	"psm2"
#define PSMX_FABRIC_NAME_LEN	4
#else
#define PSMX_LIB_NAME		"libpsm_infinipath.so"
#define PSMX_PROV_NAME		"psm"
#define PSMX_PROV_NAME_LEN	3
#define PSMX_DOMAIN_NAME	"psm"
#define PSMX_DOMAIN_NAME_LEN	3
#define PSMX_FABRIC_NAME	"psm"
#define PSMX_FABRIC_NAME_LEN	3
#endif

#define PSMX_DEFAULT_UUID	"0FFF0FFF-0000-0000-0000-0FFF0FFF0FFF"

extern struct fi_provider psmx_prov;

#define PSMX_TIME_OUT	120

#define PSMX_OP_FLAGS	(FI_INJECT | FI_MULTI_RECV | FI_COMPLETION | \
			 FI_TRIGGER | FI_INJECT_COMPLETE | \
			 FI_TRANSMIT_COMPLETE | FI_DELIVERY_COMPLETE)

#if (PSM_VERNO_MAJOR >= 2)
#define PSMX_CAP_EXT	(FI_REMOTE_CQ_DATA)
#else
#define PSMX_CAP_EXT	(0)
#endif

#define PSMX_CAPS	(FI_TAGGED | FI_MSG | FI_ATOMICS | \
			 FI_RMA | FI_MULTI_RECV | \
                         FI_READ | FI_WRITE | FI_SEND | FI_RECV | \
                         FI_REMOTE_READ | FI_REMOTE_WRITE | \
			 FI_TRIGGER | \
			 FI_RMA_EVENT | \
			 PSMX_CAP_EXT)

#if (PSM_VERNO_MAJOR >= 2)
#define PSMX_CAPS2	(PSMX_CAPS | FI_DIRECTED_RECV)
#else
#define PSMX_CAPS2	((PSMX_CAPS | FI_DIRECTED_RECV) & ~FI_TAGGED)
#endif

#define PSMX_SUB_CAPS	(FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE | \
			 FI_SEND | FI_RECV)

#define PSMX_MODE	(FI_CONTEXT)

#define PSMX_MAX_MSG_SIZE	((0x1ULL << 32) - 1)
#define PSMX_INJECT_SIZE	(64)

#define PSMX_MSG_BIT	(0x1ULL << 63)
#define PSMX_RMA_BIT	(0x1ULL << 62)

/* Bits 60 .. 63 of the flag are provider specific */
#define PSMX_NO_COMPLETION	(1ULL << 60)

enum psmx_context_type {
	PSMX_NOCOMP_SEND_CONTEXT = 1,
	PSMX_NOCOMP_RECV_CONTEXT,
	PSMX_NOCOMP_WRITE_CONTEXT,
	PSMX_NOCOMP_READ_CONTEXT,
	PSMX_SEND_CONTEXT,
	PSMX_RECV_CONTEXT,
	PSMX_MULTI_RECV_CONTEXT,
	PSMX_TSEND_CONTEXT,
	PSMX_TRECV_CONTEXT,
	PSMX_WRITE_CONTEXT,
	PSMX_READ_CONTEXT,
	PSMX_REMOTE_WRITE_CONTEXT,
	PSMX_REMOTE_READ_CONTEXT,
};

union psmx_pi {
	void	*p;
	int	i;
};

#define PSMX_CTXT_REQ(fi_context)	((fi_context)->internal[0])
#define PSMX_CTXT_TYPE(fi_context)	(((union psmx_pi *)&(fi_context)->internal[1])->i)
#define PSMX_CTXT_USER(fi_context)	((fi_context)->internal[2])
#define PSMX_CTXT_EP(fi_context)	((fi_context)->internal[3])

#if (PSM_VERNO_MAJOR >= 2)
#define PSMX_SET_TAG(tag96,tag64,tag32)	do { \
						tag96.tag0 = (uint32_t)tag64; \
						tag96.tag1 = (uint32_t)(tag64>>32); \
						tag96.tag2 = tag32; \
					} while (0)
#endif

#define PSMX_AM_RMA_HANDLER	0
#define PSMX_AM_MSG_HANDLER	1
#define PSMX_AM_ATOMIC_HANDLER	2
#define PSMX_AM_CHUNK_SIZE	2032	/* The maximum that's actually working:
					 * 2032 for inter-node, 2072 for intra-node.
					 */

#define PSMX_AM_OP_MASK		0x0000FFFF
#define PSMX_AM_FLAG_MASK	0xFFFF0000
#define PSMX_AM_EOM		0x40000000
#define PSMX_AM_DATA		0x20000000
#define PSMX_AM_FORCE_ACK	0x10000000

#ifndef PSMX_AM_USE_SEND_QUEUE
#define PSMX_AM_USE_SEND_QUEUE	0
#endif

enum {
	PSMX_AM_REQ_WRITE = 1,
	PSMX_AM_REQ_WRITE_LONG,
	PSMX_AM_REP_WRITE,
	PSMX_AM_REQ_READ,
	PSMX_AM_REQ_READ_LONG,
	PSMX_AM_REP_READ,
	PSMX_AM_REQ_SEND,
	PSMX_AM_REP_SEND,
	PSMX_AM_REQ_ATOMIC_WRITE,
	PSMX_AM_REP_ATOMIC_WRITE,
	PSMX_AM_REQ_ATOMIC_READWRITE,
	PSMX_AM_REP_ATOMIC_READWRITE,
	PSMX_AM_REQ_ATOMIC_COMPWRITE,
	PSMX_AM_REP_ATOMIC_COMPWRITE,
};

enum {
	PSMX_AM_STATE_NEW,
	PSMX_AM_STATE_QUEUED,
	PSMX_AM_STATE_PROCESSED,
	PSMX_AM_STATE_DONE
};

struct psmx_am_request {
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
	struct psmx_fid_ep *ep;
	int state;
	int no_event;
	int error;
	struct slist_entry list_entry;
};

struct psmx_unexp {
	psm_epaddr_t		sender_addr;
	uint64_t		sender_context;
	uint32_t		len_received;
	uint32_t		done;
	struct slist_entry	list_entry;
	char			buf[0];
};

struct psmx_req_queue {
	pthread_mutex_t	lock;
	struct slist	list;
};

struct psmx_multi_recv {
	uint64_t	tag;
	uint64_t	tagsel;
	void		*buf;
	size_t		len;
	size_t		offset;
	int		min_buf_size;
	int		flag;
	void		*context;
};

struct psmx_fid_fabric {
	struct fid_fabric	fabric;
	int			refcnt;
	struct psmx_fid_domain	*active_domain;
	psm_uuid_t		uuid;
};

struct psmx_fid_domain {
	struct fid_domain	domain;
	struct psmx_fid_fabric	*fabric;
	int			refcnt;
	psm_ep_t		psm_ep;
	psm_epid_t		psm_epid;
	psm_mq_t		psm_mq;
	struct psmx_fid_ep	*tagged_ep;
	struct psmx_fid_ep	*msg_ep;
	struct psmx_fid_ep	*rma_ep;
	struct psmx_fid_ep	*atomics_ep;
	uint64_t		mode;
	uint64_t		caps;
	enum fi_mr_mode		mr_mode;

	int			am_initialized;

#if PSMX_AM_USE_SEND_QUEUE
	pthread_cond_t		progress_cond;
	pthread_mutex_t		progress_mutex;
	pthread_t		progress_thread;
#endif

	/* incoming req queue for AM based RMA request. */
	struct psmx_req_queue	rma_queue;

#if PSMX_AM_USE_SEND_QUEUE
	/* send queue for AM based messages. */
	struct psmx_req_queue	send_queue;
#endif

	/* recv queue for AM based messages. */
	struct psmx_req_queue	recv_queue;
	struct psmx_req_queue	unexp_queue;

	/* triggered operations that are ready to be processed */
	struct psmx_req_queue	trigger_queue;

	/* certain bits in the tag space can be reserved for non tag-matching
	 * purpose. The tag-matching functions automatically treat these bits
	 * as 0. This field is a bit mask, with reserved bits valued as "1".
	 */
	uint64_t		reserved_tag_bits; 
};

struct psmx_cq_event {
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

struct psmx_eq_event {
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

struct psmx_fid_wait {
	struct fid_wait			wait;
	struct psmx_fid_fabric		*fabric;
	int				type;
	union {
		int			fd[2];
		struct {
			pthread_mutex_t	mutex;
			pthread_cond_t	cond;
		};
	};
};

struct psmx_poll_list {
	struct dlist_entry		entry;
	struct fid			*fid;
};

struct psmx_fid_poll {
	struct fid_poll			poll;
	struct psmx_fid_domain		*domain;
	struct dlist_entry		poll_list_head;
};

struct psmx_fid_cq {
	struct fid_cq			cq;
	struct psmx_fid_domain		*domain;
	int 				format;
	int				entry_size;
	size_t				event_count;
	struct slist			event_queue;
	struct slist			free_list;
	pthread_mutex_t			mutex;
	struct psmx_cq_event		*pending_error;
	struct psmx_fid_wait		*wait;
	int				wait_cond;
	int				wait_is_local;
};

struct psmx_fid_eq {
	struct fid_eq			eq;
	struct psmx_fid_fabric		*fabric;
	struct slist			event_queue;
	struct slist			error_queue;
	struct slist			free_list;
	pthread_mutex_t			mutex;
	struct psmx_fid_wait		*wait;
	int				wait_is_local;
};

enum psmx_triggered_op {
	PSMX_TRIGGERED_SEND,
	PSMX_TRIGGERED_RECV,
	PSMX_TRIGGERED_TSEND,
	PSMX_TRIGGERED_TRECV,
	PSMX_TRIGGERED_WRITE,
	PSMX_TRIGGERED_READ,
	PSMX_TRIGGERED_ATOMIC_WRITE,
	PSMX_TRIGGERED_ATOMIC_READWRITE,
	PSMX_TRIGGERED_ATOMIC_COMPWRITE,
};

struct psmx_trigger {
	enum psmx_triggered_op	op;
	struct psmx_fid_cntr	*cntr;
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
#if (PSM_VERNO_MAJOR >= 2)
			uint32_t	data;
#endif
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
#if (PSM_VERNO_MAJOR >= 2)
			uint32_t	data;
#endif
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
	struct psmx_trigger *next;	/* used for randomly accessed trigger list */
	struct slist_entry list_entry;	/* used for ready-to-fire trigger queue */
};

struct psmx_fid_cntr {
	struct fid_cntr		cntr;
	struct psmx_fid_domain	*domain;
	int			events;
	uint64_t		flags;
	volatile uint64_t	counter;
	volatile uint64_t	error_counter;
	uint64_t		counter_last_read;
	uint64_t		error_counter_last_read;
	struct psmx_fid_wait	*wait;
	int			wait_is_local;
	struct psmx_trigger	*trigger;
	pthread_mutex_t		trigger_lock;
};

struct psmx_fid_av {
	struct fid_av		av;
	struct psmx_fid_domain	*domain;
	struct psmx_fid_eq	*eq;
	int			type;
	uint64_t		flags;
	size_t			addrlen;
	size_t			count;
	size_t			last;
	psm_epid_t		*psm_epids;
	psm_epaddr_t		*psm_epaddrs;
};

struct psmx_fid_ep {
	struct fid_ep		ep;
	struct psmx_fid_domain	*domain;
	struct psmx_fid_av	*av;
	struct psmx_fid_cq	*send_cq;
	struct psmx_fid_cq	*recv_cq;
	struct psmx_fid_cntr	*send_cntr;
	struct psmx_fid_cntr	*recv_cntr;
	struct psmx_fid_cntr	*write_cntr;
	struct psmx_fid_cntr	*read_cntr;
	struct psmx_fid_cntr	*remote_write_cntr;
	struct psmx_fid_cntr	*remote_read_cntr;
	int			send_selective_completion:1;
	int			recv_selective_completion:1;
	uint64_t		flags;
	uint64_t		caps;
	struct fi_context	nocomp_send_context;
	struct fi_context	nocomp_recv_context;
	size_t			min_multi_recv;
};

struct psmx_fid_stx {
	struct fid_stx		stx;
	struct psmx_fid_domain	*domain;
};

struct psmx_fid_mr {
	struct fid_mr		mr;
	struct psmx_fid_domain	*domain;
	struct psmx_fid_cntr	*cntr;
	uint64_t		access;
	uint64_t		flags;
	uint64_t		offset;
	size_t			iov_count;
	struct iovec		iov[0];	/* must be the last field */
};

struct psmx_epaddr_context {
	struct psmx_fid_domain	*domain;
	psm_epid_t		epid;
};

struct psmx_env {
	int name_server;
	int am_msg;
	int tagged_rma;
	char *uuid;
	int delay;
};

extern struct fi_ops_mr		psmx_mr_ops;
extern struct fi_ops_cm		psmx_cm_ops;
extern struct fi_ops_tagged	psmx_tagged_ops;
extern struct fi_ops_tagged	psmx_tagged_ops_no_flag_av_map;
extern struct fi_ops_tagged	psmx_tagged_ops_no_flag_av_table;
extern struct fi_ops_tagged	psmx_tagged_ops_no_event_av_map;
extern struct fi_ops_tagged	psmx_tagged_ops_no_event_av_table;
extern struct fi_ops_tagged	psmx_tagged_ops_no_send_event_av_map;
extern struct fi_ops_tagged	psmx_tagged_ops_no_send_event_av_table;
extern struct fi_ops_tagged	psmx_tagged_ops_no_recv_event_av_map;
extern struct fi_ops_tagged	psmx_tagged_ops_no_recv_event_av_table;
extern struct fi_ops_msg	psmx_msg_ops;
extern struct fi_ops_msg	psmx_msg2_ops;
extern struct fi_ops_rma	psmx_rma_ops;
extern struct fi_ops_atomic	psmx_atomic_ops;
extern struct psm_am_parameters psmx_am_param;
extern struct psmx_env		psmx_env;
extern struct psmx_fid_fabric	*psmx_active_fabric;

int	psmx_domain_open(struct fid_fabric *fabric, struct fi_info *info,
			 struct fid_domain **domain, void *context);
int	psmx_wait_open(struct fid_fabric *fabric, struct fi_wait_attr *attr,
		       struct fid_wait **waitset);
int	psmx_ep_open(struct fid_domain *domain, struct fi_info *info,
		     struct fid_ep **ep, void *context);
int	psmx_stx_ctx(struct fid_domain *domain, struct fi_tx_attr *attr,
		     struct fid_stx **stx, void *context);
int	psmx_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		     struct fid_cq **cq, void *context);
int	psmx_eq_open(struct fid_fabric *fabric, struct fi_eq_attr *attr,
		     struct fid_eq **eq, void *context);
int	psmx_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		     struct fid_av **av, void *context);
int	psmx_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		       struct fid_cntr **cntr, void *context);
int	psmx_poll_open(struct fid_domain *domain, struct fi_poll_attr *attr,
		       struct fid_poll **pollset);

int	psmx_domain_check_features(struct psmx_fid_domain *domain, int ep_cap);
int	psmx_domain_enable_ep(struct psmx_fid_domain *domain, struct psmx_fid_ep *ep);
void	psmx_domain_disable_ep(struct psmx_fid_domain *domain, struct psmx_fid_ep *ep);
void 	*psmx_name_server(void *args);
void	*psmx_resolve_name(const char *servername, int port);
void	psmx_get_uuid(psm_uuid_t uuid);
int	psmx_uuid_to_port(psm_uuid_t uuid);
int	psmx_errno(int err);
int	psmx_epid_to_epaddr(struct psmx_fid_domain *domain,
			    psm_epid_t epid, psm_epaddr_t *epaddr);
void	psmx_query_mpi(void);

void	psmx_eq_enqueue_event(struct psmx_fid_eq *eq, struct psmx_eq_event *event);
struct	psmx_eq_event *psmx_eq_create_event(struct psmx_fid_eq *eq,
					uint32_t event_num,
					void *context, uint64_t data,
					int err, int prov_errno,
					void *err_data, size_t err_data_size);
void	psmx_cq_enqueue_event(struct psmx_fid_cq *cq, struct psmx_cq_event *event);
struct	psmx_cq_event *psmx_cq_create_event(struct psmx_fid_cq *cq,
					void *op_context, void *buf,
					uint64_t flags, size_t len,
					uint64_t data, uint64_t tag,
					size_t olen, int err);
int	psmx_cq_poll_mq(struct psmx_fid_cq *cq, struct psmx_fid_domain *domain,
			struct psmx_cq_event *event, int count, fi_addr_t *src_addr);
int	psmx_wait_get_obj(struct psmx_fid_wait *wait, void *arg);
int	psmx_wait_wait(struct fid_wait *wait, int timeout);
void	psmx_wait_signal(struct fid_wait *wait);

int	psmx_am_init(struct psmx_fid_domain *domain);
int	psmx_am_fini(struct psmx_fid_domain *domain);
int	psmx_am_progress(struct psmx_fid_domain *domain);
int	psmx_am_process_send(struct psmx_fid_domain *domain,
				struct psmx_am_request *req);
int	psmx_am_process_rma(struct psmx_fid_domain *domain,
				struct psmx_am_request *req);
int	psmx_process_trigger(struct psmx_fid_domain *domain,
				struct psmx_trigger *trigger);
#if (PSM_VERNO_MAJOR >= 2)
int	psmx_am_msg_handler(psm_am_token_t token,
				psm_amarg_t *args, int nargs, void *src, uint32_t len);
int	psmx_am_rma_handler(psm_am_token_t token,
				psm_amarg_t *args, int nargs, void *src, uint32_t len);
int	psmx_am_atomic_handler(psm_am_token_t token,
				psm_amarg_t *args, int nargs, void *src, uint32_t len);
#else
int	psmx_am_msg_handler(psm_am_token_t token, psm_epaddr_t epaddr,
				psm_amarg_t *args, int nargs, void *src, uint32_t len);
int	psmx_am_rma_handler(psm_am_token_t token, psm_epaddr_t epaddr,
				psm_amarg_t *args, int nargs, void *src, uint32_t len);
int	psmx_am_atomic_handler(psm_am_token_t token, psm_epaddr_t epaddr,
				psm_amarg_t *args, int nargs, void *src, uint32_t len);
#endif

void	psmx_am_ack_rma(struct psmx_am_request *req);

struct	psmx_fid_mr *psmx_mr_hash_get(uint64_t key);
int	psmx_mr_validate(struct psmx_fid_mr *mr, uint64_t addr, size_t len, uint64_t access);
void	psmx_cntr_check_trigger(struct psmx_fid_cntr *cntr);
void	psmx_cntr_add_trigger(struct psmx_fid_cntr *cntr, struct psmx_trigger *trigger);

static inline void psmx_cntr_inc(struct psmx_fid_cntr *cntr)
{
	cntr->counter++;
	psmx_cntr_check_trigger(cntr);
	if (cntr->wait)
		psmx_wait_signal((struct fid_wait *)cntr->wait);
}

static inline void psmx_progress(struct psmx_fid_domain *domain)
{
	if (domain) {
		psmx_cq_poll_mq(NULL, domain, NULL, 0, NULL);
		if (domain->am_initialized)
			psmx_am_progress(domain);
	}
}

#if (PSM_VERNO_MAJOR >= 2)
ssize_t _psmx_send(struct fid_ep *ep, const void *buf, size_t len,
		   void *desc, fi_addr_t dest_addr, void *context,
		   uint64_t flags, uint32_t data);
#else
ssize_t _psmx_send(struct fid_ep *ep, const void *buf, size_t len,
		   void *desc, fi_addr_t dest_addr, void *context,
		   uint64_t flags);
#endif
ssize_t _psmx_recv(struct fid_ep *ep, void *buf, size_t len,
		   void *desc, fi_addr_t src_addr, void *context,
		   uint64_t flags);
#if (PSM_VERNO_MAJOR >= 2)
ssize_t _psmx_tagged_send(struct fid_ep *ep, const void *buf, size_t len,
			  void *desc, fi_addr_t dest_addr, uint64_t tag,
			  void *context, uint64_t flags, uint32_t data);
#else
ssize_t _psmx_tagged_send(struct fid_ep *ep, const void *buf, size_t len,
			  void *desc, fi_addr_t dest_addr, uint64_t tag,
			  void *context, uint64_t flags);
#endif
ssize_t _psmx_tagged_recv(struct fid_ep *ep, void *buf, size_t len,
			  void *desc, fi_addr_t src_addr, uint64_t tag,
			  uint64_t ignore, void *context, uint64_t flags);
ssize_t _psmx_write(struct fid_ep *ep, const void *buf, size_t len,
		    void *desc, fi_addr_t dest_addr,
		    uint64_t addr, uint64_t key, void *context,
		    uint64_t flags, uint64_t data);
ssize_t _psmx_read(struct fid_ep *ep, void *buf, size_t len,
		   void *desc, fi_addr_t src_addr,
		   uint64_t addr, uint64_t key, void *context,
		   uint64_t flags);
ssize_t _psmx_atomic_write(struct fid_ep *ep,
			   const void *buf,
			   size_t count, void *desc,
			   fi_addr_t dest_addr,
			   uint64_t addr, uint64_t key,
			   enum fi_datatype datatype,
			   enum fi_op op, void *context,
			   uint64_t flags);
ssize_t _psmx_atomic_readwrite(struct fid_ep *ep,
				const void *buf,
				size_t count, void *desc,
				void *result, void *result_desc,
				fi_addr_t dest_addr,
				uint64_t addr, uint64_t key,
				enum fi_datatype datatype,
				enum fi_op op, void *context,
				uint64_t flags);
ssize_t _psmx_atomic_compwrite(struct fid_ep *ep,
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

