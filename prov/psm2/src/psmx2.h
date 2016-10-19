/*
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
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
#include <sys/stat.h>
#include <sys/socket.h>
#include <netdb.h>
#include <complex.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_trigger.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_errno.h>
#include "fi.h"
#include "fi_enosys.h"
#include "fi_list.h"
#include "fi_util.h"
#include "rbtree.h"
#include "version.h"

extern struct fi_provider psmx2_prov;

#define PSMX2_VERSION	(FI_VERSION(1,3))

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

#define PSMX2_MSG_BIT	(0x80000000)
#define PSMX2_RMA_BIT	(0x40000000)
#define PSMX2_IOV_BIT	(0x20000000)
#define PSMX2_SEQ_BITS	(0x0FFF0000)
#define PSMX2_SRC_BITS	(0x0000FF00)
#define PSMX2_DST_BITS	(0x000000FF)

#define PSMX2_TAG32(base, src, dst)	((base) | ((src)<<8) | (dst))
#define PSMX2_TAG32_GET_SRC(tag32)	(((tag32) & PSMX2_SRC_BITS) >> 8)
#define PSMX2_TAG32_GET_DST(tag32)	((tag32) & PSMX2_DST_BITS)
#define PSMX2_TAG32_GET_SEQ(tag32)	(((tag32) & PSMX2_SEQ_BITS) >> 16)
#define PSMX2_TAG32_SET_SEQ(tag32,seq)	do { \
						tag32 |= ((seq << 16) & PSMX2_SEQ_BITS); \
					} while (0)

#define PSMX2_SET_TAG(tag96,tag64,tag32) do { \
						tag96.tag0 = (uint32_t)tag64; \
						tag96.tag1 = (uint32_t)(tag64>>32); \
						tag96.tag2 = tag32; \
					} while (0)

#define PSMX2_GET_TAG64(tag96)		(tag96.tag0 | ((uint64_t)tag96.tag1<<32))

/* Canonical virtual address on X86_64 only uses 48 bits and the higher 16 bits
 * are sign extensions. We can put vlane into part of these 16 bits of an epaddr.
 */
#define PSMX2_MAX_VL			(0xFF)
#define PSMX2_EP_MASK			(0x00FFFFFFFFFFFFFFUL)
#define PSMX2_SIGN_MASK  		(0x0080000000000000UL)
#define PSMX2_SIGN_EXT			(0xFF00000000000000UL)
#define PSMX2_VL_MASK			(0xFF00000000000000UL)

#define PSMX2_EP_TO_ADDR(ep,vl)		((((uint64_t)vl) << 56) | \
						((uint64_t)ep & PSMX2_EP_MASK))
#define PSMX2_ADDR_TO_VL(addr)		((uint8_t)((addr & PSMX2_VL_MASK) >> 56))
#define PSMX2_ADDR_TO_EP(addr)		((psm2_epaddr_t) \
						((addr & PSMX2_SIGN_MASK) ? \
                                                 (addr | PSMX2_SIGN_EXT) : \
                                                 (addr & PSMX2_EP_MASK)))

/* Bits 60 .. 63 of the flag are provider specific */
#define PSMX2_NO_COMPLETION	(1ULL << 60)

#define PSMX2_CTXT_ALLOC_FLAG		0x80000000
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
	PSMX2_SENDV_CONTEXT,
	PSMX2_IOV_SEND_CONTEXT,
	PSMX2_IOV_RECV_CONTEXT,
	PSMX2_NOCOMP_RECV_CONTEXT_ALLOC = PSMX2_NOCOMP_RECV_CONTEXT | PSMX2_CTXT_ALLOC_FLAG,
};

struct psmx2_context {
	struct fi_context fi_context;
	struct slist_entry list_entry;
};

union psmx2_pi {
	void	*p;
	uint32_t i[2];
};

#define PSMX2_CTXT_REQ(fi_context)	((fi_context)->internal[0])
#define PSMX2_CTXT_TYPE(fi_context)	(((union psmx2_pi *)&(fi_context)->internal[1])->i[0])
#define PSMX2_CTXT_SIZE(fi_context)	(((union psmx2_pi *)&(fi_context)->internal[1])->i[1])
#define PSMX2_CTXT_USER(fi_context)	((fi_context)->internal[2])
#define PSMX2_CTXT_EP(fi_context)	((fi_context)->internal[3])

#define PSMX2_AM_RMA_HANDLER	0
#define PSMX2_AM_ATOMIC_HANDLER	1

#define PSMX2_AM_OP_MASK	0x000000FF
#define PSMX2_AM_DST_MASK	0x0000FF00
#define PSMX2_AM_SRC_MASK	0x00FF0000
#define PSMX2_AM_FLAG_MASK	0xFF000000
#define PSMX2_AM_EOM		0x40000000
#define PSMX2_AM_DATA		0x20000000
#define PSMX2_AM_FORCE_ACK	0x10000000

#define PSMX2_AM_SET_OP(u32w0,op)	do {u32w0 &= ~PSMX2_AM_OP_MASK; u32w0 |= op;} while (0)
#define PSMX2_AM_SET_DST(u32w0,vl)	do {u32w0 &= ~PSMX2_AM_DST_MASK; u32w0 |= ((uint32_t)vl << 8);} while (0)
#define PSMX2_AM_SET_SRC(u32w0,vl)	do {u32w0 &= ~PSMX2_AM_SRC_MASK; u32w0 |= ((uint32_t)vl << 16);} while (0)
#define PSMX2_AM_SET_FLAG(u32w0,flag)	do {u32w0 &= ~PSMX2_AM_FLAG_MASK; u32w0 |= flag;} while (0)
#define PSMX2_AM_GET_OP(u32w0)		(u32w0 & PSMX2_AM_OP_MASK)
#define PSMX2_AM_GET_DST(u32w0)		((uint8_t)((u32w0 & PSMX2_AM_DST_MASK) >> 8))
#define PSMX2_AM_GET_SRC(u32w0)		((uint8_t)((u32w0 & PSMX2_AM_SRC_MASK) >> 16))
#define PSMX2_AM_GET_FLAG(u32w0)	(u32w0 & PSMX2_AM_FLAG_MASK)

enum {
	PSMX2_AM_REQ_WRITE = 1,
	PSMX2_AM_REQ_WRITE_LONG,
	PSMX2_AM_REP_WRITE,
	PSMX2_AM_REQ_READ,
	PSMX2_AM_REQ_READ_LONG,
	PSMX2_AM_REP_READ,
	PSMX2_AM_REQ_ATOMIC_WRITE,
	PSMX2_AM_REP_ATOMIC_WRITE,
	PSMX2_AM_REQ_ATOMIC_READWRITE,
	PSMX2_AM_REP_ATOMIC_READWRITE,
	PSMX2_AM_REQ_ATOMIC_COMPWRITE,
	PSMX2_AM_REP_ATOMIC_COMPWRITE,
	PSMX2_AM_REQ_WRITEV,
	PSMX2_AM_REQ_READV,
};

struct psmx2_am_request {
	int op;
	union {
		struct {
			uint8_t	*buf;
			size_t	len;
			uint64_t addr;
			uint64_t key;
			void	*context;
			void	*peer_addr;
			uint8_t	vl;
			uint8_t	peer_vl;
			uint64_t data;
		} write;
		struct {
			union {
				uint8_t	*buf;	   /* for read */
				size_t	iov_count; /* for readv */
			};
			size_t	len;
			uint64_t addr;
			uint64_t key;
			void	*context;
			void	*peer_addr;
			uint8_t	vl;
			uint8_t	peer_vl;
			size_t	len_read;
		} read;
		struct {
			union {
				uint8_t	*buf;	   /* for result_count == 1 */
				size_t	iov_count; /* for result_count > 1 */
			};
			size_t	len;
			uint64_t addr;
			uint64_t key;
			void	*context;
			uint8_t *result;
			int	datatype;
		} atomic;
	};
	uint64_t cq_flags;
	struct fi_context fi_context;
	struct psmx2_fid_ep *ep;
	int no_event;
	int error;
	struct slist_entry list_entry;
	union {
		struct iovec iov[0];	/* for readv, must be the last field */
		struct fi_ioc ioc[0];	/* for atomic read, must be the last field */
	};
};

#define PSMX2_IOV_PROTO_PACK	0
#define PSMX2_IOV_PROTO_MULTI	1
#define PSMX2_IOV_MAX_SEQ_NUM	0x0FFF
#define PSMX2_IOV_BUF_SIZE	PSMX2_INJECT_SIZE
#define PSMX2_IOV_MAX_COUNT	(PSMX2_IOV_BUF_SIZE / sizeof(uint32_t) - 3)

struct psmx2_iov_info {
	uint32_t seq_num;
	uint32_t total_len;
	uint32_t count;
	uint32_t len[PSMX2_IOV_MAX_COUNT];
};

struct psmx2_sendv_request {
	struct fi_context fi_context;
	struct fi_context fi_context_iov;
	void *user_context;
	int iov_protocol;
	int no_completion;
	int comp_flag;
	uint32_t iov_done;
	union {
		struct psmx2_iov_info iov_info;
		char buf[PSMX2_IOV_BUF_SIZE];
	};
};

struct psmx2_sendv_reply {
	struct fi_context fi_context;
	int no_completion;
	int multi_recv;
	uint8_t *buf;
	void *user_context;
	size_t iov_done;
	size_t bytes_received;
	size_t msg_length;
	int error_code;
	int comp_flag;
	struct psmx2_iov_info iov_info;
};

struct psmx2_req_queue {
	fastlock_t	lock;
	struct slist	list;
};

struct psmx2_multi_recv {
	psm2_epaddr_t	src_addr;
	psm2_mq_tag_t	tag;
	psm2_mq_tag_t	tagsel;
	uint8_t		*buf;
	size_t		len;
	size_t		offset;
	int		min_buf_size;
	int		flag;
	void		*context;
};

struct psmx2_fid_fabric {
	struct util_fabric	util_fabric;
	struct psmx2_fid_domain	*active_domain;
	psm2_uuid_t		uuid;
	pthread_t		name_server_thread;
};

struct psmx2_fid_domain {
	struct util_domain	util_domain;
	struct psmx2_fid_fabric	*fabric;
	psm2_ep_t		psm2_ep;
	psm2_epid_t		psm2_epid;
	psm2_mq_t		psm2_mq;
	uint64_t		mode;
	uint64_t		caps;

	enum fi_mr_mode		mr_mode;
	fastlock_t		mr_lock;
	uint64_t		mr_reserved_key;
	RbtHandle		mr_map;

	fastlock_t		vl_lock;
	uint64_t		vl_map[(PSMX2_MAX_VL+1)/sizeof(uint64_t)];
	int			vl_alloc;
	struct psmx2_fid_ep	*eps[PSMX2_MAX_VL+1];

	int			am_initialized;

	/* incoming req queue for AM based RMA request. */
	struct psmx2_req_queue	rma_queue;

	/* triggered operations that are ready to be processed */
	struct psmx2_req_queue	trigger_queue;

	/* lock to prevent the sequence of psm2_mq_ipeek and psm2_mq_test be
	 * interleaved in a multithreaded environment.
	 */
	fastlock_t		poll_lock;

	int			progress_thread_enabled;
	pthread_t		progress_thread;
};

struct psmx2_ep_name {
	psm2_epid_t		epid;
	uint8_t			vlane;
};

#define PSMX2_DEFAULT_UNIT	(-1)
#define PSMX2_DEFAULT_PORT	0
#define PSMX2_DEFAULT_SERVICE	0

struct psmx2_src_name {
	int	unit;		/* start from 0. -1 means any */
	int	port;		/* start from 1. 0 means any */
	int	service;	/* 0 means any */
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
	fi_addr_t source;
	struct slist_entry list_entry;
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
	struct util_wait		*wait;
	int				wait_cond;
	int				wait_is_local;
};

enum psmx2_triggered_op {
	PSMX2_TRIGGERED_SEND,
	PSMX2_TRIGGERED_SENDV,
	PSMX2_TRIGGERED_RECV,
	PSMX2_TRIGGERED_TSEND,
	PSMX2_TRIGGERED_TSENDV,
	PSMX2_TRIGGERED_TRECV,
	PSMX2_TRIGGERED_WRITE,
	PSMX2_TRIGGERED_WRITEV,
	PSMX2_TRIGGERED_READ,
	PSMX2_TRIGGERED_READV,
	PSMX2_TRIGGERED_ATOMIC_WRITE,
	PSMX2_TRIGGERED_ATOMIC_WRITEV,
	PSMX2_TRIGGERED_ATOMIC_READWRITE,
	PSMX2_TRIGGERED_ATOMIC_READWRITEV,
	PSMX2_TRIGGERED_ATOMIC_COMPWRITE,
	PSMX2_TRIGGERED_ATOMIC_COMPWRITEV,
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
			uint64_t	data;
		} send;
		struct {
			struct fid_ep	*ep;
			const struct iovec *iov;
			size_t		count;
			void		**desc;
			fi_addr_t	dest_addr;
			void		*context;
			uint64_t	flags;
			uint64_t	data;
		} sendv;
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
			uint64_t	data;
		} tsend;
		struct {
			struct fid_ep	*ep;
			const struct iovec *iov;
			size_t		count;
			void		**desc;
			fi_addr_t	dest_addr;
			uint64_t	tag;
			void		*context;
			uint64_t	flags;
			uint64_t	data;
		} tsendv;
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
			const struct iovec *iov;
			size_t		count;
			void		*desc;
			fi_addr_t	dest_addr;
			uint64_t	addr;
			uint64_t	key;
			void		*context;
			uint64_t	flags;
			uint64_t	data;
		} writev;
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
			const struct iovec *iov;
			size_t		count;
			void		*desc;
			fi_addr_t	src_addr;
			uint64_t	addr;
			uint64_t	key;
			void		*context;
			uint64_t	flags;
		} readv;
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
			const struct fi_ioc *iov;
			size_t		count;
			void		*desc;
			fi_addr_t	dest_addr;
			uint64_t	addr;
			uint64_t	key;
			enum fi_datatype datatype;
			enum fi_op	atomic_op;
			void		*context;
			uint64_t	flags;
		} atomic_writev;
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
			const struct fi_ioc *iov;
			size_t		count;
			void		**desc;
			struct fi_ioc	*resultv;
			void		**result_desc;
			size_t		result_count;
			fi_addr_t	dest_addr;
			uint64_t	addr;
			uint64_t	key;
			enum fi_datatype datatype;
			enum fi_op	atomic_op;
			void		*context;
			uint64_t	flags;
		} atomic_readwritev;
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
		struct {
			struct fid_ep	*ep;
			const struct fi_ioc *iov;
			size_t		count;
			void		**desc;
			const struct fi_ioc *comparev;
			void		**compare_desc;
			size_t		compare_count;
			struct fi_ioc	*resultv;
			void		**result_desc;
			size_t		result_count;
			fi_addr_t	dest_addr;
			uint64_t	addr;
			uint64_t	key;
			enum fi_datatype datatype;
			enum fi_op	atomic_op;
			void		*context;
			uint64_t	flags;
		} atomic_compwritev;
	};
	struct psmx2_trigger *next;	/* used for randomly accessed trigger list */
	struct slist_entry list_entry;	/* used for ready-to-fire trigger queue */
};

struct psmx2_fid_cntr {
	union {
		struct fid_cntr		cntr;
		struct util_cntr	util_cntr; /* for util_poll_run */
	};
	struct psmx2_fid_domain	*domain;
	int			events;
	uint64_t		flags;
	atomic_t		counter;
	atomic_t		error_counter;
	struct util_wait	*wait;
	int			wait_is_local;
	struct psmx2_trigger	*trigger;
	pthread_mutex_t		trigger_lock;
};

struct psmx2_fid_av {
	struct fid_av		av;
	struct psmx2_fid_domain	*domain;
	struct fid_eq		*eq;
	int			type;
	uint64_t		flags;
	size_t			addrlen;
	size_t			count;
	size_t			last;
	psm2_epid_t		*epids;
	psm2_epaddr_t		*epaddrs;
	uint8_t			*vlanes;
};

struct psmx2_fid_ep {
	struct fid_ep		ep;
	struct psmx2_fid_ep	*base_ep;
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
	uint8_t			vlane;
	unsigned		send_selective_completion:1;
	unsigned		recv_selective_completion:1;
	unsigned		enabled:1;
	uint64_t		tx_flags;
	uint64_t		rx_flags;
	uint64_t		caps;
	atomic_t		ref;
	struct fi_context	nocomp_send_context;
	struct fi_context	nocomp_recv_context;
	struct slist		free_context_list;
	fastlock_t		context_lock;
	size_t			min_multi_recv;
	uint32_t		iov_seq_num;
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
int	psmx2_ep_open(struct fid_domain *domain, struct fi_info *info,
		     struct fid_ep **ep, void *context);
int	psmx2_stx_ctx(struct fid_domain *domain, struct fi_tx_attr *attr,
		     struct fid_stx **stx, void *context);
int	psmx2_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		     struct fid_cq **cq, void *context);
int	psmx2_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		     struct fid_av **av, void *context);
int	psmx2_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		       struct fid_cntr **cntr, void *context);
int	psmx2_wait_open(struct fid_fabric *fabric, struct fi_wait_attr *attr,
			struct fid_wait **waitset);
int	psmx2_wait_trywait(struct fid_fabric *fabric, struct fid **fids,
			   int count);

static inline void psmx2_fabric_acquire(struct psmx2_fid_fabric *fabric)
{
	atomic_inc(&fabric->util_fabric.ref);
}

static inline void psmx2_fabric_release(struct psmx2_fid_fabric *fabric)
{
	atomic_dec(&fabric->util_fabric.ref);
}

static inline void psmx2_domain_acquire(struct psmx2_fid_domain *domain)
{
	atomic_inc(&domain->util_domain.ref);
}

static inline void psmx2_domain_release(struct psmx2_fid_domain *domain)
{
	atomic_dec(&domain->util_domain.ref);
}

int	psmx2_domain_check_features(struct psmx2_fid_domain *domain, int ep_cap);
int	psmx2_domain_enable_ep(struct psmx2_fid_domain *domain, struct psmx2_fid_ep *ep);
void 	*psmx2_name_server(void *args);
void	*psmx2_resolve_name(const char *servername, int port);
void	psmx2_get_uuid(psm2_uuid_t uuid);
int	psmx2_uuid_to_port(psm2_uuid_t uuid);
char	*psmx2_uuid_to_string(psm2_uuid_t uuid);
int	psmx2_errno(int err);
int	psmx2_epid_to_epaddr(struct psmx2_fid_domain *domain,
			    psm2_epid_t epid, psm2_epaddr_t *epaddr);
void	psmx2_query_mpi(void);

struct	fi_context *psmx2_ep_get_op_context(struct psmx2_fid_ep *ep);
void	psmx2_ep_put_op_context(struct psmx2_fid_ep *ep, struct fi_context *fi_context);
void	psmx2_cq_enqueue_event(struct psmx2_fid_cq *cq, struct psmx2_cq_event *event);
struct	psmx2_cq_event *psmx2_cq_create_event(struct psmx2_fid_cq *cq,
					void *op_context, void *buf,
					uint64_t flags, size_t len,
					uint64_t data, uint64_t tag,
					size_t olen, int err);
int	psmx2_cq_poll_mq(struct psmx2_fid_cq *cq, struct psmx2_fid_domain *domain,
			struct psmx2_cq_event *event, int count, fi_addr_t *src_addr);

int	psmx2_am_init(struct psmx2_fid_domain *domain);
int	psmx2_am_fini(struct psmx2_fid_domain *domain);
int	psmx2_am_progress(struct psmx2_fid_domain *domain);
int	psmx2_am_process_send(struct psmx2_fid_domain *domain,
				struct psmx2_am_request *req);
int	psmx2_am_process_rma(struct psmx2_fid_domain *domain,
				struct psmx2_am_request *req);
int	psmx2_process_trigger(struct psmx2_fid_domain *domain,
				struct psmx2_trigger *trigger);
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

int	psmx2_handle_sendv_req(struct psmx2_fid_ep *ep, psm2_mq_status2_t *psm2_status,
			       int multi_recv);

static inline void psmx2_cntr_inc(struct psmx2_fid_cntr *cntr)
{
	atomic_inc(&cntr->counter);
	psmx2_cntr_check_trigger(cntr);
	if (cntr->wait)
		cntr->wait->signal(cntr->wait);
}

static inline void psmx2_progress(struct psmx2_fid_domain *domain)
{
	if (domain) {
		psmx2_cq_poll_mq(NULL, domain, NULL, 0, NULL);
		if (domain->am_initialized)
			psmx2_am_progress(domain);
	}
}

/* The following functions are used by triggered operations */

ssize_t psmx2_send_generic(
			struct fid_ep *ep,
			const void *buf, size_t len,
			void *desc, fi_addr_t dest_addr,
			void *context, uint64_t flags,
			uint64_t data);

ssize_t psmx2_sendv_generic(
			struct fid_ep *ep,
			const struct iovec *iov, void *desc,
			size_t count, fi_addr_t dest_addr,
			void *context, uint64_t flags,
			uint64_t data);

ssize_t psmx2_recv_generic(
			struct fid_ep *ep,
			void *buf, size_t len, void *desc,
			fi_addr_t src_addr, void *context,
			uint64_t flags);

ssize_t psmx2_tagged_send_generic(
			struct fid_ep *ep,
			const void *buf, size_t len,
			void *desc, fi_addr_t dest_addr,
			uint64_t tag, void *context,
			uint64_t flags, uint64_t data);

ssize_t psmx2_tagged_sendv_generic(
			struct fid_ep *ep,
			const struct iovec *iov, void *desc,
			size_t count, fi_addr_t dest_addr,
			uint64_t tag, void *context,
			uint64_t flags, uint64_t data);

ssize_t psmx2_tagged_recv_generic(
			struct fid_ep *ep,
			void *buf, size_t len,
			void *desc, fi_addr_t src_addr,
			uint64_t tag, uint64_t ignore,
			void *context, uint64_t flags);

ssize_t psmx2_write_generic(
			struct fid_ep *ep,
			const void *buf, size_t len,
			void *desc, fi_addr_t dest_addr,
			uint64_t addr, uint64_t key,
			void *context, uint64_t flags,
			uint64_t data);

ssize_t psmx2_writev_generic(
			struct fid_ep *ep,
			const struct iovec *iov, void **desc,
			size_t count, fi_addr_t dest_addr,
			uint64_t addr, uint64_t key,
			void *context, uint64_t flags,
			uint64_t data);

ssize_t psmx2_read_generic(
			struct fid_ep *ep,
			void *buf, size_t len,
			void *desc, fi_addr_t src_addr,
			uint64_t addr, uint64_t key,
			void *context, uint64_t flags);

ssize_t psmx2_readv_generic(
			struct fid_ep *ep,
			const struct iovec *iov, void *desc,
			size_t count, fi_addr_t src_addr,
			uint64_t addr, uint64_t key,
			void *context, uint64_t flags);

ssize_t psmx2_atomic_write_generic(
			struct fid_ep *ep,
			const void *buf,
			size_t count, void *desc,
			fi_addr_t dest_addr,
			uint64_t addr, uint64_t key,
			enum fi_datatype datatype,
			enum fi_op op, void *context,
			uint64_t flags);

ssize_t psmx2_atomic_readwrite_generic(
			struct fid_ep *ep,
			const void *buf,
			size_t count, void *desc,
			void *result, void *result_desc,
			fi_addr_t dest_addr,
			uint64_t addr, uint64_t key,
			enum fi_datatype datatype,
			enum fi_op op, void *context,
			uint64_t flags);

ssize_t psmx2_atomic_compwrite_generic(
			struct fid_ep *ep,
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

