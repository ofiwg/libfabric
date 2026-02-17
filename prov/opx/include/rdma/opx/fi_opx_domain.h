/*
 * Copyright (C) 2016 by Argonne National Laboratory.
 * Copyright (C) 2021-2026 Cornelis Networks.
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
#ifndef _FI_PROV_OPX_DOMAIN_H_
#define _FI_PROV_OPX_DOMAIN_H_

#include <unistd.h>
#include <stdint.h>
#include <pthread.h>
#include <uuid/uuid.h>
#include <uthash.h>

#include "rdma/fi_domain.h"
#include "ofi_atom.h"

#include "rdma/opx/fi_opx_reliability.h"

#include "rdma/opx/fi_opx_tid_domain.h"

#include "rdma/opx/opx_hmem_domain.h"
#include "rdma/opx/opx_hfisvc_keyset.h"

#if HAVE_HFISVC
#include <infiniband/hfi1dv.h>
#include <infiniband/verbs.h>
#include <infiniband/hfisvc_client.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct fi_opx_ep; /* forward declaration */

struct opx_tid_fabric;
struct opx_hmem_fabric;

struct fi_opx_fabric {
	struct fid_fabric fabric_fid;

	ofi_atomic64_t	       ref_cnt;
	struct opx_tid_fabric *tid_fabric;
#ifdef OPX_HMEM
	struct opx_hmem_fabric *hmem_fabric;
#endif
};

struct fi_opx_node {
	volatile uint64_t ep_count;
};

#define OPX_JOB_KEY_STR_SIZE	33
#define OPX_DEFAULT_JOB_KEY_STR "00112233445566778899aabbccddeeff"

#define OPX_SDMA_BOUNCE_BUF_MIN	      FI_OPX_SDMA_MIN_PAYLOAD_BYTES_MIN
#define OPX_SDMA_BOUNCE_BUF_THRESHOLD FI_OPX_SDMA_MIN_PAYLOAD_BYTES_DEFAULT
#define OPX_SDMA_BOUNCE_BUF_MAX	      FI_OPX_SDMA_MIN_PAYLOAD_BYTES_MAX

struct fi_opx_domain {
	struct fid_domain     domain_fid;
	struct fi_opx_fabric *fabric;

	enum fi_threading     threading;
	enum fi_resource_mgmt resource_mgmt;
	int		      mr_mode;
	enum fi_progress      data_progress;

	uuid_t unique_job_key;
	char   unique_job_key_str[OPX_JOB_KEY_STR_SIZE];

	char *progress_affinity_str;

	uint32_t rx_count;
	uint32_t tx_count;
	uint8_t	 ep_count;

	uint64_t	  num_mr_keys;
	struct fi_opx_mr *mr_hashmap;

	struct opx_tid_domain *tid_domain;
#ifdef OPX_HMEM
	struct opx_hmem_domain *hmem_domain;
#endif

#if HAVE_HFISVC
	struct {
		struct ibv_context *ctx;
		/**
		 * @brief Command queue used by the domain for issuing commands to the
		 * hfisvc where we are opening/closing hfisvc memory regions.
		 */
		hfisvc_client_command_queue_t mr_command_queue;

		/**
		 * @brief Completion queue used by the domain for handling completions from the
		 * hfisvc where we are opening/closing hfisvc memory regions.
		 */
		hfisvc_client_completion_queue_t mr_completion_queue;
		opx_hfisvc_keyset_t		 access_key_set;
		ofi_atomic64_t			 ref_cnt;
		hfisvc_client_key_t		 client_key;
		uint32_t			 padding;
		void				*libhfi1verbs;
		int (*initialize)(struct ibv_context *ctx);
		int (*get_client_key)(struct ibv_context *ctx, hfisvc_client_key_t *key);
		int (*command_queue_open)(hfisvc_client_command_queue_t *command_queue, struct ibv_context *ctx);
		int (*command_queue_close)(hfisvc_client_command_queue_t *command_queue);
		int (*completion_queue_open)(hfisvc_client_completion_queue_t *completion_queue,
					     struct ibv_context		      *ctx);
		int (*completion_queue_close)(hfisvc_client_completion_queue_t *completion_queue);
		size_t (*cq_read)(hfisvc_client_completion_queue_t completion_queue, uint64_t flags,
				  struct hfisvc_client_cq_entry *buf, size_t buf_size_bytes, size_t count);
		int (*cmd_dma_access_once_va)(hfisvc_client_command_queue_t   command_queue,
					      struct hfisvc_client_completion completion, uint64_t flags,
					      uint32_t access_key, uint32_t len, void *vaddr);
		int (*cmd_rdma_read_va)(hfisvc_client_command_queue_t	command_queue,
					struct hfisvc_client_completion completion, uint64_t flags, uint32_t lid,
					hfisvc_client_key_t client, uint32_t len, uint64_t imm_data,
					uint32_t access_key, uint64_t remote_offset, void *vaddr);
		int (*doorbell)(struct ibv_context *ctx);
	} hfisvc;
#endif
	uint8_t	       use_hfisvc;
	uint8_t	       padding[7];
	ofi_atomic64_t ref_cnt;

	struct slist	    deferred_work_queue;
	struct ofi_bufpool *deferred_work_pool;
};

struct fi_opx_av {
	/* == CACHE LINE 0 == */

	struct fid_av	      av_fid; /* 32 bytes */
	struct fi_opx_domain *domain;
	void		     *map_addr;
	ofi_atomic64_t	      ref_cnt;
	uint32_t	      addr_count;
	enum fi_av_type	      type;
	unsigned	      ep_tx_count;

	/* == CACHE LINE 1..20 == */

	struct fi_opx_ep *ep_tx[160];

	/* == ALL OTHER CACHE LINES == */

	union fi_opx_addr *table_addr; /* allocated buffer to free */
	uint64_t	   rx_ctx_bits;
	uint32_t	   table_count; /* table, not av, count */
};

/**
 * Keeps track of the state of the MR with regards to HFI service.
 *
 * Each state indicates the following:
 *
 * NOT_REGISTERED      : The MR has not been registered with HFI service at all.
 * PENDING_OPEN        : A request to register this MR with HFI service has been submitted, and we're waiting
 *                       for a completion from HFI service to let us know it's done.
 * PENDING_KEY_ALLOC   : This MR is registered with HFI service, but does not yet have an access_key assigned
 *                       and registered with HFI service.
 * PENDING_KEY_ENABLE  : An access_key has been assigned to this MR, and a request to enable DMA access for
 *                       the key has been submitted to HFI service, and we're waiting for a completion from
 *                       HFI service to let us know it's done.
 * OPENED              : This MR is registered with HFI service, and may be used for DMA operations in HFI service
 *                       using its access_key.
 * PENDING_KEY_DISABLE : A request to deregister the access_key associated with this MR has been submitted to
 *                       HFI service, and we're waiting for a completion from HFI service to let us know it's done.
 * PENDING_DEREGISTER  : The access_key has been successfully deregistered from HFI service and freed, and now
 *                       we need to deregister the MR from HFI service.
 * PENDING_CLOSE       : A request to deregister this MR with HFI service has been submitted, and we're waiting
 *                       for a completion from HFI service to let us know it's done.
 * CLOSED              : The MR has been fully deregisterd with HFI service, and may be closed/freed.
 */
enum opx_mr_hfisvc_state {
	OPX_MR_HFISVC_STATE_NOT_REGISTERED = 0,
	OPX_MR_HFISVC_STATE_PENDING_OPEN,
	OPX_MR_HFISVC_STATE_PENDING_KEY_ALLOC,
	OPX_MR_HFISVC_STATE_PENDING_KEY_ENABLE,
	OPX_MR_HFISVC_STATE_OPENED,
	OPX_MR_HFISVC_STATE_PENDING_KEY_DISABLE,
	OPX_MR_HFISVC_STATE_PENDING_DEREGISTER,
	OPX_MR_HFISVC_STATE_PENDING_CLOSE,
	OPX_MR_HFISVC_STATE_CLOSED,
	OPX_MR_HFISVC_STATE_CLOSE_ISSUED = 0x80000000,
};

struct fi_opx_mr {
	/* == CACHE LINE 0-2 == */
	struct fid_mr	      mr_fid; // 40 bytes
	struct fi_mr_attr     attr;   // 112 bytes
	struct fi_opx_domain *domain;
	void		     *base_addr;
	uint64_t	      flags;
	uint64_t	      cntr_bflags;
	struct fi_opx_cntr   *cntr;

	/* == CACHE LINE 3 == */
	union {
		struct {
			int	 dmabuf_fd;
			uint32_t reserved_dw;
			uint64_t reserved_qw[3];
		};
		struct {
			uint32_t     reserved_fd;
			uint32_t     unused_pad[3];
			struct iovec iov;
		};
		struct fi_mr_dmabuf dmabuf;
	};
	struct {
		union {
			uint32_t reserved;
#ifdef HFISVC
			hfisvc_client_mr_t mr_handle;
#endif
		};
		enum opx_mr_hfisvc_state state;
		uint32_t		 access_key;
		uint32_t		 padding;
	} hfisvc;
	uint8_t		     hmem_unified;
	uint8_t		     dmabuf_internal; /* 1 if OPX created dmabuf fd and must close it */
	uint8_t		     unused[6];
	struct ofi_mr_entry *cache_entry;

	/* == CACHE LINE 4 == */
	UT_hash_handle hh; // 56 bytes
	uint64_t       unused_cacheline4_qw;
} __attribute__((__aligned__(FI_OPX_CACHE_LINE_SIZE))) __attribute__((__packed__));
OPX_COMPILE_TIME_ASSERT(sizeof(struct fi_opx_mr) == (FI_OPX_CACHE_LINE_SIZE * 5),
			"Size of fi_opx_mr should be 5 cachelines!");
OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_mr, dmabuf_fd) == (FI_OPX_CACHE_LINE_SIZE * 3),
			"Offset of fi_opx_mr->dmabuf_fd should start at cacheline 3!");
OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_mr, dmabuf_fd) == offsetof(struct fi_opx_mr, dmabuf.fd),
			"Offset of fi_opx_mr->dmabuf_fd should start at the same point as fi_opx_mr->dmabuf.fd!");
OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_mr, hh) == (FI_OPX_CACHE_LINE_SIZE * 4),
			"Offset of fi_opx_mr->hh should start at cacheline 4!");

static inline uint64_t opx_mr_dmabuf_local_offset(const struct fi_opx_mr *opx_mr, const void *buf)
{
	assert((uintptr_t) (buf) >= (uintptr_t) (opx_mr->dmabuf.base_addr));
	return (uint64_t) ((uintptr_t) buf - (uintptr_t) opx_mr->dmabuf.base_addr);
}

struct opx_domain_deferred_work {
	union {
		struct slist_entry		 slist_entry;
		struct opx_domain_deferred_work *next;
	};
	int (*work_fn)(struct opx_domain_deferred_work *work);
	struct fi_opx_mr *opx_mr;
	uint64_t	  unused;
} __attribute__((__aligned__(32))) __attribute__((__packed__));

static inline uint32_t fi_opx_domain_get_tx_max(struct fid_domain *domain)
{
	return 160;
}

static inline uint32_t fi_opx_domain_get_rx_max(struct fid_domain *domain)
{
	return 160;
}

int opx_hfisvc_mr_deferred_open(struct opx_domain_deferred_work *work);
int opx_hfisvc_mr_deferred_close(struct opx_domain_deferred_work *work);

#if HAVE_HFISVC
__OPX_FORCE_INLINE__
int opx_domain_deferred_work_enqueue(struct fi_opx_domain *opx_domain, struct fi_opx_mr *opx_mr,
				     int (*work_fn)(struct opx_domain_deferred_work *work))
{
	struct opx_domain_deferred_work *work = ofi_buf_alloc(opx_domain->deferred_work_pool);
	if (OFI_UNLIKELY(work == NULL)) {
		FI_WARN(fi_opx_global.prov, FI_LOG_DOMAIN, "Error allocating deferred work for hfisvc mr open\n");
		return -FI_ENOMEM;
	}
	work->next    = NULL;
	work->opx_mr  = opx_mr;
	work->work_fn = work_fn;

	int ret = work_fn(work);
	if (OFI_UNLIKELY(ret == FI_SUCCESS)) {
		OPX_BUF_FREE(work);
	} else {
		slist_insert_tail((struct slist_entry *) work, &opx_domain->deferred_work_queue);
	}

	return FI_SUCCESS;
}

__OPX_FORCE_INLINE__
int opx_domain_deferred_work_enqueue_open(struct fi_opx_domain *opx_domain, struct fi_opx_mr *opx_mr)
{
	return opx_domain_deferred_work_enqueue(opx_domain, opx_mr, opx_hfisvc_mr_deferred_open);
}

__OPX_FORCE_INLINE__
int opx_domain_deferred_work_enqueue_close(struct fi_opx_domain *opx_domain, struct fi_opx_mr *opx_mr)
{
	return opx_domain_deferred_work_enqueue(opx_domain, opx_mr, opx_hfisvc_mr_deferred_close);
}

__OPX_FORCE_INLINE__
void opx_domain_deferred_work_do(struct fi_opx_domain *opx_domain)
{
	struct opx_domain_deferred_work *prev_item = NULL;
	struct opx_domain_deferred_work *work_item =
		(struct opx_domain_deferred_work *) opx_domain->deferred_work_queue.head;

	while (work_item) {
		int ret = (work_item->work_fn)(work_item);

		if (ret == FI_SUCCESS) {
			struct opx_domain_deferred_work *next_item = work_item->next;
			slist_remove(&opx_domain->deferred_work_queue, (struct slist_entry *) work_item,
				     (struct slist_entry *) prev_item);
			OPX_BUF_FREE(work_item);
			work_item = next_item;
		} else {
			prev_item = work_item;
			work_item = work_item->next;
		}
	}
}

__OPX_FORCE_INLINE__
void opx_domain_hfisvc_poll(struct fi_opx_domain *opx_domain)
{
	struct hfisvc_client_cq_entry hfisvc_out[64];
	size_t n = (*opx_domain->hfisvc.cq_read)(opx_domain->hfisvc.mr_completion_queue, 0ul /* flags */, hfisvc_out,
						 sizeof(struct hfisvc_client_cq_entry) * 64, 64);
	while (n > 0) {
		OPX_HFISVC_DEBUG_LOG("HFIService: Polled %lu completions from mr_completion_queue!\n", n);
		for (size_t i = 0; i < n; ++i) {
			if (hfisvc_out[i].status != HFISVC_CLIENT_CQ_ENTRY_STATUS_SUCCESS) {
				// TODO: FI_WARN, post some kind of error to the error queue
				fprintf(stderr,
					"(%d) %s:%s():%d Completion error: status was %d type=%d app_context=%lX\n",
					getpid(), __FILE__, __func__, __LINE__, hfisvc_out[i].status,
					hfisvc_out[i].type, hfisvc_out[i].app_context);
				abort();
			}
			struct fi_opx_mr  *opx_mr    = (struct fi_opx_mr *) hfisvc_out[i].app_context;
			hfisvc_client_mr_t mr_handle = hfisvc_out[i].type_mr.mr;

			assert(hfisvc_out[i].status == HFISVC_CLIENT_CQ_ENTRY_STATUS_SUCCESS);

			if (opx_mr->hfisvc.state == OPX_MR_HFISVC_STATE_PENDING_OPEN) {
				OPX_HFISVC_DEBUG_LOG(
					"MR State transition opx_mr=%p hfisvc.mr_handle=%u state=PENDING_OPEN -> KEY_ALLOC\n",
					opx_mr, (uint32_t) mr_handle);
				opx_mr->hfisvc.mr_handle = mr_handle;
				opx_mr->hfisvc.state	 = OPX_MR_HFISVC_STATE_PENDING_KEY_ALLOC;
			} else if (opx_mr->hfisvc.state == OPX_MR_HFISVC_STATE_PENDING_KEY_ENABLE) {
				OPX_HFISVC_DEBUG_LOG(
					"MR State transition opx_mr=%p state=PENDING_KEY_ENABLE -> OPENED\n", opx_mr);
				opx_mr->hfisvc.state = OPX_MR_HFISVC_STATE_OPENED;
			} else if (opx_mr->hfisvc.state == OPX_MR_HFISVC_STATE_OPENED) {
				assert(hfisvc_out[i].type_notify.access_key == opx_mr->hfisvc.access_key);
				OPX_HFISVC_DEBUG_LOG("Notify completion opx_mr=%p imm_data=%lX\n", opx_mr,
						     hfisvc_out[i].type_notify.imm_data);

				// TODO: Use generic struct for hvisvc mr completion
				struct opx_hfisvc_rzv_completion_tmp {
					struct opx_context *context;
					union {
						struct {
							uint64_t tid_length;
							uint64_t tid_vaddr;
						};
						struct {
							// uintptr_t app_context;
							uint64_t unused;
							uint32_t access_key;
							uint32_t unused_also;
						};
					};
					uint64_t byte_counter;
					uint64_t bytes_accumulated;
				} *rzv_comp =
					(struct opx_hfisvc_rzv_completion_tmp *) hfisvc_out[i].type_notify.imm_data;

				struct opx_context *context = rzv_comp->context;
				// TODO: Once hfisvc_client provides xfer_len in completion, we'll know how much to
				//       decrement from the context->byte_counter. Until then, just zero out
				//       context->byte_counter
				// uint64_t completed_len = hfisvc_out[i].type_default.xfer_len;
				uint64_t completed_len = context->byte_counter;
				assert(completed_len <= context->byte_counter);
				OPX_HFISVC_DEBUG_LOG(
					"Got completion entry for context=%p completed_len=%lu byte_counter=%lu -> %lu\n",
					context, completed_len, context->byte_counter,
					context->byte_counter - completed_len);

				context->byte_counter -= completed_len;

				/* free the rendezvous completion structure */
				OPX_BUF_FREE(rzv_comp);
			} else if (opx_mr->hfisvc.state == OPX_MR_HFISVC_STATE_PENDING_KEY_DISABLE) {
				opx_hfisvc_keyset_free_key(opx_domain->hfisvc.access_key_set, opx_mr->hfisvc.access_key,
							   NULL);
				opx_mr->hfisvc.access_key = (uint32_t) -1;
				OPX_HFISVC_DEBUG_LOG(
					"MR State transition opx_mr=%p state=PENDING_KEY_DISABLE -> PENDING_DEREGISTER\n",
					opx_mr);
				opx_mr->hfisvc.state = OPX_MR_HFISVC_STATE_PENDING_DEREGISTER;
			} else if (opx_mr->hfisvc.state == OPX_MR_HFISVC_STATE_PENDING_DEREGISTER) {
				OPX_HFISVC_DEBUG_LOG(
					"MR State transition opx_mr=%p state=PENDING_DEREGISTER -> PENDING_CLOSE\n",
					opx_mr);
				opx_mr->hfisvc.state = OPX_MR_HFISVC_STATE_PENDING_CLOSE;
			} else if (opx_mr->hfisvc.state == OPX_MR_HFISVC_STATE_PENDING_CLOSE) {
				OPX_HFISVC_DEBUG_LOG("MR State transition opx_mr=%p state=PENDING_CLOSE -> CLOSED\n",
						     opx_mr);
				assert(opx_mr->hfisvc.mr_handle == mr_handle);
				opx_mr->hfisvc.state = OPX_MR_HFISVC_STATE_CLOSED;
			} else if (opx_mr->hfisvc.state & OPX_MR_HFISVC_STATE_CLOSE_ISSUED) {
				if (opx_mr->hfisvc.state ==
				    (OPX_MR_HFISVC_STATE_PENDING_OPEN | OPX_MR_HFISVC_STATE_CLOSE_ISSUED)) {
					OPX_HFISVC_DEBUG_LOG(
						"MR State transition opx_mr=%p hfisvc.mr_handle=%u state=PENDING_OPEN with CLOSE_ISSUED -> PENDING_DEREGISTER\n",
						opx_mr, (uint32_t) mr_handle);
					opx_mr->hfisvc.mr_handle = mr_handle;
					opx_mr->hfisvc.state	 = OPX_MR_HFISVC_STATE_PENDING_DEREGISTER;
				} else if (opx_mr->hfisvc.state == (OPX_MR_HFISVC_STATE_PENDING_KEY_ENABLE |
								    OPX_MR_HFISVC_STATE_CLOSE_ISSUED)) {
					OPX_HFISVC_DEBUG_LOG(
						"MR State transition opx_mr=%p state=PENDING_KEY_ENABLE with CLOSE_ISSUED -> OPENED\n",
						opx_mr);
					opx_mr->hfisvc.state = OPX_MR_HFISVC_STATE_OPENED;
				}
			} else {
				// TODO: FI_WARN, post some kind of error to the error queue
				fprintf(stderr,
					"(%d) %s:%s():%d Got unexpected completion for opx_mr=%p state=%d completion: type=%d status=%d\n",
					getpid(), __FILE__, __func__, __LINE__, opx_mr, opx_mr->hfisvc.state,
					hfisvc_out[i].type, hfisvc_out[i].status);
				assert(0);
			}
		}
		n = (*opx_domain->hfisvc.cq_read)(opx_domain->hfisvc.mr_completion_queue, 0ul /* flags */, hfisvc_out,
						  sizeof(struct hfisvc_client_cq_entry) * 64, 64);
	}

	opx_domain_deferred_work_do(opx_domain);
}

int opx_domain_hfisvc_init(struct fi_opx_domain *domain);
#endif

#ifdef __cplusplus
}
#endif

#endif /* _FI_PROV_OPX_DOMAIN_H_ */
