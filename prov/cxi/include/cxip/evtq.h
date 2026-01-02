/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_EVTQ_H_
#define _CXIP_EVTQ_H_

#include <ofi_list.h>
#include <stdbool.h>
#include <stddef.h>

/* Forward declarations */
struct cxip_cq;
struct cxip_req;

/* Type definitions */
struct cxip_evtq {
	struct cxi_eq *eq;
	void *buf;
	size_t len;
	struct cxi_md *md;
	bool mmap;
	unsigned int unacked_events;
	unsigned int ack_batch_size;
	struct c_eq_status prev_eq_status;
	bool eq_saturated;
	/* Reference to wait_obj allocated outside scope of event queue */
	struct cxil_wait_obj *event_wait_obj;
	struct cxil_wait_obj *status_wait_obj;

	/* Point back to CQ */
	struct cxip_cq *cq;

	/* Protected with ep_ob->lock */
	struct ofi_bufpool *req_pool;
	struct indexer req_table;
	struct dlist_entry req_list;

	/* CQ completion batching state.
	 * When cq_batching_active is true, completions are added to
	 * the batch array instead of being written immediately.
	 * The batch is flushed at end of progress or when full.
	 */
	unsigned int cq_batch_size;
	unsigned int cq_batch_count;
	bool cq_batching_active;
	struct cxip_cq_batch_entry cq_batch[CXIP_CQ_BATCH_MAX];
};

struct def_event_ht {
	struct dlist_entry bh[CXIP_DEF_EVENT_HT_BUCKETS];
};

/* Function declarations */
int cxip_evtq_init(struct cxip_evtq *evtq, struct cxip_cq *cq,
		   size_t num_events, size_t num_fc_events,
		   struct cxil_wait_obj *priv_wait);

void cxip_evtq_fini(struct cxip_evtq *eq);

bool cxip_evtq_saturated(struct cxip_evtq *evtq);

int cxip_evtq_req_cancel(struct cxip_evtq *evtq, void *req_ctx, void *op_ctx,
			 bool match);

void cxip_evtq_req_discard(struct cxip_evtq *evtq, void *req_ctx);

void cxip_evtq_flush_trig_reqs(struct cxip_evtq *evtq);

struct cxip_req *cxip_evtq_req_alloc(struct cxip_evtq *evtq, int remap,
				     void *req_ctx);

void cxip_evtq_req_free(struct cxip_req *req);

void cxip_evtq_progress(struct cxip_evtq *evtq, bool internal);

int cxip_evtq_adjust_reserved_fc_event_slots(struct cxip_evtq *evtq, int value);

#endif /* _CXIP_EVTQ_H_ */
