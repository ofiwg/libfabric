/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_MR_H_
#define _CXIP_MR_H_


#include <stdint.h>
#include <stdbool.h>
#include <ofi_list.h>
#include <ofi_atom.h>
#include <ofi_lock.h>

/* Forward declarations */
struct cxip_cntr;
struct cxip_domain;
struct cxip_ep;
struct cxip_pte;

/* Macros */
#define CXIP_MR_CACHE_EVENTS_DISABLE_POLL_NSECS 100000U

#define CXIP_MR_CACHE_EVENTS_DISABLE_LE_POLL_NSECS 1000000000U

#define CXIP_MR_PROV_KEY_MASK ((1ULL << 61) - 1)

#define CXIP_MR_PROV_KEY_ID_MASK ((1ULL << 16) - 1)

#define CXIP_MR_UNCACHED_KEY_TO_IDX(key) ((key) & CXIP_MR_PROV_KEY_ID_MASK)

#define CXIP_MR_KEY_SIZE sizeof(uint32_t)

#define CXIP_MR_KEY_MASK ((1ULL << (8 * CXIP_MR_KEY_SIZE)) - 1)

#define CXIP_MR_VALID_OFFSET_MASK ((1ULL << 56) - 1)

#define CXIP_MR_PROV_KEY_SIZE sizeof(struct cxip_mr_key)

#define CXIP_MR_DOMAIN_HT_BUCKETS 16

/* Type definitions */
struct cxip_mr_key {
	union {
		/* Provider generated standard cached */
		struct {
			uint64_t lac	: 3;
			uint64_t lac_off: 58;
			uint64_t opt	: 1;
			uint64_t cached	: 1;
			uint64_t unused1: 1;
			/* shares CXIP_CTRL_LE_TYPE_MR */
		};
		/* Client or Provider non-cached */
		struct {
			uint64_t key	: 61;
			uint64_t unused2: 3;
			/* Provider shares opt */
			/* Provider shares cached == 0 */
			/* Provider shares CXIP_CTRL_LE_TYPE_MR */
		};
		/* Provider Key Only */
		struct {
			/* Non-cached key consists of unique MR ID and sequence
			 * number. The same MR ID can be used with sequence
			 * number to create 2^44 unique keys. That is, a
			 * single standard MR repeatedly created and destroyed
			 * every micro-second, would take months before
			 * it repeated.
			 */
			uint64_t id     : 16;  /* Unique - 64K MR */
			uint64_t seqnum : 44;  /* Sequence with random seed */
			uint64_t events : 1;   /* Requires event generation */
			uint64_t unused3: 2;
			uint64_t is_prov: 1;
			/* Overloads CXIP_CTRL_LE_TYPE_MR and must be cleared
			 * before appending MR LE or TX using in match bits.
			 */
		};
		uint64_t raw;
	};
};

struct cxip_mr_util_ops {
	bool is_cached;
	int (*init_key)(struct cxip_mr *mr, uint64_t req_key);
	int (*enable_opt)(struct cxip_mr *mr);
	int (*disable_opt)(struct cxip_mr *mr);
	int (*enable_std)(struct cxip_mr *mr);
	int (*disable_std)(struct cxip_mr *mr);
};

struct cxip_md {
	struct cxip_domain *dom;
	struct cxi_md *md;
	struct ofi_mr_info info;
	uint64_t map_flags;
	uint64_t handle;
	int dmabuf_fd;
	bool handle_valid;
	bool cached;
	bool dmabuf_fd_valid;
};

struct cxip_mr_domain {
	struct dlist_entry buckets[CXIP_MR_DOMAIN_HT_BUCKETS];
	ofi_spin_t lock;
};

struct cxip_mr {
	struct fid_mr mr_fid;
	struct cxip_domain *domain;	// parent domain
	struct cxip_ep *ep;		// endpoint for remote memory
	uint64_t key;			// memory key
	uint64_t flags;			// special flags
	struct fi_mr_attr attr;		// attributes
	struct cxip_cntr *cntr;		// if bound to cntr

	/* Indicates if FI_RMA_EVENT was specified at creation and
	 * will be used to enable fi_writedata() and fi_inject_writedata()
	 * support for this MR (TODO).
	 */
	bool rma_events;

	/* If requested then count MR events to determine if RMA are in
	 * progress. At close if no RMA are in progress bypass the invalidate
	 * of the PTLTE LE. This improves non-cached key close performance,
	 * enabling their use so that after closing the MR the associated
	 * memory cannot be remotely accessed, even if it remains in the
	 * libfabric MR cache.
	 */
	bool count_events;
	ofi_atomic32_t  match_events;
	ofi_atomic32_t  access_events;

	ofi_spin_t lock;

	struct cxip_mr_util_ops *mr_util;
	bool enabled;
	struct cxip_pte *pte;
	enum cxip_mr_state mr_state;
	int64_t mr_id;			// Non-cached provider key uniqueness
	struct cxip_ctrl_req req;
	bool optimized;

	void *buf;			// memory buffer VA
	uint64_t len;			// memory length
	struct cxip_md *md;		// buffer IO descriptor
	struct dlist_entry ep_entry;

	struct dlist_entry mr_domain_entry;
};

/* Function declarations */
int cxip_generic_mr_key_to_ptl_idx(struct cxip_domain *dom,
				   uint64_t key, bool write);

bool cxip_generic_is_mr_key_opt(uint64_t key);

bool cxip_generic_is_mr_key_events(uint64_t caps, uint64_t key);

bool cxip_generic_is_valid_mr_key(uint64_t key);

void cxip_mr_domain_init(struct cxip_mr_domain *mr_domain);

void cxip_mr_domain_fini(struct cxip_mr_domain *mr_domain);

#endif /* _CXIP_MR_H_ */
