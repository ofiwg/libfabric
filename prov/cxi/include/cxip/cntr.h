/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_CNTR_H_
#define _CXIP_CNTR_H_

#include <ofi_atom.h>
#include <ofi_list.h>
#include <stdbool.h>
#include <stdint.h>

/* Forward declarations */
struct cxip_cmdq;
struct cxip_domain;

/* Type definitions */
struct cxip_cntr {
	struct fid_cntr cntr_fid;
	struct cxip_domain *domain; // parent domain
	ofi_atomic32_t ref;
	struct fi_cntr_attr attr; // copy of user or default attributes
	struct fid_wait *wait;
	/* Contexts to which counter is bound */
	struct dlist_entry ctx_list;

	/* Triggered cmdq for bound counters */
	struct cxip_cmdq *trig_cmdq;

	struct ofi_genlock lock;

	struct cxi_ct *ct;
	struct c_ct_writeback *wb;
	uint64_t wb_device;
	enum fi_hmem_iface wb_iface;
	uint64_t wb_handle;
	bool wb_handle_valid;
	struct c_ct_writeback lwb;

	struct dlist_entry dom_entry;

	/* Counter for number of operations which need progress. A separate lock
	 * is needed since these functions may be called without counter lock
	 * held.
	 */
	struct ofi_genlock progress_count_lock;
	int progress_count;
};

/* Function declarations */
int cxip_cntr_mod(struct cxip_cntr *cxi_cntr, uint64_t value, bool set,
		  bool err);

int cxip_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		   struct fid_cntr **cntr, void *context);

#endif /* _CXIP_CNTR_H_ */
