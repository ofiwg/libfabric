/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_IF_H_
#define _CXIP_IF_H_

#include <ofi_atom.h>
#include <ofi_list.h>
#include <ofi_lock.h>
#include <pthread.h>
#include <stdint.h>

/* Type definitions */
struct cxip_if {
	struct slist_entry if_entry;

	/* Device description */
	struct cxil_devinfo *info;
	int speed;
	int link;

	struct cxil_dev *dev;

	/* PtlTEs (searched during state change events) */
	struct dlist_entry ptes;

	ofi_atomic32_t ref;
	ofi_spin_t lock;
};

struct cxip_remap_cp {
	struct dlist_entry remap_entry;
	struct cxi_cp remap_cp;
	struct cxi_cp *hw_cp;
};

struct cxip_lni {
	struct cxip_if *iface;
	struct cxil_lni *lni;

	/* Hardware communication profiles */
	struct cxi_cp *hw_cps[MAX_HW_CPS];
	int n_cps;

	/* Software remapped communication profiles. */
	struct dlist_entry remap_cps;

	pthread_rwlock_t cp_lock;
};

/* Function declarations */
struct cxip_if *cxip_if_lookup_addr(uint32_t nic_addr);

struct cxip_if *cxip_if_lookup_name(const char *name);

int cxip_get_if(uint32_t nic_addr, struct cxip_if **dev_if);

void cxip_put_if(struct cxip_if *dev_if);

int cxip_if_valid_rgroup_vni(struct cxip_if *iface, unsigned int rgroup_id,
			     unsigned int vni);

int cxip_alloc_lni(struct cxip_if *iface, uint32_t svc_id,
		   struct cxip_lni **if_lni);

void cxip_free_lni(struct cxip_lni *lni);

const char *cxi_tc_str(enum cxi_traffic_class tc);

void cxip_if_init(void);

void cxip_if_fini(void);

#endif /* _CXIP_IF_H_ */
