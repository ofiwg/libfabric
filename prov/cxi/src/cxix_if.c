/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include "config.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>

#include "ofi_prov.h"
#include "ofi_osd.h"

#include "cxi_prov.h"

#define CXIX_LOG_DBG(...) _CXI_LOG_DBG(FI_LOG_DOMAIN, __VA_ARGS__)
#define CXIX_LOG_ERROR(...) _CXI_LOG_ERROR(FI_LOG_DOMAIN, __VA_ARGS__)

struct slist cxix_if_list;
int cxix_num_pids;

struct cxix_if *cxix_if_lookup(uint32_t nic_addr)
{
	struct slist_entry *entry, *prev;
	struct cxix_if *if_entry;

	(void) prev; /* Makes compiler happy */
	slist_foreach(&cxix_if_list, entry, prev) {
		if_entry = container_of(entry, struct cxix_if, entry);
		if (if_entry->if_nic == nic_addr)
			return if_entry;
	}

	return NULL;
}

static struct cxix_if_domain *
cxix_if_domain_lookup(struct cxix_if *dev_if, uint32_t vni, uint32_t pid,
		      uint32_t pid_granule)
{
	struct dlist_entry *entry;
	struct cxix_if_domain *dom;

	dlist_foreach(&dev_if->if_doms, entry) {
		dom = container_of(entry, struct cxix_if_domain, entry);
		if (dom->vni == vni &&
		    dom->pid == pid &&
		    dom->pid_granule == pid_granule)
			return dom;
	}

	return NULL;
}

/*
 * cxix_get_if() - Get a reference to the interface associated with a provided
 * NIC address.  A process can open each interface once to support many FI
 * Domains.  An IF is used to allocate the various device resources including
 * CMDQs, EQs, PtlTEs.
 */
int cxix_get_if(uint32_t nic_addr, struct cxix_if **dev_if)
{
	struct cxix_if *if_entry;
	int ret;
	struct cxi_eq_alloc_opts evtq_opts;

	/* The IF list device info is static, no need to lock */
	if_entry = cxix_if_lookup(nic_addr);
	if (!if_entry) {
		CXIX_LOG_DBG("interface not found\n");
		return -FI_ENODEV;
	}

	/* Lock the IF to serialize opening the device */
	fastlock_acquire(&if_entry->lock);

	if (!if_entry->if_lni) {
		ret = cxil_open_device(if_entry->if_idx, &if_entry->if_dev);
		if (ret) {
			CXIX_LOG_DBG("cxil_open_device returned: %d\n", ret);
			ret = -FI_ENODEV;
			goto unlock;
		}

		ret = cxil_alloc_lni(if_entry->if_dev, 0, &if_entry->if_lni);
		if (ret) {
			CXIX_LOG_DBG("cxil_alloc_lni returned: %d\n", ret);
			ret = -FI_ENODEV;
			goto close_dev;
		}

		/* TODO Temporary fake CP setup, needed for CMDQ allocation */
		if_entry->n_cps = 1;
		if_entry->cps[0].vni = 0;
		if_entry->cps[0].dscp = 0;
		ret = cxil_set_cps(if_entry->if_lni, if_entry->cps,
				   if_entry->n_cps);
		assert(!ret);

		/* TODO Temporary allocation of CMDQ and EQ specifically for MR
		 * allocation.
		 */
		ret = cxil_alloc_cmdq(if_entry->if_lni, 64, 0,
				      &if_entry->mr_cmdq);
		if (ret != FI_SUCCESS) {
			CXIX_LOG_DBG("Unable to allocate MR CMDQ, ret: %d\n",
				     ret);
			ret = -FI_ENODEV;
			goto free_lni;
		}

		evtq_opts.count = 1024;
		evtq_opts.reserved_fc = 1;

		ret = cxil_alloc_evtq(if_entry->if_lni, &evtq_opts,
				      &if_entry->mr_evtq);
		if (ret != FI_SUCCESS) {
			CXIX_LOG_DBG("Unable to allocate MR EVTQ, ret: %d\n",
				     ret);
			ret = -FI_ENODEV;
			goto free_mr_cmdq;
		}

		CXIX_LOG_DBG("Allocated IF, NIC: %u ID: %u\n",
			     if_entry->if_nic, if_entry->if_idx);
	} else {
		CXIX_LOG_DBG("Using IF, NIC: %u ID: %u ref: %u\n",
			     if_entry->if_nic, if_entry->if_idx,
			     ofi_atomic_get32(&if_entry->ref));
	}

	ofi_atomic_inc32(&if_entry->ref);
	fastlock_release(&if_entry->lock);

	*dev_if = if_entry;

	return FI_SUCCESS;

free_mr_cmdq:
	ret = cxil_destroy_cmdq(if_entry->mr_cmdq);
	if (ret)
		CXIX_LOG_ERROR("Failed to destroy CMDQ: %d\n", ret);
free_lni:
	ret = cxil_destroy_lni(if_entry->if_lni);
	if (ret)
		CXIX_LOG_ERROR("Failed to destroy LNI: %d\n", ret);
	if_entry->if_lni = NULL;
close_dev:
	cxil_close_device(if_entry->if_dev);
	if_entry->if_dev = NULL;
unlock:
	fastlock_release(&if_entry->lock);

	return ret;
}

/*
 * cxix_put_if() - Drop a reference to the IF.
 */
void cxix_put_if(struct cxix_if *dev_if)
{
	int ret;

	fastlock_acquire(&dev_if->lock);

	if (!ofi_atomic_dec32(&dev_if->ref)) {
		cxil_destroy_evtq(dev_if->mr_evtq);

		ret = cxil_destroy_cmdq(dev_if->mr_cmdq);
		if (ret)
			CXIX_LOG_ERROR("Failed to destroy CMDQ: %d\n", ret);

		ret = cxil_destroy_lni(dev_if->if_lni);
		if (ret)
			CXIX_LOG_ERROR("Failed to destroy LNI: %d\n", ret);
		dev_if->if_lni = NULL;

		cxil_close_device(dev_if->if_dev);
		dev_if->if_dev = NULL;

		dev_if->n_cps = 0;

		CXIX_LOG_DBG("Released IF, NIC: %u ID: %u\n",
			     dev_if->if_nic, dev_if->if_idx);
	}

	dev_if->if_dev = NULL;

	fastlock_release(&dev_if->lock);
}

/*
 * cxix_get_if_domain() - Get a reference to an IF Domain.  An IF Domain
 * represents an address space of logical network endpoints (LEPs).  LEPs are
 * network addressable endpoints where a PtlTE resource can be mapped.  A PtlTE
 * is basically an RX queue.  The IF Domain address space is identified by the
 * three-tuple:
 *
 *    ( dev_if->if_nic, vni, pid )
 *
 * The IF Domain pid_granule value represents the address space size.  All IF
 * Domains created on a node (among all processes on the node) with matching
 * NIC and VNI values must use a matching pid_granule value.
 */
int cxix_get_if_domain(struct cxix_if *dev_if, uint32_t vni, uint32_t pid,
		       uint32_t pid_granule, struct cxix_if_domain **if_dom)
{
	struct cxix_if_domain *dom;
	int ret;

	fastlock_acquire(&dev_if->lock);

	dom = cxix_if_domain_lookup(dev_if, vni, pid, pid_granule);
	if (!dom) {
		dom = malloc(sizeof(*dom));
		if (!dom) {
			CXIX_LOG_DBG("Unable to allocate IF domain\n");
			ret = -FI_ENOMEM;
			goto unlock;
		}

		ret = cxil_alloc_domain(dev_if->if_lni, vni, pid, pid_granule,
					&dom->if_dom);
		if (ret) {
			CXIX_LOG_DBG("cxil_alloc_domain returned: %d\n", ret);
			goto free_dom;
		}

		dlist_insert_tail(&dom->entry, &dev_if->if_doms);
		dom->dev_if = dev_if;
		dom->vni = vni;
		dom->pid = pid;
		dom->pid_granule = pid_granule;
		memset(&dom->lep_map, 0, sizeof(dom->lep_map));
		ofi_atomic_initialize32(&dom->ref, 0);
		fastlock_init(&dom->lock);

		CXIX_LOG_DBG("Allocated IF Domain, NIC: %u VNI: %u PID: %u PG: %u\n",
			     dev_if->if_nic, vni, pid, pid_granule);
	} else {
		CXIX_LOG_DBG("Using IF Domain, NIC: %u VNI: %u PID: %u PG: %u ref: %u\n",
			     dev_if->if_nic, vni, pid, pid_granule,
			     ofi_atomic_get32(&dom->ref));
	}

	ofi_atomic_inc32(&dom->ref);
	fastlock_release(&dev_if->lock);

	*if_dom = dom;

	return FI_SUCCESS;

free_dom:
	free(dom);
unlock:
	fastlock_release(&dev_if->lock);

	return ret;
}

/*
 * cxix_put_if_domain() - Drop a reference to the IF Domain.
 */
void cxix_put_if_domain(struct cxix_if_domain *if_dom)
{
	int ret;

	fastlock_acquire(&if_dom->dev_if->lock);

	if (!ofi_atomic_dec32(&if_dom->ref)) {
		CXIX_LOG_DBG("Released IF Domain, NIC: %u VNI: %u PID: %u PG: %u\n",
			     if_dom->dev_if->if_nic, if_dom->vni, if_dom->pid,
			     if_dom->pid_granule);

		fastlock_destroy(&if_dom->lock);

		dlist_remove(&if_dom->entry);

		ret = cxil_destroy_domain(if_dom->if_dom);
		if (ret)
			CXIX_LOG_ERROR("Failed to destroy domain: %d\n", ret);

		free(if_dom);
	}

	fastlock_release(&if_dom->dev_if->lock);
}

/*
 * cxix_if_domain_lep_alloc() - Allocate a logical endpoint from the IF Domain
 *
 * A logical endpoint is an address where a PtlTE may be bound.  PtlTEs are
 * used to implement RX interactions for the various OFI protocols.
 */
int cxix_if_domain_lep_alloc(struct cxix_if_domain *if_dom, uint64_t lep_idx)
{
	int ret;
	void *lep;

	fastlock_acquire(&if_dom->lock);

	lep = ofi_idm_lookup(&if_dom->lep_map, lep_idx);
	if (!lep) {
		/* TODO The IDM is used as a bitmap. */
		ret = ofi_idm_set(&if_dom->lep_map, lep_idx, (void *)1);
		if (ret != lep_idx) {
			fastlock_release(&if_dom->lock);
			return -errno;
		}
	}

	fastlock_release(&if_dom->lock);

	return lep ? -FI_EADDRINUSE : FI_SUCCESS;
}

/*
 * cxix_if_domain_lep_free() - Free a logical endpoint from the IF Domain.
 */
int cxix_if_domain_lep_free(struct cxix_if_domain *if_dom, uint64_t lep_idx)
{
	void *lep;

	fastlock_acquire(&if_dom->lock);

	lep = ofi_idm_lookup(&if_dom->lep_map, lep_idx);
	if (!lep) {
		CXIX_LOG_ERROR("Attempt to free unallocated lep_idx: %lu\n",
			       lep_idx);
		fastlock_release(&if_dom->lock);
		return -FI_EINVAL;
	}

	ofi_idm_clear(&if_dom->lep_map, lep_idx);

	fastlock_release(&if_dom->lock);

	return FI_SUCCESS;
}

/*
 * cxix_query_if_list() - Populate static IF data during initialization.
 */
static void cxix_query_if_list(struct slist *if_list)
{
	struct cxix_if *if_entry;
	struct cxil_devinfo *info;
	int ret;

	slist_init(&cxix_if_list);

	/* TODO Query all interfaces */
	ret = cxil_query_devinfo(0, &info);
	if (ret) {
		CXIX_LOG_DBG("No IFs found\n");
		return;
	}

	if_entry = calloc(1, sizeof(struct cxix_if));
	if_entry->if_nic = info->nic_addr;
	if_entry->if_idx = info->dev_id;
	if_entry->if_fabric = 0; /* TODO Find real network ID */
	ofi_atomic_initialize32(&if_entry->ref, 0);
	dlist_init(&if_entry->if_doms);
	fastlock_init(&if_entry->lock);
	slist_insert_tail(&if_entry->entry, if_list);
}

/*
 * cxix_free_if_list() - Tears down static IF data.
 */
static void cxix_free_if_list(struct slist *if_list)
{
	struct slist_entry *entry;
	struct cxix_if *if_entry;

	while (!slist_empty(if_list)) {
		entry = slist_remove_head(if_list);
		if_entry = container_of(entry, struct cxix_if, entry);
		fastlock_destroy(&if_entry->lock);
		free(if_entry);
	}
}

/*
 * cxix_if_init() - The provider IF constructor.  Initializes static IF data.
 */
void cxix_if_init(void)
{
	cxix_num_pids = CXIX_NUM_PIDS_DEF;

	cxix_query_if_list(&cxix_if_list);
}

/*
 * cxix_if_init() - The provider IF destructor.  Tears down IF data.
 */
void cxix_if_fini(void)
{
	cxix_free_if_list(&cxix_if_list);
}

