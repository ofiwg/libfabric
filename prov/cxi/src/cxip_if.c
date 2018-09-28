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

#include "cxip.h"

#define CXIP_LOG_DBG(...) _CXIP_LOG_DBG(FI_LOG_DOMAIN, __VA_ARGS__)
#define CXIP_LOG_ERROR(...) _CXIP_LOG_ERROR(FI_LOG_DOMAIN, __VA_ARGS__)

struct slist cxip_if_list;

struct cxip_if *cxip_if_lookup(uint32_t nic_addr)
{
	struct slist_entry *entry, *prev __attribute__ ((unused));
	struct cxip_if *if_entry;

	slist_foreach(&cxip_if_list, entry, prev) {
		if_entry = container_of(entry, struct cxip_if, entry);
		if (if_entry->if_nic == nic_addr)
			return if_entry;
	}

	return NULL;
}

static struct cxip_if_domain *cxip_if_domain_lookup(struct cxip_if *dev_if,
						    uint32_t vni, uint32_t pid)
{
	struct dlist_entry *entry;
	struct cxip_if_domain *dom;

	dlist_foreach(&dev_if->if_doms, entry) {
		dom = container_of(entry, struct cxip_if_domain, entry);
		if (dom->vni == vni && dom->pid == pid)
			return dom;
	}

	return NULL;
}

/*
 * cxip_get_if() - Get a reference to the interface associated with a provided
 * NIC address.  A process can open each interface once to support many FI
 * Domains.  An IF is used to allocate the various device resources including
 * CMDQs, EQs, PtlTEs.
 */
int cxip_get_if(uint32_t nic_addr, struct cxip_if **dev_if)
{
	struct cxip_if *if_entry;
	int ret;
	struct cxi_eq_alloc_opts evtq_opts;

	/* The IF list device info is static, no need to lock */
	if_entry = cxip_if_lookup(nic_addr);
	if (!if_entry) {
		CXIP_LOG_DBG("interface not found\n");
		return -FI_ENODEV;
	}

	/* Lock the IF to serialize opening the device */
	fastlock_acquire(&if_entry->lock);

	if (!if_entry->if_lni) {
		ret = cxil_open_device(if_entry->if_idx, &if_entry->if_dev);
		if (ret) {
			CXIP_LOG_DBG("cxil_open_device returned: %d\n", ret);
			ret = -FI_ENODEV;
			goto unlock;
		}

		ret = cxil_alloc_lni(if_entry->if_dev, 0, &if_entry->if_lni);
		if (ret) {
			CXIP_LOG_DBG("cxil_alloc_lni returned: %d\n", ret);
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
			CXIP_LOG_DBG("Unable to allocate MR CMDQ, ret: %d\n",
				     ret);
			ret = -FI_ENODEV;
			goto free_lni;
		}

		evtq_opts.count = 1024;

		ret = cxil_alloc_evtq(if_entry->if_lni, &evtq_opts, NULL,
				      &if_entry->mr_evtq);
		if (ret != FI_SUCCESS) {
			CXIP_LOG_DBG("Unable to allocate MR EVTQ, ret: %d\n",
				     ret);
			ret = -FI_ENODEV;
			goto free_mr_cmdq;
		}

		CXIP_LOG_DBG("Allocated IF, NIC: %u ID: %u\n", if_entry->if_nic,
			     if_entry->if_idx);
	} else {
		CXIP_LOG_DBG("Using IF, NIC: %u ID: %u ref: %u\n",
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
		CXIP_LOG_ERROR("Failed to destroy CMDQ: %d\n", ret);
free_lni:
	ret = cxil_destroy_lni(if_entry->if_lni);
	if (ret)
		CXIP_LOG_ERROR("Failed to destroy LNI: %d\n", ret);
	if_entry->if_lni = NULL;
close_dev:
	cxil_close_device(if_entry->if_dev);
	if_entry->if_dev = NULL;
unlock:
	fastlock_release(&if_entry->lock);

	return ret;
}

/*
 * cxip_put_if() - Drop a reference to the IF.
 */
void cxip_put_if(struct cxip_if *dev_if)
{
	int ret;

	fastlock_acquire(&dev_if->lock);

	if (!ofi_atomic_dec32(&dev_if->ref)) {
		cxil_destroy_evtq(dev_if->mr_evtq);

		ret = cxil_destroy_cmdq(dev_if->mr_cmdq);
		if (ret)
			CXIP_LOG_ERROR("Failed to destroy CMDQ: %d\n", ret);

		ret = cxil_destroy_lni(dev_if->if_lni);
		if (ret)
			CXIP_LOG_ERROR("Failed to destroy LNI: %d\n", ret);
		dev_if->if_lni = NULL;

		cxil_close_device(dev_if->if_dev);
		dev_if->if_dev = NULL;

		dev_if->n_cps = 0;

		CXIP_LOG_DBG("Released IF, NIC: %u ID: %u\n", dev_if->if_nic,
			     dev_if->if_idx);
	}

	dev_if->if_dev = NULL;

	fastlock_release(&dev_if->lock);
}

/*
 * cxip_get_if_domain() - Get a reference to an IF Domain.  An IF Domain
 * represents an address space of logical network endpoints (LEPs).  LEPs are
 * network addressable endpoints where a PtlTE resource can be mapped.  A PtlTE
 * is basically an RX queue.  The IF Domain address space is identified by the
 * three-tuple:
 *
 *    ( dev_if->if_nic, vni, pid )
 */
int cxip_get_if_domain(struct cxip_if *dev_if, uint32_t vni, uint32_t pid,
		       struct cxip_if_domain **if_dom)
{
	struct cxip_if_domain *dom;
	int ret;

	fastlock_acquire(&dev_if->lock);

	dom = cxip_if_domain_lookup(dev_if, vni, pid);
	if (!dom) {
		dom = malloc(sizeof(*dom));
		if (!dom) {
			CXIP_LOG_DBG("Unable to allocate IF domain\n");
			ret = -FI_ENOMEM;
			goto unlock;
		}

		ret = cxil_alloc_domain(dev_if->if_lni, vni, pid, &dom->if_dom);
		if (ret) {
			CXIP_LOG_DBG("cxil_alloc_domain returned: %d\n", ret);
			goto free_dom;
		}

		dlist_insert_tail(&dom->entry, &dev_if->if_doms);
		dom->dev_if = dev_if;
		dom->vni = vni;
		dom->pid = pid;
		memset(&dom->lep_map, 0, sizeof(dom->lep_map));
		ofi_atomic_initialize32(&dom->ref, 0);
		fastlock_init(&dom->lock);

		CXIP_LOG_DBG(
			"Allocated IF Domain, NIC: %u VNI: %u PID: %u\n",
			dev_if->if_nic, vni, pid);
	} else {
		CXIP_LOG_DBG(
			"Using IF Domain, NIC: %u VNI: %u PID: %u ref: %u\n",
			dev_if->if_nic, vni, pid, ofi_atomic_get32(&dom->ref));
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
 * cxip_put_if_domain() - Drop a reference to the IF Domain.
 */
void cxip_put_if_domain(struct cxip_if_domain *if_dom)
{
	int ret;

	fastlock_acquire(&if_dom->dev_if->lock);

	if (!ofi_atomic_dec32(&if_dom->ref)) {
		CXIP_LOG_DBG(
			"Released IF Domain, NIC: %u VNI: %u PID: %u\n",
			if_dom->dev_if->if_nic, if_dom->vni, if_dom->pid);

		fastlock_destroy(&if_dom->lock);

		dlist_remove(&if_dom->entry);

		ret = cxil_destroy_domain(if_dom->if_dom);
		if (ret)
			CXIP_LOG_ERROR("Failed to destroy domain: %d\n", ret);

		free(if_dom);
	}

	fastlock_release(&if_dom->dev_if->lock);
}

/*
 * cxip_if_domain_lep_alloc() - Allocate a logical endpoint from the IF Domain
 *
 * A logical endpoint is an address where a PtlTE may be bound.  PtlTEs are
 * used to implement RX interactions for the various OFI protocols.
 *
 * The full LEP address is specified by the (dev, vni, pid, pid_idx) vector. The
 * (dev, vni, pid) is implied by the libfabric domain. The pid_idx completes the
 * specification.
 */
int cxip_if_domain_lep_alloc(struct cxip_if_domain *if_dom, uint64_t pid_idx)
{
	int ret;
	void *lep;

	fastlock_acquire(&if_dom->lock);

	lep = ofi_idm_lookup(&if_dom->lep_map, pid_idx);
	if (!lep) {
		/* TODO The IDM is used as a bitmap. */
		ret = ofi_idm_set(&if_dom->lep_map, pid_idx, (void *)1);
		if (ret != pid_idx) {
			fastlock_release(&if_dom->lock);
			return -errno;
		}
	}

	fastlock_release(&if_dom->lock);

	return lep ? -FI_EADDRINUSE : FI_SUCCESS;
}

/*
 * cxip_if_domain_lep_free() - Free a logical endpoint from the IF Domain.
 */
int cxip_if_domain_lep_free(struct cxip_if_domain *if_dom, uint64_t pid_idx)
{
	void *lep;

	fastlock_acquire(&if_dom->lock);

	lep = ofi_idm_lookup(&if_dom->lep_map, pid_idx);
	if (!lep) {
		CXIP_LOG_ERROR("Attempt to free unallocated pid_idx: %lu\n",
			       pid_idx);
		fastlock_release(&if_dom->lock);
		return -FI_EINVAL;
	}

	ofi_idm_clear(&if_dom->lep_map, pid_idx);

	fastlock_release(&if_dom->lock);

	return FI_SUCCESS;
}

int cxip_pte_alloc(struct cxip_if_domain *if_dom, struct cxi_evtq *evtq,
		   uint64_t pid_idx, struct cxi_pt_alloc_opts *opts,
		   struct cxip_pte **pte)
{
	struct cxip_pte *new_pte;
	int ret;

	new_pte = malloc(sizeof(*new_pte));
	if (!new_pte) {
		CXIP_LOG_ERROR("Unable to allocate PTE structure\n");
		return -FI_ENOMEM;
	}

	/* Allocate a PTE */
	ret = cxil_alloc_pte(if_dom->dev_if->if_lni, evtq, opts, &new_pte->pte);
	if (ret) {
		CXIP_LOG_DBG("Failed to allocate PTE: %d\n", ret);
		ret = -FI_ENOSPC;
		goto free_mem;
	}

	/* Reserve LEP where PTE will be mapped */
	ret = cxip_if_domain_lep_alloc(if_dom, pid_idx);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Failed to reserve LEP (%lu): %d\n", pid_idx, ret);
		goto free_pte;
	}

	/* Map the PTE to the LEP */
	ret = cxil_map_pte(new_pte->pte, if_dom->if_dom, pid_idx, 0,
			   &new_pte->pte_map);
	if (ret) {
		CXIP_LOG_DBG("Failed to map PTE: %d\n", ret);
		ret = -FI_EADDRINUSE;
		goto free_lep;
	}

	fastlock_acquire(&if_dom->dev_if->lock);
	dlist_insert_tail(&new_pte->entry, &if_dom->dev_if->ptes);
	fastlock_release(&if_dom->dev_if->lock);

	new_pte->if_dom = if_dom;
	new_pte->pid_idx = pid_idx;
	new_pte->state = C_PTLTE_DISABLED;

	*pte = new_pte;

	return FI_SUCCESS;

free_lep:
	ret = cxip_if_domain_lep_free(if_dom, pid_idx);
	if (ret)
		CXIP_LOG_ERROR("cxip_if_domain_lep_free returned: %d\n", ret);
free_pte:
	ret = cxil_destroy_pte(new_pte->pte);
	if (ret)
		CXIP_LOG_ERROR("cxil_destroy_pte returned: %d\n", ret);
free_mem:
	free(new_pte);

	return ret;
}

void cxip_pte_free(struct cxip_pte *pte)
{
	int ret;

	fastlock_acquire(&pte->if_dom->dev_if->lock);
	dlist_remove(&pte->entry);
	fastlock_release(&pte->if_dom->dev_if->lock);

	ret = cxil_unmap_pte(pte->pte_map);
	if (ret)
		CXIP_LOG_ERROR("Failed to unmap PTE: %d\n", ret);

	ret = cxip_if_domain_lep_free(pte->if_dom, pte->pid_idx);
	if (ret)
		CXIP_LOG_ERROR("Failed to free LEP: %d\n", ret);

	ret = cxil_destroy_pte(pte->pte);
	if (ret)
		CXIP_LOG_ERROR("Failed to free PTE: %d\n", ret);

	free(pte);
}

int cxip_pte_state_change(struct cxip_if *dev_if, uint32_t pte_num,
			  enum c_ptlte_state new_state)
{
	struct cxip_pte *pte;

	fastlock_acquire(&dev_if->lock);

	dlist_foreach_container(&dev_if->ptes, struct cxip_pte, pte, entry) {
		if (pte->pte->ptn == pte_num) {
			pte->state = new_state;
			fastlock_release(&dev_if->lock);
			return FI_SUCCESS;
		}
	}

	fastlock_release(&dev_if->lock);

	return -FI_EINVAL;
}

/*
 * cxip_query_if_list() - Populate static IF data during initialization.
 */
static void cxip_query_if_list(struct slist *if_list)
{
	struct cxip_if *if_entry;
	struct cxil_device_list *dev_list;
	int ret;

	slist_init(&cxip_if_list);

	ret = cxil_get_device_list(&dev_list);
	if (ret) {
		CXIP_LOG_DBG("cxil_get_device_list failed\n");
		return;
	}

	if (dev_list->count == 0) {
		CXIP_LOG_DBG("No IFs found\n");
		cxil_free_device_list(dev_list);
		return;
	}

	/* Pick first device */
	if_entry = calloc(1, sizeof(struct cxip_if));
	if_entry->if_nic = dev_list->info[0].nic_addr;
	if_entry->if_idx = dev_list->info[0].dev_id;
	if_entry->if_pid_granule = dev_list->info[0].pid_granule;
	if_entry->if_fabric = 0; /* TODO Find real network ID */
	ofi_atomic_initialize32(&if_entry->ref, 0);
	dlist_init(&if_entry->if_doms);
	dlist_init(&if_entry->ptes);
	fastlock_init(&if_entry->lock);
	slist_insert_tail(&if_entry->entry, if_list);

	cxil_free_device_list(dev_list);
}

/*
 * cxip_free_if_list() - Tears down static IF data.
 */
static void cxip_free_if_list(struct slist *if_list)
{
	struct slist_entry *entry;
	struct cxip_if *if_entry;

	while (!slist_empty(if_list)) {
		entry = slist_remove_head(if_list);
		if_entry = container_of(entry, struct cxip_if, entry);
		fastlock_destroy(&if_entry->lock);
		free(if_entry);
	}
}

/*
 * cxip_if_init() - The provider IF constructor.  Initializes static IF data.
 */
void cxip_if_init(void)
{
	cxip_query_if_list(&cxip_if_list);
}

/*
 * cxip_if_init() - The provider IF destructor.  Tears down IF data.
 */
void cxip_if_fini(void)
{
	cxip_free_if_list(&cxip_if_list);
}
