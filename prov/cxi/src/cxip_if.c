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
#include <dirent.h>

#include "ofi_prov.h"
#include "ofi_osd.h"

#include "cxip.h"

#define CXIP_LOG_DBG(...) _CXIP_LOG_DBG(FI_LOG_DOMAIN, __VA_ARGS__)
#define CXIP_LOG_INFO(...) _CXIP_LOG_INFO(FI_LOG_DOMAIN, __VA_ARGS__)
#define CXIP_LOG_ERROR(...) _CXIP_LOG_ERROR(FI_LOG_DOMAIN, __VA_ARGS__)

struct slist cxip_if_list;

/*
 * cxip_if_lookup() - Return a provider NIC interface descriptor associated
 * with a specified NIC address, if available.
 */
struct cxip_if *cxip_if_lookup(uint32_t nic_addr)
{
	struct slist_entry *entry, *prev __attribute__ ((unused));
	struct cxip_if *if_entry;

	slist_foreach(&cxip_if_list, entry, prev) {
		if_entry = container_of(entry, struct cxip_if, if_entry);
		if (if_entry->info.nic_addr == nic_addr)
			return if_entry;
	}

	return NULL;
}

/*
 * cxip_lni_res_count() - Return usage information for LNI resource.
 */
int cxip_lni_res_cnt(struct cxip_lni *lni, char *res_str, uint32_t *count)
{
	struct dirent *de;
	char path[100];
	uint32_t c = 0;
	DIR *dr;

	sprintf(path, "/sys/kernel/debug/cxi/cxi%u/lni/%u/%s",
		lni->iface->info.dev_id, lni->lni->id, res_str);

	dr = opendir(path);
	if (!dr)
		return -FI_ENOSYS;

	while ((de = readdir(dr))) {
		if (strncmp(de->d_name, ".", 1))
			c++;
	}

	closedir(dr);

	*count = c;

	return FI_SUCCESS;
}

/*
 * cxip_lni_res_dump() - Dump resource usage information for an LNI.
 */
void cxip_lni_res_dump(struct cxip_lni *lni)
{
	int ret;
	uint32_t pt_count = 0;
	uint32_t txq_count = 0;
	uint32_t tgq_count = 0;
	uint32_t eq_count = 0;
	uint32_t ct_count = 0;
	uint32_t ac_count = 0;

	ret = cxip_lni_res_cnt(lni, "txq", &txq_count);

	/* Expect failure if debugfs isn't mounted. */
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Resource usage info unavailable: cxi%u RGID: %u.\n",
			     lni->iface->info.dev_id, lni->lni->id);
		return;
	}

	cxip_lni_res_cnt(lni, "tgq", &tgq_count);
	cxip_lni_res_cnt(lni, "pt", &pt_count);
	cxip_lni_res_cnt(lni, "eq", &eq_count);
	cxip_lni_res_cnt(lni, "ct", &ct_count);
	cxip_lni_res_cnt(lni, "ac", &ac_count);

	CXIP_LOG_INFO("Resource usage: cxi%u RGID: %u TXQ: %u TGQ: %u PTE: %u EQ: %u CT: %u AC: %u\n",
		      lni->iface->info.dev_id, lni->lni->id, txq_count,
		      tgq_count, pt_count, eq_count, ct_count, ac_count);
}

/*
 * cxip_get_if() - Get a reference to the device interface associated with a
 * provided NIC address. A process can open each interface once to support many
 * FI Domains. An IF is used to allocate the various device resources including
 * CMDQs, EVTQs, and PtlTEs.
 */
int cxip_get_if(uint32_t nic_addr, struct cxip_if **iface)
{
	int ret;
	struct cxip_if *if_entry;

	/* The IF list device info is static, no need to lock */
	if_entry = cxip_if_lookup(nic_addr);
	if (!if_entry) {
		CXIP_LOG_DBG("interface not found\n");
		return -FI_ENODEV;
	}

	/* Lock the IF to serialize opening the device */
	fastlock_acquire(&if_entry->lock);

	if (!if_entry->dev) {
		ret = cxil_open_device(if_entry->info.dev_id, &if_entry->dev);
		if (ret) {
			CXIP_LOG_DBG("cxil_open_device returned: %d\n", ret);
			ret = -FI_ENODEV;
			goto unlock;
		}

		CXIP_LOG_DBG("Opened cxi%u\n", if_entry->info.dev_id);
	}

	ofi_atomic_inc32(&if_entry->ref);
	*iface = if_entry;

	fastlock_release(&if_entry->lock);

	return FI_SUCCESS;

unlock:
	fastlock_release(&if_entry->lock);

	return ret;
}

/*
 * cxip_put_if() - Drop a reference to the device interface.
 */
void cxip_put_if(struct cxip_if *iface)
{
	fastlock_acquire(&iface->lock);

	if (!ofi_atomic_dec32(&iface->ref)) {
		cxil_close_device(iface->dev);
		iface->dev = NULL;

		CXIP_LOG_DBG("Closed cxi%u\n", iface->info.dev_id);
	}

	fastlock_release(&iface->lock);
}

/*
 * cxip_alloc_lni() - Allocate an LNI
 */
int cxip_alloc_lni(struct cxip_if *iface, struct cxip_lni **if_lni)
{
	struct cxip_lni *lni;
	int ret;

	lni = malloc(sizeof(*lni));
	if (!lni) {
		CXIP_LOG_DBG("Unable to allocate LNI\n");
		return -FI_ENOMEM;
	}

	ret = cxil_alloc_lni(iface->dev, &lni->lni);
	if (ret) {
		CXIP_LOG_DBG("cxil_alloc_lni returned: %d\n", ret);
		ret = -FI_ENOSPC;
		goto free_lni;
	}

	lni->iface = iface;

	CXIP_LOG_DBG("Allocated LNI, cxi%u RGID: %u\n",
		     lni->iface->info.dev_id, lni->lni->id);

	*if_lni = lni;

	return FI_SUCCESS;

free_lni:
	free(lni);

	return ret;
}

/*
 * cxip_free_lni() - Free an LNI
 */
void cxip_free_lni(struct cxip_lni *lni)
{
	int ret;

	cxip_lni_res_dump(lni);

	CXIP_LOG_DBG("Freeing LNI, cxi%u RGID: %u\n",
		     lni->iface->info.dev_id, lni->lni->id);

	ret = cxil_destroy_lni(lni->lni);
	if (ret)
		CXIP_LOG_ERROR("Failed to destroy LNI: %d\n", ret);

	free(lni);
}

/*
 * cxip_alloc_if_domain() - Allocate an IF Domain
 */
int cxip_alloc_if_domain(struct cxip_lni *lni, uint32_t vni, uint32_t pid,
			 struct cxip_if_domain **if_dom)
{
	struct cxip_if_domain *dom;
	int ret;

	dom = malloc(sizeof(*dom));
	if (!dom) {
		CXIP_LOG_DBG("Unable to allocate IF domain\n");
		return -FI_ENOMEM;
	}

	ret = cxil_alloc_domain(lni->lni, vni, pid, &dom->dom);
	if (ret) {
		CXIP_LOG_DBG("cxil_alloc_domain returned: %d\n", ret);
		ret = -FI_ENOSPC;
		goto free_dom;
	}

	dom->lni = lni;

	CXIP_LOG_DBG("Allocated IF Domain, cxi%u VNI: %u PID: %u\n",
		     lni->iface->info.dev_id, vni, pid);

	*if_dom = dom;

	return FI_SUCCESS;

free_dom:
	free(dom);

	return ret;
}

/*
 * cxip_free_if_domain() - Free an IF Domain.
 */
void cxip_free_if_domain(struct cxip_if_domain *if_dom)
{
	int ret;

	CXIP_LOG_DBG("Freeing IF Domain, cxi%u VNI: %u PID: %u\n",
		     if_dom->lni->iface->info.dev_id, if_dom->dom->vni,
		     if_dom->dom->pid);

	ret = cxil_destroy_domain(if_dom->dom);
	if (ret)
		CXIP_LOG_ERROR("Failed to destroy domain: %d\n", ret);

	free(if_dom);
}

/*
 * cxip_pte_append() - Append a buffer to a PtlTE.
 */
int cxip_pte_append(struct cxip_pte *pte, uint64_t iova, size_t len,
		    unsigned int lac, enum c_ptl_list list,
		    uint32_t buffer_id, uint64_t match_bits,
		    uint64_t ignore_bits, uint32_t match_id,
		    uint64_t min_free, uint32_t flags,
		    struct cxip_cntr *cntr, struct cxip_cmdq *cmdq,
		    bool ring)
{
	union c_cmdu cmd = {};
	int rc;

	cmd.command.opcode                = C_CMD_TGT_APPEND;
	cmd.target.ptl_list               = list;
	cmd.target.ptlte_index            = pte->pte->ptn;
	cmd.target.buffer_id              = buffer_id;
	cmd.target.lac                    = lac;
	cmd.target.start                  = iova;
	cmd.target.length                 = len;
	cmd.target.ct                     = cntr ? cntr->ct->ctn : 0;
	cmd.target.match_bits             = match_bits;
	cmd.target.ignore_bits            = ignore_bits;
	cmd.target.match_id               = match_id;
	cmd.target.min_free               = min_free;

	cxi_target_cmd_setopts(&cmd.target, flags);

	fastlock_acquire(&cmdq->lock);

	rc = cxi_cq_emit_target(cmdq->dev_cmdq, &cmd);
	if (rc) {
		CXIP_LOG_DBG("Failed to write Append command: %d\n", rc);

		fastlock_release(&cmdq->lock);

		/* Return error according to Domain Resource Management */
		return -FI_EAGAIN;
	}

	if (ring)
		cxi_cq_ring(cmdq->dev_cmdq);

	fastlock_release(&cmdq->lock);

	return FI_SUCCESS;
}

/*
 * cxip_pte_unlink() - Unlink a buffer from a PtlTE.
 */
int cxip_pte_unlink(struct cxip_pte *pte, enum c_ptl_list list,
		    int buffer_id, struct cxip_cmdq *cmdq)
{
	union c_cmdu cmd = {};
	int rc;

	cmd.command.opcode = C_CMD_TGT_UNLINK;
	cmd.target.ptl_list = list;
	cmd.target.ptlte_index  = pte->pte->ptn;
	cmd.target.buffer_id = buffer_id;

	fastlock_acquire(&cmdq->lock);

	rc = cxi_cq_emit_target(cmdq->dev_cmdq, &cmd);
	if (rc) {
		CXIP_LOG_DBG("Failed to write Append command: %d\n", rc);

		fastlock_release(&cmdq->lock);

		/* Return error according to Domain Resource Management */
		return -FI_EAGAIN;
	}

	cxi_cq_ring(cmdq->dev_cmdq);

	fastlock_release(&cmdq->lock);

	return FI_SUCCESS;
}

/*
 * cxip_pte_alloc() - Allocate and map a PTE for use.
 */
int cxip_pte_alloc(struct cxip_if_domain *if_dom, struct cxi_evtq *evtq,
		   uint64_t pid_idx, struct cxi_pt_alloc_opts *opts,
		   void (*state_change_cb)(struct cxip_pte *pte,
					   enum c_ptlte_state state),
		   void *ctx, struct cxip_pte **pte)
{
	struct cxip_pte *new_pte;
	int ret, tmp;

	new_pte = malloc(sizeof(*new_pte));
	if (!new_pte) {
		CXIP_LOG_ERROR("Unable to allocate PTE structure\n");
		return -FI_ENOMEM;
	}

	/* Allocate a PTE */
	ret = cxil_alloc_pte(if_dom->lni->lni, evtq, opts,
			     &new_pte->pte);
	if (ret) {
		CXIP_LOG_DBG("Failed to allocate PTE: %d\n", ret);
		ret = -FI_ENOSPC;
		goto free_mem;
	}

	/* Map the PTE to the LEP */
	ret = cxil_map_pte(new_pte->pte, if_dom->dom, pid_idx, 0,
			   &new_pte->pte_map);
	if (ret) {
		CXIP_LOG_DBG("Failed to map PTE: %d\n", ret);
		ret = -FI_EADDRINUSE;
		goto free_pte;
	}

	fastlock_acquire(&if_dom->lni->iface->lock);
	dlist_insert_tail(&new_pte->pte_entry, &if_dom->lni->iface->ptes);
	fastlock_release(&if_dom->lni->iface->lock);

	new_pte->if_dom = if_dom;
	new_pte->pid_idx = pid_idx;
	new_pte->state_change_cb = state_change_cb;
	new_pte->ctx = ctx;

	*pte = new_pte;

	return FI_SUCCESS;

free_pte:
	tmp = cxil_destroy_pte(new_pte->pte);
	if (tmp)
		CXIP_LOG_ERROR("cxil_destroy_pte returned: %d\n", tmp);
free_mem:
	free(new_pte);

	return ret;
}

/*
 * cxip_pte_free() - Free a PTE.
 */
void cxip_pte_free(struct cxip_pte *pte)
{
	int ret;

	fastlock_acquire(&pte->if_dom->lni->iface->lock);
	dlist_remove(&pte->pte_entry);
	fastlock_release(&pte->if_dom->lni->iface->lock);

	ret = cxil_unmap_pte(pte->pte_map);
	if (ret)
		CXIP_LOG_ERROR("Failed to unmap PTE: %d\n", ret);

	ret = cxil_destroy_pte(pte->pte);
	if (ret)
		CXIP_LOG_ERROR("Failed to free PTE: %d\n", ret);

	free(pte);
}

/*
 * cxip_pte_state_change() - Atomically update PTE state. Used during
 * STATE_CHANGE event processing.
 */
int cxip_pte_state_change(struct cxip_if *dev_if, uint32_t pte_num,
			  enum c_ptlte_state new_state)
{
	struct cxip_pte *pte;

	fastlock_acquire(&dev_if->lock);

	dlist_foreach_container(&dev_if->ptes,
				struct cxip_pte, pte, pte_entry) {
		if (pte->pte->ptn == pte_num) {
			if (pte->state_change_cb)
				pte->state_change_cb(pte, new_state);

			fastlock_release(&dev_if->lock);
			return FI_SUCCESS;
		}
	}

	fastlock_release(&dev_if->lock);

	return -FI_EINVAL;
}

/*
 * cxip_cmdq_alloc() - Allocate a command queue.
 */
int cxip_cmdq_alloc(struct cxip_lni *lni, struct cxi_evtq *evtq,
		    struct cxi_cq_alloc_opts *cq_opts,
		    struct cxip_cmdq **cmdq)
{
	int ret;
	struct cxi_cmdq *dev_cmdq;
	struct cxip_cmdq *new_cmdq;

	new_cmdq = calloc(1, sizeof(*new_cmdq));
	if (!new_cmdq) {
		CXIP_LOG_ERROR("Unable to allocate CMDQ structure\n");
		return -FI_ENOMEM;
	}

	ret = cxil_alloc_cmdq(lni->lni, evtq, cq_opts, &dev_cmdq);
	if (ret) {
		CXIP_LOG_DBG("cxil_alloc_cmdq() failed, ret: %d\n", ret);
		ret = -FI_ENOSPC;
		goto free_cmdq;
	}

	new_cmdq->dev_cmdq = dev_cmdq;
	fastlock_init(&new_cmdq->lock);
	*cmdq = new_cmdq;

	return FI_SUCCESS;

free_cmdq:
	free(new_cmdq);

	return ret;
}

/*
 * cxip_cmdq_free() - Free a command queue.
 */
void cxip_cmdq_free(struct cxip_cmdq *cmdq)
{
	int ret;

	ret = cxil_destroy_cmdq(cmdq->dev_cmdq);
	if (ret)
		CXIP_LOG_ERROR("cxil_destroy_cmdq failed, ret: %d\n", ret);

	fastlock_destroy(&cmdq->lock);
	free(cmdq);
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

	if (dev_list->info[0].min_free_shift) {
		CXIP_LOG_DBG("Non-zero min_free_shift not supported\n");
		cxil_free_device_list(dev_list);
		return;
	}

	/* Pick first device */
	if_entry = calloc(1, sizeof(struct cxip_if));
	if_entry->info = dev_list->info[0];
	ofi_atomic_initialize32(&if_entry->ref, 0);
	dlist_init(&if_entry->ptes);
	fastlock_init(&if_entry->lock);
	slist_insert_tail(&if_entry->if_entry, if_list);

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
		if_entry = container_of(entry, struct cxip_if, if_entry);
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
