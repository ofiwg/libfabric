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

#define CXIP_DBG(...) _CXIP_DBG(FI_LOG_DOMAIN, __VA_ARGS__)
#define CXIP_INFO(...) _CXIP_INFO(FI_LOG_DOMAIN, __VA_ARGS__)
#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_DOMAIN, __VA_ARGS__)

struct slist cxip_if_list;
static struct cxil_device_list *cxi_dev_list;

/*
 * cxip_if_lookup_addr() - Return a provider NIC interface descriptor
 * associated with a specified NIC address, if available.
 */
struct cxip_if *cxip_if_lookup_addr(uint32_t nic_addr)
{
	struct slist_entry *entry, *prev __attribute__ ((unused));
	struct cxip_if *if_entry;

	slist_foreach(&cxip_if_list, entry, prev) {
		if_entry = container_of(entry, struct cxip_if, if_entry);
		if (if_entry->info->nic_addr == nic_addr)
			return if_entry;
	}

	return NULL;
}

/*
 * cxip_if_lookup() - Return a provider NIC interface descriptor associated
 * with a specified NIC device name, if available.
 */
struct cxip_if *cxip_if_lookup_name(const char *name)
{
	struct slist_entry *entry, *prev __attribute__ ((unused));
	struct cxip_if *if_entry;

	slist_foreach(&cxip_if_list, entry, prev) {
		if_entry = container_of(entry, struct cxip_if, if_entry);
		if (!strcmp(if_entry->info->device_name, name))
			return if_entry;
	}

	return NULL;
}

/*
 * cxip_lni_res_count() - Return usage information for LNI resource.
 */
int cxip_lni_res_cnt(struct cxip_lni *lni, char *res_str)
{
	struct dirent *de;
	char path[100];
	uint32_t c = 0;
	DIR *dr;

	sprintf(path, "/sys/kernel/debug/cxi/cxi%u/lni/%u/%s",
		lni->iface->info->dev_id, lni->lni->id, res_str);

	dr = opendir(path);
	if (!dr)
		return 0;

	while ((de = readdir(dr))) {
		if (strncmp(de->d_name, ".", 1))
			c++;
	}

	closedir(dr);

	return c;
}

/*
 * cxip_lni_res_dump() - Dump resource usage information for an LNI.
 */
void cxip_lni_res_dump(struct cxip_lni *lni)
{
	DIR *dr;
	uint32_t pt_count = 0;
	uint32_t cq_count = 0;
	uint32_t eq_count = 0;
	uint32_t ct_count = 0;
	uint32_t ac_count = 0;

	/* Check if debugfs is available. */
	dr = opendir("/sys/kernel/debug/cxi");
	if (!dr) {
		CXIP_INFO("Resource usage info unavailable: %s RGID: %u.\n",
			  lni->iface->info->device_name, lni->lni->id);
		return;
	}

	closedir(dr);

	cq_count = cxip_lni_res_cnt(lni, "cq");
	pt_count = cxip_lni_res_cnt(lni, "pt");
	eq_count = cxip_lni_res_cnt(lni, "eq");
	ct_count = cxip_lni_res_cnt(lni, "ct");
	ac_count = cxip_lni_res_cnt(lni, "ac");

	CXIP_INFO("Resource usage: %s RGID: %u CQ: %u PTE: %u EQ: %u CT: %u AC: %u\n",
		  lni->iface->info->device_name, lni->lni->id, cq_count,
		  pt_count, eq_count, ct_count, ac_count);
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
	if_entry = cxip_if_lookup_addr(nic_addr);
	if (!if_entry) {
		CXIP_DBG("interface not found\n");
		return -FI_ENODEV;
	}

	if (!if_entry->info->link_state) {
		CXIP_INFO("Interface %s link down.\n",
			  if_entry->info->device_name);
		return -FI_ENODEV;
	}

	/* Lock the IF to serialize opening the device */
	fastlock_acquire(&if_entry->lock);

	if (!if_entry->dev) {
		ret = cxil_open_device(if_entry->info->dev_id, &if_entry->dev);
		if (ret) {
			CXIP_WARN("Failed to open CXI Device, ret: %d\n", ret);
			ret = -FI_ENODEV;
			goto unlock;
		}

		CXIP_DBG("Opened %s\n", if_entry->info->device_name);
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

		CXIP_DBG("Closed %s\n", iface->info->device_name);
	}

	fastlock_release(&iface->lock);
}

/*
 * cxip_alloc_lni() - Allocate an LNI
 */
int cxip_alloc_lni(struct cxip_if *iface, uint32_t svc_id,
		   struct cxip_lni **if_lni)
{
	struct cxip_lni *lni;
	int ret;

	lni = calloc(1, sizeof(*lni));
	if (!lni) {
		CXIP_WARN("Unable to allocate LNI\n");
		return -FI_ENOMEM;
	}

	ret = cxil_alloc_lni(iface->dev, &lni->lni, svc_id);
	if (ret) {
		CXIP_WARN("Failed to allocate LNI, ret: %d\n", ret);
		ret = -FI_ENOSPC;
		goto free_lni;
	}

	lni->iface = iface;
	fastlock_init(&lni->lock);

	CXIP_DBG("Allocated LNI, %s RGID: %u\n",
		 lni->iface->info->device_name, lni->lni->id);

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
	int i;

	cxip_lni_res_dump(lni);

	CXIP_DBG("Freeing LNI, %s RGID: %u\n",
		 lni->iface->info->device_name, lni->lni->id);

	for (i = 0; i < lni->n_cps; i++) {
		ret = cxil_destroy_cp(lni->cps[i]);
		if (ret)
			CXIP_WARN("Failed to destroy CP: %d\n", ret);
	}

	ret = cxil_destroy_lni(lni->lni);
	if (ret)
		CXIP_WARN("Failed to destroy LNI: %d\n", ret);

	free(lni);
}

static const char * const tc_strs[] = {
	[CXI_TC_DEDICATED_ACCESS] = "DEDICATED_ACCESS",
	[CXI_TC_LOW_LATENCY] = "LOW_LATENCY",
	[CXI_TC_BULK_DATA] = "BUlK_DATA",
	[CXI_TC_BEST_EFFORT] = "BEST_EFFORT",
};

const char *cxi_tc_str(enum cxi_traffic_class tc)
{
	if (tc < CXI_TC_DEDICATED_ACCESS || tc > CXI_TC_MAX)
		return NULL;
	return tc_strs[tc];
}

enum cxi_traffic_class cxip_ofi_to_cxi_tc(uint32_t ofi_tclass)
{
	switch (ofi_tclass) {
	case FI_TC_BULK_DATA:
		return CXI_TC_BULK_DATA;
	case FI_TC_DEDICATED_ACCESS:
		return CXI_TC_DEDICATED_ACCESS;
	case FI_TC_LOW_LATENCY:
		return CXI_TC_LOW_LATENCY;
	case FI_TC_BEST_EFFORT:
	case FI_TC_NETWORK_CTRL:
	case FI_TC_SCAVENGER:
	default:
		return CXI_TC_BEST_EFFORT;
	}
}

static int cxip_cp_get(struct cxip_lni *lni, uint16_t vni,
		       enum cxi_traffic_class tc, bool hrp, struct cxi_cp **cp)
{
	int ret;
	int i;

	fastlock_acquire(&lni->lock);

	for (i = 0; i < lni->n_cps; i++) {
		if (lni->cps[i]->vni == vni && lni->cps[i]->tc == tc &&
		    lni->cps[i]->hrp == hrp) {
			*cp = lni->cps[i];
			ret = FI_SUCCESS;
			goto unlock;
		}
	}

	ret = cxil_alloc_cp(lni->lni, vni, tc, hrp, &lni->cps[lni->n_cps]);
	if (!ret) {
		CXIP_DBG("Allocated CP: %u VNI: %u TC: %s HRP: %u\n",
			 lni->cps[lni->n_cps]->lcid, vni, cxi_tc_str(tc),
			 hrp);
		*cp = lni->cps[lni->n_cps++];
		ret = FI_SUCCESS;
	} else {
		CXIP_DBG("Failed to allocate CP, ret: %d VNI: %u TC: %u HRP: %u\n",
			  ret, vni, tc, hrp);
		ret = -FI_EINVAL;
	}

unlock:
	fastlock_release(&lni->lock);

	return ret;
}

int cxip_txq_cp_set(struct cxip_cmdq *cmdq, uint16_t vni,
		     enum cxi_traffic_class tc, bool hrp)
{
	struct cxi_cp *cp;
	int ret;

	if (cmdq->vni == vni && cmdq->tc == tc && cmdq->hrp == hrp)
		return FI_SUCCESS;

	ret = cxip_cp_get(cmdq->lni, vni, tc, hrp, &cp);
	if (ret != FI_SUCCESS) {
		CXIP_DBG("Failed to get CP: %d\n", ret);
		return -FI_EOTHER;
	}

	ret = cxi_cq_emit_cq_lcid(cmdq->dev_cmdq, cp->lcid);
	if (ret) {
		CXIP_DBG("Failed to update CMDQ(%p) CP: %d\n", cmdq, ret);
		ret = -FI_EAGAIN;
	} else {
		ret = FI_SUCCESS;
		cmdq->vni = vni;
		cmdq->tc = tc;
		cmdq->hrp = hrp;

		CXIP_DBG("Updated CMDQ(%p) CP: %d VNI: %u TC: %s HRP: %u\n",
			 cmdq, cp->lcid, vni, cxi_tc_str(tc), hrp);
	}

	return ret;
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
		CXIP_WARN("Failed to allocate IF domain\n");
		return -FI_ENOMEM;
	}

	ret = cxil_alloc_domain(lni->lni, vni, pid, &dom->dom);
	if (ret) {
		CXIP_WARN("Failed to allocate CXI Domain, ret: %d\n", ret);
		ret = -FI_ENOSPC;
		goto free_dom;
	}

	dom->lni = lni;

	CXIP_DBG("Allocated IF Domain, %s VNI: %u PID: %u\n",
		 lni->iface->info->device_name, vni, dom->dom->pid);

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

	CXIP_DBG("Freeing IF Domain, %s VNI: %u PID: %u\n",
		 if_dom->lni->iface->info->device_name, if_dom->dom->vni,
		 if_dom->dom->pid);

	ret = cxil_destroy_domain(if_dom->dom);
	if (ret)
		CXIP_WARN("Failed to destroy domain: %d\n", ret);

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
		CXIP_DBG("Failed to write Append command: %d\n", rc);

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
		CXIP_DBG("Failed to write Append command: %d\n", rc);

		fastlock_release(&cmdq->lock);

		/* Return error according to Domain Resource Management */
		return -FI_EAGAIN;
	}

	cxi_cq_ring(cmdq->dev_cmdq);

	fastlock_release(&cmdq->lock);

	return FI_SUCCESS;
}

/*
 * cxip_pte_map() - Map a PtlTE to a specific PID index. A single PtlTE can be
 * mapped into MAX_PTE_MAP_COUNT different PID indices.
 */
int cxip_pte_map(struct cxip_pte *pte, uint64_t pid_idx, bool is_multicast)
{
	int ret;

	if (pte->pte_map_count >= MAX_PTE_MAP_COUNT)
		return -FI_ENOSPC;

	ret = cxil_map_pte(pte->pte, pte->if_dom->dom, pid_idx, is_multicast,
			   &pte->pte_map[pte->pte_map_count]);
	if (ret) {
		CXIP_WARN("Failed to map PTE: %d\n", ret);
		return -FI_EADDRINUSE;
	}

	pte->pte_map_count++;

	return FI_SUCCESS;
}

/*
 * cxip_pte_alloc_nomap() - Allocate a PtlTE without performing any mapping
 * during allocation.
 */
int cxip_pte_alloc_nomap(struct cxip_if_domain *if_dom, struct cxi_eq *evtq,
			 struct cxi_pt_alloc_opts *opts,
			 void (*state_change_cb)(struct cxip_pte *pte,
						 enum c_ptlte_state state),
			 void *ctx, struct cxip_pte **pte)
{
	struct cxip_pte *new_pte;
	int ret;

	new_pte = calloc(1, sizeof(*new_pte));
	if (!new_pte) {
		CXIP_WARN("Failed to allocate PTE structure\n");
		return -FI_ENOMEM;
	}

	/* Allocate a PTE */
	ret = cxil_alloc_pte(if_dom->lni->lni, evtq, opts,
			     &new_pte->pte);
	if (ret) {
		CXIP_WARN("Failed to allocate PTE: %d\n", ret);
		ret = -FI_ENOSPC;
		goto free_mem;
	}

	fastlock_acquire(&if_dom->lni->iface->lock);
	dlist_insert_tail(&new_pte->pte_entry, &if_dom->lni->iface->ptes);
	fastlock_release(&if_dom->lni->iface->lock);

	new_pte->if_dom = if_dom;
	new_pte->state_change_cb = state_change_cb;
	new_pte->ctx = ctx;

	*pte = new_pte;

	return FI_SUCCESS;

free_mem:
	free(new_pte);

	return ret;
}

/*
 * cxip_pte_alloc() - Allocate and map a PTE for use.
 */
int cxip_pte_alloc(struct cxip_if_domain *if_dom, struct cxi_eq *evtq,
		   uint64_t pid_idx, bool is_multicast,
		   struct cxi_pt_alloc_opts *opts,
		   void (*state_change_cb)(struct cxip_pte *pte,
					   enum c_ptlte_state state),
		   void *ctx, struct cxip_pte **pte)
{
	int ret;

	ret = cxip_pte_alloc_nomap(if_dom, evtq, opts, state_change_cb,
				   ctx, pte);
	if (ret)
		return ret;

	ret = cxip_pte_map(*pte, pid_idx, is_multicast);
	if (ret)
		goto free_pte;

	return FI_SUCCESS;

free_pte:
	cxip_pte_free(*pte);

	return ret;
}

/*
 * cxip_pte_free() - Free a PTE.
 */
void cxip_pte_free(struct cxip_pte *pte)
{
	int ret;
	int i;

	fastlock_acquire(&pte->if_dom->lni->iface->lock);
	dlist_remove(&pte->pte_entry);
	fastlock_release(&pte->if_dom->lni->iface->lock);

	for (i = pte->pte_map_count; i > 0; i--) {
		ret = cxil_unmap_pte(pte->pte_map[i - 1]);
		if (ret)
			CXIP_WARN("Failed to unmap PTE: %d\n", ret);
	}

	ret = cxil_destroy_pte(pte->pte);
	if (ret)
		CXIP_WARN("Failed to free PTE: %d\n", ret);

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
int cxip_cmdq_alloc(struct cxip_lni *lni, struct cxi_eq *evtq,
		    struct cxi_cq_alloc_opts *cq_opts, uint16_t vni,
		    enum cxi_traffic_class tc, bool hrp,
		    struct cxip_cmdq **cmdq)
{
	int ret;
	struct cxi_cq *dev_cmdq;
	struct cxip_cmdq *new_cmdq;
	struct cxi_cp *cp = NULL;

	new_cmdq = calloc(1, sizeof(*new_cmdq));
	if (!new_cmdq) {
		CXIP_WARN("Unable to allocate CMDQ structure\n");
		return -FI_ENOMEM;
	}

	if (cq_opts->flags & CXI_CQ_IS_TX) {
		ret = cxip_cp_get(lni, vni, tc, hrp, &cp);
		if (ret != FI_SUCCESS) {
			CXIP_WARN("Failed to allocate CP: %d\n", ret);
			return ret;
		}
		cq_opts->lcid = cp->lcid;
	}

	ret = cxil_alloc_cmdq(lni->lni, evtq, cq_opts, &dev_cmdq);
	if (ret) {
		CXIP_WARN("Failed to allocate %s, ret: %d\n",
			  cq_opts->flags & CXI_CQ_IS_TX ? "TXQ" : "TGQ", ret);
		ret = -FI_ENOSPC;
		goto free_cmdq;
	}

	new_cmdq->dev_cmdq = dev_cmdq;
	new_cmdq->vni = vni;
	new_cmdq->tc = tc;
	new_cmdq->hrp = hrp;
	new_cmdq->lni = lni;

	if (lni->iface->info->device_platform == CXI_PLATFORM_NETSIM)
		new_cmdq->llring_mode = CXIP_LLRING_NEVER;
	else
		new_cmdq->llring_mode = cxip_env.llring_mode;

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
		CXIP_WARN("cxil_destroy_cmdq failed, ret: %d\n", ret);

	fastlock_destroy(&cmdq->lock);
	free(cmdq);
}

/*
 * cxip_query_if_list() - Populate static IF data during initialization.
 */
static void cxip_query_if_list(struct slist *if_list)
{
	struct cxip_if *if_entry;
	int ret;
	int i;

	slist_init(if_list);

	ret = cxil_get_device_list(&cxi_dev_list);
	if (ret) {
		CXIP_WARN("cxil_get_device_list failed\n");
		return;
	}

	if (cxi_dev_list->count == 0) {
		CXIP_DBG("No IFs found\n");
		cxil_free_device_list(cxi_dev_list);
		return;
	}

	if (cxi_dev_list->info[0].min_free_shift) {
		CXIP_WARN("Non-zero min_free_shift not supported\n");
		cxil_free_device_list(cxi_dev_list);
		return;
	}

	for (i = 0; i < cxi_dev_list->count; i++) {
		if_entry = calloc(1, sizeof(struct cxip_if));
		if_entry->info = &cxi_dev_list->info[i];
		ofi_atomic_initialize32(&if_entry->ref, 0);
		dlist_init(&if_entry->ptes);
		fastlock_init(&if_entry->lock);
		slist_insert_tail(&if_entry->if_entry, if_list);
	}

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

	cxil_free_device_list(cxi_dev_list);
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
