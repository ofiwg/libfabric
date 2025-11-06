// SPDX-License-Identifier: GPL-2.0
/* Copyright 2018-2020 Hewlett Packard Enterprise Development LP */

/* Create and destroy Cassini portal tables */

#include <linux/kernel.h>
#include <linux/types.h>
#include <linux/hpe/cxi/cxi.h>
#include <linux/iopoll.h>
#include <linux/uaccess.h>
#include <linux/delay.h>

#include "cass_core.h"
#include "cass_ss1_debugfs.h"

/* Use the last logical endpoint for rendezvous offload. */
#define RDZV_GET_IDX(pid_granule) ((pid_granule) - 1)

static bool rdzv_get_en_default = true;
module_param(rdzv_get_en_default, bool, 0444);
MODULE_PARM_DESC(rdzv_get_en_default, "Hardware rendezvous get enable default");

#define PE_LE_MAX (C_LPE_STS_LIST_ENTRIES_ENTRIES / C_PE_COUNT)

/* By default, all pools share all LEs. Limits are set via use of a service.
 * Reduce pe_total_les to limit the number of LEs available.
 */
unsigned int pe_total_les = PE_LE_MAX;
module_param(pe_total_les, uint, 0644);
MODULE_PARM_DESC(pe_total_les, "Maximum number of LEs accessible to a PE");

static void init_le_pools(struct cass_dev *hw)
{
	int pe;
	int pool;
	union c_lpe_cfg_pe_le_pools le_pool;

	/* Validate module parameters */
	if (pe_total_les > PE_LE_MAX) {
		pe_total_les = PE_LE_MAX;
		cxidev_warn(&hw->cdev, "Total LEs (%u) exceeded maximum (%u).\n",
			    pe_total_les, PE_LE_MAX);
	}

	/* Configure LE Pools */
	for (pe = 0; pe < C_PE_COUNT; pe++) {
		union c_lpe_cfg_pe_le_shared le_shared = {
			.num_total = pe_total_les,
			.num_shared = pe_total_les,
		};

		cass_write(hw, C_LPE_CFG_PE_LE_SHARED(pe), &le_shared,
			   sizeof(le_shared));

		for (pool = 0; pool < CASS_NUM_LE_POOLS; pool++) {

			le_pool.max_alloc = pe_total_les;
			le_pool.num_reserved = 0;
			cass_config_lpe_reserve_pool(hw, pe, pool, &le_pool);

			cxidev_dbg(&hw->cdev, "pe: %u pool: %u max: %u res: %u\n",
				   pe, pool, le_pool.max_alloc,
				   le_pool.num_reserved);
		}

		cxidev_dbg(&hw->cdev, "pe: %u total: %u shared: %u\n",
			   pe, pe_total_les, le_shared.num_shared);

		/* Reliquish the core. This takes a long time on the Z1. */
		if (HW_PLATFORM_Z1(hw))
			cond_resched();
	}
}

/* Do necessary CSR updates to reserve or release(share) LEs */
void cass_cfg_le_pools(struct cass_dev *hw, int pool_id, int pe,
		       const struct cxi_limits *les, bool release)
{
	int sign = 1;
	union c_lpe_cfg_pe_le_shared le_shared;
	union c_lpe_cfg_pe_le_pools le_pool = {
		.max_alloc = les->max,
		.num_reserved = les->res,
	};

	if (release) {
		le_pool.max_alloc = les->max;
		le_pool.num_reserved = 0;
		sign = -1;
	}

	cass_read(hw, C_LPE_CFG_PE_LE_SHARED(pe), &le_shared,
		  sizeof(le_shared));
	le_shared.num_shared -= (les->res * sign);

	/*
	 * NUM_SHARED should be reduced before
	 * dedicating entries to pools.
	 */
	if (release) {
		cass_config_lpe_reserve_pool(hw, pe, pool_id, &le_pool);
		cass_write(hw, C_LPE_CFG_PE_LE_SHARED(pe), &le_shared,
			   sizeof(le_shared));
	} else {
		cass_write(hw, C_LPE_CFG_PE_LE_SHARED(pe), &le_shared,
			   sizeof(le_shared));
		cass_config_lpe_reserve_pool(hw, pe, pool_id, &le_pool);
	}
}

void cass_pte_set_get_ctrl(struct cass_dev *hw)
{
	union c_lpe_cfg_get_ctrl cfg_get_ctrl = {
		.transaction_type = TRANSACTION_TYPE,
		.get_en = hw->cdev.prop.rdzv_get_en,
		.dfa_nid = hw->cdev.prop.nid,
	};

	cfg_get_ctrl.get_index = cxi_build_dfa_ep(0, hw->cdev.prop.pid_bits,
						  hw->cdev.prop.rdzv_get_idx);
	cfg_get_ctrl.get_index_ext =
		cxi_build_dfa_ext(hw->cdev.prop.rdzv_get_idx);

	cass_write(hw, C_LPE_CFG_GET_CTRL, &cfg_get_ctrl,
		   sizeof(cfg_get_ctrl));
}

void cass_pte_init(struct cass_dev *hw)
{
	const union c_lpe_cfg_initiator_ctrl cfg_initiator_ctrl = {
		.pid_bits = hw->cdev.prop.pid_bits,
	};

	hw->cdev.prop.rdzv_get_idx = RDZV_GET_IDX(hw->cdev.prop.pid_granule);

	/*
	 * Configure CSRs for HW rendezvous offload. The target NIC will build
	 * the GET DFA by utilizing the source PID within the initiator field
	 * from rendezvous command. In order for the target NIC to target the
	 * correct PtlTE, SW must allocate and map a PtlTE at the max PID offset
	 * value. For example, if number of PIDs equal 512 (9 bits for PID), 8
	 * bits remain for PID offset. A rendezvous PtlTE needs to be mapped at
	 * PID offset 0xFF.
	 */
	cass_write(hw, C_LPE_CFG_INITIATOR_CTRL, &cfg_initiator_ctrl,
		   sizeof(cfg_initiator_ctrl));

	hw->cdev.prop.rdzv_get_en = rdzv_get_en_default;
	cass_pte_set_get_ctrl(hw);

	init_le_pools(hw);
}

/* Update an MST return code, and return true if the entry is still
 * valid.
 */
static bool mst_rc_update(struct cass_dev *hw,
			  const union c_mst_cfg_rc_update *mst_update)
{
	const void __iomem *csr = cass_csr(hw, C_MST_STS_RC_STS);
	union c_mst_sts_rc_sts mst_update_sts;
	int rc;

	mutex_lock(&hw->mst_update_lock);

	cass_write(hw, C_MST_CFG_RC_UPDATE, mst_update, sizeof(*mst_update));

	/* Wait for MST entry update to complete. */
	rc = readq_poll_timeout(csr, mst_update_sts.qw,
				mst_update_sts.done == 1, 1, 1000);

	mutex_unlock(&hw->mst_update_lock);

	cxidev_WARN_ONCE(&hw->cdev, rc, "Timeout waiting for MST RC update");

	return mst_update_sts.done == 1 && mst_update_sts.match == 1;
}

/*
 * Clean the entries in the MST table that refer to that PT entry.
 */
static void pt_mst_cleanup(struct cxi_pte_priv *pt)
{
	struct cxi_dev *cdev = pt->lni_priv->dev;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	const unsigned int ptlte_index = pt->pte.id;
	int i;
	union c_mst_cfg_rc_update cfg_rc_update = {
		.return_code = C_RC_MST_CANCELLED,
		.portal_vld = 1,
		.portal_idx = ptlte_index,
	};
	union c_mst_sts_num_req mst_sts;

	if (!HW_PLATFORM_NETSIM(hw))
		cass_read(hw, C_MST_STS_NUM_REQ, &mst_sts, sizeof(mst_sts));
	else
		mst_sts.cnt = C_MST_DBG_MST_TABLE_ENTRIES;

	for (i = 0; i < C_MST_DBG_MST_TABLE_ENTRIES && mst_sts.cnt; i++) {
		/* Read only the 3rd 64 bits word from the entry, for
		 * the portal_index field.
		 */

		/* TODO: (TAG_NO_PTE_0) we need to determine whether
		 * an MST is valid before accessing it.
		 * Currently, we rely on the pte index to not be 0.
		 * This prevents lots on unneeded writes to C_MST_CFG_RC_UPDATE.
		 */
		struct c_mst_table_simple mst_entry;
		u64 offset = C_MST_DBG_MST_TABLE(i) + C_MST_TABLE_SIMPLE_OFFSET;

		cass_read(hw, offset, &mst_entry, sizeof(mst_entry));

		if (mst_entry.portal_index != ptlte_index)
			continue;

		cfg_rc_update.mst_idx = i;

		if (mst_rc_update(hw, &cfg_rc_update))
			set_bit(i, pt->mst_rc_update);

		mst_sts.cnt--;
	}

	cass_flush_pci(hw);
}

#define PE_FREE_MASK_LEN (C_LPE_STS_PE_FREE_MASK_ENTRIES / C_PE_COUNT)

/*
 * Set the free bit in the corresponding FREE_MASK CSRs.
 */
static void cxi_pte_le_mark_free(struct cass_dev *hw, unsigned int le_idx,
				 unsigned int pe_num)
{
	unsigned int fm_idx;
	unsigned int fm_bit;
	u64 offset;
	union c_lpe_sts_pe_free_mask sts_pe_free_mask;

	fm_idx = le_idx % PE_FREE_MASK_LEN;
	fm_bit = le_idx / PE_FREE_MASK_LEN;
	sts_pe_free_mask.free_mask = BIT(fm_bit);

	/* All 4 pools are consecutive in memory, so only use pool0
	 * for the address computation.
	 */
	offset = C_LPE_STS_PE_FREE_MASK(pe_num * PE_FREE_MASK_LEN + fm_idx);

	cass_write(hw, offset, &sts_pe_free_mask, sizeof(sts_pe_free_mask));
}

/* Walk a PtlTE list, and free the entries. The list must be valid. */
static void cxi_pte_le_free(struct cass_dev *hw,
			    const struct c_lpe_cfg_ptl_table_ptrs *list,
			    unsigned int pe_num, unsigned int le_pool)
{
	const union c_lpe_msc_le_pool_free le_pool_free = {
		.free_en = 1,
		.le_pool = le_pool,
	};
	unsigned int le_idx = list->head;

	/* Walk the list entries marking each one as ready to be freed. */
	while (true) {
		/* Read only the 6th 64 bits from the entry. */
		struct c_list_entry_simple le;
		u64 offset =
			C_LPE_STS_LIST_ENTRIES(pe_num * PE_LE_MAX + le_idx) +
			5 * sizeof(u64);

		cass_read(hw, offset, &le, sizeof(le));

		cxi_pte_le_mark_free(hw, le_idx, pe_num);

		/* Tell NIC an entry was freed */
		cass_write(hw, C_LPE_MSC_LE_POOL_FREE(pe_num),
			   &le_pool_free, sizeof(le_pool_free));

		if (le.vld) {
			le_idx = le.addr;
		} else {
			WARN_ON_ONCE(list->tail != le_idx);
			break;
		}
	}
}

/* Clean up resources associated with the PtlTE. */
static void pt_cleanup(struct cxi_pte_priv *pt)
{
	struct cxi_lni_priv *lni = pt->lni_priv;
	struct cxi_dev *cdev = lni->dev;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	union c_lpe_cfg_ptl_table ptl_table;
	int i;

	/* Cleaning Up Portal Table Entries */

	spin_lock(&hw->lpe_shadow_lock);
	cass_read_pte(hw, pt->pte.id, &ptl_table);
	spin_unlock(&hw->lpe_shadow_lock);

	/* Walk the various lists in the PtlTE, free LEs, and
	 * invalidate the list.
	 */
	for (i = 0; i < ARRAY_SIZE(ptl_table.l); i++) {
		struct c_lpe_cfg_ptl_table_ptrs *list = &ptl_table.l[i];

		if (!list->list_vld)
			continue;

		cxi_pte_le_free(hw, list, ptl_table.pe_num,
				ptl_table.le_pool);
	}

	pt_mst_cleanup(pt);

	/* Clear the entire PTE descriptor including list_vld and list
	 * pointers.
	 */
	memset(&ptl_table, 0, sizeof(ptl_table));
	cass_write(hw, C_LPE_CFG_PTL_TABLE(pt->pte.id),
		   &ptl_table, sizeof(ptl_table));

	cass_flush_pci(hw);
}

/**
 * cxi_pte_alloc() - Allocate a portal table entry (PTE)
 *
 * @lni: LNI used for PTE allocation
 * @evtq: Event Queue to associate with the PTE
 * @opts: Options affecting the PTE
 *
 * @return: the PTE or an error pointer
 */
struct cxi_pte *cxi_pte_alloc(struct cxi_lni *lni, struct cxi_eq *evtq,
			      const struct cxi_pt_alloc_opts *opts)
{
	struct cxi_lni_priv *lni_priv =
		container_of(lni, struct cxi_lni_priv, lni);
	struct cxi_dev *cdev = lni_priv->dev;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	struct cxi_pte_priv *pt;
	struct cxi_eq_priv *eq = NULL;
	int pt_n;
	int rc;
	union c_lpe_cfg_ptl_table ptl_table = {
		.en_event_match = opts->en_event_match,
		.clr_remote_offset = opts->clr_remote_offset,
		.en_flowctrl = opts->en_flowctrl,
		.use_long_event = opts->use_long_event,
		.lossless = opts->lossless,
		.en_restricted_unicast_lm = opts->en_restricted_unicast_lm,
		.use_logical = opts->use_logical,
		.is_matching = opts->is_matching,
		.en_match_on_vni = opts->en_match_on_vni,
		.do_space_check = opts->do_space_check,
		.en_align_lm = opts->en_align_lm,
		.en_sw_hw_st_chng = opts->en_sw_hw_st_chng,
		.ptl_state = opts->ethernet ?
			C_PTLTE_ETHERNET : C_PTLTE_DISABLED,
		.signal_invalid = opts->signal_invalid,
		.signal_underflow = opts->signal_underflow,
		.signal_overflow = opts->signal_overflow,
		.signal_inexact = opts->signal_inexact,
	};
	bool pt_reuse = false;
	int count;

	if (opts->ethernet && !capable(CAP_NET_RAW))
		return ERR_PTR(-EPERM);

	if (evtq)
		eq = container_of(evtq, struct cxi_eq_priv, eq);

	/* Prefer a PTE already allocated to this LNI */
	spin_lock(&lni_priv->res_lock);
	pt = list_first_entry_or_null(&lni_priv->pt_cleanups_list,
				      struct cxi_pte_priv, list);
	if (pt) {
		list_del(&pt->list);
		spin_unlock(&lni_priv->res_lock);

		pt_reuse = true;
		pt_n = pt->pte.id;
	} else {
		spin_unlock(&lni_priv->res_lock);

		pt = kzalloc(sizeof(*pt), GFP_KERNEL);
		if (pt == NULL)
			return ERR_PTR(-ENOMEM);

		/* Check the associated service to see if this PTE can be
		 * allocated
		 */
		rc = cxi_rgroup_alloc_resource(lni_priv->rgroup,
					       CXI_RESOURCE_PTLTE);
		if (rc)
			goto pt_free;

		/* TODO: Don't allocate PTE 0 until we can determine whether
		 * an MST entry is valid. See tag TAG_NO_PTE_0 in this file.
		 */
		pt_n = ida_simple_get(&hw->pte_table, 1,
				      C_LPE_CFG_PTL_TABLE_ENTRIES, GFP_KERNEL);
		if (pt_n < 0) {
			rc = pt_n;
			goto dec_rsrc_use;
		}
	}

	pt->lni_priv = lni_priv;
	pt->eq = eq;
	pt->pte.id = pt_n;
	pt->mcast_n = -1;
	refcount_set(&pt->refcount, 1);
	bitmap_zero(pt->mst_rc_update, C_MST_DBG_MST_TABLE_ENTRIES);

	if (!ptl_table.is_matching && ptl_table.ptl_state != C_PTLTE_ETHERNET &&
	    !ptl_table.do_space_check) {
		ptl_table.do_space_check = true;

		while (true) {
			count = atomic_read(&hw->plec_count);
			if (count > PLEC_SIZE) {
				cxidev_dbg(cdev, "plec full\n");
				break;
			}

			rc = atomic_cmpxchg(&hw->plec_count, count, count + 1);
			if (rc == count) {
				pt->plec_enabled = true;
				ptl_table.do_space_check = false;
				break;
			}
		}
	}

	/* TODO: Have separate EQs for no rendezvous get events. */
	if (eq) {
		ptl_table.eq_handle = eq->eq.eqn;
		ptl_table.eq_handle_no_rget = eq->eq.eqn;
	} else {
		ptl_table.eq_handle = C_EQ_NONE;
		ptl_table.eq_handle_no_rget = C_EQ_NONE;
	}

	cass_assign_ptlte_to_rgid(hw, pt->pte.id, lni_priv->lni.rgid);

	ptl_table.pe_num = cxi_lni_get_pe_num(lni_priv);
	pt->pe_num = ptl_table.pe_num;
	ptl_table.le_pool = pt->le_pool =
		cxi_rgroup_le_pool_id(pt->lni_priv->rgroup, pt->pe_num);

	spin_lock(&hw->lpe_shadow_lock);
	cass_config_pte(hw, pt_n, &ptl_table);
	spin_unlock(&hw->lpe_shadow_lock);

	spin_lock(&lni_priv->res_lock);
	list_add_tail(&pt->list, &lni_priv->pt_list);
	spin_unlock(&lni_priv->res_lock);

	if (!pt_reuse) {
		atomic_inc(&hw->stats.pt);

		pt_debugfs_create(pt->pte.id, pt, hw, lni_priv);
		refcount_inc(&lni_priv->refcount);
	}

	if (eq)
		refcount_inc(&eq->refcount);

	cass_flush_pci(hw);

	return &pt->pte;

dec_rsrc_use:
	if (!pt_reuse)
		cxi_rgroup_free_resource(lni_priv->rgroup, CXI_RESOURCE_PTLTE);
pt_free:
	if (pt_reuse) {
		spin_lock(&lni_priv->res_lock);
		list_add_tail(&pt->list, &lni_priv->pt_cleanups_list);
		spin_unlock(&lni_priv->res_lock);
	} else {
		kfree(pt);
	}

	return ERR_PTR(rc);
}
EXPORT_SYMBOL(cxi_pte_alloc);

/**
 * cxi_pte_free() - Release a portal table entry
 *
 * @pt: the portal table entry
 */
void cxi_pte_free(struct cxi_pte *pt)
{
	struct cxi_pte_priv *pt_priv =
		container_of(pt, struct cxi_pte_priv, pte);
	struct cxi_lni_priv *lni_priv = pt_priv->lni_priv;
	struct cxi_dev *cdev = lni_priv->dev;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	const u64 ptl_state = C_PTLTE_RESET;
	u64 before;
	u64 after;

	cxidev_WARN_ONCE(cdev, !refcount_dec_and_test(&pt_priv->refcount),
			 "Resource leaks - PT refcount not zero: %d\n",
			 refcount_read(&pt_priv->refcount));

	cass_assign_ptlte_to_rgid(hw, pt_priv->pte.id, C_RESERVED_RGID);

	/* Set the PtlTE to the Reset state. Only write the last
	 * 64-bits word, at offset 0x18, as to not overwrite the list
	 * pointers.
	 * Repeat as long as PLEC_FREES_CSR increases (Cassini ERRATA-4383).
	 */
	do {
		static const unsigned int offset =
			offsetof(struct c_lpe_cntrs_group, lpe.plec_frees_csr) /
			sizeof(u64);

		cass_read(hw, C_LPE_STS_EVENT_CNTS(offset), &before, sizeof(before));

		cass_write(hw, C_LPE_CFG_PTL_TABLE(pt_priv->pte.id) + 0x18,
			   &ptl_state, sizeof(ptl_state));

		cass_read(hw, C_LPE_STS_EVENT_CNTS(offset), &after, sizeof(after));
	} while (before != after);

	pt_cleanup(pt_priv);

	if (pt_priv->eq) {
		refcount_dec(&pt_priv->eq->refcount);
		pt_priv->eq = NULL;
	}

	if (pt_priv->plec_enabled) {
		atomic_dec(&hw->plec_count);
		pt_priv->plec_enabled = false;
	}

	spin_lock(&lni_priv->res_lock);
	list_del(&pt_priv->list);
	list_add_tail(&pt_priv->list, &lni_priv->pt_cleanups_list);
	spin_unlock(&lni_priv->res_lock);
}
EXPORT_SYMBOL(cxi_pte_free);

/* Check whether an MST entry is valid */
static bool is_mst_valid(struct cass_dev *hw, unsigned int mst)
{
	union c_mst_dbg_match_done match_done = {
		.mst_idx = mst,
	};

	spin_lock(&hw->mst_match_done_lock);
	cass_write(hw, C_MST_DBG_MATCH_DONE, &match_done, sizeof(match_done));
	cass_read(hw, C_MST_DBG_MATCH_DONE, &match_done, sizeof(match_done));
	spin_unlock(&hw->mst_match_done_lock);

	return match_done.mst_idx_st == 1;
}

/* Go through each MST that where the return_code was updated, and
 * check whether they are now not valid or reused.
 */
static bool is_pt_done(struct cass_dev *hw, struct cxi_pte_priv *pt)
{
	union c_mst_cfg_rc_update cfg_rc_update = {
		.return_code = C_RC_MST_CANCELLED,
		.portal_vld = 1,
		.portal_idx = pt->pte.id,
	};
	unsigned int mst;

	for_each_set_bit(mst, pt->mst_rc_update, C_MST_DBG_MST_TABLE_ENTRIES) {
		if (is_mst_valid(hw, mst)) {
			/* MST entry is still valid, but it may have
			 * been reused for another portal entry.
			 */
			cfg_rc_update.mst_idx = mst;
			if (mst_rc_update(hw, &cfg_rc_update))
				return false;
		}

		clear_bit(mst, pt->mst_rc_update);
	}

	return true;
}

/* When the LNI closes, the PTEs are ready to be cleaned up and
 * released.
 */
void finalize_pt_cleanups(struct cxi_lni_priv *lni, bool force)
{
	struct cxi_dev *cdev = lni->dev;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	struct cxi_pte_priv *pt;
	struct cxi_pte_priv *tmp;

	list_for_each_entry_safe(pt, tmp, &lni->pt_cleanups_list, list) {
		if (!force && !is_pt_done(hw, pt))
			continue;

		list_del(&pt->list);

		debugfs_remove(pt->lni_dir);
		debugfs_remove_recursive(pt->debug_dir);

		refcount_dec(&lni->refcount);
		atomic_dec(&hw->stats.pt);
		cxi_rgroup_free_resource(lni->rgroup, CXI_RESOURCE_PTLTE);
		ida_simple_remove(&hw->pte_table, pt->pte.id);
		kfree(pt);
	}
}

/**
 * cxi_pte_map() - Map a portal entry to a portal index entry
 *
 * A portal entry can be mapped several times to a portal index entry.
 *
 * @pt: the portal entry
 * @domain: the domain
 * @pid_offset: offset of the portal in the domain's PID slice for
 *   unicast, or the (union cxi_pte_map_offset) uintval field for multicast.
 * @is_multicast: whether the address is multicast
 * @pt_index: the allocated portal index entry, to be passed when
 *   unmapping.
 *
 * @return: 0 on success or a negative errno
 */
int cxi_pte_map(struct cxi_pte *pt, struct cxi_domain *domain,
		unsigned int pid_offset, bool is_multicast,
		unsigned int *pt_index)
{
	struct cxi_pte_priv *pt_priv =
			container_of(pt, struct cxi_pte_priv, pte);
	struct cxi_domain_priv *domain_priv =
			container_of(domain, struct cxi_domain_priv, domain);
	struct cxi_dev *cdev = pt_priv->lni_priv->dev;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	unsigned int ptl_idx;
	unsigned int index_ext;
	union c_rmu_cfg_portal_list rmu_cfg_portal_list = {};
	int mcast_n;
	int pti_n;
	int ret;

	/* Sanity checks */
	if (domain_priv->lni_priv != pt_priv->lni_priv)
		return -EINVAL;

	/* Generate ptl_idx and index_ext */
	if (is_multicast) {
		union cxi_pte_map_offset map_offset;

		if (pid_offset >= (1 << (C_DFA_MULTICAST_ID_BITS +
					 C_DFA_INDEX_EXT_BITS)))
			return -EINVAL;

		if (pt_priv->mcast_n != -1)
			return -EINVAL;

		/* structure maps directly to RMU fields.
		 * | vni | ... | 1:1 | pad:2 | ptl_idx:13    | index_ext:5  |
		 */
		map_offset.uintval = pid_offset;
		ptl_idx = map_offset.mcast_id;
		index_ext = map_offset.mcast_pte_index;
	} else {
		/* pid_offset does not map directly to RMU fields.
		 * | vni | ... | 0:1 | pad:3 | pid:9      |    pid_offset:8 |
		 * | vni | ... | 0:1 | pad:3 | ptl_idx:12    | index_ext:5  |
		 */
		if (pid_offset >= cdev->prop.pid_granule)
			return -EINVAL;
		ptl_idx = cxi_build_dfa_ep(domain->pid, cdev->prop.pid_bits,
					   pid_offset);
		index_ext = cxi_build_dfa_ext(pid_offset);
		if (ptl_idx >= (1 << C_DFA_ENDPOINT_DEFINED_BITS))
			return -EINVAL;
	}

	/* Acquire an unused RMU_CFG_PORTAL_LIST slot */
	pti_n = ida_simple_get(&hw->pt_index_table, 0,
			       C_RMU_CFG_PORTAL_LIST_ENTRIES,
			       GFP_KERNEL);
	if (pti_n < 0)
		return pti_n;

	if (is_multicast) {
		/* Prevent re-issue of this mcast_id/index_ext pair */
		mcast_n = ida_simple_get(&hw->multicast_table,
					 pid_offset,
					 pid_offset+1,
					 GFP_KERNEL);
		if (mcast_n < 0) {
			ret = mcast_n;
			goto rls_pti_n;
		}
		pt_priv->mcast_n = mcast_n;
	}

	refcount_inc(&domain_priv->refcount);
	refcount_inc(&pt_priv->refcount);

	rmu_cfg_portal_list.index_ext = index_ext;
	rmu_cfg_portal_list.multicast_id = ptl_idx;
	rmu_cfg_portal_list.is_multicast = is_multicast;
	rmu_cfg_portal_list.vni_list_idx =
				domain_priv->rx_profile->config.rmu_index;

	spin_lock(&hw->rmu_portal_list_lock);
	cass_config_portal_list(hw, pti_n, pt_priv->pte.id,
				&rmu_cfg_portal_list);
	spin_unlock(&hw->rmu_portal_list_lock);

	*pt_index = pti_n;

	return 0;
rls_pti_n:
	ida_simple_remove(&hw->pt_index_table, pti_n);
	return ret;
}
EXPORT_SYMBOL(cxi_pte_map);

/**
 * cxi_pte_unmap() - Unmap a portal entry from the index table
 *
 * @pt: the portal table entry
 * @domain: the domain to use
 * @pt_index: the portal entry index returned by cxi_pte_map()
 */
int cxi_pte_unmap(struct cxi_pte *pt, struct cxi_domain *domain, int pt_index)
{
	struct cxi_pte_priv *pt_priv =
			container_of(pt, struct cxi_pte_priv, pte);
	struct cxi_domain_priv *domain_priv =
			container_of(domain, struct cxi_domain_priv, domain);
	struct cxi_dev *cdev = domain_priv->lni_priv->dev;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);

	cass_invalidate_portal_list(hw, pt_index);

	/* If this was a multicast, release the ID and free the memory */
	if (pt_priv->mcast_n != -1) {
		ida_simple_remove(&hw->multicast_table, pt_priv->mcast_n);
		pt_priv->mcast_n = -1;
	}

	/* Release the pte index value */
	ida_simple_remove(&hw->pt_index_table, pt_index);

	refcount_dec(&pt_priv->refcount);
	refcount_dec(&domain_priv->refcount);

	return 0;
}
EXPORT_SYMBOL(cxi_pte_unmap);

/**
 * cxi_pte_le_invalidate() - Invalidate a non-locally managed persistent LE.
 * @pt: Portal table entry.
 * @buffer_id: Buffer ID used to identify the LE.
 * @list: Portal list the LE was appended to.
 *
 * Non-locally managed persistent LE invalidation is required due to the LE
 * unlink not cleaning up all resources.
 *
 * If the LE has success and communication events enabled, the PtlTE EQ should
 * be drained to ensure that no outstanding events exist on the EQ using this
 * buffer ID, PtlTE, and Ptl list (assuming the PtlTE was configured with an
 * EQ).
 *
 * If a user has enabled matching events for the PtlTE and is correctly counting
 * the number of match events and target DMA operation events
 * (Put, Get, etc...), persistent LE invalidation does not need to occur.
 *
 * Note: If the PtlTE is configured with a EQ, target DMA operation events may
 * complete with a RC of C_RC_MST_CANCELLED. These target events should be
 * ignored by the user.
 */
void cxi_pte_le_invalidate(struct cxi_pte *pt, unsigned int buffer_id,
			   enum c_ptl_list list)
{
	struct cxi_pte_priv *pt_priv =
		container_of(pt, struct cxi_pte_priv, pte);
	struct cxi_dev *cdev = pt_priv->lni_priv->dev;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	union c_mst_dbg_mst_table *mst_entry = hw->mst_entries;
	union c_mst_cfg_rc_update mst_update = {
		.return_code = C_RC_MST_CANCELLED,
		.buffer_idx = buffer_id,
		.portal_idx = pt->id,
		.ptl_list = list,
		.ptl_vld = 1,
		.portal_vld = 1,
		.buffer_vld = 1,
	};
	int i;
	int rc;

	mutex_lock(&hw->mst_table_lock);

	rc = cxi_dmac_xfer(cdev, hw->dmac_pt_id);
	if (cxidev_WARN_ONCE(cdev, rc, "Unable to get MST table: %d\n", rc)) {
		mutex_unlock(&hw->mst_table_lock);
		return;
	}

	/* Walk all the MST entries cleaning up matching entries. */
	for (i = 0; i < C_MST_DBG_MST_TABLE_ENTRIES; i++, mst_entry++) {
		if (mst_entry->portal_index == pt->id &&
		    mst_entry->ptl_list == list &&
		    mst_entry->buffer_id == buffer_id) {
			/* Issue a MST RC update to the NIC. */
			mst_update.mst_idx = i;

			mst_rc_update(hw, &mst_update);
		}
	}

	mutex_unlock(&hw->mst_table_lock);
}
EXPORT_SYMBOL(cxi_pte_le_invalidate);

#define ULE_START_OFFSET 24

/**
 * cxi_pte_status() - Return collection of PTE stats
 *
 * @pt: Portal table entry.
 * @status: Pointer to struct where PTE stats will be written
 *
 * status->ule_offsets and status->ule_count are used as inputs. ule_offsets
 * is an array where ULE information is stored on output. ule_count is the
 * size in entries of ule_offsets.
 *
 * If ule_offsets is set, the PTE unexpected list is walked. The remote
 * offset field of each ULE is written into ule_offsets (if space is
 * available). The total number of ULEs found is returned in ule_count. If
 * ule_offsets is unset, the Unexpected list walk is skipped. This interface
 * is used to query ULE remote offset information needed for Rendezvous Puts
 * that are dequeued using a SearchAndDelete command.
 */
int cxi_pte_status(struct cxi_pte *pt, struct cxi_pte_status *status)
{
	struct cxi_pte_priv *pt_priv =
		container_of(pt, struct cxi_pte_priv, pte);
	struct cxi_dev *cdev = pt_priv->lni_priv->dev;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	union c_lpe_cfg_ptl_table ptl_table;
	struct c_lpe_cfg_ptl_table_ptrs ule_list;
	union c_lpe_cfg_pe_le_pools le_pool;
	union c_lpe_sts_pe_le_alloc le_alloc;
	size_t ule_count = 0;
	unsigned int idx;

	spin_lock(&hw->lpe_shadow_lock);
	cass_read_pte(hw, pt_priv->pte.id, &ptl_table);
	spin_unlock(&hw->lpe_shadow_lock);

	idx = (pt_priv->pe_num * 16) + pt_priv->le_pool;

	cass_read(hw, C_LPE_CFG_PE_LE_POOLS(idx), &le_pool,
		  sizeof(le_pool));
	cass_read(hw, C_LPE_STS_PE_LE_ALLOC(idx), &le_alloc,
		  sizeof(le_alloc));

	status->drop_count = ptl_table.drop_count;
	status->state = ptl_table.ptl_state;
	status->les_reserved = le_pool.num_reserved;
	status->les_max = le_pool.max_alloc;
	status->les_allocated = le_alloc.num_allocated;
	ule_list = ptl_table.l[C_PTL_LIST_UNEXPECTED];

	if (status->ule_offsets && ule_list.list_vld) {
		unsigned int le_idx = ule_list.head;
		u64 ule_offset;
		struct c_ule_entry ule;
		void *ule_ptr = (void *)&ule + ULE_START_OFFSET;

#if KERNEL_VERSION(4, 18, 0) > LINUX_VERSION_CODE
		if (!user_access_begin(VERIFY_WRITE, status->ule_offsets,
				       status->ule_count * sizeof(u64)))
			return -EFAULT;
#else
		if (!user_access_begin(status->ule_offsets,
				       status->ule_count * sizeof(u64)))
			return -EFAULT;
#endif

		while (true) {
			/* Read only the last 3 QWs of the ULE */
			ule_offset = C_LPE_STS_LIST_ENTRIES(pt_priv->pe_num *
							    PE_LE_MAX +
							    le_idx) +
					ULE_START_OFFSET;
			cass_read(hw, ule_offset, ule_ptr,
				  sizeof(ule) - ULE_START_OFFSET);

			if (ule_count < status->ule_count) {
				u64 offset = ule.start_or_offset;

				unsafe_put_user(offset,
						&status->ule_offsets[ule_count],
						err_put_user);
			}
			ule_count++;

			if (ule.vld) {
				le_idx = ule.addr;
			} else {
				WARN_ON_ONCE(ule_list.tail != le_idx);
				break;
			}
		}

		user_access_end();
	}

	/* Return the real number of ULEs */
	status->ule_count = ule_count;

	return 0;

err_put_user:
	user_access_end();

	return -EFAULT;
}
EXPORT_SYMBOL(cxi_pte_status);

/**
 * cxi_pte_transition_sm() - Transition a disabled PTE to software managed.
 *
 * @pt: Portal table entry.
 * @drop_count: Expected drop_count
 *
 * If the PTE is in the disable state, and its drop_count matches,
 * this will transition it into the software managed state. The
 * drop_count is reset to 0.
 *
 * Returns -EINVAL if the PTE is not disabled, -ETIMEDOUT if the
 * hardware has issues, -EAGAIN if the drop_count doesn't match, or 0 on
 * success.
 */
int cxi_pte_transition_sm(struct cxi_pte *pt, unsigned int drop_count)
{
	struct cxi_pte_priv *pt_priv = container_of(pt, struct cxi_pte_priv, pte);
	struct cxi_dev *cdev = pt_priv->lni_priv->dev;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	union c_lpe_sts_cdts_in_use cdts_in_use_mask = {
		.event_cdts_in_use = ~0,
		.ixe_mst_cdts_in_use = ~0,
		.ixe_get_cdts_in_use = ~0,
		.ixe_put_cdts_in_use = ~0,
	};
	union c_lpe_sts_cdts_in_use cdts_in_use;
	union c_lpe_cfg_cdt_limits cdt_limits;
	union c_lpe_cfg_ptl_table ptl_table;
	u8 old_rrq_cdts;
	int count;
	int ret;

	spin_lock(&hw->lpe_shadow_lock);
	cass_read_pte(hw, pt_priv->pte.id, &ptl_table);
	spin_unlock(&hw->lpe_shadow_lock);

	if (ptl_table.ptl_state == C_PTLTE_SOFTWARE_MANAGED)
		return 0;

	if (ptl_table.ptl_state != C_PTLTE_DISABLED)
		return -EINVAL;

	mutex_lock(&hw->pte_transition_sm_lock);

	/* Stop all LPE requests arbitration before the point where
	 * the PtlTE RAM is read.
	 */
	cass_read(hw, C_LPE_CFG_CDT_LIMITS, &cdt_limits, sizeof(cdt_limits));
	old_rrq_cdts = cdt_limits.rrq_cdts;
	cdt_limits.rrq_cdts = 0;
	cass_write(hw, C_LPE_CFG_CDT_LIMITS, &cdt_limits, sizeof(cdt_limits));

	/* Read stats until load and max are 0 */
	count = 0;
	do {
		static union c_lpe_msc_load_stats_doorbell doorbell = {
			.doorbell = 1,
		};
		union c_lpe_sts_load_stats stats;

		cass_write(hw, C_LPE_MSC_LOAD_STATS_DOORBELL,
			   &doorbell, sizeof(doorbell));

		cass_read(hw, C_LPE_STS_LOAD_STATS(ptl_table.pe_num),
			  &stats, sizeof(stats));

		if (stats.max == 0 && stats.load == 0)
			break;

		count++;
		if (count >= 250) {
			ret = -ETIMEDOUT;
			goto out;
		}

		usleep_range(5000, 10000);
	} while (true);

	/* Add the matching PE*_RRQ_CDTS_IN_USE to the mask */
	switch (ptl_table.pe_num) {
	case 0:
		cdts_in_use_mask.pe0_rrq_cdts_in_use = ~0;
		break;
	case 1:
		cdts_in_use_mask.pe1_rrq_cdts_in_use = ~0;
		break;
	case 2:
		cdts_in_use_mask.pe2_rrq_cdts_in_use = ~0;
		break;
	case 3:
		cdts_in_use_mask.pe3_rrq_cdts_in_use = ~0;
		break;
	}

	/* Poll until selected fields in C_LPE_STS_CDTS_IN_USE are all
	 * zeroes.
	 */
	ret = readq_poll_timeout(cass_csr(hw, C_LPE_STS_CDTS_IN_USE),
				 cdts_in_use.qw,
				 (cdts_in_use.qw & cdts_in_use_mask.qw) == 0,
				 1, 1000000);
	if (ret)
		goto out;

	/* Read the state again since the PE is now quiesced. */
	spin_lock(&hw->lpe_shadow_lock);
	cass_read_pte(hw, pt_priv->pte.id, &ptl_table);
	spin_unlock(&hw->lpe_shadow_lock);

	if (ptl_table.ptl_state == C_PTLTE_SOFTWARE_MANAGED) {
		ret = 0;
	} else if (ptl_table.ptl_state != C_PTLTE_DISABLED) {
		ret = -EINVAL;
	} else if (drop_count == ptl_table.drop_count) {
		/* Change the state and drop count. Only write a
		 * single u64 at offset 0x18, as to not overwrite
		 * other fields.
		 */
		u64 *state_dc = ((u64 *)&ptl_table) + 3;

		ptl_table.ptl_state = C_PTLTE_SOFTWARE_MANAGED;
		ptl_table.drop_count = 0;
		cass_write(hw, C_LPE_CFG_PTL_TABLE(pt_priv->pte.id) + 0x18,
			   state_dc, sizeof(u64));

		ret = 0;
	} else {
		ret = -EAGAIN;
	}

out:
	cdt_limits.rrq_cdts = old_rrq_cdts;
	cass_write(hw, C_LPE_CFG_CDT_LIMITS, &cdt_limits, sizeof(cdt_limits));

	mutex_unlock(&hw->pte_transition_sm_lock);

	return ret;
}
EXPORT_SYMBOL(cxi_pte_transition_sm);
