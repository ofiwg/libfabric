// SPDX-License-Identifier: GPL-2.0
/* Copyright 2018 Hewlett Packard Enterprise Development LP */

/* Create and destroy Cassini command queues */

#include <linux/hpe/cxi/cxi.h>
#include <linux/dma-mapping.h>
#include <linux/iopoll.h>
#include <linux/kernel.h>
#include <linux/types.h>

#include "cass_core.h"
#include "cass_ss1_debugfs.h"

#ifdef CONFIG_ARM64
/* Provide a workaround for avoiding writecombine on platforms where it is broken
 * and for which the Linux kernel does not already provide a workaround, such as the
 * Ampere Altra, used in the RL300.
 */
DEFINE_STATIC_KEY_FALSE(avoid_writecombine);
#endif

/* Lower bound for number of 64 byte commands. */
#define MIN_CQ_COUNT (PAGE_SIZE / C_CQ_CMD_SIZE)

void cass_cq_init(struct cass_dev *hw)
{
	static const union c_cq_cfg_cq_policy cq_policy = {
		.policy[CXI_CQ_UPDATE_ALWAYS] = {
			.shift = 0,
		},
		.policy[CXI_CQ_UPDATE_HIGH_FREQ_EMPTY] = {
			.shift = 4,
			.empty = 1,
		},
		.policy[CXI_CQ_UPDATE_LOW_FREQ_EMPTY] = {
			.shift = 8,
			.empty = 1,
		},
		.policy[CXI_CQ_UPDATE_LOW_FREQ] = {
			.shift = 8,
		},
	};
	union c_cq_cfg_tg_thresh tg_thresh = {
		.tg[0].thresh = tg_threshold[0],
		.tg[1].thresh = tg_threshold[1],
		.tg[2].thresh = tg_threshold[2],
		.tg[3].thresh = tg_threshold[3],
	};

	/* Initialize CQ policies. */
	cass_write(hw, C_CQ_CFG_CQ_POLICY, &cq_policy, sizeof(cq_policy));

	/* LPE append credits */
	cass_write(hw, C_CQ_CFG_TG_THRESH, &tg_thresh, sizeof(tg_thresh));

	/* Zero FQ allocation settings. */
	cass_clear(hw, C_CQ_CFG_FQ_RESRV(0), C_CQ_CFG_FQ_RESRV_SIZE);

#ifdef CONFIG_ARM64
	if (pci_get_device(PCI_VENDOR_ID_AMPERE, 0xe100, NULL))
		static_branch_enable(&avoid_writecombine);
#endif
}

static phys_addr_t
cq_mmio_phys_addr(const struct cass_dev *hw, int cq_id)
{
	return hw->regs_base + C_CQ_LAUNCH_TXQ_BASE +
		C_CQ_LAUNCH_PAGE_SIZE * cq_id;
}

/* Write TX CQ configuration to hardware */
static int setup_hw_tx_cq(struct cass_dev *hw, unsigned int cq_n,
			  const struct cxi_cq_alloc_opts *opts,
			  struct cxi_lni_priv *lni_priv,
			  struct cxi_cq_priv *cq,
			  struct cxi_eq_priv *eq,
			  struct cxi_cp_priv *cp_priv)
{
	size_t cmds_count = cq->cmds_len / C_CQ_CMD_SIZE;
	union c_cq_txq_tc_table tc = {
		.mem_q_eth = !!(opts->flags & CXI_CQ_TX_ETHERNET),
		.mem_q_trig = !!(opts->flags & CXI_CQ_TX_WITH_TRIG_CMDS),
	};
	union c_cq_txq_base_table txq_cfg = {
		.mem_q_eq = eq ? eq->eq.eqn : C_EQ_NONE,
		.mem_q_base = cq->cmds_dma_addr >> C_ADDR_SHIFT,
		.mem_q_max_ptr = cmds_count - 1,
		.mem_q_rgid = lni_priv->lni.rgid,
		.mem_q_stat_cnt_pool = opts->stat_cnt_pool,
		.mem_q_lcid = opts->lcid,
		.mem_q_acid = ATU_PHYS_AC,
		.mem_q_policy = opts->policy,
	};
	union c_cq_cfg_init_txq_hw_state txq_init;
	union c_cq_cfg_cp_fl_table flow_cfg;
	void __iomem *csr;
	int rc;

	/* Transmit CQ needs to be configured with the same PFQ and TC
	 * values the corresponding communication profile is using.
	 */
	cass_read(hw, C_CQ_CFG_CP_FL_TABLE(cp_priv->cass_cp->id),
		  &flow_cfg, sizeof(flow_cfg));

	tc.mem_q_tc = flow_cfg.tc;
	txq_cfg.mem_q_pfq = flow_cfg.pfq;

	if (opts->flags & CXI_CQ_TX_WITH_TRIG_CMDS)
		txq_cfg.mem_q_pfq = cxi_rgroup_tle_pool_id(lni_priv->rgroup);

	spin_lock(&hw->cq_shadow_lock);
	cass_tx_cq_init(hw, &cq->cass_cq, cq_n, &tc, &txq_cfg);
	spin_unlock(&hw->cq_shadow_lock);

	mutex_lock(&hw->cq_init_lock);
	cass_read(hw, C_CQ_CFG_INIT_TXQ_HW_STATE, &txq_init,
		  sizeof(txq_init));

	cxidev_WARN_ONCE(&hw->cdev, txq_init.pending == 1,
			 "TXQ init pending CSR should not be set");
	if (txq_init.pending == 1) {
		mutex_unlock(&hw->cq_init_lock);
		return -EIO;
	}

	txq_init.cq_handle = cq_n;

	cass_write(hw, C_CQ_CFG_INIT_TXQ_HW_STATE, &txq_init,
		   sizeof(txq_init));

	csr = cass_csr(hw, C_CQ_CFG_INIT_TXQ_HW_STATE);
	rc = readq_poll_timeout(csr, txq_init.qw, txq_init.pending == 0,
				1, 1000);

	cxidev_WARN_ONCE(&hw->cdev, rc, "Timeout waiting for TXQ init");
	if (rc) {
		mutex_unlock(&hw->cq_init_lock);
		return -EIO;
	}
	mutex_unlock(&hw->cq_init_lock);

	cass_tx_cq_enable(hw, cq_n);

	return 0;
}

/* Write TGT CQ configuration to hardware */
#define MEM_Q_TC_REG_SIZE BIT(3)
static int setup_hw_tgt_cq(struct cass_dev *hw, unsigned int cq_n,
			   const struct cxi_cq_alloc_opts *opts,
			   struct cxi_lni_priv *lni_priv,
			   struct cxi_cq_priv *cq,
			   struct cxi_eq_priv *eq)
{
	size_t cmds_count = cq->cmds_len / C_CQ_CMD_SIZE;
	// TODO: Handle TGT TC
	const union c_cq_tgq_table tgt_cfg = {
		.mem_q_eq = eq ? eq->eq.eqn : C_EQ_NONE,
		.mem_q_base = cq->cmds_dma_addr >> C_ADDR_SHIFT,
		.mem_q_max_ptr = cmds_count - 1,
		.mem_q_rgid = lni_priv->lni.rgid,
		.mem_q_stat_cnt_pool = opts->stat_cnt_pool,
		.mem_q_tc_reg = cq_n % MEM_Q_TC_REG_SIZE,
		.mem_q_acid = ATU_PHYS_AC,
		.mem_q_policy = opts->policy,
	};
	union c_cq_cfg_init_tgq_hw_state tgq_init;
	union c_cq_cfg_thresh_map thresh_map = {
		.thresh_id = opts->lpe_cdt_thresh_id,
	};
	void __iomem *csr;
	int rc;

	spin_lock(&hw->cq_shadow_lock);
	cass_tgt_cq_init(hw, &cq->cass_cq, cq_n, &tgt_cfg);
	spin_unlock(&hw->cq_shadow_lock);

	mutex_lock(&hw->cq_init_lock);
	cass_read(hw, C_CQ_CFG_INIT_TGQ_HW_STATE, &tgq_init,
		  sizeof(tgq_init));

	cxidev_WARN_ONCE(&hw->cdev, tgq_init.pending == 1,
			 "TGQ init pending CSR should not be set");
	if (tgq_init.pending == 1) {
		mutex_unlock(&hw->cq_init_lock);
		return -EIO;
	}

	cass_write(hw, C_CQ_CFG_THRESH_MAP(cq_n),
		   &thresh_map, sizeof(thresh_map));

	tgq_init.cq_handle = cq_n;
	cass_write(hw, C_CQ_CFG_INIT_TGQ_HW_STATE, &tgq_init,
		   sizeof(tgq_init));

	/* TODO: Check delay and timeouts are correct for real NIC. */
	csr = cass_csr(hw, C_CQ_CFG_INIT_TGQ_HW_STATE);
	rc = readq_poll_timeout(csr, tgq_init.qw, tgq_init.pending == 0,
				1, 1000);

	cxidev_WARN_ONCE(&hw->cdev, rc, "Timeout waiting for TGQ init");
	if (rc) {
		mutex_unlock(&hw->cq_init_lock);
		return -EIO;
	}
	mutex_unlock(&hw->cq_init_lock);

	cass_tgt_cq_enable(hw, cq_n);

	return 0;
}

/* Return an available CQ with an ID */
static struct cxi_cq_priv *pf_get_cq_id(struct cxi_lni_priv *lni_priv,
					const struct cxi_cq_alloc_opts *opts)
{
	struct cxi_dev *dev = lni_priv->dev;
	struct cass_dev *hw =
		container_of(dev, struct cass_dev, cdev);
	struct cxi_cq_priv *cq;
	enum cxi_resource_type rsrc_type;
	int rc;

	cq = kzalloc(sizeof(struct cxi_cq_priv), GFP_KERNEL);
	if (cq == NULL)
		return ERR_PTR(-ENOMEM);

	/* Check the associated service to see if this CQ can be allocated */
	if (opts->flags & CXI_CQ_IS_TX)
		rsrc_type = CXI_RESOURCE_TXQ;
	else
		rsrc_type = CXI_RESOURCE_TGQ;

	rc = cxi_rgroup_alloc_resource(lni_priv->rgroup, rsrc_type);
	if (rc) {
		/* Attempt to cleanup all CQs on this LNI to return service
		 * credits and then reattempt to allocate resource.
		 */
		finalize_cq_cleanups(lni_priv, false);

		rc = cxi_rgroup_alloc_resource(lni_priv->rgroup, rsrc_type);
		if (rc)
			goto cq_free;
	}

	if (opts->flags & CXI_CQ_IS_TX)
		rc = ida_simple_get(&hw->cq_table, 0,
				    C_NUM_TRANSMIT_CQS, GFP_KERNEL);
	else
		rc = ida_simple_get(&hw->cq_table, C_NUM_TRANSMIT_CQS,
				    C_NUM_TRANSMIT_CQS + C_NUM_TARGET_CQS,
				    GFP_KERNEL);

	if (rc < 0) {
		/* Out of CQs. Accelerate possible CQ cleanup. */
		lni_cleanups(hw, false);

		if (opts->flags & CXI_CQ_IS_TX)
			rc = ida_simple_get(&hw->cq_table, 0,
					    C_NUM_TRANSMIT_CQS, GFP_KERNEL);
		else
			rc = ida_simple_get(&hw->cq_table, C_NUM_TRANSMIT_CQS,
					    C_NUM_TRANSMIT_CQS +
					    C_NUM_TARGET_CQS,
					    GFP_KERNEL);
	}

	if (rc < 0)
		goto dec_rsrc_use;

	cq->cass_cq.idx = rc;

	return cq;

dec_rsrc_use:
	cxi_rgroup_free_resource(lni_priv->rgroup, rsrc_type);

cq_free:
	kfree(cq);

	return ERR_PTR(rc);
}

static void put_cq_id(struct cxi_cq_priv *cq)
{
	struct cxi_lni_priv *lni_priv = cq->lni_priv;
	struct cxi_dev *dev = lni_priv->dev;
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);

	ida_simple_remove(&hw->cq_table, cq->cass_cq.idx);
	if (cq->flags & CXI_CQ_IS_TX)
		cxi_rgroup_free_resource(lni_priv->rgroup, CXI_RESOURCE_TXQ);
	else
		cxi_rgroup_free_resource(lni_priv->rgroup, CXI_RESOURCE_TGQ);
	kfree(cq);
}

/**
 * cxi_cq_alloc() - Allocate a new command queue
 *
 * The new CQ is attached to attached to the NI.
 *
 * @lni: LNI to associate with the command queue.
 * @evtq: Event Queue to associate with the command queue, for error
 *        reporting. May be NULL.
 * @opts: Options affecting the CQ.
 * @numa_node: NUMA node ID CQ memory should be allocated from.
 *
 * @return: the command queue or an error pointer
 */
struct cxi_cq *cxi_cq_alloc(struct cxi_lni *lni, struct cxi_eq *evtq,
			    const struct cxi_cq_alloc_opts *opts,
			    int numa_node)
{
	struct cxi_lni_priv *lni_priv =
		container_of(lni, struct cxi_lni_priv, lni);
	struct cxi_dev *cdev = lni_priv->dev;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	struct cxi_eq_priv *eq;
	struct cxi_cq_priv *cq;
	int cq_idx;
	int cq_n;
	int rc;
	size_t cmds_count;
	size_t cmds_len;
	size_t cmds_order;
	struct page *cmds_pages;
	struct cxi_cp_priv *cp_priv = NULL;
	bool is_user = opts->flags & CXI_CQ_USER;

	if (opts->policy < 0 || opts->policy > CXI_CQ_UPDATE_LOW_FREQ)
		return ERR_PTR(-EINVAL);

	if (!(opts->flags & CXI_CQ_IS_TX) && opts->lpe_cdt_thresh_id > 3)
		return ERR_PTR(-EINVAL);

	/*
	 * Use provided commands count to calculate commands length. Then, use
	 * commands length to recalculate commands count to consume entire
	 * buffer space.
	 */
	cmds_count = max_t(size_t, MIN_CQ_COUNT, opts->count);
	cmds_order = get_order(cmds_count * C_CQ_CMD_SIZE);
	cmds_len = (1 << cmds_order) * PAGE_SIZE;
	cmds_count = cmds_len / C_CQ_CMD_SIZE;

	if (cmds_count > CXI_MAX_CQ_COUNT)
		return ERR_PTR(-E2BIG);

	/* At least one communication profile is needed for a transmit
	 * CQ.
	 */
	if (opts->flags & CXI_CQ_IS_TX) {
		/* For triggered CQs, only LCID zero can be used. */
		if ((opts->flags & CXI_CQ_TX_WITH_TRIG_CMDS) && opts->lcid != 0)
			return ERR_PTR(-EINVAL);

		cp_priv = cass_cp_find(hw, lni_priv->lni.rgid, opts->lcid);
		if (!cp_priv)
			return ERR_PTR(-EINVAL);
	}

	/* Only privileged client can use the C_CMD_ETHERNET_TX command. */
	if ((opts->flags & CXI_CQ_TX_ETHERNET) && !capable(CAP_NET_RAW))
		return ERR_PTR(-EPERM);

	/* CQ needs contiguous memory region. */
	cmds_pages = alloc_pages_node(numa_node, GFP_KERNEL | __GFP_ZERO,
				      cmds_order);
	if (cmds_pages == NULL)
		return ERR_PTR(-ENOMEM);

	cq = pf_get_cq_id(lni_priv, opts);
	if (IS_ERR(cq)) {
		rc = PTR_ERR(cq);
		goto cmds_free;
	}

	cq_idx = cq->cass_cq.idx;

	if (evtq)
		eq = container_of(evtq, struct cxi_eq_priv, eq);
	else
		eq = NULL;

	cq->lni_priv = lni_priv;
	cq->eq = eq;
	cq->flags = opts->flags;
	cq->cmds_len = cmds_len;
	cq->cmds_order = cmds_order;
	cq->cmds_pages = cmds_pages;
	cq->cmds = page_address(cmds_pages);
	cq->cmds_dma_addr = dma_map_page(&hw->cdev.pdev->dev, cmds_pages, 0,
					 cq->cmds_len, DMA_BIDIRECTIONAL);

	if (dma_mapping_error(&hw->cdev.pdev->dev, cq->cmds_dma_addr)) {
		rc = -ENOMEM;
		goto put_id;
	}

	if (!is_user) {
		phys_addr_t mmio_phys = cq_mmio_phys_addr(hw, cq_idx);
#ifdef CONFIG_ARM64
		if (static_branch_unlikely(&avoid_writecombine))
			cq->cq_mmio = ioremap(mmio_phys, PAGE_SIZE);
		else
#endif
		cq->cq_mmio = ioremap_wc(mmio_phys, PAGE_SIZE);
		if (!cq->cq_mmio) {
			cxidev_warn_once(cdev, "ioremap_wc failed\n");
			rc = -EFAULT;
			goto cq_release;
		}
	}

	/* Initialize client interface structure */
	cxi_cq_init(&cq->cass_cq, cq->cmds, cmds_count,
		    (__force void *)cq->cq_mmio, cq_idx);

	cq_n = cxi_cq_get_cqn(&cq->cass_cq);
	if (opts->flags & CXI_CQ_IS_TX)
		rc = setup_hw_tx_cq(hw, cq_n, opts, lni_priv, cq, eq, cp_priv);
	else
		rc = setup_hw_tgt_cq(hw, cq_n, opts, lni_priv, cq, eq);

	if (rc)
		goto cq_unmap;

	if (cq->flags & CXI_CQ_IS_TX)
		atomic_inc(&hw->stats.txq);
	else
		atomic_inc(&hw->stats.tgq);

	cq_debugfs_create(cq_idx, cq, hw, lni_priv);

	spin_lock(&lni_priv->res_lock);
	list_add_tail(&cq->list, &lni_priv->cq_list);
	spin_unlock(&lni_priv->res_lock);

	refcount_inc(&lni_priv->refcount);
	if (eq)
		refcount_inc(&eq->refcount);

	return &cq->cass_cq;

cq_unmap:
	if (!is_user)
		iounmap(cq->cq_mmio);

cq_release:
	dma_unmap_page(&hw->cdev.pdev->dev, cq->cmds_dma_addr,
		       cq->cmds_len, DMA_BIDIRECTIONAL);

put_id:
	put_cq_id(cq);

cmds_free:
	__free_pages(cmds_pages, cmds_order);

	return ERR_PTR(rc);
}
EXPORT_SYMBOL(cxi_cq_alloc);

/* Test whether a transmit CQ is done working */
static bool is_txq_done(struct cass_dev *hw, int id)
{
	union c_cq_txq_rdptr_table rd_ptr;
	union c_cq_txq_preptr_table pre_ptr;
	union c_cq_txq_ack_ctr ack_ctr;

	cass_read(hw, C_CQ_TXQ_ACK_CTR(id), &ack_ctr, sizeof(ack_ctr));
	if (ack_ctr.mem_q_ack_ctr != 0)
		return false;

	cass_read(hw, C_CQ_TXQ_RDPTR_TABLE(id), &rd_ptr, sizeof(rd_ptr));
	cass_read(hw, C_CQ_TXQ_PREPTR_TABLE(id), &pre_ptr, sizeof(pre_ptr));

	if (rd_ptr.mem_q_rd_ptr != pre_ptr.mem_q_pre_ptr)
		return false;

	return true;
}

/* Test whether a target CQ is done working */
static bool is_tgq_done(struct cass_dev *hw, int id)
{
	union c_cq_tgq_rdptr_table rd_ptr;
	union c_cq_tgq_preptr_table pre_ptr;
	union c_cq_tgq_ack_ctr ack_ctr;

	cass_read(hw, C_CQ_TGQ_ACK_CTR(id), &ack_ctr, sizeof(ack_ctr));
	if (ack_ctr.mem_q_ack_ctr != 0)
		return false;

	cass_read(hw, C_CQ_TGQ_RDPTR_TABLE(id), &rd_ptr, sizeof(rd_ptr));
	cass_read(hw, C_CQ_TGQ_PREPTR_TABLE(id), &pre_ptr, sizeof(pre_ptr));
	if (rd_ptr.mem_q_rd_ptr != pre_ptr.mem_q_pre_ptr)
		return false;

	return true;
}

/**
 * cxi_cq_free() - Destroy a command queue
 *
 * @cmdq: the command queue to release
 */
void cxi_cq_free(struct cxi_cq *cmdq)
{
	struct cxi_cq_priv *cq = container_of(cmdq, struct cxi_cq_priv, cass_cq);
	struct cxi_lni_priv *lni_priv = cq->lni_priv;
	struct cxi_eq_priv *eq = cq->eq;
	struct cxi_dev *cdev = lni_priv->dev;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	unsigned int cq_n = cxi_cq_get_cqn(cmdq);
	unsigned int mem_q_rd_ptr;

	/* CQ invalidation requires disabling and setting the drain bit for the
	 * CQ. This is done in cass_tx_cq_disable() and cass_tgt_cq_disable().
	 *
	 * By the time the read pointer is stable, the buffer is not
	 * accessed anymore and can be freed.
	 */
	if (cq->flags & CXI_CQ_IS_TX) {
		union c_cq_txq_rdptr_table rd_ptr;

		cass_tx_cq_disable(hw, cq_n);

		cass_read(hw, C_CQ_TXQ_RDPTR_TABLE(cq_n),
			  &rd_ptr, sizeof(rd_ptr));
		mem_q_rd_ptr = rd_ptr.mem_q_rd_ptr;
		while (true) {
			cass_read(hw, C_CQ_TXQ_RDPTR_TABLE(cq_n),
				  &rd_ptr, sizeof(rd_ptr));
			if (mem_q_rd_ptr == rd_ptr.mem_q_rd_ptr)
				break;
			mem_q_rd_ptr = rd_ptr.mem_q_rd_ptr;
			udelay(1);
		}
	} else {
		union c_cq_tgq_rdptr_table rd_ptr;

		cass_tgt_cq_disable(hw, cq_n);

		cass_read(hw, C_CQ_TGQ_RDPTR_TABLE(cq_n),
			  &rd_ptr, sizeof(rd_ptr));
		mem_q_rd_ptr = rd_ptr.mem_q_rd_ptr;
		while (true) {
			cass_read(hw, C_CQ_TGQ_RDPTR_TABLE(cq_n),
				  &rd_ptr, sizeof(rd_ptr));
			if (mem_q_rd_ptr == rd_ptr.mem_q_rd_ptr)
				break;
			mem_q_rd_ptr = rd_ptr.mem_q_rd_ptr;
			udelay(1);
		}
	}

	if (!(cq->flags & CXI_CQ_USER))
		iounmap(cq->cq_mmio);

	dma_unmap_page(&hw->cdev.pdev->dev, cq->cmds_dma_addr,
		       cq->cmds_len, DMA_BIDIRECTIONAL);
	__free_pages(cq->cmds_pages, cq->cmds_order);

	if (eq)
		refcount_dec(&eq->refcount);

	spin_lock(&lni_priv->res_lock);
	list_del(&cq->list);
	list_add_tail(&cq->list, &lni_priv->cq_cleanups_list);
	spin_unlock(&lni_priv->res_lock);
}
EXPORT_SYMBOL(cxi_cq_free);

/* Go through the list of CQ waiting for cleanup. If the CQ is quiet,
 * it can be freed, otherwise it stays.
 *
 * If force is true, the CQ will be release unconditionally.
 */
void finalize_cq_cleanups(struct cxi_lni_priv *lni, bool force)
{
	struct cxi_dev *cdev = lni->dev;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	struct cxi_cq_priv *cq;
	struct cxi_cq_priv *tmp;
	struct list_head free_list;

	INIT_LIST_HEAD(&free_list);

	spin_lock(&lni->res_lock);

	list_for_each_entry_safe(cq, tmp, &lni->cq_cleanups_list, list) {
		unsigned int cq_n = cxi_cq_get_cqn(&cq->cass_cq);

		if (!force) {
			/* Ensure that in-flight commands are done. */
			if (cq->flags & CXI_CQ_IS_TX) {
				if (!is_txq_done(hw, cq_n))
					continue;
			} else {
				if (!is_tgq_done(hw, cq_n))
					continue;
			}
		}

		list_del(&cq->list);
		list_add(&cq->list, &free_list);
	}

	spin_unlock(&lni->res_lock);

	while ((cq = list_first_entry_or_null(&free_list,
					      struct cxi_cq_priv, list))) {
		list_del(&cq->list);

		debugfs_remove(cq->lni_dir);
		debugfs_remove_recursive(cq->debug_dir);

		refcount_dec(&lni->refcount);

		if (cq->flags & CXI_CQ_IS_TX)
			atomic_dec(&hw->stats.txq);
		else
			atomic_dec(&hw->stats.tgq);

		put_cq_id(cq);
	}
}

/**
 * cxi_cq_user_info() - Get the information to mmap a cq to userspace
 *
 * @cmdq: the command queue
 * @cmds_size: on return, will hold the size of the queue in 64-bytes blocks
 * @cmds_pages: on return, will hold the pages to mmap
 * @wp_addr: on return, will hold the CSR to control the CQ
 * @wp_addr_size: on return, size of the CSR
 *
 * @return: 0, or -EINVAL is the CQ isn't for userspace
 */
int cxi_cq_user_info(struct cxi_cq *cmdq, size_t *cmds_size,
		     struct page **cmds_pages, phys_addr_t *wp_addr,
		     size_t *wp_addr_size)
{
	struct cxi_cq_priv *cq = container_of(cmdq, struct cxi_cq_priv, cass_cq);
	struct cxi_lni_priv *lni_priv = cq->lni_priv;
	struct cxi_dev *cdev = lni_priv->dev;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);

	if (!(cq->flags & CXI_CQ_USER))
		return -EINVAL;

	/* userspace will need to know what pages to mmap. */
	*cmds_size = cq->cmds_len;
	*cmds_pages = cq->cmds_pages;

	*wp_addr = cq_mmio_phys_addr(hw, cq->cass_cq.idx);
	*wp_addr_size = PAGE_SIZE;

	return 0;
}
EXPORT_SYMBOL(cxi_cq_user_info);

/**
 * cxi_cq_ack_counter() - Get the current ack counter for a CQ.
 *
 * @cq: CQ to get current ack counter for.
 *
 * Return: Current ack counter value.
 */
unsigned int cxi_cq_ack_counter(struct cxi_cq *cq)
{
	struct cxi_cq_priv *cq_priv =
		container_of(cq, struct cxi_cq_priv, cass_cq);
	struct cass_dev *hw =
		container_of(cq_priv->lni_priv->dev, struct cass_dev, cdev);
	unsigned int cq_n = cxi_cq_get_cqn(cq);
	union c_cq_txq_ack_ctr txq_ack_ctr;
	union c_cq_tgq_ack_ctr tgq_ack_ctr;
	unsigned int ack_ctr;

	if (cq_priv->flags & CXI_CQ_IS_TX) {
		cass_read(hw, C_CQ_TXQ_ACK_CTR(cq_n), &txq_ack_ctr,
			  sizeof(txq_ack_ctr));
		ack_ctr = txq_ack_ctr.mem_q_ack_ctr;
	} else {
		cass_read(hw, C_CQ_TGQ_ACK_CTR(cq_n), &tgq_ack_ctr,
			  sizeof(tgq_ack_ctr));
		ack_ctr = tgq_ack_ctr.mem_q_ack_ctr;
	}
	return ack_ctr;
}
EXPORT_SYMBOL(cxi_cq_ack_counter);
