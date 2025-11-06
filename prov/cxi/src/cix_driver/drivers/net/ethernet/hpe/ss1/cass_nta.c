// SPDX-License-Identifier: GPL-2.0
/* Copyright 2019 Hewlett Packard Enterprise Development LP */

/*
 * Address Translation Unit (ATU) management
 * Nic Translation Agent (NTA) mode
 */

#include <linux/interrupt.h>
#include <linux/pci.h>
#include <linux/types.h>
#include <linux/hugetlb.h>
#include <linux/iova.h>
#include <linux/bvec.h>
#include <linux/iopoll.h>
#include <linux/workqueue.h>
#include <linux/mmu_notifier.h>

#include "cass_core.h"

/* Default size of the root page table for a two level table.
 * The root page table size is 2 ^ (PAGE_SHIFT + ac_size_dbl) (currently
 * 16 KB) and covers 2 ^ ac_size_dbl GB.
 */
static int ac_size_dbl = 5;
module_param(ac_size_dbl, int, 0644);
MODULE_PARM_DESC(ac_size_dbl, "Default two level AC size");

/* Default size of the root page table for a single level table.
 * The root page table size is 2 ^ (PAGE_SHIFT + ac_size_sgl) (currently
 * 8 MB) and covers 2 ^ ((PAGE_SHIFT * 2) + ac_size_sgl) bytes.
 */
static int ac_size_sgl = 8;
module_param(ac_size_sgl, int, 0644);
MODULE_PARM_DESC(ac_size_sgl, "Default one level AC size");

static ulong min_iova_offset = 1UL * 1024 * 1024 * 1024;
module_param(min_iova_offset, ulong, 0644);
MODULE_PARM_DESC(min_iova_offset, "IOVA offset if VA range straddles AC");

static int atucq_timeout_us = 100000;
module_param(atucq_timeout_us, int, 0644);
MODULE_PARM_DESC(atucq_timeout_us, "ATUCQ timeout in microseconds");

static int decouple_timeout_us = 150;
module_param(decouple_timeout_us, int, 0644);
MODULE_PARM_DESC(decouple_timeout_us,
		 "ATUCQ timeout in microseconds when decoupling is enabled");

bool odp_sw_decouple = true;
module_param(odp_sw_decouple, bool, 0644);
MODULE_PARM_DESC(odp_sw_decouple, "Enable software ODP decoupling");

static int pri_shift_set(const char *val, const struct kernel_param *kp)
{
	int n;
	int ret;

	ret = kstrtoint(val, 10, &n);
	if (ret || n < 12 || n > 30)
		return -EINVAL;

	return param_set_int(val, kp);
}

static const struct kernel_param_ops pri_shift_ops = {
	.set = pri_shift_set,
	.get = param_get_int,
};

static int pri_shift = 16;
module_param_cb(pri_shift, &pri_shift_ops, &pri_shift, 0644);
MODULE_PARM_DESC(pri_shift, "ODP default number of pages to fault");

static irqreturn_t cass_pri_int_cb(int irq, void *context);
static void cass_pri_worker(struct work_struct *work);
static void cass_prb_worker(struct work_struct *work);
static void cass_odpq_write(struct cass_dev *hw, int index,
			    enum c_odp_status status);

static int min_root_entries(struct cass_ac *cac)
{
	int default_ac_size = (cac->ptg_mode == C_ATU_PTG_MODE_SGL) ?
			ac_size_sgl : ac_size_dbl;

	return BIT(PAGE_SHIFT + default_ac_size) / ATU_PTE_ENTRY_SIZE;
}

static void cass_nta_cmdproc_write(struct cass_dev *hw, unsigned int wr_index)
{
	union c_atu_cfg_cmdproc atu_ptrs = {
		.wr_ptr = wr_index
	};

	cass_write(hw, C_ATU_CFG_CMDPROC, &atu_ptrs, sizeof(atu_ptrs));
}

static void cass_nta_cq_write(struct cass_dev *hw, unsigned int index,
			     const union c_atu_cfg_cq_table *entry)
{
	cass_write(hw, C_ATU_CFG_CQ_TABLE(index), entry, sizeof(*entry));
}

void cass_nta_cq_fini(struct cass_dev *hw)
{
	free_irq(pci_irq_vector(hw->cdev.pdev, hw->atu_cq_vec), hw);
}

static irqreturn_t cass_nta_cq_cb(int irq, void *context)
{
	/* Not expected to ever be called */
	BUG();

	return IRQ_HANDLED;
}

/**
 * cass_nta_cq_init() - NTA command queue initialization
 *
 * @hw: Cassini device
 */
int cass_nta_cq_init(struct cass_dev *hw)
{
	struct cass_atu_cq *cq = &hw->atu_cq;
	static const union c_atu_cfg_wb_data wb_data = {
		.data = WAIT_RSP_DATA
	};
	union c_atu_cfg_cmds cfg_cmds = {
		.writeback_acid = ATU_NTA_DEF_PI_ACXT
	};
	int rc;

	cq->rsp_dma_addr = dma_map_single(&hw->cdev.pdev->dev, &cq->wait_rsp_data,
					  sizeof(cq->wait_rsp_data), DMA_FROM_DEVICE);
	if (dma_mapping_error(&hw->cdev.pdev->dev, cq->rsp_dma_addr))
		return -ENOMEM;

	cq->cmp_wait_rsp_addr = (cq->rsp_dma_addr + offsetof(struct wait_rsp_data, cmp)) >> 3;
	cq->ib_wait_rsp_addr = (cq->rsp_dma_addr + offsetof(struct wait_rsp_data, ib)) >> 3;

	mutex_init(&cq->atu_cq_mutex);
	mutex_init(&cq->atu_ib_mutex);
	cass_write(hw, C_ATU_CFG_WB_DATA, &wb_data, sizeof(wb_data));

	scnprintf(cq->cmpl_wait_int_name, sizeof(cq->cmpl_wait_int_name),
		  "%s_atu_cq", hw->cdev.name);
	rc = request_irq(pci_irq_vector(hw->cdev.pdev, hw->atu_cq_vec),
			 cass_nta_cq_cb, 0, cq->cmpl_wait_int_name, hw);
	if (rc) {
		dev_err(&hw->cdev.pdev->dev, "Failed to request IRQ.\n");
		dma_unmap_single(&hw->cdev.pdev->dev, cq->rsp_dma_addr,
				 sizeof(cq->wait_rsp_data), DMA_FROM_DEVICE);
		return rc;
	}

	disable_irq(pci_irq_vector(hw->cdev.pdev, hw->atu_cq_vec));

	cfg_cmds.ib_wait_int_idx = hw->atu_cq_vec;
	cfg_cmds.cmpl_wait_int_idx = hw->atu_cq_vec;
	cass_write(hw, C_ATU_CFG_CMDS, &cfg_cmds, sizeof(cfg_cmds));

	return 0;
}

void cass_nta_pri_fini(struct cass_dev *hw)
{
	if (cass_version(hw, CASSINI_1_0))
		return;

	cass_clear(hw, C_ATU_CFG_ODP, C_ATU_CFG_ODP_SIZE);
	cass_clear(hw, C_ATU_CFG_PRI, C_ATU_CFG_PRI_SIZE);
	destroy_workqueue(hw->prb_wq);
	destroy_workqueue(hw->pri_wq);
	dma_unmap_single(&hw->cdev.pdev->dev, hw->prt_dma_addr,
			 sizeof(struct c_page_request_entry), DMA_FROM_DEVICE);
	kfree(hw->page_request_table);
	free_irq(pci_irq_vector(hw->cdev.pdev, hw->atu_pri_vec), hw);
	dma_unmap_single(&hw->cdev.pdev->dev, hw->atu_cq.rsp_dma_addr,
			 sizeof(struct wait_rsp_data), DMA_FROM_DEVICE);
}

/**
 * cass_nta_pri_init() - Initialize the NTA Page Request interface
 *
 * @hw: Cassini device
 */
int cass_nta_pri_init(struct cass_dev *hw)
{
	int rc;
	union c_atu_cfg_pri cpa;
	union c_atu_cfg_odp codp = {
		.acid = ATU_PHYS_AC,
		.int_enable = 1,
		.odp_en = 1
	};

	if (cass_version(hw, CASSINI_1_0))
		return 0;

	hw->page_request_table = kcalloc(C_ATU_PRB_ENTRIES,
					 sizeof(struct c_page_request_entry),
					 GFP_KERNEL);
	if (!hw->page_request_table)
		return -ENOMEM;

	hw->prt_dma_addr = dma_map_single(&hw->cdev.pdev->dev, hw->page_request_table,
					  sizeof(struct c_page_request_entry),
					  DMA_FROM_DEVICE);
	if (dma_mapping_error(&hw->cdev.pdev->dev, hw->prt_dma_addr)) {
		rc = -ENOMEM;
		goto free_table;
	}

	hw->pri_rd_ptr = 0;

	scnprintf(hw->odp_pri_int_name, sizeof(hw->odp_pri_int_name),
		  "%s_odp_pri", hw->cdev.name);
	rc = request_irq(pci_irq_vector(hw->cdev.pdev, hw->atu_pri_vec),
			 cass_pri_int_cb, 0, hw->odp_pri_int_name, hw);
	if (rc) {
		pr_err("Failed to request IRQ.\n");
		goto free_mapping;
	}

	/* limit to a single threaded worker */
	hw->pri_wq = alloc_workqueue("pri_wq", WQ_UNBOUND, 1);
	if (!hw->pri_wq) {
		rc = -ENOMEM;
		pr_err("Failed to allocate PRI workqueue\n");
		goto alloc_wq_failed;
	}

	hw->prb_wq = alloc_workqueue("prb_wq", WQ_UNBOUND, 1);
	if (!hw->prb_wq) {
		rc = -ENOMEM;
		pr_err("Failed to allocate PRB workqueue\n");
		goto alloc_prb_wq_failed;
	}

	cpa.base_addr = hw->prt_dma_addr >> C_ADDR_SHIFT;
	cass_write(hw, C_ATU_CFG_PRI, &cpa, sizeof(cpa));

	codp.int_idx = hw->atu_pri_vec;
	cass_write(hw, C_ATU_CFG_ODP, &codp, sizeof(codp));

	INIT_WORK(&hw->pri_work, cass_pri_worker);
	INIT_WORK(&hw->prb_work, cass_prb_worker);

	return 0;

alloc_prb_wq_failed:
	destroy_workqueue(hw->pri_wq);
alloc_wq_failed:
	free_irq(pci_irq_vector(hw->cdev.pdev, hw->atu_pri_vec), hw);
free_mapping:
	dma_unmap_single(&hw->cdev.pdev->dev, hw->prt_dma_addr,
			 sizeof(struct c_page_request_entry), DMA_FROM_DEVICE);
free_table:
	kfree(hw->page_request_table);

	return rc;
}

int cass_sts_idle(struct sts_idle *sidle)
{
	union c_atu_sts_cmdproc sts;
	struct cass_dev *hw = sidle->hw;

	cass_read(hw, C_ATU_STS_CMDPROC, &sts, sizeof(sts));

	return sidle->ib_wait ? sts.inbound_wait_idle : sts.atucq_idle;
}

static int cass_poll_sts_cmdproc_idle(struct cass_dev *hw, bool ib_wait)
{
	int ret;
	int idle;
	int timeout_us = ib_wait ? ATU_STS_IBW_IDLE_TIMEOUT :
					ATU_STS_ATUCQ_IDLE_TIMEOUT;
	struct sts_idle sidle = {
		.hw = hw,
		.ib_wait = ib_wait
	};

	ret = readx_poll_timeout(cass_sts_idle, &sidle, idle, (idle),
				 10, timeout_us);
	if (ret)
		pr_crit("Timed out waiting for %s idle\n",
			ib_wait ? "IB wait" : "ATUCQ");

	return ret;
}

#define err_dump_csr64(name) do {		\
	u64 val;				\
	cass_read(hw, name, &val, sizeof(val));	\
	cxidev_warn_ratelimited(&hw->cdev, #name " %016llx\n", val);	\
	} while (0)

static bool cass_cmdproc_debug(struct cass_dev *hw, bool inbound_wait)
{
	union c_mb_sts_rev rev;

	/* check if HW is still alive */
	cass_read(hw, C_MB_STS_REV, &rev, sizeof(rev));
	if (rev.vendor_id != hw->rev.vendor_id) {
		pr_err("HW is in a bad state\n");
		return true;
	}

	cxidev_warn_ratelimited(&hw->cdev, "%s wait timeout\n",
				inbound_wait ? "Inbound" : "Completion");
	err_dump_csr64(C_ATU_STS_CMDPROC);
	err_dump_csr64(C_ATU_STS_ODP_DECOUPLE);
	err_dump_csr64(C_ATU_STS_AT_EPOCH);
	err_dump_csr64(C_ATU_STS_IB_EPOCH);

	return false;
}

static void cass_dcpl_debug(struct cass_dev *hw)
{
	u64 val1;
	u64 val2;

	cass_read(hw, C_ATU_STS_CMDPROC, &val1, sizeof(val1));
	cass_read(hw, C_ATU_STS_AT_EPOCH, &val2, sizeof(val2));
	cxidev_warn_ratelimited(&hw->cdev,
				"Completion wait timeout after decouple. atu_sts_cmdproc:0x%llx atu_sts_at_epoch:0x%llx rsp_data:0x%llx\n",
				val1, val2, hw->atu_cq.wait_rsp_data.cmp);
}

static void cass_completion_wait(struct cass_dev *hw, unsigned int wr_index)
{
	int rc;
	int ret;
	u64 cw_rsp_read_data;
	bool hw_failure;
	int decoupled = 0;
	union c_atu_cfg_cq_table entry = {};
	struct cass_atu_cq *cq = &hw->atu_cq;
	int timeout = odp_sw_decouple ? decouple_timeout_us : atucq_timeout_us;

	atomic_inc(&hw->dcpl_comp_wait);

	entry.cmd = C_ATU_COMPLETION_WAIT;
	entry.comp_wait_rsp_en = 1;
	entry.comp_wait_rsp_addr = cq->cmp_wait_rsp_addr;

	cq->wait_rsp_data.cmp = 0;

	cass_nta_cq_write(hw, ATUCQ_ENTRY(wr_index), &entry);

	/* WR_PTR should point at the entry that immediately follows
	 * the last outstanding command.
	 */
	wr_index++;

	/* Writes to C_ATU_CFG_CMDPROC are ignored if ATUCQ is not idle. */
	cass_poll_sts_cmdproc_idle(hw, false);

	/* start the processing of the command queue */
	cass_nta_cmdproc_write(hw, wr_index);
	cass_flush_pci(hw);

	do {
		ret = readq_poll_timeout_atomic(&cq->wait_rsp_data.cmp,
						cw_rsp_read_data,
						(cw_rsp_read_data == WAIT_RSP_DATA),
						1, timeout);
		if (ret) {
			if (odp_sw_decouple) {
				if (decoupled) {
					cass_dcpl_debug(hw);
					break;
				}

				rc = cass_odp_decouple(hw);
				if (rc) {
					cxidev_warn_ratelimited(&hw->cdev,
						"Decoupling failed\n");
					break;
				}

				decoupled++;
				timeout = atucq_timeout_us;
			} else {
				hw_failure = cass_cmdproc_debug(hw, false);
				if (hw_failure)
					return;
			}
		}
	} while (ret);
}

/**
 * cass_invalidate_range() - Invalidate a range of IOVAs
 *
 * Invalidate a range of IOVAs by powers of two. Does not return until
 * the commands have been executed - the completion wait is finished.
 * inval_addr is modified with the length when invalidating sizes larger
 * than a page. Treat the inval_size as bit 11 of the address.
 *
 * @cac:    Address Context struct
 * @iova:   IO virtual address
 * @length: Length of address range to invalidate
 */
void cass_invalidate_range(const struct cass_ac *cac, u64 iova,
			   size_t length)
{
	u64 len;
	u64 align;
	u64 size_bits;
	struct cass_dev *hw = container_of(cac->lni_priv->dev, struct cass_dev,
					   cdev);
	struct cass_atu_cq *cq = &hw->atu_cq;

	if (cac->cfg_ac.do_not_cache) {
		/* Completion wait needs to be performed so that the ODP
		 * pri_worker is not referencing the AC while it is being
		 * destroyed.
		 */
		if (cac->flags & CXI_MAP_USER_ADDR) {
			mutex_lock(&cq->atu_cq_mutex);
			cass_completion_wait(hw, 0);
			mutex_unlock(&cq->atu_cq_mutex);
		}

		return;
	}

	pr_debug("ac:%d iova:%llx length:%lx\n", cac->ac.acid, iova, length);

	mutex_lock(&cq->atu_cq_mutex);

	while (length > 0) {
		union c_atu_cfg_cq_table entry = {};
		unsigned int wr_index = 0;

		/* save room for the completion wait command */
		while (length > 0 &&
		       (C_ATU_CFG_CQ_TABLE_ENTRIES - wr_index) > 1) {
			/* Walk through the iova and invalidate a len
			 * based on its current alignment and the length
			 * remaining.
			 */
			len = BIT(flsl(length));
			align = !iova ? len : BIT(ffsl(iova));

			if (len > align)
				len = align;

			/*
			 * size_bits gives the 1's necessary to fill in the
			 * lower bits of inval_addr and the inval_size fields
			 */
			size_bits = ((len - 1) & ~MASK(C_ADDR_SHIFT)) >> 1;
			entry.qw = iova | size_bits;
			entry.cmd = C_ATU_INVALIDATE_PAGES;
			entry.inval_acid = cac->ac.acid;

			atu_debug("ac:%d entry:%llx addr:%llx len:%llx\n",
				  cac->ac.acid, entry.qw, iova, len);
			cass_nta_cq_write(hw, ATUCQ_ENTRY(wr_index), &entry);

			length -= len;
			iova += len;
			wr_index++;
		}

		cass_completion_wait(hw, wr_index);
	}

	mutex_unlock(&cq->atu_cq_mutex);
}

/**
 * cass_inbound_wait() - flush internal Cassini buffers
 *
 * Called during resource cleanup.
 *
 * @hw: the device
 * @wait_for_response: wait for hardware completion
 *
 * Return: 0 on success, negative errno on failure
 */
int cass_inbound_wait(struct cass_dev *hw, bool wait_for_response)
{
	int ret;
	int rc = 0;
	int count = 0;
	u64 rsp_read_data;
	struct cass_atu_cq *cq = &hw->atu_cq;
	union c_atu_cfg_inbound_wait cfg_iw = {
		.rsp_en = wait_for_response,
		.rsp_addr = cq->ib_wait_rsp_addr,
	};
	static bool hw_failure;

	/* hw_failure was set previously. No need to progress */
	if (hw_failure)
		return -EHOSTDOWN;

	cq->wait_rsp_data.ib = 0;

	/* Writes to C_ATU_CFG_CMDPROC are ignored if ATUCQ is not idle. */
	ret = cass_poll_sts_cmdproc_idle(hw, true);
	if (ret) {
		rc = -EALREADY;
		goto done;
	}

	/* process an inbound wait */
	cass_write(hw, C_ATU_CFG_INBOUND_WAIT, &cfg_iw, sizeof(cfg_iw));
	cass_flush_pci(hw);

	if (wait_for_response) {
		do {
			ret = readq_poll_timeout_atomic(
				&cq->wait_rsp_data.ib, rsp_read_data,
				(rsp_read_data == WAIT_RSP_DATA), 1,
				ATU_IBW_TIMEOUT);
			if (ret) {
				hw_failure = cass_cmdproc_debug(hw, true);

				if (hw_failure) {
					rc = -EHOSTDOWN;
					break;
				}

				if (count++ > 4) {
					rc = -ETIMEDOUT;
					pr_err("Inbound wait is stuck - bailing out\n");
					break;
				}
			}
		} while (ret);
	}

done:
	return rc;
}

extern bool ibw_epoch_cntr_is_0(struct cass_dev *hw, bool *epoch_ret);
/*
 * cxi_inbound_wait - Flush internal cassini buffers
 *
 * @cdev: Cassini Device
 * Return: 0 on success, negative errno on failure
 */
int cxi_inbound_wait(struct cxi_dev *cdev)
{
	int ret;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	struct cass_atu_cq *cq = &hw->atu_cq;

	mutex_lock(&cq->atu_ib_mutex);
	ret = cass_inbound_wait(hw, true);
	mutex_unlock(&cq->atu_ib_mutex);

	return ret;
}
EXPORT_SYMBOL(cxi_inbound_wait);

/**
 * cass_nta_iova_alloc() - Allocate an IOVA from the domain
 *
 * @cac: Address Context
 * @m_opts: Map options
 */
u64 cass_nta_iova_alloc(struct cass_ac *cac, const struct ac_map_opts *m_opts)
{
	u64 iova;

	if (!cac->iovad)
		return 0;

	iova = ALLOC_IOVA_FAST(cac->iovad, m_opts->va_len, cac->iova_end);

	/* Check for an iova overflow because of a 5.11+ (sles15sp4+)
	 * iova bug. Free the current allocation and return iova not found.
	 */
	if (iova && iova < cac->iova_base) {
		pr_debug("alloc_iova overflow iova:%llx iova_base:%llx\n",
			 iova, cac->iova_base);
		FREE_IOVA_FAST(cac->iovad, iova, m_opts->va_len);
		iova = 0;
	}

	return iova;
}

void cass_nta_iova_fini(struct cass_ac *cac)
{
	if (cac->iovad) {
		put_iova_domain(cac->iovad);
		kfree(cac->iovad);
	}
}

int cass_nta_iova_init(struct cass_ac *cac, size_t va_len)
{
#ifdef HAVE_IOVA_INIT_RCACHES
	int ret;
#endif

	cac->iovad = kzalloc(sizeof(*cac->iovad), GFP_KERNEL);
	if (!cac->iovad)
		return -ENOMEM;

	INIT_IOVA_DOMAIN(cac->iovad, PAGE_SIZE, cac->iova_base,
			 cac->iova_end);

/* linux 5.18 removed the initialization of the rcaches from init_iova_domain */
#ifdef HAVE_IOVA_INIT_RCACHES
	ret = iova_domain_init_rcaches(cac->iovad);
	if (ret) {
		cass_nta_iova_fini(cac);
		return ret;
	}
#endif

	return 0;
}

void cass_nta_free(struct cass_dev *hw, struct cass_nta *nta)
{
	dma_unmap_page(&hw->cdev.pdev->dev, nta->root_page_dma_addr,
		       PAGE_SIZE << nta->root_order, DMA_TO_DEVICE);
	__free_pages(nta->root_page, nta->root_order);
	kfree(nta);
}

static struct cass_nta *cass_nta_alloc(struct cass_ac *cac, int l0_entries,
				       int l1_entries)
{
	struct cass_dev *hw = container_of(cac->lni_priv->dev, struct cass_dev, cdev);
	struct cass_nta *nta;
	int ret;

	nta = kzalloc(struct_size(nta, l1, l0_entries), GFP_KERNEL);
	if (!nta)
		return ERR_PTR(-ENOMEM);

	nta->numa_node = dev_to_node(&hw->cdev.pdev->dev);
	nta->root_order = get_order(l0_entries * ATU_PTE_ENTRY_SIZE);
	nta->l1_order = get_order(l1_entries * ATU_PTE_ENTRY_SIZE);

	nta->root_page = alloc_pages_node(nta->numa_node, GFP_KERNEL | __GFP_ZERO,
					  nta->root_order);
	if (!nta->root_page) {
		ret = -ENOMEM;
		goto alloc_pages_fail;
	}

	nta->root_page_dma_addr = dma_map_page(&hw->cdev.pdev->dev, nta->root_page, 0,
					       PAGE_SIZE << nta->root_order,
					       DMA_TO_DEVICE);
	if (dma_mapping_error(&hw->cdev.pdev->dev, nta->root_page_dma_addr)) {
		ret = -ENOMEM;
		goto map_fail;
	}

	nta->root_ptr = page_address(nta->root_page);
	nta->l0_entries = l0_entries;
	nta->l1_entries = l1_entries;
	cac->cfg_ac.nta_root_ptr = nta->root_page_dma_addr >> C_ADDR_SHIFT;
	cac->cfg_ac.mem_base = cac->iova_base >> ATU_CFG_AC_TABLE_MB_SHIFT;
	cac->cfg_ac.mem_size = cac->iova_len >> C_ADDR_SHIFT;
	cac->cfg_ac.odp_mode = C_ATU_ODP_MODE_NIC_PRI;
	cac->cfg_ac.ta_mode = C_ATU_NTA_MODE;

	return nta;

map_fail:
	__free_pages(nta->root_page, nta->root_order);
alloc_pages_fail:
	kfree(nta);

	return ERR_PTR(ret);
}

static int cass_ac_size(const struct ac_map_opts *m_opts, struct cass_ac *cac,
			ulong npfns, int l1_entries)
{
	u64 l0_entries;

	l0_entries = round_up(npfns / l1_entries, min_root_entries(cac));
	cac->iova_base = ATU_VA_MASK(cass_iova_base(cac));
	cac->iova_len = l0_entries * (u64)l1_entries * BIT(cac->page_shift);
	cac->iova_base &= ~(cac->iova_len - 1);
	cac->iova_end = cac->iova_base + cac->iova_len - 1;

	pr_debug("ac:%d iova_base:%016llx iova_end:%016llx iova_len:%016llx\n",
		  cac->ac.acid, cac->iova_base, cac->iova_end, cac->iova_len);

	return l0_entries;
}

/**
 * cass_nta_init() - Initialize an NTA Address Context
 *
 * @lni_priv: the Logical Network Interface
 * @m_opts: Map options
 * @cac:    Address Context
 */
int cass_nta_init(struct cxi_lni_priv *lni_priv, struct ac_map_opts *m_opts,
		  struct cass_ac *cac)
{
	int ret;
	ulong npfns;
	int l0_entries;
	int l1_entries;
	int len_order;

	l1_entries = BIT(cac->cfg_ac.pg_table_size);
	npfns = m_opts->va_len >> m_opts->page_shift;
	npfns = ALIGN(npfns, l1_entries);

	/* Round up to the next power of 2 if not aligned. If the length
	 * passed to init_iova_domain is not aligned to a power of 2, calling
	 * iova_alloc_fast with a length that is larger than half of the
	 * domain size can fail since the iova is aligned to the length.
	 */
	len_order = flsl(npfns);
	if (npfns & MASK(len_order))
		npfns = BIT(len_order + 1);

	l0_entries = cass_ac_size(m_opts, cac, npfns, l1_entries);

	ret = cass_nta_iova_init(cac, m_opts->va_len);
	if (ret)
		goto iova_init_fail;

	m_opts->iova = cass_nta_iova_alloc(cac, m_opts);
	if (!m_opts->iova) {
		if (cac->iovad) {
			pr_debug("cass_iova_alloc failed %lld\n", m_opts->iova);
			ret = -ENOENT;
			goto alloc_iova_fail;
		}
	}

	cac->nta = cass_nta_alloc(cac, l0_entries, l1_entries);
	if (IS_ERR(cac->nta)) {
		ret = PTR_ERR(cac->nta);
		goto iova_cleanup;
	}

	return 0;

iova_cleanup:
	if (cac->iovad)
		FREE_IOVA_FAST(cac->iovad, m_opts->iova, m_opts->va_len);
alloc_iova_fail:
	cass_nta_iova_fini(cac);
iova_init_fail:

	return ret;
}

static void cass_l1_table_free(struct cass_dev *hw, struct cass_nta *nta,
			       unsigned int l0_index)
{
	struct l1_entry *l1 = &nta->l1[l0_index];

	dma_unmap_page(&hw->cdev.pdev->dev, l1->l1_pages_dma_addr,
		       PAGE_SIZE << nta->l1_order, DMA_TO_DEVICE);
	__free_pages(l1->l1_pages, nta->l1_order);
	bitmap_free(l1->bitmap);

	memset(l1, 0, sizeof(*l1));
}

/* Return the virtual address of a given PTE */
static union a_pte *get_virt_pte(const struct cass_nta *nta, unsigned int l0_index,
				 unsigned int l1_index)
{
	union a_pte *l1_array;

	l1_array = page_address(nta->l1[l0_index].l1_pages);

	return &l1_array[l1_index];
}

static void cass_pte_clear(struct cass_dev *hw, const struct cxi_md_priv *md_priv,
			   unsigned int l0_index, unsigned int l1_index,
			   bool unmap)
{
	struct cass_ac *cac = md_priv->cac;
	struct cass_nta *nta = cac->nta;
	union a_pte *pte = get_virt_pte(nta, l0_index, l1_index);
	unsigned long *map;

	if (pte && pte->pte.p) {
		if (unmap)
			dma_unmap_page(&hw->cdev.pdev->dev, PTE_ADDR(pte->pte),
				       1 << cac->page_shift, DMA_BIDIRECTIONAL);
		pte->qw = 0;
	}

	map = nta->l1[l0_index].bitmap;
	if (map)
		clear_bit(l1_index, map);
}

static void cass_clear_pde(struct cass_dev *hw, union a_pte *pde)
{
	pde->qw = 0;
}

void cass_l1_tables_free(struct cass_dev *hw, struct cass_nta *nta)
{
	int i;
	union a_pte *pde;

	for (i = 0; i < nta->l0_entries; i++) {
		if (!nta->l1[i].bitmap)
			continue;

		pde = &nta->root_ptr[i];
		if (!pde->qw) {
			pr_warn("%d map entry but pde 0\n", i);
			continue;
		}

		cass_l1_table_free(hw, nta, i);
		cass_clear_pde(hw, pde);
	}
}

static void cass_pg_tbl_indices(const struct cass_ac *cac, s64 iova_offset,
				int *l0_index, int *l1_index)
{
	int pts = cac->cfg_ac.pg_table_size;
	int l1_shift = cac->cfg_ac.base_pg_size + C_ADDR_SHIFT;
	int l0_shift = l1_shift + pts;

	*l0_index = iova_offset >> l0_shift;
	*l1_index = (iova_offset >> l1_shift) & MASK(pts);
}

/**
 * cass_clear_range() - clear a range of ptes and invalidate the ATU cache
 *
 * @md_priv: Private memory descriptor
 * @iova: IO virtual address
 * @len: IOVA range to clear
 */
void cass_clear_range(const struct cxi_md_priv *md_priv, u64 iova, u64 len)
{
	struct cass_dev *hw = container_of(md_priv->cac->lni_priv->dev,
					   struct cass_dev, cdev);
	int l0_index;
	int l1_index;
	struct cass_ac *cac = md_priv->cac;
	int page_shift = cac->page_shift;
	int entries = len >> page_shift;
	s64 iova_offset = iova - cac->iova_base;
	union a_pte *pde;
	/* Only ODP needs to be unmapped here. Device and pinned memory will
	 * be unmapped when the SG table is cleaned up.
	 */
	bool do_unmap = !md_priv->sgt;

	if (!len)
		return;

	for (; entries > 0; iova_offset += BIT(page_shift), entries--) {
		cass_pg_tbl_indices(cac, iova_offset, &l0_index, &l1_index);

		pde = &cac->nta->root_ptr[l0_index];
		if (!pde->qw)
			continue;

		if (!cac->cfg_ac.pg_table_size) {
			/* single level table */
			if (do_unmap)
				dma_unmap_page(&hw->cdev.pdev->dev,
					       PTE_ADDR(pde->pte),
					       1 << cac->page_shift,
					       DMA_BIDIRECTIONAL);
			pde->qw = 0;
		} else if (pde->pte.leaf) {
			if (do_unmap)
				dma_unmap_page(&hw->cdev.pdev->dev,
					       PTE_ADDR(pde->pte),
					       1 << cac->huge_shift,
					       DMA_BIDIRECTIONAL);
			cass_clear_pde(hw, pde);
			/*
			 * the rest of the entries for this pde don't need to
			 * be checked
			 */
			entries -= ATU_PMD_NR(cac) - 1;
			iova_offset += BIT(page_shift) * (ATU_PMD_NR(cac) - 1);
		} else {
			if (!cac->nta->l1[l0_index].bitmap)
				pr_err("ac:%d iova:%llx offset:%llx map %d is NULL\n",
				       cac->ac.acid, iova, iova_offset,
				       l0_index);

			cass_pte_clear(hw, md_priv, l0_index, l1_index,
				       do_unmap);
		}
	}
}

/**
 * cass_pte_build() - construct a page table entry based on the pfn and flags
 *
 * @hpfn: pfn from the CPU page table containing read and write permissions
 * @pte: constructed page table entry
 * @dma_addr: DMA address of the page frame number
 * @leaf: indication of the final translation - l0 hugepage entry or l1 entry
 * @flags: user requested read and write permissions
 * @is_hmm_pfn: DMA address is from an hpfn - use the hpfn flags
 * @l0_entry: indicates this is a root table entry
 */
static void cass_pte_build(u64 hpfn, union a_pte *pte, dma_addr_t dma_addr,
			   bool leaf, u32 flags, bool is_hmm_pfn,
			   bool l0_entry)
{
	/* Always set rd/wr permissions for a non-leaf root table entry
	 * since the 2nd level could be either rd or wr and the NIC checks
	 * both levels for proper permissions.
	 */
	bool rw = l0_entry && !leaf;
	bool rd = rw || (flags & CXI_MAP_READ);
	bool wr = rw || (flags & CXI_MAP_WRITE);
	bool h_wr = rw || ATU_PFN_WRITE(hpfn);

	pte->qw = dma_addr;
	pte->pte.leaf = leaf;

	if (is_hmm_pfn) {
		if (ATU_PFN_INVALID(hpfn)) {
			pte->qw = 0;
			return;
		}

		pte->pte.r = rd;
		pte->pte.w = h_wr && wr;
		pte->pte.p = !leaf || ATU_PFN_VALID(hpfn);
	} else {
		/* TODO: check page attributes? */
		pte->pte.r = rd;
		pte->pte.w = wr;
		pte->pte.p = 1;
	}
}

/**
 * cass_l1_table_empty - Free l1 map buffer if table is empty.
 *
 * @cac:   Address context
 * @pde:   Page directory entry
 * @index: L1 map to free and clean up
 * @iova:  IO virtual address
 */
static int cass_l1_table_empty(const struct cass_ac *cac,
			       union a_pte *pde, int index, u64 iova)
{
	struct cass_dev *hw = container_of(cac->lni_priv->dev,
					   struct cass_dev, cdev);
	struct cass_nta *nta = cac->nta;

	if (!nta->l1[index].bitmap) {
		pr_err("ac:%d iova:%llx map %d is NULL\n", cac->ac.acid,
		       iova, index);
		return -EINVAL;
	}

	if (!bitmap_empty(nta->l1[index].bitmap, nta->l1_entries)) {
		pr_err("bitmap is not empty iova:%llx\n", iova);
		return -EINVAL;
	}

	cass_l1_table_free(hw, nta, index);

	return 0;
}

static int cass_l1_table_alloc(struct cass_dev *hw, struct cass_nta *nta,
			       int l0_index)
{
	struct l1_entry *l1 = &nta->l1[l0_index];

	l1->bitmap = bitmap_zalloc(nta->l1_entries, GFP_KERNEL);
	if (!l1->bitmap)
		return -ENOMEM;

	l1->l1_pages = alloc_pages_node(nta->numa_node, GFP_KERNEL | __GFP_ZERO,
					nta->l1_order);
	if (!l1->l1_pages)
		goto free_bitmap;

	l1->l1_pages_dma_addr = dma_map_page(&hw->cdev.pdev->dev, l1->l1_pages,
					     0, PAGE_SIZE << nta->l1_order,
					     DMA_TO_DEVICE);
	if (dma_mapping_error(&hw->cdev.pdev->dev, l1->l1_pages_dma_addr))
		goto free_pages;

	return 0;

free_pages:
	__free_pages(l1->l1_pages, nta->l1_order);
free_bitmap:
	bitmap_free(l1->bitmap);

	memset(l1, 0, sizeof(*l1));

	return -ENOMEM;
}

int cass_dma_addr_mirror(dma_addr_t dma_addr, u64 iova, struct cass_ac *cac,
			 u32 flags, bool is_huge_page, bool *invalidate)
{
	int ret;
	int l0_index;
	int l1_index;
	union a_pte *ppde;
	union a_pte *ppte;
	union a_pte pde = {};
	union a_pte pte = {};
	bool update_pde = false;
	s64 iova_offset = iova - cac->iova_base;
	bool ptg_mode_sgl = !cac->cfg_ac.pg_table_size;
	struct cass_dev *hw = container_of(cac->lni_priv->dev, struct cass_dev,
					   cdev);

	if (iova_offset > cac->iova_len) {
		pr_warn("iova_offset:0x%llx > iova_len:0x%llx iova:%llx iova_base:%llx\n",
			iova_offset, cac->iova_len, iova, cac->iova_base);
		return -EINVAL;
	}

	cass_pg_tbl_indices(cac, iova_offset, &l0_index, &l1_index);
	ppde = &cac->nta->root_ptr[l0_index];

	if (is_huge_page && ppde->qw) {
		if (l1_index) {
			pr_err("Huge page not aligned iova:%llx\n", iova);
			return -EINVAL;
		}

		/* Replacing a non-leaf entry with a leaf entry. */
		/* TODO: check if this is still necessary */
		if (!ppde->pte.leaf) {
			ret = cass_l1_table_empty(cac, ppde, l0_index,
						  iova_offset);
			if (ret)
				return ret;

			pr_debug("Replacing non-leaf pde iova:%llx entry:%lx\n",
				 iova_offset, (ulong)PTE_ADDR(ppde->pte));
			update_pde = true;
			goto new_pde;
		}

		/* It is possible to get an ODP page request for the same
		 * IOVA since we try to fill more than one entry at a time.
		 */
		cass_pte_build(0, &pde, dma_addr, true, flags, false, true);

		if (pde.qw != ppde->qw) {
			pr_err("Overlapping huge page ac:%d iova:%llx\n",
			       cac->ac.acid, iova_offset);
			return -EINVAL;
		}

		return 0;
	}

new_pde:
	if (!ppde->qw || update_pde) {
		if (is_huge_page || ptg_mode_sgl) {
			cass_pte_build(0, &pde, dma_addr, true, flags, false,
				       true);
		} else {
			if (update_pde) {
				pr_err("Expected hugepage update iova:%llx l0_index:%d l1_index:%d\n",
				       iova_offset, l0_index, l1_index);
				return -EINVAL;
			}

			ret = cass_l1_table_alloc(hw, cac->nta, l0_index);
			if (ret) {
				pr_err("Error allocating l1 table\n");
				return ret;
			}

			cass_pte_build(0, &pde,
				       cac->nta->l1[l0_index].l1_pages_dma_addr,
				       false, flags, false, true);
		}

		atu_debug("pde iova:%llx dma_addr:0x%llx entry:%llx\n",
			  iova, dma_addr, pde.qw);
		ppde->qw = pde.qw;

		if (update_pde)
			cass_invalidate_range(cac, cac->iova_base + iova_offset,
					      BIT(cac->huge_shift));

		if (is_huge_page || ptg_mode_sgl)
			return 0;
	} else if (ppde->pte.leaf) {
		pr_err("Unexpected leaf iova:%llx pde %llx, l0_index:%d l1_index:%d\n",
		       iova_offset, ppde->qw, l0_index, l1_index);

		return -EINVAL;
	}

	/* fill in the l1 table entry */
	cass_pte_build(0, &pte, dma_addr, true, flags, false, false);
	ppte = get_virt_pte(cac->nta, l0_index, l1_index);

	atu_debug("pte %p phys:%lx iova:%llx entry:0x%llx\n", ppte,
		  (ulong)PTE_ADDR(pte.pte) + l1_index * ATU_PTE_ENTRY_SIZE,
		  iova, pte.qw);

	ppte->qw = pte.qw;
	set_bit(l1_index, cac->nta->l1[l0_index].bitmap);

	return 0;
}

/**
 * cass_pfns_mirror() - Mirror PTEs from a list of HMM pfns
 *
 * @md_priv: Private memory descriptor
 * @m_opts: map options
 * @pfns: pfns to mirror
 * @npfns: number of pfns to mirror
 * @is_huge_page: The vma is covered by a hugepage. For device memory,
 *                indicates that the pfns are contiguous.
 *
 * Return: 0 on success or negative error value
 */
int cass_pfns_mirror(struct cxi_md_priv *md_priv, const struct ac_map_opts *m_opts,
		     u64 *pfns, int npfns, bool is_huge_page)
{
	int ret;
	int i;
	size_t size;
	bool inval = false;
	int invalidate = 0;
	struct page *page;
	dma_addr_t dma_addr;
	u64 iova = m_opts->iova;
	struct cass_ac *cac = md_priv->cac;
	size_t page_size = BIT(cac->page_shift);

	for (i = 0; i < npfns;) {
		/* Skip pfns that are invalid which may happen during
		 * prefetch.
		 */
		if (!ATU_PFN_VALID(pfns[i])) {
			i += ATU_PTE_NR(cac);
			iova += page_size;
			continue;
		}

		page = hmm_pfn_to_page(pfns[i]);

		/* Check if there are enough remaining pfns to cover an L0
		 * entry.
		 */
		if ((npfns - i) < K_PMD_NR(cac))
			is_huge_page = false;

		if (is_huge_page &&
		    ((iova & MASK(cac->huge_shift)) || !PageHuge(page)))
			is_huge_page = false;

		size = is_huge_page ? BIT(cac->huge_shift) : page_size;
		dma_addr = dma_map_page(md_priv->device, page, 0, size,
					DMA_BIDIRECTIONAL);
		if (dma_mapping_error(md_priv->device, dma_addr)) {
			ret = -EINVAL;
			goto mirror_error;
		}

		ret = cass_dma_addr_mirror(dma_addr, iova, cac, md_priv->flags,
					   is_huge_page, &inval);
		if (ret)
			goto mirror_error;

		i += KPFN_INC(cac, is_huge_page);
		invalidate += inval;
		iova += size;
	}

mirror_error:
	if (i < npfns) {
		pr_err("Error populating %d PTEs at index %d\n", npfns, i);

		/* Clean up previously populated ptes. */
		cass_clear_range(md_priv, m_opts->iova, i * page_size);
	}

	/* If any PTEs are replaced they will need to be invalidated.
	 * Invalidate the whole range.
	 */
	if (invalidate)
		cass_invalidate_range(cac, m_opts->iova, npfns * page_size);

	return ret;
}

static struct page *cass_virt_to_page(u64 addr)
{
	if (is_vmalloc_addr((void *)addr))
		return vmalloc_to_page((void *)addr);
	else
		return virt_to_page(addr);
}

static bool cass_error_inject(const struct cxi_lni_priv *lni_priv)
{
	struct cass_dev *hw =
		container_of(lni_priv->dev, struct cass_dev, cdev);

	/* decrement if not 0 */
	atomic_add_unless(&hw->atu_error_inject, -1, 0);

	if (atomic_read(&hw->atu_error_inject) == 1)
		return true;

	return false;
}

/**
 * cass_pin() - Call pin_user_pages() to pin an address range
 *
 * @cac:     Address Context metadata
 * @pages:   Array of pages returned. Only used if there is an error.
 * @npages:  Number of pages to pin
 * @addr:    Virtual address
 * @write:   Write permission
 *
 * Return: Success or -EFAULT or return value of gup
 */
int cass_pin(const struct cass_ac *cac, struct page **pages, int npages,
	     u64 addr, bool write)
{
	int i;
	int gotn;
	unsigned int gup_flags = write ? FOLL_WRITE : 0;

	gotn = pin_user_pages_fast(addr, npages, gup_flags, pages);
	if (gotn < 0)
		return gotn;

	if ((gotn != npages) || cass_error_inject(cac->lni_priv)) {
		pr_debug("gup failed gotn %d of %d\n", gotn, npages);

		for (i = 0; i < gotn; i++)
			unpin_user_page(pages[i]);

		return -EFAULT;
	}

	return 0;
}

/**
 * cass_mirror_hp() - Mirror simulated hugepages from a large hugepage
 *
 * Support testing of arbitrary hugepage sizes.
 * These pages do not have a notifier callback.
 * The pinned pages are contiguous pages masquerading as a larger page size.
 *
 * @m_opts: map options
 * @md_priv: Private memory descriptor
 *
 * Return: number of mirrored pages or error
 */
int cass_mirror_hp(const struct ac_map_opts *m_opts, struct cxi_md_priv *md_priv)
{
	int i;
	int ret = 0;
	int l0_index;
	int l1_index;
	dma_addr_t dma_addr;
	union a_pte *ppde;
	union a_pte pde = {};
	struct page *page;
	u64 iova = m_opts->iova;
	struct cass_ac *cac = md_priv->cac;
	uintptr_t addr = m_opts->va_start;
	int n_mir = 1 + ((m_opts->va_len - 1) >> cac->huge_shift);

	mutex_lock(&cac->ac_mutex);

	for (i = 0; i < n_mir; i++) {
		ret = cass_pin(cac, &page, 1, addr, true);
		if (ret)
			break;

		unpin_user_page(page);

		dma_addr = dma_map_page(md_priv->device, page, 0,
					1 << cac->huge_shift,
					DMA_BIDIRECTIONAL);
		if (dma_mapping_error(md_priv->device, dma_addr)) {
			ret = -ENOMEM;
			break;
		}

		cass_pg_tbl_indices(cac, iova - cac->iova_base, &l0_index,
				    &l1_index);
		ppde = &cac->nta->root_ptr[l0_index];
		cass_pte_build(0, &pde, dma_addr, true, m_opts->flags, false,
			       true);

		atu_debug("pde dma_addr:0x%llx entry:%llx\n", dma_addr, pde.qw);
		ppde->qw = pde.qw;

		iova += BIT(cac->huge_shift);
		addr += BIT(cac->huge_shift);
	}

	if (ret) {
		pr_debug("ret:%d\n", ret);
		cass_clear_range(md_priv, m_opts->iova, i << cac->huge_shift);
	}

	mutex_unlock(&cac->ac_mutex);

	return ret;
}

/**
 * cass_mirror_device() - Mirror a GPU device memory address range into
 *                        device page tables
 *
 * @md_priv: Private memory descriptor
 * @m_opts:  User options containing the IOVA range
 *
 * Return: 0 on success, -ENOENT if not GPU memory or error
 */
int cass_mirror_device(struct cxi_md_priv *md_priv,
		       const struct ac_map_opts *m_opts)
{
	int i;
	int j = 0;
	int ret;
	bool inval;
	struct scatterlist *sg;
	u64 iova = md_priv->md.iova;
	struct cass_ac *cac = md_priv->cac;
	size_t plen = BIT(cac->page_shift);
	size_t hlen = BIT(cac->huge_shift);

	mutex_lock(&md_priv->cac->ac_mutex);

	i = 0;
	for_each_sgtable_dma_sg(md_priv->sgt, sg, j) {
		long len = sg_dma_len(sg);
		dma_addr_t dma_addr = sg_dma_address(sg);

		atu_debug("dma_addr:%llx len:%lx\n", dma_addr, len);

		while (len > 0) {
			bool is_hp = ffsl(dma_addr) >= cac->huge_shift &&
					ffsl(iova) >= cac->huge_shift &&
					len >= hlen;
			size_t inc = is_hp ? hlen : plen;

			atu_debug("iova:%llx dma_addr:%llx inc:%lx\n",
				  iova, dma_addr, inc);
			ret = cass_dma_addr_mirror(dma_addr, iova, cac,
						   md_priv->flags,
						   is_hp, &inval);
			if (ret)
				goto mirror_error;

			iova += inc;
			dma_addr += inc;
			len -= inc;
			i += KPFN_INC(cac, is_hp);
		}
	}

	mutex_unlock(&md_priv->cac->ac_mutex);

	return 0;

mirror_error:
	if (i) {
		pr_err("Error populating length 0x%lx at index %d\n",
		       md_priv->md.len, i);

		/* Clean up previously populated ptes. */
		cass_clear_range(md_priv, md_priv->md.iova, md_priv->md.len);
	}

	mutex_unlock(&md_priv->cac->ac_mutex);

	return ret;
}
/**
 * cass_mirror_odp() - Mirror a virtual address range into device page tables
 *
 * @m_opts:  User options containing the IOVA range
 * @cac:     Address Context
 * @npfns:   Number of pages
 * @addr:    Virtual address
 *
 * @return: 0 on success or negative error value
 */
int cass_mirror_odp(const struct ac_map_opts *m_opts, struct cass_ac *cac,
		    int npfns, u64 addr)
{
	int ret;
	u64 *pfns;

	pfns = kcalloc(npfns, sizeof(*pfns), GFP_KERNEL);
	if (!pfns)
		return -ENOMEM;

	ret = cass_mirror_fault(m_opts, pfns, npfns, addr, m_opts->va_len);

	kfree(pfns);

	return ret;
}

static int cass_dma_map_pages(struct cxi_md_priv *md_priv,
			      struct page **pages, int npages, bool is_huge_page)
{
	int ret;
	struct sg_table *sgt;

	sgt = kzalloc(sizeof(*sgt), GFP_KERNEL);
	if (!sgt)
		return -ENOMEM;

	ret = sg_alloc_table_from_pages_segment(sgt, pages, npages,
			0, md_priv->md.len, UINT_MAX, GFP_KERNEL);
	if (ret)
		goto sg_table_error;

	ret = dma_map_sgtable(md_priv->device, sgt, DMA_BIDIRECTIONAL, 0);
	if (ret)
		goto dma_map_error;

	md_priv->sgt = sgt;

	return 0;

dma_map_error:
	sg_free_table(sgt);
sg_table_error:
	kfree(sgt);

	return ret;
}

void cass_dma_unmap_pages(struct cxi_md_priv *md_priv)
{
	if (!md_priv->sgt)
		return;

	if (md_priv->external_sgt_owner)
		return;

	dma_unmap_sgtable(md_priv->device, md_priv->sgt, DMA_BIDIRECTIONAL, 0);
	sg_free_table(md_priv->sgt);
	kfree(md_priv->sgt);
	md_priv->sgt = NULL;
}

/**
 * cass_pin_mirror() - Pin and mirror a virtual address range into device
 *                     page tables
 *
 * @m_opts:  User options containing the IOVA range
 * @md_priv: Private memory descriptor
 *
 * Return: 0 on success or error
 */
int cass_pin_mirror(struct cxi_md_priv *md_priv, struct ac_map_opts *m_opts)
{
	int i;
	int ret;
	int addr_inc;
	bool inval = false;
	int invalidate = 0;
	bool is_huge_page;
	u64 iova = m_opts->iova;
	struct page **pages = NULL;
	struct cass_ac *cac = md_priv->cac;
	uintptr_t addr = m_opts->va_start;
	uintptr_t end = m_opts->va_end;
	int hlen = BIT(m_opts->huge_shift);
	int plen = BIT(m_opts->page_shift);
	u64 hmask = hlen - 1;
	size_t md_len = m_opts->va_len;
	int npages = m_opts->va_len >> PAGE_SHIFT;
	struct scatterlist *sg;

	pages = kvmalloc_array(npages, sizeof(pages), GFP_KERNEL);
	if (!pages)
		return -ENOMEM;

	mmap_read_lock(current->mm);
	ret = cass_vma_write_flag(current->mm, addr, end, m_opts->flags);
	mmap_read_unlock(current->mm);

	if (ret < 0)
		goto err;

	ret = cass_pin(cac, pages, npages, m_opts->va_start, ret);
	if (ret < 0)
		goto err;

	ret = cass_dma_map_pages(md_priv, pages, npages, m_opts->is_huge_page);
	if (ret)
		goto map_err;

	mutex_lock(&cac->ac_mutex);

	for_each_sgtable_dma_sg(md_priv->sgt, sg, i) {
		size_t len = sg_dma_len(sg);
		dma_addr_t dma_addr = sg_dma_address(sg);

		while (len > 0 && md_len > 0) {
			is_huge_page = m_opts->is_huge_page && (len >= hlen) &&
						!(iova & hmask);
			addr_inc = is_huge_page ? hlen : plen;

			ret = cass_dma_addr_mirror(dma_addr, iova, cac,
						   m_opts->flags,
						   is_huge_page, &inval);
			if (ret) {
				pr_err("Error populating %d PTEs at index %d iova:%llx inc:%x\n",
				       npages, i, iova, addr_inc);
				goto mirror_error;
			}

			dma_addr += addr_inc;
			iova += addr_inc;
			len -= addr_inc;
			md_len -= addr_inc;
			invalidate += inval;
		}

		if (md_len <= 0)
			break;
	}

	mutex_unlock(&cac->ac_mutex);

	if (invalidate)
		cass_invalidate_range(cac, m_opts->iova, plen * i);

	md_priv->pages = pages;

	return 0;

mirror_error:
	cass_clear_range(md_priv, m_opts->iova, plen * i);
	mutex_unlock(&cac->ac_mutex);
	cass_dma_unmap_pages(md_priv);

map_err:
	unpin_user_pages(pages, npages);

err:
	kvfree(pages);

	return ret;
}

static int atu_odpq_space(struct cass_dev *hw)
{
	union c_atu_sts_odpq sts_odpq;

	cass_read(hw, C_ATU_STS_ODPQ, &sts_odpq, sizeof(sts_odpq));

	return sts_odpq.odpq_space;
}

static int cass_odpq_space(struct cass_dev *hw)
{
	int ret;
	int space;

	ret = readx_poll_timeout(atu_odpq_space, hw, space, (space > 0), 1,
				 ATU_ODPQ_SPACE_TIMEOUT);
	if (ret) {
		pr_err("Timed out waiting for ODPQ space\n");
		return ret;
	}

	return space;
}

/**
 * cass_odpq_write() - Write the result of a page request
 *
 * Also used to clear out expired page requests.
 * The queue is 8 deep and we just write one entry at
 * a time so it should never overflow.
 *
 * @hw:     the device
 * @index:  PRB entry
 * @status: C_ATU_ODP_FAILURE or C_ATU_ODP_SUCCESS
 */
static void cass_odpq_write(struct cass_dev *hw, int index,
			    enum c_odp_status status)
{
	int space;
	union c_atu_cfg_odpq odpq = {
		.status = status,
		.prb_index = index
	};

	mutex_lock(&hw->odpq_mutex);

	space = cass_odpq_space(hw);
	if (space <= 0)
		pr_err("Timed out waiting for ODPQ\n");
	else
		cass_write(hw, C_ATU_CFG_ODPQ, &odpq, sizeof(odpq));

	mutex_unlock(&hw->odpq_mutex);
}

/**
 * cass_prb_worker() - Clean up expired PRB entries
 *
 * Scan the PRB table and look for expired entries and report entry as
 * failed. Keep scanning until there are no more entries.
 *
 * @work: the work_struct
 */
static void cass_prb_worker(struct work_struct *work)
{
	int i;
	bool scan_again;
	union c_atu_sts_prb_table entry;
	struct cass_dev *hw = container_of(work, struct cass_dev, prb_work);

	do {
		scan_again = false;

		for (i = 0; i < C_ATU_STS_PRB_TABLE_ENTRIES; i++) {
			cass_read(hw, C_ATU_STS_PRB_TABLE(i), &entry,
				  sizeof(entry));

			if (entry.valid && entry.expired) {
				pr_warn("Entry %d expired ac:%d addr:%lx\n",
					i, entry.acid,
					(ulong)entry.addr << PAGE_SHIFT);
				scan_again = true;
				cass_odpq_write(hw, i, C_ATU_ODP_FAILURE);
				atomic_inc(&hw->atu_prb_expired);
			}
		}
	} while (scan_again);
}

static void add_md_time_to_bin(struct cass_dev *hw, ktime_t time)
{
	int i;
	int bin = FIRST_MDBIN;

	if (ktime_compare(hw->pri_max_md_time, time) < 0)
		hw->pri_max_md_time = time;

	for (i = 0; i < MDBINS; i++, bin <<= 1) {
		if (time < bin) {
			hw->pri_md_time[i]++;
			break;
		}
	}
}

static void add_fault_time_to_bin(struct cass_dev *hw, ktime_t time)
{
	int i;
	int bin = FIRST_FBIN;

	if (ktime_compare(hw->pri_max_fault_time, time) < 0)
		hw->pri_max_fault_time = time;

	for (i = 0; i < FBINS; i++, bin <<= 1) {
		if (time < bin) {
			hw->pri_fault_time[i]++;
			break;
		}
	}
}

/**
 * cass_pri_fault() - Respond to a NIC PRI page request
 *
 * The page request interface consists of a buffer of 512
 * page request entries containing an AC, index and iova.
 * Fault in the page requested, clear the request in the
 * buffer and write a response to the ODP queue.
 * The page request interface can only request one page
 * at a time.
 *
 * @hw:  the device
 * @pre: page request entry
 */
static void cass_pri_fault(struct cass_dev *hw,
			   struct c_page_request_entry *pre)
{
	int rc = -1;
	u64 iova = 0;
	u64 addr = 0;
	int npages;
	int align_shift;
	int index = pre->index;
	struct cass_ac *cac;
	struct cxi_md_priv *e = NULL;
	struct ac_map_opts m_opts = {};
	union c_atu_sts_prb_table entry;
	enum c_odp_status status = C_ATU_ODP_FAILURE;
	ktime_t fault_time;
	ktime_t fault_start;
	ktime_t get_md_time;
	ktime_t start = ktime_get_raw();

	cac = hw->cac_table[pre->acid];
	if (!cac)
		goto no_cac;

	iova = (u64)pre->addr << C_ADDR_SHIFT;

	if (!(cac->flags & CXI_MAP_USER_ADDR)) {
		pr_err("Kern iova:%llx page not present\n", iova);
		goto no_cac;
	}

	mutex_lock(&cac->md_mutex);
	list_for_each_entry(e, &cac->md_list, md_entry) {
		u64 md_end = e->md.iova + e->md.len;

		if ((e->md.iova <= iova) && (iova < md_end)) {
			addr = e->md.va + (iova - e->md.iova);
			m_opts.flags = e->flags;
			break;
		}
	}

	fault_start = ktime_get_raw();
	get_md_time = fault_start - start;
	add_md_time_to_bin(hw, get_md_time);

	if (!addr)
		goto no_md;

	align_shift = pri_shift;
	m_opts.page_shift = cac->page_shift;
	m_opts.va_end = addr + PAGE_SIZE;
	rc = cass_cpu_page_size(hw, &m_opts, e->mn_sub.mm, addr, NULL,
				&align_shift);
	if (rc)
		goto no_md;

	/* There may be vmas with larger hugepage sizes than the AC
	 * is configured with. Use the smaller of AC configured
	 * hugepage size or vma hugepage size (if present).
	 */
	align_shift = min_t(int, align_shift, cac->huge_shift);

	/* Align the m_opts.va_* to align_shift */
	cass_align_start_len(&m_opts, addr, PAGE_SIZE, align_shift);
	/* Bound the m_opts.va_* to the MD */
	m_opts.va_start = max_t(u64, m_opts.va_start, e->md.va);
	m_opts.va_end = min_t(u64, m_opts.va_end, e->md.va + e->md.len);
	m_opts.va_len = m_opts.va_end - m_opts.va_start;
	/* Adjust iova to match va_start offset from addr */
	m_opts.iova = iova - (addr - m_opts.va_start);
	m_opts.md_priv = e;

	atu_debug("req_iova:%llx addr:%llx md:%d md.va:%llx iova:%llx start:%llx end:%llx va_len:%lx\n",
		  iova, addr, e->md.id, e->md.va, m_opts.iova, m_opts.va_start,
		  m_opts.va_end, m_opts.va_len);

	npages = m_opts.va_len >> cac->page_shift;
	rc = cass_mirror_odp(&m_opts, cac, npages, m_opts.va_start);
	if (!rc)
		status = C_ATU_ODP_SUCCESS;
	else {
		atomic_inc(&hw->atu_odp_fails);
		cass_read(hw, C_ATU_STS_PRB_TABLE(index), &entry,
			  sizeof(entry));
		pr_warn("ODP failure rc:%d PRT:ac:%d iova:%llx mdva:%llx SPRBT:i:%d v:%d e:%d ac:%d client:%d addr:%lx w:%d r:%d\n",
			rc, pre->acid, iova, e->md.va,
			index, entry.valid, entry.expired, entry.acid,
			entry.client, (ulong)entry.addr << PAGE_SHIFT,
			entry.w, entry.r);
	}

	fault_time = ktime_get_raw() - fault_start;
	add_fault_time_to_bin(hw, fault_time);

no_md:
	mutex_unlock(&cac->md_mutex);
no_cac:
	atu_debug("sts:%s rc:%d ac:%d index:%d iova:%llx md.va:%llx\n",
		  status ? "f" : "s", rc, pre->acid, index, iova, e ? e->md.va : 0);

	*(u64 *)pre = 0;
	/* make sure entry is zeroed before reporting to hw */
	wmb();
	atomic_inc(&hw->atu_odp_requests);
	cass_odpq_write(hw, index, status);
}

static void cass_pri_worker(struct work_struct *work)
{
	struct c_page_request_entry *pre;
	struct cass_dev *hw = container_of(work, struct cass_dev, pri_work);
	int rd_ptr = hw->pri_rd_ptr;

	while (1) {
		pre = &hw->page_request_table[rd_ptr];

		if (!pre->acid)
			break;

		cass_pri_fault(hw, pre);
		rd_ptr = (rd_ptr + 1) % C_ATU_PRB_ENTRIES;
	}

	hw->pri_rd_ptr = rd_ptr;
}

static irqreturn_t cass_pri_int_cb(int irq, void *context)
{
	struct cass_dev *hw = context;

	queue_work(hw->pri_wq, &hw->pri_work);

	return IRQ_HANDLED;
}

int cass_nta_mirror_sgt(struct cxi_md_priv *md_priv, bool need_lock)
{
	int i;
	int j = 0;
	int ret;
	bool inval;
	struct scatterlist *sg;
	u64 iova = md_priv->md.iova;
	struct cass_ac *cac = md_priv->cac;

	cass_cond_lock(&cac->ac_mutex, need_lock);

	for_each_sgtable_dma_sg(md_priv->sgt, sg, i) {
		long len = sg_dma_len(sg);
		dma_addr_t dma_addr = sg_dma_address(sg) & PAGE_MASK;

		atu_debug("dma_addr:%llx len:%lx\n", dma_addr, len);

		while (len > 0) {
			ret = cass_dma_addr_mirror(dma_addr, iova, cac,
						   md_priv->flags, false,
						   &inval);
			if (ret)
				goto mirror_error;

			dma_addr += PAGE_SIZE;
			iova += PAGE_SIZE;
			len -= PAGE_SIZE;
			j++;
		}
	}

	cass_cond_unlock(&cac->ac_mutex, need_lock);

	return 0;

mirror_error:
	cass_clear_range(md_priv, md_priv->md.iova, j * PAGE_SIZE);
	cass_cond_unlock(&cac->ac_mutex, need_lock);

	return ret;
}

int cass_nta_mirror_kern(struct cxi_md_priv *md_priv,
			 const struct iov_iter *iter, bool need_lock)
{
	int i;
	int j;
	int ret;
	int alen;
	struct page *page;
	struct sg_table *sgt;
	struct scatterlist *sg;
	u64 iova = md_priv->md.iova;
	uintptr_t addr = md_priv->md.va;
	int npages = md_priv->md.len >> PAGE_SHIFT;

	sgt = kmalloc(sizeof(struct sg_table), GFP_KERNEL);
	if (!sgt)
		return -ENOMEM;

	ret = sg_alloc_table(sgt, npages, GFP_KERNEL);
	if (ret)
		goto sg_alloc_error;

	pr_debug("va:%lx iova:%llx len:%lx\n", addr, iova, md_priv->md.len);

	sg = sgt->sgl;
	if (iter == NULL) {
		for (i = 0; i < npages; i++) {
			if (!sg) {
				pr_info("sg is null at %d\n", i);
				goto dma_map_error;
			}
			page = cass_virt_to_page(addr + (i * PAGE_SIZE));
			sg_set_page(sg, page, PAGE_SIZE, 0);
			sg = sg_next(sg);
		}
	} else if (iov_iter_is_bvec(iter)) {
		const struct bio_vec *bvec = iter->bvec;

		for (i = 0; i < npages; i++) {
			sg_set_page(sg, bvec->bv_page, PAGE_SIZE, 0);
			sg = sg_next(sg);
			bvec++;
		}
	} else if (iov_iter_is_kvec(iter)) {
		int ipages;
		const struct kvec *kvec = iter->kvec;

		for (i = 0; i < npages;) {
			alen = PAGE_ALIGN(kvec->iov_len);
			ipages = alen >> PAGE_SHIFT;

			for (j = 0; j < ipages; j++) {
				page = cass_virt_to_page((u64)kvec->iov_base +
								j * PAGE_SIZE);
				sg_set_page(sg, page, PAGE_SIZE, 0);
				sg = sg_next(sg);
				i++;
			}

			kvec++;
		}
	}

	ret = dma_map_sgtable(md_priv->device, sgt, DMA_BIDIRECTIONAL, 0);
	if (ret)
		goto dma_map_error;

	md_priv->sgt = sgt;

	ret = cass_nta_mirror_sgt(md_priv, need_lock);
	if (ret)
		goto mirror_error;

	md_priv->need_lock = false;

	return 0;

mirror_error:
	dma_unmap_sgtable(md_priv->device, sgt, DMA_BIDIRECTIONAL, 0);

dma_map_error:
	sg_free_table(sgt);

sg_alloc_error:
	kfree(sgt);

	return ret;
}
