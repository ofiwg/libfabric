// SPDX-License-Identifier: GPL-2.0
/* Copyright 2018 Hewlett Packard Enterprise Development LP */

/* Portals-like interface for Cassini */

#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/hpe/cxi/cxi.h>
#include <linux/iopoll.h>

#include "cass_core.h"
#include "cass_ss1_debugfs.h"

static unsigned int ee_timestamp_period_ms = 1000;
module_param(ee_timestamp_period_ms, uint, 0644);
MODULE_PARM_DESC(ee_timestamp_period_ms,
		 "Period between EE timestamp events in milliseconds.");

#define LT_LIMIT_MAX (BIT(9) - 1)
#define GRAN_DEF 50

static const union c_ee_cfg_latency_monitor lat_monitor_def = {
	/* The EQ latency_tolerance configuration limit. An EQ's
	 * latency_tolerance * EE granularity is the ECB expiration timeout.
	 * Limit is 9 bits wide. Allow any configuration.
	 */
	.lt_limit = LT_LIMIT_MAX,

	/* Latency step granularity. An EQ's latency_tolerance * EE granularity
	 * is the ECB expiration timeout. Granularity is effectively the step
	 * size of latency_tolerance in ns. 16 bits wide, ~65us max step size.
	 *
	 * A default of 50 allows a range of ~0-25us (50ns*511).
	 */
	.granularity = GRAN_DEF,

	/* When a configured portion of ECBs are in use, the latency monitor
	 * transitions to using a separate granularity/latency_tolerance
	 * configuration which can accelerate ECB expiration.
	 *
	 * There are two stages, A and B. Stage transitions are defined by a
	 * threshold and hysteresis value. When ECB usage hits the threshold,
	 * the latency monitor enters the corresponding stage. When ECB usage
	 * drops below the hysteresis value, the stage is exited.
	 */
	.thresh_a       = 128,
	.hysteresis_a   = 100,
	.granularity_a  = GRAN_DEF/2, /* 2x speedup in stage A */
	.lt_limit_a     = LT_LIMIT_MAX,
	.lt_reduction_a = 0,

	.thresh_b       = 196,
	.hysteresis_b   = 150,
	.granularity_b  = GRAN_DEF/10,       /* 10x speedup in stage B */
	.lt_limit_b     = 200/(GRAN_DEF/10), /* Hard limit of 200 ns */
	.lt_reduction_b = 0,
};

#define LATENCY_TOLERANCE_DEF_NS (500)

/* The max number of event entries in an event queue is calculated by:
 * The number of 4kB pages the C_EE_CFG_EQ_DESCRIPTOR BUFFER_*_SIZE value
 * can address which is 20-bits worth of addressing or 4GB (2^32) of space.
 */
#define MAX_EQ_LEN (1ull << 32)

/* Convert EQ queue length, in bytes, into queue size in ECB units. */
static size_t get_eq_queue_size(size_t queue_len)
{
	return queue_len / C_EE_CFG_ECB_SIZE;
}

void cass_ee_init(struct cass_dev *hw)
{
	union c_ee_cfg_timestamp_freq ts_freq;
	unsigned int clk_freq_khz;

	if (cass_version(hw, CASSINI_1))
		clk_freq_khz = C1_CLK_FREQ_HZ / 1000;
	else
		clk_freq_khz = C2_CLK_FREQ_HZ / 1000;

	ts_freq.clk_divider = ee_timestamp_period_ms * clk_freq_khz / C_NUM_EQS;

	cass_write(hw, C_EE_CFG_TIMESTAMP_FREQ, &ts_freq, sizeof(ts_freq));

	cass_write(hw, C_EE_CFG_LATENCY_MONITOR, &lat_monitor_def,
		   sizeof(lat_monitor_def));
}

/* Event interrupt handler for EQs */
static int eq_event_nb(struct notifier_block *nb, unsigned long action,
		       void *data)
{
	struct cxi_eq_priv *eq = container_of(nb, struct cxi_eq_priv,
					      event_nb);

	eq->event_cb(eq->event_cb_data);

	return NOTIFY_OK;
}

/* Status interrupt handler for EQs */
static int eq_status_nb(struct notifier_block *nb, unsigned long action,
			void *data)
{
	struct cxi_eq_priv *eq = container_of(nb, struct cxi_eq_priv,
					      status_nb);

	eq->status_cb(eq->status_cb_data);

	return NOTIFY_OK;
}

/* Return an EQ latency_tolerance value representing the requested nanosecond
 * delay according to the currently configured ECB latency monitor granularity.
 * Reading the currently configured value supports the expected behavior while
 * using OOB interfaces to configure the latency monitor.
 */
static unsigned int eq_compute_lt(struct cass_dev *hw, unsigned int ns)
{
	union c_ee_cfg_latency_monitor lat_mon;
	unsigned int step_size;
	unsigned int lt;

	cass_read(hw, C_EE_CFG_LATENCY_MONITOR, &lat_mon, sizeof(lat_mon));
	if (cass_version(hw, CASSINI_1))
		step_size = lat_mon.granularity / (C1_CLK_FREQ_HZ / NSEC_PER_SEC);
	else
		step_size = lat_mon.granularity / (C2_CLK_FREQ_HZ / NSEC_PER_SEC);
	lt = ns / step_size;

	if (lt > LT_LIMIT_MAX)
		lt = LT_LIMIT_MAX;

	return lt;
}

/* Toggle EQ timestamp events. Use eq_shadow_lock to serialize calls. */
static void eq_timestamp_toggle(struct cass_dev *hw, int eq_n, bool enable)
{
	union c_ee_cfg_periodic_tstamp_table ts_tbl;
	int ts_tbl_idx = eq_n / 64;
	int ts_tbl_bit = eq_n % 64;
	u64 csr_off = C_EE_CFG_PERIODIC_TSTAMP_TABLE(ts_tbl_idx);

	cass_read(hw, csr_off, &ts_tbl.qw, 8);
	if (enable)
		ts_tbl.n63_n0_enable_periodic_tstamp |= BIT(ts_tbl_bit);
	else
		ts_tbl.n63_n0_enable_periodic_tstamp &= ~BIT(ts_tbl_bit);
	cass_write(hw, csr_off, &ts_tbl.qw, 8);
}

/* Configure an EQ descriptor's status write-back settings given a queue size.
 * This must be done each time the EQ size is changed. eq_status_config() does
 * not write the new settings to hardware.
 */
static void eq_status_config(struct cxi_eq_priv *eq, size_t queue_size)
{
	unsigned int base_slots;
	unsigned int delta_slots;
	int thld_shift;

	if (eq->attr.status_thresh_count) {
		/* Don't count EQ status slot in queue size. */
		queue_size -= 1;
		/* Configure status update thresholds */
		base_slots = queue_size * (100 - eq->attr.status_thresh_base) / 100;
		delta_slots = queue_size * eq->attr.status_thresh_delta / 100;

		/* Set threshold shift to preserve the 6 MSBs of the larger of
		 * the base and delta values.
		 */
		thld_shift = (base_slots > delta_slots) ?
				fls(base_slots) : fls(delta_slots);
		thld_shift = thld_shift > 6 ? thld_shift - 6 : 0;

		eq->cfg.eq_sts_num_thld = eq->attr.status_thresh_count - 1;
		eq->cfg.eq_sts_thld_base = base_slots >> thld_shift;
		eq->cfg.eq_sts_thld_offst = delta_slots >> thld_shift;
		eq->cfg.eq_sts_thld_shift = thld_shift;
	} else {
		/* Max shift value disables threshold updates. */
		eq->cfg.eq_sts_thld_shift = 0x1f;
	}
}

#define EQ_MAX_RESERVED_FC (BIT(14) - 1)

static int is_valid_eq_reserved_fc(struct cxi_eq_priv *eq, int fc_value)
{
	int new_fc_value = (int)eq->cfg.reserved_fc + fc_value;
	int max_fc_value = eq->queue_size - 1;

	if (abs(fc_value) > EQ_MAX_RESERVED_FC ||
	    new_fc_value > EQ_MAX_RESERVED_FC || new_fc_value < 0)
		return -EINVAL;

	if (new_fc_value > max_fc_value)
		return -ENOSPC;

	return 0;
}

/**
 * cxi_eq_adjust_reserved_fc() - Adjust EQ reserved FC value.
 * @eq: EQ
 * @value: Amount to adjust reserved FC value by. Can be negative or positive.
 *
 * Return: On success, value greater than or equal to zero representing current
 * EQ reserved FC value. Else, -EINVAL if value is invalid or -ENOSPC if a valid
 * value is provided but could not be applied.
 */
int cxi_eq_adjust_reserved_fc(struct cxi_eq *eq, int value)
{
	struct cxi_eq_priv *eq_priv;
	struct cxi_dev *cdev;
	struct cass_dev *hw;
	int rc;

	if (!eq)
		return -EINVAL;

	eq_priv = container_of(eq, struct cxi_eq_priv, eq);
	cdev = eq_priv->lni_priv->dev;
	hw = container_of(cdev, struct cass_dev, cdev);

	/* Use EQ shadow_lock to serialize EQ access. */
	spin_lock(&hw->eq_shadow_lock);

	rc = is_valid_eq_reserved_fc(eq_priv, value);
	if (rc)
		goto unlock_out;

	eq_priv->cfg.reserved_fc += value;

	cass_shadow_write(hw, C_EE_BASE, C_EE_CFG_EQ_DESCRIPTOR(eq->eqn),
			  &eq_priv->cfg, sizeof(eq_priv->cfg));

	rc = eq_priv->cfg.reserved_fc;

unlock_out:
	spin_unlock(&hw->eq_shadow_lock);

	return rc;
}
EXPORT_SYMBOL(cxi_eq_adjust_reserved_fc);

/**
 * get_user_pages_contig() - Fault and pin contiguous user pages.
 *
 * If pages are contiguous, take a reference to each page and return zero.
 * Otherwise, return negative errno.
 *
 * @hw: The cassini device
 * @desc: EQ buffer descriptor
 */
static int get_user_pages_contig(struct cass_dev *hw, struct eq_buf_desc *desc)
{
	size_t npg = desc->events_len >> PAGE_SHIFT;
	struct page **pages;
	dma_addr_t pa;
	int rc;
	int i;

	pages = kmalloc_array(npg, sizeof(struct page *), GFP_KERNEL);
	if (!pages)
		return -ENOMEM;

	rc = pin_user_pages_fast((unsigned long)desc->events, npg, 1, pages);
	if (rc < 0)
		goto free_pages;
	else if (rc != npg)
		goto put_pages;

	/* Naively check that each page is in sequence. */
	pa = page_to_phys(pages[0]);
	for (i = 1; i < npg; i++) {
		pa += PAGE_SIZE;
		if (pa != page_to_phys(pages[i])) {
			rc = -EINVAL;
			goto put_pages;
		}
	}

	desc->pages = pages;
	desc->dma_addr = dma_map_page(&hw->cdev.pdev->dev,
				      pages[0], 0, desc->events_len,
				      DMA_FROM_DEVICE);
	if (dma_mapping_error(&hw->cdev.pdev->dev, desc->dma_addr)) {
		rc = -ENOMEM;
		goto put_pages;
	}

	return 0;

put_pages:
	for (i = 0; i < rc; i++)
		unpin_user_page(pages[i]);

free_pages:
	kfree(pages);

	return rc;
}

/**
 * put_pages_contig() - Un-pin a sequence of physical pages.
 *
 * @hw: The cassini device
 * @desc: EQ buffer descriptor
 */
static void put_pages_contig(struct cass_dev *hw, struct eq_buf_desc *desc)
{
	size_t npg = desc->events_len >> PAGE_SHIFT;

	dma_unmap_page(&hw->cdev.pdev->dev, desc->dma_addr, desc->events_len,
		       DMA_FROM_DEVICE);

	unpin_user_pages(desc->pages, npg);
	kfree(desc->pages);
}

/*
 * cxi_eq_resize() - Resize a CXI Event Queue buffer.
 *
 * Resizing an Event Queue is a multi-step process. The first step is to call
 * cxi_eq_resize() to pass a new event buffer to the device. After this call,
 * evtq will continue to reference the old EQ buffer. The device may write a
 * small number of events to the old EQ buffer followed by a special
 * C_EVENT_EQ_SWITCH event to indicate that hardware has transitioned to
 * writing to the new EQ buffer. When this event is detected, software must
 * call cxi_eq_resize_complete() in order to start reading events from the new
 * EQ buffer.
 *
 * The new event queue buffer must use the same translation mechanism as was
 * used to allocate the EQ. If translation is used, the Addressing Context (AC)
 * used by the new MD must match the MD used to allocate the EQ.
 *
 * The new event queue buffer must be cleared before calling cxi_eq_resize().
 *
 * @evtq: The Event Queue to resize.
 * @queue: The new event queue buffer. Must be page aligned.
 * @queue_len: The new event queue buffer length in bytes. Must be page aligned.
 * @queue_md: The new event queue memory descriptor.
 *
 * Return: On success, 0 is returned and a resized buffer has been
 * submitted to the device.
 */
int cxi_eq_resize(struct cxi_eq *evtq, void *queue, size_t queue_len,
		  struct cxi_md *queue_md)
{
	struct cxi_eq_priv *eq = container_of(evtq, struct cxi_eq_priv, eq);
	struct cxi_dev *cdev = eq->lni_priv->dev;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	struct cxi_md_priv *md_priv;
	int rc;
	int passthrough = !!(eq->attr.flags & CXI_EQ_PASSTHROUGH);

	if (!queue_len || queue_len > MAX_EQ_LEN || !queue)
		return -EINVAL;

	/* Queue buffer address and length must be page aligned. */
	if (!IS_ALIGNED((uintptr_t)queue, PAGE_SIZE))
		return -EINVAL;

	if (!IS_ALIGNED(queue_len, PAGE_SIZE))
		return -EINVAL;

	if (!passthrough) {
		if (!queue_md || !CXI_MD_CONTAINS(queue_md, queue, queue_len)) {
			cxidev_err(cdev, "MD does not contain queue buffer\n");
			return -EINVAL;
		}

		if (eq->active.md->lac != queue_md->lac) {
			cxidev_err(cdev, "New buffer AC does not match EQ\n");
			return -EINVAL;
		}
	}

	mutex_lock(&eq->resize_mutex);

	if (eq->resized) {
		rc = -EINVAL;
		goto unlock;
	}

	eq->resize.md = queue_md;
	eq->resize.events = queue;
	eq->resize.events_len = queue_len;

	if (passthrough) {
		if (eq->attr.flags & CXI_EQ_USER) {
			rc = get_user_pages_contig(hw, &eq->resize);
		} else {
			eq->resize.dma_addr = dma_map_single(&hw->cdev.pdev->dev,
							     queue, queue_len,
							     DMA_FROM_DEVICE);
			if (dma_mapping_error(&hw->cdev.pdev->dev, eq->resize.dma_addr))
				rc = -ENOMEM;
			else
				rc = 0;
		}

		if (rc)
			goto unlock;
	} else {
		eq->resize.dma_addr = CXI_VA_TO_IOVA(queue_md, queue);

		md_priv = container_of(queue_md, struct cxi_md_priv, md);
		refcount_inc(&md_priv->refcount);
	}

	if (eq->cfg.use_buffer_b) {
		eq->cfg.buffer_a_size = queue_len >> C_ADDR_SHIFT;
		eq->cfg.buffer_a_en   = 1;
		eq->cfg.buffer_a_addr = eq->resize.dma_addr >> C_ADDR_SHIFT;
		eq->cfg.use_buffer_b = 0;
	} else {
		eq->cfg.buffer_b_size = queue_len >> C_ADDR_SHIFT;
		eq->cfg.buffer_b_en   = 1;
		eq->cfg.buffer_b_addr = eq->resize.dma_addr >> C_ADDR_SHIFT;
		eq->cfg.use_buffer_b = 1;
	}

	/* Reconfigure EQ status settings using new queue length. */
	eq_status_config(eq, queue_len / C_EE_CFG_ECB_SIZE);

	/* Re-write the EQ descriptor with the new buffer description. Read
	 * back the descriptor to ensure it has been updated before returning
	 * to the client in order to guarantee that a subsequent client write
	 * to EQ_SW_STATE will force the resize to be processed by the device.
	 */
	spin_lock(&hw->eq_shadow_lock);
	cass_shadow_write(hw, C_EE_BASE, C_EE_CFG_EQ_DESCRIPTOR(eq->eq.eqn),
			  &eq->cfg, sizeof(eq->cfg));
	cass_shadow_read(hw, C_EE_BASE, C_EE_CFG_EQ_DESCRIPTOR(eq->eq.eqn),
			 &eq->cfg,
			 sizeof(eq->cfg));
	spin_unlock(&hw->eq_shadow_lock);

	eq->resized = true;

	mutex_unlock(&eq->resize_mutex);

	return 0;

unlock:
	mutex_unlock(&eq->resize_mutex);

	return rc;
}
EXPORT_SYMBOL(cxi_eq_resize);

/**
 * cxi_eq_resize_complete() - Complete resizing a CXI Event Queue buffer.
 *
 * cxi_eq_resize_complete() must be called after an EQ has been resized
 * using cxi_eq_resize() and a C_EVENT_EQ_SWITCH event was delivered. See
 * the documentation for cxi_eq_resize().
 *
 * @evtq: The Event Queue to being resized.
 *
 * Return: On success, 0 is returned and evtq references the new, resized
 * buffer.
 */
int cxi_eq_resize_complete(struct cxi_eq *evtq)
{
	struct cxi_eq_priv *eq = container_of(evtq, struct cxi_eq_priv, eq);
	struct cxi_dev *cdev = eq->lni_priv->dev;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	struct cxi_md_priv *md_priv;

	mutex_lock(&eq->resize_mutex);

	if (!eq->resized) {
		mutex_unlock(&eq->resize_mutex);
		return -EINVAL;
	}

	if (eq->attr.flags & CXI_EQ_PASSTHROUGH) {
		if (eq->attr.flags & CXI_EQ_USER)
			put_pages_contig(hw, &eq->active);
		else
			dma_unmap_single(&hw->cdev.pdev->dev, eq->active.dma_addr,
					 eq->active.events_len, DMA_FROM_DEVICE);
	} else {
		md_priv = container_of(eq->active.md, struct cxi_md_priv, md);
		refcount_dec(&md_priv->refcount);
	}

	eq->active = eq->resize;

	eq->attr.queue = eq->active.events;
	eq->attr.queue_len = eq->active.events_len;

	eq->queue_size = get_eq_queue_size(eq->active.events_len);

	/* Re-initialize the user evtq to reference new buffer */
	cxi_eq_init(evtq, eq->active.events, eq->active.events_len, eq->eq.eqn,
		    (__force u64 *)(hw->regs + C_MEMORG_EE +
				    eq->eq.eqn * C_EE_SW_STATE_PAGE_SIZE));
	evtq->sw_state.reading_buffer_b = !evtq->sw_state.reading_buffer_b;

	eq->resized = false;

	mutex_unlock(&eq->resize_mutex);

	return 0;
}
EXPORT_SYMBOL(cxi_eq_resize_complete);

/* Register the PCT EQ in all 3 places */
static void set_pct_eq(struct cass_dev *hw, unsigned int eq_n)
{
	union c_pct_cfg_eq_retry_q_handle rh_q = {
		.eq_handle = eq_n,
	};

	cass_write(hw, C_PCT_CFG_EQ_RETRY_Q_HANDLE, &rh_q, sizeof(rh_q));
	cass_write(hw, C_PCT_CFG_EQ_TGT_Q_HANDLE, &rh_q, sizeof(rh_q));
	cass_write(hw, C_PCT_CFG_EQ_CONN_LD_Q_HANDLE, &rh_q, sizeof(rh_q));
}

/* Return an available event queue with an ID */
static struct cxi_eq_priv *pf_get_eq_id(struct cxi_lni_priv *lni_priv)
{
	struct cxi_dev *cdev = lni_priv->dev;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	struct cxi_eq_priv *eq;
	int rc;

	/* Try to reuse an existing EQ that was put on the cleanup list */
	spin_lock(&lni_priv->res_lock);

	eq = list_first_entry_or_null(&lni_priv->eq_cleanups_list,
				      struct cxi_eq_priv, list);
	if (eq)
		list_del(&eq->list);

	spin_unlock(&lni_priv->res_lock);

	if (eq) {
		memset(&eq->cfg, 0, sizeof(eq->cfg));
		eq->reused = true;
		return eq;
	}

	/* Check the associated service to see if this EQ can be
	 * allocated.
	 */
	rc = cxi_rgroup_alloc_resource(lni_priv->rgroup, CXI_RESOURCE_EQ);
	if (rc)
		return ERR_PTR(rc);

	eq = kzalloc(sizeof(*eq), GFP_KERNEL);
	if (eq == NULL) {
		rc = -ENOMEM;
		goto dec_rsrc_use;
	}

	/* Get event queue ID */
	rc = ida_simple_get(&hw->eq_index_table, 1, C_NUM_EQS,
			    GFP_KERNEL);
	if (rc < 0) {
		cxidev_err(cdev, "ida_simple_get failed %d\n", rc);
		goto eq_free;
	}

	eq->eq.eqn = rc;

	return eq;

eq_free:
	kfree(eq);

dec_rsrc_use:
	cxi_rgroup_free_resource(lni_priv->rgroup, CXI_RESOURCE_EQ);

	return ERR_PTR(rc);
}

/* Free an EQ allocated with pf_get_eq_id */
static void pf_put_eq_id(struct cxi_eq_priv *eq)
{
	struct cxi_lni_priv *lni_priv = eq->lni_priv;
	struct cxi_dev *cdev = lni_priv->dev;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);

	if (eq->reused) {
		spin_lock(&lni_priv->res_lock);
		list_add_tail(&eq->list, &lni_priv->eq_cleanups_list);
		spin_unlock(&lni_priv->res_lock);
	} else {
		ida_simple_remove(&hw->eq_index_table, eq->eq.eqn);
		cxi_rgroup_free_resource(lni_priv->rgroup, CXI_RESOURCE_EQ);
		kfree(eq);
	}
}

/* Prepare and write the EQ config to the adapter */
static int pf_write_eq_config(struct cxi_eq_priv *eq)
{
	struct cxi_dev *cdev = eq->lni_priv->dev;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	union c_ee_cfg_init_eq_hw_state init_eq_hw_state;
	unsigned int eq_n = eq->eq.eqn;
	const struct cxi_eq_attr *attr = &eq->attr;
	void __iomem *csr;
	int rc;

	if (eq->event_cb) {
		eq->cfg.event_int_idx = eq->event_msi_irq->idx;
		eq->cfg.event_int_en = 1;
	}

	if (eq->status_cb) {
		eq->cfg.eq_sts_int_idx = eq->status_msi_irq->idx;
		eq->cfg.eq_sts_int_en = 1;
	}

	if (eq->attr.flags & CXI_EQ_PASSTHROUGH) {
		eq->cfg.acid = ATU_PHYS_AC;
		eq->cfg.buffer_a_addr = eq->active.dma_addr >> C_ADDR_SHIFT;
	} else {
		struct cxi_md_priv *md_priv =
			container_of(eq->active.md, struct cxi_md_priv, md);

		eq->cfg.acid = md_priv->cac->ac.acid;
		eq->cfg.buffer_a_addr =
			CXI_VA_TO_IOVA(eq->active.md, eq->active.events) >>
			C_ADDR_SHIFT;
	}

	eq->cfg.buffer_a_size = eq->active.events_len >> C_ADDR_SHIFT;
	eq->cfg.buffer_a_en = 1;
	eq->cfg.eq_enable = 1;

	rc = is_valid_eq_reserved_fc(eq, eq->attr.reserved_slots);
	if (rc)
		return rc;
	eq->cfg.reserved_fc = eq->attr.reserved_slots;

	eq_status_config(eq, eq->queue_size);

	eq->cfg.eq_sts_dropped_en =
		eq->attr.flags & CXI_EQ_DROP_STATUS_DISABLE ? 0 : 1;

	if (eq->attr.flags & CXI_EQ_EC_DISABLE) {
		/* Disable all event combining. */
		eq->cfg.latency_tolerance = 0;
	} else if (attr->ec_delay) {
		/* Set custom ECB expiration delay. */
		eq->cfg.latency_tolerance = eq_compute_lt(hw, attr->ec_delay);
	} else {
		/* Use default ECB expiration configuration. */
		eq->cfg.latency_tolerance =
			eq_compute_lt(hw, LATENCY_TOLERANCE_DEF_NS);
	}

	mutex_lock(&hw->init_eq_hw_state);

	/* Wait for C_EE_CFG_INIT_EQ_HW_STATE to be ready for use */
	csr = cass_csr(hw, C_EE_CFG_INIT_EQ_HW_STATE);
	rc = readq_poll_timeout(csr, init_eq_hw_state.qw,
				init_eq_hw_state.pending == 0, 1, 1000);
	if (rc) {
		cxidev_warn_once(cdev, "C_EE_CFG_INIT_EQ_HW_STATE was still pending.\n");
		mutex_unlock(&hw->init_eq_hw_state);
		return rc;
	}

	/* Ensure no other PCT EQ exist before activating it. */
	if (attr->flags & CXI_EQ_REGISTER_PCT) {
		if (hw->pct_eq_n != C_EQ_NONE) {
			mutex_unlock(&hw->init_eq_hw_state);
			return -EEXIST;
		}

		hw->pct_eq_n = eq_n;
	}

	/* Write EQ configuration to hardware */
	spin_lock(&hw->eq_shadow_lock);

	cass_eq_init(hw, eq_n, eq->lni_priv->lni.rgid, &eq->cfg,
		     attr->flags & CXI_EQ_INIT_LONG,
		     attr->flags & CXI_EQ_TGT_LONG);

	if (attr->flags & CXI_EQ_TIMESTAMP_EVENTS)
		eq_timestamp_toggle(hw, eq_n, true);

	spin_unlock(&hw->eq_shadow_lock);

	if (attr->flags & CXI_EQ_REGISTER_PCT) {
		/* The EQ is reserved, and up and running now. The PCT
		 * EQ can be programmed.
		 */
		set_pct_eq(hw, eq_n);
	}

	mutex_unlock(&hw->init_eq_hw_state);

	return 0;
}

/**
 * cxi_eq_alloc() - Allocate a new event queue resource.
 *
 * @lni: LNI to associate with the event queue.
 * @md: Memory descriptor for the queue
 * @attr: Event queue creation attributes.
 * @event_cb: Event interrupt callback for that EQ.
 * @event_cb_data: Opaque owner context pointer. Can be used in the event
 *                 interrupt callback to retrieve some internal structures.
 * @status_cb: Status interrupt callback for that EQ.
 * @status_cb_data: Opaque owner context pointer. Can be used in the status
 *                  interrupt callback to retrieve some internal structures.
 *
 * Return: On success, a pointer to the allocated event queue structure. On
 * error, an error pointer containing a negative error number.
 */
struct cxi_eq *cxi_eq_alloc(struct cxi_lni *lni, const struct cxi_md *md,
			    const struct cxi_eq_attr *attr,
			    void (*event_cb)(void *cb_data),
			    void *event_cb_data,
			    void (*status_cb)(void *cb_data),
			    void *status_cb_data)
{
	struct cxi_lni_priv *lni_priv =
		container_of(lni, struct cxi_lni_priv, lni);
	struct cxi_dev *cdev;
	struct cass_dev *hw;
	struct cxi_eq_priv *eq;
	struct cxi_md_priv *md_priv = NULL;
	int eq_n;
	int rc;
	int passthrough = !!(attr->flags & CXI_EQ_PASSTHROUGH);
	bool is_user = attr->flags & CXI_EQ_USER;
	struct cpumask cpus = {};
	struct cass_irq *irq;

	if (!attr->queue_len || attr->queue_len > MAX_EQ_LEN || !attr->queue)
		return ERR_PTR(-EINVAL);

	/* If status updates are enabled, ensure all requested updates are in
	 * the range [0-100].
	 */
	if (attr->status_thresh_count &&
	    (attr->status_thresh_count > 4 ||
	     attr->status_thresh_base > 100 ||
	     attr->status_thresh_delta > 100 ||
	     ((attr->status_thresh_count - 1) * attr->status_thresh_delta +
	      (100 - attr->status_thresh_base)) > 100))
		return ERR_PTR(-EINVAL);

	/* Queue buffer address and length must be page aligned. */
	if (!IS_ALIGNED((uintptr_t)attr->queue, PAGE_SIZE))
		return ERR_PTR(-EINVAL);

	if (!IS_ALIGNED(attr->queue_len, PAGE_SIZE))
		return ERR_PTR(-EINVAL);

	/* If using EQ translation, the MD must cover the aligned buffer. */
	if (!passthrough &&
	    (!md ||
	     !CXI_MD_CONTAINS(md, attr->queue, attr->queue_len)))
		return ERR_PTR(-EINVAL);

	/* Only admin can create the EQ for PCT */
	if ((attr->flags & CXI_EQ_REGISTER_PCT) && !capable(CAP_SYS_ADMIN))
		return ERR_PTR(-EPERM);

	if ((size_t)attr->reserved_slots >
	    (get_eq_queue_size(attr->queue_len) - 1))
		return ERR_PTR(-EINVAL);

	cdev = lni_priv->dev;
	hw = container_of(cdev, struct cass_dev, cdev);

	eq = pf_get_eq_id(lni_priv);
	if (IS_ERR(eq))
		return ERR_PTR(PTR_ERR(eq));

	eq_n = eq->eq.eqn;

	eq->attr = *attr;
	eq->lni_priv = lni_priv;
	eq->active.md = md;
	eq->active.events = eq->attr.queue;
	eq->active.events_len = eq->attr.queue_len;
	eq->queue_size = get_eq_queue_size(eq->active.events_len);
	mutex_init(&eq->resize_mutex);
	eq->resized = false;

	refcount_set(&eq->refcount, 1);

	if (!is_user) {
		phys_addr_t mmio_phys = hw->regs_base + C_MEMORG_EE +
				eq_n * C_EE_SW_STATE_PAGE_SIZE;
		eq->eq_mmio = ioremap(mmio_phys, PAGE_SIZE);
		if (!eq->eq_mmio) {
			cxidev_warn_once(cdev, "ioremap failed\n");
			rc = -EFAULT;
			goto id_remove;
		}
	}

	if (eq->attr.cpu_affinity < nr_cpumask_bits)
		cpumask_set_cpu(eq->attr.cpu_affinity, &cpus);

	if (event_cb) {
		eq->event_cb = event_cb;
		eq->event_cb_data = event_cb_data;
		eq->event_nb.notifier_call = eq_event_nb;

		irq = cass_comp_irq_attach(hw, &cpus, &eq->event_nb);
		if (IS_ERR(irq)) {
			rc = PTR_ERR(irq);
			goto eq_unmap;
		}
		eq->event_msi_irq = irq;
	} else {
		eq->event_msi_irq = NULL;
	}

	if (status_cb) {
		eq->status_cb = status_cb;
		eq->status_cb_data = status_cb_data;
		eq->status_nb.notifier_call = eq_status_nb;

		irq = cass_comp_irq_attach(hw, &cpus, &eq->status_nb);
		if (IS_ERR(irq)) {
			rc = PTR_ERR(irq);
			goto event_irq_detach;
		}
		eq->status_msi_irq = irq;
	} else {
		eq->status_msi_irq = NULL;
	}

	if (passthrough) {
		if (is_user) {
			rc = get_user_pages_contig(hw, &eq->active);
		} else {
			eq->active.dma_addr = dma_map_single(&hw->cdev.pdev->dev,
							     eq->active.events,
							     eq->active.events_len,
							     DMA_FROM_DEVICE);
			if (dma_mapping_error(&hw->cdev.pdev->dev, eq->active.dma_addr))
				rc = -ENOMEM;
			else
				rc = 0;
		}

		if (rc)
			goto status_irq_detach;
	} else {
		md_priv = container_of(eq->active.md, struct cxi_md_priv, md);

		refcount_inc(&md_priv->refcount);
	}

	cxi_eq_init(&eq->eq, eq->active.events, eq->active.events_len,
		    eq_n, (__force u64 *)eq->eq_mmio);

	if (!is_user)
		memset(eq->active.events, 0, eq->active.events_len);

	rc = pf_write_eq_config(eq);
	if (rc)
		goto free_md;

	spin_lock(&lni_priv->res_lock);
	list_add_tail(&eq->list, &lni_priv->eq_list);
	spin_unlock(&lni_priv->res_lock);

	if (!eq->reused) {
		atomic_inc(&hw->stats.eq);

		eq_debugfs_create(eq_n, eq, hw, lni_priv);

		refcount_inc(&lni_priv->refcount);
	}

	return &eq->eq;

free_md:
	if (passthrough) {
		if (is_user)
			put_pages_contig(hw, &eq->active);
		else
			dma_unmap_single(&hw->cdev.pdev->dev, eq->active.dma_addr,
					 eq->attr.queue_len, DMA_FROM_DEVICE);
	} else {
		refcount_dec(&md_priv->refcount);
	}

status_irq_detach:
	if (eq->status_msi_irq)
		cass_comp_irq_detach(hw, eq->status_msi_irq, &eq->status_nb);
event_irq_detach:
	if (eq->event_msi_irq)
		cass_comp_irq_detach(hw, eq->event_msi_irq, &eq->event_nb);
eq_unmap:
	if (!is_user)
		iounmap(eq->eq_mmio);
id_remove:
	pf_put_eq_id(eq);

	return ERR_PTR(rc);
}
EXPORT_SYMBOL(cxi_eq_alloc);

/**
 * cxi_eq_free() - Free event queue resource.
 *
 * @evtq: A pointer to previously allocated event queue structure.
 *
 * Return: On success, returns zero.
 */
int cxi_eq_free(struct cxi_eq *evtq)
{
	struct cxi_eq_priv *eq;
	struct cxi_lni_priv *lni_priv;
	struct cxi_dev *cdev;
	struct cass_dev *hw;
	struct cxi_md_priv *md_priv;
	bool is_user;
	union c_ee_cfg_sts_eq_hw_state hw_state;
	unsigned int wr_ptr;

	if (!evtq)
		return -EINVAL;

	eq = container_of(evtq, struct cxi_eq_priv, eq);
	lni_priv = eq->lni_priv;
	cdev = lni_priv->dev;
	hw = container_of(cdev, struct cass_dev, cdev);
	is_user = eq->attr.flags & CXI_EQ_USER;

	cxidev_WARN_ONCE(cdev, !refcount_dec_and_test(&eq->refcount),
			 "Resource leaks - EQ refcount not zero: %d\n",
			 refcount_read(&eq->refcount));

	spin_lock(&lni_priv->res_lock);
	list_del(&eq->list);
	spin_unlock(&lni_priv->res_lock);

	if (eq->attr.flags & CXI_EQ_REGISTER_PCT) {
		set_pct_eq(hw, C_EQ_NONE);

		mutex_lock(&hw->init_eq_hw_state);
		hw->pct_eq_n = C_EQ_NONE;
		mutex_unlock(&hw->init_eq_hw_state);
	}

	if (eq->status_msi_irq)
		cass_comp_irq_detach(hw, eq->status_msi_irq, &eq->status_nb);

	if (eq->event_msi_irq)
		cass_comp_irq_detach(hw, eq->event_msi_irq, &eq->event_nb);

	spin_lock(&hw->eq_shadow_lock);

	if (eq->attr.flags & CXI_EQ_TIMESTAMP_EVENTS)
		eq_timestamp_toggle(hw, eq->eq.eqn, false);

	cass_eq_clear(hw, &eq->eq);

	spin_unlock(&hw->eq_shadow_lock);

	/* Wait for the EQ write pointer to stabilize before freeing
	 * the buffer
	 */
	cass_read(hw, C_EE_CFG_STS_EQ_HW_STATE(eq->eq.eqn),
		  &hw_state, sizeof(hw_state));
	wr_ptr = hw_state.wr_ptr;
	while (true) {
		cass_read(hw, C_EE_CFG_STS_EQ_HW_STATE(eq->eq.eqn),
			  &hw_state, sizeof(hw_state));
		if (wr_ptr == hw_state.wr_ptr)
			break;
		wr_ptr = hw_state.wr_ptr;
		udelay(1);
	}

	if (eq->attr.flags & CXI_EQ_PASSTHROUGH) {
		if (eq->attr.flags & CXI_EQ_USER) {
			put_pages_contig(hw, &eq->active);

			if (eq->resized)
				put_pages_contig(hw, &eq->resize);
		} else {
			dma_unmap_single(&hw->cdev.pdev->dev,
					 eq->active.dma_addr, eq->attr.queue_len,
					 DMA_FROM_DEVICE);
			if (eq->resized)
				dma_unmap_single(&hw->cdev.pdev->dev,
						 eq->resize.dma_addr,
						 eq->resize.events_len,
						 DMA_FROM_DEVICE);
		}
	} else {
		md_priv = container_of(eq->active.md, struct cxi_md_priv, md);
		refcount_dec(&md_priv->refcount);

		if (eq->resized) {
			md_priv = container_of(eq->resize.md,
					       struct cxi_md_priv, md);
			refcount_dec(&md_priv->refcount);
		}
	}

	if (!is_user)
		iounmap(eq->eq_mmio);

	/* Hang the EQ on the LNI pending deletion list, to be
	 * released later
	 */
	spin_lock(&lni_priv->res_lock);
	list_add_tail(&eq->list, &lni_priv->eq_cleanups_list);
	spin_unlock(&lni_priv->res_lock);

	return 0;
}
EXPORT_SYMBOL(cxi_eq_free);

/* Finish releasing all the EQs on the deletion pending list */
void finalize_eq_cleanups(struct cxi_lni_priv *lni)
{
	struct cxi_dev *dev = lni->dev;
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);
	struct cxi_eq_priv *eq;

	while ((eq = list_first_entry_or_null(&lni->eq_cleanups_list,
					      struct cxi_eq_priv, list))) {
		list_del(&eq->list);

		debugfs_remove(eq->lni_dir);
		debugfs_remove(eq->debug_file);

		refcount_dec(&lni->refcount);
		atomic_dec(&hw->stats.eq);
		cxi_rgroup_free_resource(lni->rgroup, CXI_RESOURCE_EQ);
		ida_simple_remove(&hw->eq_index_table, eq->eq.eqn);
		kfree(eq);
	}
}

/**
 * cxi_eq_user_info() - Get userspace mmap info for an event queue.
 *
 * @evtq: A pointer to previously allocated event queue structure.
 * @csr_addr: On success, will contain the address of the event queue CSRs.
 * @csr_size: On success, will contain the size of the event queue CSRs.
 *
 * Return: On success, returns zero. On error, returns a negative error number.
 */
int cxi_eq_user_info(struct cxi_eq *evtq,
		     phys_addr_t *csr_addr, size_t *csr_size)
{
	struct cxi_eq_priv *eq = container_of(evtq, struct cxi_eq_priv, eq);
	struct cxi_dev *cdev = eq->lni_priv->dev;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);

	/* Userspace needs the SW State descriptor CSR address */
	*csr_addr = hw->regs_base + C_MEMORG_EE +
		(eq->eq.eqn * C_EE_SW_STATE_PAGE_SIZE);
	*csr_size = PAGE_SIZE;

	return 0;
}
EXPORT_SYMBOL(cxi_eq_user_info);
