// SPDX-License-Identifier: GPL-2.0
/* Copyright 2020-2021,2025 Hewlett Packard Enterprise Development LP */
/* Cassini pseudo PHY interface
 * Copy the interface from drivers/net/phy/phy.c
 */

#include <linux/debugfs.h>
#include <linux/iopoll.h>
#include <linux/hpe/cxi/cxi.h>

#include "cass_core.h"
#include "cass_sbl.h"
#include <linux/sbl.h>

/**
 * cass_phy_is_started() - Convenience function to check whether PHY is started
 *
 * @hw: Cassini device.
 */
static bool cass_phy_is_started(struct cass_dev *hw)
{
	return hw->phy.state >= CASS_PHY_UP;
}

bool cass_phy_is_headshell_removed(struct cass_dev *hw)
{
	return hw->phy.state == CASS_PHY_HEADSHELL_REMOVED;
}

void cass_phy_set_state(enum cass_phy_state state, struct cass_dev *hw)
{
	spin_lock(&hw->phy.lock);
	hw->phy.state = state;
	spin_unlock(&hw->phy.lock);
}

static void cass_phy_queue_state_machine(struct cass_dev *hw,
					 unsigned long jiffies)
{
	bool is_started;

	spin_lock(&hw->phy.lock);
	is_started = cass_phy_is_started(hw);
	spin_unlock(&hw->phy.lock);

	if (!is_started)
		return;

	if (!(work_busy(&hw->phy.state_queue.work) & WORK_BUSY_PENDING))
		mod_delayed_work(system_power_efficient_wq, &hw->phy.state_queue, jiffies);
}

void cass_phy_trigger_machine(struct cass_dev *hw)
{
	cass_phy_queue_state_machine(hw, 0);
}

void cass_phy_link_up(struct cass_dev *hw)
{
	cxidev_warn(&hw->cdev, "CXI_EVENT_LINK_UP");
	update_hni_link_up(hw);
	update_oxe_link_up(hw);
	cass_set_outstanding_limit(hw);
	cxi_send_async_event(&hw->cdev, CXI_EVENT_LINK_UP);
}

void cass_phy_link_down(struct cass_dev *hw)
{
	cxidev_warn(&hw->cdev, "CXI_EVENT_LINK_DOWN");
	cxi_send_async_event(&hw->cdev, CXI_EVENT_LINK_DOWN);
}

static void cass_phy_check_link_status(struct cass_dev *hw)
{
	enum cass_link_status state = cass_link_get_state(hw);

	if ((state == CASS_LINK_STATUS_UP) &&
	    (hw->phy.state != CASS_PHY_RUNNING)) {
		// TODO - phy.state and lstate duplicate one another
		hw->phy.state = CASS_PHY_RUNNING;
		cass_phy_link_up(hw);
	} else if ((state != CASS_LINK_STATUS_UP) &&
		   (hw->phy.state != CASS_PHY_NOLINK)) {
		hw->phy.state = CASS_PHY_NOLINK;
		cass_phy_link_down(hw);
	}
}

/* cass_phy_start_aneg - start auto-negotiation for the device */
static void cass_phy_start_aneg(struct cass_dev *hw)
{
	bool is_started;

	cass_lmon_request_up(hw);

	spin_lock(&hw->phy.lock);
	is_started = cass_phy_is_started(hw);
	spin_unlock(&hw->phy.lock);

	if (is_started)
		cass_phy_check_link_status(hw);
}

static void cass_phy_state_machine(struct work_struct *work)
{
	struct delayed_work *dwork = to_delayed_work(work);
	struct cass_dev *hw =
		container_of(dwork, struct cass_dev, phy.state_queue);
	enum cass_phy_state old_state;
	enum cass_phy_state new_state;
	bool needs_aneg = false;

	if (atomic_read(&hw->phy.cancel_state_machine) == 1)
		return;

	spin_lock(&hw->phy.lock);
	old_state = hw->phy.state;
	spin_unlock(&hw->phy.lock);

	switch (old_state) {
	case CASS_PHY_DOWN:
	case CASS_PHY_READY:
	case CASS_PHY_HEADSHELL_REMOVED:
		break;

	case CASS_PHY_UP:
		needs_aneg = true;
		break;

	case CASS_PHY_NOLINK:
	case CASS_PHY_RUNNING:
		cass_lmon_request_up(hw);
		cass_phy_check_link_status(hw);
		break;
	case CASS_PHY_HALTED:
		if (cass_link_get_state(hw) == CASS_LINK_STATUS_UP)
			cass_phy_link_down(hw);
		break;
	}

	if (needs_aneg)
		cass_phy_start_aneg(hw);

	if (old_state != hw->phy.state) {
		cxidev_dbg(&hw->cdev, "PHY state change %d -> %d\n",
			   old_state, hw->phy.state);
	}

	spin_lock(&hw->phy.lock);
	new_state = hw->phy.state;
	spin_unlock(&hw->phy.lock);

	if (new_state != CASS_PHY_RUNNING)
		cass_phy_queue_state_machine(hw, HZ);
}

void cass_phy_bounce(struct cass_dev *hw)
{
	cass_phy_stop(hw, true);
	cass_phy_start(hw, false);
}

void cass_phy_start(struct cass_dev *hw, bool force_reconfig)
{
	atomic_set(&hw->phy.cancel_state_machine, 0);

	/* Start the PHY directly in UP mode */
	INIT_DELAYED_WORK(&hw->phy.state_queue, cass_phy_state_machine);

	if (hw->link_config_dirty || force_reconfig)
		cass_sbl_configure(hw);

	if (hw->phy.state != CASS_PHY_HEADSHELL_REMOVED)
		hw->phy.state = CASS_PHY_UP;

	cass_phy_queue_state_machine(hw, 0);
}

void cass_phy_stop(struct cass_dev *hw, bool block)
{
	int err;

	atomic_set(&hw->phy.cancel_state_machine, 1);

	cancel_delayed_work_sync(&hw->phy.state_queue);

	spin_lock(&hw->phy.lock);
	if (cass_phy_is_started(hw))
		hw->phy.state = CASS_PHY_UP;
	spin_unlock(&hw->phy.lock);

	if (!block)
		err = cass_lmon_request_down(hw);
	else
		err = cass_link_async_down_wait(hw, CASS_DOWN_ORIGIN_CONFIG);

	if (err || (cass_link_get_state(hw) == CASS_LINK_STATUS_ERROR)) {
		cass_link_async_reset_wait(hw, CASS_DOWN_ORIGIN_CONFIG);
		cass_sbl_configure(hw);
		hw->phy.state = CASS_PHY_UP;
	}
}
