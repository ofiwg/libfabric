// SPDX-License-Identifier: GPL-2.0
/* Copyright 2020,2025 Hewlett Packard Enterprise Development LP */

#include <linux/types.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/device.h>
#include <linux/workqueue.h>
#include <linux/delay.h>
#include <linux/sbl.h>

#include "cass_core.h"

/*
 * The link has asynchronously gone down
 *
 *   i.e. the fabric LSM has gone down, the base link has gone down or
 *        the headshell/media removed
 *
 */
void cass_link_async_down(struct cass_dev *hw, u32 origin)
{
	cxidev_dbg(&hw->cdev,
		   "async link down, origin %s, (current dirn %s, origin %s)\n",
		   cass_link_down_origin_str(origin),
		   cass_lmon_dirn_str(cass_lmon_get_dirn(hw)),
		   cass_link_down_origin_str(cass_link_get_down_origin(hw)));

	if (cass_version(hw, CASSINI_1))
		atomic_inc(&hw->sbl_counters[link_async_down_tries]);

	if (cass_lmon_get_dirn(hw) == CASS_LMON_DIRECTION_RESET) {
		/*
		 * Do nothing
		 * This will take us down anyway and is only cleared
		 * when reset had completed
		 */
		return;
	}

	spin_lock(&hw->port->lock);
	if (hw->port->link_down_origin != CASS_DOWN_ORIGIN_CMD) {
		/*
		 * We are executing a command don't overwrite origin
		 * as the link keep-up code requires it
		 */
		hw->port->link_down_origin = origin;
	}
	spin_unlock(&hw->port->lock);
	if (cass_version(hw, CASSINI_1))
		atomic_inc(&hw->sbl_counters[link_async_down]);
	cass_lmon_request_down(hw);
}

/*
 * blocking async link down
 *
 *   wait until the link is in a suitable state ("down") for the
 *   media to be changed
 *
 *   we would link to use a completion here but its too complicated so
 *   will simply poll for it to be in a "down" state.
 */
int cass_link_async_down_wait(struct cass_dev *hw, u32 origin)
{
	unsigned long timeout = jiffies +
		msecs_to_jiffies(CASS_LINK_ASYNC_DOWN_TIMEOUT);

	cxidev_dbg(&hw->cdev, "async down wait - origin %s\n",
		   cass_link_down_origin_str(origin));

	if (cass_version(hw, CASSINI_1))
		atomic_inc(&hw->sbl_counters[link_async_down_tries]);

	/* stop the link if its coming up */
	spin_lock(&hw->port->lock);
	switch (hw->port->lstate) {

	case CASS_LINK_STATUS_STARTING:
	case CASS_LINK_STATUS_UP:
		hw->port->link_down_origin = origin;
		spin_unlock(&hw->port->lock);
		cass_lmon_request_down(hw);
		break;

	default:
		spin_unlock(&hw->port->lock);
	}

	/* wait for it to get to a "down" state */
	do {
		spin_lock(&hw->port->lock);
		switch (hw->port->lstate) {

		case CASS_LINK_STATUS_DOWN:
		case CASS_LINK_STATUS_UNCONFIGURED:
		case CASS_LINK_STATUS_ERROR:
		case CASS_LINK_STATUS_UNKNOWN:
			cxidev_dbg(&hw->cdev, "async down wait done (%s)\n",
				   cass_link_state_str(hw->port->lstate));
			if (cass_version(hw, CASSINI_1))
				atomic_inc(&hw->sbl_counters[link_async_down]);
			spin_unlock(&hw->port->lock);
			return 0;

		default:
			spin_unlock(&hw->port->lock);
			msleep(CASS_LINK_ASYNC_DOWN_INTERVAL);
			break;
		}
	} while time_is_after_jiffies(timeout);

	cxidev_err(&hw->cdev, "async down wait timed out (%s)\n",
		   cass_link_state_str(hw->port->lstate));
	return -ETIMEDOUT;
}

/*
 * blocking async link reset
 *
 *   block until the link reset completes
 *
 *   we would link to use a completion here but its too complicated so
 *   will simply poll for it to be in an "unconfigured" state.
 */
int cass_link_async_reset_wait(struct cass_dev *hw, u32 origin)
{
	unsigned long timeout = jiffies +
		msecs_to_jiffies(CASS_LINK_ASYNC_RESET_TIMEOUT);

	cxidev_dbg(&hw->cdev, "async reset wait - origin %s\n",
		   cass_link_down_origin_str(origin));

	/* stop the link if its coming up */
	spin_lock(&hw->port->lock);
	switch (hw->port->lstate) {

	case CASS_LINK_STATUS_STARTING:
	case CASS_LINK_STATUS_UP:
	case CASS_LINK_STATUS_STOPPING:
	case CASS_LINK_STATUS_DOWN:
	case CASS_LINK_STATUS_ERROR:
	case CASS_LINK_STATUS_UNCONFIGURED:
		hw->port->link_down_origin = origin;
		spin_unlock(&hw->port->lock);
		cass_lmon_request_reset(hw);
		break;

	default:
		spin_unlock(&hw->port->lock);
	}

	/* wait for it to get to an "unconfigured" state */
	do {
		spin_lock(&hw->port->lock);
		switch (hw->port->lstate) {

		case CASS_LINK_STATUS_UNCONFIGURED:
			cxidev_dbg(&hw->cdev, "async reset wait done (%s)\n",
				   cass_link_state_str(hw->port->lstate));
			spin_unlock(&hw->port->lock);
			return 0;

		default:
			spin_unlock(&hw->port->lock);
			msleep(CASS_LINK_ASYNC_RESET_INTERVAL);
			break;
		}
	} while time_is_after_jiffies(timeout);

	cxidev_err(&hw->cdev, "async reset wait timed out (%s)\n",
		   cass_link_state_str(hw->port->lstate));
	return -ETIMEDOUT;
}

/*
 * print out link state, info etc for sysfs diags
 *
 */
int cass_link_sysfs_sprint(struct cass_dev *hw, char *buf, size_t size)
{
	int rc;
	int state;

	spin_lock(&hw->port->lock);
	state = hw->port->lstate;
	spin_unlock(&hw->port->lock);

	rc = scnprintf(buf, size, "%s", cass_link_state_str(state));

	return rc;
}
