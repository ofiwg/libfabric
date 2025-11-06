// SPDX-License-Identifier: GPL-2.0
/*
 * Cassini link monitor
 * Copyright 2020-2024,2025 Hewlett Packard Enterprise Development LP
 */

#include <linux/iopoll.h>
#include <linux/hpe/cxi/cxi.h>
#include <linux/kthread.h>
#include <sbl/sbl_mb.h>
#include <linux/sbl.h>
#include <uapi/sbl.h>
#include <uapi/sbl_serdes.h>

#include "cass_core.h"
#include "cassini-telemetry-items.h"

/*
 * make sure code is aligned with address map
 */
#if (CASS_PCS_NUM_FECL_CNTRS != 8)
#error fecl counter number mismatch
#endif

/* FEC monitoring */
#define CASS_LMON_WAKEUP_INTERVAL                 1000    /* ms */
#define CASS_LMON_MAX_FEC_WARNINGS                   3
#define CASS_LMON_MIN_FEC_WINDOW                  1000    /* ms */

/* wakeup rate limiting */
#define CASS_LMON_MAX_WAKEUPS                       10
#define CASS_LMON_WAKEUP_DELAY                     100    /* ms */
#define CASS_LMON_WAKEUP_LIMITER                   500    /* ms */
#define CASS_LMON_WAKEUP_WINDOW                   1000    /* ms */

/* thread init data */
struct cass_lmon_data {
	struct cass_dev *hw;
	struct completion running;
};

/*
 * test to see if link should be up
 *
 * This function is called with the port locked
 */
static bool cass_lmon_link_should_auto_restart(struct cass_dev *hw)
{
	if (kthread_should_stop())
		return false;

	if (hw->port->link_down_origin == CASS_DOWN_ORIGIN_CONFIG) {
		/* link has been told to come down */
		return false;
	}

	if (!(hw->port->lattr.options & CASS_LINK_OPT_UP_AUTO_RESTART)) {
		/* auto restart not configured */
		return false;
	}

	if (hw->port->lmon_up_pause) {
		/* auto restart has been paused */
		return false;
	}

	/* should auto restart */
	return true;
}

/*
 * test to see if monitor should wake up and do something
 */
static bool cass_lmon_wakeup(struct cass_dev *hw)
{
	int lmon_dirn;
	int lstate;
	bool auto_restart;

	if (kthread_should_stop())
		return true;

	spin_lock(&hw->port->lock);
	lmon_dirn = hw->port->lmon_dirn;
	lstate = hw->port->lstate;
	auto_restart = cass_lmon_link_should_auto_restart(hw);
	spin_unlock(&hw->port->lock);

	if (lmon_dirn == CASS_LMON_DIRECTION_RESET) {
		/* always do reset */
		return true;
	} else if (lstate == CASS_LINK_STATUS_ERROR) {
		/* error can generally only be cleared by a reset */
		return false;
	} else if (lstate == CASS_LINK_STATUS_UNCONFIGURED) {
		/* can't do anything without configuration */
		return false;
	} else if ((lmon_dirn == CASS_LMON_DIRECTION_UP &&
		    lstate != CASS_LINK_STATUS_UP) ||
		   (lmon_dirn == CASS_LMON_DIRECTION_DOWN &&
		    (lstate != CASS_LINK_STATUS_DOWN || auto_restart))) {
		/* if we should be up or down and not, then there is
		 *  something to do
		 */
		return true;
	}

	return false;
}

static void cass_lmon_wakeup_limiter_reset(struct cass_dev *hw)
{
	spin_lock(&hw->port->lock);
	hw->port->lmon_wake_cnt = 0;
	hw->port->lmon_wake_jiffies = jiffies +
		msecs_to_jiffies(CASS_LMON_WAKEUP_WINDOW);
	hw->port->lmon_limiter_on = false;
	spin_unlock(&hw->port->lock);
}

/*
 * make sure we are not waking too frequently
 */
static void cass_lmon_wakeup_limiter(struct cass_dev *hw)
{
	spin_lock(&hw->port->lock);
	if (time_is_after_jiffies(hw->port->lmon_wake_jiffies)) {
		if (++hw->port->lmon_wake_cnt < CASS_LMON_MAX_WAKEUPS) {
			spin_unlock(&hw->port->lock);
			msleep(CASS_LMON_WAKEUP_DELAY);
			cxidev_dbg(&hw->cdev, "lmon wakeup limiter - %d\n",
				   hw->port->lmon_wake_cnt);
			return;
		}

		cxidev_dbg(&hw->cdev, "lmon wakeup limiter - sleeping\n");
		hw->port->lmon_limiter_on = true;
		spin_unlock(&hw->port->lock);
		atomic_inc(&hw->port->lmon_counters[lmon_limiter]);
		msleep(CASS_LMON_WAKEUP_LIMITER);
		return;
	}
	spin_unlock(&hw->port->lock);
	cass_lmon_wakeup_limiter_reset(hw);
}

/*
 * called from thread other than the link monitor
 */
int cass_lmon_request_up(struct cass_dev *hw)
{
	bool wakeup_lmon = false;
	int err = 0;

	spin_lock(&hw->port->lock);
	switch (hw->port->lstate) {

	case CASS_LINK_STATUS_UNKNOWN:
	case CASS_LINK_STATUS_UNCONFIGURED:
	case CASS_LINK_STATUS_ERROR:
		cxidev_err_ratelimited(&hw->cdev,
			"lmon: up request failed, wrong state (%s)\n",
			cass_link_state_str(hw->port->lstate));
		err = -ESTALE;
		break;

	case CASS_LINK_STATUS_STARTING:
	case CASS_LINK_STATUS_UP:
		/* continue coming up */
		if (cass_version(hw, CASSINI_1))
			sbl_base_link_enable_start(hw->sbl, 0);
		break;

	case CASS_LINK_STATUS_STOPPING:
	case CASS_LINK_STATUS_DOWN:
		/* start it coming up */
		if (cass_version(hw, CASSINI_1))
			sbl_base_link_enable_start(hw->sbl, 0);
		hw->port->lmon_dirn = CASS_LMON_DIRECTION_UP;
		hw->port->link_restart_count = 0;
		hw->port->link_restart_time_idx = 0;
		memset(hw->port->link_restart_time_buf, 0, sizeof(hw->port->link_restart_time_buf));
		wakeup_lmon = true;
		break;

	case CASS_LINK_STATUS_RESETTING:
		/* have to wait for this to finish - do nothing */
		err = -EBUSY;
		break;
	}
	spin_unlock(&hw->port->lock);

	if (wakeup_lmon)
		wake_up_interruptible(&hw->port->lmon_wq);

	return err;
}

int cass_lmon_request_down(struct cass_dev *hw)
{
	bool wakeup_lmon = false;
	int err = 0;

	/*
	 * first change direction
	 * then, if we need to cancel an ongoing startup, it
	 * will restarts with the new direction
	 */
	spin_lock(&hw->port->lock);
	switch (hw->port->lstate) {

	case CASS_LINK_STATUS_UNCONFIGURED:
	case CASS_LINK_STATUS_UNKNOWN:
	case CASS_LINK_STATUS_ERROR:
		cxidev_err(&hw->cdev, "lmon: down request failed, wrong state (%s)\n",
			   cass_link_state_str(hw->port->lstate));
		err = -ESTALE;
		break;

	case CASS_LINK_STATUS_STARTING:
	case CASS_LINK_STATUS_UP:
		/* start coming down*/
		hw->port->lmon_dirn = CASS_LMON_DIRECTION_DOWN;
		wakeup_lmon = true;
		break;

	case CASS_LINK_STATUS_STOPPING:
	case CASS_LINK_STATUS_DOWN:
		/* nothing to do */
		break;

	case CASS_LINK_STATUS_RESETTING:
		/* have to wait for this to finish - do nothing */
		err = -EBUSY;
		break;
	}

	/* cancel any starting operation */
	if (cass_version(hw, CASSINI_1) &&
			(hw->port->lstate == CASS_LINK_STATUS_STARTING)) {
		sbl_base_link_cancel_start(hw->sbl, 0);
	}
	spin_unlock(&hw->port->lock);

	if (wakeup_lmon)
		wake_up_interruptible(&hw->port->lmon_wq);

	return err;
}

int cass_lmon_request_reset(struct cass_dev *hw)
{
	bool wakeup_lmon = false;
	int err = 0;

	/*
	 * first change direction
	 * then, if we need to cancel an ongoing startup, it
	 * will restarts with the new direction
	 */
	spin_lock(&hw->port->lock);
	switch (hw->port->lstate) {

	case CASS_LINK_STATUS_UNKNOWN:
		cxidev_err(&hw->cdev, "lmon: down request failed, wrong state (%s)\n",
			   cass_link_state_str(hw->port->lstate));
		err = -ESTALE;
		break;

	case CASS_LINK_STATUS_STARTING:
	case CASS_LINK_STATUS_UP:
	case CASS_LINK_STATUS_STOPPING:
	case CASS_LINK_STATUS_DOWN:
	case CASS_LINK_STATUS_ERROR:
	case CASS_LINK_STATUS_UNCONFIGURED:
	case CASS_LINK_STATUS_RESETTING:
		/* start reset */
		hw->port->lmon_dirn = CASS_LMON_DIRECTION_RESET;
		wakeup_lmon = true;
		break;
	}

	/* cancel any starting operation */
	if (cass_version(hw, CASSINI_1) &&
			(hw->port->lstate == CASS_LINK_STATUS_STARTING)) {
		sbl_base_link_cancel_start(hw->sbl, 0);
	}
	spin_unlock(&hw->port->lock);

	if (wakeup_lmon)
		wake_up_interruptible(&hw->port->lmon_wq);

	return err;
}

int cass_lmon_get_dirn(struct cass_dev *hw)
{
	int dirn;

	spin_lock(&hw->port->lock);
	dirn = hw->port->lmon_dirn;
	spin_unlock(&hw->port->lock);

	return dirn;
}

void cass_lmon_set_dirn(struct cass_dev *hw, int dirn)
{
	spin_lock(&hw->port->lock);
	hw->port->lmon_dirn = dirn;
	spin_unlock(&hw->port->lock);
}

bool cass_lmon_get_active(struct cass_dev *hw)
{
	bool state;

	spin_lock(&hw->port->lock);
	state = hw->port->lmon_active;
	spin_unlock(&hw->port->lock);

	return state;
}

void cass_lmon_set_active(struct cass_dev *hw, bool state)
{
	spin_lock(&hw->port->lock);
	hw->port->lmon_active = state;
	spin_unlock(&hw->port->lock);
}

void cass_lmon_set_up_pause(struct cass_dev *hw, bool state)
{
	spin_lock(&hw->port->lock);
	hw->port->lmon_up_pause = state;
	spin_unlock(&hw->port->lock);
	if (!state)
		wake_up_interruptible(&hw->port->lmon_wq);
}

bool cass_lmon_get_up_pause(struct cass_dev *hw)
{
	bool state;

	spin_lock(&hw->port->lock);
	state = hw->port->lmon_up_pause;
	spin_unlock(&hw->port->lock);

	return state;
}

static int cass_lmon(void *data)
{
	struct cass_lmon_data *thrd_data = data;
	struct cass_dev *hw = thrd_data->hw;
	long timeout = msecs_to_jiffies(CASS_LMON_WAKEUP_INTERVAL);
	int down_origin;
	int status;
	char base_link_state_str[SBL_BASE_LINK_STATE_STR_LEN];

	cxidev_dbg(&hw->cdev, "lmon starting");

	complete(&thrd_data->running);

	cass_lmon_wakeup_limiter_reset(hw);
	cass_lmon_set_active(hw, true);

	// FIXME: temp delay before first link up.  Relates to NETCASSINI-6454
	if (cass_version(hw, CASSINI_2))
		msleep(5000);

	while (!kthread_should_stop()) {

		while (!cass_lmon_wakeup(hw)) {

			if (cass_lmon_get_active(hw)) {
				cxidev_dbg(&hw->cdev, "lmon sleeping\n");
				cass_lmon_set_active(hw, false);
			}

			status = wait_event_interruptible_timeout(
							  hw->port->lmon_wq,
							  cass_lmon_wakeup(hw),
							  timeout);

			atomic_inc(&hw->port->lmon_counters[lmon_wakeup]);

			if (kthread_should_stop())
				goto out;

			if (!status) {
				/* timed out and nothing needing attention */
				continue;
			}

			cxidev_dbg(&hw->cdev, "lmon active\n");
			cass_lmon_set_active(hw, true);
		}

		atomic_inc(&hw->port->lmon_counters[lmon_active]);

		if (kthread_should_stop())
			goto out;

		cass_lmon_wakeup_limiter(hw);

		switch (cass_lmon_get_dirn(hw)) {

		case CASS_LMON_DIRECTION_UP:

			if (cass_lmon_get_up_pause(hw)) {
				cxidev_dbg(&hw->cdev, "lmon up paused\n");
				break;
			}

			if (cass_phy_is_headshell_removed(hw))
				break;

			cxidev_dbg(&hw->cdev, "lmon up action\n");

			hw->link_ops->link_up(hw);
			break;

		case CASS_LMON_DIRECTION_DOWN:
			if (cass_version(hw, CASSINI_1)) {
				down_origin = cass_link_get_down_origin(hw);
				cass_sbl_counters_down_origin_inc(hw, down_origin);
				sbl_base_link_state_str(hw->sbl, 0, base_link_state_str,
							SBL_BASE_LINK_STATE_STR_LEN);
				cxidev_info(&hw->cdev,
					    "lmon taking link down (reason %s, bl: %s)\n",
					    cass_link_down_origin_str(down_origin),
					    base_link_state_str);
			}

			hw->link_ops->link_down(hw);

			spin_lock(&hw->port->lock);
			if (cass_lmon_link_should_auto_restart(hw)) {
				if (cass_version(hw, CASSINI_1))
					atomic_inc(&hw->sbl_counters[link_auto_restart]);
				cxidev_dbg(&hw->cdev,
					   "lmon auto restarting link up\n");
				hw->port->lmon_dirn = CASS_LMON_DIRECTION_UP;
				hw->port->link_restart_count++;
				hw->port->link_restart_time_buf[hw->port->link_restart_time_idx]
					= ktime_get_real_seconds();
				hw->port->link_restart_time_idx++;
				if (hw->port->link_restart_time_idx >=
					CASS_SBL_LINK_RESTART_TIME_SIZE)
					hw->port->link_restart_time_idx = 0;
			}
			spin_unlock(&hw->port->lock);

			if (cass_version(hw, CASSINI_2)) {
				if (cass_lmon_link_should_auto_restart(hw))
					cass_phy_trigger_machine(hw);
			}

			break;

		case CASS_LMON_DIRECTION_RESET:
			cxidev_dbg(&hw->cdev, "lmon reset action\n");
			hw->link_ops->link_reset(hw);
			hw->port->lmon_dirn = CASS_LMON_DIRECTION_NONE;
			break;

		case CASS_LMON_DIRECTION_NONE:
			/* don't do anything - thread is going to terminate */
			break;
		}
	}

out:
	return 0;
}

/**
 * cass_lmon_start_all() - Starts lmon thread
 *
 * @hw: the device
 *
 * Return: 0 on success, negative errno on failure
 */
int cass_lmon_start_all(struct cass_dev *hw)
{
	struct cass_lmon_data thrd_data;
	struct task_struct *lmon;
	int err;
	int i = 0;

	thrd_data.hw = hw;

	init_completion(&thrd_data.running);

	if (hw->port->lmon)
		return 0;

	lmon = kthread_run(cass_lmon, (void *)&thrd_data, "lmon");

	err = IS_ERR(lmon);
	if (err) {
		cxidev_err(&hw->cdev, "lmon %d: failed to start monitor [%d]\n",
			   i, err);
		goto out_err;
	} else {
		hw->port->lmon = lmon;
	}

	wait_for_completion(&thrd_data.running);
	cxidev_dbg(&hw->cdev, "lmon %d: started\n", i);

	return 0;

 out_err:
	cass_lmon_kill_all(hw);
	return err;

}

int cass_lmon_stop_all(struct cass_dev *hw)
{
	/* TODO add cleanup */
	cass_lmon_kill_all(hw);
	return 0;
}

void cass_lmon_kill_all(struct cass_dev *hw)
{
	int err = 0;

	if (!hw->port->lmon)
		return;

	/*
	 * cancel any starting ports and stop any more actions
	 */
	if (cass_version(hw, CASSINI_1) && (hw->sbl))
		sbl_base_link_cancel_start(hw->sbl, 0);
	if (cass_version(hw, CASSINI_2))
		hw->sl.is_canceled = true;
	cass_lmon_set_dirn(hw, CASS_LMON_DIRECTION_NONE);
	wake_up_interruptible(&hw->port->lmon_wq);

	err = kthread_stop(hw->port->lmon);
	if (err)
		cxidev_err(&hw->cdev, "thread stop failed [%d]\n", err);

	hw->port->lmon = NULL;

	cxidev_dbg(&hw->cdev, "lmon thread destroyed\n");
}

/*
 * Update lmon counters
 */
void cass_lmon_counters_init(struct cass_dev *hw)
{
	int i;

	hw->port->lmon_counters = kzalloc(sizeof(atomic_t)*CASS_LMON_NUM_COUNTERS, GFP_KERNEL);

	for (i = 0; i < CASS_LMON_NUM_COUNTERS; ++i)
		atomic_set(&hw->port->lmon_counters[i], 0);
}

/*
 * Destroy the port's counter array
 */
void cass_lmon_counters_term(struct cass_dev *hw)
{
	kfree(hw->port->lmon_counters);
	hw->port->lmon_counters = NULL;
}
