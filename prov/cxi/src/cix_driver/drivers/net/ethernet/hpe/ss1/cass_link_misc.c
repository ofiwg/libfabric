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
#include "cass_cable.h"

/*
 * thread safe access functions
 *
 *   TODO inline these
 */
int cass_link_get_state(struct cass_dev *hw)
{
	int state;

	spin_lock(&hw->port->lock);
	state = hw->port->lstate;
	spin_unlock(&hw->port->lock);

	return state;
}

void cass_link_set_led(struct cass_dev *hw)
{
	cxidev_dbg(&hw->cdev,
		   "beacon = %s, overtemp = %s, hstate = %d, lstate = %d\n",
		   hw->qsfp_beacon_active ? "true" : "false",
		   hw->qsfp_over_temp ? "true" : "false",
		   hw->port->hstate,
		   hw->port->lstate);

	if (hw->qsfp_beacon_active) {
		uc_cmd_set_link_leds(hw, LED_SLOW_GRN);
		return;
	}

	if (!cass_version(hw, CASSINI_1))
		return;

	if (hw->port->hstate == CASS_HEADSHELL_STATUS_NOT_PRESENT)
		uc_cmd_set_link_leds(hw, LED_OFF);
	else if (hw->qsfp_over_temp)
		uc_cmd_set_link_leds(hw, LED_ON_YEL);
	else if (hw->port->hstate == CASS_HEADSHELL_STATUS_ERROR)
		uc_cmd_set_link_leds(hw, LED_FAST_YEL);
	else {
		switch (hw->port->lstate) {
		case CASS_LINK_STATUS_UNKNOWN:
		case CASS_LINK_STATUS_ERROR:
			uc_cmd_set_link_leds(hw, LED_FAST_GRN_YEL);
			break;
		case CASS_LINK_STATUS_RESETTING:
		case CASS_LINK_STATUS_UNCONFIGURED:
		case CASS_LINK_STATUS_STOPPING:
		case CASS_LINK_STATUS_DOWN:
			uc_cmd_set_link_leds(hw, LED_OFF);
			break;
		case CASS_LINK_STATUS_STARTING:
			uc_cmd_set_link_leds(hw, LED_FAST_GRN);
			break;
		case CASS_LINK_STATUS_UP:
			uc_cmd_set_link_leds(hw, LED_ON_GRN);
			break;
		default:
			break;
		}
	}
}

void cass_link_set_state(struct cass_dev *hw, int state, int err)
{
	spin_lock(&hw->port->lock);
	hw->port->prev_lstate = hw->port->lstate;
	hw->port->lstate = state;
	hw->port->lerr = err;
	spin_unlock(&hw->port->lock);

	cass_link_set_led(hw);
}

int cass_link_get_down_origin(struct cass_dev *hw)
{
	int origin;

	spin_lock(&hw->port->lock);
	origin = hw->port->link_down_origin;
	spin_unlock(&hw->port->lock);

	return origin;
}

void cass_link_set_down_origin(struct cass_dev *hw, int origin)
{
	spin_lock(&hw->port->lock);
	hw->port->link_down_origin = origin;
	spin_unlock(&hw->port->lock);
}

/*
 * enum to string functions
 */
const char *cass_link_state_str(enum cass_link_status state)
{
	switch (state) {
	case CASS_LINK_STATUS_UNKNOWN:      return "unknown";
	case CASS_LINK_STATUS_RESETTING:    return "resetting";
	case CASS_LINK_STATUS_UNCONFIGURED: return "unconfigured";
	case CASS_LINK_STATUS_STARTING:     return "starting";
	case CASS_LINK_STATUS_UP:           return "up";
	case CASS_LINK_STATUS_STOPPING:     return "stopping";
	case CASS_LINK_STATUS_DOWN:         return "down";
	case CASS_LINK_STATUS_ERROR:        return "error";
	default:                            return "unrecognised";
	}
}

const char *cass_lmon_dirn_str(enum cass_lmon_direction dirn)
{
	switch (dirn) {
	case CASS_LMON_DIRECTION_INVALID: return "unknown";
	case CASS_LMON_DIRECTION_NONE:    return "none";
	case CASS_LMON_DIRECTION_UP:      return "up";
	case CASS_LMON_DIRECTION_DOWN:    return "down";
	case CASS_LMON_DIRECTION_RESET:   return "reset";
	default:                          return "unrecognised";
	}
}

const char *cass_link_down_origin_str(u32 origin)
{
	switch (origin) {
	case CASS_DOWN_ORIGIN_UNKNOWN:           return "unknown";
	case CASS_DOWN_ORIGIN_NONE:              return "none";
	case CASS_DOWN_ORIGIN_CONFIG:            return "config";
	case CASS_DOWN_ORIGIN_BL_LFAULT:         return "bl/lfault";
	case CASS_DOWN_ORIGIN_BL_RFAULT:         return "bl/rfault";
	case CASS_DOWN_ORIGIN_BL_ALIGN:          return "bl/align";
	case CASS_DOWN_ORIGIN_BL_DOWN:           return "bl/down";
	case CASS_DOWN_ORIGIN_BL_HISER:          return "bl/hiser";
	case CASS_DOWN_ORIGIN_BL_LLR:            return "bl/llr";
	case CASS_DOWN_ORIGIN_BL_UNKNOWN:        return "bl/unknown";
	case CASS_DOWN_ORIGIN_LMON_UCW:          return "lmon/ucw";
	case CASS_DOWN_ORIGIN_LMON_CCW:          return "lmon/ccw";
	case CASS_DOWN_ORIGIN_LLR_TX_REPLAY:     return "lmon/llr_tx_replay";
	case CASS_DOWN_ORIGIN_HEADSHELL_REMOVED: return "headshell-removed";
	case CASS_DOWN_ORIGIN_HEADSHELL_ERROR:   return "headshell-error";
	case CASS_DOWN_ORIGIN_MEDIA_REMOVED:     return "media-removed";
	case CASS_DOWN_ORIGIN_CMD:               return "cmd";
	default:                                 return "unrecognised";
	}
}
