// SPDX-License-Identifier: GPL-2.0
/* Copyright 2020 Hewlett Packard Enterprise Development LP */

#include <linux/types.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/device.h>
#include <linux/sbl.h>

#include "cass_core.h"

/*
 * create and initialise the port array
 */
int cass_port_new_port_db(struct cass_dev *hw)
{
	hw->port = kzalloc(sizeof(struct cass_port), GFP_KERNEL);
	if (!hw->port)
		return -ENOMEM;

	spin_lock_init(&hw->port->lock);
	hw->port->lstate = CASS_LINK_STATUS_UNCONFIGURED;
	init_waitqueue_head(&hw->port->lmon_wq);
	spin_lock_init(&hw->port->pause_lock);
	hw->port->pause_type = CASS_PAUSE_TYPE_NONE;
	hw->port->tx_pause = false;
	hw->port->rx_pause = false;
	hw->port->lmon_counters = NULL;
	hw->port->start_time = 0;
	cass_lmon_counters_init(hw);

	return 0;
}

void cass_port_del_port_db(struct cass_dev *hw)
{
	if (hw->port) {
		cass_lmon_kill_all(hw);
		cass_lmon_counters_term(hw);
		kfree(hw->port);
		hw->port = NULL;
	}
}

/*
 * text output helpers
 */
const char *cass_port_subtype_str(enum cass_port_subtype subtype)
{
	switch (subtype) {
	case CASS_PORT_SUBTYPE_IEEE:    return "ieee";
	case CASS_PORT_SUBTYPE_CASSINI: return "cassini";
	case CASS_PORT_SUBTYPE_LOCAL:   return "local";
	case CASS_PORT_SUBTYPE_GLOBAL:  return "global";
	default:                        return "unknown";
	}
}

const char *cass_pause_type_str(enum cass_pause_type type)
{
	switch (type) {
	case CASS_PAUSE_TYPE_INVALID: return "invalid";
	case CASS_PAUSE_TYPE_NONE:    return "none";
	case CASS_PAUSE_TYPE_GLOBAL:  return "global/802.3x";
	case CASS_PAUSE_TYPE_PFC:     return "pfc/802.1qbb";
	default:                      return "unrecognised";
	}
}
