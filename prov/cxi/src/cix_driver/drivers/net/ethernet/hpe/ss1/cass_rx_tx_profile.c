// SPDX-License-Identifier: GPL-2.0
/* Copyright (C) 2024 Hewlett Packard Enterprise Development LP */

/* VNI Entry manipulation routines for Cassini Device */

#include "cass_core.h"
#include "cass_rx_tx_profile.h"

#include "cxi_rxtx_profile.h"
#include "cxi_rxtx_profile_list.h"

static struct xa_limit    rx_profile_limits = {
	.min = RX_PROFILE_ID_MIN,
	.max = RX_PROFILE_ID_MAX,
};

static struct xa_limit    tx_profile_limits = {
	.min = TX_PROFILE_ID_MIN,
	.max = TX_PROFILE_ID_MAX,
};

/* **************************************************************** */
/* Cassini device initialization and shutdown                       */
/* **************************************************************** */

/**
 * cass_dev_rx_tx_profiles_init() - Create device structures to support
 *                                  RX and TX Profile lists.
 *
 * @hw: The cassini device
 */
void cass_dev_rx_tx_profiles_init(struct cass_dev *hw)
{
	cxi_rxtx_profile_list_init(&hw->rx_profile_list,
				   &rx_profile_limits,
				   RX_PROFILE_XARRAY_FLAGS,
				   RX_PROFILE_GFP_OPTS);

	cxi_rxtx_profile_list_init(&hw->tx_profile_list,
				   &tx_profile_limits,
				   TX_PROFILE_XARRAY_FLAGS,
				   TX_PROFILE_GFP_OPTS);

	mutex_init(&hw->rx_profile_get_lock);
	mutex_init(&hw->tx_profile_get_lock);
}

struct profile_destroy_data {
	struct cxi_dev   *cxi_dev;
};

static void rx_profile_destroy(struct cxi_rxtx_profile *rxtx_profile,
			       void *user_arg)
{
	struct profile_destroy_data *data = user_arg;

	refcount_dec(&rxtx_profile->state.refcount);
	cxi_rx_profile_dec_refcount(data->cxi_dev, co_rx_profile(rxtx_profile));
}

/**
 * cass_dev_rx_tx_profiles_fini() - Destroy all RX and TX Profile entries
 * associated with this device, as well as locks, etc.
 *
 * @hw: Cassini Device
 */
void cass_dev_rx_tx_profiles_fini(struct cass_dev *hw)
{
	struct profile_destroy_data  data;

	data.cxi_dev = &hw->cdev;

	cxi_rxtx_profile_list_destroy(&hw->rx_profile_list,
				      rx_profile_destroy, &data);
	cxi_tx_profile_list_destroy(hw);
}

/**
 * cass_rx_profile_init() - initialize the hardware specific
 *                                 part of the RX Profile
 * @hw: Cassini device pointer
 * @rx_profile: pointer to Profile
 */
void cass_rx_profile_init(struct cass_dev *hw,
				struct cxi_rx_profile *rx_profile)
{
	spin_lock_init(&rx_profile->config.pid_lock);
	rx_profile->config.rmu_index = DEF_RMU_INDEX;
}

/**
 * cass_tx_profile_init() - initialize the hardware specific
 *                                 part of the TX Profile
 * @hw: Cassini device pointer
 * @tx_profile: pointer to Profile
 */
void cass_tx_profile_init(struct cass_dev *hw,
			  struct cxi_tx_profile *tx_profile)
{
	tx_profile->config.hw = hw;
	idr_init(&tx_profile->config.cass_cp_table);
	spin_lock_init(&tx_profile->config.lock);
}
