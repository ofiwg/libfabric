/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2024 Hewlett Packard Enterprise Development LP */

#ifndef _CASS_RX_TX_PROFILE_H_
#define _CASS_RX_TX_PROFILE_H_

struct cass_dev;

void cass_dev_rx_tx_profiles_init(struct cass_dev *hw);
void cass_dev_rx_tx_profiles_fini(struct cass_dev *hw);

void cass_rx_profile_init(struct cass_dev *hw, struct cxi_rx_profile *rx_profile);
void cass_tx_profile_init(struct cass_dev *hw, struct cxi_tx_profile *tx_profile);

#endif /* _CASS_RX_TX_PROFILE_H_ */
