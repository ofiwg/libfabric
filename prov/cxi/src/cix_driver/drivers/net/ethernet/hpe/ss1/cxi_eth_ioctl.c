// SPDX-License-Identifier: GPL-2.0
/*
 * Cray Cassini ethernet driver - ioctls handling.
 * © Copyright 2021 Hewlett Packard Enterprise Development LP
 */

#include <linux/netdevice.h>

#include "cxi_eth.h"

static int cxi_ptp_get_ts_config(struct cxi_eth *dev, struct ifreq *ifr)
{
	int rc;

	rc = copy_to_user(ifr->ifr_data, &dev->tstamp_config,
			  sizeof(dev->tstamp_config));

	return rc ? -EFAULT : 0;
}

static int cxi_ptp_set_ts_config(struct cxi_eth *dev, struct ifreq *ifr)
{
	struct hwtstamp_config cfg;
	int rc;

	if (copy_from_user(&cfg, ifr->ifr_data, sizeof(cfg)))
		return -EFAULT;

	rc = cxi_eth_cfg_timestamp(dev->cxi_dev, &cfg);
	if (rc)
		return rc;

	dev->tstamp_config = cfg;

	dev->ptp_ts_enabled = cfg.rx_filter == HWTSTAMP_FILTER_PTP_V2_L2_EVENT;

	return copy_to_user(ifr->ifr_data, &cfg, sizeof(cfg)) ?
		-EFAULT : 0;
}

int cxi_do_ioctl(struct net_device *ndev, struct ifreq *ifr, int cmd)
{
	struct cxi_eth *dev = netdev_priv(ndev);

	switch (cmd) {
	case SIOCGHWTSTAMP:
		return cxi_ptp_get_ts_config(dev, ifr);
	case SIOCSHWTSTAMP:
		return cxi_ptp_set_ts_config(dev, ifr);
	default:
		return -EOPNOTSUPP;
	}
}
