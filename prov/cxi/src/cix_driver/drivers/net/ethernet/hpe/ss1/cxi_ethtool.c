// SPDX-License-Identifier: GPL-2.0
/*
 * Cassini ethernet driver
 * Copyright 2018,2022,2024 Hewlett Packard Enterprise Development LP
 */

#include <linux/netdevice.h>
#include <linux/etherdevice.h>
#include <linux/ethtool.h>
#include <linux/firmware.h>
#include <uapi/linux/net_tstamp.h>

#include "cxi_eth.h"
#include "cxi_link.h"
/*
 * definitions for CXI_GLOBAL_STATS_LEN, cxi_get_ethtool_stats_set,
 * and cxi_get_ethtool_stats_name.
 */
#include "cassini-telemetry-ethtool-names.h"

/* Ethtool priv_flags */
static const char priv_flags_str[PRIV_FLAGS_COUNT][ETH_GSTRING_LEN] = {
	"internal-loopback",
	"external-loopback",
	"llr",
	"precoding",
	"ifg-hpc",
	"roce-opt",
	"ignore-align",
	"disable-pml-recovery",
	"link-train",
	"ck-speed",
	"remote-fault-recovery",
	"use-unsupported-cable",
	"fec_monitor",
	"auto-lane-degrade",
	"ignore-media-error",
	"use-supported-ss200-cable",
};

/* ethtool ops */
static void cxi_get_drvinfo(struct net_device *ndev,
			    struct ethtool_drvinfo *info)
{
	struct cxi_eth *dev = netdev_priv(ndev);

	strscpy(info->driver, "cxi_eth", sizeof(info->driver));
	strscpy(info->version, "0.2", sizeof(info->version));
	strscpy(info->bus_info, dev_name(ndev->dev.parent),
		sizeof(info->bus_info));
	strscpy(info->fw_version, dev->eth_info.fw_version,
		sizeof(info->fw_version));
	strscpy(info->erom_version, dev->eth_info.erom_version,
		sizeof(info->erom_version));

	info->n_priv_flags = PRIV_FLAGS_COUNT;
}

static int cxi_get_sset_count(struct net_device *ndev, int sset)
{
	switch (sset) {
	case ETH_SS_PRIV_FLAGS:
		return PRIV_FLAGS_COUNT;
	case ETH_SS_STATS:
		return CXI_GLOBAL_STATS_LEN;
	default:
		return -EOPNOTSUPP;
	}
}

static void cxi_get_strings(struct net_device *ndev, u32 stringset, u8 *data)
{
	char *p = data;

	switch (stringset) {
	case ETH_SS_PRIV_FLAGS:
		memcpy(p, priv_flags_str, PRIV_FLAGS_COUNT * ETH_GSTRING_LEN);
		break;
	case ETH_SS_STATS:
		memcpy(p, cxi_get_ethtool_stats_name, CXI_GLOBAL_STATS_LEN *
		       ETH_GSTRING_LEN);
		break;

	}
}

/* Return the number of RSS queues. */
static void cxi_get_channels(struct net_device *ndev,
			     struct ethtool_channels *channels)
{
	channels->max_rx = max_rss_queues;
	channels->max_tx = max_tx_queues;
	channels->max_combined = 0;
	channels->max_other = 0;
	channels->rx_count = ndev->real_num_rx_queues;
	channels->tx_count = ndev->real_num_tx_queues;
	channels->other_count = 0;
	channels->combined_count = 0;
}

static int cxi_grow_tx_channels(struct cxi_eth *dev,
				unsigned int num_tx_channels)
{
	int i;
	int rc;

	for (i = dev->cur_txqs; i < num_tx_channels; i++) {
		rc = alloc_tx_queue(dev, i);
		if (rc)
			goto err_free_txqs;
	}

	for (i = dev->cur_txqs; i < num_tx_channels; i++)
		enable_tx_queue(&dev->txqs[i]);

	rc = netif_set_real_num_tx_queues(dev->ndev, num_tx_channels);
	if (rc)
		goto err_free_txqs;

	return 0;

err_free_txqs:
	for (i--; i >= dev->cur_txqs; i--)
		free_tx_queue(&dev->txqs[i]);

	return rc;
}

static int cxi_shrink_tx_channels(struct cxi_eth *dev,
				  unsigned int num_tx_channels)
{
	int i;
	int rc;

	rc = netif_set_real_num_tx_queues(dev->ndev, num_tx_channels);
	if (rc)
		return rc;

	for (i = dev->cur_txqs - 1; i >= num_tx_channels; i--)
		disable_tx_queue(&dev->txqs[i]);

	for (i = dev->cur_txqs - 1; i >= num_tx_channels; i--)
		free_tx_queue(&dev->txqs[i]);

	return 0;
}

int cxi_set_tx_channels(struct cxi_eth *dev, unsigned int num_tx_channels)
{
	int rc;

	if (dev->cur_txqs == num_tx_channels)
		return 0;
	else if (num_tx_channels < dev->cur_txqs)
		rc = cxi_shrink_tx_channels(dev, num_tx_channels);
	else
		rc = cxi_grow_tx_channels(dev, num_tx_channels);

	if (!rc)
		dev->cur_txqs = num_tx_channels;

	return rc;
}

static int cxi_grow_rx_channels(struct cxi_eth *dev,
				unsigned int num_rx_channels)
{
	int rc;
	int i;

	for (i = dev->res.rss_queues; i < num_rx_channels; i++) {
		rc = alloc_rx_queue(dev, i);
		if (rc)
			goto err_free_rxqs;

		rc = post_rx_buffers(&dev->rxqs[i], GFP_KERNEL);
		if (rc < 0) {
			netdev_info(dev->ndev, "Cannot post RX buffers: %d\n",
				    rc);
			/* Increment i to free this RXQ. */
			i++;
			goto err_free_rxqs;
		}
	}

	for (i = dev->res.rss_queues; i < num_rx_channels; i++)
		enable_rx_queue(&dev->rxqs[i]);

	return 0;

err_free_rxqs:
	for (i--; i >= dev->res.rss_queues; i--)
		free_rx_queue(&dev->rxqs[i]);

	return rc;
}

static void cxi_shrink_rx_channels(struct cxi_eth *dev,
				   unsigned int num_rx_channels)
{
	int i;

	for (i = dev->res.rss_queues - 1; i >= num_rx_channels; i--)
		disable_rx_queue(&dev->rxqs[i]);

	for (i = dev->res.rss_queues - 1; i >= num_rx_channels; i--)
		free_rx_queue(&dev->rxqs[i]);
}

static bool valid_rx_channel_count(unsigned int num_rx_channels)
{
	if (!is_power_of_2(num_rx_channels) || num_rx_channels < 1 ||
	    num_rx_channels > max_rss_queues)
		return false;
	return true;
}

int cxi_set_rx_channels(struct cxi_eth *dev, unsigned int num_rx_channels)
{
	int rc;
	int i;

	if (!valid_rx_channel_count(num_rx_channels))
		return -EINVAL;

	if (num_rx_channels == dev->res.rss_queues)
		return 0;

	/* Disable RSS support for all configure MAC addresses/set lists. This
	 * will temporarily force all traffic to the RXQ 0 which is the
	 * default/catch-all RX queue.
	 */
	cxi_eth_clear_indir_table(dev->cxi_dev, &dev->res);

	if (num_rx_channels > dev->res.rss_queues) {
		rc = cxi_grow_rx_channels(dev, num_rx_channels);
		if (rc) {
			/* Number of channels have not changed. Re-enable RSS
			 * and exit.
			 */
			cxi_eth_set_indir_table(dev->cxi_dev, &dev->res);
			goto out;
		}
	} else {
		cxi_shrink_rx_channels(dev, num_rx_channels);
	}

	dev->res.rss_queues = num_rx_channels;
	dev->res.rss_indir_size = num_rx_channels > 1 ? rss_indir_size : 0;

	rc = netif_set_real_num_rx_queues(dev->ndev, num_rx_channels);
	if (rc)
		goto out;

	for (i = 0; i < dev->res.rss_queues; i++)
		dev->res.ptn_rss[i] = dev->rxqs[i].pt->id;

	for (i = 0; i < dev->res.rss_indir_size; i++)
		dev->res.indir_table[i] =
			ethtool_rxfh_indir_default(i, dev->res.rss_queues);

	/* Re-enable RSS if number of RX queues is successfully changed. */
	cxi_eth_set_indir_table(dev->cxi_dev, &dev->res);

out:
	return rc;
}

static int cxi_set_channels(struct net_device *ndev,
			    struct ethtool_channels *ch)
{
	struct cxi_eth *dev = netdev_priv(ndev);
	int rc;

	if (ch->combined_count || ch->other_count)
		return -EINVAL;

	/* If interface is not running, update the real TX/RX queue count. The
	 * real TX/RX queue count is used to initialize TX/RX channels during
	 * hardware setup.
	 */
	if (!netif_running(dev->ndev)) {
		if (!valid_rx_channel_count(ch->rx_count))
			return -EINVAL;

		rc = netif_set_real_num_rx_queues(ndev, ch->rx_count);
		if (rc)
			return rc;

		return netif_set_real_num_tx_queues(ndev, ch->tx_count);
	}

	rc = cxi_set_rx_channels(dev, ch->rx_count);
	if (rc)
		return rc;

	return cxi_set_tx_channels(dev, ch->tx_count);
}

static int cxi_get_rxnfc(struct net_device *ndev, struct ethtool_rxnfc *cmd,
			 u32 *rule_locs)
{
	struct cxi_eth *dev = netdev_priv(ndev);
	int rc;

	switch (cmd->cmd) {
	case ETHTOOL_GRXRINGS:
		/* The catch-all queue is always present */
		cmd->data = dev->res.rss_queues ? : 1;
		rc = 0;
		break;

	default:
		rc = -EOPNOTSUPP;
		break;
	}

	return rc;
}

static u32 cxi_get_rxfh_indir_size(struct net_device *ndev)
{
	struct cxi_eth *dev = netdev_priv(ndev);

	return dev->res.rss_indir_size;
}

static u32 cxi_get_rxfh_key_size(struct net_device *ndev)
{
	return CXI_ETH_HASH_KEY_SIZE;
}

#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 4, 0) || (defined(RHEL_MAJOR) && (RHEL_MAJOR == 9 && RHEL_MINOR >= 5))
static int cxi_get_rxfh(struct net_device *ndev,
			struct ethtool_rxfh_param *rxfh)
{
	u8 *key = rxfh->key;
	u8 *hfunc = &rxfh->hfunc;
	u32 *indir = rxfh->indir;
#else
static int cxi_get_rxfh(struct net_device *ndev, u32 *indir, u8 *key, u8 *hfunc)
{
#endif
	struct cxi_eth *dev = netdev_priv(ndev);

	if (hfunc)
		*hfunc = ETH_RSS_HASH_TOP;

	if (key)
		cxi_eth_get_hash_key(dev->cxi_dev, key);

	if (indir && dev->res.rss_queues)
		memcpy(indir, dev->res.indir_table,
		       dev->res.rss_indir_size * sizeof(u32));

	return 0;
}

#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 4, 0) || (defined(RHEL_MAJOR) && (RHEL_MAJOR == 9 && RHEL_MINOR >= 5))
static int cxi_set_rxfh(struct net_device *ndev,
			struct ethtool_rxfh_param *rxfh,
			struct netlink_ext_ack *extack)
{
	u8 *key = rxfh->key;
	u8 hfunc = rxfh->hfunc;
	u32 *indir = rxfh->indir;
#else
static int cxi_set_rxfh(struct net_device *ndev, const u32 *indir,
			const u8 *key, const u8 hfunc)
{
#endif
	struct cxi_eth *dev = netdev_priv(ndev);
	int i;

	if (hfunc != ETH_RSS_HASH_NO_CHANGE && hfunc != ETH_RSS_HASH_TOP)
		return -EINVAL;

	if (key != NULL)
		return -EINVAL;

	if (indir) {
		/* Sanity check the table */
		for (i = 0; i < dev->res.rss_indir_size; i++) {
			if (indir[i] > dev->res.rss_queues) {
				netdev_warn_once(ndev, "Bad new indirection - %d %d\n",
						 indir[i],
						 dev->res.rss_queues);
				return -EINVAL;
			}
		}

		memcpy(dev->res.indir_table, indir,
		       dev->res.rss_indir_size * sizeof(u32));
		cxi_eth_set_indir_table(dev->cxi_dev, &dev->res);
	}

	return 0;
}

static int cxi_get_module_info(struct net_device *ndev,
			       struct ethtool_modinfo *modinfo)
{
	struct cxi_eth *dev = netdev_priv(ndev);

	cxi_eth_devinfo(dev->cxi_dev, &dev->eth_info);

	if (dev->eth_info.qsfp_eeprom_len == 0)
		return -ENOMEDIUM;

	switch (dev->eth_info.qsfp_eeprom[0]) {
	case 0xc: /* QSFP */
	case 0xd: /* QSFP+ */
		modinfo->type = ETH_MODULE_SFF_8436;
		modinfo->eeprom_len = ETH_MODULE_SFF_8436_LEN;
		break;
	case 0x11: /* QSFP-28 */
	case 0x1E:
		modinfo->type = ETH_MODULE_SFF_8636;
		modinfo->eeprom_len = ETH_MODULE_SFF_8636_LEN;
		break;
	default:
		netdev_err(dev->ndev, "Unsupported transceiver type 0x%x\n",
			   dev->eth_info.qsfp_eeprom[0]);
		return -EIO;
	}

	return 0;
}

static int cxi_get_module_eeprom(struct net_device *ndev,
				 struct ethtool_eeprom *eeprom, u8 *data)
{
	struct cxi_eth *dev = netdev_priv(ndev);
	int ret;

	ret = cxi_get_qsfp_data(dev->cxi_dev, eeprom->offset, eeprom->len,
				0, data);

	if (ret > 0)
		return 0;
	else
		return -EINVAL;
}

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 12, 0)
static int cxi_get_module_eeprom_by_page(struct net_device *ndev,
					 const struct ethtool_module_eeprom *req,
					 struct netlink_ext_ack *extack)
{
	struct cxi_eth *dev = netdev_priv(ndev);

	return cxi_get_qsfp_data(dev->cxi_dev, req->offset, req->length,
				 req->page, req->data);
}
#endif

static int cxi_flash_device(struct net_device *ndev,
			    struct ethtool_flash *flash)
{
	struct cxi_eth *dev = netdev_priv(ndev);
	const struct firmware *fw;
	int rc;

	if (!capable(CAP_SYS_ADMIN))
		return -EPERM;

	rc = request_firmware(&fw, flash->data, &ndev->dev);
	if (rc != 0)
		return rc;

	// TODO? stop adapter
	rc = cxi_program_firmware(dev->cxi_dev, fw);
	release_firmware(fw);
	// TODO? restart adapter

	if (!rc)
		cxi_eth_devinfo(dev->cxi_dev, &dev->eth_info);

	return rc;
}

/* Link settings. */
static int cxi_get_link_ksettings(struct net_device *ndev,
				  struct ethtool_link_ksettings *s)
{
	struct cxi_eth *dev = netdev_priv(ndev);
	struct cxi_link_info link_info;

	memset(s, 0, sizeof(struct ethtool_link_ksettings));

	cxi_link_mode_get(dev->cxi_dev, &link_info);

	s->base.speed = link_info.speed;
	s->base.port = link_info.port_type;
	s->base.duplex = DUPLEX_FULL;
	s->base.phy_address = 0; /* N/A */
	s->base.autoneg = link_info.autoneg;
	s->base.eth_tp_mdix = ETH_TP_MDI_INVALID;
	s->base.eth_tp_mdix_ctrl = ETH_TP_MDI_INVALID;

	ethtool_link_ksettings_add_link_mode(s, supported, Autoneg);
	ethtool_link_ksettings_add_link_mode(s, supported, FIBRE);
	ethtool_link_ksettings_add_link_mode(s, supported, TP);
	ethtool_link_ksettings_add_link_mode(s, supported, Pause);
	ethtool_link_ksettings_add_link_mode(s, supported, 50000baseKR_Full);
	ethtool_link_ksettings_add_link_mode(s, supported, 100000baseCR2_Full);
	ethtool_link_ksettings_add_link_mode(s, supported, 100000baseCR4_Full);
	ethtool_link_ksettings_add_link_mode(s, supported, 200000baseCR4_Full);
	if (s->base.autoneg == AUTONEG_ENABLE)
		ethtool_link_ksettings_add_link_mode(s, advertising, Autoneg);

	ethtool_link_ksettings_add_link_mode(s, advertising, Pause);
	ethtool_link_ksettings_add_link_mode(s, advertising, 50000baseKR_Full);
	ethtool_link_ksettings_add_link_mode(s, advertising, 100000baseCR2_Full);
	ethtool_link_ksettings_add_link_mode(s, advertising, 100000baseCR4_Full);
	ethtool_link_ksettings_add_link_mode(s, advertising, 200000baseCR4_Full);

	if (cassini_version(&dev->cxi_dev->prop, CASSINI_2)) {
		ethtool_link_ksettings_add_link_mode(s, supported, 400000baseCR4_Full);
		ethtool_link_ksettings_add_link_mode(s, advertising, 400000baseCR4_Full);
	}

	return 0;
}

static int
cxi_set_link_ksettings(struct net_device *ndev,
		       const struct ethtool_link_ksettings *s)
{
	struct cxi_eth *dev = netdev_priv(ndev);
	struct cxi_link_info link_info;

	cxi_link_mode_get(dev->cxi_dev, &link_info);

	if (s->base.duplex != DUPLEX_UNKNOWN &&
	    s->base.duplex != DUPLEX_FULL)
		return -ENOTSUPP;

	if (dev->is_c2 && s->base.autoneg == AUTONEG_ENABLE &&
	    (link_info.flags & LOOPBACK_MODE) != 0) {
		netdev_err(ndev,
			   "autoneg must not be enabled in loopback mode\n");
		return -EINVAL;
	}

	link_info.speed = s->base.speed;
	link_info.autoneg = s->base.autoneg;

	cxi_link_mode_set(dev->cxi_dev, &link_info);

	return 0;
}

static int cxi_set_priv_flags(struct net_device *ndev, u32 flags)
{
	struct cxi_eth *dev = netdev_priv(ndev);
	struct cxi_link_info link_info;
	u32 changes;
	u32 old_flags;
	u32 debug_flags;

	cxi_link_mode_get(dev->cxi_dev, &link_info);
	cxi_link_flags_get(dev->cxi_dev, &debug_flags);

	old_flags = dev->priv_flags | link_info.flags | debug_flags;

	changes = flags ^ old_flags;

	if (!changes)
		return 0;

	if (changes & LOOPBACK_MODE) {
		u32 loopback_mode = flags & LOOPBACK_MODE;

		if (loopback_mode ==
		    (CXI_ETH_PF_INTERNAL_LOOPBACK | CXI_ETH_PF_EXTERNAL_LOOPBACK)) {
			netdev_err(dev->ndev,
				   "Loopback private flags are mutually exclusive\n");
			return -EINVAL;
		}
	}

	if (changes & CXI_ETH_PF_FEC_MONITOR) {
		if (flags & CXI_ETH_PF_FEC_MONITOR) {
			dev->priv_flags |= CXI_ETH_PF_FEC_MONITOR;
			cxi_link_fec_monitor(dev->cxi_dev, true);
		} else {
			dev->priv_flags &= ~CXI_ETH_PF_FEC_MONITOR;
			cxi_link_fec_monitor(dev->cxi_dev, false);
		}
	}

	link_info.flags = flags;
	cxi_link_mode_set(dev->cxi_dev, &link_info);

	if (changes & CXI_ETH_PF_ROCE_OPT) {
		if (flags & CXI_ETH_PF_ROCE_OPT)
			dev->priv_flags |= CXI_ETH_PF_ROCE_OPT;
		else
			dev->priv_flags &= ~CXI_ETH_PF_ROCE_OPT;

		cxi_set_roce_rcv_seg(dev->cxi_dev, flags & CXI_ETH_PF_ROCE_OPT);
	}

	if (changes & CXI_ETH_PF_IGNORE_ALIGN) {
		if (flags & CXI_ETH_PF_IGNORE_ALIGN)
			cxi_link_flags_set(dev->cxi_dev,
					   0, CXI_ETH_PF_IGNORE_ALIGN);
		else
			cxi_link_flags_set(dev->cxi_dev,
					   CXI_ETH_PF_IGNORE_ALIGN, 0);
	}

	if (changes & CXI_ETH_PF_DISABLE_PML_RECOVERY) {
		if (flags & CXI_ETH_PF_DISABLE_PML_RECOVERY) {
			dev->priv_flags |= CXI_ETH_PF_DISABLE_PML_RECOVERY;
			cxi_pml_recovery_set(dev->cxi_dev, true);
		} else {
			dev->priv_flags &= ~CXI_ETH_PF_DISABLE_PML_RECOVERY;
			cxi_pml_recovery_set(dev->cxi_dev, false);
		}
	}

	if (changes & CXI_ETH_PF_REMOTE_FAULT_RECOVERY) {
		if (flags & CXI_ETH_PF_REMOTE_FAULT_RECOVERY)
			cxi_link_flags_set(dev->cxi_dev,
					   0, CXI_ETH_PF_REMOTE_FAULT_RECOVERY);
		else
			cxi_link_flags_set(dev->cxi_dev,
					   CXI_ETH_PF_REMOTE_FAULT_RECOVERY, 0);
	}

	if (changes & CXI_ETH_PF_USE_UNSUPPORTED_CABLE) {
		if (flags & CXI_ETH_PF_USE_UNSUPPORTED_CABLE) {
			dev->priv_flags |= CXI_ETH_PF_USE_UNSUPPORTED_CABLE;
			cxi_link_use_unsupported_cable(dev->cxi_dev, true);
		} else {
			dev->priv_flags &= ~CXI_ETH_PF_USE_UNSUPPORTED_CABLE;
			cxi_link_use_unsupported_cable(dev->cxi_dev, false);
		}
	}

	if (changes & CXI_ETH_PF_USE_SUPPORTED_SS200_CABLE) {
		if (flags & CXI_ETH_PF_USE_SUPPORTED_SS200_CABLE) {
			dev->priv_flags |= CXI_ETH_PF_USE_SUPPORTED_SS200_CABLE;
			cxi_link_use_supported_ss200_cable(dev->cxi_dev, true);
		} else {
			dev->priv_flags &= ~CXI_ETH_PF_USE_SUPPORTED_SS200_CABLE;
			cxi_link_use_supported_ss200_cable(dev->cxi_dev, false);
		}
	}

	if (changes & CXI_ETH_PF_IGNORE_MEDIA_ERROR) {
		if (flags & CXI_ETH_PF_IGNORE_MEDIA_ERROR) {
                        dev->priv_flags |= CXI_ETH_PF_IGNORE_MEDIA_ERROR;
                        cxi_link_ignore_media_error(dev->cxi_dev, true);
                } else {
                        dev->priv_flags &= ~CXI_ETH_PF_IGNORE_MEDIA_ERROR;
                        cxi_link_ignore_media_error(dev->cxi_dev, false);
                }
        }

       if (changes & CXI_ETH_PF_ALD) {
                if (flags & CXI_ETH_PF_ALD) {
                        dev->priv_flags |= CXI_ETH_PF_ALD;
                        cxi_link_auto_lane_degrade(dev->cxi_dev, true);
                } else {
                        dev->priv_flags &= ~CXI_ETH_PF_ALD;
                        cxi_link_auto_lane_degrade(dev->cxi_dev, false);
                }
        }

	return 0;
}

static u32 cxi_get_priv_flags(struct net_device *ndev)
{
	struct cxi_eth *dev = netdev_priv(ndev);
	struct cxi_link_info link_info;
	u32 debug_flags;

	cxi_link_mode_get(dev->cxi_dev, &link_info);
	cxi_link_flags_get(dev->cxi_dev, &debug_flags);

	return dev->priv_flags | link_info.flags | debug_flags;
}

static void cxi_get_ethtool_stats(struct net_device *ndev,
				  struct ethtool_stats *stats, u64 *data)
{
	struct cxi_eth *dev = netdev_priv(ndev);
	struct cxi_dev *cdev = dev->cxi_dev;
	int retval;
	int i;

	retval = cxi_telem_get_selected(cdev, cxi_get_ethtool_stats_set, data,
					CXI_GLOBAL_STATS_LEN);
	if (retval != 0)
		for (i = 0 ; i < CXI_GLOBAL_STATS_LEN ; ++i)
			data[i] = 0;
}

static int cxi_get_ts_info(struct net_device *ndev,
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 11, 0) || defined(HAVE_KERNEL_ETHTOOL_TS_INFO)
			   struct kernel_ethtool_ts_info *info
#else
			   struct ethtool_ts_info *info
#endif
	)
{
	struct cxi_eth *dev = netdev_priv(ndev);

	info->so_timestamping =
		SOF_TIMESTAMPING_RX_SOFTWARE | SOF_TIMESTAMPING_RX_HARDWARE |
		SOF_TIMESTAMPING_TX_SOFTWARE | SOF_TIMESTAMPING_TX_HARDWARE |
		SOF_TIMESTAMPING_SOFTWARE | SOF_TIMESTAMPING_RAW_HARDWARE;

	info->phc_index = dev->eth_info.ptp_clock_index;

	info->tx_types = BIT(HWTSTAMP_TX_OFF) | BIT(HWTSTAMP_TX_ON);

	/* The HW only supports PTP Ethernet filter. Timestamp cannot
	 * be added to PTP UDP packet.
	 */
	info->rx_filters = BIT(HWTSTAMP_FILTER_NONE) |
			   BIT(HWTSTAMP_FILTER_PTP_V2_L2_EVENT);

	return 0;
}

static void cxi_get_pauseparam(struct net_device *ndev,
			       struct ethtool_pauseparam *pause)
{
	struct cxi_eth *dev = netdev_priv(ndev);

	cxi_eth_get_pause(dev->cxi_dev, pause);
	pause->autoneg = 0;
}

static int cxi_set_pauseparam(struct net_device *ndev,
			      struct ethtool_pauseparam *pause)
{
	struct cxi_eth *dev = netdev_priv(ndev);

	if (pause->autoneg != 0)
		return -EINVAL;

	cxi_eth_set_pause(dev->cxi_dev, pause);

	return 0;
}

#ifdef HAVE_KERNEL_RINGPARAM
static void cxi_get_ringparam(struct net_device *ndev,
			      struct ethtool_ringparam *ring,
			      struct kernel_ethtool_ringparam *kernel_ring,
			      struct netlink_ext_ack *extack)
#else
static void cxi_get_ringparam(struct net_device *ndev,
			      struct ethtool_ringparam *ring)
#endif
{
	struct cxi_eth *dev = netdev_priv(ndev);

	*ring = dev->ringparam;

#ifdef HAVE_KERNEL_RINGPARAM
#ifdef ETHTOOL_RING_USE_CQE_SIZE
	kernel_ring->cqe_size = get_rxq_eq_buf_size(dev) /
		sizeof(struct c_event_target_enet);
#endif
	kernel_ring->rx_buf_len =
		dev->ringparam.rx_pending * dev->eth_info.max_segment_size +
		small_pkts_buf_count * small_pkts_buf_size;
#endif
}

#ifdef HAVE_KERNEL_RINGPARAM
static int cxi_set_ringparam(struct net_device *ndev,
			     struct ethtool_ringparam *ring,
			     struct kernel_ethtool_ringparam *kernel_ring,
			     struct netlink_ext_ack *extack)
#else
static int cxi_set_ringparam(struct net_device *ndev,
			     struct ethtool_ringparam *ring)
#endif
{
	struct cxi_eth *dev = netdev_priv(ndev);
	int rc = 0;

	if (ring->rx_jumbo_pending)
		return -EINVAL;

	dev->ringparam.rx_pending = clamp(ring->rx_pending,
					  LARGE_PKTS_BUF_COUNT_MIN,
					  LARGE_PKTS_BUF_COUNT_MAX);

	dev->ringparam.rx_mini_pending = num_small_packets(
		clamp(round_up(ring->rx_mini_pending, 4096) /
		      (small_pkts_buf_size / buffer_threshold),
		      SMALL_PKTS_BUF_COUNT_MIN, SMALL_PKTS_BUF_COUNT_MAX));

	dev->ringparam.tx_pending = clamp_t(unsigned int, ring->tx_pending,
					    1, CXI_MAX_CQ_COUNT);

	if (netif_running(ndev)) {
		hw_cleanup(dev);
		rc = hw_setup(dev);
		if (rc)
			netdev_err(ndev,
				   "Failed to restart after ringparam change\n");
	}

	return rc;
}

static int cxi_set_phys_id(struct net_device *ndev,
			  enum ethtool_phys_id_state state)
{
	struct cxi_eth *dev = netdev_priv(ndev);

	switch (state) {
	case ETHTOOL_ID_ACTIVE:
		cxi_set_led_beacon(dev->cxi_dev, true);
		return 0;

	case ETHTOOL_ID_INACTIVE:
		cxi_set_led_beacon(dev->cxi_dev, false);
		return 0;
	default:
		return -EOPNOTSUPP;
	}
}

const struct ethtool_ops cxi_eth_ethtool_ops = {
#ifdef HAVE_KERNEL_RINGPARAM
	.supported_ring_params =
#ifdef ETHTOOL_RING_USE_CQE_SIZE
	ETHTOOL_RING_USE_CQE_SIZE |
#endif
	ETHTOOL_RING_USE_RX_BUF_LEN,
#endif
	.get_drvinfo	= cxi_get_drvinfo,
	.get_sset_count	= cxi_get_sset_count,
	.get_strings	= cxi_get_strings,
	.get_rxnfc	= cxi_get_rxnfc,
	.get_channels	= cxi_get_channels,
	.set_channels	= cxi_set_channels,
	.get_rxfh_indir_size = cxi_get_rxfh_indir_size,
	.get_rxfh       = cxi_get_rxfh,
	.set_rxfh       = cxi_set_rxfh,
	.get_rxfh_key_size = cxi_get_rxfh_key_size,
	.get_module_info = cxi_get_module_info,
	.get_module_eeprom = cxi_get_module_eeprom,
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 12, 0)
	.get_module_eeprom_by_page = cxi_get_module_eeprom_by_page,
#endif
	.flash_device   = cxi_flash_device,
	.get_link	= ethtool_op_get_link,
	.get_link_ksettings = cxi_get_link_ksettings,
	.set_link_ksettings = cxi_set_link_ksettings,
	.get_priv_flags     = cxi_get_priv_flags,
	.set_priv_flags     = cxi_set_priv_flags,
	.get_ethtool_stats  = cxi_get_ethtool_stats,
	.get_ts_info        = cxi_get_ts_info,
	.set_phys_id        = cxi_set_phys_id,
	.get_pauseparam     = cxi_get_pauseparam,
	.set_pauseparam     = cxi_set_pauseparam,
	.get_ringparam      = cxi_get_ringparam,
	.set_ringparam      = cxi_set_ringparam,
};
