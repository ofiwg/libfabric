// SPDX-License-Identifier: GPL-2.0
/*
 * Cray Cassini ethernet driver
 * © Copyright 2018-2020 Cray Inc
 * © Copyright 2020-2021 Hewlett Packard Enterprise Development LP
 */

/* TODO:
 *
 * - address all the TODOs
 * - a lot of error handling
 * - support PCI device removal (see srcu usage in cxi_user_core.c)
 * - find a way to cleanly pass the hash types. Modify ethtool?
 * - add support for RSS contexts
 */

#include <linux/module.h>
#include <linux/netdevice.h>
#include <linux/etherdevice.h>
#include <linux/pci.h>

#include "cxi_eth.h"
#include "cxi_eth_debugfs.h"

unsigned int rss_indir_size = 64;
module_param(rss_indir_size, int, 0444);
MODULE_PARM_DESC(rss_indir_size, "Size of RSS indirection table");

unsigned int max_rss_queues = 16;
module_param(max_rss_queues, int, 0444);
MODULE_PARM_DESC(max_rss_queues, "Maximum number of RSS queues (power of 2)");

unsigned int max_tx_queues = 16;
module_param(max_tx_queues, int, 0444);
MODULE_PARM_DESC(max_tx_queues, "Maximum number of TX queues");

unsigned int small_pkts_buf_size = CXI_ETH_1_MB;
module_param(small_pkts_buf_size, uint, 0444);
MODULE_PARM_DESC(small_pkts_buf_size, "Size of a small packets buffer (4 KiB to 1 GiB, 4 KiB aligned)");

unsigned int small_pkts_buf_count = 4;
module_param(small_pkts_buf_count, uint, 0444);
MODULE_PARM_DESC(small_pkts_buf_count, "Number of small packets buffers (1 to 1024)");

unsigned int large_pkts_buf_count = 1024;
module_param(large_pkts_buf_count, uint, 0444);
MODULE_PARM_DESC(large_pkts_buf_count, "Number of large packets buffers (1 to 10000)");

unsigned int buffer_threshold = 256;
module_param(buffer_threshold, uint, 0444);
MODULE_PARM_DESC(buffer_threshold, "Ethernet packets with length greater than"
		 " or equal to this threshold are large packets (0 to 16383)");

unsigned int lpe_cdt_thresh_id = 2;
module_param(lpe_cdt_thresh_id, uint, 0444);
MODULE_PARM_DESC(lpe_cdt_thresh_id, "LPE append credit id to use (0 to 3)");

static unsigned int rx_repost_retry_ms = 20;
module_param(rx_repost_retry_ms, uint, 0444);
MODULE_PARM_DESC(rx_repost_retry_ms,
		 "Time to wait before retrying to repost an RX buffer, in ms");

static struct dentry *cxieth_debug_dir;
unsigned long rx_repost_retry_jiffies;

static int idc_dma_threshold_set(const char *val, const struct kernel_param *kp)
{
	unsigned long num;
	int ret;

	if (!val)
		return -EINVAL;

	ret = kstrtoul(val, 0, &num);
	if (ret || num > C_MAX_IDC_PAYLOAD_RES)
		return -EINVAL;

	*((unsigned int *)kp->arg) = num;

	return 0;
}

static const struct kernel_param_ops idc_dma_threshold_ops = {
	.set = idc_dma_threshold_set,
	.get = param_get_uint,
};
unsigned int idc_dma_threshold = C_MAX_IDC_PAYLOAD_RES;
module_param_cb(idc_dma_threshold, &idc_dma_threshold_ops,
		&idc_dma_threshold, 0644);
MODULE_PARM_DESC(idc_dma_threshold, "Ethernet packets up to this size will be"
		 " sent inline. Larger packets will use DMA. (0 to 224)");

/* List of devices registered with this client */
static LIST_HEAD(dev_list);
static DEFINE_MUTEX(dev_list_mutex);

static const struct net_device_ops cxi_eth_netdev_ops = {
	.ndo_open = cxi_eth_open,
	.ndo_start_xmit = cxi_eth_start_xmit,
	.ndo_stop = cxi_eth_close,
	.ndo_set_mac_address = cxi_eth_mac_addr,
	.ndo_set_rx_mode = cxi_eth_set_rx_mode,
	.ndo_change_mtu = cxi_change_mtu,
	.ndo_do_ioctl = cxi_do_ioctl,
};

/* Core is adding a new device */
static int add_device(struct cxi_dev *cxi_dev)
{
	struct net_device *ndev;
	struct cxi_eth *dev;
	int rc;

	/* Disable Ethernet devices on VF devices for now */
	if (!cxi_dev->is_physfn)
		return -ENODEV;

	ndev = alloc_etherdev_mqs(sizeof(struct cxi_eth), max_tx_queues,
				  max_rss_queues + 1);
	if (!ndev) {
		rc = -ENOMEM;
		goto err;
	}

	cxi_set_ethernet_threshold(cxi_dev, buffer_threshold);
	cxi_set_roce_rcv_seg(cxi_dev, false);

	SET_NETDEV_DEV(ndev, &cxi_dev->pdev->dev);

	dev = netdev_priv(ndev);

	dev->txqs = kcalloc(max_tx_queues, sizeof(*dev->txqs), GFP_KERNEL);
	if (!dev->txqs) {
		rc = -ENOMEM;
		goto err_free_ether;
	}

	dev->rxqs = kcalloc(max_rss_queues + 1, sizeof(*dev->rxqs), GFP_KERNEL);
	if (!dev->rxqs) {
		rc = -ENOMEM;
		goto err_free_txq;
	}

	dev->ndev = ndev;
	dev->cxi_dev = cxi_dev;
	spin_lock_init(&dev->cq_tgt_req_lock);
	dev->is_c2 = cassini_version(&dev->cxi_dev->prop, CASSINI_2);

	ndev->netdev_ops = &cxi_eth_netdev_ops;
	ndev->ethtool_ops = &cxi_eth_ethtool_ops;
	ndev->watchdog_timeo = CXI_ETH_TX_TIMEOUT;

	//ndev->irq = cxi_dev->pdev->irq;
	ndev->mtu = ETH_DATA_LEN;
	ndev->min_mtu = ETH_ZLEN;
	ndev->max_mtu = CXI_ETH_MAX_MTU;
	ndev->tx_queue_len = TX_QUEUE_LEN_DEFAULT;

	cxi_eth_devinfo(cxi_dev, &dev->eth_info);

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 14, 0)
	eth_hw_addr_set(ndev, dev->eth_info.default_mac_addr);
#else
	ether_addr_copy(ndev->dev_addr, dev->eth_info.default_mac_addr);
#endif

	dev->ringparam.rx_max_pending = LARGE_PKTS_BUF_COUNT_MAX;
	dev->ringparam.rx_mini_max_pending = num_small_packets(SMALL_PKTS_BUF_COUNT_MAX);
	dev->ringparam.tx_max_pending = CXI_MAX_CQ_COUNT;

	dev->ringparam.rx_pending = large_pkts_buf_count;
	dev->ringparam.rx_mini_pending = num_small_packets(small_pkts_buf_count);
	dev->ringparam.tx_pending = min_t(unsigned int, CXI_MAX_CQ_COUNT,
					  2 * dev->ndev->tx_queue_len);

	/* Used to set min_free to be equal or higher than buffer_threshold. */
	dev->min_free = roundup_pow_of_two(buffer_threshold) >>
		dev->eth_info.min_free_shift;

	ndev->hw_features = NETIF_F_HW_CSUM | NETIF_F_GSO | NETIF_F_SG |
		NETIF_F_GRO | NETIF_F_RXCSUM | NETIF_F_HIGHDMA |
		NETIF_F_RXHASH;

	ndev->features = ndev->hw_features;
	ndev->vlan_features = ndev->hw_features;
	ndev->priv_flags |= IFF_UNICAST_FLT | IFF_LIVE_ADDR_CHANGE;

	netif_set_real_num_rx_queues(ndev, 8);
	netif_set_real_num_tx_queues(ndev, 8);

	rc = register_netdev(ndev);
	if (rc)
		goto err_free_rxq;

	/* If the SerDes are up, we must report the carrier as on here,
	 * since we've already missed the link up event which was sent from
	 * cxi-ss1
	 */
	if (cxi_is_link_up(cxi_dev)) {
		/* Without the off->on transition, the ip link state will go to
		 * UNKNOWN.
		 */
		netif_carrier_off(ndev);
		netif_carrier_on(ndev);
	} else {
		netif_carrier_off(ndev);
	}

	device_debugfs_create(cxi_dev->name, dev, cxieth_debug_dir);

	mutex_lock(&dev_list_mutex);
	list_add_tail(&dev->dev_list, &dev_list);
	mutex_unlock(&dev_list_mutex);

	dev_info(&cxi_dev->pdev->dev, "%s[%s]: Added Ethernet device",
		 cxi_dev->name, netdev_name(ndev));

	return 0;

err_free_rxq:
	kfree(dev->rxqs);
err_free_txq:
	kfree(dev->txqs);
err_free_ether:
	free_netdev(ndev);
err:
	dev_err(&cxi_dev->pdev->dev,
		 "%s[]: Failure(%d): Ethernet device not added",
		 cxi_dev->name, rc);
	return rc;
}

/* From a given CXI device, find the ethernet device. Must be locked. */
static struct cxi_eth *find_eth_device(const struct cxi_dev *cxi_dev)
{
	struct cxi_eth *dev;

	list_for_each_entry_reverse(dev, &dev_list, dev_list) {
		if (dev->cxi_dev == cxi_dev)
			return dev;
	}

	return NULL;
}

static void remove_device(struct cxi_dev *cxi_dev)
{
	struct cxi_eth *dev;

	/* Ethernet devices on VF devices are not added for now, so not removed */
	if (!cxi_dev->is_physfn)
		return;

	mutex_lock(&dev_list_mutex);
	dev = find_eth_device(cxi_dev);
	if (dev)
		list_del(&dev->dev_list);
	mutex_unlock(&dev_list_mutex);

	if (!dev) {
		dev_info(&cxi_dev->pdev->dev,
			 "%s[]: Failure(NOT FOUND): no Ethernet device to remove",
			 cxi_dev->name);
		return;
	}

	debugfs_remove(dev->debug);

	unregister_netdev(dev->ndev);

	kfree(dev->rxqs);
	kfree(dev->txqs);
	free_netdev(dev->ndev);

	dev_info(&cxi_dev->pdev->dev, "%s[]: Removed Ethernet device",
		 cxi_dev->name);
}

static void async_event(struct cxi_dev *cxi_dev, enum cxi_async_event event)
{
	struct cxi_eth *dev;

	mutex_lock(&dev_list_mutex);
	dev = find_eth_device(cxi_dev);
	mutex_unlock(&dev_list_mutex);

	if (!dev)
		return;

	netdev_dbg(dev->ndev, "got async event %d\n", event);

	switch (event) {
	case CXI_EVENT_LINK_UP:
		netif_carrier_on(dev->ndev);
		break;

	case CXI_EVENT_LINK_DOWN:
		netif_carrier_off(dev->ndev);
		break;

	default:
		break;
	}
}

static int cxi_netdev_event(struct notifier_block *this, unsigned long event,
			    void *ptr)
{
	struct net_device *ndev = netdev_notifier_info_to_dev(ptr);

	if (ndev->netdev_ops->ndo_open == &cxi_eth_open &&
	    event == NETDEV_CHANGENAME) {
		struct cxi_eth *dev = netdev_priv(ndev);

		cxi_set_eth_name(dev->cxi_dev, netdev_name(ndev));
	}

	return NOTIFY_DONE;
}

static struct notifier_block cxi_netdev_notifier = {
	.notifier_call = cxi_netdev_event,
};

static struct cxi_client cxiu_client = {
	.add = add_device,
	.remove = remove_device,
	.async_event = async_event,
};

static int __init init(void)
{
	unsigned int new;
	int ret;

	if (rss_indir_size && (!is_power_of_2(rss_indir_size) ||
			   rss_indir_size > CXI_ETH_MAX_INDIR_ENTRIES)) {
		rss_indir_size = min_t(unsigned int,
				       roundup_pow_of_two(rss_indir_size),
				       CXI_ETH_MAX_INDIR_ENTRIES);
		pr_info("Adjusting rss_indir_size to %d\n", rss_indir_size);
	}

	if (max_rss_queues > CXI_ETH_MAX_RSS_QUEUES ||
	    !is_power_of_2(max_rss_queues)) {
		max_rss_queues = min_t(unsigned int,
				       roundup_pow_of_two(max_rss_queues),
				       CXI_ETH_MAX_RSS_QUEUES);
		pr_info("Adjusting max_rss_queues to %d\n", max_rss_queues);
	}

	if (max_tx_queues > CXI_ETH_MAX_TX_QUEUES) {
		max_tx_queues = CXI_ETH_MAX_TX_QUEUES;
		pr_info("Adjusting max_tx_queues to %d\n", max_tx_queues);
	}

	if (small_pkts_buf_size < PAGE_SIZE) {
		small_pkts_buf_size = PAGE_SIZE;
		pr_info("Adjusting small_pkts_buf_size to %u\n",
			small_pkts_buf_size);
	} else if (small_pkts_buf_size > CXI_ETH_1_GB) {
		small_pkts_buf_size = CXI_ETH_1_GB;
		pr_info("Adjusting small_pkts_buf_size to %u\n",
			small_pkts_buf_size);
	} else if ((small_pkts_buf_size % PAGE_SIZE) != 0) {
		small_pkts_buf_size = roundup(small_pkts_buf_size, PAGE_SIZE);
		pr_info("Adjusting small_pkts_buf_size to %u\n",
			small_pkts_buf_size);
	}

	new = clamp(small_pkts_buf_count,
		    SMALL_PKTS_BUF_COUNT_MIN, SMALL_PKTS_BUF_COUNT_MAX);
	if (new != small_pkts_buf_count) {
		small_pkts_buf_count = new;
		pr_info("Adjusting small_pkts_buf_count to %u\n",
			small_pkts_buf_count);
	}

	new = clamp(large_pkts_buf_count,
		    LARGE_PKTS_BUF_COUNT_MIN, LARGE_PKTS_BUF_COUNT_MAX);
	if (new != large_pkts_buf_count) {
		large_pkts_buf_count = new;
		pr_info("Adjusting large_pkts_buf_count to %u\n",
			large_pkts_buf_count);
	}

	if (buffer_threshold > 16383) {
		buffer_threshold = 16383;
		pr_info("Adjusting buffer_threshold to %u\n", buffer_threshold);
	}

	if (lpe_cdt_thresh_id > 3) {
		lpe_cdt_thresh_id = 0;
		pr_info("Resetting lpe_cdt_thresh_id to 0\n");
	}

	if (rx_repost_retry_ms < 1 || rx_repost_retry_ms > 5000) {
		rx_repost_retry_ms = 20;
		pr_info("Resetting rx_repost_retry_ms to 20ms\n");
	}
	rx_repost_retry_jiffies = msecs_to_jiffies(rx_repost_retry_ms);

	ret = register_netdevice_notifier(&cxi_netdev_notifier);
	if (ret) {
		pr_err("Failed to register netdevice_notifier\n");
		return -EINVAL;
	}

	cxieth_debug_dir = device_eth_debugfs_init();

	ret = cxi_register_client(&cxiu_client);
	if (ret) {
		pr_err("Couldn't register CXI ethernet driver\n");
		goto out;
	}

	return 0;

out:
	debugfs_remove(cxieth_debug_dir);
	unregister_netdevice_notifier(&cxi_netdev_notifier);

	return ret;
}

static void __exit cleanup(void)
{
	cxi_unregister_client(&cxiu_client);
	unregister_netdevice_notifier(&cxi_netdev_notifier);
	debugfs_remove(cxieth_debug_dir);
}

module_init(init);
module_exit(cleanup);

MODULE_LICENSE("GPL v2");
MODULE_DESCRIPTION("Cray eXascale Interconnect (CXI) Ethernet driver");
MODULE_AUTHOR("Cray Inc.");
