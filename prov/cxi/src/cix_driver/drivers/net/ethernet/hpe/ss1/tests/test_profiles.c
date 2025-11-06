// SPDX-License-Identifier: GPL-2.0
/* Copyright 2025 Hewlett Packard Enterprise Development LP */

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/delay.h>
#include <linux/hpe/cxi/cxi.h>
#include "cxi_core.h"
#include "cass_core.h"

static bool test_pass;

#ifdef pr_fmt
#undef pr_fmt
#endif
#define pr_fmt(fmt) KBUILD_MODNAME ":%s:%d " fmt, __func__, __LINE__

static struct cxi_dev *dev;

static struct cxi_rx_profile *rx_profile;
static struct cxi_tx_profile *tx_profile;

static struct cxi_client cxiu_client;
struct cxi_resource_use r_save[CXI_RESOURCE_MAX];

static int test_rx_profiles(struct cxi_dev *dev)
{
	int rc;
	int match = 0x20;
	int ignore = 0x11;
	struct cxi_rx_attr rx_attr = {};
	struct cxi_rx_attr rx_attr_ret = {};

	rx_attr.vni_attr.match = 2;
	rx_attr.vni_attr.ignore = 2;
	rx_profile = cxi_dev_alloc_rx_profile(dev, &rx_attr);
	if (!IS_ERR(rx_profile)) {
		rc = PTR_ERR(rx_profile);
		pr_info("Allocate RX profile should have failed");
		return rc;
	}

	rx_attr.vni_attr.match = 0;
	rx_attr.vni_attr.ignore = 0;
	rx_profile = cxi_dev_alloc_rx_profile(dev, &rx_attr);
	if (IS_ERR(rx_profile)) {
		rc = PTR_ERR(rx_profile);
		pr_info("Allocate RX profile failed rc:%d", rc);
		return rc;
	}

	rc = cxi_rx_profile_enable(dev, rx_profile);
	if (!rc) {
		pr_info("Enable RX profile should fail rc:%d", rc);
		rc = -1;
		goto remove_rx_profile;
	}

	/* Should be ok to disable even if it is disabled */
	cxi_rx_profile_disable(dev, rx_profile);

	rx_attr.vni_attr.match = match;
	rx_attr.vni_attr.ignore = ignore;
	rc = cxi_dev_set_rx_profile_attr(dev, rx_profile, &rx_attr);
	if (rc) {
		pr_info("Failed to set rx profile attributes rc:%d", rc);
		goto remove_rx_profile;
	}

	rc = cxi_dev_set_rx_profile_attr(dev, rx_profile, &rx_attr);
	if (!rc) {
		pr_info("Should fail to set rx profile attributes again rc:%d",
			rc);
		goto remove_rx_profile;
	}

	rc = cxi_rx_profile_get_info(dev, rx_profile, &rx_attr_ret, NULL);
	if (rc) {
		pr_info("Failed to get rx profile attributes rc:%d", rc);
		goto remove_rx_profile;
	}

	if (rx_attr.vni_attr.match != rx_attr_ret.vni_attr.match ||
	    rx_attr.vni_attr.ignore != rx_attr_ret.vni_attr.ignore) {
		pr_info("Rx profile attributes not equal");
		rc = -1;
		goto remove_rx_profile;
	}

	rc = cxi_rx_profile_enable(dev, rx_profile);
	if (rc) {
		pr_info("Failed enabling RX profile rc:%d", rc);
		goto remove_rx_profile;
	}

	/* Should be ok to disable before the dec_refcount */
	cxi_rx_profile_disable(dev, rx_profile);

remove_rx_profile:
	cxi_rx_profile_dec_refcount(dev, rx_profile);

	return rc;
}

static int test_tx_profiles(struct cxi_dev *dev)
{
	int rc;
	int match = 0x40;
	int ignore = 0x11;
	struct cxi_tx_attr tx_attr = {};
	struct cxi_tx_attr tx_attr_ret = {};

	tx_attr.vni_attr.match = 1;
	tx_attr.vni_attr.ignore = 1;
	tx_profile = cxi_dev_alloc_tx_profile(dev, &tx_attr);
	if (!IS_ERR(tx_profile)) {
		rc = PTR_ERR(tx_profile);
		pr_info("Allocate TX profile should have failed");
		return rc;
	}

	tx_attr.vni_attr.match = 0;
	tx_attr.vni_attr.ignore = 0;
	tx_profile = cxi_dev_alloc_tx_profile(dev, &tx_attr);
	if (IS_ERR(tx_profile)) {
		rc = PTR_ERR(tx_profile);
		pr_info("Allocate TX profile failed rc:%d", rc);
		return rc;
	}

	rc = cxi_tx_profile_enable(dev, tx_profile);
	if (!rc) {
		pr_info("Allocate TX profile should fail rc:%d", rc);
		rc = -1;
		goto remove_tx_profile;
	}

	tx_attr.vni_attr.match = match;
	tx_attr.vni_attr.ignore = ignore;
	rc = cxi_dev_set_tx_profile_attr(dev, tx_profile, &tx_attr);
	if (rc) {
		pr_info("Failed to set tx profile attributes rc:%d", rc);
		goto remove_tx_profile;
	}

	rc = cxi_dev_set_tx_profile_attr(dev, tx_profile, &tx_attr);
	if (!rc) {
		pr_info("Should fail to set tx profile attributes again rc:%d",
			rc);
		goto remove_tx_profile;
	}

	rc = cxi_tx_profile_get_info(dev, tx_profile, &tx_attr_ret, NULL);
	if (rc) {
		pr_info("Failed to get tx profile attributes rc:%d", rc);
		goto remove_tx_profile;
	}

	if (tx_attr.vni_attr.match != tx_attr_ret.vni_attr.match ||
	    tx_attr.vni_attr.ignore != tx_attr_ret.vni_attr.ignore) {
		pr_info("Tx profile attributes not equal");
		goto remove_tx_profile;
	}

remove_tx_profile:
	cxi_tx_profile_dec_refcount(dev, tx_profile, true);

	return rc;
}

static int run_tests(struct cxi_dev *dev)
{
	int rc;
	struct cass_dev *hw = cxi_to_cass_dev(dev);
	int refcount_final;
	int refcount = refcount_read(&hw->refcount);

	rc = test_rx_profiles(dev);
	if (rc) {
		test_pass = false;
		pr_err("test_rx_profiles failed: %d\n", rc);
		goto done;
	}

	rc = test_tx_profiles(dev);
	if (rc) {
		test_pass = false;
		pr_err("test_tx_profiles failed: %d\n", rc);
		goto done;
	}

	refcount_final = refcount_read(&hw->refcount);

	if (refcount != refcount_final) {
		pr_err("hw refcount %s was:%d now:%d\n",
		       refcount < refcount_final ? "leak" : "missing",
		       refcount, refcount_final);
		test_pass = false;
		rc = -EIO;
	}

	pr_info("All tests pass\n");

done:
	return rc;
}

static int add_device(struct cxi_dev *cdev)
{
	dev = cdev;
	return 0;

}

static void remove_device(struct cxi_dev *cdev)
{
	dev = NULL;
}

static struct cxi_client cxiu_client = {
	.add = add_device,
	.remove = remove_device,
};

static int __init init(void)
{
	int rc;

	test_pass = true;

	rc = cxi_register_client(&cxiu_client);
	if (rc) {
		pr_err("cxi_register_client failed: %d\n", rc);
		return rc;
	}

	rc = run_tests(dev);
	if (rc)
		pr_err("Failed:%d\n", rc);

	if (!test_pass) {
		cxi_unregister_client(&cxiu_client);
		rc = -EIO;
	}

	return rc;
}

static void __exit cleanup(void)
{
	pr_info("unregistering client\n");
	cxi_unregister_client(&cxiu_client);
}

module_init(init);
module_exit(cleanup);

MODULE_LICENSE("GPL v2");
MODULE_DESCRIPTION("CXI profile test suite");
MODULE_AUTHOR("HPE");
