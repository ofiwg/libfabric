// SPDX-License-Identifier: GPL-2.0
/* Copyright 2021,2024 Hewlett Packard Enterprise Development LP */

/* Test driver for CXI TELEM API functionality. */

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/hpe/cxi/cxi.h>
#include <linux/delay.h>

#include "cassini-telemetry-items.h"
#include "cassini-telemetry-test.h"

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

static void testcase_cxi_telem_api_00(struct cxi_dev *dev,
				      const unsigned int *items,
				      u64 *data,
				      unsigned int count)
{
	int retval;
	int i;

	pr_err("TESTCASE_CXI_TELEM_API_00: START\n");

	for (i = 0 ; i < count + 1 ; ++i)
		data[i] = ~0ULL;

	retval = cxi_telem_get_selected(dev, items, data, 0);
	if (retval != -EINVAL)
		pr_err("TEST-ERROR: %d: unexpected retval %d for get selected "
		       "instead of %d\n",
		       __LINE__, retval, -EINVAL);
	for (i = 0 ; i < count + 1 ; ++i)
		if (data[i] != ~0ULL)
			pr_err("TEST-ERROR: %d: unexpected change in "
			       "data[%d]=0x%016llx\n",
			       __LINE__, i, data[i]);

	retval = cxi_telem_get_selected(dev, items, data, count + 1);
	if (retval != -EINVAL)
		pr_err("TEST-ERROR: %d: unexpected retval %d for get selected "
		       "instead of %d\n",
		       __LINE__, retval, -EINVAL);
	for (i = 0 ; i < count + 1 ; ++i)
		if (data[i] != ~0ULL)
			pr_err("TEST-ERROR: %d: unexpected change in "
			       "data[%d]=0x%016llx\n",
			       __LINE__, i, data[i]);

	retval = cxi_telem_get_selected(dev, items, data, count);
	if (retval != 0)
		pr_err("TEST-ERROR: %d: unexpected retval %d for get selected "
		       "instead of %d\n",
		       __LINE__, retval, 0);
	for (i = 0 ; i < count ; ++i)
		if (data[i] == ~0ULL)
			pr_err("TEST-ERROR: %d: data[%d]=0x%016llx did not "
			       "change\n",
			       __LINE__, i, data[i]);

	if (data[count] != ~0ULL)
		pr_err("TEST-ERROR: %d: unexpected change in "
		       "data[%d]=0x%016llx\n",
		       __LINE__, count, data[i]);

	pr_err("TESTCASE_CXI_TELEM_API_00: FINISH\n");
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/*
 * Core is adding a new device.
 */
static int add_device(struct cxi_dev *dev)
{
	unsigned int count;
	const unsigned int *items;
	u64 *data;

	if (cassini_version(&dev->prop, CASSINI_2)) {
		count = C2_TELEM_SIZE;
		items = c2items;
	} else {
		count = C1_TELEM_SIZE;
		items = c1items;
	}

	data = kmalloc((count + 1) * sizeof(u64), GFP_KERNEL);
	if (data == NULL)
		return 1;
	pr_err("TESTSUITE_CXI_TELEM_API: START\n");
	testcase_cxi_telem_api_00(dev, items, data, count);
	pr_err("TESTSUITE_CXI_TELEM_API: FINISH\n");
	kfree(data);
	return 0;
}

static void remove_device(struct cxi_dev *dev)
{
}

static struct cxi_client cxiu_client = {
	.add	= add_device,
	.remove = remove_device,
};

static int __init init(void)
{
	int retval;

	retval = cxi_register_client(&cxiu_client);
	if (retval != 0)
		pr_err("Could not register client\n");
	return retval;
}

static void __exit cleanup(void)
{
	cxi_unregister_client(&cxiu_client);
}

module_init(init);
module_exit(cleanup);

MODULE_LICENSE("GPL v2");
MODULE_DESCRIPTION("CXI CNTRS API test driver");
MODULE_AUTHOR("Cray Inc.");
