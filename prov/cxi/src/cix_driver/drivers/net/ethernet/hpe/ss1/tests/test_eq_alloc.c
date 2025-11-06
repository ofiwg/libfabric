// SPDX-License-Identifier: GPL-2.0
/* Copyright 2021 Hewlett Packard Enterprise Development LP */

#include <linux/hpe/cxi/cxi.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/slab.h>

static int test_1_realloc_eq_same_reserved_slots(struct cxi_dev *cdev)
{
	struct cxi_lni *lni;
	struct cxi_eq_attr eq_attr = {};
	struct cxi_eq *eq;
	size_t eq_buf_size = 4096;
	void *eq_buf;
	int rc;

	lni = cxi_lni_alloc(cdev, CXI_DEFAULT_SVC_ID);
	if (IS_ERR(lni)) {
		rc = PTR_ERR(lni);
		pr_err("cxi_lni_alloc() failed: %d\n", rc);
		return rc;
	}

	eq_buf = kmalloc(eq_buf_size, GFP_KERNEL);
	if (!eq_buf) {
		rc = -ENOMEM;
		pr_err("kmalloc() failed");
		goto err_free_lni;
	}

	eq_attr.queue = eq_buf;
	eq_attr.queue_len = eq_buf_size;
	eq_attr.flags = CXI_EQ_PASSTHROUGH;
	eq_attr.reserved_slots = (eq_buf_size / C_EE_CFG_ECB_SIZE) - 1;

	eq = cxi_eq_alloc(lni, NULL, &eq_attr, NULL, NULL, NULL, NULL);
	if (IS_ERR(eq)) {
		rc = PTR_ERR(eq);
		pr_err("%d:%s cxi_eq_alloc() failed: %d", __LINE__, __func__,
		       rc);
		goto err_free_eq_buf;
	}

	cxi_eq_free(eq);

	eq = cxi_eq_alloc(lni, NULL, &eq_attr, NULL, NULL, NULL, NULL);
	if (IS_ERR(eq)) {
		rc = PTR_ERR(eq);
		pr_err("%d:%s cxi_eq_alloc() failed: %d", __LINE__, __func__,
		       rc);
		goto err_free_eq_buf;
	}

	cxi_eq_free(eq);

	rc = 0;

err_free_eq_buf:
	kfree(eq_buf);

err_free_lni:
	cxi_lni_free(lni);

	return rc;
}

/* Core is adding a new device */
static int add_device(struct cxi_dev *cdev)
{
	int rc;

	pr_info("Using device %s\n", cdev->name);

	rc = test_1_realloc_eq_same_reserved_slots(cdev);
	if (rc)
		pr_err("test_1_realloc_eq_same_reserved_slots failed: %d\n",
		       rc);
	else
		pr_info("test_1_realloc_eq_same_reserved_slots succeeded\n");

	return 0;
}

static void remove_device(struct cxi_dev *dev)
{
}

static struct cxi_client client = {
	.add = add_device,
	.remove = remove_device,
};

static int __init init(void)
{
	int rc;

	rc = cxi_register_client(&client);
	if (rc)
		pr_err("cxi_register_client() failed: %d\n", rc);

	return rc;
}

static void __exit cleanup(void)
{
	cxi_unregister_client(&client);
}

module_init(init);
module_exit(cleanup);

MODULE_AUTHOR("Hewlett Packard Enterprise Development LP");
MODULE_DESCRIPTION("CXI EQ test suite");
MODULE_LICENSE("GPL v2");
