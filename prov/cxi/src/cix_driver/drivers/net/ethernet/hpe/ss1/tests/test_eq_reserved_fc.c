// SPDX-License-Identifier: GPL-2.0
/* Copyright 2021 Hewlett Packard Enterprise Development LP */

#include <linux/hpe/cxi/cxi.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/slab.h>

struct reserved_fc_values {
	int adjust_value;
	int expected_rc;
};

static int test_reserve_fc_values(struct cxi_dev *cdev)
{
	size_t eq_size = 2 * 1024 * 1024;
	struct cxi_eq_attr eq_attr = {};
	struct cxi_lni *lni;
	void *eq_buf;
	struct cxi_eq *eq;
	int value;
	int expected_rc;
	struct reserved_fc_values bad_reserved_fc_values[] = {
		{
			.adjust_value = (int)BIT(14),
			.expected_rc = -EINVAL,
		},
		{
			.adjust_value = -1 * (int)BIT(14),
			.expected_rc = -EINVAL,
		},
	};
	struct reserved_fc_values good_reserved_fc_values[] = {
		{
			.adjust_value = (int)BIT(14) - 1,
			.expected_rc = (int)BIT(14) - 1,
		},
		{
			.adjust_value = -1,
			.expected_rc = (int)BIT(14) - 2,
		},
		{
			.adjust_value = -1 * ((int)BIT(14) - 2),
			.expected_rc = 0,
		},
	};
	int rc;
	int i;

	pr_info("Using device %s\n", cdev->name);

	lni = cxi_lni_alloc(cdev, CXI_DEFAULT_SVC_ID);
	if (IS_ERR(lni)) {
		rc = PTR_ERR(lni);
		pr_err("cxi_lni_alloc() failed: %d\n", rc);
		goto err_out;
	}

	eq_buf = kmalloc(eq_size, GFP_KERNEL);
	if (!eq_buf) {
		rc = -ENOMEM;
		pr_err("kmalloc() failed: %d", rc);
		goto err_free_lni;
	}

	eq_attr.queue = eq_buf;
	eq_attr.queue_len = eq_size;
	eq_attr.flags = CXI_EQ_PASSTHROUGH;

	eq = cxi_eq_alloc(lni, NULL, &eq_attr, NULL, NULL, NULL, NULL);
	if (IS_ERR(eq)) {
		rc = PTR_ERR(eq);
		pr_err("cxi_eq_alloc() failed: %d", rc);
		goto err_free_eq_buf;
	}

	/* Test invalid reserve FC values. */
	rc = 0;
	for (i = 0; i < ARRAY_SIZE(bad_reserved_fc_values); i++) {
		value = bad_reserved_fc_values[i].adjust_value;
		expected_rc = bad_reserved_fc_values[i].expected_rc;
		if (cxi_eq_adjust_reserved_fc(eq, value) != expected_rc) {
			rc = -EINVAL;
			pr_err("cxi_eq_adjust_reserved_fc() did not return %d with %d value\n",
			       expected_rc, value);
			break;
		}
	}

	if (rc)
		pr_err("%s failed: %d\n", __func__, rc);

	/* Test valid reserve FC values. */
	rc = 0;
	for (i = 0; i < ARRAY_SIZE(good_reserved_fc_values); i++) {
		value = good_reserved_fc_values[i].adjust_value;
		expected_rc = good_reserved_fc_values[i].expected_rc;
		if (cxi_eq_adjust_reserved_fc(eq, value) != expected_rc) {
			rc = -EINVAL;
			pr_err("cxi_eq_adjust_reserved_fc() did not return %d with %d value\n",
			       expected_rc, value);
			break;
		}
	}


	if (rc)
		pr_err("%s failed: %d\n", __func__, rc);

	cxi_eq_free(eq);
err_free_eq_buf:
	kfree(eq_buf);
err_free_lni:
	cxi_lni_free(lni);
err_out:
	return rc;
}

/* Core is adding a new device */
static int add_device(struct cxi_dev *dev)
{
	int rc;

	rc = test_reserve_fc_values(dev);
	if (rc)
		pr_err("reserved fc tests failed: %d\n", rc);
	else
		pr_info("reserved fc tests succeeded\n");

	return rc;
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
