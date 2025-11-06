// SPDX-License-Identifier: GPL-2.0
/* Copyright 2019 Cray Inc. All rights reserved */

/* Test the PF/VF communication. The requirement for this test is that
 * the VF is also accessible in the host. ie. the VF is not given to a
 * virtual machine.
 *
 * This test could be split into 2 drivers at some point. One to
 * handle the PF part, and the other for the VFs.
 */

#include <linux/module.h>
#include <linux/pci.h>
#include <linux/slab.h>
#include <linux/workqueue.h>
#include <uapi/ethernet/cxi-abi.h>
#include <linux/hpe/cxi/cxi.h>

#include <cxi_prov_hw.h>

#include "cxi_core.h"

/* List of devices registered with this client */
static LIST_HEAD(dev_list);
static DEFINE_MUTEX(dev_list_mutex);

/* Request and reply. Must be odd length for now. The NUL character
 * will make the message of even length.
 */
static const char *request_msg = "hello there";
static const char *reply_msg =   "what's up buddy";

/* Keep track of known devices. Protected by dev_list_mutex. */
struct tdev {
	struct list_head dev_list;
	struct cxi_dev *dev;

	struct delayed_work test_work;
};

/* Read a message from a VF. This would be in cxi-user. */
static int msg_relay(void *data, unsigned int vf_num,
		     const void *req, size_t req_len,
		     void *rsp, size_t *rsp_len)
{
	pr_info("Got message from VF %d, len %zu\n", vf_num, req_len);

	BUG_ON(data == NULL);

	/* Process the request */
	if (req_len != strlen(request_msg) + 1)
		pr_err("BAD: Unexpected request length\n");
	else if (memcmp(request_msg, req, req_len))
		pr_err("BAD: Request has unexpected data\n");
	else
		pr_info("Request is valid\n");

	/* Prepare the reply */
	*rsp_len = strlen(reply_msg) + 1;
	strcpy(rsp, reply_msg);

	return 0;
}

/* Send a message to the PF, which will be relayed to msg_relay(), and
 * get the reply. This function is only run on a VF.
 */
static void test_work(struct work_struct *work)
{
	struct tdev *tdev = container_of(work, struct tdev, test_work.work);
	struct cxi_dev *dev = tdev->dev;
	char reply[100];
	size_t reply_len;
	int rc;

	reply_len = sizeof(reply);
	rc = cxi_send_msg_to_pf(dev, request_msg, strlen(request_msg) + 1,
				reply, &reply_len);

	if (rc != 0)
		pr_err("BAD: Reply has return code %d\n", rc);
	else if (reply_len != strlen(reply_msg) + 1)
		pr_err("BAD: Reply has unexpected length %zu\n", reply_len);
	else if (memcmp(reply_msg, reply, strlen(reply_msg) + 1))
		pr_err("BAD: Reply has unexpected data\n");
	else
		pr_info("Reply is valid\n");

	pr_info("Test done\n");
}

/* Core is adding a new device */
static int add_device(struct cxi_dev *cdev)
{
	struct tdev *tdev;
	int rc;

	tdev = kzalloc(sizeof(*tdev), GFP_KERNEL);
	if (tdev == NULL)
		return -ENOMEM;

	tdev->dev = cdev;

	if (cdev->is_physfn) {
		rc = cxi_register_msg_relay(cdev, msg_relay, tdev);
		if (rc) {
			dev_err(&cdev->pdev->dev, "BAD: msg_relay registration failed\n");
			kfree(tdev);
			return rc;
		}
	} else {
		INIT_DELAYED_WORK(&tdev->test_work, test_work);
		schedule_delayed_work(&tdev->test_work, 2 * HZ);
	}

	mutex_lock(&dev_list_mutex);
	list_add_tail(&tdev->dev_list, &dev_list);
	mutex_unlock(&dev_list_mutex);

	return 0;
}

static void remove_device(struct cxi_dev *dev)
{
	struct tdev *tdev;
	bool found = false;

	/* Find the device in the list */
	mutex_lock(&dev_list_mutex);
	list_for_each_entry_reverse(tdev, &dev_list, dev_list) {
		if (tdev->dev == dev) {
			found = true;
			list_del(&tdev->dev_list);
			break;
		}
	}
	mutex_unlock(&dev_list_mutex);

	if (!found)
		return;

	if (dev->is_physfn)
		cxi_unregister_msg_relay(dev);
	else
		cancel_delayed_work_sync(&tdev->test_work);

	kfree(tdev);

	pr_info("Removing VF/PF comm client device %s\n", dev->name);
}

static struct cxi_client cxiu_client = {
	.add = add_device,
	.remove = remove_device,
};

static int __init init(void)
{
	int ret;

	ret = cxi_register_client(&cxiu_client);
	if (ret) {
		pr_err("BAD: Couldn't register client\n");
		goto out;
	}

	return 0;

out:
	return ret;
}

static void __exit cleanup(void)
{
	cxi_unregister_client(&cxiu_client);
}

module_init(init);
module_exit(cleanup);

MODULE_LICENSE("GPL v2");
MODULE_DESCRIPTION("Cray eXascale Interconnect (CXI) VF/PF test driver");
MODULE_AUTHOR("Cray Inc.");
