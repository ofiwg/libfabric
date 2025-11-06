// SPDX-License-Identifier: GPL-2.0
/* Copyright 2018,2024 Hewlett Packard Enterprise Development LP */

#include <linux/module.h>
#include <linux/debugfs.h>
#include <linux/moduleparam.h>
#include <linux/pci.h>

#include "cxi_core.h"

#define DEFAULT_PID_BITS MAX_PID_BITS

/* TODO: Make this a per device configurable. */
unsigned int pid_bits = DEFAULT_PID_BITS;
module_param(pid_bits, uint, 0444);
MODULE_PARM_DESC(pid_bits, "Width of the PID field within a NIC address");

/* System info to determine if a system is mixed (C1 & C2)
 * or homogeneous (only C1 or only C2).
 */
unsigned int system_type_identifier;
module_param(system_type_identifier, uint, 0444);
MODULE_PARM_DESC(system_type_identifier,
		"Identifies what type of Cassini NICs are present in the system\n"
		"0 - Cassini 1 and Cassini 2 (Default)\n"
		"1 - Cassini 1 Only\n"
		"2 - Cassini 2 Only\n");

unsigned int min_free_shift;
module_param(min_free_shift, uint, 0444);
MODULE_PARM_DESC(min_free_shift, "min_free shift (0 to 8)");

unsigned int vni_matching;
module_param(vni_matching, uint, 0444);
MODULE_PARM_DESC(vni_matching, "vni matching configuration (0 to 3)");

unsigned int tg_threshold[4] = { 0, 64, 128, 256 };
module_param_array(tg_threshold, uint, NULL, 0644);
MODULE_PARM_DESC(tg_threshold, "Array of 4 LPE append credits. Credits are 0 for immediate append, or 4 to 16383 for delayed append.");

int untagged_eth_pcp = -1;
module_param(untagged_eth_pcp, int, 0444);
MODULE_PARM_DESC(untagged_eth_pcp, "PCP used by untagged Ethernet frames. Valid PCPs 0-7. Use -1 to use the PCP specified in chosen QOS Profile");

bool switch_connected = true;
module_param(switch_connected, bool, 0644);
MODULE_PARM_DESC(switch_connected, "Connected to a switch or another NIC");

struct class cxi_class = {
	.name = "cxi",
};

/* For device naming purposes */
atomic_t cxi_num = ATOMIC_INIT(-1);

/* root debugfs directory for all devices */
struct dentry *cxi_debug_dir;

/* List of devices and its mutex. */
static LIST_HEAD(dev_list);
static DEFINE_MUTEX(dev_list_mutex);

/* List of clients. Protected by dev_list_mutex. */
static LIST_HEAD(client_list);

/* Return the first VF */
struct pci_dev *cxi_get_vf0_dev(void)
{
	struct cxi_dev *dev;

	list_for_each_entry(dev, &dev_list, dev_list)
		if (dev->pdev->is_virtfn)
			return dev->pdev;

	return ERR_PTR(-ENODEV);
}

int cxi_get_lpe_append_credits(unsigned int lpe_cdt_thresh_id)
{
	if (lpe_cdt_thresh_id >= ARRAY_SIZE(tg_threshold))
		return -EINVAL;
	return tg_threshold[lpe_cdt_thresh_id];
}
EXPORT_SYMBOL(cxi_get_lpe_append_credits);

/**
 * cxi_register_client() - Register a CXI client
 *
 * @client: client to be registered
 */
int cxi_register_client(struct cxi_client *client)
{
	struct cxi_dev *dev;

	mutex_lock(&dev_list_mutex);

	list_add_tail(&client->list, &client_list);

	list_for_each_entry(dev, &dev_list, dev_list)
		if (client->add)
			client->add(dev);

	mutex_unlock(&dev_list_mutex);

	return 0;
}
EXPORT_SYMBOL(cxi_register_client);

/**
 * cxi_unregister_client() - Unregister a CXI client
 *
 * @client: client to be unregistered
 */
void cxi_unregister_client(struct cxi_client *client)
{
	struct cxi_dev *dev;

	mutex_lock(&dev_list_mutex);

	list_for_each_entry_reverse(dev, &dev_list, dev_list)
		if (client->remove)
			client->remove(dev);

	list_del(&client->list);

	mutex_unlock(&dev_list_mutex);
}
EXPORT_SYMBOL(cxi_unregister_client);

/* A device driver is declaring a new CXI device */
void cxi_add_device(struct cxi_dev *cdev)
{
	struct cxi_client *client;

	mutex_lock(&dev_list_mutex);
	list_add_tail(&cdev->dev_list, &dev_list);

	list_for_each_entry(client, &client_list, list)
		if (client->add)
			client->add(cdev);

	mutex_unlock(&dev_list_mutex);
}

/* A device driver signals it's going away. */
void cxi_remove_device(struct cxi_dev *cdev)
{
	struct cxi_client *client;

	mutex_lock(&dev_list_mutex);
	list_del(&cdev->dev_list);

	list_for_each_entry_reverse(client, &client_list, list)
		if (client->remove)
			client->remove(cdev);

	mutex_unlock(&dev_list_mutex);
}

/* Send an asynchronous event to the clients */
void cxi_send_async_event(struct cxi_dev *cdev, enum cxi_async_event event)
{
	struct cxi_client *client;

	mutex_lock_nested(&dev_list_mutex, SINGLE_DEPTH_NESTING);

	list_for_each_entry(client, &client_list, list)
		if (client->async_event)
			client->async_event(cdev, event);

	mutex_unlock(&dev_list_mutex);
}

void cxi_apply_for_all(void (*callback)(struct cxi_dev *dev, void *p), void *p)
{
	struct cxi_dev *dev;

	mutex_lock(&dev_list_mutex);

	list_for_each_entry(dev, &dev_list, dev_list)
		(*callback)(dev, p);

	mutex_unlock(&dev_list_mutex);
}

static int __init cxi_init(void)
{
	int rc;
	int i;

	pr_info("CXI driver revision %s\n", CXI_COMMIT);

	if (pid_bits > MAX_PID_BITS) {
		pr_warn("PID count (%u) exceeded max (%u). Using default (%u).\n",
			pid_bits, MAX_PID_BITS, DEFAULT_PID_BITS);
		pid_bits = DEFAULT_PID_BITS;
	}

	if (pid_bits < MIN_PID_BITS) {
		pr_warn("PID bits (%u) less than min (%u). Using default (%u).\n",
			pid_bits, MIN_PID_BITS, DEFAULT_PID_BITS);
		pid_bits = DEFAULT_PID_BITS;
	}

	if (untagged_eth_pcp > 7 || untagged_eth_pcp < -1) {
		pr_err("Invalid untagged Ethernet PCP value: %u\n",
		       untagged_eth_pcp);
		return -EINVAL;
	}

	if (min_free_shift > 8) {
		pr_warn("min_free_shift (%d) must be 0 to 8.\n",
			min_free_shift);
		return -EINVAL;
	}

	if (vni_matching > 3) {
		pr_warn("vni_matching (%d) must be 0 to 3.\n", vni_matching);
		return -EINVAL;
	}

	for (i = 0; i < ARRAY_SIZE(tg_threshold); i++) {
		if ((tg_threshold[i] >= 1 && tg_threshold[i] <= 3) ||
		    tg_threshold[i] > 16383) {
			pr_warn("tg_threshold must be 0 or 4 to 16383\n");
			return -EINVAL;
		}
	}

	cxi_debug_dir = debugfs_create_dir("cxi", NULL);

	rc = class_register(&cxi_class);
	if (rc) {
		pr_warn("Couldn't create Cray eXascale Interconnect device class\n");
		goto err_class_reg;
	}

	rc = cxi_configfs_subsys_init();
	if (rc) {
		pr_err("Error while registering configfs subsystem\n");
		goto err_class_reg;
	}

	rc = hw_register();
	if (rc < 0) {
		pr_err("Driver registration failed\n");
		goto err_drv_reg;
	}

	if (!(system_type_identifier == CASSINI_1_ONLY ||
		system_type_identifier == CASSINI_2_ONLY ||
		system_type_identifier == CASSINI_MIX)) {
		pr_warn("Unknown system_type_identifier (%u). Using default (%u).\n",
				system_type_identifier, CASSINI_MIX);
		system_type_identifier = CASSINI_MIX;
	}

	return 0;

err_drv_reg:
	class_unregister(&cxi_class);

err_class_reg:
	debugfs_remove(cxi_debug_dir);

	cxi_p2p_fini();

	return rc;
}

static void __exit cxi_exit(void)
{
	hw_unregister();
	class_unregister(&cxi_class);
	cxi_configfs_exit();
	debugfs_remove(cxi_debug_dir);
	cxi_p2p_fini();
}

module_init(cxi_init);
module_exit(cxi_exit);

MODULE_LICENSE("GPL v2");
MODULE_DESCRIPTION("Cray Cassini Nic Driver");
MODULE_AUTHOR("Cray Inc.");
