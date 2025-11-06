// SPDX-License-Identifier: GPL-2.0
/* Copyright 2024 Hewlett Packard Enterprise Development LP */

#include <linux/hpe/cxi/cxi.h>
#include <linux/delay.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include <linux/dma-mapping.h>
#include <linux/pci.h>
#include <linux/bvec.h>
#include <uapi/ethernet/cxi-abi.h>
#include <linux/version.h>

#include "cxi_prov_hw.h"
#include "cxi_core.h"
#include "cass_core.h"

static struct cxi_dev *cdev;

#define PLEC_SIZE 256U
#define PTE_COUNT (PLEC_SIZE + 1)

static struct cxi_lni *lni;
static struct cxi_pte *ptes[PTE_COUNT] = {};

static int test_1_verify_plec_limit(void)
{
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	int rc = 0;
	struct cxi_pte_priv *pte_priv;
	struct cxi_pt_alloc_opts opts = {};
	int i;

	if (atomic_read(&hw->plec_count)) {
		pr_err("PLEC in-use count non-zero\n");
		return -EINVAL;
	}

	lni = cxi_lni_alloc(cdev, CXI_DEFAULT_SVC_ID);
	if (IS_ERR(lni)) {
		rc = PTR_ERR(lni);
		pr_err("cxi_lni_alloc failed: %d\n", rc);
		goto out;
	}

	for (i = 0; i < PTE_COUNT; i++) {
		ptes[i] = cxi_pte_alloc(lni, NULL, &opts);
		if (IS_ERR(ptes[i])) {
			rc = PTR_ERR(ptes[i]);
			pr_err("cxi_pte_alloc failed: %d\n", rc);
			goto free_ptes;
		}

		pte_priv = container_of(ptes[i], struct cxi_pte_priv, pte);
		if (i > PLEC_SIZE && pte_priv->plec_enabled) {
			rc = -EINVAL;
			pr_err("PtlTE %d incorrectly marked for PLEC enabled\n", i);
			goto free_ptes;
		} else if (!pte_priv->plec_enabled) {
			rc = -EINVAL;
			pr_err("PtlTE %d incorrectly marked for PLEC disabled\n", i);
			goto free_ptes;
		}
	}

free_ptes:
	for (i = 0; i < PTE_COUNT; i++) {
		if (!IS_ERR_OR_NULL(ptes[i]))
			cxi_pte_free(ptes[i]);
	}

	if (atomic_read(&hw->plec_count)) {
		pr_err("PLEC in-use count non-zero\n");
		rc = -EINVAL;
	}

	cxi_lni_free(lni);

out:
	return rc;
}

static int add_device(struct cxi_dev *dev)
{
	cdev = dev;
	return 0;
}

static void remove_device(struct cxi_dev *dev)
{
	cdev = NULL;
}

static struct cxi_client cxi_client = {
	.add = add_device,
	.remove = remove_device,
};

static int __init init(void)
{
	int rc;

	rc = cxi_register_client(&cxi_client);
	if (rc) {
		pr_err("Couldn't register client\n");
		goto out;
	}

	rc = test_1_verify_plec_limit();
	if (rc)
		pr_err("test_1_verify_plec_limit failed: %d\n", rc);

	cxi_unregister_client(&cxi_client);

out:
	return rc;
}

static void __exit cleanup(void)
{

}

module_init(init);
module_exit(cleanup);

MODULE_LICENSE("GPL v2");
MODULE_DESCRIPTION("Cray eXascale Interconnect (CXI) PtlTE test driver");
MODULE_AUTHOR("Hewlett Packard Enterprise Development LP");
