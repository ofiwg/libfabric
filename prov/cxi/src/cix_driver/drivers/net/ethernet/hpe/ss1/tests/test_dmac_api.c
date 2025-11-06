// SPDX-License-Identifier: GPL-2.0
/* Copyright 2021 Hewlett Packard Enterprise Development LP */

/* Test driver for CXI DMAC API functionality. */

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/hpe/cxi/cxi.h>
#include <linux/delay.h>

#include "cass_core.h"

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/*
 * Since this test module (test dma api) is run/loaded into the system with
 * the cassini sshot11 module loaded as in a standard system,
 * a consequence is that some dmac resources are already in use.
 * Thus, these tests are somewhat limited in their ability to do complete
 * coverage since the testing starting condition is _NOT_ with all dmac
 * resources free.
 */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

static void testcase_cxi_dmac_api_00(struct cxi_dev *dev)
{
	int retval;

	pr_err("TESTCASE_CXI_DMAC_API_00: START\n");

	retval = cxi_dmac_desc_set_alloc(dev, 1, NULL);
	if (retval < 0) {
		pr_err("TEST-ERROR: alloc failed with %d\n", retval);
		goto __testcase_cxi_dmac_api_00_A;
	}
	retval = cxi_dmac_desc_set_free(dev, retval);
	if (retval < 0) {
		pr_err("TEST-ERROR: free failed with %d\n", retval);
		goto __testcase_cxi_dmac_api_00_A;
	}

	/*
	 * verify still functioning after one usage pair.
	 */

	retval = cxi_dmac_desc_set_alloc(dev, 1, NULL);
	if (retval < 0) {
		pr_err("TEST-ERROR: alloc failed with %d\n", retval);
		goto __testcase_cxi_dmac_api_00_A;
	}
	retval = cxi_dmac_desc_set_free(dev, retval);
	if (retval < 0) {
		pr_err("TEST-ERROR: free failed with %d\n", retval);
		/* falling through to label __testcase_cxi_dmac_api_00_A */
	}

__testcase_cxi_dmac_api_00_A:
	pr_err("TESTCASE_CXI_DMAC_API_00: FINISH\n");
}

static void testcase_cxi_dmac_api_01(struct cxi_dev *dev)
{
	int retval;

	pr_err("TESTCASE_CXI_DMAC_API_01: START\n");

	retval = cxi_dmac_desc_set_alloc(dev, 0, NULL);
	if (retval != -EINVAL) {
		pr_err("TEST-ERROR: unexpected retval %d for alloc instead of %d\n",
		       retval, -EINVAL);
		goto __testcase_cxi_dmac_api_01_A;
	}

	retval = cxi_dmac_desc_set_alloc(dev, ~0, NULL);
	if (retval != -EINVAL) {
		pr_err("TEST-ERROR: unexpected retval %d for alloc instead of %d\n",
		       retval, -EINVAL);
		goto __testcase_cxi_dmac_api_01_A;
	}

	/*
	 * verify still functioning for trivial case after failure
	 */

	retval = cxi_dmac_desc_set_alloc(dev, 1, NULL);
	if (retval < 0) {
		pr_err("TEST-ERROR: alloc failed with %d\n", retval);
		goto __testcase_cxi_dmac_api_01_A;
	}
	retval = cxi_dmac_desc_set_free(dev, retval);
	if (retval < 0) {
		pr_err("TEST-ERROR: free failed with %d\n", retval);
		/* falling through to label __testcase_cxi_dmac_api_01_A */
	}

__testcase_cxi_dmac_api_01_A:
	pr_err("TESTCASE_CXI_DMAC_API_01: FINISH\n");
}

static void testcase_cxi_dmac_api_02(struct cxi_dev *dev)
{
	int retval;
	int i;
	int set_id[16];
	int num_allocated_set_ids = 0;
	int num_allocated_set_ids2 = 0;
	int num_set_ids_to_cleanup;

	pr_err("TESTCASE_CXI_DMAC_API_02: START\n");

	for (i = 0; i < 16; ++i) {
		retval = cxi_dmac_desc_set_alloc(dev, 1, NULL);
		if (retval < 0) {
			if (retval != -ENOSPC) {

				pr_err("TEST-ERROR: unexpected retval %d for alloc instead of %d\n",
				       retval, -ENOSPC);
				num_set_ids_to_cleanup = num_allocated_set_ids;
				goto __testcase_cxi_dmac_api_02_B;
			}
			break;
		}
		set_id[i] = retval;
		++num_allocated_set_ids;
	}
	for (i = 0; i < num_allocated_set_ids; ++i) {
		retval = cxi_dmac_desc_set_free(dev, set_id[i]);
		if (retval < 0) {
			pr_err("TEST-ERROR: test cleanup free failed with %d for setid[%d]=%d\n",
			       retval, i, set_id[i]);
			goto __testcase_cxi_dmac_api_02_A;
		}

	}
	for (i = 0; i < 16; ++i) {
		retval = cxi_dmac_desc_set_alloc(dev, 1, NULL);
		if (retval < 0) {
			if (retval != -ENOSPC) {
				pr_err("TEST-ERROR: unexpected retval %d for alloc instead of %d\n",
				       retval, -ENOSPC);
				num_set_ids_to_cleanup = num_allocated_set_ids2;
				goto __testcase_cxi_dmac_api_02_B;
			}
			break;
		}
		set_id[i] = retval;
		++num_allocated_set_ids2;
	}
	if (num_allocated_set_ids != num_allocated_set_ids2) {
		pr_err("TEST-ERROR: num_allocated_set_ids(%d) == num_allocated_set_ids2(%d)\n",
		       num_allocated_set_ids,
		       num_allocated_set_ids2);
		/* falling through to cleanup */
	}

	num_set_ids_to_cleanup = num_allocated_set_ids2;

__testcase_cxi_dmac_api_02_B:
	for (i = 0; i < num_set_ids_to_cleanup; ++i) {
		retval = cxi_dmac_desc_set_free(dev, set_id[i]);
		if (retval < 0) {
			pr_err("TEST-ERROR: test cleanup free failed with %d for setid[%d]=%d\n",
			       retval, i, set_id[i]);
			goto __testcase_cxi_dmac_api_02_A;
		}

	}

__testcase_cxi_dmac_api_02_A:
	pr_err("TESTCASE_CXI_DMAC_API_02: FINISH\n");
}

static void testcase_cxi_dmac_api_03(struct cxi_dev *dev)
{
	int retval;

	pr_err("TESTCASE_CXI_DMAC_API_03: START\n");

	retval = cxi_dmac_desc_set_free(dev, ~0);
	if (retval != -EINVAL)
		pr_err("TEST-ERROR: unexpected retval %d for free instead of %d\n",
		       retval, -EINVAL);

	retval = cxi_dmac_desc_set_free(dev, -1);
	if (retval != -EINVAL)
		pr_err("TEST-ERROR: unexpected retval %d for free instead of %d\n",
		       retval, -EINVAL);

	retval = cxi_dmac_desc_set_free(dev, 15);
	if (retval != -EINVAL)
		pr_err("TEST-ERROR: unexpected retval %d for free instead of %d\n",
		       retval, -EINVAL);

	pr_err("TESTCASE_CXI_DMAC_API_03: FINISH\n");
}

static void testcase_cxi_dmac_api_04(struct cxi_dev *dev)
{
	int retval;
	int set_id;

	pr_err("TESTCASE_CXI_DMAC_API_04: START\n");

	retval = cxi_dmac_xfer(dev, ~0);
	if (retval != -EINVAL)
		pr_err("TEST-ERROR: unexpected retval %d for xfer instead of %d\n",
		       retval, -EINVAL);

	retval = cxi_dmac_xfer(dev, -1);
	if (retval != -EINVAL)
		pr_err("TEST-ERROR: unexpected retval %d for xfer instead of %d\n",
		       retval, -EINVAL);

	retval = cxi_dmac_xfer(dev, 15);
	if (retval != -EINVAL)
		pr_err("TEST-ERROR: unexpected retval %d for xfer instead of %d\n",
		       retval, -EINVAL);

	retval = cxi_dmac_desc_set_alloc(dev, 1, NULL);
	if (retval < 0) {
		pr_err("TEST-ERROR: alloc failed with %d\n",
		       retval);
		goto __testcase_cxi_dmac_api_04_A;
	}
	set_id = retval;
	retval = cxi_dmac_xfer(dev, set_id);
	if (retval != -EINVAL)
		pr_err("TEST-ERROR: unexpected retval %d for xfer instead of %d\n",
		       retval, -EINVAL);

	cxi_dmac_desc_set_free(dev, set_id);

__testcase_cxi_dmac_api_04_A:
	pr_err("TESTCASE_CXI_DMAC_API_04: FINISH\n");
}

static void testcase_cxi_dmac_api_05(struct cxi_dev *dev)
{
	struct device *dma_dev = &dev->pdev->dev;
	const size_t dma_size = 7 * sizeof(u64);
	dma_addr_t dma_addr;
	int retval;
	u64 *p;
	int set_id;
	int i;

	pr_err("TESTCASE_CXI_DMAC_API_05: START\n");

	p = dma_alloc_coherent(dma_dev, dma_size, &dma_addr, GFP_KERNEL);
	if (p == NULL) {
		pr_err("TEST-ERROR: memory allocation failed\n");
		goto __testcase_cxi_dmac_api_05_A;
	}
	p[0] = 0x0123456789abcdefUL;
	p[6] = 0x0123456789abcdefUL;
	for (i = 1; i < 6; ++i)
		p[i] = ~0UL;
	retval = cxi_dmac_desc_set_alloc(dev, 1, NULL);
	if (retval < 0) {
		pr_err("TEST-ERROR: alloc failed with %d\n", retval);
		goto __testcase_cxi_dmac_api_05_B;
	}
	set_id = retval;

	retval = cxi_dmac_desc_set_add(dev, set_id, dma_addr + sizeof(u64),
				       C_PI_IPD_STS_EVENT_CNTS(0),
				       5 * sizeof(u64));
	if (retval != 0) {
		pr_err("TEST-ERROR: desc add failed with %d\n", retval);
		goto __testcase_cxi_dmac_api_05_C;
	}

	retval = cxi_dmac_xfer(dev, set_id);
	if (retval != 0) {
		pr_err("TEST-ERROR: xfer failed with %d\n", retval);
		goto __testcase_cxi_dmac_api_05_C;
	}

	if (p[0] != 0x0123456789abcdefUL)
		pr_err("TEST-ERROR: p[0] corrupted\n");
	if (p[6] != 0x0123456789abcdefUL)
		pr_err("TEST-ERROR: p[6] corrupted\n");
	for ((i = 1) ; (i < 6) ; (++i))
		if (p[i] == ~0UL)
			pr_err("TEST-ERROR: p[%d] still ~0UL\n", i);

__testcase_cxi_dmac_api_05_C:
	retval = cxi_dmac_desc_set_free(dev, set_id);
	if (retval != 0)
		pr_err("TEST-ERROR: free failed with %d\n", retval);

__testcase_cxi_dmac_api_05_B:
	dma_free_coherent(dma_dev, dma_size, p, dma_addr);

__testcase_cxi_dmac_api_05_A:
	pr_err("TESTCASE_CXI_DMAC_API_05: FINISH\n");
}

static void testcase_cxi_dmac_api_06(struct cxi_dev *dev)
{
	struct device *dma_dev = &dev->pdev->dev;
	const size_t dma_size = (6 + 15) * sizeof(u64);
	dma_addr_t dma_addr;
	int retval;
	u64 *p;
	int set_id;
	int i;

	pr_err("TESTCASE_CXI_DMAC_API_06: START\n");

	p = dma_alloc_coherent(dma_dev, dma_size, &dma_addr, GFP_KERNEL);
	if (p == NULL) {
		pr_err("TEST-ERROR: memory allocation failed\n");
		goto __testcase_cxi_dmac_api_06_A;
	}
	memset(p, 0, dma_size);

	for (i = 0 ; i < 6 + 15; ++i)
		if (i == 0 || i == 2 || i == 5 || i == 9 || i == 14 || i == 20)
			p[i] = 0x0123456789abcdefUL;
		else
			p[i] = ~0UL;

	retval = cxi_dmac_desc_set_alloc(dev, 5, NULL);
	if (retval < 0) {
		pr_err("TEST-ERROR: alloc failed with %d\n", retval);
		goto __testcase_cxi_dmac_api_06_B;
	}
	set_id = retval;

	retval = cxi_dmac_desc_set_add(dev, set_id, dma_addr + sizeof(u64),
				       C_PI_IPD_STS_EVENT_CNTS(0),
				       1 * sizeof(u64));
	if (retval != 0) {
		pr_err("TEST-ERROR: desc add failed with %d\n", retval);
		goto __testcase_cxi_dmac_api_06_C;
	}

	retval = cxi_dmac_desc_set_add(dev, set_id, dma_addr + 3 * sizeof(u64),
				       C_MB_STS_EVENT_CNTS(0),
				       2 * sizeof(u64));
	if (retval != 0) {
		pr_err("TEST-ERROR: desc add failed with %d\n", retval);
		goto __testcase_cxi_dmac_api_06_C;
	}

	retval = cxi_dmac_desc_set_add(dev, set_id, dma_addr + 6 * sizeof(u64),
				       C_CQ_STS_EVENT_CNTS(0),
				       3 * sizeof(u64));
	if (retval != 0) {
		pr_err("TEST-ERROR: desc add failed with %d\n", retval);
		goto __testcase_cxi_dmac_api_06_C;
	}

	retval = cxi_dmac_desc_set_add(dev, set_id, dma_addr + 10 * sizeof(u64),
				       C_LPE_STS_EVENT_CNTS(0),
				       4 * sizeof(u64));
	if (retval != 0) {
		pr_err("TEST-ERROR: desc add failed with %d\n", retval);
		goto __testcase_cxi_dmac_api_06_C;
	}

	retval = cxi_dmac_desc_set_add(dev, set_id, dma_addr + 15 * sizeof(u64),
				       C_HNI_STS_EVENT_CNTS(0),
				       5 * sizeof(u64));
	if (retval != 0) {
		pr_err("TEST-ERROR: desc add failed with %d\n", retval);
		goto __testcase_cxi_dmac_api_06_C;
	}

	retval = cxi_dmac_xfer(dev, set_id);
	if (retval != 0) {
		pr_err("TEST-ERROR: xfer failed with %d\n", retval);
		goto __testcase_cxi_dmac_api_06_C;
	}

	for (i = 0; i < 6 + 15; ++i)
		if (i == 0 || i == 2 || i == 5 || i == 9 || i == 14 ||
		    i == 20) {
			if (p[i] != 0x0123456789abcdefUL)
				pr_err("TEST-ERROR: p[%d] corrupted\n", i);
		} else {
			if (p[i] == ~0UL)
				pr_err("TEST-ERROR: p[%d] still ~0UL\n", i);
		}

__testcase_cxi_dmac_api_06_C:
	retval = cxi_dmac_desc_set_free(dev, set_id);
	if (retval != 0)
		pr_err("TEST-ERROR: free failed with %d\n", retval);

__testcase_cxi_dmac_api_06_B:
	dma_free_coherent(dma_dev, dma_size, p, dma_addr);

__testcase_cxi_dmac_api_06_A:
	pr_err("TESTCASE_CXI_DMAC_API_06: FINISH\n");
}

static void testcase_cxi_dmac_api_07(struct cxi_dev *dev)
{
	struct device *dma_dev = &dev->pdev->dev;
	unsigned int numu64s = (4 + 3 + C_IXE_CFG_DSCP_WRQ_MAP_ENTRIES
				+ C_IXE_CFG_DSCP_DSCP_TC_MAP_ENTRIES);
	const size_t dma_size = numu64s * sizeof(u64);
	dma_addr_t dma_addr;
	int retval;
	u64 *p;
	int set_id;
	int i;
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);
	u64 t;
	int pidx;

	pr_err("TESTCASE_CXI_DMAC_API_07: START\n");

	p = dma_alloc_coherent(dma_dev, dma_size, &dma_addr, GFP_KERNEL);
	if (p == NULL) {
		pr_err("TEST-ERROR: memory allocation failed\n");
		goto __testcase_cxi_dmac_api_07_A;
	}
	memset(p, 0, dma_size);
	for (i = 0; i < numu64s; ++i)
		if (i == 0 || i == 2 || i == 5 || i == numu64s - 1)
			p[i] = 0x0123456789abcdefUL;
		else
			p[i] = ~0UL;

	retval = cxi_dmac_desc_set_alloc(dev, 3, NULL);
	if (retval < 0) {
		pr_err("TEST-ERROR: alloc failed with %d\n", retval);
		goto __testcase_cxi_dmac_api_07_B;
	}
	set_id = retval;

	retval = cxi_dmac_desc_set_add(dev, set_id, dma_addr + sizeof(u64),
				       C_MB_STS_REV, 1 * sizeof(u64));
	if (retval != 0) {
		pr_err("TEST-ERROR: desc add failed with %d\n", retval);
		goto __testcase_cxi_dmac_api_07_C;
	}

	retval = cxi_dmac_desc_set_add(dev, set_id, dma_addr + 3 * sizeof(u64),
				       C_OXE_CFG_ARB_CONFIG, 2 * sizeof(u64));
	if (retval != 0) {
		pr_err("TEST-ERROR: desc add failed with %d\n", retval);
		goto __testcase_cxi_dmac_api_07_C;
	}

	retval = cxi_dmac_desc_set_add(dev, set_id, dma_addr + 6 * sizeof(u64),
				       C_IXE_CFG_DSCP_WRQ_MAP(0),
				       (C_IXE_CFG_DSCP_WRQ_MAP_ENTRIES +
					C_IXE_CFG_DSCP_DSCP_TC_MAP_ENTRIES) *
				       sizeof(u64));
	if (retval != 0) {
		pr_err("TEST-ERROR: desc add failed with %d\n", retval);
		goto __testcase_cxi_dmac_api_07_C;
	}

	retval = cxi_dmac_xfer(dev, set_id);
	if (retval != 0) {
		pr_err("TEST-ERROR: xfer failed with %d\n", retval);
		goto __testcase_cxi_dmac_api_07_C;
	}

	for (i = 0; i < numu64s; ++i) {
		if (i == 0 || i == 2 || i == 5 || i == numu64s - 1) {
			if (p[i] != 0x0123456789abcdefUL)
				pr_err("TEST-ERROR: p[%d] corrupted\n", i);
		} else {
			if (p[i] == ~0UL)
				pr_err("TEST-ERROR: p[%d] still ~0UL\n", i);
		}
	}

	cass_read(hw, C_MB_STS_REV, &t, sizeof(t));
	if (t != p[1])
		pr_err("TEST-ERROR: p[1]=0x%llx != t=0x%llx\n", p[1], t);

	cass_read(hw, C_OXE_CFG_ARB_CONFIG, &t, sizeof(t));
	if (t != p[3])
		pr_err("TEST-ERROR: p[3]=0x%llx != t=0x%llx\n", p[3], t);

	cass_read(hw, C_OXE_CFG_ARB_MFS_OUT, &t, sizeof(t));
	if (t != p[4])
		pr_err("TEST-ERROR: p[4]=0x%llx != t=0x%llx\n", p[4], t);

	for (i = 0; i < C_IXE_CFG_DSCP_WRQ_MAP_ENTRIES; ++i) {
		cass_read(hw, C_IXE_CFG_DSCP_WRQ_MAP(i), &t, sizeof(t));
		pidx = 6 + i;
		if (t != p[pidx])
			pr_err("TEST-ERROR: p[%d]=0x%llx != t=0x%llx\n",
			       pidx, p[pidx], t);
	}

	for (i = 0; i < C_IXE_CFG_DSCP_DSCP_TC_MAP_ENTRIES; ++i) {
		cass_read(hw, C_IXE_CFG_DSCP_DSCP_TC_MAP(i), &t, sizeof(t));
		pidx = 6 + C_IXE_CFG_DSCP_WRQ_MAP_ENTRIES + i;
		if (t != p[pidx])
			pr_err("TEST-ERROR: p[%d]=0x%llx != t=0x%llx\n",
			       pidx, p[pidx], t);
	}

__testcase_cxi_dmac_api_07_C:
	retval = cxi_dmac_desc_set_free(dev, set_id);
	if (retval != 0)
		pr_err("TEST-ERROR: free failed with %d\n", retval);

__testcase_cxi_dmac_api_07_B:
	dma_free_coherent(dma_dev, dma_size, p, dma_addr);

__testcase_cxi_dmac_api_07_A:
	pr_err("TESTCASE_CXI_DMAC_API_07: FINISH\n");
}

static void testcase_cxi_dmac_api_08(struct cxi_dev *dev)
{
	int retval;

	pr_err("TESTCASE_CXI_DMAC_API_08: START\n");

	retval = cxi_dmac_desc_set_reserve(dev, 1, 1111, NULL);
	if (retval != -EINVAL)
		pr_err("TEST-ERROR: unexpected retval %d for reserve instead of %d\n",
		       retval, -EINVAL);

	retval = cxi_dmac_desc_set_reserve(dev, 0, 0, NULL);
	if (retval != -EINVAL)
		pr_err("TEST-ERROR: unexpected retval %d for reserve instead of %d\n",
		       retval, -EINVAL);

	retval = cxi_dmac_desc_set_reserve(dev, 10000, 0, NULL);
	if (retval != -EINVAL)
		pr_err("TEST-ERROR: unexpected retval %d for reserve instead of %d\n",
		       retval, -EINVAL);

	pr_err("TESTCASE_CXI_DMAC_API_08: FINISH\n");
}

static void testcase_cxi_dmac_api_09(struct cxi_dev *dev)
{
	int retval;
	int set_id;

	pr_err("TESTCASE_CXI_DMAC_API_09: START\n");

	retval = cxi_dmac_desc_set_reserve(dev, 1, 500, NULL);
	if (retval < 0) {
		pr_err("TEST-ERROR: reserve failed with %d\n", retval);
		goto __testcase_cxi_dmac_api_09_A;
	}

	set_id = retval;

	/*
	 * reserving the same range should fail.
	 */
	retval = cxi_dmac_desc_set_reserve(dev, 1, 500, NULL);
	if (retval != -ENOSPC)
		pr_err("TEST-ERROR: unexpected retval %d for reserve instead of %d\n",
		       retval, -ENOSPC);

	retval = cxi_dmac_desc_set_free(dev, set_id);
	if (retval != 0) {
		pr_err("TEST-ERROR: free failed with %d\n", retval);
		goto __testcase_cxi_dmac_api_09_A;
	}

	/*
	 * reserving the same range again after freeing it should succeed
	 */
	retval = cxi_dmac_desc_set_reserve(dev, 1, 500, NULL);
	if (retval < 0) {
		pr_err("TEST-ERROR: reserve failed with %d\n", retval);
		goto __testcase_cxi_dmac_api_09_A;
	}

	set_id = retval;

	retval = cxi_dmac_desc_set_free(dev, set_id);
	if (retval != 0) {
		pr_err("TEST-ERROR: free failed with %d\n", retval);
		/* falling through to __testcase_cxi_dmac_api_09_A; */
	}

__testcase_cxi_dmac_api_09_A:
	pr_err("TESTCASE_CXI_DMAC_API_09: FINISH\n");
}

static void testcase_cxi_dmac_api_10(struct cxi_dev *dev)
{
	int retval;
	int set_id;
	int i;

	pr_err("TESTCASE_CXI_DMAC_API_10: START\n");

	retval = cxi_dmac_desc_set_reserve(dev, 100, 500, NULL);
	if (retval < 0) {
		pr_err("TEST-ERROR: reserve failed with %d\n", retval);
		goto __testcase_cxi_dmac_api_10_A;
	}

	set_id = retval;

	for (i = 0; i < 100; ++i) {
		retval = cxi_dmac_desc_set_reserve(dev, 1, 500 + i, NULL);
		if (retval != -ENOSPC)
			pr_err("TEST-ERROR: unexpected retval %d for reserve instead of %d\n",
			       retval, -ENOSPC);
	}

	retval = cxi_dmac_desc_set_free(dev, set_id);
	if (retval != 0) {
		pr_err("TEST-ERROR: free failed with %d\n", retval);
		/* falling through to __testcase_cxi_dmac_api_10_A; */
	}

__testcase_cxi_dmac_api_10_A:
	pr_err("TESTCASE_CXI_DMAC_API_10: FINISH\n");
}

static void testcase_cxi_dmac_api_11(struct cxi_dev *dev)
{
	int retval;

	pr_err("TESTCASE_CXI_DMAC_API_11: START\n");

	retval = cxi_dmac_desc_set_reset(dev, ~0);
	if (retval != -EINVAL)
		pr_err("TEST-ERROR: unexpected retval %d for reset instead of %d\n",
		       retval, -EINVAL);

	retval = cxi_dmac_desc_set_reset(dev, -1);
	if (retval != -EINVAL)
		pr_err("TEST-ERROR: unexpected retval %d for reset instead of %d\n",
		       retval, -EINVAL);

	retval = cxi_dmac_desc_set_reset(dev, 15);
	if (retval != -EINVAL)
		pr_err("TEST-ERROR: unexpected retval %d for reset instead of %d\n",
		       retval, -EINVAL);

	pr_err("TESTCASE_CXI_DMAC_API_11: FINISH\n");
}

static void testcase_cxi_dmac_api_12(struct cxi_dev *dev)
{
	dma_addr_t dma_addr = 0;
	int retval;
	int set_id;

	pr_err("TESTCASE_CXI_DMAC_API_12: START\n");

	retval = cxi_dmac_desc_set_alloc(dev, 1, NULL);
	if (retval < 0) {
		pr_err("TEST-ERROR: alloc failed with %d\n", retval);
		goto __testcase_cxi_dmac_api_12_A;
	}
	set_id = retval;

	retval = cxi_dmac_desc_set_add(dev, set_id, dma_addr + 1,
				       C_MB_STS_REV, 1 * sizeof(u64));
	if (retval != -EINVAL)
		pr_err("TEST-ERROR: unexpected retval %d for add instead of %d\n",
		       retval, -EINVAL);

	retval = cxi_dmac_desc_set_add(dev, set_id, dma_addr, C_MB_STS_REV, 2);
	if (retval != -EINVAL)
		pr_err("TEST-ERROR: unexpected retval %d for add instead of %d\n",
		       retval, -EINVAL);

	retval = cxi_dmac_desc_set_add(dev, set_id, dma_addr, C_MB_STS_REV, 0);
	if (retval != -EINVAL)
		pr_err("TEST-ERROR: unexpected retval %d for add instead of %d\n",
		       retval, -EINVAL);

	retval = cxi_dmac_desc_set_add(dev, set_id, dma_addr, C_MB_STS_REV,
				       0x00100001UL * sizeof(u64));
	if (retval != -EINVAL)
		pr_err("TEST-ERROR: unexpected retval %d for add instead of %d\n",
		       retval, -EINVAL);

	retval = cxi_dmac_desc_set_add(dev, set_id, dma_addr, 1, 1);
	if (retval != -EINVAL)
		pr_err("TEST-ERROR: unexpected retval %d for add instead of %d\n",
		       retval, -EINVAL);

	retval = cxi_dmac_desc_set_add(dev, set_id, dma_addr, C_MEMORG_CSR_SIZE,
				       1 * sizeof(u64));
	if (retval != -EINVAL)
		pr_err("TEST-ERROR: unexpected retval %d for add instead of %d\n",
		       retval, -EINVAL);

	retval = cxi_dmac_desc_set_add(dev, set_id, dma_addr, C_MEMORG_CSR_SIZE -
				       sizeof(u64), 2 * sizeof(u64));
	if (retval != -EINVAL)
		pr_err("TEST-ERROR: unexpected retval %d for add instead of %d\n",
		       retval, -EINVAL);

	retval = cxi_dmac_desc_set_add(dev, set_id, dma_addr, 0x00fffff8UL,
				       2 * sizeof(u64));
	if (retval != -EINVAL)
		pr_err("TEST-ERROR: unexpected retval %d for add instead of %d\n",
		       retval, -EINVAL);

	retval = cxi_dmac_desc_set_add(dev, ~0, dma_addr, C_MB_STS_REV,
				       1 * sizeof(u64));
	if (retval != -EINVAL)
		pr_err("TEST-ERROR: unexpected retval %d for add instead of %d\n",
		       retval, -EINVAL);

	retval = cxi_dmac_desc_set_add(dev, -1, dma_addr, C_MB_STS_REV,
				       1 * sizeof(u64));
	if (retval != -EINVAL)
		pr_err("TEST-ERROR: unexpected retval %d for add instead of %d\n",
		       retval, -EINVAL);

	retval = cxi_dmac_desc_set_add(dev, 15, dma_addr, C_MB_STS_REV,
				       1 * sizeof(u64));
	if (retval != -EINVAL)
		pr_err("TEST-ERROR: unexpected retval %d for add instead of %d\n",
		       retval, -EINVAL);

	retval = cxi_dmac_desc_set_add(dev, set_id, dma_addr, C_MB_STS_REV,
				       1 * sizeof(u64));
	if (retval != 0)
		pr_err("TEST-ERROR: unexpected retval %d for add instead of %d\n",
		       retval, 0);

	retval = cxi_dmac_desc_set_add(dev, set_id, dma_addr, C_OXE_CFG_ARB_CONFIG,
				       2 * sizeof(u64));
	if (retval != -ENOSPC)
		pr_err("TEST-ERROR: unexpected retval %d for add instead of %d\n",
		       retval, -ENOSPC);

	retval = cxi_dmac_desc_set_free(dev, set_id);
	if (retval != 0) {
		pr_err("TEST-ERROR: free failed with %d\n", retval);
		/* falling through to __testcase_cxi_dmac_api_12_C; */
	}

__testcase_cxi_dmac_api_12_A:
	pr_err("TESTCASE_CXI_DMAC_API_12: FINISH\n");
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/*
 * Core is adding a new device.
 */
static int add_device(struct cxi_dev *dev)
{
	pr_err("TESTSUITE_CXI_DMAC_API: START\n");
	testcase_cxi_dmac_api_00(dev);
	testcase_cxi_dmac_api_01(dev);
	testcase_cxi_dmac_api_02(dev);
	testcase_cxi_dmac_api_03(dev);
	testcase_cxi_dmac_api_04(dev);
	testcase_cxi_dmac_api_05(dev);
	testcase_cxi_dmac_api_06(dev);
	testcase_cxi_dmac_api_07(dev);
	testcase_cxi_dmac_api_08(dev);
	testcase_cxi_dmac_api_09(dev);
	testcase_cxi_dmac_api_10(dev);
	testcase_cxi_dmac_api_11(dev);
	testcase_cxi_dmac_api_12(dev);
	pr_err("TESTSUITE_CXI_DMAC_API: FINISH\n");
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
MODULE_DESCRIPTION("CXI DMAC API test driver");
MODULE_AUTHOR("Cray Inc.");
