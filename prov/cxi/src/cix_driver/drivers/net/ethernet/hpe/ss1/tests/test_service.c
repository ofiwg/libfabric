// SPDX-License-Identifier: GPL-2.0
/* Copyright 2022 Hewlett Packard Enterprise Development LP */
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/hpe/cxi/cxi.h>
#include "cass_core.h"

#define VNI 8U
#define TLE_COUNT 10U
#define TIMEOUT 2U

/* List of devices registered with this client */
static LIST_HEAD(dev_list);
static DEFINE_MUTEX(device_list_mutex);

/* Keep track of known devices. Protected by device_list_mutex. */
struct tdev {
	struct list_head dev_list;
	struct cxi_dev *dev;
};

#define test_err(fmt, ...) pr_err("%s:%d " fmt, __func__, __LINE__, \
	##__VA_ARGS__)

static int test_service_tle_in_use(struct cxi_dev *dev)
{
	int rc;
	struct cxi_svc_desc desc = {
		.enable = 1,
		.is_system_svc = 1,
		.num_vld_vnis = 1,
		.resource_limits = 1,
		.restricted_vnis = 1,
		.vnis[0] = VNI,
		.limits.type[CXI_RSRC_TYPE_PTE].max = 100,
		.limits.type[CXI_RSRC_TYPE_PTE].res = 100,
		.limits.type[CXI_RSRC_TYPE_TXQ].max = 100,
		.limits.type[CXI_RSRC_TYPE_TXQ].res = 100,
		.limits.type[CXI_RSRC_TYPE_TGQ].max = 100,
		.limits.type[CXI_RSRC_TYPE_TGQ].res = 100,
		.limits.type[CXI_RSRC_TYPE_EQ].max = 100,
		.limits.type[CXI_RSRC_TYPE_EQ].res = 100,
		.limits.type[CXI_RSRC_TYPE_CT].max = 100,
		.limits.type[CXI_RSRC_TYPE_CT].res = 100,
		.limits.type[CXI_RSRC_TYPE_LE].max = 100,
		.limits.type[CXI_RSRC_TYPE_LE].res = 100,
		.limits.type[CXI_RSRC_TYPE_TLE].max = 100,
		.limits.type[CXI_RSRC_TYPE_TLE].res = 100,
		.limits.type[CXI_RSRC_TYPE_AC].max = 4,
		.limits.type[CXI_RSRC_TYPE_AC].res = 4,
		.tcs[CXI_TC_BEST_EFFORT] = true,
	};
	struct cxi_svc_fail_info info;
	struct cxi_lni *lni;
	struct cxi_ct *ct;
	struct c_ct_writeback *wb;
	struct cxi_cp *cp;
	struct cxi_cq *trig_cq;
	struct cxi_cq_alloc_opts cq_opts = {};
	struct cxi_rsrc_use in_use;
	struct c_ct_cmd ct_cmd = {};
	int i;
	ktime_t timeout;

	rc = cxi_svc_alloc(dev, &desc, &info, "tle-in-use");
	if (rc < 0) {
		test_err("cxi_svc_alloc failed: %d\n", rc);
		goto err;
	}

	desc.svc_id = rc;

	lni = cxi_lni_alloc(dev, desc.svc_id);
	if (IS_ERR(lni)) {
		rc = PTR_ERR(lni);
		test_err("cxi_lni_alloc failed: %d\n", rc);
		goto err_free_svc;
	}

	wb = kzalloc(sizeof(*wb), GFP_KERNEL);
	if (!wb) {
		pr_err("TEST-ERROR: Failed to allocate writeback buffer\n");
		goto err_free_lni;
	}

	ct = cxi_ct_alloc(lni, wb, false);
	if (IS_ERR(ct)) {
		rc = PTR_ERR(ct);
		test_err("cxi_ct_alloc failed: %d\n", rc);
		goto err_free_wb;
	}

	cp = cxi_cp_alloc(lni, VNI, CXI_TC_BEST_EFFORT, CXI_TC_TYPE_DEFAULT);
	if (IS_ERR(cp)) {
		rc = PTR_ERR(cp);
		test_err("cxi_cp_alloc failed: %d\n", rc);
		goto err_free_ct;
	}

	cq_opts.count = 4096;
	cq_opts.lcid = cp->lcid;
	cq_opts.flags = CXI_CQ_IS_TX | CXI_CQ_TX_WITH_TRIG_CMDS;

	trig_cq = cxi_cq_alloc(lni, NULL, &cq_opts, 0);
	if (IS_ERR(trig_cq)) {
		rc = PTR_ERR(trig_cq);
		test_err("cxi_cq_alloc failed: %d\n", rc);
		goto err_free_cp;
	}

	rc = cxi_svc_rsrc_get(dev, desc.svc_id, &in_use);
	if (rc) {
		test_err("cxi_svc_rsrc_get failed: %d\n", rc);
		goto err_free_cq;
	}

	if (in_use.in_use[CXI_RSRC_TYPE_TLE] != 0) {
		rc = -EINVAL;
		test_err("Incorrect TLE in use: expected=0 got=%d\n",
			 in_use.in_use[CXI_RSRC_TYPE_TLE]);
		goto err_free_cq;
	}

	ct_cmd.trig_ct = ct->ctn;
	ct_cmd.ct = ct->ctn;
	ct_cmd.ct_success = 1;
	ct_cmd.set_ct_success = 1;

	for (i = 0; i < TLE_COUNT; i++) {
		ct_cmd.threshold = i + 1;

		rc = cxi_cq_emit_ct(trig_cq, C_CMD_CT_TRIG_INC, &ct_cmd);
		if (rc) {
			test_err("cxi_cq_emit_ct failed: %d\n", rc);
			goto err_free_cq;
		}

		cxi_cq_ring(trig_cq);
	}

	rc = cxi_svc_rsrc_get(dev, desc.svc_id, &in_use);
	if (rc) {
		test_err("cxi_svc_rsrc_get failed: %d\n", rc);
		goto err_free_cq;
	}

	if (in_use.in_use[CXI_RSRC_TYPE_TLE] != TLE_COUNT) {
		rc = -EINVAL;
		test_err("Incorrect TLE in use: expected=%u got=%d\n",
			 TLE_COUNT, in_use.in_use[CXI_RSRC_TYPE_TLE]);
		goto err_free_cq;
	}

	rc = cxi_cq_emit_ct(trig_cq, C_CMD_CT_INC, &ct_cmd);
	if (rc) {
		test_err("cxi_cq_emit_ct failed: %d\n", rc);
		goto err_free_cq;
	}

	cxi_cq_ring(trig_cq);

	timeout = ktime_add(ktime_get(), ktime_set(TIMEOUT, 0));

	do {
		rc = cxi_svc_rsrc_get(dev, desc.svc_id, &in_use);
		if (rc) {
			test_err("cxi_svc_rsrc_get failed: %d\n", rc);
			goto err_free_cq;
		}

		if (ktime_compare(ktime_get(), timeout) > 1) {
			rc = -ETIMEDOUT;
			test_err("timeout reaching TLE count\n");
			goto err_free_cq;
		}
	} while (in_use.in_use[CXI_RSRC_TYPE_TLE] != 0);

	rc = 0;

err_free_cq:
	cxi_cq_free(trig_cq);
err_free_cp:
	cxi_cp_free(cp);
err_free_ct:
	cxi_ct_free(ct);
err_free_wb:
	kfree(wb);
err_free_lni:
	cxi_lni_free(lni);
err_free_svc:
	cxi_svc_destroy(dev, desc.svc_id);
err:
	return rc;
}

static int test_service_modify(struct cxi_dev *dev)
{
	int i, rc;
	struct cxi_svc_desc desc = {
		.enable = 1,
		.is_system_svc = 1,
		.num_vld_vnis = 1,
		.restricted_vnis = 1,
		.vnis[0] = VNI,
		.limits.type[CXI_RSRC_TYPE_PTE].max = 100,
		.limits.type[CXI_RSRC_TYPE_PTE].res = 100,
		.limits.type[CXI_RSRC_TYPE_TXQ].max = 100,
		.limits.type[CXI_RSRC_TYPE_TXQ].res = 100,
		.limits.type[CXI_RSRC_TYPE_TGQ].max = 100,
		.limits.type[CXI_RSRC_TYPE_TGQ].res = 100,
		.limits.type[CXI_RSRC_TYPE_EQ].max = 100,
		.limits.type[CXI_RSRC_TYPE_EQ].res = 100,
		.limits.type[CXI_RSRC_TYPE_CT].max = 100,
		.limits.type[CXI_RSRC_TYPE_CT].res = 100,
		.limits.type[CXI_RSRC_TYPE_LE].max = 100,
		.limits.type[CXI_RSRC_TYPE_LE].res = 100,
		.limits.type[CXI_RSRC_TYPE_TLE].max = 100,
		.limits.type[CXI_RSRC_TYPE_TLE].res = 100,
		.limits.type[CXI_RSRC_TYPE_AC].max = 4,
		.limits.type[CXI_RSRC_TYPE_AC].res = 4,
	};
	struct cxi_svc_desc comp;
	struct cxi_svc_fail_info info;
	struct cxi_lni *lni;

	rc = cxi_svc_alloc(dev, &desc, &info, "svc-modify");
	if (rc < 0) {
		test_err("cxi_svc_alloc failed: %d\n", rc);
		goto err;
	}

	desc.svc_id = rc;

	/* Modify portions of descriptor */
	desc.restricted_tcs = 1;
	desc.tcs[0] = true;
	desc.tcs[1] = true;

	lni = cxi_lni_alloc(dev, desc.svc_id);
	if (IS_ERR(lni)) {
		rc = PTR_ERR(lni);
		test_err("cxi_lni_alloc failed: %d\n", rc);
		goto err_free_svc;
	}

	/* Should not be able to modify service while it is being used. */
	rc = cxi_svc_update(dev, &desc);
	if (rc != -EBUSY) {
		rc = -EINVAL;
		test_err("cxi_svc_update did not returned -EBUSY\n");
		goto err_free_lni;
	}

	cxi_lni_free(lni);

	/* Should be able to modify service since it is not being used. */
	rc = cxi_svc_update(dev, &desc);
	if (rc) {
		test_err("cxi_svc_update failed: %d\n", rc);
		goto err_free_svc;
	}

	/* Get the updated descriptor back from the kernel */
	rc = cxi_svc_get(dev, desc.svc_id, &comp);
	if (rc) {
		test_err("cxi_svc_get failed: %d\n", rc);
		goto err_free_svc;
	}

	if (comp.restricted_members != desc.restricted_members) {
		test_err("Inconsistent result. desc.restricted_members:%d comp.restricted_members:%d",
			 desc.restricted_members, comp.restricted_members);
		rc = -EINVAL;
		goto err_free_svc;
	}

	if (comp.restricted_tcs != desc.restricted_tcs) {
		test_err("Inconsistent result. desc.restricted_tcs:%d comp.restricted_tcs:%d",
			 desc.restricted_tcs, comp.restricted_tcs);
		rc = -EINVAL;
		goto err_free_svc;
	}

	/* Compare TCs */
	for (i = 0; i < CXI_TC_MAX - 1; i++) {
		if (comp.tcs[i] != desc.tcs[i]) {
			test_err("Inconsistent result. desc.tcs[%d]:%d comp.tcs[%d]:%d",
				 i, desc.tcs[i], i, comp.tcs[i]);
			rc = -EINVAL;
			goto err_free_svc;
		}
	}

	cxi_svc_destroy(dev, desc.svc_id);

	return 0;

err_free_lni:
	cxi_lni_free(lni);
err_free_svc:
	cxi_svc_destroy(dev, desc.svc_id);
err:
	return rc;
}

static int test_disabled_service(struct cxi_dev *dev)
{
	int rc;
	struct cxi_svc_desc desc = {
		.enable = 0,
		.is_system_svc = 1,
		.num_vld_vnis = 1,
		.restricted_vnis = 1,
		.vnis[0] = VNI,
		.limits.type[CXI_RSRC_TYPE_PTE].max = 100,
		.limits.type[CXI_RSRC_TYPE_PTE].res = 100,
		.limits.type[CXI_RSRC_TYPE_TXQ].max = 100,
		.limits.type[CXI_RSRC_TYPE_TXQ].res = 100,
		.limits.type[CXI_RSRC_TYPE_TGQ].max = 100,
		.limits.type[CXI_RSRC_TYPE_TGQ].res = 100,
		.limits.type[CXI_RSRC_TYPE_EQ].max = 100,
		.limits.type[CXI_RSRC_TYPE_EQ].res = 100,
		.limits.type[CXI_RSRC_TYPE_CT].max = 100,
		.limits.type[CXI_RSRC_TYPE_CT].res = 100,
		.limits.type[CXI_RSRC_TYPE_LE].max = 100,
		.limits.type[CXI_RSRC_TYPE_LE].res = 100,
		.limits.type[CXI_RSRC_TYPE_TLE].max = 100,
		.limits.type[CXI_RSRC_TYPE_TLE].res = 100,
		.limits.type[CXI_RSRC_TYPE_AC].max = 4,
		.limits.type[CXI_RSRC_TYPE_AC].res = 4,
	};
	struct cxi_svc_fail_info info;
	struct cxi_lni *lni;

	rc = cxi_svc_alloc(dev, &desc, &info, "disabled-svc");
	if (rc < 0) {
		test_err("cxi_svc_alloc failed: %d\n", rc);
		goto err;
	}

	desc.svc_id = rc;

	rc = cxi_svc_update(dev, &desc);
	if (rc) {
		test_err("cxi_svc_update failed: %d\n", rc);
		goto err_free_svc;
	}

	lni = cxi_lni_alloc(dev, desc.svc_id);
	if (PTR_ERR(lni) != -EKEYREVOKED) {
		rc = -EINVAL;
		test_err("cxi_lni_alloc did not return -EKEYREVOKED: %ld\n",
			 PTR_ERR(lni));
		cxi_lni_free(lni);
		goto err_free_svc;
	}

	rc = 0;

err_free_svc:
	cxi_svc_destroy(dev, desc.svc_id);
err:
	return rc;
}

static int test_default_service(struct cxi_dev *dev)
{
	int rc = 0;
	struct cxi_cp *cp;
	struct cxi_lni *lni;
	struct cxi_svc_desc desc;

	/* Get the updated descriptor back from the kernel */
	rc = cxi_svc_get(dev, CXI_DEFAULT_SVC_ID, &desc);
	if (rc) {
		test_err("cxi_svc_get failed: %d\n", rc);
		goto err;
	}

	/* Enable Default Service */
	desc.enable = true;
	rc = cxi_svc_update(dev, &desc);
	if (rc) {
		test_err("cxi_svc_update failed: %d\n", rc);
		goto err;
	}

	/* Allocate LNI against default service */
	lni = cxi_lni_alloc(dev, CXI_DEFAULT_SVC_ID);
	if (IS_ERR(lni)) {
		rc = PTR_ERR(lni);
		test_err("Couldn't allocate LNI with default service: %d\n",
			 rc);
		goto err;
	}

	/* Try to use non-default VNI. Should fail */
	cp = cxi_cp_alloc(lni, 500, CXI_TC_BEST_EFFORT,
			  CXI_TC_TYPE_DEFAULT);

	if (!IS_ERR(cp)) {
		test_err("Allocated CP with non-default VNI using default service\n");
		cxi_cp_free(cp);
		/* TODO cause failure when tx_profile is not allocated
		 * when no profile with vni is found.
		 */
		// rc = -EINVAL;
	}

	cxi_lni_free(lni);
err:
	return rc;
}

static int test_service_busy(struct cxi_dev *dev)
{
	int rc;
	struct cxi_lni *lni;
	struct cxi_svc_fail_info info;
	struct cxi_svc_desc desc = {
		.enable = 1,
		.is_system_svc = 1,
		.restricted_vnis = 1,
		.num_vld_vnis = 1,
		.vnis[0] = VNI,
		.limits.type[CXI_RSRC_TYPE_AC].max = 4,
		.limits.type[CXI_RSRC_TYPE_AC].res = 4,
	};

	rc = cxi_svc_alloc(dev, &desc, &info, "svc-busy");
	if (rc < 0) {
		test_err("cxi_svc_alloc failed: %d\n", rc);
		goto err;
	}

	desc.svc_id = rc;

	lni = cxi_lni_alloc(dev, desc.svc_id);
	if (IS_ERR(lni)) {
		rc = PTR_ERR(lni);
		test_err("cxi_lni_alloc failed: %d\n", rc);
		goto err_free_svc;
	}

	/* should be busy */
	rc = cxi_svc_destroy(dev, desc.svc_id);
	if (rc != -EBUSY) {
		test_err("cxi_svc_destroy should fail\n");
		rc = -1;
		goto err_free_lni;
	}

	/* should still be busy */
	rc = cxi_svc_destroy(dev, desc.svc_id);
	if (rc != -EBUSY) {
		test_err("cxi_svc_destroy should be busy rc:%d\n", rc);
		rc = -1;
		goto err_free_lni;
	}

	cxi_lni_free(lni);

	/* should not be busy any longer */
	rc = cxi_svc_destroy(dev, desc.svc_id);
	if (rc) {
		test_err("cxi_svc_destroy failed: %d\n", rc);
		goto err;
	}

	return 0;

err_free_lni:
	cxi_lni_free(lni);
err_free_svc:
	cxi_svc_destroy(dev, desc.svc_id);
err:
	return rc;
}

static int test_restricted_members_service(struct cxi_dev *dev)
{
	int rc;
	struct cxi_lni *lni;
	struct cxi_svc_fail_info info;
	struct cxi_svc_desc desc = {
		.enable = 0,
		.is_system_svc = 1,
		.restricted_members = 0,
		.num_vld_vnis = 1,
		.restricted_vnis = 1,
		.vnis[0] = VNI,
		.members[0].svc_member.uid = 1,
		.members[0].type = CXI_SVC_MEMBER_UID,
		.members[1].svc_member.gid = 2,
		.members[1].type = CXI_SVC_MEMBER_GID,
	};

	rc = cxi_svc_alloc(dev, &desc, &info, "restricted-members");
	if (rc < 0) {
		test_err("cxi_svc_alloc failed: %d\n", rc);
		goto err;
	}

	desc.svc_id = rc;

	lni = cxi_lni_alloc(dev, desc.svc_id);
	if (IS_ERR(lni)) {
		rc = PTR_ERR(lni);
		test_err("cxi_lni_alloc failed:%d\n", rc);
		goto err_free_svc;
	}

	cxi_lni_free(lni);

	desc.restricted_members = 1;
	rc = cxi_svc_update(dev, &desc);
	if (rc) {
		test_err("cxi_svc_update failed: %d\n", rc);
		goto err_free_svc;
	}

	lni = cxi_lni_alloc(dev, desc.svc_id);
	if (!IS_ERR(lni)) {
		test_err("cxi_lni_alloc did not return error\n");
		rc = -1;
		cxi_lni_free(lni);
		goto err_free_svc;
	}

	rc = 0;

err_free_svc:
	cxi_svc_destroy(dev, desc.svc_id);
err:
	return rc;
}

static int test_service_vni_range(struct cxi_dev *dev)
{
	int rc;
	struct cxi_lni *lni;
	struct cxi_svc_desc desc = {
		.enable = 1,
		.is_system_svc = 1,
		.num_vld_vnis = 0,
		.restricted_vnis = 0,
		.resource_limits = 1,
		.limits.type[CXI_RSRC_TYPE_PTE].max = 100,
		.limits.type[CXI_RSRC_TYPE_PTE].res = 100,
		.limits.type[CXI_RSRC_TYPE_TXQ].max = 100,
		.limits.type[CXI_RSRC_TYPE_TXQ].res = 100,
		.limits.type[CXI_RSRC_TYPE_TGQ].max = 100,
		.limits.type[CXI_RSRC_TYPE_TGQ].res = 100,
		.limits.type[CXI_RSRC_TYPE_EQ].max = 100,
		.limits.type[CXI_RSRC_TYPE_EQ].res = 100,
		.limits.type[CXI_RSRC_TYPE_CT].max = 100,
		.limits.type[CXI_RSRC_TYPE_CT].res = 100,
		.limits.type[CXI_RSRC_TYPE_LE].max = 100,
		.limits.type[CXI_RSRC_TYPE_LE].res = 100,
		.limits.type[CXI_RSRC_TYPE_TLE].max = 100,
		.limits.type[CXI_RSRC_TYPE_TLE].res = 100,
		.limits.type[CXI_RSRC_TYPE_AC].max = 4,
		.limits.type[CXI_RSRC_TYPE_AC].res = 4,
	};
	struct cxi_svc_fail_info info;
	unsigned int vni_min, vni_max;
	unsigned int test_min = 64;
	unsigned int test_max = 127; /* Good range (64 values, aligned) */
	unsigned int bad_min = 32;
	unsigned int bad_max = 95; /* Bad range (64 values, not aligned) */

	/* Allocate a Service with restricted_vnis = 0
	 * it should be disabled after creation.
	 */
	rc = cxi_svc_alloc(dev, &desc, &info, "vni-range-svc");
	if (rc < 0) {
		test_err("cxi_svc_alloc failed: %d\n", rc);
		goto err;
	}
	desc.svc_id = rc;

	/* Allocating an LNI should fail at this point */
	lni = cxi_lni_alloc(dev, desc.svc_id);
	if (PTR_ERR(lni) != -EKEYREVOKED) {
		rc = -EINVAL;
		test_err("cxi_lni_alloc did not return -EKEYREVOKED: %ld\n",
			 PTR_ERR(lni));
		cxi_lni_free(lni);
		goto err_free_svc;
	}

	/* Add a "bad" VNI Range */
	rc = cxi_svc_set_vni_range(dev, desc.svc_id, bad_min, bad_max);
	if (rc == 0) {
		test_err("cxi_svc_set_vni_range accepted invalid range [%u, %u]\n",
			 bad_min, bad_max);
		rc = -EINVAL;
		goto err_free_svc;
	}

	/* Add a good VNI range */
	rc = cxi_svc_set_vni_range(dev, desc.svc_id, test_min, test_max);
	if (rc) {
		test_err("cxi_svc_set_vni_range failed for valid range [%u, %u]: %d\n",
			 test_min, test_max, rc);
		goto err_free_svc;
	}

	/* Get the VNI range of the svc to verify it */
	rc = cxi_svc_get_vni_range(dev, desc.svc_id, &vni_min, &vni_max);
	if (rc) {
		test_err("cxi_svc_get_vni_range failed: %d\n", rc);
		goto err_free_svc;
	}
	if (vni_min != test_min || vni_max != test_max) {
		test_err("cxi_svc_get_vni_range mismatch: got [%u, %u], expected [%u, %u]\n",
			 vni_min, vni_max, test_min, test_max);
		rc = -EINVAL;
		goto err_free_svc;
	}

	/* Enable the Service */
	rc = cxi_svc_enable(dev, desc.svc_id, true);
	if (rc) {
		test_err("cxi_svc_enable failed: %d\n", rc);
		goto err_free_svc;
	}

	/* Allocating an LNI should work now */
	lni = cxi_lni_alloc(dev, desc.svc_id);
	if (IS_ERR(lni)) {
		rc = PTR_ERR(lni);
		test_err("cxi_lni_alloc failed:%d\n", rc);
		goto err_free_svc;
	}

	cxi_lni_free(lni);

err_free_svc:
	cxi_svc_destroy(dev, desc.svc_id);
err:
	return rc;
}

static int test_le_full(struct cxi_dev *dev)
{
	int i;
	int j;
	int rc;
	int rc1;
	int pools = CASS_NUM_LE_POOLS;
	struct cxi_svc_fail_info info = {};
	int svc_ids[CASS_NUM_LE_POOLS + 1];
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);
	struct cxi_svc_desc desc = {
		.enable = 1,
		.is_system_svc = 1,
		.restricted_vnis = 1,
		.num_vld_vnis = 1,
		.vnis[0] = 16,
		.resource_limits = 1,
		.limits.type[CXI_RSRC_TYPE_LE].max = 1,
		.limits.type[CXI_RSRC_TYPE_LE].res = 1,
	};

	/* get the next le pool id */
	rc = ida_simple_get(&hw->le_pool_ids[0], DEFAULT_LE_POOL_ID,
			    CASS_NUM_LE_POOLS, GFP_NOWAIT);
	if (rc < 0) {
		test_err("ida_simple_get failed %d\n", rc);
		goto err;
	}
	ida_simple_remove(&hw->le_pool_ids[0], rc);
	pools -= rc;
	rc = 0;

	for (i = 0; i <= pools; i++) {
		desc.vnis[0] = 16 + i;
		rc = cxi_svc_alloc(dev, &desc, &info, "le-full");
		if (i == pools && rc == -ENOSPC) {
			if (!info.no_le_pools)
				test_err("iter:%d no_le_pools check failed\n",
					 i);
			rc = 0;
			break;
		}

		if (rc < 0) {
			test_err("iter:%d cxi_svc_alloc failed:%d\n", i, rc);
			test_err("failinfo no_le_pools:%d no_tle_pools:%d\n",
				 info.no_le_pools, info.no_tle_pools);
			break;
		}
		svc_ids[i] = rc;
	}

	for (j = i - 1; j >= 0; j--) {
		rc1 = cxi_svc_destroy(dev, svc_ids[j]);
		if (rc1) {
			test_err("iter:%d cxi_svc_destroy svc_id:%d failed:%d\n",
				 j, svc_ids[j], rc1);
			rc = rc1;
			goto err;
		}
	}

err:
	return rc;
}

static int test_tle_full(struct cxi_dev *dev)
{
	int i;
	int j;
	int rc;
	int rc1;
	int pools = C_CQ_CFG_TLE_POOL_ENTRIES;
	struct cxi_svc_fail_info info = {};
	int svc_ids[C_CQ_CFG_TLE_POOL_ENTRIES + 1];
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);
	struct cxi_svc_desc desc = {
		.enable = 1,
		.is_system_svc = 1,
		.restricted_vnis = 1,
		.num_vld_vnis = 1,
		.vnis[0] = 16,
		.resource_limits = 1,
		.limits.type[CXI_RSRC_TYPE_TLE].max = 8,
		.limits.type[CXI_RSRC_TYPE_TLE].res = 8,
	};

	/* get the next tle pool id */
	rc = ida_simple_get(&hw->tle_pool_ids, DEFAULT_TLE_POOL_ID,
			    C_CQ_CFG_TLE_POOL_ENTRIES, GFP_NOWAIT);
	if (rc < 0) {
		test_err("ida_simple_get failed %d\n", rc);
		goto err;
	}
	ida_simple_remove(&hw->tle_pool_ids, rc);
	pools -= rc;
	rc = 0;

	for (i = 0; i <= pools; i++) {
		desc.vnis[0] = 16 + i;
		rc = cxi_svc_alloc(dev, &desc, &info, "tle-full");
		if (i == pools && rc == -ENOSPC) {
			if (!info.no_tle_pools)
				test_err("iter:%d no_tle_pools check failed\n",
					 i);
			rc = 0;
			break;
		}

		if (rc < 0) {
			test_err("iter:%d cxi_svc_alloc failed:%d\n", i, rc);
			test_err("failinfo no_le_pools:%d no_tle_pools:%d\n",
				 info.no_le_pools, info.no_tle_pools);
			break;
		}
		svc_ids[i] = rc;
	}

	for (j = i - 1; j >= 0; j--) {
		rc1 = cxi_svc_destroy(dev, svc_ids[j]);
		if (rc1) {
			test_err("iter:%d cxi_svc_destroy svc_id:%d failed:%d\n",
				 j, svc_ids[j], rc1);
			rc = rc1;
			goto err;
		}
	}

err:
	return rc;
}

static int run_tests(struct cxi_dev *dev)
{
	int rc;

	rc = test_le_full(dev);
	if (rc) {
		test_err("test_le_full failed: %d\n", rc);
		return -EIO;
	}

	rc = test_tle_full(dev);
	if (rc) {
		test_err("test_tle_full failed: %d\n", rc);
		return -EIO;
	}

	rc = test_service_busy(dev);
	if (rc) {
		test_err("test_service_busy failed: %d\n", rc);
		return -EIO;
	}

	rc = test_service_tle_in_use(dev);
	if (rc) {
		test_err("test_service_tle_in_use failed: %d\n", rc);
		return -EIO;
	}

	rc = test_service_modify(dev);
	if (rc) {
		test_err("test_service_modify failed: %d\n", rc);
		return -EIO;
	}

	rc = test_disabled_service(dev);
	if (rc) {
		test_err("test_service_tle_in_use failed: %d\n", rc);
		return -EIO;
	}

	rc = test_default_service(dev);
	if (rc) {
		test_err("test_default_service failed: %d\n", rc);
		return -EIO;
	}

	rc = test_restricted_members_service(dev);
	if (rc) {
		test_err("test_restricted_members_service failed: %d\n", rc);
		return -EIO;
	}
	rc = test_service_vni_range(dev);
	if (rc) {
		test_err("test_service_vni failed: %d\n", rc);
		return -EIO;
	}

	return 0;
}

static int add_device(struct cxi_dev *dev)
{
	int rc;
	struct tdev *tdev;

	tdev = kzalloc(sizeof(*tdev), GFP_KERNEL);
	if (!tdev)
		return -ENOMEM;

	tdev->dev = dev;

	rc = run_tests(dev);
	if (rc)
		goto fail;

	pr_info("Tests passed\n");

	pr_info("Adding template client for device %s\n", dev->name);
	mutex_lock(&device_list_mutex);
	list_add_tail(&tdev->dev_list, &dev_list);
	mutex_unlock(&device_list_mutex);

	return 0;

fail:
	return -ENODEV;
}

static void remove_device(struct cxi_dev *dev)
{
	struct tdev *tdev;
	bool found = false;

	/* Find the device in the list */
	mutex_lock(&device_list_mutex);
	list_for_each_entry_reverse(tdev, &dev_list, dev_list) {
		if (tdev->dev == dev) {
			found = true;
			list_del(&tdev->dev_list);
			break;
		}
	}
	mutex_unlock(&device_list_mutex);

	if (!found)
		return;

	kfree(tdev);
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
		pr_err("Couldn't register client\n");
		goto out;
	}

	return 0;

out:
	return ret;
}

static void __exit cleanup(void)
{
	pr_info("Removing template client\n");
	cxi_unregister_client(&cxiu_client);
}

module_init(init);
module_exit(cleanup);

MODULE_LICENSE("GPL v2");
MODULE_DESCRIPTION("CXI service test suite");
MODULE_AUTHOR("HPE");
