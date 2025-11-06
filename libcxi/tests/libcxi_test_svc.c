/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause */
/* Copyright 2018 Hewlett Packard Enterprise Development LP */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>

#include "libcxi_test_common.h"
#include "libcxi.h"

TestSuite(svc, .init = dev_setup, .fini = dev_teardown);

struct svc_alloc_params {
	bool resource_limits;
	struct cxi_rsrc_limits limits;
	bool rsrc_unavail[CXI_RSRC_TYPE_MAX];
	bool pass;
};

ParameterizedTestParameters(svc, svc_alloc)
{
	size_t param_sz;

	static struct svc_alloc_params params[] = {
		/* Good test */
		{
			.resource_limits = false,
			.limits = {},
			.pass = true,
		},
		/* Too many TXQs */
		{
			.resource_limits = true,
			.limits = (struct cxi_rsrc_limits) {
					.txqs = {
							.max = 1000,
							.res = 1025,
					},
					.eqs = {
							.max = 10,
							.res = 5,
					},
			},
			.rsrc_unavail[CXI_RSRC_TYPE_TXQ] = true,
			.pass = false,
		},
		/* All TXQ Res - (those in use by eth service)  */
		{
			.resource_limits = true,
			.limits = (struct cxi_rsrc_limits) {
					.txqs = {
							.max = 1024,
							.res = 1024 - 16,
					},
					.eqs = {
							.max = 10,
							.res = 5,
					},
			},
			.pass = true,
		},
	};

	param_sz = ARRAY_SIZE(params);
	return cr_make_param_array(struct svc_alloc_params, params,
				   param_sz);

}

ParameterizedTest(struct svc_alloc_params *param, svc, svc_alloc)
{
	int i;
	struct cxi_svc_fail_info fail_info = {};
	struct cxi_svc_desc svc_desc = {
		.resource_limits = param->resource_limits,
		.limits = param->limits,
		.num_vld_vnis = 1,
		.vnis[0] = 8,
	};

	int rc = cxil_alloc_svc(dev, &svc_desc, &fail_info);

	if (param->pass) {
		/* Check svc_id */
		cr_assert_gt(rc, 0, "cxil_alloc_svc(): Failed. Expected Success! rc:%d",
			     rc);
		svc_desc.svc_id = rc;

		/* Free svc */
		rc = cxil_destroy_svc(dev, svc_desc.svc_id);
		cr_assert_eq(rc, 0, "cxil_destroy_svc(): Failed. Couldn't free svc: %d. rc: %d",
			     svc_desc.svc_id, rc);
	} else {
		cr_assert_neq(rc, 0, "cxil_alloc_svc(): Succeeded. Expected Failure!");
		for (i = 0; i < CXI_RSRC_TYPE_MAX; i++) {
			if (param->rsrc_unavail[i]) {
				cr_assert_lt(fail_info.rsrc_avail[i],
					     param->limits.type[i].res,
					     "cxil_alloc_svc(): fail_info incorrect. Expected avail:%d to be less than requested:%d",
					     fail_info.rsrc_avail[i],
					     param->limits.type[i].res);
			}
		}
	}
}

Test(svc, svc_null)
{
	int rc;

	rc = cxil_alloc_svc(dev, NULL, NULL);
	cr_assert_eq(rc, -EINVAL);

	rc = cxil_destroy_svc(dev, 9001);
	cr_assert_eq(rc, -EINVAL);
}

Test(svc, svc_get)
{
	int rc;
	struct cxi_svc_desc desc;

	/* Garbage ID should fail */
	rc = cxil_get_svc(dev, 99, &desc);
	cr_assert_eq(rc, -EINVAL, "cxil_get_svc(): Succeeded. Expected Failure! rc:%d",
		     rc);

	/* Should be able to retrieve default service */
	rc = cxil_get_svc(dev, CXI_DEFAULT_SVC_ID, &desc);
	cr_assert_eq(rc, 0, "cxil_get_svc(): Failed. Expected Success! rc:%d",
		     rc);
}

Test(svc, svc_update)
{
	int rc, svc_id;
	struct cxi_svc_fail_info fail_info = {};
	struct cxi_svc_desc svc_desc = {
		.restricted_vnis = 1,
		.num_vld_vnis = 1,
		.vnis[0] = 8,
	};
	struct cxi_svc_desc comp_desc = {
		.restricted_vnis = 1,
		.num_vld_vnis = 1,
		.vnis[0] = 8,
	};

	/* Initial allocation of service */
	rc = cxil_alloc_svc(dev, &svc_desc, &fail_info);
	cr_assert_gt(rc, 0, "cxil_alloc_svc(): Failed. Expected Success! rc:%d",
		     rc);

	svc_id = rc;

	/* Updating nonexistent service should fail */
	svc_desc.svc_id = 13;
	rc = cxil_update_svc(dev, &svc_desc, &fail_info);
	cr_assert_eq(rc, -EINVAL, "cxil_update_svc(): Succeeded. (Bad svc_id) Expected Failure!");

	/* Put good ID back in descriptor */
	svc_desc.svc_id = svc_id;

	/* Changing resource limits should fail */
	svc_desc.resource_limits = 1;
	rc = cxil_update_svc(dev, &svc_desc, &fail_info);
	cr_assert_eq(rc, -EINVAL, "cxil_update_svc(): Succeeded.(resource_limits) Expected Failure!");

	/* Revert resource limits change */
	svc_desc.resource_limits = 0;

	/* Updating VNIs properly should work */
	svc_desc.num_vld_vnis = 1;
	svc_desc.vnis[0] = 11;
	rc = cxil_update_svc(dev, &svc_desc, &fail_info);
	cr_assert_eq(rc, 0, "cxil_update_svc(): Failed with rc: %d.(restricted_vnis) Expected Success!",
		     rc);

	/* Updating members should work */
	svc_desc.restricted_members = 1;
	rc = cxil_update_svc(dev, &svc_desc, &fail_info);
	cr_assert_eq(rc, 0, "cxil_update_svc(): Failed with rc: %d.(restricted_members) Expected Success!",
		     rc);

	/* Updating TCs should work */
	svc_desc.restricted_tcs = 1;
	rc = cxil_update_svc(dev, &svc_desc, &fail_info);
	cr_assert_eq(rc, 0, "cxil_update_svc(): Failed with rc: %d.(restricted_tcs) Expected Success!",
		     rc);

	/* Disabling Service should work */
	svc_desc.enable = 0;
	rc = cxil_update_svc(dev, &svc_desc, &fail_info);
	cr_assert_eq(rc, 0, "cxil_update_svc(): Failed with rc: %d.(enable) Expected Success!",
		     rc);

	/* Updating lnis_per_rgid should work */
	rc = cxil_set_svc_lpr(dev, svc_id, 2);
	cr_assert_eq(rc, 0, "cxil_set_svc_lpr(): Failed with rc: %d. Expected Success!",
		     rc);

	/* Compare our descriptor with Kernel's view of descriptor */
	rc = cxil_get_svc(dev, svc_desc.svc_id, &comp_desc);
	cr_assert_eq(rc, 0, "cxil_get_svc(): Failed with rc: %d. Expected Success!",
		     rc);

	cr_assert_eq(comp_desc.num_vld_vnis,
		     svc_desc.num_vld_vnis,
		     "cxil_get_svc(): Inconsistent result (num_vld_vnis). Local: %d Kernel: %d",
		     svc_desc.num_vld_vnis,
		     comp_desc.num_vld_vnis);
	cr_assert_eq(comp_desc.resource_limits,
		     svc_desc.resource_limits,
		     "cxil_get_svc(): Inconsistent result (resource_limits). Local: %d Kernel: %d",
		     svc_desc.resource_limits,
		     comp_desc.resource_limits);
	cr_assert_eq(comp_desc.restricted_members,
		     svc_desc.restricted_members,
		     "cxil_get_svc(): Inconsistent result (restricted_members). Local: %d Kernel: %d",
		     svc_desc.restricted_members,
		     comp_desc.restricted_members);

	/* Destroy SVC*/
	rc = cxil_destroy_svc(dev, svc_desc.svc_id);
	cr_assert_eq(rc, 0, "cxil_destroy_svc(): Failed. Couldn't free svc: %d, rc: %d",
		     svc_desc.svc_id, rc);
}

struct svc_ugid_params {
	bool pass;
	bool su;
	enum cxi_svc_member_type type;
	uid_t ugid;
};

ParameterizedTestParameters(svc, svc_ugid)
{
	size_t param_sz;

	static struct svc_ugid_params params[] = {
		/* Root GID  - as root */
		{
			.pass = true,
			.su = false,
			.type = CXI_SVC_MEMBER_GID,
			.ugid = 0,
		},
		/* Root UID - as root */
		{
			.pass = true,
			.su = false,
			.type = CXI_SVC_MEMBER_UID,
			.ugid = 0,
		},
		/* random GID - as root */
		{
			.pass = false,
			.su = false,
			.type = CXI_SVC_MEMBER_GID,
			.ugid = 747,
		},
		/* 'nobody' UID - as root */
		{
			.pass = false,
			.su = false,
			.type = CXI_SVC_MEMBER_UID,
			.ugid = 65534,
		},
		/* 'nobody' UID - su to 'nobody' */
		{
			.pass = true,
			.su = true,
			.type = CXI_SVC_MEMBER_UID,
			.ugid = 65534,
		},

	};

	param_sz = ARRAY_SIZE(params);
	return cr_make_param_array(struct svc_ugid_params, params,
				   param_sz);
}

ParameterizedTest(struct svc_ugid_params *param, svc, svc_ugid)
{
	int rc;
	struct cxi_svc_desc svc_desc = {
		.restricted_members = 1,
		.restricted_vnis = 1,
		.num_vld_vnis = 1,
		.vnis[0] = 8,
	};

	svc_desc.members[0].svc_member.gid = param->ugid;
	svc_desc.members[0].type = param->type;

	/* Allocate SVC */
	rc = cxil_alloc_svc(dev, &svc_desc, NULL);
	cr_assert_gt(rc, 0, "cxil_alloc_svc() Failed. rc: %d",
		     rc);

	/* Save off svc_id */
	svc_desc.svc_id = rc;

	if (param->su) {
		rc = seteuid(param->ugid);
		cr_assert_eq(rc, 0, "seteuid() failed");
	}

	/* Allocate LNI */
	rc = cxil_alloc_lni(dev, &lni, svc_desc.svc_id);
	if (param->pass)
		cr_assert_eq(rc, 0, "cxil_alloc_lni() Failed. Expected rc: 0 received rc: %d", rc);
	else
		cr_assert_neq(rc, 0, "Shouldn't have been able to allocate LNI. expected: %d received %d",
			      -EPERM, rc);
	/* Revert to root if needed */
	if (param->su) {
		rc = seteuid(0);
		cr_assert_eq(rc, 0, "seteuid() failed");
	}

	/* Destroy LNI if it was allocated */
	if (param->pass) {
		rc = cxil_destroy_lni(lni);
		cr_assert_eq(rc, 0, "Destroy LNI failed rc: %d", rc);
	}

	/* Destroy SVC */
	rc = cxil_destroy_svc(dev, svc_desc.svc_id);
	cr_assert_eq(rc, 0, "cxil_destroy_svc(): Failed. rc: %d",
		     rc);
}

#define TEST_UID 200
#define TEST_GID 300
Test(svc, svc_profile_ignore)
{
	int rc;
	struct cxi_svc_desc svc_desc = {
		.restricted_members = 1,
		.restricted_vnis = 1,
		.num_vld_vnis = 1,
		.vnis[0] = 8,
	};

	/* member[1].type is ignore therefore no ac entry */
	svc_desc.members[0].svc_member.uid = TEST_UID;
	svc_desc.members[0].type = CXI_SVC_MEMBER_UID;

	/* Allocate SVC */
	rc = cxil_alloc_svc(dev, &svc_desc, NULL);
	cr_assert_gt(rc, 0, "cxil_alloc_svc() Failed. rc: %d", rc);

	svc_desc.svc_id = rc;

	rc = cxil_alloc_lni(dev, &lni, svc_desc.svc_id);
	cr_assert_eq(rc, -EPERM, "cxil_alloc_lni Expected rc:%d received rc:%d",
		      -EPERM, rc);

	rc = seteuid(TEST_UID);
	cr_assert_eq(rc, 0, "seteuid() failed");

	rc = cxil_alloc_lni(dev, &lni, svc_desc.svc_id);
	cr_assert_eq(rc, 0, "cxil_alloc_lni Expected rc:0 received rc:%d", rc);

	rc = cxil_destroy_lni(lni);
	cr_assert_eq(rc, 0, "Destroy LNI failed rc: %d", rc);

	/* Set back to root to destroy svc */
	rc = seteuid(0);
	cr_assert_eq(rc, 0, "seteuid() failed");

	/* Destroy SVC */
	rc = cxil_destroy_svc(dev, svc_desc.svc_id);
	cr_assert_eq(rc, 0, "cxil_destroy_svc(): Failed. rc: %d",
		     rc);
}

Test(svc, svc_member_perm)
{
	int rc;
	struct cxi_svc_desc svc_desc = {
		.restricted_members = 1,
		.restricted_vnis = 1,
		.num_vld_vnis = 1,
		.vnis[0] = 8,
	};

	svc_desc.members[0].svc_member.uid = TEST_UID;
	svc_desc.members[0].type = CXI_SVC_MEMBER_UID;
	svc_desc.members[1].svc_member.uid = TEST_GID;
	svc_desc.members[1].type = CXI_SVC_MEMBER_GID;

	/* Allocate SVC */
	rc = cxil_alloc_svc(dev, &svc_desc, NULL);
	cr_assert_gt(rc, 0, "cxil_alloc_svc() Failed. rc: %d", rc);

	svc_desc.svc_id = rc;

	/* Verify cannot allocate lni as root user */
	rc = cxil_alloc_lni(dev, &lni, svc_desc.svc_id);
	cr_assert_eq(rc, -EPERM, "cxil_alloc_lni Expected rc:%d received rc:%d",
		      -EPERM, rc);

	/* Test with TEST uid */
	rc = seteuid(TEST_UID);
	cr_assert_eq(rc, 0, "seteuid() failed");

	rc = cxil_alloc_lni(dev, &lni, svc_desc.svc_id);
	cr_assert_eq(rc, 0, "cxil_alloc_lni Expected rc:0 received rc:%d", rc);

	rc = cxil_alloc_domain(lni, svc_desc.vnis[0], 10, &domain);
	cr_assert_eq(rc, 0, "cxil_alloc_domain failed rc:%d", rc);

	rc = cxil_destroy_domain(domain);
	cr_assert_eq(rc, 0, "cxil_destroy_domain failed rc:%d", rc);

	rc = cxil_alloc_cp(lni, svc_desc.vnis[0], CXI_TC_BEST_EFFORT,
			   CXI_TC_TYPE_DEFAULT, &cp);
	cr_assert_eq(rc, 0, "cxil_alloc_cp() failed %d", rc);
	cr_assert_neq(cp, NULL);

	rc = cxil_destroy_cp(cp);
	cr_assert_eq(rc, 0, "Destroy CP failed %d", rc);

	rc = cxil_destroy_lni(lni);
	cr_assert_eq(rc, 0, "Destroy LNI failed rc: %d", rc);

	/* Back to root uid */
	rc = seteuid(0);
	cr_assert_eq(rc, 0, "seteuid() failed");

	/* Now test with TEST gid */
	rc = setegid(TEST_GID);
	cr_assert_eq(rc, 0, "setegid() failed");

	rc = cxil_alloc_lni(dev, &lni, svc_desc.svc_id);
	cr_assert_eq(rc, 0, "cxil_alloc_lni Expected rc:0 received rc:%d", rc);

	rc = cxil_alloc_domain(lni, svc_desc.vnis[0], 10, &domain);
	cr_assert_eq(rc, 0, "cxil_alloc_domain failed rc:%d", rc);

	rc = cxil_destroy_domain(domain);
	cr_assert_eq(rc, 0, "cxil_destroy_domain failed rc:%d", rc);

	rc = cxil_alloc_cp(lni, svc_desc.vnis[0], CXI_TC_BEST_EFFORT,
			   CXI_TC_TYPE_DEFAULT, &cp);
	cr_assert_eq(rc, 0, "cxil_alloc_cp() failed %d", rc);
	cr_assert_neq(cp, NULL);

	rc = cxil_destroy_cp(cp);
	cr_assert_eq(rc, 0, "Destroy CP failed %d", rc);

	/* gid back to root */
	rc = setegid(0);
	cr_assert_eq(rc, 0, "seteuid() failed");

	rc = cxil_alloc_domain(lni, svc_desc.vnis[0], 10, &domain);
	cr_assert_eq(rc, -EPERM, "cxil_alloc_domain expected:%d received:%d",
		      -EPERM, rc);

	rc = cxil_alloc_cp(lni, svc_desc.vnis[0], CXI_TC_BEST_EFFORT,
			   CXI_TC_TYPE_DEFAULT, &cp);
	cr_assert_eq(rc, -EPERM, "cxil_alloc_cp expected:%d received:%d",
		      -EPERM, rc);

	rc = cxil_destroy_lni(lni);
	cr_assert_eq(rc, 0, "Destroy LNI failed rc: %d", rc);

	/* Destroy SVC */
	rc = cxil_destroy_svc(dev, svc_desc.svc_id);
	cr_assert_eq(rc, 0, "cxil_destroy_svc(): Failed. rc: %d",
		     rc);
}

/* Start as root, allocate an LNI, and change user to allocate an AC */
Test(svc, svc_change_uid)
{
	int rc;
	size_t qlen = s_page_size;
	void *buf;
	struct cxi_md *buf_md;

	struct cxi_svc_desc svc_desc = {
		.restricted_members = 1,
		.restricted_vnis = 1,
		.num_vld_vnis = 1,
		.vnis[0] = 8,
	};

	svc_desc.members[0].svc_member.uid = 0;
	svc_desc.members[0].type = CXI_SVC_MEMBER_UID;

	/* Allocate SVC */
	rc = cxil_alloc_svc(dev, &svc_desc, NULL);
	cr_assert_gt(rc, 0, "cxil_alloc_svc() Failed. rc: %d",
		     rc);

	/* Save off svc_id */
	svc_desc.svc_id = rc;

	/* Allocate LNI */
	rc = cxil_alloc_lni(dev, &lni, svc_desc.svc_id);
	cr_assert_eq(rc, 0, "cxil_alloc_lni() Failed. Expected rc: 0 received rc: %d", rc);

	/* Setup cp */
	rc = cxil_alloc_cp(lni, vni, CXI_TC_BEST_EFFORT, CXI_TC_TYPE_DEFAULT,
			   &cp);
	cr_assert_eq(rc, 0, "cxil_alloc_cp() failed %d", rc);
	cr_assert_neq(cp, NULL);
	cr_log_info("assigned LCID: %u\n", cp->lcid);

	/* Switch to non-root user. Should still be able to allocate an AC */
	rc = seteuid(65534);
	cr_assert_eq(rc, 0, "seteuid() failed");

	/* Setup buffer */
	buf = aligned_alloc(s_page_size, qlen);
	cr_assert(buf);
	memset(buf, 0, qlen);

	rc = cxil_map(lni, buf, qlen,
		      CXI_MAP_PIN | CXI_MAP_READ | CXI_MAP_WRITE,
		      NULL, &buf_md);
	cr_assert(!rc, "Received :%d", rc);

	cxil_unmap(buf_md);
	free(buf);

	/* Destroy cp */
	rc = cxil_destroy_cp(cp);
	cr_assert_eq(rc, 0, "Destroy CP failed %d", rc);
	cp = NULL;

	/* Switch back to root user */
	rc = seteuid(0);
	cr_assert_eq(rc, 0, "seteuid() failed");

	/* Destroy LNI */
	rc = cxil_destroy_lni(lni);
	cr_assert_eq(rc, 0, "Destroy LNI failed %d", rc);

	/* Destroy SVC*/
	rc = cxil_destroy_svc(dev, svc_desc.svc_id);
	cr_assert_eq(rc, 0, "cxil_destroy_svc(): Failed. Couldn't free svc: %d, rc: %d",
		     svc_desc.svc_id, rc);
}

Test(svc, svc_max)
{
	int rc, i;
	size_t limit = 10;
	size_t qlen = s_page_size;
	void *eq_buf, *buf;
	struct cxi_md *eq_buf_md, *buf_md;
	struct cxi_cq **txqs;
	struct cxi_cq **tgqs;
	struct cxi_eq **eqs;
	struct cxil_pte **ptes;
	struct cxi_ct **cts;
	struct c_ct_writeback *wb;
	struct cxi_eq_attr attr = {};
	struct cxi_pt_alloc_opts pte_opts = {};
	struct cxi_rsrc_use rsrc_use;

	struct cxi_rsrc_limits limits = {
		.acs = {
			.max = 1,
			.res = 1,
		},
		.eqs = {
			.max = limit,
			.res = limit,
		},
		.cts = {
			.max = limit,
			.res = limit,
		},
		.ptes = {
			.max = limit,
			.res = limit,
		},
		.txqs = {
			.max = limit,
			.res = limit,
		},
		.tgqs = {
			.max = limit,
			.res = limit,
		},
	};
	struct cxi_svc_desc svc_desc = {
		.resource_limits = true,
		.limits = limits,
		.restricted_vnis = 1,
		.num_vld_vnis = 1,
		.vnis[0] = 8,
	};
	struct cxi_cq_alloc_opts opts = {
		.count = 1024,
		.flags = CXI_CQ_IS_TX,
	};

	/* Allocate svc */
	rc = cxil_alloc_svc(dev, &svc_desc, NULL);
	cr_assert_gt(rc, 0, "cxil_alloc_svc(): Failed. Expected Success! rc:%d",
		     rc);

	svc_desc.svc_id = rc;

	/* Allocate lni */
	rc = cxil_alloc_lni(dev, &lni, svc_desc.svc_id);
	cr_assert_eq(rc, 0, "rc mismatch, expected: %d received %d",
		     0, rc);

	/* Setup cp */
	rc = cxil_alloc_cp(lni, vni, CXI_TC_BEST_EFFORT, CXI_TC_TYPE_DEFAULT,
			   &cp);
	cr_assert_eq(rc, 0, "cxil_alloc_cp() failed %d", rc);
	cr_assert_neq(cp, NULL);
	cr_log_info("assigned LCID: %u\n", cp->lcid);

	opts.lcid = cp->lcid,

	/* Setup buffer */
	eq_buf = aligned_alloc(s_page_size, qlen);
	cr_assert(eq_buf);
	memset(eq_buf, 0, qlen);

	rc = cxil_map(lni, eq_buf, qlen,
		      CXI_MAP_PIN | CXI_MAP_READ | CXI_MAP_WRITE,
		      NULL, &eq_buf_md);
	cr_assert(!rc, "Received :%d", rc);

	attr.queue = eq_buf;
	attr.queue_len = qlen;

	/* Ensure second AC is not allocated.
	 * Note that different MAP flags are needed so that a second AC
	 * will even be attempted to be allocated. (WRITE and READ are
	 * not considered).
	 */
	buf = aligned_alloc(s_page_size, qlen);
	rc = cxil_map(lni, buf, qlen,
		      CXI_MAP_PIN | CXI_MAP_READ | CXI_MAP_WRITE | 0x80000,
		      NULL, &buf_md);
	cr_assert_eq(rc, -ENOSPC,
		     "cxil_map Expected -ENOSPC, Received: %d", rc);
	free(buf);

	/* Allocate multiple EQs */
	eqs = calloc(limit+1, sizeof(*eqs));
	cr_assert(eqs);

	for (i = 0; i < limit; i++) {
		rc = cxil_alloc_evtq(lni, eq_buf_md, &attr, NULL,
				     NULL, &eqs[i]);
		cr_assert(!rc);
	}

	/* Ensure extra EQ can't be allocated */
	rc = cxil_alloc_evtq(lni, eq_buf_md, &attr, NULL,
			     NULL, &eqs[i]);
	cr_assert_eq(rc, -ENOSPC,
		     "cxil_alloc_evtq Expected -ENOSPC. Received :%d", rc);

	/* Allocate multiple PTES */
	ptes = calloc(limit+1, sizeof(*ptes));
	cr_assert(ptes);

	for (i = 0; i < limit; i++) {
		rc = cxil_alloc_pte(lni, eqs[i], &pte_opts, &ptes[i]);
		cr_assert(!rc);
	}

	/* Ensure extra PTE can't be allocated */
	rc = cxil_alloc_pte(lni, eqs[i], &pte_opts, &ptes[i]);
	cr_assert_eq(rc, -ENOSPC,
		     "cxil_alloc_pte Expected -ENOSPC. Received :%d", rc);

	/* Allocate multiple TXQs */
	txqs = calloc(limit+1, sizeof(*txqs));
	cr_assert(txqs);

	for (i = 0; i < limit; i++) {
		rc = cxil_alloc_cmdq(lni, NULL, &opts, &txqs[i]);
		cr_assert(!rc);
	}

	/* Ensure extra TXQ can't be allocated */
	rc = cxil_alloc_cmdq(lni, NULL, &opts, &txqs[limit]);
	cr_assert_eq(rc, -ENOSPC,
		     "cxil_alloc_cmdq Expected -ENOSPC. Received :%d", rc);

	/* Allocate multiple TGQs */
	opts.flags = 0;
	tgqs = calloc(limit+1, sizeof(*tgqs));
	cr_assert(tgqs);

	for (i = 0; i < limit; i++) {
		rc = cxil_alloc_cmdq(lni, NULL, &opts, &tgqs[i]);
		cr_assert(!rc);
	}

	/* Ensure extra TGQ can't be allocated */
	rc = cxil_alloc_cmdq(lni, NULL, &opts, &tgqs[limit]);
	cr_assert_eq(rc, -ENOSPC,
		     "cxil_alloc_cmdq Expected -ENOSPC. Received :%d", rc);

	/* Allocate multiple CTs */
	wb = calloc(1, sizeof(*wb));
	cr_assert(wb);
	cts = calloc(limit+1, sizeof(*cts));
	cr_assert(cts);

	for (i = 0; i < limit; i++) {
		rc = cxil_alloc_ct(lni, wb, &cts[i]);
		cr_assert_eq(rc, 0, "Failed cxil_alloc_ct() failed %d", rc);
	}

	/* Ensure extra CT can't be allocated */
	rc = cxil_alloc_ct(lni, wb, &cts[i]);
	cr_assert_eq(rc, -ENOSPC,
		     "cxil_alloc_ct Expected -ENOSPC. Received: %d", rc);

	/* Validate Resource Usage */
	rc = cxil_get_svc_rsrc_use(dev, svc_desc.svc_id, &rsrc_use);
	cr_assert_eq(rc, 0,
		     "cxil_get_svc_rsrc_use Expected rc 0. Received: %d", rc);

	cr_assert_eq(rsrc_use.in_use[CXI_RSRC_TYPE_AC], 1,
		     "Expected: %u in use. Observed: %u", 1,
		     rsrc_use.in_use[CXI_RSRC_TYPE_AC]);
	cr_assert_eq(rsrc_use.in_use[CXI_RSRC_TYPE_EQ], limit,
		     "Expected: %lu in use. Observed: %u", limit,
		     rsrc_use.in_use[CXI_RSRC_TYPE_EQ]);
	cr_assert_eq(rsrc_use.in_use[CXI_RSRC_TYPE_PTE], limit,
		     "Expected: %lu in use. Observed: %u", limit,
		     rsrc_use.in_use[CXI_RSRC_TYPE_PTE]);
	cr_assert_eq(rsrc_use.in_use[CXI_RSRC_TYPE_TXQ], limit,
		     "Expected: %lu in use. Observed: %u", limit,
		     rsrc_use.in_use[CXI_RSRC_TYPE_TXQ]);
	cr_assert_eq(rsrc_use.in_use[CXI_RSRC_TYPE_TGQ], limit,
		     "Expected: %lu in use. Observed: %u", limit,
		     rsrc_use.in_use[CXI_RSRC_TYPE_TGQ]);
	cr_assert_eq(rsrc_use.in_use[CXI_RSRC_TYPE_CT], limit,
		     "Expected: %lu in use. Observed: %u", limit,
		     rsrc_use.in_use[CXI_RSRC_TYPE_CT]);

	/* Free CTs*/
	for (i = 0; i < limit; i++) {
		rc = cxil_destroy_ct(cts[i]);
		cr_assert_eq(rc, 0);
	}
	free(cts);
	free(wb);

	/* Free TGQs */
	for (i = 0; i < limit; i++) {
		rc = cxil_destroy_cmdq(tgqs[i]);
		cr_assert(!rc);
	}
	free(tgqs);

	/* Free TXQs */
	for (i = 0; i < limit; i++) {
		rc = cxil_destroy_cmdq(txqs[i]);
		cr_assert(!rc);
	}
	free(txqs);

	/* Free PTEs */
	for (i = 0; i < limit; i++) {
		rc = cxil_destroy_pte(ptes[i]);
		cr_assert_eq(rc, 0);
	}
	free(ptes);

	/* Free EQs */
	for (i = 0; i < limit; i++) {
		rc = cxil_destroy_evtq(eqs[i]);
		cr_assert(!rc);
	}
	cxil_unmap(eq_buf_md);
	free(eq_buf);
	free(eqs);

	/* Destroy cp */
	rc = cxil_destroy_cp(cp);
	cr_assert_eq(rc, 0, "Destroy CP failed %d", rc);
	cp = NULL;

	/* Destroy LNI */
	rc = cxil_destroy_lni(lni);
	cr_assert_eq(rc, 0, "Destroy LNI failed %d", rc);

	/* Destroy SVC*/
	rc = cxil_destroy_svc(dev, svc_desc.svc_id);
	cr_assert_eq(rc, 0, "cxil_destroy_svc(): Failed. Couldn't free svc: %d, rc: %d",
		     svc_desc.svc_id, rc);
}

Test(svc, svc_vni_range)
{
	int rc;
	uint16_t vni_min = 16;
	uint16_t vni_max = 31;
	uint16_t min;
	uint16_t max;
	bool exclusive = false;
	struct cxi_svc_desc desc = {};

	desc.limits.type[CXI_RSRC_TYPE_PTE].max = C_NUM_PTLTES;
	desc.limits.type[CXI_RSRC_TYPE_TXQ].max = C_NUM_TRANSMIT_CQS;
	desc.limits.type[CXI_RSRC_TYPE_TGQ].max = C_NUM_TARGET_CQS;
	desc.limits.type[CXI_RSRC_TYPE_EQ].max = C_NUM_EQS - 1;
	desc.limits.type[CXI_RSRC_TYPE_CT].max = C_NUM_CTS - 1;
	desc.limits.type[CXI_RSRC_TYPE_LE].max = C_LPE_STS_LIST_ENTRIES_ENTRIES / C_PE_COUNT;
	desc.limits.type[CXI_RSRC_TYPE_AC].max = C_ATU_CFG_AC_TABLE_ENTRIES - 2;

	/* Using the default svc should fail */
	rc = cxil_alloc_svc(dev, &desc, NULL);
	cr_assert_gt(rc, 0, "cxil_alloc_svc failed rc:%d", rc);
	desc.svc_id = rc;

	rc = cxil_svc_set_vni_range(dev, desc.svc_id, vni_min, vni_max);
	cr_assert_eq(rc, 0, "cxil_svc_set_vni_range failed svc_id:%d rc: %d",
		     desc.svc_id, rc);

	rc = cxil_svc_get_vni_range(dev, desc.svc_id, &min, &max);
	cr_assert_eq(rc, 0, "cxil_svc_get_vni_range failed svc_id:%d rc: %d",
		     desc.svc_id, rc);
	cr_assert(vni_min == min && vni_max == max,
		     "vni_min:%d exp:%d vni_max:%d exp:%d",
		     min, vni_min, max, vni_max);

	rc = cxil_svc_set_exclusive_cp(dev, desc.svc_id, true);
	cr_assert_eq(rc, 0, "cxil_svc_set_exclusive_cp failed svc_id:%d rc: %d",
		     desc.svc_id, rc);

	rc = cxil_svc_get_exclusive_cp(dev, desc.svc_id, &exclusive);
	cr_assert_eq(rc, 0, "cxil_get_svc_exclusive_cp failed svc_id:%d rc: %d",
		     desc.svc_id, rc);
	cr_log_info("exclusive:%d\n", exclusive);

	rc = cxil_svc_enable(dev, desc.svc_id, true);
	cr_assert_eq(rc, 0, "cxil_svc_enable failed svc_id:%d rc: %d",
		     desc.svc_id, rc);

	rc = cxil_destroy_svc(dev, desc.svc_id);
	cr_assert_eq(rc, 0, "cxil_destroy_svc(): Failed. Couldn't free svc: %d. rc: %d",
		     desc.svc_id, rc);
}

Test(svc, svc_vni_overlap)
{
	int rc;
	struct cxi_svc_desc svc_desc = {};

	svc_desc.restricted_vnis = 1;
	svc_desc.num_vld_vnis = 1;
	svc_desc.vnis[0] = CXI_DEFAULT_SVC_ID;

	/* Using the default svc should fail */
	rc = cxil_alloc_svc(dev, &svc_desc, NULL);
	cr_assert_eq(rc, -EEXIST, "cxil_alloc_svc() Failed. rc: %d", rc);
}

struct le_tle_params {
	struct cxi_rsrc_limits limits;
	int num_services;
};

ParameterizedTestParameters(svc, le_tle)
{
	size_t param_sz;

	static struct le_tle_params params[] = {
		/* LEs */
		{
			.limits = (struct cxi_rsrc_limits) {
				.les = {
					.max = 16384,
					.res = 1092,
				},
			},
			.num_services = 15,
		},
		/* TLEs */
		{
			.limits = (struct cxi_rsrc_limits) {
				.tles = {
					.max = 10,
					.res = 7,
				},
			},
			.num_services = 3,
		},
	};

	param_sz = ARRAY_SIZE(params);
	return cr_make_param_array(struct le_tle_params, params,
				   param_sz);

}

ParameterizedTest(struct le_tle_params *param, svc, le_tle)
{
	int rc, i;
	int num_svcs = param->num_services;
	struct cxi_svc_desc *svcs;
	bool alloc_svcs_failed = false;

	/* When running on actual HW, the LE test case needs to account for
	 * LEs currently in use to determine max number of svcs to allocate.
	 */
	if (!is_vm() && num_svcs == 15) {
		FILE *fp = popen("cat /sys/kernel/debug/cxi/cxi0/services | awk '$0 ~ / Res / {s+=$8} END {print s}'", "r");
		if (fp) {
			int les_in_use;
			int ret;

			ret = fscanf(fp, "%d", &les_in_use);
			cr_assert_eq(ret, 1, "ret=%d\n", ret);
			pclose(fp);
			num_svcs = (param->limits.les.max - les_in_use) / param->limits.les.res;
			cr_log_info("les_in_use:%d num_svcs:%d\n", les_in_use,
				    num_svcs);
			if (!num_svcs)
				cr_skip("Too many LEs are currently in use to run this test");
		} else {
			cr_log_info("Unable to determine LEs in use");
		}
	}

	/* Allocate max number of services with le/tle limits */
	svcs = calloc(num_svcs + 1, sizeof(*svcs));
	for (i = 0; i < num_svcs; i++) {
		svcs[i].resource_limits = true;
		svcs[i].limits = param->limits;
		svcs[i].restricted_vnis = 1;
		svcs[i].num_vld_vnis = 1,
		svcs[i].vnis[0] = 11 + i,
		rc = cxil_alloc_svc(dev, &svcs[i], NULL);
		if (rc < 0) {
			alloc_svcs_failed = true;
			cr_log_info("cxil_alloc_svc()[%d]: rc:%d\n", i, rc);
			break;
		}
		svcs[i].svc_id = rc;
	}

	if (alloc_svcs_failed) {
		for (i--; i >= 0; i--) {
			rc = cxil_destroy_svc(dev, svcs[i].svc_id);
			if (rc)
				cr_log_info("cxil_destroy_svc() i:%d svc_id:%d rc:%d\n", i,
				     svcs[i].svc_id, rc);
		}
		cr_assert(0, "alloc svc failed");
	}

	/* Show that another svc with le/tle reservations cannot be allocated */
	svcs[num_svcs].resource_limits = true;
	svcs[num_svcs].limits = param->limits;
	svcs[num_svcs].restricted_vnis = 1;
	svcs[num_svcs].num_vld_vnis = 1;
	svcs[num_svcs].vnis[0] = 8;
	rc = cxil_alloc_svc(dev, &svcs[num_svcs], NULL);
	cr_assert_eq(rc, -ENOSPC);

	/* Destroy services */
	for (i = 0; i < num_svcs; i++) {
		rc = cxil_destroy_svc(dev, svcs[i].svc_id);
		cr_assert_eq(rc, 0, "cxil_destroy_svc(): Failed. Couldn't free svc: %d, rc: %d",
			     svcs[i].svc_id, rc);
	}
	free(svcs);
}

Test(svc, dev_resource_limits)
{
	struct cxil_devinfo *info = &dev->info;

	cr_assert_neq(0, info->num_ptes, "pte resource_limit not set");
	cr_assert_neq(0, info->num_txqs, "txq resource_limit not set");
	cr_assert_neq(0, info->num_tgqs, "tgq resource_limit not set");
	cr_assert_neq(0, info->num_eqs, "eq resource_limit not set");
	cr_assert_neq(0, info->num_cts, "ct resource_limit not set");
	cr_assert_neq(0, info->num_acs, "ac resource_limit not set");
	cr_assert_neq(0, info->num_tles, "tle resource_limit not set");
	cr_assert_neq(0, info->num_les, "le resource_limit not set");
}

Test(svc, svc_list)
{
	int rc;
	struct cxil_svc_list *list = NULL;

	rc = cxil_get_svc_list(dev, &list);
	cr_assert_eq(0, rc, "svc_list rc: %d", rc);
	cr_assert_neq(0, list->count, "0 svc descriptors were copied over");
	cxil_free_svc_list(list);
}

Test(svc, svc_demo)
{
	int rc;
	int i;
	struct cxil_svc_list *list = NULL;
	struct cxil_devinfo *info = &dev->info;
	int res_pte = 0, res_txq = 0, res_tgq = 0, res_eq = 0, res_ct = 0,
			res_ac = 0, res_tle = 0, res_le = 0;

	rc = cxil_get_svc_list(dev, &list);
	cr_assert_eq(0, rc, "svc_list rc: %d", rc);
	cr_assert_neq(0, list->count, "0 svc descriptors were copied over");

	printf("Found %d services\n", list->count);
	for (i = 0; i < list->count; i++) {
		rc = cxil_get_svc_lpr(dev, list->descs[i].svc_id);
		cr_assert_geq(rc, 1, "cxil_get_svc_lpr(): Failed with rc: %d. Expected Success!",
			     rc);

		printf("svc:%d LNIs/RGID:%d\n", i, rc);

		if (!list->descs[i].is_system_svc)
			continue;

		res_pte += list->descs[i].limits.ptes.res;
		res_txq += list->descs[i].limits.txqs.res;
		res_tgq += list->descs[i].limits.tgqs.res;
		res_eq += list->descs[i].limits.eqs.res;
		res_ct += list->descs[i].limits.cts.res;
		res_ac += list->descs[i].limits.acs.res;
		res_tle += list->descs[i].limits.tles.res;
		res_le += list->descs[i].limits.les.res;
	}

	printf("PTEs total: %d system: %d remainder: %d\n", info->num_ptes,
			res_pte, info->num_ptes - res_pte);
	printf("TXQs total: %d system: %d remainder: %d\n", info->num_txqs,
			res_txq, info->num_txqs - res_txq);
	printf("TGQs total: %d system: %d remainder: %d\n", info->num_tgqs,
			res_tgq, info->num_tgqs - res_tgq);
	printf("EQs total: %d system: %d remainder: %d\n", info->num_eqs,
			res_eq, info->num_eqs - res_eq);
	printf("CTs total: %d system: %d remainder: %d\n", info->num_cts,
			res_ct, info->num_cts - res_ct);
	printf("ACs total: %d system: %d remainder: %d\n", info->num_acs,
			res_ac, info->num_acs - res_ac);
	printf("TLEs total: %d system: %d remainder: %d\n", info->num_tles,
			res_tle, info->num_tles - res_tle);
	printf("LEs total: %d system: %d remainder: %d\n", info->num_les,
			res_le, info->num_les - res_le);

	cxil_free_svc_list(list);
}
