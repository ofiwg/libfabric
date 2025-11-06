/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause */
/* Copyright 2018,2024 Hewlett Packard Enterprise Development LP */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <linux/mman.h>
#include <time.h>

#include "libcxi_test_common.h"

int svc_id2;
struct cxil_lni *lni2;

/* Allocate a service with no restrictions - use for lni2 */
void lni2_setup(void)
{
	int ret;
	struct cxi_svc_desc svc_desc = {
		.num_vld_vnis = 1,
		.vnis[0] = 8,
	};

	svc_id2 = cxil_alloc_svc(dev, &svc_desc, NULL);
	cr_assert_gt(svc_id2, 0, "cxil_alloc_svc(): Failed. ret:%d", svc_id2);

	svc_desc.svc_id = svc_id2;

	ret = cxil_alloc_lni(dev, &lni2, svc_id2);
	cr_assert_eq(ret, 0, "ret = (%d) %s", ret, strerror(-ret));
	cr_assert_neq(lni2, NULL);
}

void lni2_teardown(void)
{
	int ret;

	if (!lni2)
		return;

	ret = cxil_destroy_lni(lni2);
	cr_expect_eq(ret, 0, "%s: cxil_destroy_lni() returns (%d) %s",
		     __func__, ret, strerror(-ret));
	lni2 = NULL;

	ret = cxil_destroy_svc(dev, svc_id2);
	cr_assert_eq(ret, 0, "ret:%d", ret);
}

TestSuite(dev);

struct dev_open_params {
	int dev_id;
	int rc;
};

ParameterizedTestParameters(dev, dev_open)
{
	size_t param_sz;

	static struct dev_open_params params[] = {
		{.dev_id = 0,
		 .rc = 0 },
		{.dev_id = 10,
		 .rc = -ENOENT },
	};

	param_sz = ARRAY_SIZE(params);
	return cr_make_param_array(struct dev_open_params, params,
				   param_sz);
}

ParameterizedTest(struct dev_open_params *param, dev, dev_open)
{
	int rc = cxil_open_device(param->dev_id, &dev);
	cr_assert_eq(rc, param->rc,
		     "rc mismatch, expected: %d received %d",
		     param->rc, rc);

	if (!rc)
		cxil_close_device(dev);
}

Test(dev, dev_null)
{
	int rc = cxil_open_device(0, NULL);
	cr_assert_eq(rc, -EINVAL);

	cxil_close_device(NULL);
}

TestSuite(lni, .init = dev_setup, .fini = dev_teardown);

struct lni_alloc_params {
	int rc;
};

ParameterizedTestParameters(lni, lni_alloc)
{
	size_t param_sz;

	static struct lni_alloc_params params[] = {
		{ .rc = 0 },
	};

	param_sz = ARRAY_SIZE(params);
	return cr_make_param_array(struct lni_alloc_params, params,
				   param_sz);
}

ParameterizedTest(struct lni_alloc_params *param, lni, lni_alloc)
{
	/* TODO Additional testing for svc_id field */
	int rc = cxil_alloc_lni(dev, &lni, CXI_DEFAULT_SVC_ID);

	cr_assert_eq(rc, param->rc, "rc mismatch, expected: %d received %d",
		     param->rc, rc);

	if (!rc)
		cxil_destroy_lni(lni);
}

Test(lni, lni_null)
{
	int rc = cxil_alloc_lni(NULL, &lni, CXI_DEFAULT_SVC_ID);
	cr_assert_eq(rc, -EINVAL);

	rc = cxil_alloc_lni(dev, NULL, CXI_DEFAULT_SVC_ID);
	cr_assert_eq(rc, -EINVAL);

	rc = cxil_alloc_lni(dev, &lni, 0);
	cr_assert_eq(rc, -EINVAL);

	cxil_destroy_lni(NULL);
}

TestSuite(domain, .init = lni_setup, .fini = lni_teardown);

struct domain_alloc_params {
	unsigned int vni;
	unsigned int pid;
	int rc;
};

ParameterizedTestParameters(domain, domain_alloc)
{
	size_t param_sz;

	static struct domain_alloc_params params[] = {
		{.vni = 1,
		 .pid = 0,
		 .rc = 0 },
	};

	param_sz = ARRAY_SIZE(params);
	return cr_make_param_array(struct domain_alloc_params, params,
				   param_sz);
}

ParameterizedTest(struct domain_alloc_params *param, domain, domain_alloc)
{
	int rc = cxil_alloc_domain(lni, param->vni, param->pid, &domain);
	cr_assert_eq(rc, param->rc,
		     "rc mismatch, expected: %d received %d",
		     param->rc, rc);

	if (!rc)
		cxil_destroy_domain(domain);
}

Test(domain, domain_null)
{
	int rc = cxil_alloc_domain(NULL, 0, 0, &domain);
	cr_assert_eq(rc, -EINVAL);

	rc = cxil_alloc_domain(lni, 0, 0, NULL);
	cr_assert_eq(rc, -EINVAL);

	cxil_destroy_domain(NULL);
}

Test(domain, reserve)
{
	int rc;
	int i;
	int max_doms = 511;
	struct cxil_domain *domains[max_doms];
	struct cxil_lni *lni2;
	int base_pid;

	rc = cxil_alloc_lni(dev, &lni2, CXI_DEFAULT_SVC_ID);

	/* Bad vni */
	rc = cxil_reserve_domain(lni, 10000, 0, 0);
	cr_assert_eq(rc, -EINVAL);

	/* Bad LNI */
	rc = cxil_reserve_domain(NULL, 1, 0, 0);
	cr_assert_eq(rc, -EINVAL);

	/* Bad count */
	rc = cxil_reserve_domain(lni, 1, 0, 0);
	cr_assert_eq(rc, -EINVAL);

	/* reserve 1 PID */
	rc = cxil_reserve_domain(lni, 1, 0, 1);
	cr_assert_eq(rc, 0);

	/* Clean up reservation */
	rc = cxil_destroy_lni(lni);
	cr_assert_eq(rc, 0);

	rc = cxil_alloc_lni(dev, &lni, CXI_DEFAULT_SVC_ID);
	cr_assert_eq(rc, 0);

	/* reserve 1 PID */
	rc = cxil_reserve_domain(lni, 1, 0, 1);
	cr_assert_eq(rc, 0);

	/* Use reserved PID */
	rc = cxil_alloc_domain(lni, 1, 0, &domain);
	cr_assert_eq(rc, 0);

	rc = cxil_destroy_domain(domain);
	cr_assert_eq(rc, 0);

	/* reserve many PIDs */
	rc = cxil_reserve_domain(lni, 1, 0, max_doms);
	cr_assert_eq(rc, 0);

	for (i = 0; i < max_doms; i++) {
		rc = cxil_alloc_domain(lni, 1, i, &domains[i]);
		cr_assert_eq(rc, 0);
	}

	for (i = 0; i < max_doms; i++) {
		rc = cxil_destroy_domain(domains[i]);
		cr_assert_eq(rc, 0);
	}

	/* reserve many PIDs with wildcard */
	rc = cxil_reserve_domain(lni, 1, C_PID_ANY, 10);
	cr_assert_eq(rc, 0);

	/* attempt to reserve reserved PIDs */
	rc = cxil_reserve_domain(lni, 1, rc, 10);
	cr_assert_eq(rc, -EEXIST, "rc is %d\n", rc);

	/* reserve many PIDs with wildcard */
	base_pid = cxil_reserve_domain(lni, 1, C_PID_ANY, 10);
	cr_assert_eq(base_pid, 10, "pid is %d\n", base_pid);

	/* Attempt to allocate a reserved PID */
	rc = cxil_alloc_domain(lni2, 1, base_pid, &domains[0]);
	cr_assert_eq(rc, -EEXIST, "rc is %d\n", rc);

	for (i = 0; i < 10; i++) {
		rc = cxil_alloc_domain(lni, 1, base_pid + i, &domains[i]);
		cr_assert_eq(rc, 0);
	}

	for (i = 0; i < 10; i++) {
		rc = cxil_destroy_domain(domains[i]);
		cr_assert_eq(rc, 0);
	}

	/* reserve 1 non-zero PID */
	rc = cxil_alloc_lni(dev, &lni2, CXI_DEFAULT_SVC_ID);
	cr_assert_eq(rc, 0);

	rc = cxil_reserve_domain(lni2, 1, 100, 1);
	cr_assert_eq(rc, 100);

	/* reserve many PIDs with defined base */
	base_pid = cxil_reserve_domain(lni, 1, 101, 10);
	cr_assert_eq(base_pid, 101, "pid is %d\n", base_pid);

	/* Attempt to allocate a reserved PID */
	rc = cxil_alloc_domain(lni2, 1, base_pid, &domains[0]);
	cr_assert_eq(rc, -EEXIST, "rc is %d\n", rc);

	for (i = 0; i < 10; i++) {
		rc = cxil_alloc_domain(lni, 1, base_pid + i, &domains[i]);
		cr_assert_eq(rc, 0);
	}

	for (i = 0; i < 10; i++) {
		rc = cxil_destroy_domain(domains[i]);
		cr_assert_eq(rc, 0);
	}

	cxil_destroy_lni(lni2);
}

TestSuite(cps, .init = lni_setup, .fini = lni_teardown);

struct alloc_cps_params {
	unsigned int vni;
	enum cxi_traffic_class tc;
	enum cxi_traffic_class_type tc_type;
	unsigned int count;
	int rc;
	bool null_lni;
	bool null_cp;
};

ParameterizedTestParameters(cps, alloc_cps)
{
	size_t param_sz;

	static struct alloc_cps_params params[] = {
		/* Good test */
		{
			.vni = 1,
			.tc = CXI_TC_BEST_EFFORT,
			.tc_type = CXI_TC_TYPE_DEFAULT,
			.count = 1,
			.rc = 0,
			.null_lni = false,
			.null_cp = false,
		},
		/* Good test */
		{
			.vni = 1,
			.tc = CXI_TC_DEDICATED_ACCESS,
			.tc_type = CXI_TC_TYPE_DEFAULT,
			.count = 1,
			.rc = 0,
			.null_lni = false,
			.null_cp = false,
		},
		/* Good test */
		{
			.vni = 1,
			.tc = CXI_TC_BULK_DATA,
			.tc_type = CXI_TC_TYPE_DEFAULT,
			.count = 1,
			.rc = 0,
			.null_lni = false,
			.null_cp = false,
		},
		/* Good test */
		{
			.vni = 1,
			.tc = CXI_TC_LOW_LATENCY,
			.tc_type = CXI_TC_TYPE_DEFAULT,
			.count = 1,
			.rc = -EINVAL,
			.null_lni = false,
			.null_cp = false,
		},
		/* Good VNI */
		{
			.vni = 1,
			.tc = CXI_TC_BEST_EFFORT,
			.tc_type = CXI_TC_TYPE_DEFAULT,
			.count = 1,
			.rc = 0,
			.null_lni = false,
			.null_cp = false,
		},

		/* Good VNI */
		{
			.vni = 10,
			.tc = CXI_TC_BEST_EFFORT,
			.tc_type = CXI_TC_TYPE_DEFAULT,
			.count = 1,
			.rc = 0,
			.null_lni = false,
			.null_cp = false,
		},

		/* Bad VNI */
		{
			.vni = 1 << 16,
			.tc = CXI_TC_BEST_EFFORT,
			.tc_type = CXI_TC_TYPE_DEFAULT,
			.count = 1,
			.rc = -EINVAL,
			.null_lni = false,
			.null_cp = false,
		},

		/* Bad TC */
		{
			.vni = 1,
			.tc = CXI_ETH_TC2 + 1,
			.tc_type = CXI_TC_TYPE_DEFAULT,
			.count = 1,
			.rc = -EINVAL,
			.null_lni = false,
			.null_cp = false,
		},

		/* Missing LNI */
		{
			.vni = 1,
			.tc = CXI_TC_BEST_EFFORT,
			.tc_type = CXI_TC_TYPE_DEFAULT,
			.count = 1,
			.rc = -EINVAL,
			.null_lni = true,
			.null_cp = false,
		},

		/* Missing CP */
		{
			.vni = 1,
			.tc = CXI_TC_BEST_EFFORT,
			.tc_type = CXI_TC_TYPE_DEFAULT,
			.count = 1,
			.rc = -EINVAL,
			.null_lni = false,
			.null_cp = true,
		},
		/* Bad TC type */
		{
			.vni = 1,
			.tc = CXI_TC_BEST_EFFORT,
			.tc_type = CXI_TC_TYPE_MAX,
			.count = 1,
			.rc = -EINVAL,
			.null_lni = false,
			.null_cp = false,
		},

	};

	param_sz = ARRAY_SIZE(params);
	return cr_make_param_array(struct alloc_cps_params, params,
				   param_sz);
}

ParameterizedTest(struct alloc_cps_params *param, cps, alloc_cps)
{
	int rc;
	struct cxi_cp **cps = NULL;
	struct cxi_cp **cp;
	int i;
	int cp_cnt = 0;
	struct cxil_lni *cp_lni;

	if (param->null_lni)
		cp_lni = NULL;
	else
		cp_lni = lni;

	if (param->count > 0) {
		cps = calloc(param->count, sizeof(*cps));
		cr_assert_neq(cps, NULL, "Failed to allocated memory");
	}

	for (i = 0; i < param->count; i++) {
		if (param->null_cp)
			cp = NULL;
		else
			cp = &cps[i];

		rc = cxil_alloc_cp(cp_lni, param->vni + i, param->tc,
				   param->tc_type, cp);

		if ((i + 1) == param->count) {
			cr_assert_eq(rc, param->rc, "cxil_alloc_cp(), expected: %d received %d",
				     param->rc, rc);
		} else {
			cr_assert_eq(rc, 0, "cxil_alloc_cp(), expected: %d received %d",
				     0, rc);
		}

		if (!rc)
			cp_cnt++;
	}

	for (cp_cnt--; cp_cnt >= 0; cp_cnt--) {
		rc = cxil_destroy_cp(cps[cp_cnt]);
		cr_assert_eq(rc, 0, "cxil_destroy_cp(), expected: %d received %d",
			     0, rc);
	}

	if (param->count > 0)
		free(cps);
}

TestSuite(cmdq, .init = lni_setup, .fini = lni_teardown);

struct cmdq_alloc_params {
	struct cxi_cq_alloc_opts opts;
	bool alloc_cp;
	int rc;
};

ParameterizedTestParameters(cmdq, cmdq_alloc)
{
	size_t param_sz;

	static struct cmdq_alloc_params params[] = {
		{.opts.count = 0, /* TODO Should size 0 work? */
		 .alloc_cp = true,
		 .opts.flags = CXI_CQ_IS_TX,
		 .rc = 0 },
		{.opts.count = 1024,
		 .alloc_cp = true,
		 .opts.flags = CXI_CQ_IS_TX,
		 .rc = 0 },
		{.opts.count = 0,
		 .alloc_cp = true,
		 .opts.flags = 0,
		 .rc = 0 },
		{.opts.count = 1024,
		 .alloc_cp = true,
		 .opts.flags = 0,
		 .rc = 0 },
		{.opts.count = 1024,
		 .alloc_cp = true,
		 .opts.flags = 0,
		 .rc = 0 },
		{.opts.count = 1024,
		 .alloc_cp = false,  /* cps required for transmit CMDQ */
		 .opts.flags = CXI_CQ_IS_TX,
		 .rc = -EINVAL },

		/* Bad policy value. */
		{.opts.count = 1024,
		 .alloc_cp = true,
		 .opts.policy = -1,
		 .rc = -EINVAL },

		/* Another bad policy value. */
		{.opts.count = 1024,
		 .alloc_cp = true,
		 .opts.policy = CXI_CQ_UPDATE_LOW_FREQ + 1,
		 .rc = -EINVAL },
	};

	param_sz = ARRAY_SIZE(params);
	return cr_make_param_array(struct cmdq_alloc_params, params,
				   param_sz);
}

ParameterizedTest(struct cmdq_alloc_params *param, cmdq, cmdq_alloc)
{
	int rc;
	struct cxi_cp *cp;
	unsigned int ack_counter;


	if ((param->opts.flags & CXI_CQ_IS_TX) && param->alloc_cp) {
		rc = cxil_alloc_cp(lni, vni, CXI_TC_BEST_EFFORT,
				   CXI_TC_TYPE_DEFAULT, &cp);
		cr_assert_eq(rc, 0, "Failed to allocate comm profile: %d\n",
			     rc);

		param->opts.lcid = cp->lcid;
	}

	rc = cxil_alloc_cmdq(lni, NULL, &param->opts, &transmit_cmdq);
	cr_assert_eq(rc, param->rc, "rc mismatch, expected: %d received %d",
		     param->rc, rc);
	if (!rc) {
		rc = cxil_cmdq_ack_counter(transmit_cmdq, &ack_counter);
		cr_assert_eq(rc, 0, "Failed to get CQ ack counter: %d\n", rc);

		rc = cxil_destroy_cmdq(transmit_cmdq);
		cr_assert_eq(rc, 0, "Failed to destroy CQ: %d\n", rc);
	}

	if ((param->opts.flags & CXI_CQ_IS_TX) && param->alloc_cp) {
		rc = cxil_destroy_cp(cp);
		cr_assert_eq(rc, 0, "Failed to destroy comm profile: %d\n", rc);
	}
}

Test(cmdq, cmdq_null)
{
	struct cxi_cq_alloc_opts opts = {
		.count = 1024,
		.flags = CXI_CQ_IS_TX,
	};
	int rc = cxil_alloc_cmdq(NULL, NULL, &opts, &transmit_cmdq);
	cr_assert_eq(rc, -EINVAL);

	rc = cxil_alloc_cmdq(lni, NULL, &opts, NULL);
	cr_assert_eq(rc, -EINVAL);

	cxil_destroy_cmdq(NULL);
}

Test(devicelist, list_valid)
{
	int rc;
	struct cxil_device_list *dev_list = NULL;

	rc = cxil_get_device_list(&dev_list);
	cr_assert_eq(rc, 0, "rc is %d", rc);
	cr_assert_neq(dev_list, NULL);
	cr_assert_gt(dev_list->count, 0);
	cxil_free_device_list(dev_list);
}

TestSuite(md, .init = test_data_setup, .fini = test_data_teardown);

Test(md, map_md_null)
{
	int rc;

	rc = cxil_map(NULL, NULL, 0, 0, NULL, NULL);
	cr_assert_eq(rc, -EINVAL);

	rc = cxil_map(lni, NULL, 0, 0, NULL, NULL);
	cr_assert_eq(rc, -EINVAL);

	rc = cxil_map(NULL, test_data[0].addr, 0, 0, NULL, NULL);
	cr_assert_eq(rc, -EINVAL);

	rc = cxil_map(NULL, NULL, 0, 0, NULL, &test_data[0].md);
	cr_assert_eq(rc, -EINVAL);

	rc = cxil_map(lni, test_data[0].addr, 0, 0, NULL, NULL);
	cr_assert_eq(rc, -EINVAL);

	rc = cxil_map(NULL, test_data[0].addr, 0, 0, NULL, &test_data[0].md);
	cr_assert_eq(rc, -EINVAL);

	rc = cxil_map(lni, NULL, 0, 0, NULL, &test_data[0].md);
	cr_assert_eq(rc, -EINVAL);
}

Test(md, unmap_md_null)
{
	int rc;

	rc = cxil_unmap(NULL);
	cr_assert_eq(rc, -EINVAL);
}

Test(md, map_md)
{
	int rc;
	int i;

	for (i = 0; i < test_data_len; i++) {
		rc = cxil_map(lni, test_data[i].addr, test_data[i].len,
			      CXI_MAP_WRITE | CXI_MAP_READ | CXI_MAP_PIN,
			      NULL, &test_data[i].md);
		cr_assert_eq(rc, 0, "rc = %d", rc);
		cr_assert_neq(test_data[i].md->iova, 0);
	}

	for (i = 0; i < test_data_len; i++) {
		rc = cxil_unmap(test_data[i].md);
		cr_assert_eq(rc, 0);
	}
}

TestSuite(le_invalidate, .init = data_xfer_setup,
	  .fini = data_xfer_teardown);

struct le_invalidate_params {
	bool null_pte;
	int rc;
	int test_num;
};

ParameterizedTestParameters(le_invalidate, le_invalidate_params)
{
	size_t param_sz;

	static struct le_invalidate_params params[] = {
		/* NUll PTE. */
		{
			.null_pte = true,
			.rc = -EINVAL,
			.test_num = 1,
		},

		/* Successful invalidate. Note no LEs are append so no
		 * C_RC_MST_CANCELLED errors will occur.
		 */
		{
			.null_pte = false,
			.rc = 0,
			.test_num = 2,
		},
	};

	param_sz = ARRAY_SIZE(params);
	return cr_make_param_array(struct le_invalidate_params, params,
				   param_sz);
}

ParameterizedTest(struct le_invalidate_params *p, le_invalidate,
		  le_invalidate_params)
{
	int rc;
	struct cxil_pte *pte;

	ptlte_setup(0, true, false);

	if (p->null_pte)
		pte = NULL;
	else
		pte = rx_pte;

	rc = cxil_invalidate_pte_le(pte, 0, C_PTL_LIST_PRIORITY);

	cr_assert_eq(rc, p->rc, "rc mismatch(%d) expected: %d received %d",
		     p->test_num, p->rc, rc);

	ptlte_teardown();
}

TestSuite(evtq, .init = lni_setup, .fini = lni_teardown);

struct evtq_alloc_params {
	size_t queue_len_add;
	int rc;
	int test_num;
};

ParameterizedTestParameters(evtq, evtq_alloc)
{
	size_t param_sz;

	static struct evtq_alloc_params params[] = {
		/* Unaligned size */
		{.queue_len_add = 1,
		 .rc = -EINVAL,
		 .test_num = 0},
		/* Expected to work */
		{.queue_len_add = 0,
		 .rc = 0,
		 .test_num = 1},
	};

	param_sz = ARRAY_SIZE(params);
	return cr_make_param_array(struct evtq_alloc_params, params,
				   param_sz);
}

ParameterizedTest(struct evtq_alloc_params *p, evtq, evtq_alloc)
{
	int rc;
	void *eq_buf = NULL;
	struct cxi_md *eq_buf_md;
	struct cxi_eq_attr attr = {};
	size_t queue_len = s_page_size + p->queue_len_add;

	rc = posix_memalign(&eq_buf, s_page_size, queue_len);
	cr_assert(rc == 0);
	memset(eq_buf, 0, queue_len);

	rc = cxil_map(lni, eq_buf, queue_len,
		      CXI_MAP_PIN | CXI_MAP_READ | CXI_MAP_WRITE,
		      NULL, &eq_buf_md);
	cr_assert(!rc);

	memset(&transmit_evtq, 0, sizeof(transmit_evtq));

	attr.queue = eq_buf;
	attr.queue_len = queue_len;

	rc = cxil_alloc_evtq(lni, eq_buf_md, &attr, NULL, NULL, &transmit_evtq);
	cr_assert_eq(rc, p->rc, "rc mismatch(%d) expected: %d received %d",
		     p->test_num, p->rc, rc);

	if (!rc) {
		/* Check evtq */
		cr_expect_neq(transmit_evtq->sw_state_addr, 0);
		cr_expect_geq(transmit_evtq->byte_size, queue_len);
		cr_expect_eq(transmit_evtq->rd_offset, C_EE_CFG_ECB_SIZE);
		cr_expect_eq(transmit_evtq->prev_rd_offset, 64);

		rc = cxil_destroy_evtq(transmit_evtq);
		cr_assert_eq(rc, 0, "Destroy evtq failed. %d", rc);
	}

	cxil_unmap(eq_buf_md);
	free(eq_buf);
}

Test(evtq, evtq_null_params)
{
	int rc;
	void *eq_buf;
	size_t eq_buf_len = s_page_size;
	struct cxi_md *eq_buf_md;
	struct cxi_eq *evtq1;
	struct cxi_eq *evtq2;
	struct cxi_eq_attr attr = {};

	eq_buf = aligned_alloc(s_page_size, eq_buf_len);
	cr_assert(eq_buf);
	memset(eq_buf, 0, eq_buf_len);

	rc = cxil_map(lni, eq_buf, eq_buf_len,
		      CXI_MAP_PIN | CXI_MAP_READ | CXI_MAP_WRITE,
		      NULL, &eq_buf_md);
	cr_assert(!rc);

	attr.queue = eq_buf;
	attr.queue_len = eq_buf_len;

	/* NULL alloc LNI */
	rc = cxil_alloc_evtq(NULL, eq_buf_md, &attr,
			     NULL, NULL, &transmit_evtq);
	cr_assert_eq(rc, -EINVAL);

	/* NULL alloc attr */
	rc = cxil_alloc_evtq(lni, eq_buf_md, NULL, NULL, NULL, &transmit_evtq);
	cr_assert_eq(rc, -EINVAL);

	/* NULL alloc EQ */
	rc = cxil_alloc_evtq(lni, eq_buf_md, &attr, NULL, NULL, NULL);
	cr_assert_eq(rc, -EINVAL);

	/* NULL destroy EQ */
	cr_assert_eq(cxil_destroy_evtq(NULL), -EINVAL);

	/* Allocate multiple EQs */
	rc = cxil_alloc_evtq(lni, eq_buf_md, &attr, NULL, NULL, &evtq1);
	cr_assert_eq(rc, 0, "Alloc evtq1 expected: %d received %d", 0, rc);

	rc = cxil_alloc_evtq(lni, eq_buf_md, &attr, NULL, NULL, &evtq2);
	cr_assert_eq(rc, 0, "Alloc evtq2 expected: %d received %d", 0, rc);

	rc = cxil_destroy_evtq(evtq2);
	cr_assert_eq(rc, 0, "Destroy evtq2 expected %d received %d", 0, rc);

	rc = cxil_destroy_evtq(evtq1);
	cr_assert_eq(rc, 0, "Destroy evtq1 expected %d received %d", 0, rc);

	cxil_unmap(eq_buf_md);
	free(eq_buf);
}

Test(evtq, evtq_passthrough)
{
	int rc;
	void *eq_buf;
	size_t eq_buf_len = TWO_MB;
	struct cxi_eq_attr attr = {};
	int prot = PROT_WRITE | PROT_READ;
	int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB |
			 MAP_HUGE_1GB;
	int hp_cnt;

	eq_buf = aligned_alloc(s_page_size, eq_buf_len);
	cr_assert(eq_buf);
	memset(eq_buf, 0, eq_buf_len);

	attr.queue = eq_buf;
	attr.queue_len = eq_buf_len;
	attr.flags = CXI_EQ_PASSTHROUGH;

	/* Discontig buffer with passthrough */
	rc = cxil_alloc_evtq(lni, NULL, &attr, NULL, NULL, &transmit_evtq);
	cr_assert_eq(rc, -EINVAL,
		     "EQ Alloc w/discontig PT buffer succeeded, expected %d received %d",
		     -EINVAL, rc);

	free(eq_buf);

	hp_cnt = huge_pg_setup(ONE_GB, 1);
	if (hp_cnt < 0)
		cr_skip_test("1 GB hugepage not available");

	check_huge_pg_free(ONE_GB, 1);

	eq_buf = mmap(NULL, eq_buf_len, prot, mmap_flags, 0, 0);
	cr_assert_not_null(eq_buf, "hugepage mmap() failed.\n");
	cr_assert_neq(eq_buf, MAP_FAILED,
		      "mmap failed. ret:%p errno:%d", eq_buf,
		      errno);

	attr.queue = eq_buf;
	attr.queue_len = eq_buf_len;
	attr.flags = CXI_EQ_PASSTHROUGH;

	rc = cxil_alloc_evtq(lni, NULL, &attr, NULL, NULL, &transmit_evtq);
	cr_assert_eq(rc, 0,
		     "EQ Alloc w/ contig PT buffer failed, expected %d received %d",
		     0, rc);

	rc = cxil_destroy_evtq(transmit_evtq);
	cr_assert_eq(rc, 0);

	munmap(eq_buf, eq_buf_len);

	huge_pg_setup(ONE_GB, hp_cnt);
}

/* Test basic use of the resize interface. Skip reading events. */
Test(evtq, resize)
{
	void *eq_buf;
	size_t eq_buf_len = s_page_size * 2;
	struct cxi_md *eq_buf_md;
	struct cxi_eq_attr attr = {};
	int rc;

	memset(&transmit_evtq, 0, sizeof(transmit_evtq));

	eq_buf = aligned_alloc(s_page_size, eq_buf_len);
	cr_assert(eq_buf);
	memset(eq_buf, 0, eq_buf_len);

	rc = cxil_map(lni, eq_buf, eq_buf_len,
		      CXI_MAP_PIN | CXI_MAP_READ | CXI_MAP_WRITE,
		      NULL, &eq_buf_md);
	cr_assert(!rc);

	attr.queue = eq_buf;
	attr.queue_len = eq_buf_len / 2;

	rc = cxil_alloc_evtq(lni, eq_buf_md, &attr, NULL, NULL, &transmit_evtq);
	cr_assert(!rc);

	cr_expect_neq(transmit_evtq->sw_state_addr, 0);
	cr_expect_geq(transmit_evtq->byte_size, eq_buf_len / 2);
	cr_expect_eq(transmit_evtq->rd_offset, C_EE_CFG_ECB_SIZE);
	cr_expect_eq(transmit_evtq->prev_rd_offset, 64);

	rc = cxil_evtq_resize(transmit_evtq, eq_buf, eq_buf_len, eq_buf_md);
	cr_assert(!rc, "cxil_evtq_resize() failed");

	rc = cxil_evtq_resize_complete(transmit_evtq);
	cr_assert(!rc, "cxil_evtq_resize_complete() failed");

	/* Expect EQ pointers to be reset */
	cr_expect_geq(transmit_evtq->byte_size, eq_buf_len);
	cr_expect_eq(transmit_evtq->rd_offset, C_EE_CFG_ECB_SIZE);
	cr_expect_eq(transmit_evtq->prev_rd_offset, 64);

	rc = cxil_destroy_evtq(transmit_evtq);
	cr_assert_eq(rc, 0, "Destroy evtq failed. %d", rc);

	rc = cxil_unmap(eq_buf_md);
	cr_assert_eq(rc, 0);

	free(eq_buf);
}

TestSuite(evtq_poll, .init = data_xfer_setup, .fini = data_xfer_teardown);

Test(evtq_poll, resize_poll, .disabled = true)
{
	int rc;
	int pid_idx = 0;
	int xfer_len = 64;
	struct mem_window snd_mem;
	struct mem_window rcv_mem;
	void *eq_buf = NULL;
	size_t eq_buf_len = s_page_size;
	struct cxi_md *eq_buf_md;

	/* Allocate buffers */
	alloc_iobuf(1024, &snd_mem, CXI_MAP_WRITE);
	alloc_iobuf(1024, &rcv_mem, CXI_MAP_READ);

	/* Initialize Send Memory */
	for (int i = 0; i < snd_mem.length; i++)
		snd_mem.buffer[i] = i;
	memset(rcv_mem.buffer, 0, rcv_mem.length);

	/* Initialize recv PtlTE */
	ptlte_setup(pid_idx, true, false);

	/* Post receive buffer */
	append_le_sync(rx_pte, &rcv_mem, C_PTL_LIST_PRIORITY,
		       0, 0, 0, CXI_MATCH_ID_ANY, 0,
		       false, false, true, false, true, true, false,
		       NULL);

	/* Initiate Put Operation */
	do_put_sync(snd_mem, xfer_len, 0, 0, pid_idx, false, 0, 0, 0, false);

	/* Gather Unlink event from recv buffer */
	process_eqe(target_evtq, EQE_TGT_LONG, C_EVENT_UNLINK,
		    0, NULL);

	/* Gather PUT event from recv buffer */
	process_eqe(target_evtq, EQE_TGT_SHORT, C_EVENT_PUT,
		    0, NULL);

	eq_buf = aligned_alloc(s_page_size, eq_buf_len);
	cr_assert(eq_buf);
	memset(eq_buf, 0, eq_buf_len);

	rc = cxil_map(lni, eq_buf, eq_buf_len,
		      CXI_MAP_PIN | CXI_MAP_READ | CXI_MAP_WRITE,
		      NULL, &eq_buf_md);
	cr_assert(!rc);

	rc = cxil_evtq_resize(transmit_evtq, eq_buf, eq_buf_len, eq_buf_md);
	cr_assert(!rc);

	/* Gather EQ_SWITCH event from transmit EQ */
	process_eqe(transmit_evtq, EQE_EQ_SWITCH, C_EVENT_EQ_SWITCH,
		    0, NULL);

	rc = cxil_evtq_resize_complete(transmit_evtq);
	cr_assert(!rc);

	/* Perform a second op and switch back to the original buffer. */

	/* Post receive buffer */
	append_le_sync(rx_pte, &rcv_mem, C_PTL_LIST_PRIORITY,
		       0, 0, 0, CXI_MATCH_ID_ANY, 0,
		       false, false, true, false, true, true, false,
		       NULL);

	/* Initiate Put Operation */
	do_put_sync(snd_mem, xfer_len, 0, 0, pid_idx, false, 0, 0, 0, false);

	/* Gather Unlink event from recv buffer */
	process_eqe(target_evtq, EQE_TGT_LONG, C_EVENT_UNLINK,
		    0, NULL);

	/* Gather PUT event from recv buffer */
	process_eqe(target_evtq, EQE_TGT_SHORT, C_EVENT_PUT,
		    0, NULL);

	rc = cxil_evtq_resize(transmit_evtq, transmit_eq_buf,
			      transmit_eq_buf_len, transmit_eq_md);
	cr_assert(!rc);

	/* Gather EQ_SWITCH event from transmit EQ */
	process_eqe(transmit_evtq, EQE_EQ_SWITCH, C_EVENT_EQ_SWITCH,
		    0, NULL);

	rc = cxil_evtq_resize_complete(transmit_evtq);
	cr_assert(!rc);

	rc = cxil_unmap(eq_buf_md);
	cr_assert_eq(rc, 0);

	free(eq_buf);

	/* Free receive queue PtlTE */
	ptlte_teardown();

	/* Deallocate buffers */
	free_iobuf(&snd_mem);
	free_iobuf(&rcv_mem);
}

TestSuite(pte, .init = data_xfer_setup, .fini = data_xfer_teardown);

Test(pte, pte_status)
{
	int rc;
	struct cxi_pte_status status = {};
	struct mem_window snd_mem;
	struct mem_window rcv_mem;
	__u64 *ule_offs;
	__u64 tgt_iova;

	ptlte_setup(0, true, false);

	/* No PTE */
	rc = cxil_pte_status(NULL, &status);
	cr_assert_eq(rc, -EINVAL);

	/* No struct to store results */
	rc = cxil_pte_status(rx_pte, NULL);
	cr_assert_eq(rc, -EINVAL);

	/* Expect Success */
	rc = cxil_pte_status(rx_pte, &status);
	cr_assert_eq(rc, 0, "rc mismatch, expected: %d received %d",
		     0, rc);

	/* Do a short Put that matches in overflow list */
	alloc_iobuf(1024, &snd_mem, CXI_MAP_WRITE|CXI_MAP_READ);
	alloc_iobuf(1024, &rcv_mem, CXI_MAP_WRITE|CXI_MAP_READ);
	append_le_sync(rx_pte, &rcv_mem, C_PTL_LIST_OVERFLOW,
		       0, 0, 0, CXI_MATCH_ID_ANY, 0,
		       false, false, true, false, true, true, false,
		       NULL);
	do_put_sync(snd_mem, 8, 0, 0, 0, false, 0, 0, 0, false);

	ule_offs = malloc(10 * sizeof(*ule_offs));
	assert(ule_offs);

	status.ule_offsets = ule_offs;
	status.ule_count = 0;

	rc = cxil_pte_status(rx_pte, &status);
	cr_assert_eq(rc, 0, "rc mismatch, expected: %d received %d",
		     0, rc);
	cr_assert_eq(status.ule_count, 1, "ule_count: %d (exp: %d)",
		     status.ule_count, 1);

	status.ule_offsets = ule_offs;
	status.ule_count = 10;

	rc = cxil_pte_status(rx_pte, &status);
	cr_assert_eq(rc, 0, "rc mismatch, expected: %d received %d",
		     0, rc);

	tgt_iova = CXI_VA_TO_IOVA(rcv_mem.md, rcv_mem.buffer);
	cr_assert_eq(tgt_iova, ule_offs[0],
		     "start mismatch: %#llx (exp: %#llx)",
		     ule_offs[0], tgt_iova);
	printf("ule_count: %d offset[0]: %#llx\n",
	       status.ule_count, ule_offs[0]);

	free(ule_offs);

	ptlte_teardown();

	free_iobuf(&snd_mem);
	free_iobuf(&rcv_mem);
}

TestSuite(cntr, .init = dev_setup, .fini = dev_teardown);

/* buf needs to store 30 characters */
int timespec_fmt(char *buf, uint len, struct timespec *ts)
{
	int ret;
	struct tm t;

	tzset();
	if (localtime_r(&(ts->tv_sec), &t) == NULL)
		return 1;

	ret = strftime(buf, len, "%F %T", &t);
	if (ret == 0)
		return 2;
	len -= ret - 1;

	ret = snprintf(&buf[strlen(buf)], len, ".%09ld", ts->tv_nsec);
	if (ret >= len)
		return 3;

	return 0;
}

Test(cntr, read)
{
	int rc;
	uint64_t val;
	static uint64_t data[C1_CNTR_SIZE]; /* static -> off the stack */
	struct timespec ts;
	struct timespec ts2;
	char buf[30];
	int i;
	enum c_cntr_type cntrs[10];

	val = ~0ULL;
	rc = cxil_read_cntr(NULL, 0, &val, NULL);
	cr_expect_eq(rc, -EINVAL);
	cr_expect_eq(val, ~0ULL);

	val = ~0ULL;
	rc = cxil_read_cntr(dev, 0, NULL, NULL);
	cr_expect_eq(rc, -EINVAL);
	cr_expect_eq(val, ~0ULL);

	val = ~0ULL;
	rc = cxil_read_cntr(dev, 112, &val, NULL);
	cr_expect_eq(rc, 0);
	cr_expect_neq(val, ~0ULL);

	val = ~0ULL;
	rc = cxil_read_cntr(dev, C1_CNTR_SIZE, &val, NULL);
	cr_expect_eq(rc, -EINVAL);
	cr_expect_eq(val, ~0ULL);

	val = ~0ULL;
	rc = cxil_read_cntr(dev, C_CNTR_LPE_SUCCESS_CNTR, &val, NULL);
	cr_expect_eq(rc, 0);
	cr_expect_neq(val, ~0ULL);

	val = ~0ULL;
	rc = cxil_read_cntr(dev, C_CNTR_LPE_SUCCESS_CNTR, &val, &ts);
	cr_expect_eq(rc, 0);
	cr_expect_neq(val, ~0ULL);

	rc = timespec_fmt(buf, 30, &ts);
	cr_expect_eq(rc, 0);

	printf("LPE_SUCCESS_CNTR(%u): %lu ts: %s\n",
	       C_CNTR_LPE_SUCCESS_CNTR, val, buf);

	for (i = 0 ; i < C1_CNTR_SIZE ; ++i)
		data[i] = ~0ULL;

	rc = cxil_read_all_cntrs(dev, data, NULL);
	cr_expect_eq(rc, 0);

	for (i = 0 ; i < C1_CNTR_SIZE ; ++i)
		cr_expect_neq(data[i], ~0ULL);

	for (i = 0 ; i < C1_CNTR_SIZE ; ++i)
		data[i] = ~0ULL;

	rc = cxil_read_all_cntrs(dev, data, &ts);
	cr_expect_eq(rc, 0);

	for (i = 0 ; i < C1_CNTR_SIZE ; ++i)
		cr_expect_neq(data[i], ~0ULL);

	rc = cxil_read_n_cntrs(dev, 0, cntrs, data, NULL);
	cr_expect_eq(rc, -EINVAL);

	rc = cxil_read_n_cntrs(dev, 0, cntrs, data, &ts);
	cr_expect_eq(rc, -EINVAL);

	rc = cxil_read_n_cntrs(dev, C1_CNTR_COUNT + 1, cntrs, data, NULL);
	cr_expect_eq(rc, -EINVAL);

	rc = cxil_read_n_cntrs(dev, C1_CNTR_COUNT + 1, cntrs, data, &ts);
	cr_expect_eq(rc, -EINVAL);

	cntrs[0] = -1;

	rc = cxil_read_n_cntrs(dev, 1, cntrs, data, NULL);
	cr_expect_eq(rc, -EINVAL);

	rc = cxil_read_n_cntrs(dev, 1, cntrs, data, &ts);
	cr_expect_eq(rc, -EINVAL);

	cntrs[0] = C1_CNTR_SIZE;

	rc = cxil_read_n_cntrs(dev, 1, cntrs, data, NULL);
	cr_expect_eq(rc, -EINVAL);

	rc = cxil_read_n_cntrs(dev, 1, cntrs, data, &ts);
	cr_expect_eq(rc, -EINVAL);

	/*
	 * no counter exist at C1_CNTR_PI_IPD_IPD_TRIGGER_EVENTS_1 + 1
	 */
	cntrs[0] = C_CNTR_PI_IPD_IPD_TRIGGER_EVENTS_1 + 1;

	rc = cxil_read_n_cntrs(dev, 1, cntrs, data, NULL);
	cr_expect_eq(rc, -EINVAL);

	rc = cxil_read_n_cntrs(dev, 1, cntrs, data, &ts);
	cr_expect_eq(rc, -EINVAL);

	cntrs[0] = C_CNTR_PI_IPD_IPD_TRIGGER_EVENTS_1 + 1;
	cntrs[1] = C_CNTR_PI_IPD_PRI_RTRGT_TLPS;
	cntrs[2] = C_CNTR_PI_IPD_IPD_TRIGGER_EVENTS_1;
	cntrs[3] = C_CNTR_MB_CRMC_RING_SBE_0;
	cntrs[4] = C_CNTR_ATU_INVAL_STALL_CMPL_WAIT;
	cntrs[5] = C_CNTR_CQ_SUCCESS_TX_CNTR;
	cntrs[6] = C_CNTR_EE_EQ_STATE_UCOR_ERR_CNTR;
	cntrs[7] = C_CNTR_LPE_SUCCESS_CNTR;
	cntrs[8] = C_CNTR_HNI_COR_ECC_ERR_CNTR;
	cntrs[9] = C1_CNTR_OXE_PRF_SET1_OCC_HIST_BIN3;

	rc = cxil_read_n_cntrs(dev, 10, cntrs, data, NULL);
	cr_expect_eq(rc, -EINVAL);

	rc = cxil_read_n_cntrs(dev, 10, cntrs, data, &ts);
	cr_expect_eq(rc, -EINVAL);

	cntrs[0] = C_CNTR_PI_IPD_PRI_RTRGT_TLPS;
	cntrs[1] = C_CNTR_PI_IPD_IPD_TRIGGER_EVENTS_1;
	cntrs[2] = C_CNTR_MB_CRMC_RING_SBE_0;
	cntrs[3] = C_CNTR_ATU_INVAL_STALL_CMPL_WAIT;
	cntrs[4] = C_CNTR_CQ_SUCCESS_TX_CNTR;
	cntrs[5] = C_CNTR_EE_EQ_STATE_UCOR_ERR_CNTR;
	cntrs[6] = C_CNTR_LPE_SUCCESS_CNTR;
	cntrs[7] = C_CNTR_HNI_COR_ECC_ERR_CNTR;
	cntrs[8] = C1_CNTR_OXE_PRF_SET1_OCC_HIST_BIN3;
	cntrs[9] = C_CNTR_PI_IPD_IPD_TRIGGER_EVENTS_1 + 1;

	rc = cxil_read_n_cntrs(dev, 10, cntrs, data, NULL);
	cr_expect_eq(rc, -EINVAL);

	rc = cxil_read_n_cntrs(dev, 10, cntrs, data, &ts);
	cr_expect_eq(rc, -EINVAL);

	cntrs[0] = C_CNTR_PI_IPD_PRI_RTRGT_TLPS;
	cntrs[1] = C_CNTR_PI_IPD_IPD_TRIGGER_EVENTS_1;
	cntrs[2] = C_CNTR_MB_CRMC_RING_SBE_0;
	cntrs[3] = C_CNTR_ATU_INVAL_STALL_CMPL_WAIT;
	cntrs[4] = C_CNTR_PI_IPD_IPD_TRIGGER_EVENTS_1 + 1;
	cntrs[5] = C_CNTR_CQ_SUCCESS_TX_CNTR;
	cntrs[6] = C_CNTR_EE_EQ_STATE_UCOR_ERR_CNTR;
	cntrs[7] = C_CNTR_LPE_SUCCESS_CNTR;
	cntrs[8] = C_CNTR_HNI_COR_ECC_ERR_CNTR;
	cntrs[9] = C1_CNTR_OXE_PRF_SET1_OCC_HIST_BIN3;

	rc = cxil_read_n_cntrs(dev, 10, cntrs, data, NULL);
	cr_expect_eq(rc, -EINVAL);

	rc = cxil_read_n_cntrs(dev, 10, cntrs, data, &ts);
	cr_expect_eq(rc, -EINVAL);

	cntrs[0] = C_CNTR_PI_IPD_PRI_RTRGT_TLPS;
	cntrs[1] = C_CNTR_PI_IPD_IPD_TRIGGER_EVENTS_1;
	cntrs[2] = C_CNTR_MB_CRMC_RING_SBE_0;
	cntrs[3] = C_CNTR_ATU_INVAL_STALL_CMPL_WAIT;
	cntrs[4] = C_CNTR_CQ_SUCCESS_TX_CNTR;
	cntrs[5] = C_CNTR_EE_EQ_STATE_UCOR_ERR_CNTR;
	cntrs[6] = C_CNTR_LPE_SUCCESS_CNTR;
	cntrs[7] = C_CNTR_HNI_COR_ECC_ERR_CNTR;
	cntrs[8] = C1_CNTR_OXE_PRF_SET1_OCC_HIST_BIN3;
	cntrs[9] = C_CNTR_IXE_TC_REQ_ECN_PKTS_3;

	for (i = 0 ; i < C1_CNTR_SIZE ; ++i)
		data[i] = ~0ULL;

	rc = cxil_read_n_cntrs(dev, 10, cntrs, data, NULL);
	cr_expect_eq(rc, 0);

	for (i = 0 ; i < 10 ; ++i)
		cr_expect_neq(data[i], ~0ULL);

	for (i = 10 ; i < C1_CNTR_SIZE ; ++i)
		cr_expect_eq(data[i], ~0ULL);

	for (i = 0 ; i < C1_CNTR_SIZE ; ++i)
		data[i] = ~0ULL;

	rc = cxil_read_n_cntrs(dev, 10, cntrs, data, &ts);
	cr_expect_eq(rc, 0);

	for (i = 0 ; i < 10 ; ++i)
		cr_expect_neq(data[i], ~0ULL);

	for (i = 10 ; i < C1_CNTR_SIZE ; ++i)
		cr_expect_eq(data[i], ~0ULL);

	sleep(5);

	for (i = 0 ; i < C1_CNTR_SIZE ; ++i)
		data[i] = ~0ULL;

	rc = cxil_read_n_cntrs(dev, 10, cntrs, data, &ts2);
	cr_expect_eq(rc, 0);

	for (i = 0 ; i < 10 ; ++i)
		cr_expect_neq(data[i], ~0ULL);

	for (i = 10 ; i < C1_CNTR_SIZE ; ++i)
		cr_expect_eq(data[i], ~0ULL);

	cr_expect_neq(ts.tv_sec, ts2.tv_sec);
	cr_expect_neq(ts.tv_nsec, ts2.tv_nsec);
}

TestSuite(evtq_adjust_reserved_fc, .init = data_xfer_setup,
	  .fini = data_xfer_teardown);

struct evtq_adjust_reserved_fc_params {
	int value;
	int rc;
	int test_num;
};

ParameterizedTestParameters(evtq_adjust_reserved_fc, single_value_adjustments)
{
	size_t param_sz;

	static struct evtq_adjust_reserved_fc_params params[] = {
		/* Invalid value (too high) */
		{
			.value = 1 << 14,
			.rc = -EINVAL,
			.test_num = 0,
		},

		/* Invalid value (too low) */
		{
			.value = -1 * (1 << 14),
			.rc = -EINVAL,
			.test_num = 1,
		},

		/* Valid value but EQ cannot support it based EQ size and/or
		 * current reserved FC value.
		 */
		{
			.value = (1 << 14) - 1,
			.rc = -ENOSPC,
			.test_num = 2,
		},

		/* Valid value. */
		{
			.value = 1,
			.rc = 1,
			.test_num = 3,
		},
	};

	param_sz = ARRAY_SIZE(params);
	return cr_make_param_array(struct evtq_adjust_reserved_fc_params,
				   params, param_sz);
}

ParameterizedTest(struct evtq_adjust_reserved_fc_params *p,
		  evtq_adjust_reserved_fc, single_value_adjustments)
{
	int rc;

	rc = cxil_evtq_adjust_reserved_fc(transmit_evtq, p->value);
	cr_assert_eq(rc, p->rc, "rc mismatch(%d) expected: %d received %d",
		     p->test_num, p->rc, rc);
}

Test(evtq_adjust_reserved_fc, multiple_value_adjustments)
{
	int rc;
	int reserved_fc;

	reserved_fc = 10;
	rc = cxil_evtq_adjust_reserved_fc(transmit_evtq, reserved_fc);
	cr_assert_eq(rc, reserved_fc, "rc mismatch expected: %d received %d",
		     reserved_fc, rc);

	reserved_fc -= 1;
	rc = cxil_evtq_adjust_reserved_fc(transmit_evtq, -1);
	cr_assert_eq(rc, reserved_fc, "rc mismatch expected: %d received %d",
		     reserved_fc, rc);

	reserved_fc = 0;
	rc = cxil_evtq_adjust_reserved_fc(transmit_evtq, -9);
	cr_assert_eq(rc, reserved_fc, "rc mismatch expected: %d received %d",
		     reserved_fc, rc);
}

TestSuite(misc, .init = dev_setup, .fini = dev_teardown);

Test(misc, inbound_wait)
{
	int rc;

	rc = cxil_inbound_wait(dev);
	cr_assert_eq(rc, 0, "rc mismatch expected: 0 received %d",
		      rc);
}
