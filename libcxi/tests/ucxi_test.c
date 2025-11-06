/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause */
/* Copyright 2018 Hewlett Packard Enterprise Development LP */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <string.h>

#include "libcxi_priv.h"
#include "libcxi_test_common.h"

TestSuite(ucxi_lni, .init = dev_setup, .fini = dev_teardown);

struct lni_alloc_params {
	int write_sz;
	bool with_resp;
	int destroy;
	int destroy_hndl_off;
	int write_rc;
	int write_errno;
};

ParameterizedTestParameters(ucxi_lni, lni_alloc)
{
	size_t param_sz;
	static const struct lni_alloc_params params[] = {
		/* good LNI */
		{.write_sz = sizeof(struct cxi_lni_alloc_cmd),
		 .with_resp = true,
		 .destroy = 1,
		 .destroy_hndl_off = 0,
		 .write_rc = sizeof(struct cxi_lni_alloc_cmd),
		 .write_errno = 0 },
		/* auto cleanup */
		{.write_sz = sizeof(struct cxi_lni_alloc_cmd),
		 .with_resp = true,
		 .destroy = 0,
		 .destroy_hndl_off = 0,
		 .write_rc = sizeof(struct cxi_lni_alloc_cmd),
		 .write_errno = 0 },
		/* bad size */
		{.write_sz = sizeof(struct cxi_lni_alloc_cmd) - 1,
		 .with_resp = true,
		 .destroy = 0,
		 .destroy_hndl_off = 0,
		 .write_rc = -1,
		 .write_errno = EINVAL},
		/* bad size */
		{.write_sz = sizeof(struct cxi_lni_alloc_cmd) + 1,
		 .with_resp = true,
		 .destroy = 0,
		 .destroy_hndl_off = 0,
		 .write_rc = -1,
		 .write_errno = EINVAL},
		/* invalid response */
		{.write_sz = sizeof(struct cxi_lni_alloc_cmd),
		 .with_resp = false,
		 .destroy = 0,
		 .destroy_hndl_off = 0,
		 .write_rc = -1,
		 .write_errno = EFAULT},
		/* invalid destroy handle */
		{.write_sz = sizeof(struct cxi_lni_alloc_cmd),
		 .with_resp = true,
		 .destroy = 1,
		 .destroy_hndl_off = 1,
		 .write_rc = sizeof(struct cxi_lni_alloc_cmd),
		 .write_errno = 0 },
	};

	param_sz = ARRAY_SIZE(params);
	return cr_make_param_array(struct lni_alloc_params, params,
				   param_sz);
}

ParameterizedTest(const struct lni_alloc_params *param, ucxi_lni, lni_alloc)
{
	struct cxil_dev_priv *pdev = (struct cxil_dev_priv *) dev;
	struct cxi_lni_alloc_cmd lni_alloc = {};
	struct cxi_lni_alloc_resp resp;
	struct cxi_lni_free_cmd lni_free = { .op = CXI_OP_LNI_FREE };
	int rc;

	lni_alloc.op = CXI_OP_LNI_ALLOC;
	lni_alloc.resp = param->with_resp ? &resp : NULL;
	lni_alloc.svc_id = CXI_DEFAULT_SVC_ID;
	rc = write(pdev->fd, &lni_alloc, param->write_sz);
	cr_assert_eq(rc, param->write_rc,
		     "RC mismatch, expected: %d received: %d",
		     param->write_rc, rc);
	if (param->write_errno != 0)
		cr_assert_eq(errno, param->write_errno,
			     "errno mismatch, expected: %d received: %d",
			     param->write_errno, errno);

	if (param->destroy && param->write_rc > 0) {
		lni_free.lni = ((struct cxi_lni_alloc_resp *)
				(lni_alloc.resp))->lni;
		lni_free.lni += param->destroy_hndl_off;
		rc = write(pdev->fd, &lni_free, sizeof(lni_free));
		if (param->destroy_hndl_off)
			cr_assert_eq(rc, -1);
		else
			cr_assert_eq(rc, sizeof(lni_free));
	}
}

Test(lni, null_buf)
{
	struct cxil_dev_priv *pdev = (struct cxil_dev_priv *) dev;
	struct cxi_lni_alloc_cmd cmd = {};
	int rc;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnonnull"
	rc = write(pdev->fd, NULL, sizeof(cmd));
#pragma GCC diagnostic pop
	cr_assert_eq(rc, -1,
		     "RC mismatch, expected: %d received: %d",
		     -1, rc);
	cr_assert_eq(errno, EFAULT,
		     "errno mismatch, expected: %d received: %d",
		     EFAULT, errno);
}

Test(lni, null_len)
{
	struct cxil_dev_priv *pdev = (struct cxil_dev_priv *) dev;
	struct cxi_lni_alloc_cmd cmd = {};
	int rc;

	rc = write(pdev->fd, &cmd, 0);
	cr_assert_eq(rc, -1,
		     "RC mismatch, expected: %d received: %d",
		     -1, rc);
	cr_assert_eq(errno, EINVAL,
		     "errno mismatch, expected: %d received: %d",
		     EINVAL, errno);
}

TestSuite(ucxi_domain, .init = lni_setup, .fini = lni_teardown);

struct domain_alloc_params {
	int write_sz;
	int lni_hndl_off;
	bool with_resp;
	unsigned int vni;
	unsigned int pid;
	int destroy;
	int destroy_hndl_off;
	int write_rc;
	int write_errno;
};

ParameterizedTestParameters(ucxi_domain, domain_alloc)
{
	size_t param_sz;
	static const struct domain_alloc_params params[] = {
		/* good domain */
		{.write_sz = sizeof(struct cxi_domain_alloc_cmd),
		 .lni_hndl_off = 0,
		 .with_resp = true,
		 .vni = 1,
		 .pid = 0,
		 .destroy = 1,
		 .destroy_hndl_off = 0,
		 .write_rc = sizeof(struct cxi_domain_alloc_cmd),
		 .write_errno = 0 },
		/* inval size */
		{.write_sz = sizeof(struct cxi_domain_alloc_cmd) - 1,
		 .lni_hndl_off = 0,
		 .with_resp = true,
		 .vni = 1,
		 .pid = 0,
		 .destroy = 0,
		 .destroy_hndl_off = 0,
		 .write_rc = -1,
		 .write_errno = EINVAL },
		/* inval size */
		{.write_sz = sizeof(struct cxi_domain_alloc_cmd) + 1,
		 .lni_hndl_off = 0,
		 .with_resp = true,
		 .vni = 1,
		 .pid = 0,
		 .destroy = 0,
		 .destroy_hndl_off = 0,
		 .write_rc = -1,
		 .write_errno = EINVAL },
		/* invalid response ptr */
		{.write_sz = sizeof(struct cxi_domain_alloc_cmd),
		 .lni_hndl_off = 0,
		 .with_resp = false,
		 .vni = 1,
		 .pid = 0,
		 .destroy = 0,
		 .destroy_hndl_off = 0,
		 .write_rc = -1,
		 .write_errno = EFAULT },
		/* VNI too big */
		{.write_sz = sizeof(struct cxi_domain_alloc_cmd),
		 .lni_hndl_off = 0,
		 .with_resp = true,
		 .vni = UINT16_MAX+1,
		 .pid = 0,
		 .destroy = 0,
		 .destroy_hndl_off = 0,
		 .write_rc = -1,
		 .write_errno = EINVAL },
		/* PID too big */
		{.write_sz = sizeof(struct cxi_domain_alloc_cmd),
		 .lni_hndl_off = 0,
		 .with_resp = true,
		 .vni = 1,
		 .pid = 1024 * 1024,
		 .destroy = 0,
		 .destroy_hndl_off = 0,
		 .write_rc = -1,
		 .write_errno = EINVAL },
		/* inval LNI */
		{.write_sz = sizeof(struct cxi_domain_alloc_cmd),
		 .lni_hndl_off = 1,
		 .with_resp = true,
		 .vni = 1,
		 .pid = 0,
		 .destroy = 0,
		 .destroy_hndl_off = 0,
		 .write_rc = -1,
		 .write_errno = EINVAL },
		/* LNI too big */
		{.write_sz = sizeof(struct cxi_domain_alloc_cmd),
		 .lni_hndl_off = UINT32_MAX/2,
		 .with_resp = true,
		 .vni = 1,
		 .pid = 0,
		 .destroy = 0,
		 .destroy_hndl_off = 0,
		 .write_rc = -1,
		 .write_errno = EINVAL },
		/* inval destroy handle */
		{.write_sz = sizeof(struct cxi_domain_alloc_cmd),
		 .lni_hndl_off = 0,
		 .with_resp = true,
		 .vni = 1,
		 .pid = 0,
		 .destroy = 1,
		 .destroy_hndl_off = 1,
		 .write_rc = sizeof(struct cxi_domain_alloc_cmd),
		 .write_errno = 0 },
	};

	param_sz = ARRAY_SIZE(params);
	return cr_make_param_array(struct domain_alloc_params, params,
				   param_sz);
}

ParameterizedTest(const struct domain_alloc_params *param, ucxi_domain,
		  domain_alloc)
{
	struct cxil_dev_priv *pdev = (struct cxil_dev_priv *) dev;
	struct cxi_domain_alloc_cmd domain_alloc = {};
	struct cxi_domain_alloc_resp resp;
	struct cxi_domain_free_cmd domain_free = { .op = CXI_OP_DOMAIN_FREE };
	int rc;

	domain_alloc.op = CXI_OP_DOMAIN_ALLOC;
	domain_alloc.resp = param->with_resp ? &resp : NULL;
	domain_alloc.lni = lni->id + param->lni_hndl_off;
	domain_alloc.vni = param->vni;
	domain_alloc.pid = param->pid;
	rc = write(pdev->fd, &domain_alloc, param->write_sz);
	cr_assert_eq(rc, param->write_rc,
		     "RC mismatch, expected: %d received: %d",
		     param->write_rc, rc);
	if (param->write_errno != 0)
		cr_assert_eq(errno, param->write_errno,
			     "errno mismatch, expected: %d received: %d",
			     param->write_errno, errno);

	if (param->destroy && param->write_rc > 0) {
		domain_free.domain = ((struct cxi_domain_alloc_resp *)
				  (domain_alloc.resp))->domain;
		domain_free.domain += param->destroy_hndl_off;
		rc = write(pdev->fd, &domain_free, sizeof(domain_free));
		if (param->destroy_hndl_off) {
			cr_assert_eq(rc, -1);

			domain_free.domain = ((struct cxi_domain_alloc_resp *)
					(domain_alloc.resp))->domain;
			rc = write(pdev->fd, &domain_free, sizeof(domain_free));
		} else {
			cr_assert_eq(rc, sizeof(domain_free));
		}
	}
}

TestSuite(ucxi_cmdq, .init = cp_setup, .fini = cp_teardown);

struct cmdq_alloc_params {
	int write_sz;
	int lni_hndl_off;
	struct cxi_cq_alloc_opts opts;
	bool with_resp;
	int destroy;
	int destroy_hndl_off;
	int write_rc;
	int write_errno;
	int lcid;
};

ParameterizedTestParameters(ucxi_cmdq, cmdq_alloc)
{
	size_t param_sz;
	static const struct cmdq_alloc_params params[] = {
		/* good rx cmdq */
		{.write_sz = sizeof(struct cxi_cq_alloc_cmd),
		 .lni_hndl_off = 0,
		 .opts.count = 1,
		 .opts.flags = 0,
		 .with_resp = true,
		 .destroy = 1,
		 .destroy_hndl_off = 0,
		 .write_rc = sizeof(struct cxi_cq_alloc_cmd),
		 .write_errno = 0,
		 .lcid = -1},
		/* good tx cmdq */
		{.write_sz = sizeof(struct cxi_cq_alloc_cmd),
		 .lni_hndl_off = 0,
		 .opts.count = 1,
		 .opts.flags = CXI_CQ_IS_TX,
		 .with_resp = true,
		 .destroy = 1,
		 .destroy_hndl_off = 0,
		 .write_rc = sizeof(struct cxi_cq_alloc_cmd),
		 .write_errno = 0,
		 .lcid = -1},
		/* inval size */
		{.write_sz = sizeof(struct cxi_cq_alloc_cmd) - 1,
		 .lni_hndl_off = 0,
		 .opts.count = 1,
		 .opts.flags = 0,
		 .with_resp = true,
		 .destroy = 1,
		 .destroy_hndl_off = 0,
		 .write_rc = -1,
		 .write_errno = EINVAL,
		 .lcid = -1},
		/* inval size */
		{.write_sz = sizeof(struct cxi_cq_alloc_cmd) + 1,
		 .lni_hndl_off = 0,
		 .opts.count = 1,
		 .opts.flags = 0,
		 .with_resp = true,
		 .destroy = 1,
		 .destroy_hndl_off = 0,
		 .write_rc = -1,
		 .write_errno = EINVAL,
		 .lcid = -1},
		/* inval response */
		{.write_sz = sizeof(struct cxi_cq_alloc_cmd),
		 .lni_hndl_off = 0,
		 .opts.count = 1,
		 .opts.flags = 0,
		 .with_resp = false,
		 .destroy = 1,
		 .destroy_hndl_off = 0,
		 .write_rc = -1,
		 .write_errno = EFAULT,
		 .lcid = -1},
		/* LNI too big */
		{.write_sz = sizeof(struct cxi_cq_alloc_cmd),
		 .lni_hndl_off = UINT32_MAX/2,
		 .opts.count = 1,
		 .opts.flags = 0,
		 .with_resp = true,
		 .destroy = 1,
		 .destroy_hndl_off = 0,
		 .write_rc = -1,
		 .write_errno = EINVAL,
		 .lcid = -1},
		/* 4k queue size */
		{.write_sz = sizeof(struct cxi_cq_alloc_cmd),
		 .lni_hndl_off = 0,
		 .opts.count = 0x1000,
		 .opts.flags = 0,
		 .with_resp = true,
		 .destroy = 1,
		 .destroy_hndl_off = 0,
		 .write_rc = sizeof(struct cxi_cq_alloc_cmd),
		 .write_errno = 0,
		 .lcid = -1},
		/* max queue size */
		{.write_sz = sizeof(struct cxi_cq_alloc_cmd),
		 .lni_hndl_off = 0,
		 .opts.count = (1 << 16),
		 .opts.flags = 0,
		 .with_resp = true,
		 .destroy = 1,
		 .destroy_hndl_off = 0,
		 .write_rc = sizeof(struct cxi_cq_alloc_cmd),
		 .write_errno = 0,
		 .lcid = -1},
		/* queue size too big */
		{.write_sz = sizeof(struct cxi_cq_alloc_cmd),
		 .lni_hndl_off = 0,
		 .opts.count = (1 << 16) + 1,
		 .opts.flags = 0,
		 .with_resp = true,
		 .destroy = 1,
		 .destroy_hndl_off = 0,
		 .write_rc = -1,
		 .write_errno = E2BIG,
		 .lcid = -1},
		 /* bad tx cmdq LCID */
		{.write_sz = sizeof(struct cxi_cq_alloc_cmd),
		 .lni_hndl_off = 0,
		 .opts.count = 1,
		 .opts.flags = CXI_CQ_IS_TX,
		 .with_resp = true,
		 .destroy = 1,
		 .destroy_hndl_off = 0,
		 .write_rc = sizeof(struct cxi_cq_alloc_cmd),
		 .write_errno = 0,
		 .lcid = 13},
	};

	param_sz = ARRAY_SIZE(params);
	return cr_make_param_array(struct cmdq_alloc_params, params,
				   param_sz);
}

ParameterizedTest(const struct cmdq_alloc_params *param, ucxi_cmdq, cmdq_alloc)
{
	struct cxil_dev_priv *pdev = (struct cxil_dev_priv *) dev;
	struct cxi_cq_alloc_cmd cmdq_alloc = {};
	static struct cxi_cq_alloc_resp resp;
	struct cxi_cq_free_cmd cmdq_free = { .op = CXI_OP_CQ_FREE };
	int rc;

	cmdq_alloc.op = CXI_OP_CQ_ALLOC;
	cmdq_alloc.resp = param->with_resp ? &resp : NULL;
	cmdq_alloc.lni = lni->id + param->lni_hndl_off;
	cmdq_alloc.opts = param->opts;
	cmdq_alloc.eq = C_EQ_NONE;

	/* -1 means use allocated LCID */
	if (param->lcid == -1)
		cmdq_alloc.opts.lcid = cp->lcid;

	rc = write(pdev->fd, &cmdq_alloc, param->write_sz);
	cr_assert_eq(rc, param->write_rc,
		     "RC mismatch, expected: %d received: %d (errno %d)",
		     param->write_rc, rc, errno);
	if (param->write_errno != 0)
		cr_assert_eq(errno, param->write_errno,
			     "errno mismatch, expected: %d received: %d",
			     param->write_errno, errno);

	if (param->destroy && param->write_rc > 0) {
		cmdq_free.cq = ((struct cxi_cq_alloc_resp *)
				(cmdq_alloc.resp))->cq;
		cmdq_free.cq += param->destroy_hndl_off;
		rc = write(pdev->fd, &cmdq_free, sizeof(cmdq_free));
		if (param->destroy_hndl_off)
			cr_assert_eq(rc, -1);
		else
			cr_assert_eq(rc, sizeof(cmdq_free));
	}
}

TestSuite(ucxi_evtq, .init = lni_setup, .fini = lni_teardown);

struct evtq_alloc_params {
	unsigned int id;   /* test ID */
	bool resp;         /* define response? */
	size_t lni_offset; /* offset added to LNI handle */
	bool alloc_queue;  /* allocate EQ buffer? */
	size_t queue_len;
	size_t queue_off;  /* offset added to EQ buffer */
	size_t event_wait_off;
	size_t status_wait_off;
	unsigned long sts_tb;
	unsigned long sts_td;
	unsigned long sts_cnt;
	bool map_queue;    /* map EQ buffer? */
	uint64_t flags;    /* EQ allocation flags */
	ssize_t alloc_rc;  /* Expected EQ alloc RC */
	int eq_offset;     /* offset to add to EQ handle */
	ssize_t free_rc;   /* expected EQ free RC */
};

ParameterizedTestParameters(ucxi_evtq, evtq_alloc)
{
	size_t param_sz;
	static const struct evtq_alloc_params params[] = {
		/* Happy path */
		{.resp = true,
		 .lni_offset = 0,
		 .alloc_queue = true,
		 .queue_len = 1,
		 .queue_off = 0,
		 .map_queue = true,
		 .alloc_rc = sizeof(struct cxi_eq_alloc_cmd),
		 .eq_offset = 0,
		 .free_rc = sizeof(struct cxi_eq_free_cmd),
		 .id = 0},
		/* Invalid LNI # */
		{.resp = true,
		 .lni_offset = 1,
		 .alloc_queue = true,
		 .queue_len = 1,
		 .queue_off = 0,
		 .map_queue = true,
		 .alloc_rc = -1,
		 .id = 1},
		/* Unaligned queue length */
		{.resp = true,
		 .lni_offset = 0,
		 .alloc_queue = true,
		 .queue_len = 2,
		 .queue_off = 0,
		 .map_queue = true,
		 .alloc_rc = -1,
		 .id = 2},
		/* NULL response */
		{.resp = false,
		 .lni_offset = 0,
		 .alloc_queue = true,
		 .queue_len = 1,
		 .queue_off = 0,
		 .map_queue = true,
		 .alloc_rc = -1,
		 .id = 3},
		/* Invalid EQ to free */
		{.resp = true,
		 .lni_offset = 0,
		 .alloc_queue = true,
		 .queue_len = 1,
		 .queue_off = 0,
		 .map_queue = true,
		 .alloc_rc = sizeof(struct cxi_eq_alloc_cmd),
		 .eq_offset = 1,
		 .free_rc = -1,
		 .id = 4},
		/* Unaligned queue */
		{.resp = true,
		 .lni_offset = 0,
		 .alloc_queue = true,
		 .queue_len = 1,
		 .queue_off = 2048,
		 .map_queue = true,
		 .alloc_rc = -1,
		 .eq_offset = 0,
		 .free_rc = sizeof(struct cxi_eq_free_cmd),
		 .id = 5},
		/* Invalid queue */
		{.resp = true,
		 .lni_offset = 0,
		 .alloc_queue = false,
		 .queue_len = 1,
		 .queue_off = 0,
		 .map_queue = true,
		 .alloc_rc = -1,
		 .eq_offset = 0,
		 .free_rc = sizeof(struct cxi_eq_free_cmd),
		 .id = 6},
		/* Invalid MD */
		{.resp = true,
		 .lni_offset = 0,
		 .alloc_queue = true,
		 .queue_len = 1,
		 .queue_off = 0,
		 .map_queue = false,
		 .alloc_rc = -1,
		 .eq_offset = 0,
		 .free_rc = sizeof(struct cxi_eq_free_cmd),
		 .id = 7},
		/* Invalid event wait object */
		{.resp = true,
		 .lni_offset = 0,
		 .alloc_queue = true,
		 .queue_len = 1,
		 .queue_off = 0,
		 .event_wait_off = 1,
		 .map_queue = true,
		 .alloc_rc = -1,
		 .eq_offset = 0,
		 .free_rc = sizeof(struct cxi_eq_free_cmd),
		 .id = 8},
		/* Invalid status wait object */
		{.resp = true,
		 .lni_offset = 0,
		 .alloc_queue = true,
		 .queue_len = 1,
		 .queue_off = 0,
		 .status_wait_off = 1,
		 .map_queue = true,
		 .alloc_rc = -1,
		 .eq_offset = 0,
		 .free_rc = sizeof(struct cxi_eq_free_cmd),
		 .id = 9},
		/* Invalid thresh_base */
		{.resp = true,
		 .lni_offset = 0,
		 .alloc_queue = true,
		 .queue_len = 1,
		 .queue_off = 0,
		 .sts_tb = 101,
		 .sts_td = 10,
		 .sts_cnt = 1,
		 .map_queue = true,
		 .alloc_rc = -1,
		 .eq_offset = 0,
		 .free_rc = sizeof(struct cxi_eq_free_cmd),
		 .id = 10},
		/* Invalid thresh_delta */
		{.resp = true,
		 .lni_offset = 0,
		 .alloc_queue = true,
		 .queue_len = 1,
		 .queue_off = 0,
		 .sts_tb = 100,
		 .sts_td = 101,
		 .sts_cnt = 1,
		 .map_queue = true,
		 .alloc_rc = -1,
		 .eq_offset = 0,
		 .free_rc = sizeof(struct cxi_eq_free_cmd),
		 .id = 11},
		/* Invalid thresh_count */
		{.resp = true,
		 .lni_offset = 0,
		 .alloc_queue = true,
		 .queue_len = 1,
		 .queue_off = 0,
		 .sts_tb = 100,
		 .sts_td = 10,
		 .sts_cnt = 5,
		 .map_queue = true,
		 .alloc_rc = -1,
		 .eq_offset = 0,
		 .free_rc = sizeof(struct cxi_eq_free_cmd),
		 .id = 12},
		/* Invalid max threshold */
		{.resp = true,
		 .lni_offset = 0,
		 .alloc_queue = true,
		 .queue_len = 1,
		 .queue_off = 0,
		 .sts_tb = 90,
		 .sts_td = 40,
		 .sts_cnt = 4,
		 .map_queue = true,
		 .alloc_rc = -1,
		 .eq_offset = 0,
		 .free_rc = sizeof(struct cxi_eq_free_cmd),
		 .id = 13},
		/* Happy path with status updates */
		{.resp = true,
		 .lni_offset = 0,
		 .alloc_queue = true,
		 .queue_len = 1,
		 .queue_off = 0,
		 .sts_tb = 90,
		 .sts_td = 10,
		 .sts_cnt = 4,
		 .map_queue = true,
		 .alloc_rc = sizeof(struct cxi_eq_alloc_cmd),
		 .eq_offset = 0,
		 .free_rc = sizeof(struct cxi_eq_free_cmd),
		 .id = 14},
	};

	param_sz = ARRAY_SIZE(params);
	return cr_make_param_array(struct evtq_alloc_params, params,
				   param_sz);
}

ParameterizedTest(const struct evtq_alloc_params *p, ucxi_evtq, evtq_alloc)
{
	struct cxil_dev_priv *pdev = (struct cxil_dev_priv *) dev;
	struct cxi_eq_alloc_cmd eq_alloc = {.op = CXI_OP_EQ_ALLOC};
	struct cxi_eq_free_cmd eq_free = {.op = CXI_OP_EQ_FREE};
	struct cxi_eq_alloc_resp evtq_alloc_resp = {};
	void *eq_buf_aligned = NULL;
	void *eq_buf = NULL;
	struct cxi_md *eq_buf_md;
	struct cxil_md_priv *md_priv = NULL;
	ssize_t rc;
	size_t queue_len = s_page_size / p->queue_len;

	if (p->alloc_queue) {
		rc = posix_memalign(&eq_buf_aligned, s_page_size,
				    queue_len + p->queue_off);
		cr_assert(rc == 0);

		eq_buf = (uint8_t *)eq_buf_aligned + p->queue_off;
		memset(eq_buf, 0, queue_len);

		if (p->map_queue) {
			rc = cxil_map(lni, eq_buf, queue_len,
				      CXI_MAP_PIN | CXI_MAP_READ |
				      CXI_MAP_WRITE,
				      NULL, &eq_buf_md);
			cr_assert(!rc);

			md_priv = container_of(eq_buf_md,
					       struct cxil_md_priv,
					       md);
		}
	}

	memset(&evtq_alloc_resp, 0xff, sizeof(evtq_alloc_resp));

	rc = cxil_alloc_wait_obj(lni, &wait_obj);
	cr_assert_eq(rc, 0, "cxil_alloc_wait_obj() failed %ld", rc);

	/* Allocate the Event Queue */
	if (p->resp)
		eq_alloc.resp = &evtq_alloc_resp;
	else
		eq_alloc.resp = NULL;
	eq_alloc.lni = lni->id + p->lni_offset;
	eq_alloc.queue_md = md_priv ? md_priv->md_hndl : CXI_MD_NONE;
	eq_alloc.event_wait = wait_obj->wait + p->event_wait_off;
	eq_alloc.status_wait = wait_obj->wait + p->status_wait_off;
	eq_alloc.attr.queue = eq_buf;
	eq_alloc.attr.queue_len = queue_len;
	eq_alloc.attr.flags = p->flags;
	eq_alloc.attr.status_thresh_base = p->sts_tb;
	eq_alloc.attr.status_thresh_delta = p->sts_td;
	eq_alloc.attr.status_thresh_count = p->sts_cnt;
	rc = write(pdev->fd, &eq_alloc, sizeof(eq_alloc));
	cr_expect_eq(rc, p->alloc_rc, "EvtQ Allocate failed (%d). %zd != %zd",
		     p->id, rc, p->alloc_rc);

	/* Do not continue on error */
	if (rc == -1) {
		cr_log_info("Exiting early after allocate (%d) errno %d",
			    p->id, errno);
		goto teardown;
	}

	/* Validate driver response */
	cr_expect_lt(evtq_alloc_resp.eq, 2048, "EvtQ ID (%d) %d",
		      p->id, evtq_alloc_resp.eq);

	/* Free the Event Queue */
	eq_free.eq = evtq_alloc_resp.eq + p->eq_offset;
	rc = write(pdev->fd, &eq_free, sizeof(eq_free));
	cr_expect_eq(rc, p->free_rc,
		     "EvtQ Free failed (%d). %zd != %zd errno %d",
		     p->id, rc, p->free_rc, errno);

	/* Ensure EQ is freed if the above failed */
	if (rc == -1) {
		eq_free.eq = evtq_alloc_resp.eq;
		rc = write(pdev->fd, &eq_free, sizeof(eq_free));
		if (rc != sizeof(eq_free))
			cr_log_warn("EvtQ Free FAIL (%d). %zd != %zd errno %d",
				     p->id, rc, sizeof(eq_free), errno);
	}

teardown:
	rc = cxil_destroy_wait_obj(wait_obj);
	cr_assert(!rc);

	if (eq_buf_aligned) {
		if (p->map_queue)
			cxil_unmap(eq_buf_md);
		free(eq_buf_aligned);
	}
}

Test(ucxi_evtq, resize)
{
	int ret;
	struct cxil_eq *eq;
	void *eq_buf;
	struct cxi_eq_resize_cmd resize_cmd = {
		.op = CXI_OP_EQ_RESIZE,
	};
	struct cxi_eq_resize_complete_cmd rc_cmd = {
		.op = CXI_OP_EQ_RESIZE_COMPLETE,
	};
	struct cxi_md *eq_md;
	struct cxil_md_priv *md_priv = NULL;
	struct cxi_eq_attr eq_attr = {};

	eq_buf = aligned_alloc(s_page_size, s_page_size);
	cr_assert(eq_buf);

	eq_attr.queue = eq_buf;
	eq_attr.queue_len = s_page_size;
	eq_attr.flags = CXI_EQ_PASSTHROUGH;

	ret = cxil_alloc_evtq(lni, NULL, &eq_attr, NULL, NULL, &transmit_evtq);
	cr_assert_eq(ret, 0, "Allocate TX EQ Failed %d", ret);

	eq = container_of(transmit_evtq, struct cxil_eq, hw);

	/* resize: bad EQ */
	resize_cmd.eq_hndl = eq->evtq_hndl+1;
	resize_cmd.queue = eq_buf;
	resize_cmd.queue_len = s_page_size;
	resize_cmd.queue_md = CXI_MD_NONE;
	ret = write(eq->lni_priv->dev->fd, &resize_cmd, sizeof(resize_cmd));
	cr_assert_eq(ret, -1, "1) resize failed to fail %d", ret);
	cr_assert_eq(errno, EINVAL, "1) resize bad errno %d", errno);

	/* resize: unaligned queue */
	resize_cmd.eq_hndl = eq->evtq_hndl;
	resize_cmd.queue = ((uint8_t *)eq_buf)+1;
	resize_cmd.queue_len = s_page_size;
	resize_cmd.queue_md = CXI_MD_NONE;
	ret = write(eq->lni_priv->dev->fd, &resize_cmd, sizeof(resize_cmd));
	cr_assert_eq(ret, -1, "2) failed to fail %d", ret);
	cr_assert_eq(errno, EINVAL, "2) resize bad errno %d", errno);

	/* resize: unaligned queue length */
	resize_cmd.eq_hndl = eq->evtq_hndl;
	resize_cmd.queue = eq_buf;
	resize_cmd.queue_len = s_page_size+1;
	resize_cmd.queue_md = CXI_MD_NONE;
	ret = write(eq->lni_priv->dev->fd, &resize_cmd, sizeof(resize_cmd));
	cr_assert_eq(ret, -1, "3) failed to fail %d", ret);
	cr_assert_eq(errno, EINVAL, "3) resize bad errno %d", errno);

	/* resize: NULL queue */
	resize_cmd.eq_hndl = eq->evtq_hndl;
	resize_cmd.queue = NULL;
	resize_cmd.queue_len = s_page_size;
	resize_cmd.queue_md = CXI_MD_NONE;
	ret = write(eq->lni_priv->dev->fd, &resize_cmd, sizeof(resize_cmd));
	cr_assert_eq(ret, -1, "4) failed to fail %d", ret);
	cr_assert_eq(errno, EINVAL, "4) resize bad errno %d", errno);

	/* resize: NULL queue len */
	resize_cmd.eq_hndl = eq->evtq_hndl;
	resize_cmd.queue = eq_buf;
	resize_cmd.queue_len = 0;
	resize_cmd.queue_md = CXI_MD_NONE;
	ret = write(eq->lni_priv->dev->fd, &resize_cmd, sizeof(resize_cmd));
	cr_assert_eq(ret, -1, "5) failed to fail %d", ret);
	cr_assert_eq(errno, EINVAL, "5) resize bad errno %d", errno);

	/* resize: success */
	resize_cmd.eq_hndl = eq->evtq_hndl;
	resize_cmd.queue = eq_buf;
	resize_cmd.queue_len = s_page_size;
	resize_cmd.queue_md = CXI_MD_NONE;
	ret = write(eq->lni_priv->dev->fd, &resize_cmd, sizeof(resize_cmd));
	cr_assert_eq(ret, sizeof(resize_cmd), "resize EQ Failed %d", ret);

	/* resize_complete: bad EQ */
	rc_cmd.eq_hndl = eq->evtq_hndl+1;
	ret = write(eq->lni_priv->dev->fd, &rc_cmd, sizeof(rc_cmd));
	cr_assert_eq(ret, -1, "1) resize_complete failed to fail %d", ret);
	cr_assert_eq(errno, EINVAL, "1) resize_complete bad errno %d", ret);

	/* resize_complete: success */
	rc_cmd.eq_hndl = eq->evtq_hndl;
	ret = write(eq->lni_priv->dev->fd, &rc_cmd, sizeof(rc_cmd));
	cr_assert_eq(ret, sizeof(rc_cmd), "resize_complete EQ Failed %d", ret);

	ret = cxil_destroy_evtq(transmit_evtq);
	cr_assert_eq(ret, 0, "Destroy TX EQ Failed %d", ret);

	/* Redo all tests with EQ translation */

	ret = cxil_map(lni, eq_buf, s_page_size,
		       CXI_MAP_PIN | CXI_MAP_READ | CXI_MAP_WRITE,
		       NULL, &eq_md);
	cr_assert_eq(ret, 0, "cxil_map() failed %d", ret);

	md_priv = container_of(eq_md, struct cxil_md_priv, md);

	eq_attr.queue = eq_buf;
	eq_attr.queue_len = s_page_size;
	eq_attr.flags = 0;

	ret = cxil_alloc_evtq(lni, eq_md, &eq_attr, NULL, NULL, &transmit_evtq);
	cr_assert_eq(ret, 0, "Allocate TX EQ Failed %d", ret);

	eq = container_of(transmit_evtq, struct cxil_eq, hw);

	/* resize: bad EQ */
	resize_cmd.eq_hndl = eq->evtq_hndl+1;
	resize_cmd.queue = eq_buf;
	resize_cmd.queue_len = s_page_size;
	resize_cmd.queue_md = md_priv->md_hndl;
	ret = write(eq->lni_priv->dev->fd, &resize_cmd, sizeof(resize_cmd));
	cr_assert_eq(ret, -1, "1) resize failed to fail %d", ret);
	cr_assert_eq(errno, EINVAL, "1) resize bad errno %d", errno);

	/* resize: unaligned queue */
	resize_cmd.eq_hndl = eq->evtq_hndl;
	resize_cmd.queue = (uint8_t *)eq_buf+1;
	resize_cmd.queue_len = s_page_size;
	resize_cmd.queue_md = md_priv->md_hndl;
	ret = write(eq->lni_priv->dev->fd, &resize_cmd, sizeof(resize_cmd));
	cr_assert_eq(ret, -1, "2) failed to fail %d", ret);
	cr_assert_eq(errno, EINVAL, "2) resize bad errno %d", errno);

	/* resize: unaligned queue length */
	resize_cmd.eq_hndl = eq->evtq_hndl;
	resize_cmd.queue = eq_buf;
	resize_cmd.queue_len = s_page_size+1;
	resize_cmd.queue_md = md_priv->md_hndl;
	ret = write(eq->lni_priv->dev->fd, &resize_cmd, sizeof(resize_cmd));
	cr_assert_eq(ret, -1, "3) failed to fail %d", ret);
	cr_assert_eq(errno, EINVAL, "3) resize bad errno %d", errno);

	/* resize: NULL queue */
	resize_cmd.eq_hndl = eq->evtq_hndl;
	resize_cmd.queue = NULL;
	resize_cmd.queue_len = s_page_size;
	resize_cmd.queue_md = md_priv->md_hndl;
	ret = write(eq->lni_priv->dev->fd, &resize_cmd, sizeof(resize_cmd));
	cr_assert_eq(ret, -1, "4) failed to fail %d", ret);
	cr_assert_eq(errno, EINVAL, "4) resize bad errno %d", errno);

	/* resize: NULL queue len */
	resize_cmd.eq_hndl = eq->evtq_hndl;
	resize_cmd.queue = eq_buf;
	resize_cmd.queue_len = 0;
	resize_cmd.queue_md = md_priv->md_hndl;
	ret = write(eq->lni_priv->dev->fd, &resize_cmd, sizeof(resize_cmd));
	cr_assert_eq(ret, -1, "5) failed to fail %d", ret);
	cr_assert_eq(errno, EINVAL, "5) resize bad errno %d", errno);

	/* resize: NULL queue MD */
	resize_cmd.eq_hndl = eq->evtq_hndl;
	resize_cmd.queue = eq_buf;
	resize_cmd.queue_len = 0;
	resize_cmd.queue_md = CXI_MD_NONE;
	ret = write(eq->lni_priv->dev->fd, &resize_cmd, sizeof(resize_cmd));
	cr_assert_eq(ret, -1, "6) failed to fail %d", ret);
	cr_assert_eq(errno, EINVAL, "6) resize bad errno %d", errno);

	/* resize: success */
	resize_cmd.eq_hndl = eq->evtq_hndl;
	resize_cmd.queue = eq_buf;
	resize_cmd.queue_len = s_page_size;
	resize_cmd.queue_md = md_priv->md_hndl;
	ret = write(eq->lni_priv->dev->fd, &resize_cmd, sizeof(resize_cmd));
	cr_assert_eq(ret, sizeof(resize_cmd), "resize EQ Failed %d", ret);

	/* resize_complete: bad EQ */
	rc_cmd.eq_hndl = eq->evtq_hndl+1;
	ret = write(eq->lni_priv->dev->fd, &rc_cmd, sizeof(rc_cmd));
	cr_assert_eq(ret, -1, "1) resize_complete failed to fail %d", ret);
	cr_assert_eq(errno, EINVAL, "1) resize_complete bad errno %d", ret);

	/* resize_complete: success */
	rc_cmd.eq_hndl = eq->evtq_hndl;
	ret = write(eq->lni_priv->dev->fd, &rc_cmd, sizeof(rc_cmd));
	cr_assert_eq(ret, sizeof(rc_cmd), "resize_complete EQ Failed %d", ret);

	ret = cxil_destroy_evtq(transmit_evtq);
	cr_assert_eq(ret, 0, "Destroy TX EQ Failed %d", ret);

	/* Try to free EQ with resize pending */

	eq_attr.queue = eq_buf;
	eq_attr.queue_len = s_page_size;
	eq_attr.flags = CXI_EQ_PASSTHROUGH;

	ret = cxil_alloc_evtq(lni, NULL, &eq_attr, NULL, NULL, &transmit_evtq);
	cr_assert_eq(ret, 0, "Allocate TX EQ Failed %d", ret);

	eq = container_of(transmit_evtq, struct cxil_eq, hw);

	/* resize: success */
	resize_cmd.eq_hndl = eq->evtq_hndl;
	resize_cmd.queue = eq_buf;
	resize_cmd.queue_len = s_page_size;
	resize_cmd.queue_md = CXI_MD_NONE;
	ret = write(eq->lni_priv->dev->fd, &resize_cmd, sizeof(resize_cmd));
	cr_assert_eq(ret, sizeof(resize_cmd), "resize EQ Failed %d", ret);

	ret = cxil_destroy_evtq(transmit_evtq);
	cr_assert_eq(ret, 0, "Destroy TX EQ Failed %d", ret);

	/* Try to free EQ with resize pending with translation */

	eq_attr.queue = eq_buf;
	eq_attr.queue_len = s_page_size;
	eq_attr.flags = 0;

	ret = cxil_alloc_evtq(lni, eq_md, &eq_attr, NULL, NULL, &transmit_evtq);
	cr_assert_eq(ret, 0, "Allocate TX EQ Failed %d", ret);

	eq = container_of(transmit_evtq, struct cxil_eq, hw);

	/* resize: success */
	resize_cmd.eq_hndl = eq->evtq_hndl;
	resize_cmd.queue = eq_buf;
	resize_cmd.queue_len = s_page_size;
	resize_cmd.queue_md = md_priv->md_hndl;
	ret = write(eq->lni_priv->dev->fd, &resize_cmd, sizeof(resize_cmd));
	cr_assert_eq(ret, sizeof(resize_cmd), "resize EQ Failed %d", ret);

	ret = cxil_destroy_evtq(transmit_evtq);
	cr_assert_eq(ret, 0, "Destroy TX EQ Failed %d", ret);

	/* clean up */

	ret = cxil_unmap(eq_md);
	cr_assert_eq(ret, 0, "unmap failed %d", ret);

	free(eq_buf);
}
