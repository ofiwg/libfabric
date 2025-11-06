/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause */
/* Copyright 2018 Hewlett Packard Enterprise Development LP */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>

#include "libcxi_test_common.h"

TestSuite(PtlTE, .init = domain_setup, .fini = domain_teardown);

Test(PtlTE, null)
{
	struct cxi_pt_alloc_opts pte_opts = {};
	int rc;

	/* combinations of invalid parameters */
	rc = cxil_alloc_pte(NULL, NULL, NULL, NULL);
	cr_assert_eq(rc, -EINVAL);

	rc = cxil_alloc_pte(lni, NULL, NULL, NULL);
	cr_assert_eq(rc, -EINVAL);

	rc = cxil_alloc_pte(NULL, NULL, &pte_opts, NULL);
	cr_assert_eq(rc, -EINVAL);

	rc = cxil_alloc_pte(NULL, NULL, NULL, &rx_pte);
	cr_assert_eq(rc, -EINVAL);
	cr_assert_eq(rx_pte, NULL);

	rc = cxil_alloc_pte(lni, NULL, &pte_opts, NULL);
	cr_assert_eq(rc, -EINVAL);

	rc = cxil_alloc_pte(lni, NULL, NULL, &rx_pte);
	cr_assert_eq(rc, -EINVAL);
	cr_assert_eq(rx_pte, NULL);

	rc = cxil_alloc_pte(NULL, NULL, &pte_opts, &rx_pte);
	cr_assert_eq(rc, -EINVAL);
	cr_assert_eq(rx_pte, NULL);

	rc = cxil_destroy_pte(NULL);
	cr_assert_eq(rc, -EINVAL);
}

Test(PtlTE, basic)
{
	struct cxi_pt_alloc_opts pte_opts = {};
	struct cxi_eq *evtq;
	int rc;
	void *eq_buf;
	size_t eq_buf_len = s_page_size;
	struct cxi_md *eq_buf_md;
	struct cxi_eq_attr attr = {};

	eq_buf = aligned_alloc(s_page_size, eq_buf_len);
	cr_assert(eq_buf);
	memset(eq_buf, 0, eq_buf_len);

	rc = cxil_map(lni, eq_buf, eq_buf_len,
		      CXI_MAP_PIN | CXI_MAP_READ | CXI_MAP_WRITE,
		      NULL, &eq_buf_md);
	cr_assert(!rc);

	/* Allocate EVTQ */
	attr.queue = eq_buf;
	attr.queue_len = eq_buf_len;

	evtq = NULL;
	rc = cxil_alloc_evtq(lni, eq_buf_md, &attr, NULL, NULL, &evtq);
	cr_assert_eq(rc, 0);
	cr_assert_neq(evtq, NULL);

	/* Allocate PtlTE */
	rx_pte = NULL;
	rc = cxil_alloc_pte(lni, evtq, &pte_opts, &rx_pte);
	cr_assert_eq(rc, 0);
	cr_assert_neq(rx_pte, NULL);

	/* Should be blocked from destroying LNI or EQ */
	rc = cxil_destroy_lni(lni);
	cr_assert_eq(rc, -EBUSY);

	/* Free PtlTE */
	rc = cxil_destroy_pte(rx_pte);
	cr_assert_eq(rc, 0);

	/* Free EQ */
	rc = cxil_destroy_evtq(evtq);
	cr_assert_eq(rc, 0);

	rc = cxil_unmap(eq_buf_md);
	cr_assert_eq(rc, 0);

	free(eq_buf);
}

TestSuite(PtlTEmap, .init = pte_setup, .fini = pte_teardown);

Test(PtlTEmap, null)
{
	struct cxil_pte_map *pte_map = NULL;
	int rc;

	rc = cxil_map_pte(NULL, NULL, 0, 0, NULL);
	cr_assert_eq(rc, -EINVAL);

	rc = cxil_map_pte(NULL, NULL, 0, 0, NULL);
	cr_assert_eq(rc, -EINVAL);

	rc = cxil_map_pte(rx_pte, NULL, 0, 0, NULL);
	cr_assert_eq(rc, -EINVAL);

	rc = cxil_map_pte(NULL, domain, 0, 0, NULL);
	cr_assert_eq(rc, -EINVAL);

	rc = cxil_map_pte(NULL, NULL, 0, 0, &pte_map);
	cr_assert_eq(rc, -EINVAL);
	cr_assert_eq(pte_map, NULL);

	rc = cxil_map_pte(rx_pte, domain, 0, 0, NULL);
	cr_assert_eq(rc, -EINVAL);

	rc = cxil_map_pte(NULL, domain, 0, 0, &pte_map);
	cr_assert_eq(rc, -EINVAL);
	cr_assert_eq(pte_map, NULL);

	rc = cxil_map_pte(rx_pte, NULL, 0, 0, &pte_map);
	cr_assert_eq(rc, -EINVAL);
	cr_assert_eq(pte_map, NULL);

	rc = cxil_unmap_pte(NULL);
	cr_assert_eq(rc, -EINVAL);
}

Test(PtlTEmap, basic)
{
	struct cxil_pte_map *pte_map = NULL;
	int rc;

	/* Map PtlTE */
	pte_map = NULL;
	rc = cxil_map_pte(rx_pte, domain, 0, 0, &pte_map);
	cr_assert_eq(rc, 0);
	cr_assert_neq(pte_map, NULL);

	/* Should be blocked from destroying DOMAIN */
	rc = cxil_destroy_domain(domain);
	cr_assert_eq(rc, -EBUSY);

	/* Unmap PtlTE */
	rc = cxil_unmap_pte(pte_map);
	cr_assert_eq(rc, 0);
}
