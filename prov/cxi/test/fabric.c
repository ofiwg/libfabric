/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2015-2018 Cray Inc. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>

#include <criterion/criterion.h>

#include "cxi_prov.h"

struct fi_info *fi_hints;
static struct fid_fabric *fabric;
static struct fi_info *fi;

static void create_fabric(void)
{
	int ret;

	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION),
			 NULL, NULL, 0, fi_hints, &fi);
	cr_assert(ret == FI_SUCCESS, "fi_getinfo");

	ret = fi_fabric(fi->fabric_attr, &fabric, NULL);
	cr_assert(ret == FI_SUCCESS, "fi_fabric");
}

static void setup_getinfo(void)
{
	fi_hints = fi_allocinfo();
	cr_assert(fi_hints, "fi_allocinfo");
}

static void teardown_getinfo(void)
{
	fi_freeinfo(fi_hints);
}

static void setup_fabric(void)
{
	setup_getinfo();

	/* Always select CXI */
	fi_hints->fabric_attr->prov_name = strdup("cxi");

	create_fabric();
}

static void teardown_fabric(void)
{
	int ret;

	ret = fi_close(&fabric->fid);
	cr_assert(ret == FI_SUCCESS, "fi_close fabric");

	teardown_getinfo();
}

TestSuite(getinfo, .init = setup_getinfo,
	  .fini = teardown_getinfo);

/* Test fabric selection with provider name */
Test(getinfo, prov_name)
{
	fi_hints->fabric_attr->prov_name = strdup(cxi_prov_name);
	cr_assert(fi_hints, "fi_allocinfo");

	create_fabric();
	cr_assert(fabric != NULL);

	cr_assert(!strcmp(fi->fabric_attr->prov_name, cxi_prov_name));
}

/* Test fabric selection with domain name */
Test(getinfo, dom_name)
{
	fi_hints->domain_attr->name = strdup(cxi_dom_name);
	cr_assert(fi_hints, "fi_allocinfo");

	create_fabric();
	cr_assert(fabric != NULL);

	cr_assert(!strcmp(fi->domain_attr->name, cxi_dom_name));
}

/* Test fabric selection with fabric name */
/* TODO Fabric name selection does not work. */
Test(getinfo, fab_name, .disabled = true)
{
	fi_hints->fabric_attr->name = strdup(cxi_fab_name);
	cr_assert(fi_hints, "fi_allocinfo");

	create_fabric();
	cr_assert(fabric != NULL);

	cr_assert(!strcmp(fi->fabric_attr->name, cxi_fab_name));
}

TestSuite(fabric, .init = setup_fabric, .fini = teardown_fabric);

/* Test basic fabric creation */
Test(fabric, simple)
{
	cr_assert(fabric != NULL);
}

