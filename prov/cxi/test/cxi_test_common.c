/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>

#include <criterion/criterion.h>

#include "cxi_prov.h"

struct fi_info *cxit_fi_hints;
struct fi_info *cxit_fi;
struct fid_fabric *cxit_fabric;
struct fid_domain *cxit_domain;
char *cxit_node, *cxit_service;
uint64_t cxit_flags;
int cxit_n_ifs;

void cxit_create_fabric_info(void)
{
	int ret;

	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION),
			 cxit_node, cxit_service, cxit_flags, cxit_fi_hints,
			 &cxit_fi);
	cr_assert(ret == FI_SUCCESS, "fi_getinfo");
}

void cxit_destroy_fabric_info(void)
{
	fi_freeinfo(cxit_fi);
	cxit_fi = NULL;
}

void cxit_create_fabric(void)
{
	int ret;

	ret = fi_fabric(cxit_fi->fabric_attr, &cxit_fabric, NULL);
	cr_assert(ret == FI_SUCCESS, "fi_fabric");
}

void cxit_destroy_fabric(void)
{
	int ret;

	ret = fi_close(&cxit_fabric->fid);
	cr_assert(ret == FI_SUCCESS, "fi_close fabric");
	cxit_fabric = NULL;
}

void cxit_create_domain(void)
{
	int ret;

	ret = fi_domain(cxit_fabric, cxit_fi, &cxit_domain, NULL);
	cr_assert(ret == FI_SUCCESS, "fi_domain");
}

void cxit_destroy_domain(void)
{
	int ret;

	ret = fi_close(&cxit_domain->fid);
	cr_assert(ret == FI_SUCCESS, "fi_close domain");
	cxit_domain = NULL;
}

static void cxit_init(void)
{
	struct slist_entry *entry, *prev;
	int ret;

	/* Force provider init */
	ret = fi_getinfo(FI_VERSION(-1, -1),
			 NULL, NULL, 0, NULL,
			 NULL);
	cr_assert(ret == -FI_ENOSYS);

	(void) prev; /* Makes compiler happy */
	slist_foreach(&cxi_if_list, entry, prev) {
		cxit_n_ifs++;
	}
}

void cxit_setup_getinfo(void)
{
	cxit_init();

	cxit_fi_hints = fi_allocinfo();
	cr_assert(cxit_fi_hints, "fi_allocinfo");
}

void cxit_teardown_getinfo(void)
{
	fi_freeinfo(cxit_fi_hints);
	cxit_fi_hints = NULL;
}

void cxit_setup_fabric(void)
{
	cxit_setup_getinfo();

	/* Always select CXI */
	cxit_fi_hints->fabric_attr->prov_name = strdup(cxi_prov_name);

	cxit_create_fabric_info();
}

void cxit_teardown_fabric(void)
{
	cxit_destroy_fabric_info();
	cxit_teardown_getinfo();
}

void cxit_setup_domain(void)
{
	cxit_setup_fabric();
	cxit_create_fabric();
}

void cxit_teardown_domain(void)
{
	cxit_destroy_fabric();
	cxit_teardown_fabric();
}
