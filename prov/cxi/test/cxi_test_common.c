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
struct fid_fabric *cxit_fabric;
struct fi_info *cxit_fi;
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
}

void cxit_create_fabric(void)
{
	int ret;

	cxit_create_fabric_info();

	ret = fi_fabric(cxit_fi->fabric_attr, &cxit_fabric, NULL);
	cr_assert(ret == FI_SUCCESS, "fi_fabric");
}

void cxit_destroy_fabric(void)
{
	int ret;

	ret = fi_close(&cxit_fabric->fid);
	cr_assert(ret == FI_SUCCESS, "fi_close fabric");

	cxit_destroy_fabric_info();
}

void cxit_fabric_test_init(void)
{
	struct slist_entry *entry, *prev;

	/* init OFI providers */
	cxit_create_fabric();
	cxit_destroy_fabric();

	(void) prev; /* Makes compiler happy */
	slist_foreach(&cxi_if_list, entry, prev) {
		cxit_n_ifs++;
	}
}

void cxit_setup_getinfo(void)
{
	cxit_fabric_test_init();

	cxit_fi_hints = fi_allocinfo();
	cr_assert(cxit_fi_hints, "fi_allocinfo");
}

void cxit_teardown_getinfo(void)
{
	fi_freeinfo(cxit_fi_hints);
}

void cxit_setup_fabric(void)
{
	cxit_setup_getinfo();

	/* Always select CXI */
	cxit_fi_hints->fabric_attr->prov_name = strdup(cxi_prov_name);

	cxit_create_fabric();
}

void cxit_teardown_fabric(void)
{
	cxit_destroy_fabric();
	cxit_teardown_getinfo();
}

