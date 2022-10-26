/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2015-2018 Cray Inc. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>

#include <criterion/criterion.h>

#include "cxip.h"
#include "cxip_test_common.h"

static const char cxip_dom_fmt[] = "cxi%d";

static char *get_dom_name(int if_idx)
{
	char *dom;
	int ret;

	ret = asprintf(&dom, cxip_dom_fmt, if_idx);
	cr_assert(ret > 0);

	return dom;
}

TestSuite(getinfo, .init = cxit_setup_getinfo,
	  .fini = cxit_teardown_getinfo, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test fabric selection with provider name */
Test(getinfo, prov_name)
{
	int infos = 0;

	cxit_fi_hints->fabric_attr->prov_name = strdup(cxip_prov_name);

	cxit_create_fabric_info();
	cr_assert(cxit_fi != NULL);

	/* Make sure we have at least 1 FI for each IF */
	do {
		cr_assert(!strcmp(cxit_fi->fabric_attr->prov_name,
				  cxip_prov_name));
		infos++;
	} while ((cxit_fi = cxit_fi->next));
	cr_assert(infos >= cxit_n_ifs);
}

/* Test fabric selection with domain name */
Test(getinfo, dom_name)
{
	int infos = 0;
	struct cxip_if *if_entry;
	struct slist_entry *entry, *prev __attribute__ ((unused));

	slist_foreach(&cxip_if_list, entry, prev) {
		if_entry = container_of(entry, struct cxip_if, if_entry);
		infos = 0;

		cxit_node = get_dom_name(if_entry->info->dev_id);
		cxit_flags = FI_SOURCE;
		printf("searching %s\n", cxit_node);

		cxit_create_fabric_info();
		cr_assert(cxit_fi != NULL);

		/* Make sure we have at least 1 FI for each IF */
		do {
			cr_expect(!strcmp(cxit_fi->domain_attr->name,
					  cxit_node),
					  "%s != %s\n",
					  cxit_fi->domain_attr->name,
					  cxit_fi_hints->domain_attr->name);

			cr_assert(!strcmp(cxit_fi->fabric_attr->prov_name,
					  cxip_prov_name));

			cr_assert(!strcmp(cxit_fi->fabric_attr->name,
				  cxip_prov_name));

			infos++;
		} while ((cxit_fi = cxit_fi->next));
		cr_assert(infos >= 1);

		cxit_destroy_fabric_info();
	}
	cr_assert(infos >= 1);
}

/* Test fabric selection with fabric name */
Test(getinfo, fab_name)
{
	int infos = 0;
	struct slist_entry *entry, *prev __attribute__ ((unused));
	struct fi_info *fi;

	slist_foreach(&cxip_if_list, entry, prev) {
		infos = 0;

		cxit_fi_hints->fabric_attr->name = strdup(cxip_prov_name);

		cxit_create_fabric_info();
		cr_assert(cxit_fi != NULL);

		fi = cxit_fi;
		do {
			/* Not all providers can be trusted to filter by fabric
			 * name */
			if (strcmp(fi->fabric_attr->prov_name,
				   cxip_prov_name))
				continue;

			cr_assert(!strcmp(fi->fabric_attr->name,
					  fi->fabric_attr->name));

			infos++;
		} while ((fi = fi->next));

		cxit_destroy_fabric_info();
	}
	cr_assert(infos);
}

TestSuite(getinfo_infos, .timeout = CXIT_DEFAULT_TIMEOUT);

#define MAX_INFOS	8
#define FI_ADDR_CXI_COMPAT FI_ADDR_OPX

struct info_check {
	int mr_mode;
	uint32_t format;
};

Test(getinfo_infos, nohints)
{
	int num_info;
	int i;
	int info_per_if = 0;
	struct fi_info *fi_ptr;
	char *dom_name;
	char *odp;
	char *compat;
	struct info_check infos[MAX_INFOS];

	cxit_init();
	cr_assert(!cxit_fi_hints, "hints not NULL");

	cxit_create_fabric_info();
	cr_assert(cxit_fi != NULL);

	for (i = 0; i < MAX_INFOS; i++) {
		infos[i].format = 0;
		infos[i].mr_mode = -1;
	}

	/* By default when no hints are specified, each interface
	 * should have 2 fi_info.
	 */
	infos[info_per_if].mr_mode = FI_MR_ENDPOINT | FI_MR_ALLOCATED |
				     FI_MR_PROV_KEY;
	infos[info_per_if].format = FI_ADDR_CXI;
	info_per_if++;

	infos[info_per_if].mr_mode = FI_MR_ENDPOINT | FI_MR_ALLOCATED;
	infos[info_per_if].format = FI_ADDR_CXI;
	info_per_if++;

	/* Add ODP versions if enabled */
	odp = getenv("FI_CXI_ODP");
	if (odp && strtol(odp, NULL, 10)) {
		infos[info_per_if].format = FI_ADDR_CXI;
		infos[info_per_if].mr_mode = FI_MR_ENDPOINT | FI_MR_PROV_KEY;
		info_per_if++;

		infos[info_per_if].format = FI_ADDR_CXI;
		infos[info_per_if].mr_mode = FI_MR_ENDPOINT;
		info_per_if++;
	}

	/* If we are supporting compatibility with old constants,
	 * then fi_info are repeated with compatibility constants.
	 */
	compat = getenv("FI_CXI_COMPAT");
	if (!compat || strtol(compat, NULL, 10) == 1) {
		for (i = 0; i < info_per_if; i++) {
			infos[info_per_if + i].mr_mode =
				infos[i].mr_mode;
			infos[info_per_if + i].format =
				FI_ADDR_CXI_COMPAT;
		}
		info_per_if += i;
	}
	cr_assert(info_per_if <= MAX_INFOS, "Too many infos");

	fi_ptr = cxit_fi;

	while (fi_ptr) {
		/* Only concerned with CXI */
		if (strcmp(fi_ptr->fabric_attr->prov_name, cxip_prov_name)) {
			fi_ptr = fi_ptr->next;
			continue;
		}

		dom_name = fi_ptr->domain_attr->name;
		num_info = 0;

		/* Each info for the same NIC as the same domain name */
		while (fi_ptr) {
			/* Different interface detected */
			if (strcmp(dom_name, fi_ptr->domain_attr->name))
				break;

			num_info++;
			cr_assert(num_info <= MAX_INFOS,
				  "too many fi_info %d", num_info);

			cr_assert(infos[num_info - 1].mr_mode ==
				  fi_ptr->domain_attr->mr_mode,
				  "expected MR mode %x got %x",
				  infos[num_info - 1].mr_mode,
				  fi_ptr->domain_attr->mr_mode);

			cr_assert(infos[num_info - 1].format ==
				  fi_ptr->addr_format,
				  "expected addr_fomrat %u got %u",
				  infos[num_info - 1].format,
				  fi_ptr->addr_format);

			fi_ptr = fi_ptr->next;
		}

		cr_assert(num_info == info_per_if,
			  "Wrong number of fi_info %d got %d",
			  num_info, info_per_if);
		if (fi_ptr)
			fi_ptr = fi_ptr->next;
	}
	cxit_destroy_fabric_info();
}

Test(getinfo_infos, hints)
{
	int num_info;
	int i;
	int info_per_if = 0;
	struct fi_info *fi_ptr;
	char *dom_name;
	char *compat;
	struct info_check infos[2];

	cxit_setup_fabric();
	cr_assert(cxit_fi != NULL);
	cr_assert(cxit_fi_hints != NULL);

	for (i = 0; i < 2; i++) {
		infos[i].format = 0;
		infos[i].mr_mode = -1;
	}

	infos[0].format = FI_ADDR_CXI;
	infos[0].mr_mode = FI_MR_ENDPOINT | FI_MR_ALLOCATED;
	if (cxit_prov_key)
		infos[0].mr_mode |= FI_MR_PROV_KEY;
	info_per_if++;

	compat = getenv("FI_CXI_COMPAT");
	if (!compat || strtol(compat, NULL, 10) == 1) {
		infos[1].format = FI_ADDR_CXI_COMPAT;
		infos[1].mr_mode = infos[0].mr_mode;
		info_per_if++;
	}

	fi_ptr = cxit_fi;

	while (fi_ptr) {
		/* Should only be CXI provider */
		cr_assert(!strcmp(fi_ptr->fabric_attr->prov_name,
				  cxip_prov_name), "non-cxi provider");

		dom_name = fi_ptr->domain_attr->name;
		num_info = 0;

		/* Each info for the same NIC as the same domain name */
		while (fi_ptr) {
			/* Different interface detected */
			if (strcmp(dom_name, fi_ptr->domain_attr->name))
				break;

			num_info++;
			cr_assert(num_info <= 2, "too many fi_info %d",
				  num_info);

			cr_assert(infos[num_info - 1].mr_mode ==
				  fi_ptr->domain_attr->mr_mode,
				  "expected MR mode %x got %x",
				  infos[num_info - 1].mr_mode,
				  fi_ptr->domain_attr->mr_mode);

			cr_assert(infos[num_info - 1].format ==
				  fi_ptr->addr_format,
				  "expected addr_fomrat %u got %u",
				  infos[num_info - 1].format,
				  fi_ptr->addr_format);

			fi_ptr = fi_ptr->next;
		}

		cr_assert(num_info == info_per_if,
			  "Wrong number of fi_info %d got %d",
			  num_info, info_per_if);
		if (fi_ptr)
			fi_ptr = fi_ptr->next;
	}
	cxit_teardown_fabric();
}

TestSuite(fabric, .init = cxit_setup_fabric, .fini = cxit_teardown_fabric,
	  .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test basic fabric creation */
Test(fabric, simple)
{
	cxit_create_fabric();
	cr_assert(cxit_fabric != NULL);

	cxit_destroy_fabric();
}

