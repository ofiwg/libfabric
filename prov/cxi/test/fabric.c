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

TestSuite(fabric, .init = cxit_setup_fabric, .fini = cxit_teardown_fabric,
	  .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test basic fabric creation */
Test(fabric, simple)
{
	cxit_create_fabric();
	cr_assert(cxit_fabric != NULL);

	cxit_destroy_fabric();
}

