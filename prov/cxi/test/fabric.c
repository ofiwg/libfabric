/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2015-2018 Cray Inc. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>

#include <criterion/criterion.h>

#include "cxi_prov.h"
#include "cxi_test_common.h"

static char *get_dom_name(int if_idx, int dom_id)
{
	char *dom;
	int ret;

	ret = asprintf(&dom, cxi_dom_fmt, if_idx, dom_id);
	cr_assert(ret > 0);

	return dom;
}

static char *get_fab_name(int fab_id)
{
	char *fab;
	int ret;

	ret = asprintf(&fab, cxi_fab_fmt, fab_id);
	cr_assert(ret > 0);

	return fab;
}

TestSuite(getinfo, .init = cxit_setup_getinfo,
	  .fini = cxit_teardown_getinfo);

/* Test fabric selection with provider name */
Test(getinfo, prov_name)
{
	int infos = 0;

	cxit_fi_hints->fabric_attr->prov_name = strdup(cxi_prov_name);

	cxit_create_fabric_info();
	cr_assert(cxit_fi != NULL);

	/* Make sure we have 1 FI for each IF */
	do {
		cr_assert(!strcmp(cxit_fi->fabric_attr->prov_name,
				  cxi_prov_name));
		infos++;
	} while ((cxit_fi = cxit_fi->next));
	cr_assert(infos == (cxit_n_ifs * cxi_num_vpids));
}

/* Test fabric selection with domain name */
Test(getinfo, dom_name)
{
	int infos = 0;
	struct cxi_if_list_entry *if_entry;
	struct slist_entry *entry, *prev;
	char *fab_name;

	(void) prev; /* Makes compiler happy */
	slist_foreach(&cxi_if_list, entry, prev) {
		if_entry = container_of(entry, struct cxi_if_list_entry, entry);
		infos = 0;

		cxit_fi_hints->domain_attr->name =
				get_dom_name(if_entry->if_idx, 0);

		cxit_create_fabric_info();
		cr_assert(cxit_fi != NULL);

		/* Make sure we have 1 FI for each IF */
		do {
			cr_assert(!strcmp(cxit_fi->domain_attr->name,
					  cxit_fi_hints->domain_attr->name));

			cr_assert(!strcmp(cxit_fi->fabric_attr->prov_name,
					  cxi_prov_name));

			fab_name = get_fab_name(if_entry->if_fabric);
			cr_assert(!strcmp(cxit_fi->fabric_attr->name,
					  fab_name));
			free(fab_name);

			infos++;
		} while ((cxit_fi = cxit_fi->next));
		cr_assert(infos == 1);

		cxit_destroy_fabric_info();
	}
	cr_assert(infos == 1);
}

/* Test fabric selection with fabric name */
Test(getinfo, fab_name)
{
	int infos = 0;
	struct cxi_if_list_entry *if_entry;
	struct slist_entry *entry, *prev;

	(void) prev; /* Makes compiler happy */
	slist_foreach(&cxi_if_list, entry, prev) {
		if_entry = container_of(entry, struct cxi_if_list_entry, entry);
		infos = 0;

		cxit_fi_hints->fabric_attr->name =
				get_fab_name(if_entry->if_fabric);

		cxit_create_fabric_info();
		cr_assert(cxit_fi != NULL);

		do {
			/* Not all providers can be trusted to filter by fabric
			 * name */
			if (strcmp(cxit_fi->fabric_attr->prov_name,
				   cxi_prov_name))
				continue;

			cr_assert(!strcmp(cxit_fi->fabric_attr->name,
					  cxit_fi_hints->fabric_attr->name));

			infos++;
		} while ((cxit_fi = cxit_fi->next));

		cxit_destroy_fabric_info();
	}
	cr_assert(infos);
}

/* Test selection by source node */
Test(getinfo, src_node)
{
	int ret, infos = 0;
	struct cxi_addr *addr;
	struct cxi_if_list_entry *if_entry;
	struct slist_entry *entry, *prev;
	char *fab_name, *dom_name;

	cxit_fi_hints->fabric_attr->prov_name = strdup(cxi_prov_name);

	(void) prev; /* Makes compiler happy */
	slist_foreach(&cxi_if_list, entry, prev) {
		if_entry = container_of(entry, struct cxi_if_list_entry, entry);
		infos = 0;

		ret = asprintf(&cxit_node, "0x%x", if_entry->if_nic);
		cr_assert(ret > 0);

		cxit_flags = FI_SOURCE;

		cxit_create_fabric_info();
		cr_assert(cxit_fi != NULL);

		/* Make sure we have only 1 FI for each IF */
		do {
			cr_assert(!strcmp(cxit_fi->fabric_attr->prov_name,
					  cxi_prov_name));

			cr_assert(cxit_fi->src_addr);
			addr = (struct cxi_addr *)cxit_fi->src_addr;
			cr_assert(addr->nic == if_entry->if_nic);
			cr_assert(addr->domain == 0);
			cr_assert(addr->port == 0);

			fab_name = get_fab_name(if_entry->if_fabric);
			cr_assert(!strcmp(cxit_fi->fabric_attr->name,
					  fab_name));
			free(fab_name);

			dom_name = get_dom_name(if_entry->if_idx, 0);
			cr_assert(!strcmp(cxit_fi->domain_attr->name,
					  dom_name));
			free(dom_name);

			cr_assert(!strcmp(cxit_fi->fabric_attr->prov_name,
					  cxi_prov_name));
			infos++;
		} while ((cxit_fi = cxit_fi->next));
		cr_assert(infos == 1);

		cxit_destroy_fabric_info();
		free(cxit_node);
	}
	cr_assert(infos == 1);
}

/* Test selection by source node and service */
Test(getinfo, src_node_service)
{
	int ret, infos = 0;
	struct cxi_addr *addr;
	struct cxi_if_list_entry *if_entry;
	struct slist_entry *entry, *prev;
	int dom, port;
	char *fab_name, *dom_name;

	cxit_fi_hints->fabric_attr->prov_name = strdup(cxi_prov_name);

	(void) prev; /* Makes compiler happy */
	slist_foreach(&cxi_if_list, entry, prev) {
		if_entry = container_of(entry, struct cxi_if_list_entry, entry);
		infos = 0;

		ret = asprintf(&cxit_node, "0x%x", if_entry->if_nic);
		cr_assert(ret > 0);

		dom = if_entry->if_idx+4;
		port = if_entry->if_idx+5;
		ret = asprintf(&cxit_service, "%d:%d", dom, port);
		cr_assert(ret > 0);

		cxit_flags = FI_SOURCE;

		cxit_create_fabric_info();
		cr_assert(cxit_fi != NULL);

		/* Make sure we have only 1 FI for each IF */
		do {
			cr_assert(!strcmp(cxit_fi->fabric_attr->prov_name,
					  cxi_prov_name));

			cr_assert(cxit_fi->src_addr);
			addr = (struct cxi_addr *)cxit_fi->src_addr;
			cr_assert(addr->nic == if_entry->if_nic);
			cr_assert(addr->domain == dom);
			cr_assert(addr->port == port);

			fab_name = get_fab_name(if_entry->if_fabric);
			cr_assert(!strcmp(cxit_fi->fabric_attr->name,
					  fab_name));
			free(fab_name);

			dom_name = get_dom_name(if_entry->if_idx, dom);
			cr_assert(!strcmp(cxit_fi->domain_attr->name,
					  dom_name));
			free(dom_name);

			cr_assert(!strcmp(cxit_fi->fabric_attr->prov_name,
					  cxi_prov_name));
			infos++;
		} while ((cxit_fi = cxit_fi->next));
		cr_assert(infos == 1);

		cxit_destroy_fabric_info();
		free(cxit_service);
		free(cxit_node);
	}
	cr_assert(infos == 1);
}

/* Select fabric by node */
Test(getinfo, dest_node)
{
	int ret, infos = 0;
	struct cxi_addr *addr;
	struct cxi_if_list_entry *if_entry;
	char *fab_name, *dom_name;
	int nic_id = 130;

	cxit_fi_hints->fabric_attr->prov_name = strdup(cxi_prov_name);

	infos = 0;

	/* Make up the NIC of a remote node */
	ret = asprintf(&cxit_node, "0x%x", nic_id);
	cr_assert(ret > 0);

	cxit_create_fabric_info();
	cr_assert(cxit_fi != NULL);

	/* All FIs use the first available src address.  All FI data
	 * will match the first interface found.  Additional, node is
	 * used to create a dest addr.
	 */
	if_entry = container_of((cxi_if_list.head), struct cxi_if_list_entry,
				entry);

	/* Make sure we have only 1 FI */
	do {
		cr_assert(!strcmp(cxit_fi->fabric_attr->prov_name,
				  cxi_prov_name));

		cr_assert(cxit_fi->src_addr);
		addr = (struct cxi_addr *)cxit_fi->src_addr;
		cr_assert(addr->nic == if_entry->if_nic);
		cr_assert(addr->domain == 0);
		cr_assert(addr->port == 0);

		fab_name = get_fab_name(if_entry->if_fabric);
		cr_assert(!strcmp(cxit_fi->fabric_attr->name,
				  fab_name));
		free(fab_name);

		dom_name = get_dom_name(if_entry->if_idx, 0);
		cr_assert(!strcmp(cxit_fi->domain_attr->name,
				  dom_name));
		free(dom_name);

		cr_assert(!strcmp(cxit_fi->fabric_attr->prov_name,
				  cxi_prov_name));

		cr_assert(cxit_fi->dest_addr);
		addr = (struct cxi_addr *)cxit_fi->dest_addr;
		cr_assert(addr->nic == nic_id);
		cr_assert(addr->domain == 0);
		cr_assert(addr->port == 0);

		infos++;
	} while ((cxit_fi = cxit_fi->next));

	cr_assert(infos == 1);

	cxit_destroy_fabric_info();
	free(cxit_node);
}

/* Select fabric by node and service */
Test(getinfo, dest_node_service)
{
	int ret, infos = 0;
	struct cxi_addr *addr;
	struct cxi_if_list_entry *if_entry;
	char *fab_name, *dom_name;
	int nic_id = 130, dom_id = 5, port_id = 6;

	cxit_fi_hints->fabric_attr->prov_name = strdup(cxi_prov_name);

	infos = 0;

	ret = asprintf(&cxit_node, "0x%x", nic_id);
	cr_assert(ret > 0);

	ret = asprintf(&cxit_service, "%d:%d", dom_id, port_id);
	cr_assert(ret > 0);

	cxit_create_fabric_info();
	cr_assert(cxit_fi != NULL);

	/* All FIs use the first available src address.  All FI data
	 * will match the first interface found.  Additionally, node
	 * and service are used to create a dest addr.
	 */
	if_entry = container_of((cxi_if_list.head), struct cxi_if_list_entry,
				entry);

	/* Make sure we have only 1 FI */
	do {
		cr_assert(!strcmp(cxit_fi->fabric_attr->prov_name,
				  cxi_prov_name));

		cr_assert(cxit_fi->src_addr);
		addr = (struct cxi_addr *)cxit_fi->src_addr;
		cr_assert(addr->nic == if_entry->if_nic);
		cr_assert(addr->domain == 0);
		cr_assert(addr->port == 0);

		fab_name = get_fab_name(if_entry->if_fabric);
		cr_assert(!strcmp(cxit_fi->fabric_attr->name,
				  fab_name));
		free(fab_name);

		dom_name = get_dom_name(if_entry->if_idx, 0);
		cr_assert(!strcmp(cxit_fi->domain_attr->name,
				  dom_name));
		free(dom_name);

		cr_assert(!strcmp(cxit_fi->fabric_attr->prov_name,
				  cxi_prov_name));

		cr_assert(cxit_fi->dest_addr);
		addr = (struct cxi_addr *)cxit_fi->dest_addr;
		cr_assert(addr->nic == nic_id);
		cr_assert(addr->domain == dom_id);
		cr_assert(addr->port == port_id);

		infos++;
	} while ((cxit_fi = cxit_fi->next));
	cr_assert(infos == 1);

	cxit_destroy_fabric_info();
	free(cxit_service);
	free(cxit_node);
}

/* Select fabric by service.  FI_SOURCE is effectively ignored. */
Test(getinfo, service)
{
	int ret, infos = 0;
	struct cxi_addr *addr;
	struct cxi_if_list_entry *if_entry;
	struct slist_entry *entry, *prev;
	int dom, port;
	char *fab_name, *dom_name;

	cxit_fi_hints->fabric_attr->prov_name = strdup(cxi_prov_name);

	dom = 6;
	port = 7;
	ret = asprintf(&cxit_service, "%d:%d", dom, port);
	cr_assert(ret > 0);

	cxit_flags = FI_SOURCE;

	cxit_create_fabric_info();
	cr_assert(cxit_fi != NULL);

	/* Make sure we have only 1 FI for each IF */
	do {
		cr_assert(!strcmp(cxit_fi->fabric_attr->prov_name,
				  cxi_prov_name));

		(void) prev; /* Makes compiler happy */
		slist_foreach(&cxi_if_list, entry, prev) {
			if_entry = container_of(entry,
					struct cxi_if_list_entry, entry);

			cr_assert(cxit_fi->src_addr);
			addr = (struct cxi_addr *)cxit_fi->src_addr;
			if (addr->nic != if_entry->if_nic)
				continue;

			cr_assert(addr->nic == if_entry->if_nic);
			cr_assert(addr->domain == dom);
			cr_assert(addr->port == port);

			fab_name = get_fab_name(if_entry->if_fabric);
			cr_assert(!strcmp(cxit_fi->fabric_attr->name,
					  fab_name));
			free(fab_name);

			dom_name = get_dom_name(if_entry->if_idx, dom);
			cr_assert(!strcmp(cxit_fi->domain_attr->name,
					  dom_name));
			free(dom_name);

			cr_assert(!strcmp(cxit_fi->fabric_attr->prov_name,
						cxi_prov_name));

			infos++;
			break;
		}
	} while ((cxit_fi = cxit_fi->next));
	cr_assert(infos == (cxit_n_ifs));

	cxit_destroy_fabric_info();
	free(cxit_service);
}

TestSuite(fabric, .init = cxit_setup_fabric, .fini = cxit_teardown_fabric);

/* Test basic fabric creation */
Test(fabric, simple)
{
	cxit_create_fabric();
	cr_assert(cxit_fabric != NULL);

	cxit_destroy_fabric();
}

