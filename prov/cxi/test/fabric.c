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
char *node = NULL, *service = NULL;
uint64_t flags = 0;
int n_ifs;

static void create_fabric_info(void)
{
	int ret;

	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION),
			 node, service, flags, fi_hints, &fi);
	cr_assert(ret == FI_SUCCESS, "fi_getinfo");
}

static void destroy_fabric_info(void)
{
	fi_freeinfo(fi);
}

static void create_fabric(void)
{
	int ret;

	create_fabric_info();

	ret = fi_fabric(fi->fabric_attr, &fabric, NULL);
	cr_assert(ret == FI_SUCCESS, "fi_fabric");
}

static void destroy_fabric(void)
{
	int ret;

	ret = fi_close(&fabric->fid);
	cr_assert(ret == FI_SUCCESS, "fi_close fabric");

	destroy_fabric_info();
}

static void cxi_fabric_test_init(void)
{
	struct slist_entry *entry, *prev;

	/* init OFI providers */
	create_fabric();
	destroy_fabric();

	(void) prev; /* Makes compiler happy */
	slist_foreach(&cxi_if_list, entry, prev) {
		n_ifs++;
	}
}

static void setup_getinfo(void)
{
	cxi_fabric_test_init();

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
	fi_hints->fabric_attr->prov_name = strdup(cxi_prov_name);

	create_fabric();
}

static void teardown_fabric(void)
{
	destroy_fabric();
	teardown_getinfo();
}

static char *get_dom_name(int if_idx, int dom_id)
{
	char *dom = malloc(100);
	cr_assert(dom);

	snprintf(dom, 100, cxi_dom_fmt, if_idx, dom_id);

	return dom;
}

static char *get_fab_name(int fab_id)
{
	char *fab = malloc(100);
	cr_assert(fab);

	snprintf(fab, 100, cxi_fab_fmt, fab_id);

	return fab;
}

TestSuite(getinfo, .init = setup_getinfo,
	  .fini = teardown_getinfo);

/* Test fabric selection with provider name */
Test(getinfo, prov_name)
{
	int infos = 0;

	fi_hints->fabric_attr->prov_name = strdup(cxi_prov_name);

	create_fabric_info();
	cr_assert(fi != NULL);

	/* Make sure we have 1 FI for each IF */
	do {
		cr_assert(!strcmp(fi->fabric_attr->prov_name, cxi_prov_name));
		infos++;
	} while ((fi = fi->next));
	cr_assert(infos == n_ifs);
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

		fi_hints->domain_attr->name = get_dom_name(if_entry->if_idx, 0);

		create_fabric_info();
		cr_assert(fi != NULL);

		/* Make sure we have 1 FI for each IF */
		do {
			cr_assert(!strcmp(fi->domain_attr->name,
					  fi_hints->domain_attr->name));

			cr_assert(!strcmp(fi->fabric_attr->prov_name,
					  cxi_prov_name));

			fab_name = get_fab_name(if_entry->if_fabric);
			cr_assert(!strcmp(fi->fabric_attr->name, fab_name));
			free(fab_name);

			infos++;
		} while ((fi = fi->next));
		cr_assert(infos == 1);

		destroy_fabric_info();
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

		fi_hints->fabric_attr->name = get_fab_name(if_entry->if_fabric);

		create_fabric_info();
		cr_assert(fi != NULL);

		do {
			/* Not all providers can be trusted to filter by fabric
			 * name */
			if (strcmp(fi->fabric_attr->prov_name, cxi_prov_name))
				continue;

			cr_assert(!strcmp(fi->fabric_attr->name,
					  fi_hints->fabric_attr->name));

			infos++;
		} while ((fi = fi->next));

		destroy_fabric_info();
	}
	cr_assert(infos);
}

/* Test selection by source node */
Test(getinfo, src_node)
{
	int infos = 0;
	struct cxi_addr *addr;
	struct cxi_if_list_entry *if_entry;
	struct slist_entry *entry, *prev;
	char *fab_name, *dom_name;

	fi_hints->fabric_attr->prov_name = strdup(cxi_prov_name);

	(void) prev; /* Makes compiler happy */
	slist_foreach(&cxi_if_list, entry, prev) {
		if_entry = container_of(entry, struct cxi_if_list_entry, entry);
		infos = 0;

		node = malloc(10);
		cr_assert(node);
		snprintf(node, 10, "0x%x", if_entry->if_nic);
		flags = FI_SOURCE;

		create_fabric_info();
		cr_assert(fi != NULL);

		/* Make sure we have only 1 FI for each IF */
		do {
			cr_assert(!strcmp(fi->fabric_attr->prov_name,
					  cxi_prov_name));

			addr = (struct cxi_addr *)fi->src_addr;
			cr_assert(addr->nic == if_entry->if_nic);
			cr_assert(addr->domain == 0);
			cr_assert(addr->port == 0);

			fab_name = get_fab_name(if_entry->if_fabric);
			cr_assert(!strcmp(fi->fabric_attr->name, fab_name));
			free(fab_name);

			dom_name = get_dom_name(if_entry->if_idx, 0);
			cr_assert(!strcmp(fi->domain_attr->name, dom_name));
			free(dom_name);

			cr_assert(!strcmp(fi->fabric_attr->prov_name,
					  cxi_prov_name));
			infos++;
		} while ((fi = fi->next));
		cr_assert(infos == 1);

		destroy_fabric_info();
		free(node);
	}
	cr_assert(infos == 1);
}

/* Test selection by source node and service */
Test(getinfo, src_node_service)
{
	int infos = 0;
	struct cxi_addr *addr;
	struct cxi_if_list_entry *if_entry;
	struct slist_entry *entry, *prev;
	int dom, port;
	char *fab_name, *dom_name;

	fi_hints->fabric_attr->prov_name = strdup(cxi_prov_name);

	(void) prev; /* Makes compiler happy */
	slist_foreach(&cxi_if_list, entry, prev) {
		if_entry = container_of(entry, struct cxi_if_list_entry, entry);
		infos = 0;

		node = malloc(10);
		cr_assert(node);
		snprintf(node, 10, "0x%x", if_entry->if_nic);

		service = malloc(10);
		cr_assert(service);
		dom = if_entry->if_idx+4;
		port = if_entry->if_idx+5;
		snprintf(service, 10, "%d:%d", dom, port);

		flags = FI_SOURCE;

		create_fabric_info();
		cr_assert(fi != NULL);

		/* Make sure we have only 1 FI for each IF */
		do {
			cr_assert(!strcmp(fi->fabric_attr->prov_name,
					  cxi_prov_name));

			addr = (struct cxi_addr *)fi->src_addr;
			cr_assert(addr->nic == if_entry->if_nic);
			cr_assert(addr->domain == dom);
			cr_assert(addr->port == port);

			fab_name = get_fab_name(if_entry->if_fabric);
			cr_assert(!strcmp(fi->fabric_attr->name, fab_name));
			free(fab_name);

			dom_name = get_dom_name(if_entry->if_idx, dom);
			cr_assert(!strcmp(fi->domain_attr->name, dom_name));
			free(dom_name);

			cr_assert(!strcmp(fi->fabric_attr->prov_name,
					  cxi_prov_name));
			infos++;
		} while ((fi = fi->next));
		cr_assert(infos == 1);

		destroy_fabric_info();
		free(service);
		free(node);
	}
	cr_assert(infos == 1);
}

/* Select fabric by service */
Test(getinfo, src_service)
{
	int infos = 0;
	struct cxi_addr *addr;
	struct cxi_if_list_entry *if_entry;
	struct slist_entry *entry, *prev;
	int dom, port;
	char *fab_name, *dom_name;

	fi_hints->fabric_attr->prov_name = strdup(cxi_prov_name);

	service = malloc(10);
	cr_assert(service);
	dom = 6;
	port = 7;
	snprintf(service, 10, "%d:%d", dom, port);

	flags = FI_SOURCE;

	create_fabric_info();
	cr_assert(fi != NULL);

	/* Make sure we have only 1 FI for each IF */
	do {
		cr_assert(!strcmp(fi->fabric_attr->prov_name,
				  cxi_prov_name));

		(void) prev; /* Makes compiler happy */
		slist_foreach(&cxi_if_list, entry, prev) {
			if_entry = container_of(entry, struct cxi_if_list_entry, entry);

			addr = (struct cxi_addr *)fi->src_addr;
			if (addr->nic != if_entry->if_nic)
				continue;

			cr_assert(addr->nic == if_entry->if_nic);
			cr_assert(addr->domain == dom);
			cr_assert(addr->port == port);

			fab_name = get_fab_name(if_entry->if_fabric);
			cr_assert(!strcmp(fi->fabric_attr->name, fab_name));
			free(fab_name);

			dom_name = get_dom_name(if_entry->if_idx, dom);
			cr_assert(!strcmp(fi->domain_attr->name, dom_name));
			free(dom_name);

			cr_assert(!strcmp(fi->fabric_attr->prov_name,
						cxi_prov_name));

			infos++;
			break;
		}
	} while ((fi = fi->next));
	cr_assert(infos == n_ifs);

	destroy_fabric_info();
	free(service);
}

Test(getinfo, dest_node, .disabled = true)
{
}

Test(getinfo, dest_node_service, .disabled = true)
{
}

Test(getinfo, dest_service, .disabled = true)
{
}

TestSuite(fabric, .init = setup_fabric, .fini = teardown_fabric);

/* Test basic fabric creation */
Test(fabric, simple)
{
	cr_assert(fabric != NULL);
}

