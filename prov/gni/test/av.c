/*
 * Copyright (c) 2015 Los Alamos National Security, LLC. All rights reserved.
 * Copyright (c) 2015-2016 Cray Inc.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>

#include "fi.h"
#include "rdma/fi_domain.h"
#include "rdma/fi_prov.h"

#include "gnix.h"

#include <criterion/criterion.h>

static struct fid_fabric *fab;
static struct fid_domain *dom;
static struct fi_info *hints;
static struct fi_info *fi;
struct gnix_ep_name *fake_names;
static struct fid_av *av;
static struct gnix_fid_av *gnix_av;


#define SIMPLE_EP_ENTRY(id) \
{	\
	.gnix_addr = { \
			.device_addr = id, \
			.cdm_id = id, \
	}, \
	.name_type = id, \
	.cm_nic_cdm_id = id, \
	.cookie = id, \
}

#define SIMPLE_ADDR_COUNT 16
static struct gnix_ep_name simple_ep_names[SIMPLE_ADDR_COUNT] = {
		SIMPLE_EP_ENTRY(1),
		SIMPLE_EP_ENTRY(2),
		SIMPLE_EP_ENTRY(3),
		SIMPLE_EP_ENTRY(4),
		SIMPLE_EP_ENTRY(5),
		SIMPLE_EP_ENTRY(6),
		SIMPLE_EP_ENTRY(7),
		SIMPLE_EP_ENTRY(8),
		SIMPLE_EP_ENTRY(9),
		SIMPLE_EP_ENTRY(10),
		SIMPLE_EP_ENTRY(11),
		SIMPLE_EP_ENTRY(12),
		SIMPLE_EP_ENTRY(13),
		SIMPLE_EP_ENTRY(14),
		SIMPLE_EP_ENTRY(15),
		SIMPLE_EP_ENTRY(16),
};

static void av_setup(void)
{
	int ret = 0;

	hints = fi_allocinfo();
	cr_assert(hints, "fi_allocinfo");

	hints->domain_attr->cq_data_size = 4;
	hints->mode = ~0;

	hints->fabric_attr->name = strdup("gni");

	ret = fi_getinfo(FI_VERSION(1, 0), NULL, 0, 0, hints, &fi);
	cr_assert_eq(ret, FI_SUCCESS, "fi_getinfo");

	ret = fi_fabric(fi->fabric_attr, &fab, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_fabric");

	ret = fi_domain(fab, fi, &dom, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_domain");

}

static void av_teardown(void)
{
	int ret = 0;

	ret = fi_close(&dom->fid);
	cr_assert_eq(ret, FI_SUCCESS, "failure in closing domain.");
	ret = fi_close(&fab->fid);
	cr_assert_eq(ret, FI_SUCCESS, "failure in closing fabric.");
	fi_freeinfo(fi);
	fi_freeinfo(hints);
}


static void av_full_map_setup(void)
{
	struct fi_av_attr av_table_attr = {
		.type = FI_AV_MAP,
		.count = 16,
	};
	int ret;

	av_setup();

	ret = fi_av_open(dom, &av_table_attr, &av, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "failed to open av");

	gnix_av = container_of(av, struct gnix_fid_av, av_fid);
}

static void av_full_map_teardown(void)
{
	int ret;

	ret = fi_close(&av->fid);
	cr_assert_eq(ret, FI_SUCCESS, "failed to close av");

	av_teardown();
}

static void av_full_table_setup(void)
{
	struct fi_av_attr av_table_attr = {
		.type = FI_AV_TABLE,
		.count = 16,
	};
	int ret;

	av_setup();

	ret = fi_av_open(dom, &av_table_attr, &av, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "failed to open av");

	gnix_av = container_of(av, struct gnix_fid_av, av_fid);
}

static void av_full_table_teardown(void)
{
	int ret;

	ret = fi_close(&av->fid);
	cr_assert_eq(ret, FI_SUCCESS, "failed to close av");

	av_teardown();
}

TestSuite(av_bare, .init = av_setup, .fini = av_teardown, .disabled = false);

TestSuite(av_full_map, .init = av_full_map_setup,
		.fini = av_full_map_teardown, .disabled = false);

TestSuite(av_full_table, .init = av_full_table_setup,
		.fini = av_full_table_teardown, .disabled = false);

static void invalid_addrlen_pointer_test(void)
{
	int ret;
	fi_addr_t address = 0xdeadbeef;
	void *addr = (void *) 0xb00fbabe;

	/* while the pointers to address and addr aren't valid, they are
	 * acceptable as stated by the manpage. This will only test for a
	 * proper return code from fi_av_lookup()
	 */
	ret = fi_av_lookup(av, address, addr, NULL);
	cr_assert_eq(ret, -FI_EINVAL);
}

Test(av_full_map, invalid_addrlen_pointer)
{
	invalid_addrlen_pointer_test();
}

Test(av_full_table, invalid_addrlen_pointer)
{
	invalid_addrlen_pointer_test();
}

static void remove_addr_test(void)
{
	int ret;
	int i;
	fi_addr_t addresses[SIMPLE_ADDR_COUNT];
	fi_addr_t *compare;

	/* insert addresses */
	ret = fi_av_insert(av, (void *) simple_ep_names, SIMPLE_ADDR_COUNT, addresses,
			0, NULL);
	cr_assert_eq(ret, SIMPLE_ADDR_COUNT);

	/* check address contents */
	for (i = 0; i < SIMPLE_ADDR_COUNT; i++) {
		if (gnix_av->type == FI_AV_MAP) {
			compare = (fi_addr_t *) &simple_ep_names[i].gnix_addr;
			cr_assert_eq(*compare, addresses[i]);
		} else {
			cr_assert_eq(i, addresses[i]);
		}
	}

	/* remove addresses */
	ret = fi_av_remove(av, addresses, SIMPLE_ADDR_COUNT, 0);
	cr_assert_eq(ret, FI_SUCCESS);
}

Test(av_full_map, remove_addr)
{
	remove_addr_test();
}

Test(av_full_table, remove_addr)
{
	remove_addr_test();
}

static void lookup_invalid_test(void)
{
	int ret;
	fi_addr_t addresses[SIMPLE_ADDR_COUNT];
	size_t addrlen = sizeof(struct gnix_ep_name);

	/* test null addrlen */
	ret = fi_av_lookup(av, 0xdeadbeef, (void *) 0xdeadbeef, NULL);
	cr_assert_eq(ret, -FI_EINVAL);

	/* test null addr */
	ret = fi_av_lookup(av, 0xdeadbeef, NULL, &addrlen);
	cr_assert_eq(ret, -FI_EINVAL);

	/* test invalid lookup */
	if (gnix_av->type == FI_AV_TABLE) {
		ret = fi_av_lookup(av, 2000, &addresses[0], &addrlen);
		cr_assert_eq(ret, -FI_EINVAL);

		/* test within range, but not inserted case */
		ret = fi_av_lookup(av, 1, &addresses[1], &addrlen);
		cr_assert_eq(ret, -FI_EINVAL);
	} else {
		ret = fi_av_lookup(av, 0xdeadbeef, &addresses[0], &addrlen);
		cr_assert_eq(ret, -FI_ENOENT);
	}
}

Test(av_full_map, lookup_invalid)
{
	lookup_invalid_test();
}

Test(av_full_table, lookup_invalid)
{
	lookup_invalid_test();
}

static void lookup_test(void)
{
	int ret;
	int i;
	fi_addr_t addresses[SIMPLE_ADDR_COUNT];
	fi_addr_t *compare;
	fi_addr_t found;
	size_t addrlen = sizeof(struct gnix_ep_name);

	/* insert addresses */
	ret = fi_av_insert(av, (void *) simple_ep_names, SIMPLE_ADDR_COUNT,
			addresses, 0, NULL);
	cr_assert_eq(ret, SIMPLE_ADDR_COUNT);

	/* check address contents */
	for (i = 0; i < SIMPLE_ADDR_COUNT; i++) {
		if (gnix_av->type == FI_AV_MAP) {
			compare = (fi_addr_t *) &simple_ep_names[i].gnix_addr;
			cr_assert_eq(*compare, addresses[i]);
		} else {
			cr_assert_eq(i, addresses[i]);
		}
	}

	ret = fi_av_lookup(av, addresses[1], &found, &addrlen);
	cr_assert_eq(ret, FI_SUCCESS);
}

Test(av_full_map, lookup)
{
	lookup_test();
}

Test(av_full_table, lookup)
{
	lookup_test();
}

static void straddr_test(void)
{
	int ret;
	int i;
	const char *buf;
	char address[128];
	fi_addr_t addresses[SIMPLE_ADDR_COUNT];
	fi_addr_t *compare;
	size_t addrlen = sizeof(struct gnix_ep_name);

	/* insert addresses */
	ret = fi_av_insert(av, (void *) simple_ep_names, SIMPLE_ADDR_COUNT,
			addresses, 0, NULL);
	cr_assert_eq(ret, SIMPLE_ADDR_COUNT);

	/* check address contents */
	for (i = 0; i < SIMPLE_ADDR_COUNT; i++) {
		if (gnix_av->type == FI_AV_MAP) {
			compare = (fi_addr_t *) &simple_ep_names[i].gnix_addr;
			cr_assert_eq(*compare, addresses[i]);
		} else {
			cr_assert_eq(i, addresses[i]);
		}
	}

	buf = fi_av_straddr(av, &simple_ep_names[0], address, &addrlen);
	cr_assert_eq(buf, address);
}

Test(av_full_map, straddr)
{
	straddr_test();
}

Test(av_full_table, straddr)
{
	straddr_test();
}

#define TABLE_SIZE_INIT  16
#define TABLE_SIZE_FINAL 1024

Test(av_bare, test_capacity)
{
	int ret, i;
	fi_addr_t addresses[TABLE_SIZE_FINAL];
	struct fi_av_attr av_table_attr = {
		.type = FI_AV_TABLE,
		.count = TABLE_SIZE_INIT,
	};

	ret = fi_av_open(dom, &av_table_attr, &av, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "failed to open av");

	fake_names = (struct gnix_ep_name *)calloc(TABLE_SIZE_FINAL,
						   sizeof(*fake_names));
	cr_assert_neq(fake_names, NULL);

	for (i = 0; i < TABLE_SIZE_INIT; i++) {
		fake_names[i].gnix_addr.device_addr = i + 100;
		fake_names[i].gnix_addr.cdm_id = i;
		fake_names[i].cm_nic_cdm_id = 0xbeef;
		fake_names[i].cookie = 0xdeadbeef;
	}

	ret = fi_av_insert(av, fake_names, TABLE_SIZE_INIT,
			   addresses, 0, NULL);
	cr_assert_eq(ret, TABLE_SIZE_INIT, "av insert failed");

	/*
	 * now add some more
	 */

	for (i = TABLE_SIZE_INIT; i < TABLE_SIZE_FINAL; i++) {
		fake_names[i].gnix_addr.device_addr = i + 100;
		fake_names[i].gnix_addr.cdm_id = i;
		fake_names[i].cm_nic_cdm_id = 0xbeef;
		fake_names[i].cookie = 0xdeadbeef;
	}

	ret = fi_av_insert(av, &fake_names[TABLE_SIZE_INIT],
			   TABLE_SIZE_FINAL - TABLE_SIZE_INIT,
			   &addresses[TABLE_SIZE_INIT], 0, NULL);
	cr_assert_eq(ret, TABLE_SIZE_FINAL - TABLE_SIZE_INIT,
		     "av insert failed");

}
