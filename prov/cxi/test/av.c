/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2015-2018 Cray Inc. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <netinet/ether.h>

#include <criterion/criterion.h>

#include "cxip.h"
#include "cxip_test_common.h"

static struct cxip_addr *test_addrs;
fi_addr_t *test_fi_addrs;
#define AV_COUNT 1024
int naddrs = AV_COUNT * 10;

static char *nic_to_amac(uint32_t nic)
{
	struct ether_addr mac = {};

	mac.ether_addr_octet[5] = nic;
	mac.ether_addr_octet[4] = nic >> 8;
	mac.ether_addr_octet[3] = nic >> 16;

	return ether_ntoa(&mac);
}

/* This allocates memory for naddrs FSAs (test_addrs), and naddrs tokens
 * (test_fi_addrs), and initializes the FSAs to unique addresses.
 */
static void
test_addrs_init(void)
{
	int i;

	test_addrs = malloc(naddrs * sizeof(struct cxip_addr));
	cr_assert(test_addrs != NULL);

	test_fi_addrs = calloc(naddrs, sizeof(fi_addr_t));
	cr_assert(test_fi_addrs != NULL);

	for (i = 0; i < naddrs; i++) {
		test_addrs[i].nic = i;
		test_addrs[i].pid = i + 1;
	}
}

/* Clean up the FSA and token memory.
 */
static void
test_addrs_fini(void)
{
	free(test_fi_addrs);
	free(test_addrs);
}

/* This creates an AV with 'count' objects, and peeks at internals to ensure
 * that the structure is sound. If 'count' is 0, this should default to
 * cxip_av_dev_sz.
 */
static void
test_create(size_t count)
{
	struct cxip_av *av;
	size_t expect;
	int i;

	expect = (count) ? count : cxip_av_def_sz;
	cxit_av_attr.count = count;
	cxit_create_av();

	/* Should return the actual count   */
	cr_assert(cxit_av_attr.count == expect,
		"attr count=%ld, expect=%ld", cxit_av_attr.count, expect);
	/* Should allocate a structure   */
	cr_assert(cxit_av != NULL,
		"cxit_av=%p", cxit_av);

	/* Check the table memory   */
	av = container_of(&cxit_av->fid, struct cxip_av, av_fid.fid);
	cr_assert(av->table_hdr != NULL,
		"av->table_hdr=%p", av->table_hdr);
	cr_assert(av->table != NULL,
		"av->table=%p", av->table);
	cr_assert(av->table_hdr->size == expect,
		"av->table_hdr->size=%ld, expect %ld",
		av->table_hdr->size, expect);

	if (cxit_av_attr.name) {
		/* Named AVs should allocate shared memory   */
		cr_assert(av->idx_arr != NULL,
			"av->idx_arr=%p, expect ptr", av->idx_arr);
		/* Not read-only, so this should not return this pointer   */
		cr_assert(cxit_av_attr.map_addr == NULL,
			"map_addr=%p", cxit_av_attr.map_addr);
		/* All of the indices should be -1   */
		for (i = 0; i < av->table_hdr->size; i++) {
			cr_assert(av->idx_arr[i] == (uint64_t)-1,
				"av->idx_arr[%d]=%ld", i, av->idx_arr[i]);
		}
	} else {
		/* Unnamed AV should not allocate shared memory */
		cr_assert(av->idx_arr == NULL,
			"av->idx_arr=%p, expect (nil)", av->idx_arr);
	}

	cxit_destroy_av();
}

/* This inserts 'count' FSAs, looks up all of them, then removes all of them. It
 * repeats this 'iters' times without destroying the AV.
 */
static void
__test_insert(int count, int iters)
{
	int j, i, ret;
	struct cxip_addr addr;
	size_t addrlen;

	/* Can't test addresses we haven't set up   */
	cr_assert(naddrs >= count, "Invalid test case");

	cxit_create_av();
	test_addrs_init();

	for (j = 0; j < iters; j++) {
		/* Insert addresses   */
		for (i = 0; i < count; i++) {
			ret = fi_av_insert(cxit_av, &test_addrs[i], 1,
				&test_fi_addrs[i], 0, NULL);
			/* Should have inserted 1 item   */
			cr_assert(ret == 1,
				"fi_av_insert() iter=%d, idx=%d, ret=%d\n",
				j, i, ret);
			/* Returned tokens should match insertion order   */
			cr_assert(test_fi_addrs[i] == i,
				"fi_av_insert() iter=%d, idx=%d, index=%ld\n",
				j, i, test_fi_addrs[i]);
		}

		/* Lookup addresses   */
		for (i = 0; i < count; i++) {
			addrlen = sizeof(struct cxip_addr);
			ret = fi_av_lookup(cxit_av, test_fi_addrs[i], &addr,
				&addrlen);
			/* Should succeed   */
			cr_assert(ret == FI_SUCCESS,
				"fi_av_lookup() iter=%d, idx=%d, ret=%d",
				j, i, ret);
			/* Address should match what we expect   */
			cr_assert(addr.nic == test_addrs[i].nic,
				"fi_av_lookup() iter=%d, count=%d, i=%d, index=%ld, nic=%d, exp=%d",
				j, count, i, test_fi_addrs[i], addr.nic,
				test_addrs[i].nic);
			cr_assert(addr.pid == test_addrs[i].pid,
				"fi_av_lookup() iter=%d, idx=%d, pid=%d",
				j, i, addr.pid);
		}

		/* Spot-check. If we remove an arbitrary entry, and then insert
		 * a new address, it should always fill the hole left by the
		 * removal.
		 */

		/* Remove an arbitrary item in the middle   */
		i = count / 2;
		ret = fi_av_remove(cxit_av, &test_fi_addrs[i], 1, 0);
		cr_assert(ret == FI_SUCCESS,
			"fi_av_remove() mid iter=%d, idx=%d, ret=%d\n",
			j, i, ret);

		/* Make sure that lookup fails   */
		addrlen = sizeof(struct cxip_addr);
		ret = fi_av_lookup(cxit_av, test_fi_addrs[i], &addr, &addrlen);
		cr_assert(ret == -FI_EINVAL,
			"fi_av_lookup() mid iter=%d, idx=%d, ret=%d\n",
			j, i, ret);

		/* Insert an address   */
		ret = fi_av_insert(cxit_av, &test_addrs[i], 1,
			&test_fi_addrs[i], 0, NULL);
		cr_assert(ret == 1,
			"fi_av_insert() mid iter=%d, idx=%d, ret=%d\n",
			j, i, ret);
		cr_assert(test_fi_addrs[i] == i,
			"fi_av_insert() mid iter=%d, idx=%d, index=%ld\n",
			j, i, test_fi_addrs[i]);

		addrlen = sizeof(struct cxip_addr);
		ret = fi_av_lookup(cxit_av, test_fi_addrs[i], &addr,
			&addrlen);
		cr_assert(ret == FI_SUCCESS,
			"fi_av_lookup() mid iter=%d, idx=%d, ret=%d",
			j, i, ret);
		cr_assert(addr.nic == test_addrs[i].nic,
			"fi_av_lookup() mid iter=%d, count=%d, i=%d, index=%ld, nic=%d, exp=%d",
			j, count, i, test_fi_addrs[i], addr.nic,
			test_addrs[i].nic);
		cr_assert(addr.pid == test_addrs[i].pid,
			"fi_av_lookup() mid iter=%d, idx=%d, pid=%d",
			j, i, addr.pid);

		/* Remove all of the entries   */
		for (i = 0; i < count; i++) {
			ret = fi_av_remove(cxit_av, &test_fi_addrs[i], 1, 0);
			/* Should succeed   */
			cr_assert(ret == 0,
				"fi_av_remove() iter=%d, idx=%d, ret=%d",
				j, i, ret);
		}
	}

	test_addrs_fini();
	cxit_destroy_av();
}

/* Wrapper for insert test.
 *
 * The first call in each group only fills half of the initially allocated
 * space.
 *
 * The second call fills the entire initially allocated space.
 *
 * The third call requires multiple memory reallocations to expand the memory as
 * this inserts.
 */
static void
test_insert(void)
{
	int iters = 1;

	__test_insert(AV_COUNT / 2, iters);
	__test_insert(AV_COUNT, iters);
	__test_insert(naddrs, iters);

	iters = 3;

	__test_insert(AV_COUNT / 2, iters);
	__test_insert(AV_COUNT, iters);
	__test_insert(naddrs, iters);
}

/* Opening bracket for a read-only test. This will open a read-only AV, and
 * return information in attrRO and avRO. There must be a read-write AV open
 * when this is called.
 */
static void
shmem_readonly_init(const char *name, struct fi_av_attr *attrRO,
	struct fid_av **avRO)
{
	int ret;

	memset(attrRO, 0, sizeof(*attrRO));

	attrRO->count = 0;
	attrRO->type = FI_AV_MAP;
	attrRO->name = name;
	attrRO->flags |= FI_READ;

	*avRO = NULL;

	ret = fi_av_open(cxit_domain, attrRO, avRO, NULL);
	/* This should succeed   */
	cr_assert(ret == FI_SUCCESS,
		"fi_av_open RO");
	/* Read-only open should return a map   */
	cr_assert(attrRO->map_addr != NULL,
		"fi_av_open RO no map");
	/* Read-only count should have read the size   */
	cr_assert(attrRO->count != 0,
		"fi_av_open RO zero count");
}

static void
shmem_readonly_term(struct fid_av *avRO)
{
	int ret;

	ret = fi_close(&avRO->fid);
	cr_assert(ret == FI_SUCCESS, "fi_close RO");
}

TestSuite(av, .init = cxit_setup_av, .fini = cxit_teardown_av,
	  .timeout = CXIT_DEFAULT_TIMEOUT);

ReportHook(TEST_CRASH)(struct criterion_test_stats *stats)
{
	printf("signal = %d\n", stats->signal);
}

/* Test AV creation syntax error */
Test(av, av_open_invalid)
{
	int ret;

	ret = fi_av_open(cxit_domain, NULL, NULL, NULL);
	cr_assert(ret == -FI_EINVAL, "fi_av_open AV all NULL = %d", ret);

	ret = fi_av_open(cxit_domain, &cxit_av_attr, NULL, NULL);
	cr_assert(ret == -FI_EINVAL, "fi_av_open AV NULL av = %d", ret);

	ret = fi_av_open(cxit_domain, NULL, &cxit_av, NULL);
	cr_assert(ret == -FI_EINVAL, "fi_av_open AV NULL av_attr = %d", ret);

	cxit_av_attr.type = 99;
	ret = fi_av_open(cxit_domain, &cxit_av_attr, &cxit_av, NULL);
	cr_assert(ret == -FI_EINVAL, "fi_av_open AV bad type = %d", ret);
	cxit_av_attr.type = 0;

	/* NOTE: FI_READ means read-only */
	cxit_av_attr.flags = FI_READ;
	ret = fi_av_open(cxit_domain, &cxit_av_attr, &cxit_av, NULL);
	cr_assert(ret == -FI_EINVAL, "fi_av_open AV FI_READ with no name = %d",
		ret);
	cxit_av_attr.flags = 0;

	cxit_av_attr.rx_ctx_bits = CXIP_EP_MAX_CTX_BITS + 1;
	ret = fi_av_open(cxit_domain, &cxit_av_attr, &cxit_av, NULL);
	cr_assert(ret == -FI_EINVAL, "fi_av_open AV too many bits = %d", ret);
	cxit_av_attr.rx_ctx_bits = 0;
}

/* Test AV bind not supported */
Test(av, av_bind_invalid)
{
	int ret;

	cxit_create_av();

	ret = fi_av_bind(cxit_av, NULL, 0);
	cr_assert(ret == -FI_ENOSYS, "fi_av_bind() = %d", ret);

	cxit_destroy_av();
}

/* Test AV control not supported */
Test(av, av_control_invalid)
{
	int ret;

	cxit_create_av();

	ret = fi_control(&cxit_av->fid, 0, NULL);
	cr_assert(ret == -FI_ENOSYS, "fi_control() = %d", ret);

	cxit_destroy_av();
}

/* Test AV open_ops not supported */
Test(av, av_open_ops_invalid)
{
	int ret;

	cxit_create_av();

	ret = fi_open_ops(&cxit_av->fid, NULL, 0, NULL, NULL);
	cr_assert(ret == -FI_ENOSYS, "fi_open_ops() = %d", ret);

	cxit_destroy_av();
}

/* Test basic AV table creation */
Test(av, table_create)
{
	cxit_av_attr.type = FI_AV_TABLE;
	test_create(0);
	test_create(1024);
}

Test(av, shmem_table_create)
{
	char name[64];

	snprintf(name, sizeof(name), "/bogus%d", getpid());
	cxit_av_attr.type = FI_AV_TABLE;
	cxit_av_attr.name = name;
	test_create(0);
	test_create(1024);
	cxit_av_attr.name = NULL;
}

/* Test basic AV map creation */
Test(av, map_create)
{
	cxit_av_attr.type = FI_AV_MAP;
	test_create(0);
	test_create(1024);
}

Test(av, shmem_map_create)
{
	char name[64];

	snprintf(name, sizeof(name), "/bogus%d", getpid());
	cxit_av_attr.type = FI_AV_MAP;
	cxit_av_attr.name = name;
	test_create(0);
	test_create(1024);
	cxit_av_attr.name = NULL;
}

/* Test basic AV default creation */
Test(av, unspecified_create)
{
	cxit_av_attr.type = FI_AV_UNSPEC;
	test_create(0);
	test_create(1024);
}

Test(av, shmem_unspecified_create)
{
	char name[64];

	snprintf(name, sizeof(name), "/bogus%d", getpid());
	cxit_av_attr.type = FI_AV_UNSPEC;
	cxit_av_attr.name = name;
	test_create(0);
	test_create(1024);
	cxit_av_attr.name = NULL;
}

/* Test basic AV table insert */
Test(av, table_insert)
{
	cr_skip_test();
	cxit_av_attr.count = AV_COUNT;
	cxit_av_attr.type = FI_AV_TABLE;
	naddrs = cxit_av_attr.count * 10;

	test_insert();
}

Test(av, shmem_table_insert)
{
	char name[64];

	snprintf(name, sizeof(name), "/bogus%d", getpid());
	cxit_av_attr.count = AV_COUNT;
	cxit_av_attr.type = FI_AV_TABLE;
	cxit_av_attr.name = name;
	naddrs = cxit_av_attr.count * 10;

	test_insert();
}

/* Test basic AV map insert */
Test(av, map_insert)
{
	cr_skip_test();
	cxit_av_attr.count = AV_COUNT;
	cxit_av_attr.type = FI_AV_MAP;
	naddrs = cxit_av_attr.count * 10;

	test_insert();
}

Test(av, shmem_map_insert)
{
	char name[64];

	snprintf(name, sizeof(name), "/bogus%d", getpid());
	cxit_av_attr.count = AV_COUNT;
	cxit_av_attr.type = FI_AV_MAP;
	cxit_av_attr.name = name;
	naddrs = cxit_av_attr.count * 10;

	test_insert();
}

/* Test shared memory implementation */
Test(av, shmem_map_readonly)
{
	struct fi_av_attr attrRW, attrRO;
	struct fid_av *avRW, *avRO;
	char name[64];
	uint64_t *map;
	int ret, siz, i;

	test_addrs_init();

	memset(&attrRW, 0, sizeof(attrRW));
	memset(&avRW, 0, sizeof(avRW));

	/* Create a very small RW AV */
	snprintf(name, sizeof(name), "/bogus%d", getpid());
	attrRW.count = 32;
	attrRW.type = FI_AV_MAP;
	attrRW.name = name;

	ret = fi_av_open(cxit_domain, &attrRW, &avRW, NULL);
	cr_assert(ret == FI_SUCCESS, "fi_av_open RW");

	/* Make sure the map has invalid tokens */
	shmem_readonly_init(name, &attrRO, &avRO);
	do {
		map = (uint64_t *)attrRO.map_addr;
		for (i = 0; i < attrRO.count; i++) {
			cr_assert(map[i] == (uint64_t)-1,
				"map[%d]=%016lx\n", i, map[i]);
		}
	} while (0);
	shmem_readonly_term(avRO);


	/* Fill to increasing sizes */
	for (siz = 1; siz < 36; siz++) {
		/* Insert siz addresses   */
		ret = fi_av_insert(avRW, &test_addrs[0], siz,
			&test_fi_addrs[0], 0, NULL);
		cr_assert(ret == siz,
			"insertion failed siz=%d ret=%d", siz, ret);
		for (i = 0; i < siz; i++) {
			cr_assert(test_fi_addrs[i] == i,
				"bad index siz=%d idx=%i val=%ld",
				siz, i, test_fi_addrs[i]);
		}

		/* Look at the read-only map   */
		shmem_readonly_init(name, &attrRO, &avRO);
		do {
			map = (uint64_t *)attrRO.map_addr;
			/* There should be space in the map   */
			cr_assert(attrRO.count >= siz,
				"map too small, count=%ld, siz=%d\n",
				attrRO.count, siz);
			/* First 'siz' should be valid indices */
			for (i = 0; i < siz; i++) {
				cr_assert(map[i] == i,
					"map[%d]=%016lx\n", 0, map[0]);
			}
			/* Remainder should be invalid   */
			for (; i < attrRO.count; i++) {
				cr_assert(map[i] == (uint64_t)-1,
					"map[%d]=%016lx\n", i, map[i]);
			}
		} while (0);
		shmem_readonly_term(avRO);

		/* Remove all but the last address   */
		ret = fi_av_remove(avRW, &test_fi_addrs[0], siz - 1, 0);
		cr_assert(ret == FI_SUCCESS,
			"remove %d failed", siz - 1);

		shmem_readonly_init(name, &attrRO, &avRO);
		do {
			map = (uint64_t *)attrRO.map_addr;
			/* First siz-1 should be invalid   */
			for (i = 0; i < siz - 1; i++) {
				cr_assert(map[i] == (uint64_t)-1,
					"map[%d]=%016lx\n", i, map[i]);
			}
			/* Last item should be valid   */
			for (; i < siz; i++) {
				cr_assert(map[i] == i,
					"map[%d]=%016lx\n", 0, map[0]);
			}
			/* Remainder should be invalid   */
			for (; i < attrRO.count; i++) {
				cr_assert(map[i] == (uint64_t)-1,
					"map[%d]=%016lx\n", i, map[i]);
			}
		} while (0);
		shmem_readonly_term(avRO);

		/* Fill up again   */
		ret = fi_av_insert(avRW, &test_addrs[0], siz - 1,
			&test_fi_addrs[0], 0, NULL);
		cr_assert(ret == siz - 1,
			"reinsertion failed siz=%d ret=%d", siz, ret);
		/* These should have filled in-order   */
		for (i = 0; i < siz; i++) {
			cr_assert(test_fi_addrs[i] == i,
				"bad index siz=%d idx=%i val=%ld",
				siz, i, test_fi_addrs[i]);
		}

		/* Clean up for next pass   */
		ret = fi_av_remove(avRW, &test_fi_addrs[0], siz, 0);
		cr_assert(ret == FI_SUCCESS,
			"remove %d failed", siz);

	}

	ret = fi_close(&avRW->fid);
	cr_assert(ret == FI_SUCCESS, "fi_close RW");

	test_addrs_fini();
}

/* Test address conversion to string */
Test(av, straddr)
{
	uint64_t addr = 0xabcd1234abcd1234;
	size_t len = 0;
	char *buf = NULL;
	const char *tmp_buf;

	cxit_create_av();

	tmp_buf = fi_av_straddr(cxit_av, &addr, buf, &len);
	/* "fi_addr_cxi://0x0000000000000000\0" */
	cr_assert_eq(len, 33, "fi_av_straddr() len failure: %ld", len);
	cr_assert_null(tmp_buf, "fi_av_straddr() buffer not null %p", tmp_buf);

	buf = malloc(len);
	cr_assert(buf != NULL);

	tmp_buf = fi_av_straddr(cxit_av, &addr, buf, &len);
	cr_assert_eq(len, 33, "fi_av_straddr() len failure: %ld", len);
	cr_assert_not_null(tmp_buf, "fi_av_straddr() buffer is null");
	cr_assert_str_eq(tmp_buf, buf,
		"fi_av_straddr() buffer failure: '%s' != '%s'", tmp_buf, buf);

	free(buf);

	cxit_destroy_av();
}

Test(av, shmem_zero_size)
{
	struct util_shm shm;
	void *ptr;
	int ret;
	char name[64];

	ret = ofi_shm_map(&shm, name, 0, 0, &ptr);
	cr_assert(ret == -FI_EINVAL, "create shmem RW size 0 = %d", ret);
	ret = ofi_shm_map(&shm, name, 0, 1, &ptr);
	cr_assert(ret == -FI_EINVAL, "create shmem RO size 0 = %d", ret);
}

Test(av, insertsvc)
{
	int i, ret;
	struct cxip_addr addr;
	size_t addrlen;
	char pid_str[256];

	cxit_create_av();
	test_addrs_init();

	ret = fi_av_insertsvc(cxit_av, NULL, pid_str, &test_fi_addrs[0], 0,
			      NULL);
	cr_assert(ret == -FI_EINVAL);

	ret = fi_av_insertsvc(cxit_av, nic_to_amac(test_addrs[0].nic), NULL,
			      &test_fi_addrs[0], 0, NULL);
	cr_assert(ret == -FI_EINVAL);

	ret = fi_av_insertsvc(cxit_av, NULL, NULL, &test_fi_addrs[0], 0, NULL);
	cr_assert(ret == -FI_EINVAL);

	/* Insert addresses   */
	for (i = 0; i < naddrs; i++) {
		ret = sprintf(pid_str, "%d", test_addrs[i].pid);
		cr_assert(ret > 0);

		ret = fi_av_insertsvc(cxit_av, nic_to_amac(test_addrs[i].nic),
				      pid_str, &test_fi_addrs[i], 0, NULL);
		/* Should have inserted 1 item   */
		cr_assert(ret == 1,
			"fi_av_insertsvc() idx=%d, ret=%d\n",
			i, ret);
		/* Returned tokens should match insertion order   */
		cr_assert(test_fi_addrs[i] == i,
			"fi_av_insertsvc() idx=%d, fi_addr=%ld\n",
			i, test_fi_addrs[i]);
	}

	/* Lookup addresses   */
	for (i = 0; i < naddrs; i++) {
		addrlen = sizeof(struct cxip_addr);
		ret = fi_av_lookup(cxit_av, test_fi_addrs[i], &addr,
			&addrlen);
		/* Should succeed   */
		cr_assert(ret == FI_SUCCESS,
			"fi_av_lookup() idx=%d, ret=%d",
			i, ret);
		/* Address should match what we expect   */
		cr_assert(addr.nic == test_addrs[i].nic,
			"fi_av_lookup() naddrs=%d, i=%d, index=%ld, nic=%d, exp=%d",
			naddrs, i, test_fi_addrs[i], addr.nic,
			test_addrs[i].nic);
		cr_assert(addr.pid == test_addrs[i].pid,
			"fi_av_lookup() idx=%d, pid=%d",
			i, addr.pid);
	}

	/* Spot-check. If we remove an arbitrary entry, and then insert
	 * a new address, it should always fill the hole left by the
	 * removal.
	 */

	/* Remove an arbitrary item in the middle   */
	i = naddrs / 2;
	ret = fi_av_remove(cxit_av, &test_fi_addrs[i], 1, 0);
	cr_assert(ret == FI_SUCCESS,
		"fi_av_remove() mid idx=%d, ret=%d\n",
		i, ret);

	/* Make sure that lookup fails   */
	addrlen = sizeof(struct cxip_addr);
	ret = fi_av_lookup(cxit_av, test_fi_addrs[i], &addr, &addrlen);
	cr_assert(ret == -FI_EINVAL,
		"fi_av_lookup() mid idx=%d, ret=%d\n",
		i, ret);

	/* Insert an address   */
	ret = fi_av_insert(cxit_av, &test_addrs[i], 1,
		&test_fi_addrs[i], 0, NULL);
	cr_assert(ret == 1,
		"fi_av_insert() mid idx=%d, ret=%d\n",
		i, ret);
	cr_assert(test_fi_addrs[i] == i,
		"fi_av_insert() mid idx=%d, index=%ld\n",
		i, test_fi_addrs[i]);

	addrlen = sizeof(struct cxip_addr);
	ret = fi_av_lookup(cxit_av, test_fi_addrs[i], &addr,
		&addrlen);
	cr_assert(ret == FI_SUCCESS,
		"fi_av_lookup() mid idx=%d, ret=%d",
		i, ret);
	cr_assert(addr.nic == test_addrs[i].nic,
		"fi_av_lookup() mid naddrs=%d, i=%d, index=%ld, nic=%d, exp=%d",
		naddrs, i, test_fi_addrs[i], addr.nic,
		test_addrs[i].nic);
	cr_assert(addr.pid == test_addrs[i].pid,
		"fi_av_lookup() mid idx=%d, pid=%d",
		i, addr.pid);

	/* Remove all of the entries   */
	for (i = 0; i < naddrs; i++) {
		ret = fi_av_remove(cxit_av, &test_fi_addrs[i], 1, 0);
		/* Should succeed   */
		cr_assert(ret == 0,
			"fi_av_remove() idx=%d, ret=%d",
			i, ret);
	}

	test_addrs_fini();
	cxit_destroy_av();
}
