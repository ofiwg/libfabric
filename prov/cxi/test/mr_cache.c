/*
 * Copyright (c) 2024 Hewlett Packard Enterprise Development LP
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <ctype.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#include <pthread.h>

#include "libcxi/libcxi.h"
#include "cxip.h"
#include "cxip_test_common.h"

#define SETENV_OVERWRITE 1

TestSuite(mr_cache, .timeout = CXIT_DEFAULT_TIMEOUT);

Test(mr_cache, cache_full)
{
	static struct {
		const char *name;
		const char *value;
	} envs[] = {
		{ .name = "FI_MR_CACHE_MONITOR",   .value = "kdreg2", },
		{ .name = "FI_MR_CACHE_MAX_COUNT", .value = "4", },
	};
	struct {
		void *addr;
		struct fid_mr *mr;
	} *region_data;
	size_t i;
	int ret;
	long page_size;
	unsigned long num_regions, total_regions;
	struct ofi_mr_cache   *cache;
	struct cxip_domain    *cxip_dom;

	/* setup the environment */
	for (i = 0; i < ARRAY_SIZE(envs); i++) {
		ret = setenv(envs[i].name, envs[i].value, SETENV_OVERWRITE);
		cr_assert_eq(ret, 0, "Failed to set %s to %s: %d",
			     envs[i].name, envs[i].value, errno);
	}

	/* allocate the memory regions */
	page_size = sysconf(_SC_PAGESIZE);
	cr_assert(page_size > 0,
		  "sysconf(_SC_PAGESIZE) return %ld: errno = %d", page_size, errno);

	ret = sscanf(getenv("FI_MR_CACHE_MAX_COUNT"), "%lu", &num_regions);
	cr_assert_eq(ret, 1, "Failed to get number of regions: %d %d:%s",
		     ret, errno, strerror(errno));

	/* one extra to push one out of the cache */
	total_regions = num_regions + 1;
	region_data = calloc(total_regions, sizeof(*region_data));
	cr_assert_not_null(region_data);

	for (i = 0; i < total_regions; i++) {
		region_data[i].addr = mmap(NULL, page_size,
					   PROT_READ | PROT_WRITE,
					   MAP_ANONYMOUS | MAP_PRIVATE,
					   -1, 0);
		cr_assert_not_null(region_data[i].addr);
	}

	/* create the domain */
	cxit_setup_msg();

	/* Register the max number of regions */
	for (i = 0; i < num_regions; i++) {
		ret = fi_mr_reg(cxit_domain, region_data[i].addr,
				page_size, FI_READ | FI_WRITE,
				0, 0, 0, &region_data[i].mr, NULL);
		cr_assert_eq(ret, FI_SUCCESS,
			     "fi_mr_reg failed for region %lu: %d", i, ret);
	}

	/* See that the cache is full */
	cxip_dom = container_of(cxit_domain, struct cxip_domain,
				util_domain.domain_fid);
	cache = &cxip_dom->iomm;
	cr_assert(cache->cached_cnt == cache->cached_max_cnt,
		  "Cache is not full: %zu != %zu",
		  cache->cached_cnt, cache->cached_max_cnt);
	cr_assert(cache->uncached_cnt == 0,
		  "Cache has uncached entries: %zu",
		  cache->uncached_cnt);

	/* release the registrations, this should put them on the LRU list */
	for(i = 0; i < num_regions; i++) {
		ret = fi_close(&region_data[i].mr->fid);
		cr_assert_eq(ret, FI_SUCCESS,
			     "Failed to close mr %zu: %d",
			     i, ret);
	}

	/* Register one more, this should push one off LRU list */
	ret = fi_mr_reg(cxit_domain, region_data[num_regions].addr,
			page_size, FI_READ | FI_WRITE,
			0, 0, 0, &region_data[num_regions].mr, NULL);
	cr_assert_eq(ret, FI_SUCCESS,
		     "fi_mr_reg failed for region %lu: %d", num_regions, ret);

	/* Cache should remain full */
	cr_assert(cache->cached_cnt == cache->cached_max_cnt,
		  "Cache is not full: %zu != %zu",
		  cache->cached_cnt, cache->cached_max_cnt);
	cr_assert(cache->uncached_cnt == 0,
		  "Cache has uncached entries: %zu",
		  cache->uncached_cnt);

	ret = fi_close(&region_data[num_regions].mr->fid);
	cr_assert_eq(ret, FI_SUCCESS, "Failed to close mr: %d", ret);

	cxit_teardown_msg();
}
