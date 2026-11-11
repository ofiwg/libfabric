/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_rdm_mr_utils.h"
#include "efa.h"
#include "rdm/efa_rdm_domain.h"
#include "rdm/efa_rdm_mr.h"

typedef int (*efa_test_monitor_start_fn)(struct ofi_mem_monitor *monitor);

static struct {
	struct ofi_mem_monitor *default_monitor;
	char *requested;
	efa_test_monitor_start_fn kdreg2_start;
	efa_test_monitor_start_fn memhooks_start;
	efa_test_monitor_start_fn uffd_start;
	int mr_cache_enable;
	size_t max_cnt;
	size_t max_size;
	size_t mr_max_cached_count;
	size_t mr_max_cached_size;
} efa_test_mr_monitor_saved;

static int efa_test_mr_monitor_failed_start_count;

static struct ofi_mem_monitor *
efa_test_mr_monitor_ptr(enum efa_test_mr_monitor monitor)
{
	switch (monitor) {
	case EFA_TEST_MR_MONITOR_KDREG2:
		return kdreg2_monitor;
	case EFA_TEST_MR_MONITOR_MEMHOOKS:
		return memhooks_monitor;
	case EFA_TEST_MR_MONITOR_UFFD:
		return uffd_monitor;
	default:
		return NULL;
	}
}

static efa_test_monitor_start_fn
efa_test_mr_monitor_saved_start(enum efa_test_mr_monitor monitor)
{
	switch (monitor) {
	case EFA_TEST_MR_MONITOR_KDREG2:
		return efa_test_mr_monitor_saved.kdreg2_start;
	case EFA_TEST_MR_MONITOR_MEMHOOKS:
		return efa_test_mr_monitor_saved.memhooks_start;
	case EFA_TEST_MR_MONITOR_UFFD:
		return efa_test_mr_monitor_saved.uffd_start;
	default:
		return NULL;
	}
}

static int efa_test_mr_monitor_start_succeed(struct ofi_mem_monitor *monitor)
{
	return 0;
}

static int efa_test_mr_monitor_start_fail(struct ofi_mem_monitor *monitor)
{
	efa_test_mr_monitor_failed_start_count++;
	return -FI_ENODEV;
}

void efa_test_mr_monitor_save(void)
{
	efa_test_mr_monitor_saved.default_monitor = default_monitor;
	efa_test_mr_monitor_saved.requested = cache_params.monitor;
	efa_test_mr_monitor_saved.kdreg2_start = kdreg2_monitor->start;
	efa_test_mr_monitor_saved.memhooks_start = memhooks_monitor->start;
	efa_test_mr_monitor_saved.uffd_start = uffd_monitor->start;
	efa_test_mr_monitor_saved.mr_cache_enable = efa_mr_cache_enable;
	efa_test_mr_monitor_saved.max_cnt = cache_params.max_cnt;
	efa_test_mr_monitor_saved.max_size = cache_params.max_size;
	efa_test_mr_monitor_saved.mr_max_cached_count = efa_mr_max_cached_count;
	efa_test_mr_monitor_saved.mr_max_cached_size = efa_mr_max_cached_size;

	efa_mr_cache_enable = 1;
	efa_test_mr_monitor_failed_start_count = 0;
}

void efa_test_mr_monitor_restore(void)
{
	default_monitor = efa_test_mr_monitor_saved.default_monitor;
	cache_params.monitor = efa_test_mr_monitor_saved.requested;
	kdreg2_monitor->start = efa_test_mr_monitor_saved.kdreg2_start;
	memhooks_monitor->start = efa_test_mr_monitor_saved.memhooks_start;
	uffd_monitor->start = efa_test_mr_monitor_saved.uffd_start;
	efa_mr_cache_enable = efa_test_mr_monitor_saved.mr_cache_enable;
	cache_params.max_cnt = efa_test_mr_monitor_saved.max_cnt;
	cache_params.max_size = efa_test_mr_monitor_saved.max_size;
	efa_mr_max_cached_count = efa_test_mr_monitor_saved.mr_max_cached_count;
	efa_mr_max_cached_size = efa_test_mr_monitor_saved.mr_max_cached_size;
}

void efa_test_mr_monitor_set_default(enum efa_test_mr_monitor monitor,
				     const char *requested)
{
	default_monitor = efa_test_mr_monitor_ptr(monitor);
	cache_params.monitor = (char *) requested;
}

void efa_test_mr_monitor_set_start(enum efa_test_mr_monitor monitor,
				   enum efa_test_mr_monitor_start start)
{
	struct ofi_mem_monitor *mon = efa_test_mr_monitor_ptr(monitor);

	switch (start) {
	case EFA_TEST_MR_MONITOR_START_SUCCEED:
		mon->start = efa_test_mr_monitor_start_succeed;
		break;
	case EFA_TEST_MR_MONITOR_START_FAIL:
		mon->start = efa_test_mr_monitor_start_fail;
		break;
	default:
		mon->start = efa_test_mr_monitor_saved_start(monitor);
		break;
	}
}

int efa_test_mr_monitor_failed_starts(void)
{
	return efa_test_mr_monitor_failed_start_count;
}

enum efa_test_mr_monitor
efa_test_rdm_domain_mr_cache_monitor(struct fid_domain *domain_fid)
{
	struct efa_domain *efa_domain = container_of(
		domain_fid, struct efa_domain, util_domain.domain_fid);
	struct efa_rdm_domain *rdm_domain = container_of(
		efa_domain, struct efa_rdm_domain, efa_domain);
	struct ofi_mem_monitor *monitor;

	if (!rdm_domain->cache)
		return EFA_TEST_MR_MONITOR_NONE;

	monitor = rdm_domain->cache->monitors[FI_HMEM_SYSTEM];
	if (monitor == kdreg2_monitor)
		return EFA_TEST_MR_MONITOR_KDREG2;
	if (monitor == memhooks_monitor)
		return EFA_TEST_MR_MONITOR_MEMHOOKS;
	if (monitor == uffd_monitor)
		return EFA_TEST_MR_MONITOR_UFFD;
	return EFA_TEST_MR_MONITOR_NONE;
}

bool efa_test_memhooks_monitor_built(void)
{
	return HAVE_MEMHOOKS_MONITOR;
}
