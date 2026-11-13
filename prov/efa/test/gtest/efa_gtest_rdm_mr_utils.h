/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_GTEST_RDM_MR_UTILS_H
#define EFA_GTEST_RDM_MR_UTILS_H

#include <stdbool.h>
#include <rdma/fi_domain.h>

#ifdef __cplusplus
extern "C" {
#endif

enum efa_test_mr_monitor {
	EFA_TEST_MR_MONITOR_NONE,
	EFA_TEST_MR_MONITOR_KDREG2,
	EFA_TEST_MR_MONITOR_MEMHOOKS,
	EFA_TEST_MR_MONITOR_UFFD,
};

enum efa_test_mr_monitor_start {
	EFA_TEST_MR_MONITOR_START_REAL,
	EFA_TEST_MR_MONITOR_START_SUCCEED,
	EFA_TEST_MR_MONITOR_START_FAIL,
};

/**
 * @brief Save the memory monitor and MR cache state that the functions below
 * and opening a domain with an MR cache change, and enable the EFA MR cache.
 * efa_test_mr_monitor_restore() restores it. Libfabric must be initialized
 * first, e.g. by fi_getinfo(), since its initialization sets the default
 * monitor.
 */
void efa_test_mr_monitor_save(void);

void efa_test_mr_monitor_restore(void);

/**
 * @brief Set the default system memory monitor and the FI_MR_CACHE_MONITOR
 * value (NULL if unset) as ofi_monitors_init() would have left them.
 */
void efa_test_mr_monitor_set_default(enum efa_test_mr_monitor monitor,
				     const char *requested);

/**
 * @brief Replace the start function of @p monitor. A replaced start function
 * does not start the monitor; one that fails returns -FI_ENODEV and counts
 * the call.
 */
void efa_test_mr_monitor_set_start(enum efa_test_mr_monitor monitor,
				   enum efa_test_mr_monitor_start start);

int efa_test_mr_monitor_failed_starts(void);

/**
 * @brief The system memory monitor of the RDM domain's MR cache, or
 * EFA_TEST_MR_MONITOR_NONE if the domain has no MR cache.
 */
enum efa_test_mr_monitor
efa_test_rdm_domain_mr_cache_monitor(struct fid_domain *domain_fid);

bool efa_test_memhooks_monitor_built(void);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_RDM_MR_UTILS_H */
