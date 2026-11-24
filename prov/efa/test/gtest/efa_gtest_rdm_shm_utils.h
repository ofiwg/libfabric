/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

/* C-linkage bridge for the RDM shm tests.
 * See efa_gtest_common_helpers.h for why this exists. */

#ifndef EFA_GTEST_RDM_SHM_UTILS_H
#define EFA_GTEST_RDM_SHM_UTILS_H

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>

#ifdef __cplusplus
extern "C" {
#endif

int efa_test_rdm_ep_has_shm_ep(struct fid_ep *ep);
int efa_test_rdm_av_has_shm_av(struct fid_av *av);

/**
 * @brief Reference count of the shm peer cq of an RDM cq, or -1 if the cq
 * has no shm peer cq.
 */
int efa_test_rdm_cq_shm_cq_ref(struct fid_cq *cq);

/**
 * @brief Reference count of the shm peer counter of an RDM counter, or -1 if
 * the counter has no shm peer counter.
 */
int efa_test_rdm_cntr_shm_cntr_ref(struct fid_cntr *cntr);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_RDM_SHM_UTILS_H */
