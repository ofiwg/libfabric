/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_GTEST_RDM_AV_UTILS_H
#define EFA_GTEST_RDM_AV_UTILS_H

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int efa_test_get_shm_av_size(void);

int efa_test_set_shm_av_size(int value);

int efa_test_av_insert_local_peer(struct fid_ep *ep, struct fid_av *av,
				  uint32_t qpn, fi_addr_t *addr);

bool efa_test_av_peer_uses_shm(struct fid_ep *ep, fi_addr_t addr);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_RDM_AV_UTILS_H */
