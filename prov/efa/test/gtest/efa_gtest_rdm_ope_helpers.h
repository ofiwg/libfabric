/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_GTEST_RDM_OPE_HELPERS_H
#define EFA_GTEST_RDM_OPE_HELPERS_H

#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>

#ifdef __cplusplus
extern "C" {
#endif

int efa_test_drive_rxe_unexp_handle_error(struct fid_ep *ep, void *op_context,
					  int err, int *prov_errno_out);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_RDM_OPE_HELPERS_H */
