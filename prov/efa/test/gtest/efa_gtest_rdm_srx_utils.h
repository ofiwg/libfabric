/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_GTEST_RDM_SRX_UTILS_H
#define EFA_GTEST_RDM_SRX_UTILS_H

#include <rdma/fi_endpoint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct efa_test_srx_dispatch_result {
	int was_unexpected;
	int callback_set;
	int callback_invocations;
	int matched;
	int unexpected_packet_cleared;
};

int efa_test_srx_dispatches_receive_callback(
	struct fid_ep *ep, struct fid_av *av, int unexpected,
	struct efa_test_srx_dispatch_result *out);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_RDM_SRX_UTILS_H */
