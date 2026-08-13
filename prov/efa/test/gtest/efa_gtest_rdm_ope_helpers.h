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

enum efa_test_queued_op_kind {
	EFA_TEST_QUEUED_OP_SEND = 0,
	EFA_TEST_QUEUED_OP_READ = 1,
	EFA_TEST_QUEUED_OP_WRITE = 2,
};

struct efa_rdm_ope;
struct efa_rdm_peer;

struct efa_test_queued_op {
	struct fid_ep *ep;
	struct fid_mr *mr;
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;
	int fi_more_was_set;
	char buf[16];
};

/**
 * @brief Issue a send/read/write with FI_MORE to a peer with no handshake yet,
 * so the op is queued on the endpoint's ope_queued_list instead of posted.
 */
int efa_test_queue_op_with_fi_more(struct fid_ep *ep, struct fid_av *av,
				   struct fid_domain *domain, int op_kind,
				   struct efa_test_queued_op *qop);

/**
 * @brief Simulate handshake completion (device RDMA + p2p advertised), then
 * process the queued ope so it reposts to the QP.
 */
int efa_test_process_queued_ope_after_handshake(struct efa_test_queued_op *qop);

/**
 * @brief Release the posted pkt entry (by wr_id) and the txe, and close the MR.
 */
void efa_test_queued_op_cleanup(struct efa_test_queued_op *qop, uint64_t wr_id);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_RDM_OPE_HELPERS_H */
