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

/**
 * @brief Which EFA_RDM_OPE_QUEUED_* flag an ope is queued with, for the
 * dispatch-arm tests. BEFORE_HANDSHAKE is covered separately because it is
 * the one flag a queued op reaches through the normal send path.
 */
enum efa_test_queued_flag_kind {
	EFA_TEST_QUEUED_FLAG_RNR = 0,
	EFA_TEST_QUEUED_FLAG_CTRL = 1,
	EFA_TEST_QUEUED_FLAG_READ = 2,
};

struct efa_rdm_ope;
struct efa_rdm_peer;

struct efa_test_queued_op {
	struct fid_ep *ep;
	struct fid_mr *mr;
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;
	int fi_more_was_set;
	int queued_ctrl_type;
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

/**
 * @brief Observable state after a efa_rdm_ope_process_queued_ope() call.
 */
struct efa_test_process_queued_result {
	int ret;
	int any_queued_flag_set;
	int queued_list_empty;
	size_t before_handshake_cnt;
};

/**
 * @brief Queue a txe on the endpoint's ope_queued_list carrying exactly one of
 * the RNR / CTRL / READ flags, so the dispatch arm the derivation selects can
 * be observed. The RNR case goes through efa_rdm_ep_queue_rnr_pkt() so the
 * ope's queued_pkts list is populated as production would leave it.
 *
 * @return 0 on success, negative otherwise.
 */
int efa_test_queue_ope_with_flag(struct fid_ep *ep, struct fid_av *av,
				 int flag_kind, struct efa_test_queued_op *qop);

/**
 * @brief Drive efa_rdm_ope_process_queued_ope() on an ope queued by
 * efa_test_queue_ope_with_flag() and report the post-call bookkeeping state.
 *
 * @return 0 if @p res was filled, negative otherwise.
 */
int efa_test_process_queued_flag_op(struct efa_test_queued_op *qop,
				    struct efa_test_process_queued_result *res);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_RDM_OPE_HELPERS_H */
