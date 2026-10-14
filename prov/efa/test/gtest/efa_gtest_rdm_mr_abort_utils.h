/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_GTEST_RDM_MR_ABORT_UTILS_H
#define EFA_GTEST_RDM_MR_ABORT_UTILS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum efa_test_mr_abort_pkt {
	EFA_TEST_MR_ABORT_PKT_MEDIUM_TAGRTM = 0,
	EFA_TEST_MR_ABORT_PKT_LONGCTS_TAGRTM,
	EFA_TEST_MR_ABORT_PKT_PEER_ERROR,
	EFA_TEST_MR_ABORT_PKT_HANDSHAKE,
};

enum efa_test_mr_abort_err {
	EFA_TEST_MR_ABORT_ERR_RNR = 0,
	EFA_TEST_MR_ABORT_ERR_INVALID_LKEY,
};

struct efa_test_mr_abort_state {
	bool alive;
	bool state_err;
	bool peer_abort_pending;
	bool peer_error_emitted;
	bool queued_rnr;
	bool on_ope_queued_list;
	size_t queued_pkts;
	size_t txe_outstanding;
	size_t ep_outstanding;
	size_t ep_rnr_queued_pkt_cnt;
	bool peer_handshake_sent;
	bool peer_handshake_queued;
	size_t handshake_queued_peers;
};

int efa_test_mr_abort_setup(struct fid_ep *ep, struct fid_av *av, int req_pkt,
			    bool handshake_received, void *op_context);

void efa_test_mr_abort_set_peer_flags(bool req_sent, bool handshake_sent,
				      bool handshake_queued);

void efa_test_mr_abort_set_peer_error_emitted(void);

void *efa_test_mr_abort_forge_tx_pke(int pkt);

void efa_test_mr_abort_handle_tx_error(void *pke, int err);

void efa_test_mr_abort_txe_handle_error(void);

void efa_test_mr_abort_snapshot(struct efa_test_mr_abort_state *out);

void efa_test_mr_abort_complete_send(uintptr_t wr_id);

int efa_test_mr_abort_wire_type(int pkt);

void efa_test_mr_abort_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_RDM_MR_ABORT_UTILS_H */
