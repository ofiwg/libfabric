/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_GTEST_FI_MORE_HELPERS_H
#define EFA_GTEST_FI_MORE_HELPERS_H

#include <stdbool.h>
#include <stdint.h>
#include <infiniband/verbs.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Remote addressing for the RMA operations. Nothing is deliverable, so these
 * only have to be distinctive. */
#define EFA_TEST_RKEY	0x00C0FFEE
#define EFA_TEST_RADDR	0x0000BEEF00001000ULL

enum efa_test_post_op {
	EFA_TEST_POST_SEND = 0,
	EFA_TEST_POST_READ = 1,
	EFA_TEST_POST_WRITE = 2,
};

/**
 * @brief Which arm of the efa_qp_post_* dispatch to exercise.
 *
 * The two hand work requests to the device by different means, so a test
 * observes them in different places, but the FI_MORE contract they implement is
 * the same one.
 */
enum efa_test_dp_backend {
	/** efa_data_path_direct_post_*: submitting rings the SQ doorbell. */
	EFA_TEST_DP_DIRECT = 0,
	/** efa_ibv_post_*: submitting calls ibv_wr_complete. */
	EFA_TEST_DP_RDMA_CORE = 1,
};

/**
 * @brief Intercepts however the selected backend hands work to the device.
 *
 * Opaque to the test body; use the accessors below, which read the same way for
 * either backend.
 */
struct efa_test_dp_probe {
	int backend;
	void *qp;			/* struct efa_qp *, NULL when not installed */

	/* EFA_TEST_DP_DIRECT: the redirected send queue */
	void *sq;			/* struct efa_data_path_direct_sq * */
	uint8_t *saved_desc;
	uint32_t *saved_db;
	uint8_t *scratch_desc;
	uint32_t scratch_db;

	/* EFA_TEST_DP_RDMA_CORE: the hooked work request vtable */
	void *qpx;			/* struct ibv_qp_ex * */
	void *saved_qpx;		/* malloc'd copy of the original vtable */
	void *saved_set_hints;
	int saved_direct_enabled;
};

/**
 * @brief Redirect the selected backend's device sink at test-owned state, so a
 * post can be observed without anything reaching the hardware.
 *
 * For EFA_TEST_DP_DIRECT this points the send queue's descriptor buffer and
 * doorbell register at test memory. For EFA_TEST_DP_RDMA_CORE it clears
 * qp->data_path_direct_enabled, so that arm of the dispatch is the one taken,
 * and replaces the ibv_qp_ex work request vtable with counting no-ops.
 *
 * @return 0, -FI_EOPNOTSUPP if the backend is unavailable on this device, or
 *	   -FI_ENOMEM on allocation failure.
 */
int efa_test_dp_probe_install(struct fid_ep *ep, int backend,
			      struct efa_test_dp_probe *p);

/**
 * @brief Restore everything install() and the posts under it changed.
 * Idempotent. Must run before the endpoint is closed: a FI_MORE test
 * deliberately leaves work unsubmitted, and closing the endpoint neither rings
 * the doorbell nor flushes an open work request session.
 */
void efa_test_dp_probe_restore(struct efa_test_dp_probe *p);

/**
 * @brief Whether the backend has handed anything to the device since install()
 * or the last reset(): the doorbell rang, or ibv_wr_complete was called.
 */
bool efa_test_dp_probe_submitted(const struct efa_test_dp_probe *p);

/**
 * @brief Whether the backend is holding work it has not submitted: entries
 * staged in the send queue, or an open work request session.
 */
bool efa_test_dp_probe_pending(const struct efa_test_dp_probe *p);

/** @brief Forget any submission already observed. */
void efa_test_dp_probe_reset(struct efa_test_dp_probe *p);

/**
 * @brief Make submissions fail with @p err, until called again with 0.
 *
 * EFA_TEST_DP_RDMA_CORE only: the direct path writes a doorbell register rather
 * than calling anything that can report an error.
 */
void efa_test_dp_probe_set_submit_error(struct efa_test_dp_probe *p, int err);

/** @brief Whether the endpoint has an ibv_wr_start session open. */
int efa_test_ep_is_wr_started(struct fid_ep *ep);

/* ---------------------------------------------------------------------------
 * efa-rdm
 * ------------------------------------------------------------------------ */

/**
 * @brief Set up an RDM peer that takes the device data path: own GID with a
 * different QPN, so the address handle is real but the peer is not self, with
 * the handshake recorded as received and device RDMA plus p2p advertised.
 */
int efa_test_rdm_setup_peer(struct fid_ep *ep, struct fid_av *av,
			    fi_addr_t *peer_addr);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_FI_MORE_HELPERS_H */
