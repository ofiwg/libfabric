/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

/* C-linkage bridge for the RDM CQ tests.
 * See efa_gtest_common_helpers.h for why this exists. */

#ifndef EFA_GTEST_RDM_CQ_UTILS_H
#define EFA_GTEST_RDM_CQ_UTILS_H

#include <infiniband/verbs.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct efa_ibv_cq;
struct efa_rdm_pke;

/* extra_info[0] of the fabricated handshake packet. Uses a bit above every
 * EFA_RDM_EXTRA_FEATURE_* so the handler's feature checks are unaffected. */
#define EFA_TEST_RDM_CQ_EXTRA_INFO_SENTINEL (1ULL << 40)
#define EFA_TEST_RDM_CQ_DEVICE_VERSION	    0xEFA9

/**
 * @brief State of one fabricated RECV completion whose source address is
 * inserted into the explicit AV mid-lookup.
 */
struct efa_test_rdm_cq_race_ctx {
	struct fid_ep *ep;
	struct fid_av *av;
	struct fid_cq *cq;
	struct efa_rdm_pke *pke;
	/* wr_id of the fabricated CQE */
	uint64_t wr_id;
	size_t pkt_size;
	/* what the CQE reports, and what the racing address is keyed by */
	uint16_t ahn;
	uint32_t qpn;
	uint32_t qkey;
	union ibv_gid src_gid;
	uint32_t nextra_p3;
	/* explicit fi_addr of the racing insert, FI_ADDR_NOTAVAIL until then */
	fi_addr_t racing_addr;
};

/**
 * @brief Fabricate the RX handshake packet of a RECV completion from a peer
 * that is in neither AV.
 *
 * The packet's source address shares the endpoint's GID -- so the racing
 * insert reuses self_ah and its reverse AV entry lands on a known AHN -- but
 * carries a different QP number, so it is a distinct address.
 *
 * @param[in]	with_connid	include the optional connid header. Without it
 *				the packet carries no raw address at all, so the
 *				CQ read path cannot look one up.
 * @return 0 on success, a negative fi errno otherwise.
 */
int efa_test_rdm_cq_race_setup(struct fid_ep *ep, struct fid_av *av,
			       struct fid_cq *cq, int with_connid,
			       struct efa_test_rdm_cq_race_ctx *ctx);

/**
 * @brief The racing writer: insert the packet's source address into the
 * explicit AV, as a concurrent fi_av_insert would.
 *
 * Also marks the peer's handshake as already sent, so the recv path does not
 * post one and leave an outstanding TX op behind.
 *
 * @return the new explicit fi_addr, or FI_ADDR_NOTAVAIL on failure.
 */
fi_addr_t efa_test_rdm_cq_race_insert(struct efa_test_rdm_cq_race_ctx *ctx);

/**
 * @brief Drive the fabricated completion through efa_rdm_cq_poll_ibv_cq,
 * holding the ep_list_lock it requires.
 */
int efa_test_rdm_cq_race_poll(struct efa_test_rdm_cq_race_ctx *ctx);

/**
 * @brief Whether the CQ can report a source GID (efadv CQ), which the raw
 * address lookup depends on.
 */
int efa_test_rdm_cq_reports_sgid(struct efa_ibv_cq *ibv_cq);

struct efa_test_rdm_cq_peer_state {
	int peer_exists;
	int handshake_received;
	uint32_t nextra_p3;
	uint64_t extra_info0;
	uint32_t device_version;
	fi_addr_t explicit_fi_addr;
	fi_addr_t implicit_fi_addr;
};

/**
 * @brief Read the explicit peer at @p addr without creating one.
 */
void efa_test_rdm_cq_peer_state(struct fid_ep *ep, fi_addr_t addr,
				struct efa_test_rdm_cq_peer_state *out);

size_t efa_test_rdm_cq_implicit_av_count(struct fid_av *av);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_RDM_CQ_UTILS_H */
