/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_GTEST_RDM_OPE_CTRL_UTILS_H
#define EFA_GTEST_RDM_OPE_CTRL_UTILS_H

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Which non-REQ ctrl packet a case posts, and on which ope kind. */
enum efa_test_ctrl_case {
	EFA_TEST_CTRL_CTS_RXE = 0,
	EFA_TEST_CTRL_CTS_TXE,
	EFA_TEST_CTRL_READRSP_FITS,
	EFA_TEST_CTRL_READRSP_REMAINDER,
	EFA_TEST_CTRL_EOR,
	EFA_TEST_CTRL_RECEIPT,
	EFA_TEST_CTRL_READ_NACK,
	EFA_TEST_CTRL_ATOMRSP,
	EFA_TEST_CTRL_PEER_ERROR_RXE,
	EFA_TEST_CTRL_PEER_ERROR_TXE_PROTO,
};

/** @brief How a case should perturb the post, for the queue/error tests. */
enum efa_test_ctrl_perturb {
	EFA_TEST_CTRL_PERTURB_NONE = 0,
	/** Drive efa_outstanding_tx_ops to the max so no TX packet is free. */
	EFA_TEST_CTRL_PERTURB_TX_FULL,
	/** Set FI_MORE on the ope, which no non-REQ packet may propagate. */
	EFA_TEST_CTRL_PERTURB_FI_MORE,
};

#define EFA_TEST_CTRL_PAYLOAD_MAX 256

/** @brief The ids the bridge stamped on the ope, to assert the wire against. */
struct efa_test_ctrl_ids {
	uint32_t tx_id;
	uint32_t rx_id;
	uint32_t msg_id;
	uint32_t connid;
	int prov_errno;
};

/** @brief Decoded header of the ctrl packet that reached efa_qp_post_send. */
struct efa_test_ctrl_wire {
	int decoded;
	int pkt_type;
	int version;
	uint16_t flags;
	uint32_t send_id;
	uint32_t recv_id;
	uint32_t tx_id;
	uint32_t msg_id;
	uint32_t op_id;
	uint32_t emitter_ope_type;
	uint32_t prov_errno;
	uint32_t connid;
	uint64_t recv_length;
	uint64_t seg_length;
	size_t pkt_size;
	size_t payload_size;
	char payload_bytes[EFA_TEST_CTRL_PAYLOAD_MAX];
};

/** @brief Observable state after the post returns. */
struct efa_test_ctrl_result {
	ssize_t ret;
	struct efa_test_ctrl_ids ids;
	int queued_ctrl_flag_set;
	int queued_ctrl_type;
	int queued_list_empty;
	int on_posted_ack_list;
	int on_longcts_send_list;
	uint64_t ope_window;
	uint64_t ope_bytes_sent;
	int ope_state;
	size_t outstanding_tx_ops;
	size_t source_len;
	char source_bytes[EFA_TEST_CTRL_PAYLOAD_MAX];
};

/**
 * @brief Build the ope a case needs and drive
 * efa_rdm_ope_post_send_or_queue() on it. The caller must arm efa_qp_post_send
 * first; on the success cases it should return 0 so the packet entry survives
 * for efa_test_ctrl_decode_posted().
 *
 * @return 0 if @p out was filled, negative on setup failure.
 */
int efa_test_ctrl_post(struct fid_ep *ep, struct fid_av *av, int which,
		       int perturb, struct efa_test_ctrl_result *out);

/**
 * @brief Decode the packet the poster staged, which is still live after a
 * successful post. Call before efa_test_ctrl_cleanup().
 */
void efa_test_ctrl_decode_posted(struct fid_ep *ep,
				 struct efa_test_ctrl_wire *out);

/** @brief Release whatever the case allocated, including a posted packet. */
void efa_test_ctrl_cleanup(void);

/** @brief The packet type a case is expected to put on the wire. */
int efa_test_ctrl_expected_pkt_type(int which);

/** @brief sizeof() the header struct a case's packet type uses. */
size_t efa_test_ctrl_expected_hdr_size(int which);

/** @brief The protocol version every EFA RDM packet carries. */
int efa_test_ctrl_protocol_version(void);

/** @brief The EFA_RDM_PKT_CONNID_HDR flag bit. */
uint16_t efa_test_ctrl_connid_hdr_flag(void);

/** @brief The EFA_RDM_CTS_READ_REQ flag bit. */
uint16_t efa_test_ctrl_cts_read_req_flag(void);

/** @brief enum efa_rdm_ope_type values, for the emitter_ope_type assertions. */
int efa_test_ctrl_ope_type_txe(void);
int efa_test_ctrl_ope_type_rxe(void);

/** @brief The EFA_RDM_OPE_ID_INVALID sentinel a txe PEER_ERROR carries. */
uint32_t efa_test_ctrl_ope_id_invalid(void);

/** @brief recv_length a CTS for @p bytes_left must advertise. */
uint64_t efa_test_ctrl_expected_cts_recv_length(struct fid_ep *ep,
					       uint64_t bytes_left);

/** @brief Largest READRSP payload that still fits one packet. */
size_t efa_test_ctrl_readrsp_max_payload(struct fid_ep *ep);

/** @brief Observable state of the long CTS continuation drain. */
struct efa_test_cont_result {
	int on_longcts_send_list;
	uint64_t ope_bytes_sent;
};

/**
 * @brief Build a long CTS read responder rxe mid-transfer and place it on the
 * endpoint's ope_longcts_send_list, as a received CTS would. The rxe carries
 * the long CTS read protocol and a window with data still to send.
 *
 * @return 0 on success, negative on setup failure.
 */
int efa_test_ctrl_setup_longcts_continuation(struct fid_ep *ep,
					     struct fid_av *av,
					     struct efa_test_cont_result *out);

/**
 * @brief Run the progress engine's long CTS send-list drain once. When
 * @p tx_full is set, the TX pool is exhausted for the duration of the drain so
 * the continuation post yields -FI_EAGAIN. Reports the post-drain state.
 */
void efa_test_ctrl_drive_continuation(struct fid_ep *ep, int tx_full,
				      struct efa_test_cont_result *out);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_RDM_OPE_CTRL_UTILS_H */
