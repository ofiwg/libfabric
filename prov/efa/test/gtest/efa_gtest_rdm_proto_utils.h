/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

/* C-linkage bridge for the rdm protocol interface tests.
 * See efa_gtest_common_helpers.h for why this exists. */

#ifndef EFA_GTEST_RDM_PROTO_UTILS_H
#define EFA_GTEST_RDM_PROTO_UTILS_H

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EFA_TEST_PROTO_MAX_PKES 16

/* A message needing more than one REQ packet but still fitting inside them. */
#define EFA_TEST_PROTO_MEDIUM_LEN 16384

struct efa_test_proto_rx_callback_result {
	ssize_t after_robuf_ret;
	ssize_t unexpected_match_ret;
};

void efa_test_proto_callbacks_preserve_return_values(
	struct efa_test_proto_rx_callback_result *out);

/** @brief Whether @p len is above one eager packet and within the medium
 * threshold on the selected device. */
int efa_test_proto_medium_len_in_band(struct fid_ep *ep, size_t len);

struct efa_test_proto_plan_result {
	ssize_t ret;
	size_t pkt_entry_cnt;
	size_t data_sizes[EFA_TEST_PROTO_MAX_PKES];
};

/**
 * @brief Drive efa_rdm_proto_medium_plan_tx_pkes() on a mock txe.
 *
 * A mock txe is what makes the interface and the message length free
 * variables, including an interface this host has no memory for.
 *
 * @param[in]	iface		interface the source descriptor reports
 * @param[in]	align128	request 128 byte send/recv alignment on the ep
 * @param[in]	leave_one_tx_pkt  exhaust the ep's TX ops down to one
 * @return	0 if @p out was filled, negative on setup failure
 */
int efa_test_proto_medium_plan(struct fid_ep *ep, struct fid_av *av,
			       enum fi_hmem_iface iface, int align128,
			       size_t total_len, int leave_one_tx_pkt,
			       struct efa_test_proto_plan_result *out);

struct efa_test_proto_construct_result {
	ssize_t ret;
	int selected_medium;
	int txe_filled;
	int req_pkt_type;
	size_t total_len;
	uint64_t bytes_sent;
	size_t pke_cnt;
	size_t callbacks_set;
	size_t ope_backrefs_set;
	uint64_t msg_lengths[EFA_TEST_PROTO_MAX_PKES];
	uint64_t seg_offsets[EFA_TEST_PROTO_MAX_PKES];
	size_t payload_sizes[EFA_TEST_PROTO_MAX_PKES];
};

/**
 * @brief Run selection, txe fill and construct_tx_pkes() the way
 * efa_rdm_msg_generic_send() does, and report the packets built.
 */
int efa_test_proto_medium_construct(struct fid_ep *ep, struct fid_av *av,
				    struct fid_domain *domain,
				    struct efa_test_proto_construct_result *out);

/**
 * @brief Construct twice on one txe, as the pre-handshake repost does. The
 * post-send hook runs after the second attempt only, so @p second carries the
 * resulting bytes_sent.
 */
int efa_test_proto_medium_construct_repost(
	struct fid_ep *ep, struct fid_av *av, struct fid_domain *domain,
	struct efa_test_proto_construct_result *first,
	struct efa_test_proto_construct_result *second);

struct efa_test_proto_completion_result {
	size_t pke_cnt;
	uint64_t total_len;
	size_t ope_list_after_send;
	size_t payload_sizes[EFA_TEST_PROTO_MAX_PKES];
	uint64_t bytes_acked_after[EFA_TEST_PROTO_MAX_PKES];
	size_t ope_list_after[EFA_TEST_PROTO_MAX_PKES];
};

/**
 * @brief fi_send a medium message, then drive each packet's send completion
 * callback in turn. The caller must arm efa_qp_post_send.
 */
int efa_test_proto_medium_completion(
	struct fid_ep *ep, struct fid_av *av, struct fid_domain *domain,
	struct efa_test_proto_completion_result *out);

struct efa_test_proto_peer_abort_result {
	size_t pke_cnt;
	int req_pkt_type_is_rtm;
	int abort_pending_after_error;
	int emitted_after_error;
	ssize_t readerr_after_error;
	int emitted_after[EFA_TEST_PROTO_MAX_PKES];
	ssize_t readerr_after[EFA_TEST_PROTO_MAX_PKES];
	size_t ope_list_after_data;
	ssize_t readerr_after_data;
	size_t ope_list_final;
	ssize_t readerr_final;
	int final_err;
	int final_prov_errno;
};

/**
 * @brief fi_send a medium message, cancel its source MR, then complete every
 * data WR successfully and finally the PEER_ERROR_PKT. The caller must arm
 * efa_qp_post_send.
 */
int efa_test_proto_medium_peer_abort(
	struct fid_ep *ep, struct fid_av *av, struct fid_cq *cq,
	struct fid_domain *domain,
	struct efa_test_proto_peer_abort_result *out);

/** @brief prov_errno a peer/MR abort reports. */
int efa_test_proto_peer_abort_prov_errno(void);

/** @brief EFA_RDM_MEDIUM_MSGRTM_PKT, whose enum is not includable from C++. */
int efa_test_proto_medium_msgrtm_pkt_type(void);

struct efa_test_proto_force_result {
	ssize_t ret;
	/** Name of the protocol selection picked, empty when it picked none. */
	char proto_name[32];
};

/**
 * @brief Select a send protocol for a @p len byte host message with
 * FI_EFA_RDM_FORCE_SEND_PROTO set to @p forced.
 *
 * @param[in]	forced	FI_EFA_RDM_FORCE_SEND_PROTO value, NULL for unset
 * @return	0 if @p out was filled, negative on setup failure
 */
int efa_test_proto_force_select_send(struct fid_ep *ep, struct fid_av *av,
				     struct fid_domain *domain,
				     const char *forced, size_t len,
				     struct efa_test_proto_force_result *out);

#define EFA_TEST_PROTO_LARGE_LEN 131072
#define EFA_TEST_PROTO_RUNT_SIZE 32768

/** @brief Whether EFA_TEST_PROTO_LARGE_LEN is too large for eager and medium. */
int efa_test_proto_large_len_beyond_medium(struct fid_ep *ep);

/** @brief Whether the selected device can do RDMA read. */
int efa_test_proto_device_supports_rdma_read(void);

/**
 * @brief Save the host hmem thresholds and the device generation that the read
 * protocol tests override.
 */
void efa_test_proto_save_globals(void);

/** @brief Restore what efa_test_proto_save_globals() saved. */
void efa_test_proto_restore_globals(void);

struct efa_test_proto_select_result {
	ssize_t ret;
	/** Name of the protocol selection picked, empty when it picked none. */
	char proto_name[32];
};

/**
 * @brief Select a send protocol for an EFA_TEST_PROTO_LARGE_LEN byte host
 * message.
 *
 * @param[in]	read_peer	peer advertises RDMA read on a matching device
 * @param[in]	runt_size	host runt size, used when @p read_peer is set
 * @param[in]	with_desc	pass the registered descriptor; without it the
 *				MR cache is disabled for the selection
 * @param[in]	flags		send flags
 * @return	0 if @p out was filled, negative on setup failure
 */
int efa_test_proto_select(struct fid_ep *ep, struct fid_av *av,
			  struct fid_domain *domain, int read_peer,
			  size_t runt_size, int with_desc, uint64_t flags,
			  struct efa_test_proto_select_result *out);

struct efa_test_proto_longread_construct_result {
	ssize_t ret;
	int selected;
	size_t pke_cnt;
	int req_pkt_type;
	int expected_req_pkt_type;
	int handler_set;
	int ope_set;
	size_t payload_size;
	uint64_t msg_length;
	uint64_t total_len;
	uint32_t send_id;
	uint32_t tx_id;
	uint32_t read_iov_count;
	size_t iov_count;
	uint64_t read_iov_addr;
	uint64_t buf_addr;
	uint64_t read_iov_len;
	uint64_t buf_len;
	uint64_t read_iov_key;
	uint64_t mr_key;
	size_t pkt_size;
	size_t expected_pkt_size;
};

/** @brief Select long read, fill the txe and construct its REQ. */
int efa_test_longread_construct_builds_one_pke(
	struct fid_ep *ep, struct fid_av *av, struct fid_domain *domain,
	struct efa_test_proto_longread_construct_result *out);

struct efa_test_proto_repeat_result {
	ssize_t first_ret;
	ssize_t second_ret;
	int selected;
	size_t first_pke_cnt;
	size_t second_pke_cnt;
	int first_req_pkt_type;
	int second_req_pkt_type;
	uint64_t total_len;
	uint64_t second_msg_length;
	size_t first_payload_size;
	size_t second_payload_size;
	uint64_t first_seg_offsets[EFA_TEST_PROTO_MAX_PKES];
	uint64_t second_seg_offsets[EFA_TEST_PROTO_MAX_PKES];
	uint64_t first_bytes_runt;
	uint64_t second_bytes_runt;
	uint64_t bytes_sent_after_first_hook;
	uint64_t bytes_sent_after_second_hook;
	int read_msg_counted_after_first_hook;
	int64_t read_msg_in_flight_after_first_hook;
	int64_t read_msg_in_flight_after_second_hook;
	/** Protocol a second send picks while the first one is in flight. */
	char next_proto_name[32];
};

/**
 * @brief Construct the long read REQ twice on one txe, running the post-send
 * hook after each attempt.
 */
int efa_test_longread_construct_is_repeatable(
	struct fid_ep *ep, struct fid_av *av, struct fid_domain *domain,
	struct efa_test_proto_repeat_result *out);

/**
 * @brief Construct the runt read REQs twice on one txe, running the post-send
 * hook after each attempt, then select a protocol for a second send.
 */
int efa_test_runtread_construct_is_repeatable(
	struct fid_ep *ep, struct fid_av *av, struct fid_domain *domain,
	struct efa_test_proto_repeat_result *out);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_RDM_PROTO_UTILS_H */
