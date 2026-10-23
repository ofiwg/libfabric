/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

/* C-linkage bridge for rdm pke tests.
 * See efa_gtest_common_helpers.h for why this exists. */

#ifndef EFA_GTEST_RDM_PKE_UTILS_H
#define EFA_GTEST_RDM_PKE_UTILS_H

#include <stdbool.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calls the RTM RX handler with a LONGCTS READ_NACK packet whose msg_id
 * is missing from the peer's rxe_map.
 *
 * @param[in]	ep		RDM efa endpoint
 * @param[in]	peer_addr	pre-inserted fi_addr of the peer
 * @param[in]	tagged		non-zero drives the tagged path (proc_tagrtm)
 * @param[out]	ret		return code of the handler
 * @return	1 if the handler returned without crashing, 0 on setup failure
 */
int efa_test_rtm_read_nack_missing_rxe(struct fid_ep *ep, fi_addr_t peer_addr,
				       int tagged, ssize_t *ret);
struct efa_rdm_pke;

/**
 * @brief Allocate a linked list of n unexpected packet entries from the
 * endpoint's unexpected packet pool.
 * @return head of the chain, or NULL on allocation failure.
 */
struct efa_rdm_pke *efa_test_pke_build_unexp_chain(struct fid_ep *ep, size_t n);

void efa_test_pke_release_cloned(struct efa_rdm_pke *head);

/**
 * @brief Count allocated buffers in the unexpected packet pool.
 */
size_t efa_test_ep_unexp_pool_outstanding(struct fid_ep *ep);

int efa_test_failed_reorder_msg_releases_rx_pkt(struct fid_ep *ep,
						fi_addr_t peer_addr,
						size_t *to_post_before,
						size_t *to_post_after);

int efa_test_failed_reorder_msg_overflow_releases_rx_pkt_and_entry(
	struct fid_ep *ep, fi_addr_t peer_addr, size_t *to_post_before,
	size_t *to_post_after, size_t *overflow_free_before,
	size_t *overflow_free_after);

struct efa_test_av_remove_unexp_result {
	int was_unexpected;
	int unspec_unexp_queue_empty;
	size_t unexp_pool_outstanding_before;
	size_t unexp_pool_outstanding_queued;
	size_t unexp_pool_outstanding_after;
	size_t rx_pkts_held_before;
	size_t rx_pkts_held_queued;
	size_t rx_pkts_held_after;
	size_t rx_pkts_to_post_queued;
	size_t rx_pkts_to_post_after;
	size_t ope_count_before;
	size_t ope_count_queued;
	size_t ope_count_after;
};

int efa_test_av_remove_releases_unexp_pkt(
	struct fid_ep *ep, struct fid_av *av, fi_addr_t peer_addr,
	int copy_unexp, struct efa_test_av_remove_unexp_result *out);

struct efa_test_reorder_callback_result {
	int callback_set;
	int protocol_correct;
};

int efa_test_reordered_packet_retains_callback(
	struct fid_ep *ep, fi_addr_t peer_addr,
	struct efa_test_reorder_callback_result *out);
int efa_test_reordered_rta_retains_callback(
	struct fid_ep *ep, fi_addr_t peer_addr,
	struct efa_test_reorder_callback_result *out);

enum efa_test_order_scenario {
	EFA_TEST_ORDER_MIXED_RTM_RTA,
	EFA_TEST_ORDER_DRAIN_ERROR,
	EFA_TEST_ORDER_IN_ORDER_ERROR,
	EFA_TEST_ORDER_SINGLE_DRAIN,
};

struct efa_test_order_result {
	uint32_t log[3], exp_msg_id_before, exp_msg_id;
	size_t log_len, pre_log_len;
	unsigned int callback_count[3];
	int drained;
};

int efa_test_ordered_receive(struct fid_ep *ep, fi_addr_t peer_addr,
			     enum efa_test_order_scenario scenario,
			     struct efa_test_order_result *out);

// Sentinel values that will be used in assertions
#define EFA_TEST_RTM_TAG	 0x1234567890abcdefULL
#define EFA_TEST_RTM_MSG_ID	 0x4321u
#define EFA_TEST_RTM_SEND_ID	 0x5678u
#define EFA_TEST_RTM_EAGER_LEN	 64u
#define EFA_TEST_RTM_LONG_LEN	 65536u
#define EFA_TEST_RTM_MEDIUM_SEG	 128u
#define EFA_TEST_RTM_MEDIUM_DATA 256u
#define EFA_TEST_RTM_RUNT_LEN	 4096u
#define EFA_TEST_RTM_RUNT_SEG	 2048u
#define EFA_TEST_RTM_RUNT_DATA	 2048u

enum efa_test_rtm_family {
	EFA_TEST_RTM_FAM_EAGER,
	EFA_TEST_RTM_FAM_MEDIUM,
	EFA_TEST_RTM_FAM_LONGCTS,
	EFA_TEST_RTM_FAM_LONGREAD,
	EFA_TEST_RTM_FAM_RUNTREAD,
};

#define EFA_TEST_RTM_FAMILY_MASK 0x7u
#define EFA_TEST_RTM_TAGGED	 (1u << 3)
#define EFA_TEST_RTM_DC		 (1u << 4)
#define EFA_TEST_RTM_ZERO_HDR	 (1u << 5)

#define EFA_TEST_RTM_FAMILY(v) \
	((enum efa_test_rtm_family)((v) & EFA_TEST_RTM_FAMILY_MASK))
#define EFA_TEST_RTM_IS_TAGGED(v)   (!!((v) & EFA_TEST_RTM_TAGGED))
#define EFA_TEST_RTM_IS_DC(v)	    (!!((v) & EFA_TEST_RTM_DC))
#define EFA_TEST_RTM_IS_ZERO_HDR(v) (!!((v) & EFA_TEST_RTM_ZERO_HDR))

// All 16 init functions in efa_rdm_pke_rtm.c + eager w/o base header
enum efa_test_rtm_variant {
	EFA_TEST_RTM_EAGER_MSG = EFA_TEST_RTM_FAM_EAGER,
	EFA_TEST_RTM_EAGER_TAG = EFA_TEST_RTM_FAM_EAGER | EFA_TEST_RTM_TAGGED,
	EFA_TEST_RTM_DC_EAGER_MSG = EFA_TEST_RTM_FAM_EAGER | EFA_TEST_RTM_DC,
	EFA_TEST_RTM_DC_EAGER_TAG =
		EFA_TEST_RTM_FAM_EAGER | EFA_TEST_RTM_TAGGED | EFA_TEST_RTM_DC,
	EFA_TEST_RTM_MEDIUM_MSG = EFA_TEST_RTM_FAM_MEDIUM,
	EFA_TEST_RTM_MEDIUM_TAG = EFA_TEST_RTM_FAM_MEDIUM | EFA_TEST_RTM_TAGGED,
	EFA_TEST_RTM_DC_MEDIUM_MSG = EFA_TEST_RTM_FAM_MEDIUM | EFA_TEST_RTM_DC,
	EFA_TEST_RTM_DC_MEDIUM_TAG =
		EFA_TEST_RTM_FAM_MEDIUM | EFA_TEST_RTM_TAGGED | EFA_TEST_RTM_DC,
	EFA_TEST_RTM_LONGCTS_MSG = EFA_TEST_RTM_FAM_LONGCTS,
	EFA_TEST_RTM_LONGCTS_TAG =
		EFA_TEST_RTM_FAM_LONGCTS | EFA_TEST_RTM_TAGGED,
	EFA_TEST_RTM_DC_LONGCTS_MSG =
		EFA_TEST_RTM_FAM_LONGCTS | EFA_TEST_RTM_DC,
	EFA_TEST_RTM_DC_LONGCTS_TAG = EFA_TEST_RTM_FAM_LONGCTS |
				      EFA_TEST_RTM_TAGGED | EFA_TEST_RTM_DC,
	EFA_TEST_RTM_LONGREAD_MSG = EFA_TEST_RTM_FAM_LONGREAD,
	EFA_TEST_RTM_LONGREAD_TAG =
		EFA_TEST_RTM_FAM_LONGREAD | EFA_TEST_RTM_TAGGED,
	EFA_TEST_RTM_RUNTREAD_MSG = EFA_TEST_RTM_FAM_RUNTREAD,
	EFA_TEST_RTM_RUNTREAD_TAG =
		EFA_TEST_RTM_FAM_RUNTREAD | EFA_TEST_RTM_TAGGED,
	EFA_TEST_RTM_EAGER_MSG_ZERO_HDR =
		EFA_TEST_RTM_FAM_EAGER | EFA_TEST_RTM_ZERO_HDR,
};

/* Header fields read from the built packet */
struct efa_test_rtm_init_result {
	ssize_t ret; /* return code of the init function */
	int base_type;
	int has_msg_flag;
	int has_tagged_flag;
	int dc_requested;
	uint32_t msg_id;
	uint64_t tag;
	uint32_t send_id;
	uint32_t credit_request;
	uint32_t read_iov_count;
	uint64_t msg_length;
	uint64_t seg_offset;
	uint64_t runt_length;
	size_t payload_size;
};

/**
 * @brief Build one RTM packet via its init function and report header fields.
 *
 * Inserts a peer, constructs a txe populated with sentinels, allocates
 * a TX packet, calls the variant's init function, reads the result into @p out,
 * and releases the packet and txe.
 *
 * @param[in]	ep	the endpoint
 * @param[in]	av	the endpoint's AV (peer is inserted here)
 * @param[in]	variant	which init function and protocol family to exercise
 * @param[out]	out	populated header fields and init return code
 */
void efa_test_rtm_init_build(struct fid_ep *ep, struct fid_av *av,
			     enum efa_test_rtm_variant variant,
			     struct efa_test_rtm_init_result *out);

/** @brief rtm variant -> EFA_RDM_*_PKT lookup.
 */
int efa_test_rtm_pkt_type(enum efa_test_rtm_variant variant);

int efa_test_get_tx_min_credits(void);

/* --- TX sent / send-completion handler bridge --- */
enum efa_test_rtm_sent_op {
	EFA_TEST_RTM_OP_SENT,
	EFA_TEST_RTM_OP_COMPLETION,
};

struct efa_test_rtm_sent_result {
	uint64_t bytes_sent;
	uint64_t bytes_acked;
	uint64_t num_read_msg_in_flight;
	int64_t num_runt_bytes_in_flight;
	int send_completed;
	int cq_has_completion;
	int txe_on_ope_list;
};

/**
 * @brief Drive one TX sent/completion handler and report observable counters.
 *
 * @param[in]	ep	RDM efa endpoint
 * @param[in]	av	the endpoint's AV
 * @param[in]	variant	selects family/tagged/dc
 * @param[in]	op	sent vs send-completion handler
 * @param[in]	payload_size	bytes carried by this packet
 * @param[in]	bytes_already	txe->bytes_sent/bytes_acked before the call
 * @param[out]	out	observable counters after the call
 */
void efa_test_rtm_sent_build(struct fid_ep *ep, struct fid_av *av,
			     enum efa_test_rtm_variant variant,
			     enum efa_test_rtm_sent_op op, size_t payload_size,
			     size_t bytes_already,
			     struct efa_test_rtm_sent_result *out);

#define EFA_TEST_COPY_HMEM_HANDLE	0xabc0u
#define EFA_TEST_COPY_HMEM_DEVICE	0x77u
#define EFA_TEST_COPY_HMEM_DATA_SIZE	64u

enum efa_test_copy_src {
	EFA_TEST_COPY_SRC_HOST,
	EFA_TEST_COPY_SRC_CUDA,
	EFA_TEST_COPY_SRC_CUDA_GDRCOPY,
};

struct efa_test_copy_from_hmem_result {
	ssize_t ret;
	size_t payload_size;
	int payload_in_wiredata;
};

void efa_test_selects_copy_method(struct fid_ep *ep, struct fid_av *av,
				  size_t total_len, enum efa_test_copy_src src,
				  bool cuda_api_permitted,
				  struct efa_test_copy_from_hmem_result *out);

size_t efa_test_get_max_gdrcopy_from_hmem_size(void);
void efa_test_set_max_gdrcopy_from_hmem_size(size_t size);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_RDM_PKE_UTILS_H */
