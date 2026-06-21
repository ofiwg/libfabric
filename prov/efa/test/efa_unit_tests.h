/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_UNIT_TESTS_H
#define EFA_UNIT_TESTS_H

#define _GNU_SOURCE

#define MR_MODE_BITS FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_LOCAL

#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include "stdio.h"
#include "efa.h"
#include "efa_unit_test_mocks.h"

extern int g_ibv_ah_limit;
extern int g_ibv_ah_cnt;
extern int g_self_ah_cnt;
extern struct ibv_ah g_dummy_ah;

extern struct efa_mock_ibv_send_wr_list g_ibv_send_wr_list;
extern struct efa_unit_test_mocks g_efa_unit_test_mocks;
extern struct efa_env efa_env;

struct efa_resource {
	struct fi_info *hints;
	struct fi_info *info;
	struct fid_fabric *fabric;
	struct fid_domain *domain;
	struct fid_ep *ep;
	struct fid_eq *eq;
	struct fid_av *av;
	struct fid_cq *cq;
};

/* common functions in efa_unit_test_common.c */
struct fi_info *efa_unit_test_alloc_hints(enum fi_ep_type ep_type, char *fabric_name);
struct fi_info *efa_unit_test_alloc_hints_hmem(enum fi_ep_type ep_type, char *fabric_name);

void efa_unit_test_resource_construct(struct efa_resource *resource, enum fi_ep_type ep_type, char *fabric_name);
void efa_unit_test_resource_construct_ep_not_enabled(
	struct efa_resource *resource, enum fi_ep_type ep_type, char *fabric_name);
void efa_unit_test_resource_construct_no_cq_and_ep_not_enabled(
	struct efa_resource *resource, enum fi_ep_type ep_type, char *fabric_name);
void efa_unit_test_resource_construct_with_hints(struct efa_resource *resource,
						 enum fi_ep_type ep_type,
						 uint32_t fi_version, struct fi_info *hints,
						 bool enable_ep, bool open_cq);

void efa_unit_test_resource_construct_rdm_shm_disabled(struct efa_resource *resource);

void efa_unit_test_resource_destruct(struct efa_resource *resource);

void efa_unit_test_construct_msg(struct fi_msg *msg, struct iovec *iov,
				 size_t iov_count, fi_addr_t addr,
				 void *context, uint64_t data,
				 void **desc);

void efa_unit_test_construct_tmsg(struct fi_msg_tagged *tmsg, struct iovec *iov,
				  size_t iov_count, fi_addr_t addr,
				  void *context, uint64_t data,
				  void **desc, uint64_t tag,
				  uint64_t ignore);

void efa_unit_test_construct_msg_rma(struct fi_msg_rma *msg, struct iovec *iov,
				     void **desc, size_t iov_count,
				     fi_addr_t addr, struct fi_rma_iov *rma_iov,
				     size_t rma_iov_count, void *context,
				     uint64_t data);

void new_temp_file(char *template, size_t len);

struct efa_unit_test_buff {
	uint8_t *buff;
	size_t  size;
	struct fid_mr *mr;
};

struct efa_unit_test_eager_rtm_pkt_attr {
	uint32_t msg_id;
	uint32_t connid;
};

struct efa_unit_test_handshake_pkt_attr {
	uint32_t connid;
	uint64_t host_id;
	uint32_t device_version;
};

void efa_unit_test_buff_construct(struct efa_unit_test_buff *buff, struct efa_resource *resource, size_t buff_size);

void efa_unit_test_buff_destruct(struct efa_unit_test_buff *buff);

void efa_unit_test_eager_msgrtm_pkt_construct(struct efa_rdm_pke *pkt_entry, struct efa_unit_test_eager_rtm_pkt_attr *attr);

void efa_unit_test_construct_handshake_pkt_for_receive(struct efa_rdm_pke *pkt_entry, struct efa_unit_test_handshake_pkt_attr *attr);

struct efa_rdm_ope *efa_unit_test_alloc_txe(struct efa_resource *resource, uint32_t op);

struct efa_rdm_ope *efa_unit_test_alloc_rxe(struct efa_resource *resource, uint32_t op);

/* end of common functions in efa_unit_test_common.c */

/* test cases */

/* begin efa_unit_test_av.c */
void test_av_insert_duplicate_raw_addr(void **state);
void test_av_insert_duplicate_gid(void **state);
void test_efa_ah_cnt_one_av_efa(void **state);
void test_efa_ah_cnt_one_av_efa_direct(void **state);
void test_efa_ah_cnt_multi_av_efa(void **state);
void test_efa_ah_cnt_multi_av_efa_direct(void **state);
void test_av_multiple_ep_efa(void **state);
void test_av_multiple_ep_efa_direct(void **state);
void test_av_reinsertion(void **state);
void test_av_reverse_av_remove_qpn_collision(void **state);
void test_av_implicit(void **state);
void test_av_implicit_to_explicit(void **state);
void test_av_implicit_av_lru_insertion(void **state);
void test_av_implicit_av_lru_eviction(void **state);
void test_ah_refcnt(void **state);
void test_ah_lru_eviction_explicit_av_insert(void **state);
void test_ah_lru_eviction_implicit_av_insert(void **state);
/* end efa_unit_test_av.c */

void test_efa_device_construct_error_handling(void **state);
void test_efa_rdm_ep_ignore_missing_host_id_file(void **state);
void test_efa_rdm_ep_has_valid_host_id(void **state);
void test_efa_rdm_ep_ignore_short_host_id(void **state);
void test_efa_rdm_ep_ignore_non_hex_host_id(void **state);
void test_efa_rdm_ep_handshake_receive_and_send_valid_host_ids_with_connid(void **state);
void test_efa_rdm_ep_handshake_receive_and_send_valid_host_ids_without_connid(void **state);
void test_efa_rdm_ep_handshake_receive_valid_peer_host_id_and_do_not_send_local_host_id(void **state);
void test_efa_rdm_ep_handshake_receive_without_peer_host_id_and_do_not_send_local_host_id(void **state);
void test_efa_rdm_ep_getopt_undersized_optlen(void **state);
void test_efa_rdm_ep_getopt_oversized_optlen(void **state);
void test_efa_rdm_ep_tx_pkt_pool_flags(void **state);
void test_efa_rdm_ep_rx_pkt_pool_flags(void **state);
void test_efa_rdm_ep_pkt_pool_page_alignment(void **state);
void test_efa_rdm_ep_write_queue_before_handshake(void **state);
void test_efa_rdm_ep_read_queue_before_handshake(void **state);
void test_efa_rdm_ep_trigger_handshake(void **state);
void test_efa_rdm_txe_construct_splits_internal_flags(void **state);
void test_efa_rdm_read_copy_pkt_pool_128_alignment(void **state);
void test_efa_rdm_ep_send_with_shm_no_copy(void **state);
void test_efa_rdm_ep_rma_without_caps(void **state);
void test_efa_rdm_ep_atomic_without_caps(void **state);
void test_efa_rdm_ep_setopt_shared_memory_permitted(void **state);
void test_efa_rdm_ep_setopt_homogeneous_peers(void **state);
void test_efa_rdm_ep_enable_qp_in_order_aligned_128_bytes_good(void **state);
void test_efa_rdm_ep_enable_qp_in_order_aligned_128_bytes_bad(void **state);
void test_efa_rdm_ep_close_shm_resource_happy(void **state);
void test_efa_rdm_ep_close_shm_resource_unhappy(void **state);
void test_efa_rdm_ep_zcpy_recv_not_created_but_peer_flag_set(void **state);
void test_efa_rdm_ep_zcpy_compat_disabled_by_sas(void **state);
void test_efa_rdm_ep_handshake_receive_peer_user_recv_qp(void **state);
void test_efa_rdm_ep_handshake_receive_peer_no_user_recv_qp(void **state);
void test_efa_rdm_ep_post_handshake_error_handling_pke_exhaustion(void **state);
void test_efa_rdm_ep_rx_refill_threshold_smaller_than_rx_size(void **state);
void test_efa_rdm_ep_rx_refill_threshold_larger_than_rx_size(void **state);
void test_efa_rdm_ep_support_unsolicited_write_recv(void **state);
void test_efa_rdm_ep_default_sizes(void **state);
void test_efa_rdm_ep_outstanding_tx_ops_decremented_with_error_completion(void **state);
void test_efa_rdm_ep_get_explicit_shm_fi_addr(void **state);
void test_efa_rdm_ep_get_explicit_shm_fi_addr_no_shm(void **state);
void test_efa_base_ep_construct_ibv_qp_init_attr_ex_efa_direct_use_requested_limits(void **state);
void test_efa_base_ep_construct_ibv_qp_init_attr_ex_efa_use_requested_limits(void **state);
void test_efa_base_ep_construct_ibv_qp_init_attr_ex_efa_use_device_limits(void **state);
void test_efa_base_ep_construct_ibv_qp_init_attr_ex_dgram(void **state);
void test_dgram_cq_read_empty_cq(void **state);
void test_ibv_cq_ex_read_empty_cq(void **state);
void test_ibv_cq_ex_read_failed_poll(void **state);
void test_rdm_cq_create_error_handling(void **state);
void test_rdm_cq_read_bad_send_status_unresponsive_receiver(void **state);
void test_rdm_cq_read_bad_send_status_unresponsive_receiver_missing_peer_host_id(void **state);
void test_rdm_cq_read_bad_send_status_unreachable_receiver(void **state);
void test_rdm_cq_read_bad_send_status_invalid_qpn(void **state);
void test_rdm_cq_read_bad_send_status_message_too_long(void **state);
void test_rdm_cq_handshake_bad_send_status_bad_qpn(void **state);
void test_rdm_cq_handshake_bad_send_status_unresp_remote(void **state);
void test_rdm_cq_handshake_bad_send_status_unreach_remote(void **state);
void test_rdm_cq_handshake_bad_send_status_remote_abort(void **state);
void test_rdm_cq_handshake_bad_send_status_unsupported_op(void **state);
void test_ibv_cq_unsolicited_write_recv_status(void **state);
void test_ibv_cq_ex_read_bad_recv_status(void **state);
void test_ibv_cq_ex_read_bad_recv_rdma_with_imm_status_use_unsolicited_recv(void **state);
void test_ibv_cq_ex_read_bad_recv_rdma_with_imm_status_use_solicited_recv(void **state);
void test_ibv_cq_ex_read_recover_forgotten_peer_ah(void **state);
void test_rdm_fallback_to_ibv_create_cq_ex_cq_read_ignore_forgotton_peer(void **state);
void test_ibv_cq_ex_read_ignore_removed_peer(void **state);
void test_efa_rdm_cq_before_ep_enable(void **state);
void test_efa_rdm_cq_sread_no_wait_obj(void **state);
void test_efa_rdm_cq_sread_eagain(void **state);
void test_efa_rdm_cq_sread_with_cqe(void **state);

/* begin efa_unit_test_info.c */
void test_info_open_ep_with_wrong_info(void **state);
void test_info_rdm_attributes(void **state);
void test_info_dgram_attributes(void **state);
void test_info_direct_attributes_no_rma(void **state);
void test_info_direct_attributes_rma(void **state);
void test_info_direct_hmem_support_p2p(void **state);
void test_info_tx_rx_msg_order_rdm_order_none(void **state);
void test_info_tx_rx_msg_order_rdm_order_sas(void **state);
void test_info_tx_rx_msg_order_dgram_order_none(void **state);
void test_info_tx_rx_msg_order_dgram_order_sas(void **state);
void test_info_max_order_size_dgram_with_atomic(void **state);
void test_info_max_order_size_rdm_with_atomic_no_order(void **state);
void test_info_max_order_size_rdm_with_atomic_order(void **state);
void test_info_tx_rx_op_flags_rdm(void **state);
void test_info_tx_rx_size_rdm(void **state);
void test_info_check_shm_info_hmem(void **state);
void test_info_check_shm_info_op_flags(void **state);
void test_info_check_shm_info_threading(void **state);
void test_info_check_hmem_cuda_support_on_api_lt_1_18(void **state);
void test_info_check_hmem_cuda_support_on_api_ge_1_18(void **state);
void test_info_check_no_hmem_support_when_not_requested(void **state);
void test_info_hmem_supported_with_null_hints(void **state);
void test_info_hmem_not_advertised_with_null_hints_when_unsupported(void **state);
void test_info_hmem_requested_but_unsupported_returns_enodata(void **state);
void test_info_direct_unsupported(void **state);
void test_info_direct_ordering(void **state);
void test_info_reuse_fabric_via_fabric_attr(void **state);
void test_info_reuse_domain_via_domain_attr(void **state);
void test_info_reuse_fabric_via_name(void **state);
void test_info_reuse_domain_via_name(void **state);
void test_efa_hmem_info_p2p_dmabuf_assumed_neuron(void **state);
void test_efa_hmem_info_p2p_disabled_neuron(void **state);
void test_efa_hmem_info_p2p_disabled_synapse(void **state);
void test_efa_hmem_info_disable_p2p_cuda(void **state);
void test_efa_hmem_info_check_p2p_cuda_ctx_create_destroy_on_memalloc_fail(void **state);
void test_efa_nic_select_all_devices_matches(void);
void test_efa_nic_select_first_device_matches(void);
void test_efa_nic_select_first_device_with_surrounding_comma_matches(void);
void test_efa_nic_select_first_device_first_letter_no_match(void);
void test_efa_nic_select_empty_device_no_match(void);
void test_efa_use_device_rdma_env1_opt1(void **state);
void test_efa_use_device_rdma_env0_opt0(void **state);
void test_efa_use_device_rdma_env1_opt0(void **state);
void test_efa_use_device_rdma_env0_opt1(void **state);
void test_efa_use_device_rdma_opt1(void **state);
void test_efa_use_device_rdma_opt0(void **state);
void test_efa_use_device_rdma_env1(void **state);
void test_efa_use_device_rdma_env0(void **state);
void test_efa_use_device_rdma_opt_old(void **state);
void test_info_direct_null_hints_return_rma_and_rx_cq_data(void **state);
void test_info_direct_rma_with_rx_cq_data_when_no_unsolicited_write_recv(void **state);
void test_info_direct_rma_without_rx_cq_data_when_no_unsolicited_write_recv(void **state);
void test_info_direct_no_rma_no_rx_cq_data_when_no_unsolicited_write_recv(void **state);
void test_info_direct_rma_without_rx_cq_data_when_unsolicited_write_recv_supported(void **state);
void test_info_direct_msg_only_small_max_msg_size_success(void **state);
void test_info_direct_msg_only_large_max_msg_size_fail(void **state);
void test_info_direct_msg_rma_large_max_msg_size_success(void **state);
void test_info_direct_msg_rma_too_large_max_msg_size_fail(void **state);
void test_info_max_cntr_value_api_lt_2_5(void **state);
void test_info_max_cntr_value_api_ge_2_5_within_hw_range(void **state);
void test_info_max_cntr_value_api_ge_2_5_hint_within_hw_range(void **state);
void test_info_max_cntr_value_api_ge_2_5_above_hw_range(void **state);
void test_info_rdm_max_cntr_value_api_ge_2_5_within_hw_range(void **state);
void test_info_direct_inject_size_no_hint(void **state);
void test_info_direct_inject_size_small(void **state);
void test_info_direct_inject_size_wide_wqe(void **state);
void test_info_direct_inject_size_exceeds_max(void **state);
void test_ep_getopt_inject_size_regular_wqe(void **state);
void test_ep_getopt_inject_size_wide_wqe(void **state);
/* end efa_unit_test_info.c */

void test_efa_srx_min_multi_recv_size(void **state);
void test_efa_srx_cq(void **state);
void test_efa_srx_lock(void **state);
void test_efa_srx_unexp_pkt(void **state);
void test_efa_srx_foreach_unspec_skips_other_provider(void **state);
void test_efa_rdm_peer_construct_robuf_failure(void **state);
void test_efa_rnr_queue_and_resend_msg(void **state);
void test_efa_rnr_queue_and_resend_tagged(void **state);
void test_efa_rdm_ep_post_queued_pkts_releases_pkt_on_error(void **state);

/* begin of efa_unit_test_ope.c */
void test_efa_rdm_ope_prepare_to_post_send_with_no_enough_tx_pkts(void **state);
void test_efa_rdm_ope_prepare_to_post_send_host_memory(void **state);
void test_efa_rdm_ope_prepare_to_post_send_host_memory_align128(void **state);
void test_efa_rdm_ope_prepare_to_post_send_cuda_memory(void **state);
void test_efa_rdm_ope_prepare_to_post_send_cuda_memory_align128(void **state);
void test_efa_rdm_ope_post_write_0_byte_no_shm(void);
void test_efa_rdm_rxe_post_local_read_or_queue_unhappy(void **state);
void test_efa_rdm_rxe_post_local_read_or_queue_happy(void **state);
void test_efa_rdm_rxe_post_local_read_or_queue_clone_error(void **state);
void test_efa_rdm_txe_handle_error_write_cq(void **state);
void test_efa_rdm_txe_handle_error_not_write_cq(void **state);
void test_efa_rdm_txe_handle_error_suppressed_write(void **state);
void test_efa_rdm_txe_handle_error_suppressed_read(void **state);
void test_efa_rdm_txe_handle_error_suppressed_send(void **state);
void test_efa_rdm_txe_handle_error_inject_still_reports_cq_error(void **state);
void test_efa_rdm_rxe_handle_error_write_cq(void **state);
void test_efa_rdm_rxe_handle_error_not_write_cq(void **state);
void test_efa_rdm_rxe_map(void **state);
void test_efa_rdm_rxe_list_removal(void **state);
void test_efa_rdm_txe_list_removal(void **state);
void test_efa_rdm_txe_prepare_local_read_pkt_entry(void **state);
void test_efa_rdm_txe_handle_error_queue_flags_cleanup(void **state);
void test_efa_rdm_rxe_handle_error_queue_flags_cleanup(void **state);
void test_efa_rdm_txe_handle_error_duplicate_prevention(void **state);
void test_efa_rdm_rxe_handle_error_duplicate_prevention(void **state);
void test_efa_rdm_ope_receipt_packet_tracking_cq_read(void **state);
void test_efa_rdm_ope_receipt_packet_tracking_wait_send(void **state);
void test_efa_rdm_ope_receipt_packet_failed_posting(void **state);
void test_efa_rdm_ope_receipt_packet_tracking_unresponsive_wait_send(void **state);
void test_efa_rdm_ope_eor_packet_tracking_cq_read(void **state);
void test_efa_rdm_ope_eor_packet_tracking_wait_send(void **state);
void test_efa_rdm_ope_eor_packet_failed_posting(void **state);
void test_efa_rdm_ope_eor_packet_tracking_unresponsive_wait_send(void **state);
void test_efa_rdm_rxe_peer_abort_writes_error_completion_at_drain(void **state);
void test_efa_rdm_rxe_mark_peer_aborted_multi_recv_writes_err(void **state);
void test_efa_rdm_pke_handle_tx_error_longread_bad_address_peer_aborts(void **state);
void test_efa_rdm_pke_handle_tx_error_longread_abort_peer_aborts(void **state);
void test_efa_rdm_pke_handle_tx_error_longread_bad_length_writes_cq_err(void **state);
void test_efa_rdm_pke_handle_tx_error_longread_tagged_peer_aborts(void **state);
void test_efa_rdm_pke_handle_rma_read_completion_drains_recovered_rxe(void **state);
void test_efa_rdm_rxe_emit_peer_error_emits_pkt(void **state);
void test_efa_rdm_rxe_emit_peer_error_multi_recv_emits_pkt(void **state);
void test_efa_rdm_rxe_emit_peer_error_skips_on_peer_ep_closed(void **state);
void test_efa_rdm_rxe_emit_peer_error_with_homogeneous_peers(void **state);
void test_efa_rdm_rxe_emit_peer_error_skips_when_no_handshake(void **state);
void test_efa_rdm_pke_handle_send_completion_peer_error_releases_rxe(void **state);
void test_efa_rdm_pke_handle_tx_error_peer_error_pkt_releases_rxe(void **state);
void test_efa_rdm_rxe_emit_peer_error_reentry_safe(void **state);
void test_efa_rdm_pke_handle_tx_error_sibling_read_wr_does_not_release_rxe(void **state);
void test_efa_rdm_pke_handle_peer_error_recv_longread_fails_txe(void **state);
void test_efa_rdm_pke_handle_peer_error_recv_longcts_reaps_rxe(void **state);
void test_efa_rdm_pke_handle_peer_error_recv_longcts_tagged(void **state);
void test_efa_rdm_pke_handle_peer_error_recv_invalid_op_id_dropped(void **state);
void test_efa_rdm_txe_handle_error_emits_peer_error_on_invalid_lkey(void **state);
void test_efa_rdm_txe_handle_error_emits_peer_error_on_canceled(void **state);
void test_efa_rdm_txe_handle_error_no_emit_when_peer_unsupported(void **state);
void test_efa_rdm_txe_handle_error_emits_peer_error_with_homogeneous_peers(void **state);
void test_efa_rdm_txe_handle_error_skips_peer_error_when_no_handshake(void **state);
void test_efa_rdm_pke_handle_peer_error_recv_longcts_cts_outstanding(void **state);
void test_efa_rdm_pke_handle_tx_error_longcts_abort_drains_txe(void **state);
void test_efa_rdm_pke_handle_peer_error_recv_medium_reaps_rxe(void **state);
void test_efa_rdm_pke_handle_peer_error_recv_medium_tagged(void **state);
void test_efa_rdm_pke_handle_peer_error_recv_medium_msg_id_not_found_dropped(void **state);
void test_efa_rdm_pke_handle_peer_error_recv_medium_unexpected_tears_down(void **state);
void test_efa_rdm_txe_handle_error_no_emit_when_not_longcts(void **state);
void test_efa_rdm_atomic_compare_desc_persistence(void **state);
void test_efa_rdm_txe_dc_ctsdata_send_first(void **state);
void test_efa_rdm_txe_dc_ctsdata_resp_first(void **state);
void test_efa_rdm_txe_dc_eager_rtm_send_first(void **state);
void test_efa_rdm_txe_dc_eager_rtm_resp_first(void **state);
void test_efa_rdm_txe_short_rtr_send_first(void **state);
void test_efa_rdm_txe_short_rtr_resp_first(void **state);
void test_efa_rdm_txe_fetch_rta_send_first(void **state);
void test_efa_rdm_txe_fetch_rta_resp_first(void **state);
void test_efa_rdm_txe_compare_rta_send_first(void **state);
void test_efa_rdm_txe_compare_rta_resp_first(void **state);
void test_efa_rdm_txe_longread_tagrtm_send_first(void **state);
void test_efa_rdm_txe_longread_tagrtm_resp_first(void **state);
void test_efa_rdm_txe_longread_msgrtm_send_first(void **state);
void test_efa_rdm_txe_longread_msgrtm_resp_first(void **state);
void test_efa_rdm_txe_longread_rtw_send_first(void **state);
void test_efa_rdm_txe_longread_rtw_resp_first(void **state);
void test_efa_rdm_rxe_longcts_msg_cts_send_first(void **state);
void test_efa_rdm_rxe_longcts_msg_cts_recv_first(void **state);
void test_efa_rdm_rxe_longcts_write_cts_send_first(void **state);
void test_efa_rdm_rxe_longcts_write_cts_recv_first(void **state);
void test_efa_rdm_txe_longcts_read_cts_send_first(void **state);
void test_efa_rdm_txe_longcts_read_cts_recv_first(void **state);
void test_efa_rdm_rxe_dc_longcts_write_cts_before_receipt(void **state);
void test_efa_rdm_rxe_dc_longcts_write_receipt_before_cts(void **state);


/* end of efa_unit_test_ope.c */
void test_efa_rdm_msg_send_to_local_peer_with_null_desc(void **state);
void test_efa_fork_support_request_initialize_when_ibv_fork_support_is_needed(void **state);
void test_efa_fork_support_request_initialize_when_ibv_fork_support_is_unneeded(void **state);
void test_efa_rdm_peer_get_runt_size_no_enough_runt(void **state);
void test_efa_rdm_peer_get_runt_size_cuda_memory_smaller_than_alignment(void **state);
void test_efa_rdm_peer_get_runt_size_cuda_memory_exceeding_total_len(void **state);
void test_efa_rdm_peer_get_runt_size_cuda_memory_normal(void **state);
void test_efa_rdm_peer_get_runt_size_host_memory_smaller_than_alignment(void **state);
void test_efa_rdm_peer_get_runt_size_host_memory_exceeding_total_len(void **state);
void test_efa_rdm_peer_get_runt_size_host_memory_normal(void **state);
void test_efa_rdm_peer_get_runt_size_cuda_memory_128_multiple_alignment(void **state);
void test_efa_rdm_peer_get_runt_size_cuda_memory_non_128_multiple_alignment(void **state);
void test_efa_rdm_peer_get_runt_size_cuda_memory_smaller_than_128_alignment(void **state);
void test_efa_rdm_peer_get_runt_size_cuda_memory_exceeding_total_len_128_alignment(void **state);
void test_efa_rdm_peer_select_readbase_rtm_no_runt(void **state);
void test_efa_rdm_peer_select_readbase_rtm_do_runt(void **state);
void test_efa_rdm_pke_get_available_copy_methods_align128(void **state);

/* begin efa_unit_test_domain.c */
void test_efa_domain_info_type_efa_direct(void **state);
void test_efa_domain_direct_has_bounce_buffer(void **state);
void test_efa_domain_no_bounce_buffer_without_fi_rma_cap_requested(void **state);
void test_efa_domain_bounce_buffer_with_rdma(void **state);
void test_efa_domain_open_ops_wrong_name(void **state);
void test_efa_domain_open_ops_mr_query(void **state);
void test_efa_domain_dgram_attr_mr_allocated(void **state);
void test_efa_domain_direct_attr_mr_allocated(void **state);
void test_efa_domain_open_ops_query_addr(void **state);
void test_efa_domain_open_ops_query_qp_wqs(void **state);
void test_efa_domain_open_ops_query_cq(void **state);
void test_efa_domain_open_ops_cq_open_ext(void **state);
void test_efa_domain_open_ops_cntr_open_ext(void **state);
void test_efa_domain_open_ops_get_mr_lkey(void **state);
void test_efa_domain_direct_mr_ops(void **state);
void test_efa_domain_dgram_mr_ops(void **state);
void test_efa_domain_open_installs_base_domain_ops_efa_direct(void **state);
void test_efa_domain_open_installs_base_domain_ops_dgram(void **state);
void test_efa_domain_gda_ops_rejected_for_dgram(void **state);
/* end efa_unit_test_domain.c */

/* begin efa_unit_test_fabric.c */
void test_efa_fabric_open_ops_feature_known(void **state);
void test_efa_fabric_open_ops_feature_not_on_proto(void **state);
void test_efa_fabric_open_ops_feature_unknown(void **state);
/* end efa_unit_test_fabric.c */

/* begin efa_unit_test_rdm_domain.c */
void test_efa_domain_info_type_efa_rdm(void **state);
void test_efa_domain_rdm_no_bounce_buffer(void **state);
void test_efa_domain_rdm_attr_mr_allocated(void **state);
void test_efa_domain_peer_list_cleared(void **state);
void test_efa_domain_rdm_mr_ops(void **state);
void test_efa_domain_mr_cache_enabled(void **state);
void test_efa_domain_mr_cache_disabled_with_mr_local(void **state);
void test_efa_rdm_domain_open_installs_rdm_domain_ops(void **state);
void test_efa_domain_gda_ops_rejected_for_rdm(void **state);
/* end efa_unit_test_rdm_domain.c */

void test_efa_rdm_cq_ibv_cq_poll_list_same_tx_rx_cq_single_ep(void **state);
void test_efa_rdm_cq_ibv_cq_poll_list_separate_tx_rx_cq_single_ep(void **state);
void test_efa_rdm_cq_post_initial_rx_pkts(void **state);
void test_efa_rdm_cntr_ibv_cq_poll_list_same_tx_rx_cq_single_ep(void **state);
void test_efa_rdm_cntr_ibv_cq_poll_list_separate_tx_rx_cq_single_ep(void **state);
void test_efa_rdm_cntr_post_initial_rx_pkts(void **state);
void test_efa_rdm_cntr_read_before_ep_enable(void **state);
void test_efa_hw_cntr_open_unsupported_type_bytes(void **state);
void test_efa_hw_cntr_open_max_cntr_value_exceeded(void **state);
void test_efa_hw_cntr_open_ibv_fail(void **state);
void test_efa_hw_cntr_add(void **state);
void test_efa_hw_cntr_adderr(void **state);
void test_efa_hw_cntr_set(void **state);
void test_efa_hw_cntr_seterr(void **state);
void test_efa_hw_cntr_read(void **state);
void test_efa_hw_cntr_readerr(void **state);
void test_efa_hw_cntr_bind_ep(void **state);
void test_efa_hw_cntr_bind_ep_attach_fail(void **state);
void test_efa_hw_cntr_wait_success(void **state);
void test_efa_hw_cntr_wait_returns_einval_with_wait_none(void **state);
void test_efa_hw_cntr_open_returns_eopnotsupp_with_wait_fd(void **state);
void test_efa_hw_cntr_open_returns_eopnotsupp_with_wait_yield(void **state);
void test_efa_cntr_open_uses_hw_cntr(void **state);
void test_efa_hw_cntr_open_use_hw_cntr_disabled(void **state);

/* begin of efa_unit_test_rdm_peer.c */
void test_efa_rdm_peer_reorder_expected_msg_id(void **state);
void test_efa_rdm_peer_reorder_smaller_msg_id(void **state);
void test_efa_rdm_peer_reorder_larger_msg_id(void **state);
void test_efa_rdm_peer_reorder_overflow_msg_id(void **state);
void test_efa_rdm_peer_abort_ooo_msg_overflow_multi_segment(void **state);
void test_efa_rdm_peer_move_overflow_pke_to_recvwin(void **state);
void test_efa_rdm_peer_keep_pke_in_overflow_list(void **state);
void test_efa_rdm_peer_append_overflow_pke_to_recvwin(void **state);
void test_efa_rdm_peer_recvwin_queue_or_append_pke(void **state);
void test_efa_rdm_peer_destruct_clears_rnr_flag(void **state);
void test_efa_rdm_peer_abort_ooo_in_overflow(void **state);
void test_efa_rdm_peer_abort_ooo_in_recvwin(void **state);
void test_efa_rdm_peer_abort_ooo_miss(void **state);
void test_efa_rdm_peer_abort_ooo_recvwin_drain_progresses(void **state);
void test_efa_rdm_peer_skip_aborted_msg_id_never_arrived_unblocks_window(void **state);
void test_efa_rdm_peer_skip_aborted_msg_id_head_advances(void **state);
void test_efa_rdm_peer_skip_aborted_msg_id_already_processed_noop(void **state);
void test_efa_rdm_peer_skip_aborted_msg_id_buffered_abort_markers(void **state);
void test_efa_rdm_peer_skip_aborted_msg_id_abort_marker_behind_head(void **state);
/* end of efa_unit_test_rdm_peer.c */

/* begin of efa_unit_test_pke.c */
void test_efa_rdm_pke_handle_send_completion_peer_removed(void **state);
void test_efa_rdm_pke_handle_tx_error_peer_removed(void **state);
void test_efa_rdm_pke_handle_longcts_rtm_send_completion(void **state);
void test_efa_rdm_pke_release_rx_list(void **state);
void test_efa_rdm_pke_alloc_rta_rxe(void **state);
void test_efa_rdm_pke_alloc_rtw_rxe(void **state);
void test_efa_rdm_pke_alloc_rtr_rxe(void **state);
void test_efa_rdm_pke_get_unexp(void **state);
void test_efa_rdm_pke_flag_tracking(void **state);
void test_efa_rdm_pke_proc_matched_eager_rtm_error(void **state);
void test_efa_rdm_pke_proc_matched_mulreq_rtm_first_packet_error(void **state);
void test_efa_rdm_pke_proc_matched_mulreq_rtm_second_packet_error(void **state);
void test_efa_rdm_pke_flush_queued_blocking_copy_to_hmem_copy_size_mismatch(void **state);
void test_efa_rdm_prov_errno_is_peer_abort(void **state);
void test_efa_rdm_pkt_is_rxe_remote_read(void **state);
void test_efa_rdm_pke_init_peer_error_for_ope_ope_index(void **state);
/* end of efa_unit_test_pke.c */

void test_efa_msg_fi_recv(void **state);
void test_efa_msg_fi_recvv(void **state);
void test_efa_msg_fi_recvmsg(void **state);
void test_efa_msg_fi_send(void **state);
void test_efa_msg_fi_sendv(void **state);
void test_efa_msg_fi_sendmsg(void **state);
void test_efa_msg_fi_senddata(void **state);
void test_efa_msg_fi_inject(void **state);
void test_efa_msg_fi_injectdata(void **state);
void test_efa_rma_read(void **state);
void test_efa_rma_readv(void **state);
void test_efa_rma_readmsg(void **state);
void test_efa_rma_write(void **state);
void test_efa_rma_writev(void **state);
void test_efa_rma_writemsg(void **state);
void test_efa_rma_writedata(void **state);
void test_efa_rma_inject_write(void **state);
void test_efa_rma_inject_writedata(void **state);
void test_efa_rma_writemsg_with_inject(void **state);
void test_efa_rma_writemsg_with_wide_wqe_inject(void **state);
void test_efa_rma_read_0_byte(void **state);
void test_efa_rma_readv_0_byte(void **state);
void test_efa_rma_readmsg_0_byte(void **state);
void test_efa_rma_write_0_byte(void **state);
void test_efa_rma_writev_0_byte(void **state);
void test_efa_rma_writemsg_0_byte(void **state);
void test_efa_rma_writedata_0_byte(void **state);
void test_efa_rma_inject_write_0_byte(void **state);
void test_efa_rma_inject_writedata_0_byte(void **state);
void test_efa_rma_write_0_byte_with_inject_flag(void **state);
void test_efa_rdm_rma_write_0_byte_with_inject_flag(void **state);
void test_efa_msg_send_0_byte(void **state);
void test_efa_msg_sendv_0_byte(void **state);
void test_efa_msg_sendmsg_0_byte(void **state);
void test_efa_msg_senddata_0_byte(void **state);
void test_efa_msg_inject_0_byte(void **state);
void test_efa_msg_injectdata_0_byte(void **state);
void test_efa_msg_send_0_byte_with_inject_flag(void **state);
void test_efa_msg_sendmsg_inject_with_hmem_fails(void **state);
void test_efa_msg_sendmsg_multi_iov_second_desc_hmem_fails(void **state);
void test_efa_msg_sendmsg_inject_with_large_msg_fails(void **state);
void test_efa_msg_inject_with_large_msg_fails(void **state);
void test_efa_rdm_msg_send_0_byte_with_inject_flag(void **state);
void test_efa_rdm_msg_send_0_byte_no_shm(void **state);
void test_efa_rdm_msg_sendv_0_byte_no_shm(void **state);
void test_efa_rdm_msg_sendmsg_0_byte_no_shm(void **state);
void test_efa_rdm_msg_senddata_0_byte_no_shm(void **state);
void test_efa_rdm_msg_inject_0_byte_no_shm(void **state);
void test_efa_rdm_msg_injectdata_0_byte_no_shm(void **state);
void test_efa_rdm_tagged_send_0_byte_no_shm(void **state);
void test_efa_rdm_tagged_sendv_0_byte_no_shm(void **state);
void test_efa_rdm_tagged_sendmsg_0_byte_no_shm(void **state);
void test_efa_rdm_tagged_senddata_0_byte_no_shm(void **state);
void test_efa_rdm_tagged_inject_0_byte_no_shm(void **state);
void test_efa_rdm_tagged_injectdata_0_byte_no_shm(void **state);
void test_efa_rdm_rma_read_0_byte_no_shm(void **state);
void test_efa_rdm_rma_readv_0_byte_no_shm(void **state);
void test_efa_rdm_rma_readmsg_0_byte_no_shm(void **state);
void test_efa_rdm_rma_write_0_byte_no_shm(void **state);
void test_efa_rdm_rma_writev_0_byte_no_shm(void **state);
void test_efa_rdm_rma_writemsg_0_byte_no_shm(void **state);
void test_efa_rdm_rma_writedata_0_byte_no_shm(void **state);
void test_efa_rdm_rma_inject_write_0_byte_no_shm(void **state);
void test_efa_rdm_rma_inject_writedata_0_byte_no_shm(void **state);
void test_efa_cq_read_no_completion(void **state);
void test_efa_cq_read_send_success(void **state);
void test_efa_cq_read_senddata_success(void **state);
void test_efa_cq_read_read_success(void **state);
void test_efa_cq_read_write_success(void **state);
void test_efa_cq_read_writedata_success(void **state);
void test_efa_cq_read_recv_success(void **state);
void test_efa_cq_read_recv_rdma_with_imm_success(void **state);
void test_efa_cq_read_send_failure(void **state);
void test_efa_cq_read_recv_failure(void **state);
void test_efa_cq_recv_rdma_with_imm_failure(void **state);
void test_efa_cq_data_path_direct_disabled_by_env(void **state);
void test_efa_cq_data_path_direct_disabled_with_old_device(void **state);
void test_efa_cq_data_path_direct_enabled_with_new_device(void **state);
void test_efa_cq_data_path_direct_with_wait_obj(void **state);
void test_efa_rdm_cq_data_path_direct_disabled_with_old_device(void **state);
void test_efa_rdm_cq_data_path_direct_enabled_with_new_device(void **state);
void test_efa_cq_trywait_no_channel(void **state);
void test_efa_cq_trywait_completions_available(void **state);
void test_efa_cq_trywait_success(void **state);
void test_efa_cq_sread_enosys(void **state);
void test_efa_cq_sread_eagain(void **state);
void test_efa_cq_control_getwait_with_channel(void **state);
void test_efa_cq_control_getwait_no_channel(void **state);
void test_efa_cq_control_getwaitobj(void **state);
void test_efa_cq_control_invalid_command(void **state);
void test_efa_cq_ep_list_lock_type_no_op(void **state);
void test_efa_cq_ep_list_lock_type_mutex(void **state);

void test_efa_cq_ops_override_with_counter_binding(void **state);
void test_efa_cq_readfrom_input_validation(void **state);
void test_efa_cq_readerr_return_value_user_buffer(void **state);
void test_efa_cq_readerr_return_value_provider_buffer(void **state);
void test_efa_cq_readfrom_start_poll_error(void **state);
void test_efa_cq_readfrom_util_cq_entries(void **state);
void test_efa_cq_readerr_util_cq_error(void **state);
void test_efa_cq_poll_ep_close_bypass_path(void **state);
void test_efa_cq_next_poll_stale_cur_wq_segv_on_ep_close(void **state);
void test_efa_cq_read_mixed_success_error(void **state);
void test_efa_cq_close_returns_ebusy_with_bound_ep(void **state);
void test_efa_ep_open(void **state);
void test_efa_ep_cancel(void **state);
void test_efa_ep_getopt(void **state);
void test_efa_ep_setopt_use_device_rdma(void **state);
void test_efa_ep_setopt_hmem_p2p(void **state);
void test_efa_ep_setopt_rnr_retry(void **state);
void test_efa_ep_setopt_sizes(void **state);
void test_efa_ep_bind_and_enable(void **state);
void test_efa_ep_data_path_direct_equal_to_cq_data_path_direct_happy(void **state);
void test_efa_ep_data_path_direct_equal_to_cq_data_path_direct_unhappy(void **state);
void test_efa_rdm_ep_data_path_direct_equal_to_cq_data_path_direct_happy(void **state);
void test_efa_rdm_ep_data_path_direct_equal_to_cq_data_path_direct_unhappy(void **state);
void test_efa_ep_lock_type_no_op(void **state);
void test_efa_ep_lock_type_mutex(void **state);
void test_efa_rdm_ep_shm_ep_different_info(void **state);
void test_efa_base_ep_construct_info_and_util_ep_initialized(void **state);
void test_efa_base_ep_disable_unsolicited_write_recv_with_rx_cq_data(void **state);
void test_efa_rdm_ep_enable_ah_alloc_failure(void **state);
void test_efa_rdm_ep_ibv_create_ah_failure(void **state);
void test_efa_rdm_ep_setopt_cq_flow_control(void **state);
void test_efa_direct_ep_setopt_cq_flow_control_no_rx_cq_data(void **state);
void test_efa_direct_ep_setopt_cq_flow_control_with_rx_cq_data(void **state);

/* begin efa_unit_test_data_path_direct.c */
void test_efa_data_path_direct_rdma_read_multiple_sge_fail(void **state);
void test_efa_data_path_direct_rdma_write_multiple_sge_fail(void **state);
void test_efa_data_path_direct_qp_gen_initialization(void **state);
void test_efa_data_path_direct_dev_req_id_roundtrip(void **state);
void test_efa_data_path_direct_stale_completion_detected(void **state);
void test_efa_data_path_direct_qp_gen_increments_across_qps(void **state);
void test_efa_data_path_direct_write_high_pps_hint_set(void **state);
/* end efa_unit_test_data_path_direct.c */


void test_efa_cntr_ibv_cq_poll_list_same_tx_rx_cq_single_ep(void **state);
void test_efa_cntr_ibv_cq_poll_list_separate_tx_rx_cq_single_ep(void **state);

/* begin efa_unit_test_mr.c */
void test_efa_rdm_mr_reg_host_memory(void **state);
void test_efa_rdm_mr_reg_host_memory_no_mr_local(void **state);
void test_efa_rdm_mr_reg_host_memory_overlapping_buffers(void **state);
void test_efa_rdm_mr_reg_cuda_memory(void **state);
void test_efa_rdm_mr_reg_cuda_memory_non_p2p(void **state);
void test_efa_direct_mr_reg_cuda_memory(void **state);
void test_efa_direct_mr_reg_rdma_read_not_supported(void **state);
void test_efa_direct_mr_reg_rdma_write_not_supported(void **state);
void test_efa_mr_validate_regattr_invalid_iov_count(void **state);
void test_efa_mr_validate_regattr_uninitialized_iface(void **state);
void test_efa_rdm_mr_structure_casting(void **state);
void test_efa_mr_attr_init_system_macro(void **state);
void test_efa_mr_ofi_to_ibv_access_no_access(void **state);
void test_efa_mr_ofi_to_ibv_access_one_flag(void **state);
void test_efa_mr_ofi_to_ibv_access_read_not_supported(void **state);
void test_efa_mr_ofi_to_ibv_access_write_not_supported(void **state);
void test_efa_mr_ofi_to_ibv_access_remote_read_write_read_only_supported(void **state);
void test_efa_mr_ofi_to_ibv_access_all_flags_supported(void **state);
void test_efa_mr_ofi_to_ibv_access_all_flags_not_supported(void **state);
void test_efa_mr_close_warn_outstanding_direct_ope(void **state);
void test_efa_direct_ope_released_on_recv_error(void **state);
void test_efa_direct_ope_released_on_send_error(void **state);
void test_efa_direct_ope_released_on_read_error(void **state);
void test_efa_direct_ope_released_on_write_error(void **state);
void test_efa_mr_close_warn_outstanding_direct_ope_multi_ep(void **state);
void test_efa_mr_close_warn_outstanding_rdm_txe(void **state);
void test_efa_rdm_mr_gen_bumps_on_close(void **state);
void test_efa_rdm_mr_gen_check_ope_skips_rxe_queued_ctrl_cts(void **state);
void test_efa_rdm_mr_gen_capture_not_overwritten_on_repost(void **state);
void test_efa_rdm_mr_gen_check_ope_detects_closed_mr(void **state);
void test_efa_direct_fi_send_with_closed_mr_no_crash(void **state);
void test_efa_rdm_mr_gen_check_cancels_rnr_queued_ope(void **state);
void test_efa_rdm_mr_gen_check_cancels_longcts_ope(void **state);
void test_efa_rdm_mr_cache_regv_no_cache(void **state);
void test_efa_rdm_mr_cache_regv_with_cache(void **state);
void test_efa_rdm_mr_cache_regv_cache_hit(void **state);
void test_efa_rdm_mr_cache_encapsulation_smaller(void **state);
void test_efa_rdm_mr_cache_non_overlapping(void **state);
void test_efa_rdm_mr_cache_lru_behavior(void **state);
void test_efa_rdm_mr_cache_flush_behavior(void **state);
void test_efa_rdm_mr_cache_reference_counting(void **state);

void test_efa_mr_reg_out_of_range_iface(void **state);
/* end efa_unit_test_mr.c */

/* begin efa_unit_test_rdm_rma.c */
void test_efa_rdm_rma_should_write_using_rdma_remote_cq_data_multiple_iovs_returns_false(void **state);
void test_efa_rdm_rma_should_write_using_rdma_remote_cq_data_multiple_rma_iovs_returns_false(void **state);
void test_efa_rdm_rma_should_write_using_rdma_use_device_rdma_false_returns_false(void **state);
void test_efa_rdm_rma_should_write_using_rdma_peer_no_rdma_write_support_returns_false(void **state);
void test_efa_rdm_rma_should_write_using_rdma_no_p2p_support_returns_false(void **state);
void test_efa_rdm_rma_should_write_using_rdma_p2p_and_rdma_write_support_returns_true(void **state);
void test_efa_rdm_rma_should_write_using_rdma_remote_cq_data_single_iovs_with_rdma_support(void **state);
void test_efa_rdm_rma_should_write_using_rdma_unsolicited_write_recv_not_match(void **state);
void test_efa_rdm_rma_post_remote_write_partial_fail_no_txe_release(void **state);
void test_efa_rdm_rma_post_remote_read_partial_fail_no_txe_release(void **state);
void test_efa_rdm_rma_partial_post_retry_no_double_free(void **state);
void test_efa_rdm_rma_partial_post_retry_no_double_free_read(void **state);
void test_efa_rdm_msg_send_multi_pkt_sendv_fail_no_inflight(void **state);
void test_efa_ibv_post_write_processing_hints_with_high_pps(void **state);
void test_efa_ibv_post_write_processing_hints_without_high_pps(void **state);
/* end efa_unit_test_rdm_rma.c */

static inline
int efa_unit_test_get_dlist_length(struct dlist_entry *head)
{
	int i = 0;
	struct dlist_entry *item;

	dlist_foreach(head, item) {
		i++;
	}

	return i;
}

/*
 * efa-rdm tx and rx entries are tracked on the shared base_ep.ope_list,
 * differentiated by efa_rdm_ope::type. These helpers count / fetch entries
 * of a given type (EFA_RDM_TXE or EFA_RDM_RXE) so tests can reason about
 * tx and rx entries independently.
 */
static inline
int efa_unit_test_get_ope_list_length(struct efa_rdm_ep *ep,
				      enum efa_rdm_ope_type type)
{
	int i = 0;
	struct dlist_entry *item;
	struct efa_rdm_ope *ope;

	dlist_foreach(&ep->base_ep.ope_list, item) {
		ope = container_of(item, struct efa_rdm_ope, ep_entry);
		if (ope->type == type)
			i++;
	}

	return i;
}

static inline
struct efa_rdm_ope *efa_unit_test_get_first_ope(struct efa_rdm_ep *ep,
						enum efa_rdm_ope_type type)
{
	struct dlist_entry *item;
	struct efa_rdm_ope *ope;

	dlist_foreach(&ep->base_ep.ope_list, item) {
		ope = container_of(item, struct efa_rdm_ope, ep_entry);
		if (ope->type == type)
			return ope;
	}

	return NULL;
}

void efa_unit_test_rdm_0byte_prep(struct efa_resource *resource, fi_addr_t *addr);

#endif
