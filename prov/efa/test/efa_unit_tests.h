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

struct fi_info *efa_unit_test_alloc_hints(enum fi_ep_type ep_type);

void efa_unit_test_resource_construct(struct efa_resource *resource, enum fi_ep_type ep_type);
void efa_unit_test_resource_construct_ep_not_enabled(
	struct efa_resource *resource, enum fi_ep_type ep_type);
void efa_unit_test_resource_construct_no_cq_and_ep_not_enabled(
	struct efa_resource *resource, enum fi_ep_type ep_type);
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

int efa_device_construct(struct efa_device *efa_device,
			 int device_idx,
			 struct ibv_device *ibv_device);

void efa_unit_test_buff_construct(struct efa_unit_test_buff *buff, struct efa_resource *resource, size_t buff_size);

void efa_unit_test_buff_destruct(struct efa_unit_test_buff *buff);

void efa_unit_test_eager_msgrtm_pkt_construct(struct efa_rdm_pke *pkt_entry, struct efa_unit_test_eager_rtm_pkt_attr *attr);

void efa_unit_test_handshake_pkt_construct(struct efa_rdm_pke *pkt_entry, struct efa_unit_test_handshake_pkt_attr *attr);

/* test cases */
void test_av_insert_duplicate_raw_addr();
void test_av_insert_duplicate_gid();
void test_efa_device_construct_error_handling();
void test_efa_rdm_ep_ignore_missing_host_id_file();
void test_efa_rdm_ep_has_valid_host_id();
void test_efa_rdm_ep_ignore_short_host_id();
void test_efa_rdm_ep_ignore_non_hex_host_id();
void test_efa_rdm_ep_handshake_receive_and_send_valid_host_ids_with_connid();
void test_efa_rdm_ep_handshake_receive_and_send_valid_host_ids_without_connid();
void test_efa_rdm_ep_handshake_receive_valid_peer_host_id_and_do_not_send_local_host_id();
void test_efa_rdm_ep_handshake_receive_without_peer_host_id_and_do_not_send_local_host_id();
void test_efa_rdm_ep_getopt_undersized_optlen();
void test_efa_rdm_ep_getopt_oversized_optlen();
void test_efa_rdm_ep_pkt_pool_flags();
void test_efa_rdm_ep_pkt_pool_page_alignment();
void test_efa_rdm_ep_dc_atomic_queue_before_handshake();
void test_efa_rdm_ep_dc_send_queue_before_handshake();
void test_efa_rdm_ep_dc_send_queue_limit_before_handshake();
void test_efa_rdm_ep_write_queue_before_handshake();
void test_efa_rdm_ep_read_queue_before_handshake();
void test_efa_rdm_read_copy_pkt_pool_128_alignment();
void test_efa_rdm_ep_send_with_shm_no_copy();
void test_efa_rdm_ep_rma_without_caps();
void test_efa_rdm_ep_atomic_without_caps();
void test_efa_rdm_ep_setopt_shared_memory_permitted();
void test_efa_rdm_ep_enable_qp_in_order_aligned_128_bytes_good();
void test_efa_rdm_ep_enable_qp_in_order_aligned_128_bytes_bad();
void test_efa_rdm_ep_user_zcpy_rx_disabled();
void test_efa_rdm_ep_user_disable_p2p_zcpy_rx_disabled();
void test_efa_rdm_ep_user_zcpy_rx_unhappy_due_to_sas();
void test_efa_rdm_ep_user_p2p_not_supported_zcpy_rx_happy();
void test_efa_rdm_ep_user_zcpy_rx_unhappy_due_to_no_mr_local();
void test_efa_rdm_ep_close_discard_posted_recv();
void test_efa_rdm_ep_zcpy_recv_cancel();
void test_efa_rdm_ep_zcpy_recv_eagain();
void test_efa_rdm_ep_post_handshake_error_handling_pke_exhaustion();
void test_dgram_cq_read_empty_cq();
void test_ibv_cq_ex_read_empty_cq();
void test_ibv_cq_ex_read_failed_poll();
void test_rdm_cq_create_error_handling();
void test_rdm_cq_read_bad_send_status_unresponsive_receiver();
void test_rdm_cq_read_bad_send_status_unresponsive_receiver_missing_peer_host_id();
void test_rdm_cq_read_bad_send_status_unreachable_receiver();
void test_rdm_cq_read_bad_send_status_invalid_qpn();
void test_rdm_cq_read_bad_send_status_message_too_long();
void test_ibv_cq_ex_read_bad_recv_status();
void test_ibv_cq_ex_read_bad_recv_rdma_with_imm_status_use_unsolicited_recv();
void test_ibv_cq_ex_read_bad_recv_rdma_with_imm_status_use_solicited_recv();
void test_ibv_cq_ex_read_recover_forgotten_peer_ah();
void test_rdm_fallback_to_ibv_create_cq_ex_cq_read_ignore_forgotton_peer();
void test_ibv_cq_ex_read_ignore_removed_peer();
void test_info_open_ep_with_wrong_info();
void test_info_tx_rx_msg_order_rdm_order_none();
void test_info_tx_rx_msg_order_rdm_order_sas();
void test_info_tx_rx_msg_order_dgram_order_none();
void test_info_tx_rx_msg_order_dgram_order_sas();
void test_info_max_order_size_dgram_with_atomic();
void test_info_max_order_size_rdm_with_atomic_no_order();
void test_info_max_order_size_rdm_with_atomic_order();
void test_info_tx_rx_op_flags_rdm();
void test_info_tx_rx_size_rdm();
void test_info_check_shm_info_hmem();
void test_info_check_shm_info_op_flags();
void test_info_check_shm_info_threading();
void test_info_check_hmem_cuda_support_on_api_lt_1_18();
void test_info_check_hmem_cuda_support_on_api_ge_1_18();
void test_info_check_no_hmem_support_when_not_requested();
void test_efa_hmem_info_update_neuron();
void test_efa_hmem_info_disable_p2p_neuron();
void test_efa_hmem_info_disable_p2p_cuda();
void test_efa_nic_select_all_devices_matches();
void test_efa_nic_select_first_device_matches();
void test_efa_nic_select_first_device_with_surrounding_comma_matches();
void test_efa_nic_select_first_device_first_letter_no_match();
void test_efa_nic_select_empty_device_no_match();
void test_efa_use_device_rdma_env1_opt1();
void test_efa_use_device_rdma_env0_opt0();
void test_efa_use_device_rdma_env1_opt0();
void test_efa_use_device_rdma_env0_opt1();
void test_efa_use_device_rdma_opt1();
void test_efa_use_device_rdma_opt0();
void test_efa_use_device_rdma_env1();
void test_efa_use_device_rdma_env0();
void test_efa_use_device_rdma_opt_old();
void test_efa_srx_min_multi_recv_size();
void test_efa_srx_cq();
void test_efa_srx_lock();
void test_efa_rnr_queue_and_resend();
void test_efa_rdm_ope_prepare_to_post_send_with_no_enough_tx_pkts();
void test_efa_rdm_ope_prepare_to_post_send_host_memory();
void test_efa_rdm_ope_prepare_to_post_send_host_memory_align128();
void test_efa_rdm_ope_prepare_to_post_send_cuda_memory();
void test_efa_rdm_ope_prepare_to_post_send_cuda_memory_align128();
void test_efa_rdm_ope_post_write_0_byte();
void test_efa_rdm_rxe_post_local_read_or_queue_cleanup_txe();
void test_efa_rdm_msg_send_to_local_peer_with_null_desc();
void test_efa_fork_support_request_initialize_when_ibv_fork_support_is_needed();
void test_efa_fork_support_request_initialize_when_ibv_fork_support_is_unneeded();
void test_efa_rdm_peer_get_runt_size_no_enough_runt();
void test_efa_rdm_peer_get_runt_size_cuda_memory_smaller_than_alignment();
void test_efa_rdm_peer_get_runt_size_cuda_memory_exceeding_total_len();
void test_efa_rdm_peer_get_runt_size_cuda_memory_normal();
void test_efa_rdm_peer_get_runt_size_host_memory_smaller_than_alignment();
void test_efa_rdm_peer_get_runt_size_host_memory_exceeding_total_len();
void test_efa_rdm_peer_get_runt_size_host_memory_normal();
void test_efa_rdm_peer_get_runt_size_cuda_memory_128_multiple_alignment();
void test_efa_rdm_peer_get_runt_size_cuda_memory_non_128_multiple_alignment();
void test_efa_rdm_peer_get_runt_size_cuda_memory_smaller_than_128_alignment();
void test_efa_rdm_peer_get_runt_size_cuda_memory_exceeding_total_len_128_alignment();
void test_efa_rdm_peer_select_readbase_rtm_no_runt();
void test_efa_rdm_peer_select_readbase_rtm_do_runt();
void test_efa_rdm_pke_get_available_copy_methods_align128();
void test_efa_domain_open_ops_wrong_name();
void test_efa_domain_open_ops_mr_query();
void test_efa_rdm_cq_ibv_cq_poll_list_same_tx_rx_cq_single_ep();
void test_efa_rdm_cq_ibv_cq_poll_list_separate_tx_rx_cq_single_ep();
void test_efa_rdm_cq_post_initial_rx_pkts();
void test_efa_rdm_cntr_ibv_cq_poll_list_same_tx_rx_cq_single_ep();
void test_efa_rdm_cntr_ibv_cq_poll_list_separate_tx_rx_cq_single_ep();
void test_efa_cntr_post_initial_rx_pkts();
void test_efa_rdm_peer_reorder_expected_msg_id();
void test_efa_rdm_peer_reorder_smaller_msg_id();
void test_efa_rdm_peer_reorder_larger_msg_id();
void test_efa_rdm_peer_reorder_overflow_msg_id();
void test_efa_rdm_peer_move_overflow_pke_to_recvwin();
void test_efa_rdm_peer_keep_pke_in_overflow_list();
void test_efa_rdm_peer_append_overflow_pke_to_recvwin();
void test_efa_rdm_pke_handle_longcts_rtm_send_completion();

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
#endif
