#ifndef EFA_UNIT_TESTS_H
#define EFA_UNIT_TESTS_H

#define _GNU_SOURCE
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
void efa_unit_test_resource_construct_with_hints(struct efa_resource *resource,
						 enum fi_ep_type ep_type,
						 struct fi_info* hints);

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
void test_efa_rdm_ep_cq_create_error_handling();
void test_efa_rdm_ep_pkt_pool_flags();
void test_efa_rdm_ep_pkt_pool_page_alignment();
void test_efa_rdm_ep_dc_atomic_error_handling();
void test_efa_rdm_ep_send_with_shm_no_copy();
void test_efa_rdm_ep_rma_without_caps();
void test_efa_rdm_ep_atomic_without_caps();
void test_dgram_cq_read_empty_cq();
void test_ibv_cq_ex_read_empty_cq();
void test_ibv_cq_ex_read_failed_poll();
void test_rdm_cq_read_bad_send_status_unresponsive_receiver();
void test_rdm_cq_read_bad_send_status_unresponsive_receiver_missing_peer_host_id();
void test_rdm_cq_read_bad_send_status_invalid_qpn();
void test_rdm_cq_read_bad_send_status_message_too_long();
void test_ibv_cq_ex_read_bad_recv_status();
void test_ibv_cq_ex_read_recover_forgotten_peer_ah();
void test_rdm_fallback_to_ibv_create_cq_ex_cq_read_ignore_forgotton_peer();
void test_ibv_cq_ex_read_ignore_removed_peer();
void test_info_open_ep_with_wrong_info();
void test_info_open_ep_with_api_1_1_info();
void test_info_check_shm_info();
void test_info_check_hmem_cuda_support_on_api_lt_1_18();
void test_info_check_hmem_cuda_support_on_api_ge_1_18();
void test_info_check_no_hmem_support_when_not_requested();
void test_efa_hmem_info_update_neuron();
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
void test_efa_rdm_msg_send_to_local_peer_with_null_desc();
void test_efa_fork_support_request_initialize_when_ibv_fork_support_is_needed();
void test_efa_fork_support_request_initialize_when_ibv_fork_support_is_unneeded();

#endif
