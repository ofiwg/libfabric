#ifndef EFA_UNIT_TESTS_H
#define EFA_UNIT_TESTS_H

#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include "stdio.h"
#include "efa.h"
#include "rxr.h"
#include "efa_unit_test_mocks.h"

extern struct efa_mock_ibv_send_wr_list g_ibv_send_wr_list;
extern struct efa_unit_test_mocks g_efa_unit_test_mocks;

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

int efa_unit_test_resource_construct(struct efa_resource *resource, enum fi_ep_type ep_type);

void efa_unit_test_resource_destruct(struct efa_resource *resource);

struct efa_unit_test_buff {
	uint8_t *buff;
	size_t  size;
	struct fid_mr *mr;
};

struct efa_unit_test_eager_rtm_pkt_attr {
	uint32_t msg_id;
	uint32_t connid;
};

void efa_unit_test_buff_construct(struct efa_unit_test_buff *buff, struct efa_resource *resource, size_t buff_size);

void efa_unit_test_buff_destruct(struct efa_unit_test_buff *buff);

void efa_unit_test_eager_msgrtm_pkt_construct(struct rxr_pkt_entry *pkt_entry, struct efa_unit_test_eager_rtm_pkt_attr *attr);

/* test cases */
void test_av_insert_duplicate_raw_addr();
void test_av_insert_duplicate_gid();
void test_efa_device_construct_error_handling();
void test_rxr_ep_pkt_pool_flags();
void test_rxr_ep_pkt_pool_page_alignment();
void test_rxr_ep_dc_atomic_error_handling();
void test_dgram_cq_read_empty_cq();
void test_dgram_cq_read_bad_wc_status();
void test_ibv_cq_ex_read_empty_cq();
void test_ibv_cq_ex_read_failed_poll();
void test_ibv_cq_ex_read_bad_send_status();
void test_ibv_cq_ex_read_bad_recv_status();
void test_ibv_cq_ex_read_recover_forgotten_peer_ah();
void test_rdm_fallback_to_ibv_create_cq_ex_cq_read_ignore_forgotton_peer();
void test_ibv_cq_ex_read_ignore_removed_peer();
void test_info_open_ep_with_wrong_info();
void test_info_open_ep_with_api_1_1_info();
void test_efa_hmem_support_status_update_neuron();

#endif
