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

struct efa_resource {
	struct fi_info *hints;
	struct fi_info *info;
	struct fid_fabric *fabric;
	struct fid_domain *domain;
	struct fid_ep *ep;
	struct fi_eq_attr *eq_attr;
	struct fid_eq *eq;
	struct fid_av *av;
	struct fid_cq *cq;
};

void test_av_insert_duplicate_raw_addr();

void test_av_insert_duplicate_gid();

void test_efa_device_construct_error_handling();

void test_rxr_ep_pkt_pool_flags();

void test_rxr_ep_pkt_pool_page_alignment();

void test_rxr_ep_dc_atomic_error_handling();

int efa_unit_test_resource_construct(struct efa_resource *resource, enum fi_ep_type ep_type);

void efa_unit_test_resource_destruct(struct efa_resource *resource);

void efa_unit_test_ibv_cq_ex_use_mock(struct ibv_cq_ex *ibv_cq_ex);

void test_efa_cq_readerr_happy();
void test_efa_cq_readerr_sad();

void test_dgram_ep_progress_happy();
void test_dgram_ep_progress_with_empty_cq();
void test_dgram_ep_progress_encounter_bad_wc_status();

void test_rdm_ep_progress_send_completion_happy();
void test_rdm_ep_progress_recv_completion_happy();
void test_rdm_ep_progress_send_empty_cq();
void test_rdm_ep_progress_failed_poll();
void test_rdm_ep_progress_bad_send_wc_status();
void test_rdm_ep_progress_bad_recv_wc_status();

#endif
