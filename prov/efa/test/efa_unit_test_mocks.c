#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include "efa.h"
#include "efa_unit_test_mocks.h"

/* mock of rdma-core functions */
int g_ibv_create_ah_call_counter = 0;
struct ibv_ah *efa_mock_ibv_create_ah_increase_call_counter(struct ibv_pd *pd, struct ibv_ah_attr *attr)
{
	g_ibv_create_ah_call_counter += 1;
	return  __real_ibv_create_ah(pd, attr);
}

int efa_mock_efadv_query_device_return_mock(struct ibv_context *ibv_ctx,
					    struct efadv_device_attr *attr,
					    uint32_t inlen)
{
	return mock();
}

static int efa_mock_ibv_start_poll_return_mock(struct ibv_cq_ex *ibvcqx,
					       struct ibv_poll_cq_attr *attr)
{
	return mock();
}

static int efa_mock_ibv_next_poll_return_mock(struct ibv_cq_ex *ibvcqx)
{
	return mock();
}

static void efa_mock_ibv_end_poll_check_mock(struct ibv_cq_ex *ibvcqx)
{
	mock();
}

static uint32_t efa_mock_ibv_read_vendor_err_return_one(struct ibv_cq_ex *current)
{
	return 1;
}

static unsigned int efa_mock_ibv_read_wc_flags_return_zero(struct ibv_cq_ex *current)
{
	return 0;
}

static uint32_t efa_mock_ibv_read_qp_num_return_zero(struct ibv_cq_ex *current)
{
	return 0;
}

static uint32_t efa_mock_ibv_read_byte_len_return_one(struct ibv_cq_ex *current)
{
	return 1;
}

static uint32_t efa_mock_ibv_read_src_qp_return_zero(struct ibv_cq_ex *current)
{
	return 0;
}

static uint8_t efa_mock_ibv_read_sl_return_zero(struct ibv_cq_ex *current)
{
	return 0;
}

static uint32_t efa_mock_ibv_read_slid_return_zero(struct ibv_cq_ex *current)
{
	return 0;
}

static __be32 efa_mock_ibv_read_imm_data_return_zero(struct ibv_cq_ex *current)
{
	return 0;
}

static enum ibv_wc_opcode efa_mock_ibv_read_opcode_return_mock(struct ibv_cq_ex *current)
{
	return mock();
}

void efa_unit_test_ibv_cq_ex_use_mock(struct ibv_cq_ex *ibv_cq_ex)
{
	ibv_cq_ex->start_poll = &efa_mock_ibv_start_poll_return_mock;
	ibv_cq_ex->next_poll = &efa_mock_ibv_next_poll_return_mock;
	ibv_cq_ex->end_poll = &efa_mock_ibv_end_poll_check_mock;
	ibv_cq_ex->read_vendor_err = &efa_mock_ibv_read_vendor_err_return_one;
	ibv_cq_ex->read_wc_flags = &efa_mock_ibv_read_wc_flags_return_zero;
	ibv_cq_ex->read_qp_num = &efa_mock_ibv_read_qp_num_return_zero;
	ibv_cq_ex->read_byte_len = &efa_mock_ibv_read_byte_len_return_one;
	ibv_cq_ex->read_src_qp = &efa_mock_ibv_read_src_qp_return_zero;
	ibv_cq_ex->read_sl = &efa_mock_ibv_read_sl_return_zero;
	ibv_cq_ex->read_slid = &efa_mock_ibv_read_slid_return_zero;
	ibv_cq_ex->read_imm_data = &efa_mock_ibv_read_imm_data_return_zero;
	ibv_cq_ex->read_opcode = &efa_mock_ibv_read_opcode_return_mock;
}

/* mock of libfabric EFA provider functions */
void efa_mock_rxr_pkt_handle_send_completion_check_args_only(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry)
{
	check_expected(ep);
	check_expected(pkt_entry);
}

void efa_mock_rxr_pkt_handle_recv_completion_check_args_only(struct rxr_ep *ep,
							     struct rxr_pkt_entry *pkt_entry,
							     enum rxr_lower_ep_type lower_ep_type)
{
	check_expected(ep);
	check_expected(pkt_entry);
	check_expected(lower_ep_type);
};

void efa_mock_rxr_pkt_handle_send_error_check_args_only(struct rxr_ep *ep,
							struct rxr_pkt_entry *pkt_entry,
							int err, int prov_errno)
{
	check_expected(ep);
	check_expected(pkt_entry);
	check_expected(err);
	check_expected(prov_errno);
}

void efa_mock_rxr_pkt_handle_recv_error_check_args_only(struct rxr_ep *ep,
							struct rxr_pkt_entry *pkt_entry,
							int err, int prov_errno)
{
	check_expected(ep);
	check_expected(pkt_entry);
	check_expected(err);
	check_expected(prov_errno);
}

struct efa_unit_test_mocks g_efa_unit_test_mocks = {
	.ibv_create_ah = __real_ibv_create_ah,
	.efadv_query_device = __real_efadv_query_device,
	.rxr_pkt_handle_send_completion = __real_rxr_pkt_handle_send_completion,
	.rxr_pkt_handle_recv_completion = __real_rxr_pkt_handle_recv_completion,
	.rxr_pkt_handle_send_error = __real_rxr_pkt_handle_send_error,
	.rxr_pkt_handle_recv_error = __real_rxr_pkt_handle_recv_error,
};

struct ibv_ah *__wrap_ibv_create_ah(struct ibv_pd *pd, struct ibv_ah_attr *attr)
{
	return g_efa_unit_test_mocks.ibv_create_ah(pd, attr);
}

int __wrap_efadv_query_device(struct ibv_context *ibv_ctx, struct efadv_device_attr *attr,
			      uint32_t inlen)
{
	return g_efa_unit_test_mocks.efadv_query_device(ibv_ctx, attr, inlen);
}

void __wrap_rxr_pkt_handle_send_completion(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry)
{
	return g_efa_unit_test_mocks.rxr_pkt_handle_send_completion(ep, pkt_entry);
}

void __wrap_rxr_pkt_handle_recv_completion(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry,
					   enum rxr_lower_ep_type lower_ep_type)
{
	return g_efa_unit_test_mocks.rxr_pkt_handle_recv_completion(ep, pkt_entry, lower_ep_type);
}

void __wrap_rxr_pkt_handle_send_error(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry,
				      int err, int prov_errno)
{
	return g_efa_unit_test_mocks.rxr_pkt_handle_send_error(ep, pkt_entry, err, prov_errno);
}

void __wrap_rxr_pkt_handle_recv_error(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry,
				      int err, int prov_errno)
{
	return g_efa_unit_test_mocks.rxr_pkt_handle_recv_error(ep, pkt_entry, err, prov_errno);
}

