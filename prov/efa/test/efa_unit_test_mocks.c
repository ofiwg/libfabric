/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#define _GNU_SOURCE
#include <errno.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include "efa.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_pke_nonreq.h"
#include "efa_unit_test_mocks.h"

/* mock of rdma-core functions */

/**
 * @brief call real ibv_create_ah and mock()
 *
 * When combined with will_return_count(), this mock of ibv_create_ah() can be used to verify
 * number of times ibv_create_ah() is called.
 */
struct ibv_ah *efa_mock_ibv_create_ah_check_mock(struct ibv_pd *pd, struct ibv_ah_attr *attr)
{
	mock();
	return  __real_ibv_create_ah(pd, attr);
}

int efa_mock_efadv_query_device_return_mock(struct ibv_context *ibv_ctx,
					    struct efadv_device_attr *attr,
					    uint32_t inlen)
{
	return mock();
}


/**
 * @brief a list of work requests request's WR ID
 */
void *g_ibv_submitted_wr_id_vec[EFA_RDM_EP_MAX_WR_PER_IBV_POST_SEND];
int g_ibv_submitted_wr_id_cnt = 0;

void efa_ibv_submitted_wr_id_vec_clear()
{
	memset(g_ibv_submitted_wr_id_vec, 0,
	       g_ibv_submitted_wr_id_cnt * sizeof(void *));
	g_ibv_submitted_wr_id_cnt = 0;
}

void efa_mock_efa_qp_wr_start_no_op(struct efa_qp *qp)
{
}

/**
 * @brief save wr_id of send request in a global array
 *
 * The saved work request is then be used by efa_mock_efa_ibv_cq_start_poll_use_send_wr()
 * to make ibv_cq_ex to look like it indeed got a completion from device.
 */
void efa_mock_efa_qp_wr_send_save_wr(struct efa_qp *qp)
{
	g_ibv_submitted_wr_id_vec[g_ibv_submitted_wr_id_cnt] = (void *)qp->ibv_qp_ex->wr_id;
	g_ibv_submitted_wr_id_cnt++;
}

void efa_mock_efa_qp_wr_send_verify_handshake_pkt_local_host_id_and_save_wr(struct efa_qp *qp)
{
	struct efa_rdm_pke* pke;
	struct efa_rdm_base_hdr *efa_rdm_base_hdr;
	uint64_t *host_id_ptr;

	pke = (struct efa_rdm_pke *)qp->ibv_qp_ex->wr_id;
	efa_rdm_base_hdr = efa_rdm_pke_get_base_hdr(pke);

	assert_int_equal(efa_rdm_base_hdr->type, EFA_RDM_HANDSHAKE_PKT);

	if (g_efa_unit_test_mocks.local_host_id) {
		assert_true(efa_rdm_base_hdr->flags & EFA_RDM_HANDSHAKE_HOST_ID_HDR);
		host_id_ptr = efa_rdm_pke_get_handshake_opt_host_id_ptr(pke);
		assert_true(*host_id_ptr == g_efa_unit_test_mocks.local_host_id);
	} else {
		assert_false(efa_rdm_base_hdr->flags & EFA_RDM_HANDSHAKE_HOST_ID_HDR);
	}

	function_called();
	return efa_mock_efa_qp_wr_send_save_wr(qp);
}

void efa_mock_efa_qp_wr_send_imm_save_wr(struct efa_qp *qp, __be32 imm_data)
{
	g_ibv_submitted_wr_id_vec[g_ibv_submitted_wr_id_cnt] = (void *)qp->ibv_qp_ex->wr_id;
	g_ibv_submitted_wr_id_cnt++;
}

void efa_mock_efa_qp_wr_set_inline_data_list_no_op(struct efa_qp *qp,
						size_t num_buf,
						const struct ibv_data_buf *buf_list)
{
}

void efa_mock_efa_qp_wr_set_sge_list_no_op(struct efa_qp *qp,
					size_t num_sge,
					const struct ibv_sge *sge_list)
{
}

void efa_mock_efa_qp_wr_set_ud_addr_no_op(struct efa_qp *qp, struct efa_ah *ah,
				       uint32_t remote_qpn, uint32_t remote_qkey)
{
}

int efa_mock_efa_qp_wr_complete_no_op(struct efa_qp *qp)
{
	return 0;
}

void efa_mock_efa_qp_wr_rdma_write_save_wr(struct efa_qp *qp, uint32_t rkey,
					uint64_t remote_addr)
{
	g_ibv_submitted_wr_id_vec[g_ibv_submitted_wr_id_cnt] = (void *)qp->ibv_qp_ex->wr_id;
	g_ibv_submitted_wr_id_cnt++;
}

int efa_mock_efa_ibv_cq_start_poll_return_mock(struct efa_ibv_cq *ibv_cq,
					struct ibv_poll_cq_attr *attr)
{
	return mock();
}

static inline
int efa_mock_use_saved_send_wr(struct ibv_cq_ex *ibv_cqx, int status)
{
	int i;

	if (g_ibv_submitted_wr_id_cnt == 0)
		return ENOENT;

	ibv_cqx->wr_id = (uintptr_t)g_ibv_submitted_wr_id_vec[0];
	ibv_cqx->status = status;

	for (i = 1; i < g_ibv_submitted_wr_id_cnt; ++i)
		g_ibv_submitted_wr_id_vec[i-1] = g_ibv_submitted_wr_id_vec[i];

	g_ibv_submitted_wr_id_cnt--;
	return 0;
}

int efa_mock_efa_ibv_cq_start_poll_use_saved_send_wr_with_mock_status(struct efa_ibv_cq *ibv_cq,
							       struct ibv_poll_cq_attr *attr)
{
	return efa_mock_use_saved_send_wr(ibv_cq->ibv_cq_ex, mock());
}

int efa_mock_efa_ibv_cq_next_poll_return_mock(struct efa_ibv_cq *ibv_cq)
{
	return mock();
}

int efa_mock_efa_ibv_cq_next_poll_use_saved_send_wr_with_mock_status(struct efa_ibv_cq *ibv_cq)
{
	return efa_mock_use_saved_send_wr(ibv_cq->ibv_cq_ex, mock());
}

void efa_mock_efa_ibv_cq_end_poll_check_mock(struct efa_ibv_cq *ibv_cq)
{
	mock();
}

enum ibv_wc_opcode efa_mock_efa_ibv_cq_wc_read_opcode_return_mock(struct efa_ibv_cq *current)
{
	return mock();
}

uint32_t efa_mock_efa_ibv_cq_wc_read_vendor_err_return_mock(struct efa_ibv_cq *current)
{
	return mock();
}

uint32_t efa_mock_efa_ibv_cq_wc_read_qp_num_return_mock(struct efa_ibv_cq *current)
{
	return mock();
}

uint32_t efa_mock_efa_ibv_cq_wc_read_wc_flags_return_mock(struct efa_ibv_cq *current)
{
	return mock();
}

uint32_t efa_mock_efa_ibv_cq_wc_read_imm_data_return_mock(struct efa_ibv_cq *current)
{
	return mock();
}

bool efa_mock_efa_ibv_cq_wc_is_unsolicited_return_mock(struct efa_ibv_cq *ibv_cq)
{
	return mock();
}

int g_ofi_copy_from_hmem_iov_call_counter;
ssize_t efa_mock_ofi_copy_from_hmem_iov_inc_counter(void *dest, size_t size,
						    enum fi_hmem_iface hmem_iface, uint64_t device,
						    const struct iovec *hmem_iov,
						    size_t hmem_iov_count, uint64_t hmem_iov_offset)
{
	g_ofi_copy_from_hmem_iov_call_counter += 1;
	return __real_ofi_copy_from_hmem_iov(dest, size, hmem_iface, device, hmem_iov, hmem_iov_count, hmem_iov_offset);
}

int efa_mock_efa_rdm_pke_read_return_mock(struct efa_rdm_ope *ope)
{
	return mock();
}

ssize_t efa_mock_efa_rdm_ope_post_send_return_mock(struct efa_rdm_ope *ope, int pkt_type)
{
	return mock();
}

ssize_t efa_mock_efa_rdm_pke_proc_matched_rtm_no_op(struct efa_rdm_pke *pkt_entry)
{
	return FI_SUCCESS;
}

bool efa_mock_efa_device_support_unsolicited_write_recv()
{
	return mock();
}

int efa_mock_efa_qp_post_recv_return_mock(struct efa_qp *qp, struct ibv_recv_wr *wr,
				struct ibv_recv_wr **bad_wr)
{
	return mock();
}

void efa_mock_efa_qp_wr_rdma_read_save_wr(struct efa_qp *qp, uint32_t rkey,
				       uint64_t remote_addr)
{
	g_ibv_submitted_wr_id_vec[g_ibv_submitted_wr_id_cnt] = (void *)qp->ibv_qp_ex->wr_id;
	g_ibv_submitted_wr_id_cnt++;
}

void efa_mock_efa_qp_wr_rdma_write_imm_save_wr(struct efa_qp *qp, uint32_t rkey,
					    uint64_t remote_addr,
					    __be32 imm_data)
{
	g_ibv_submitted_wr_id_vec[g_ibv_submitted_wr_id_cnt] = (void *) qp->ibv_qp_ex->wr_id;
	g_ibv_submitted_wr_id_cnt++;
}

struct efa_unit_test_mocks g_efa_unit_test_mocks = {
	.local_host_id = 0,
	.peer_host_id = 0,
	.ibv_create_ah = __real_ibv_create_ah,
	.efadv_query_device = __real_efadv_query_device,
#if HAVE_EFADV_CQ_EX
	.efadv_create_cq = __real_efadv_create_cq,
#endif
#if HAVE_NEURON
	.neuron_alloc = __real_neuron_alloc,
#endif
#if HAVE_CUDA
	.ofi_cudaMalloc = __real_ofi_cudaMalloc,
#endif
	.ofi_copy_from_hmem_iov = __real_ofi_copy_from_hmem_iov,
	.efa_rdm_pke_read = __real_efa_rdm_pke_read,
	.efa_rdm_pke_proc_matched_rtm = __real_efa_rdm_pke_proc_matched_rtm,
	.efa_rdm_ope_post_send = __real_efa_rdm_ope_post_send,
	.efa_device_support_unsolicited_write_recv = __real_efa_device_support_unsolicited_write_recv,
	.ibv_is_fork_initialized = __real_ibv_is_fork_initialized,

	/* EFA data path ops real function assignments */
	.efa_qp_post_recv = __real_efa_qp_post_recv,
	.efa_qp_wr_complete = __real_efa_qp_wr_complete,
	.efa_qp_wr_rdma_read = __real_efa_qp_wr_rdma_read,
	.efa_qp_wr_rdma_write = __real_efa_qp_wr_rdma_write,
	.efa_qp_wr_rdma_write_imm = __real_efa_qp_wr_rdma_write_imm,
	.efa_qp_wr_send = __real_efa_qp_wr_send,
	.efa_qp_wr_send_imm = __real_efa_qp_wr_send_imm,
	.efa_qp_wr_set_inline_data_list = __real_efa_qp_wr_set_inline_data_list,
	.efa_qp_wr_set_sge_list = __real_efa_qp_wr_set_sge_list,
	.efa_qp_wr_set_ud_addr = __real_efa_qp_wr_set_ud_addr,
	.efa_qp_wr_start = __real_efa_qp_wr_start,
	.efa_ibv_cq_start_poll = __real_efa_ibv_cq_start_poll,
	.efa_ibv_cq_next_poll = __real_efa_ibv_cq_next_poll,
	.efa_ibv_cq_wc_read_opcode = __real_efa_ibv_cq_wc_read_opcode,
	.efa_ibv_cq_end_poll = __real_efa_ibv_cq_end_poll,
	.efa_ibv_cq_wc_read_qp_num = __real_efa_ibv_cq_wc_read_qp_num,
	.efa_ibv_cq_wc_read_vendor_err = __real_efa_ibv_cq_wc_read_vendor_err,
	.efa_ibv_cq_wc_read_src_qp = __real_efa_ibv_cq_wc_read_src_qp,
	.efa_ibv_cq_wc_read_slid = __real_efa_ibv_cq_wc_read_slid,
	.efa_ibv_cq_wc_read_byte_len = __real_efa_ibv_cq_wc_read_byte_len,
	.efa_ibv_cq_wc_read_wc_flags = __real_efa_ibv_cq_wc_read_wc_flags,
	.efa_ibv_cq_wc_read_imm_data = __real_efa_ibv_cq_wc_read_imm_data,
	.efa_ibv_cq_wc_is_unsolicited = __real_efa_ibv_cq_wc_is_unsolicited,
	.efa_ibv_cq_wc_read_sgid = __real_efa_ibv_cq_wc_read_sgid,
	.efa_ibv_get_cq_event = __real_efa_ibv_get_cq_event,
	.efa_ibv_req_notify_cq = __real_efa_ibv_req_notify_cq,

#if HAVE_EFADV_QUERY_MR
	.efadv_query_mr = __real_efadv_query_mr,
#endif
#if HAVE_EFA_DATA_IN_ORDER_ALIGNED_128_BYTES
	.ibv_query_qp_data_in_order = __real_ibv_query_qp_data_in_order,
#endif
#if HAVE_EFADV_QUERY_QP_WQS
	.efadv_query_qp_wqs = __real_efadv_query_qp_wqs,
#endif
#if HAVE_EFADV_QUERY_CQ
	.efadv_query_cq = __real_efadv_query_cq,
#endif
};

/* EFA data path ops wrappers */
int __wrap_efa_qp_post_recv(struct efa_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad)
{
	return g_efa_unit_test_mocks.efa_qp_post_recv(qp, wr, bad);
}

int __wrap_efa_qp_wr_complete(struct efa_qp *efaqp)
{
	return g_efa_unit_test_mocks.efa_qp_wr_complete(efaqp);
}

void __wrap_efa_qp_wr_rdma_read(struct efa_qp *efaqp, uint32_t rkey, uint64_t remote_addr)
{
	g_efa_unit_test_mocks.efa_qp_wr_rdma_read(efaqp, rkey, remote_addr);
}

void __wrap_efa_qp_wr_rdma_write(struct efa_qp *efaqp, uint32_t rkey, uint64_t remote_addr)
{
	g_efa_unit_test_mocks.efa_qp_wr_rdma_write(efaqp, rkey, remote_addr);
}

void __wrap_efa_qp_wr_rdma_write_imm(struct efa_qp *efaqp, uint32_t rkey, uint64_t remote_addr, __be32 imm_data)
{
	g_efa_unit_test_mocks.efa_qp_wr_rdma_write_imm(efaqp, rkey, remote_addr, imm_data);
}

void __wrap_efa_qp_wr_send(struct efa_qp *efaqp)
{
	g_efa_unit_test_mocks.efa_qp_wr_send(efaqp);
}

void __wrap_efa_qp_wr_send_imm(struct efa_qp *efaqp, __be32 imm_data)
{
	g_efa_unit_test_mocks.efa_qp_wr_send_imm(efaqp, imm_data);
}

void __wrap_efa_qp_wr_set_inline_data_list(struct efa_qp *efaqp, size_t num_buf, const struct ibv_data_buf *buf_list)
{
	g_efa_unit_test_mocks.efa_qp_wr_set_inline_data_list(efaqp, num_buf, buf_list);
}

void __wrap_efa_qp_wr_set_sge_list(struct efa_qp *efaqp, size_t num_sge, const struct ibv_sge *sg_list)
{
	g_efa_unit_test_mocks.efa_qp_wr_set_sge_list(efaqp, num_sge, sg_list);
}

void __wrap_efa_qp_wr_set_ud_addr(struct efa_qp *efaqp, struct efa_ah *ah, uint32_t remote_qpn, uint32_t remote_qkey)
{
	g_efa_unit_test_mocks.efa_qp_wr_set_ud_addr(efaqp, ah, remote_qpn, remote_qkey);
}

void __wrap_efa_qp_wr_start(struct efa_qp *efaqp)
{
	g_efa_unit_test_mocks.efa_qp_wr_start(efaqp);
}

int __wrap_efa_ibv_cq_start_poll(struct efa_ibv_cq *ibv_cq, struct ibv_poll_cq_attr *attr)
{
	return g_efa_unit_test_mocks.efa_ibv_cq_start_poll(ibv_cq, attr);
}

int __wrap_efa_ibv_cq_next_poll(struct efa_ibv_cq *ibv_cq)
{
	return g_efa_unit_test_mocks.efa_ibv_cq_next_poll(ibv_cq);
}

enum ibv_wc_opcode __wrap_efa_ibv_cq_wc_read_opcode(struct efa_ibv_cq *ibv_cq)
{
	return g_efa_unit_test_mocks.efa_ibv_cq_wc_read_opcode(ibv_cq);
}

void __wrap_efa_ibv_cq_end_poll(struct efa_ibv_cq *ibv_cq)
{
	g_efa_unit_test_mocks.efa_ibv_cq_end_poll(ibv_cq);
}

uint32_t __wrap_efa_ibv_cq_wc_read_qp_num(struct efa_ibv_cq *ibv_cq)
{
	return g_efa_unit_test_mocks.efa_ibv_cq_wc_read_qp_num(ibv_cq);
}

uint32_t __wrap_efa_ibv_cq_wc_read_vendor_err(struct efa_ibv_cq *ibv_cq)
{
	return g_efa_unit_test_mocks.efa_ibv_cq_wc_read_vendor_err(ibv_cq);
}

uint32_t __wrap_efa_ibv_cq_wc_read_src_qp(struct efa_ibv_cq *ibv_cq)
{
	return g_efa_unit_test_mocks.efa_ibv_cq_wc_read_src_qp(ibv_cq);
}

uint32_t __wrap_efa_ibv_cq_wc_read_slid(struct efa_ibv_cq *ibv_cq)
{
	return g_efa_unit_test_mocks.efa_ibv_cq_wc_read_slid(ibv_cq);
}

uint32_t __wrap_efa_ibv_cq_wc_read_byte_len(struct efa_ibv_cq *ibv_cq)
{
	return g_efa_unit_test_mocks.efa_ibv_cq_wc_read_byte_len(ibv_cq);
}

unsigned int __wrap_efa_ibv_cq_wc_read_wc_flags(struct efa_ibv_cq *ibv_cq)
{
	return g_efa_unit_test_mocks.efa_ibv_cq_wc_read_wc_flags(ibv_cq);
}

__be32 __wrap_efa_ibv_cq_wc_read_imm_data(struct efa_ibv_cq *ibv_cq)
{
	return g_efa_unit_test_mocks.efa_ibv_cq_wc_read_imm_data(ibv_cq);
}

bool __wrap_efa_ibv_cq_wc_is_unsolicited(struct efa_ibv_cq *ibv_cq)
{
	return g_efa_unit_test_mocks.efa_ibv_cq_wc_is_unsolicited(ibv_cq);
}

int __wrap_efa_ibv_cq_wc_read_sgid(struct efa_ibv_cq *ibv_cq, union ibv_gid *sgid)
{
	return g_efa_unit_test_mocks.efa_ibv_cq_wc_read_sgid(ibv_cq, sgid);
}

int __wrap_efa_ibv_get_cq_event(struct efa_ibv_cq *ibv_cq, void **cq_context)
{
	return g_efa_unit_test_mocks.efa_ibv_get_cq_event(ibv_cq, cq_context);
}

int __wrap_efa_ibv_req_notify_cq(struct efa_ibv_cq *ibv_cq, int solicited_only)
{
	return g_efa_unit_test_mocks.efa_ibv_req_notify_cq(ibv_cq, solicited_only);
}

struct ibv_ah *__wrap_ibv_create_ah(struct ibv_pd *pd, struct ibv_ah_attr *attr)
{
	return g_efa_unit_test_mocks.ibv_create_ah(pd, attr);
}

int __wrap_efadv_query_device(struct ibv_context *ibv_ctx, struct efadv_device_attr *attr,
			      uint32_t inlen)
{
	return g_efa_unit_test_mocks.efadv_query_device(ibv_ctx, attr, inlen);
}

struct ibv_cq_ex *efa_mock_create_cq_ex_return_null(struct ibv_context *context, struct ibv_cq_init_attr_ex *init_attr)
{
	function_called();
	return NULL;
};

#if HAVE_EFADV_CQ_EX
struct ibv_cq_ex *__wrap_efadv_create_cq(struct ibv_context *ibvctx,
										 struct ibv_cq_init_attr_ex *attr_ex,
										 struct efadv_cq_init_attr *efa_attr,
										 uint32_t inlen)
{
	return g_efa_unit_test_mocks.efadv_create_cq(ibvctx, attr_ex, efa_attr, inlen);
}

uint32_t efa_mock_efa_ibv_cq_wc_read_src_qp_return_mock(struct efa_ibv_cq *current)
{
	return mock();
}

uint32_t efa_mock_efa_ibv_cq_wc_read_byte_len_return_mock(struct efa_ibv_cq *current)
{
	return mock();
};

uint32_t efa_mock_efa_ibv_cq_wc_read_slid_return_mock(struct efa_ibv_cq *current)
{
	return mock();
}

int efa_mock_efa_ibv_cq_wc_read_sgid_return_mock(struct efa_ibv_cq *ibv_cq, union ibv_gid *sgid)
{
	return mock();
}

int efa_mock_efa_ibv_cq_wc_read_sgid_return_zero_code_and_expect_next_poll_and_set_gid(struct efa_ibv_cq *ibv_cq, union ibv_gid *sgid)
{
	/* Make sure this mock is always called before ibv_next_poll */
	expect_function_call(efa_mock_efa_ibv_cq_next_poll_check_function_called_and_return_mock);
	memcpy(sgid->raw, (uint8_t *)mock(), sizeof(sgid->raw));
	/* Must return 0 for unknown AH */
	return 0;
};

int efa_mock_efa_ibv_cq_next_poll_check_function_called_and_return_mock(struct efa_ibv_cq *ibv_cq)
{
	function_called();
	return mock();
};

struct ibv_cq_ex *efa_mock_efadv_create_cq_with_ibv_create_cq_ex(struct ibv_context *ibvctx,
																 struct ibv_cq_init_attr_ex *attr_ex,
																 struct efadv_cq_init_attr *efa_attr,
																 uint32_t inlen)
{
	function_called();
	return ibv_create_cq_ex(ibvctx, attr_ex);
}

struct ibv_cq_ex *efa_mock_efadv_create_cq_set_eopnotsupp_and_return_null(struct ibv_context *ibvctx,
																		  struct ibv_cq_init_attr_ex *attr_ex,
																		  struct efadv_cq_init_attr *efa_attr,
																		  uint32_t inlen)
{
	function_called();
	errno = EOPNOTSUPP;
	return NULL;
}
#endif

#if HAVE_NEURON
void *__wrap_neuron_alloc(void **handle, size_t size)
{
	return g_efa_unit_test_mocks.neuron_alloc(handle, size);
}

void *efa_mock_neuron_alloc_return_null(void **handle, size_t size)
{
	return NULL;
}

void *efa_mock_neuron_alloc_return_mock(void **handle, size_t size)
{
	/* Not mocking return value so this function will fail when it is called */
	return (void *) mock();
}
#endif

#if HAVE_CUDA
cudaError_t __wrap_ofi_cudaMalloc(void **ptr, size_t size)
{
	return g_efa_unit_test_mocks.ofi_cudaMalloc(ptr, size);
}

cudaError_t efa_mock_ofi_cudaMalloc_return_mock(void **ptr, size_t size)
{
	/* Not mocking return value so this function will fail when it is called */
	return (cudaError_t) mock();
}
#endif

ssize_t __wrap_ofi_copy_from_hmem_iov(void *dest, size_t size,
				      enum fi_hmem_iface hmem_iface, uint64_t device,
				      const struct iovec *hmem_iov,
				      size_t hmem_iov_count, uint64_t hmem_iov_offset)
{
	return g_efa_unit_test_mocks.ofi_copy_from_hmem_iov(dest, size, hmem_iface, device, hmem_iov, hmem_iov_count, hmem_iov_offset);
}

int __wrap_efa_rdm_pke_read(struct efa_rdm_ope *ope)
{
	return g_efa_unit_test_mocks.efa_rdm_pke_read(ope);
}

int __wrap_efa_rdm_pke_proc_matched_rtm(struct efa_rdm_pke *pkt_entry)
{
	return g_efa_unit_test_mocks.efa_rdm_pke_proc_matched_rtm(pkt_entry);
}

int __wrap_efa_rdm_ope_post_send(struct efa_rdm_ope *ope, int pkt_type)
{
	return g_efa_unit_test_mocks.efa_rdm_ope_post_send(ope, pkt_type);
}

bool __wrap_efa_device_support_unsolicited_write_recv(void)
{
	return g_efa_unit_test_mocks.efa_device_support_unsolicited_write_recv();
}

enum ibv_fork_status __wrap_ibv_is_fork_initialized(void)
{
	return g_efa_unit_test_mocks.ibv_is_fork_initialized();
}

enum ibv_fork_status efa_mock_ibv_is_fork_initialized_return_mock(void)
{
	return mock();
}

#if HAVE_EFADV_QUERY_MR
int __wrap_efadv_query_mr(struct ibv_mr *ibv_mr, struct efadv_mr_attr *attr, uint32_t inlen)
{
	return g_efa_unit_test_mocks.efadv_query_mr(ibv_mr, attr, inlen);
}

/* set recv_ic_id as 0 */
int efa_mock_efadv_query_mr_recv_ic_id_0(struct ibv_mr *ibv_mr, struct efadv_mr_attr *attr, uint32_t inlen)
{
	attr->ic_id_validity = EFADV_MR_ATTR_VALIDITY_RECV_IC_ID;
	attr->recv_ic_id = 0;
	return 0;
}

/* set rdma_read_ic_id id as 1 */
int efa_mock_efadv_query_mr_rdma_read_ic_id_1(struct ibv_mr *ibv_mr, struct efadv_mr_attr *attr, uint32_t inlen)
{
	attr->ic_id_validity = EFADV_MR_ATTR_VALIDITY_RDMA_READ_IC_ID;
	attr->rdma_read_ic_id = 1;
	return 0;
}

/* set rdma_recv_ic_id id as 2 */
int efa_mock_efadv_query_mr_rdma_recv_ic_id_2(struct ibv_mr *ibv_mr, struct efadv_mr_attr *attr, uint32_t inlen)
{
	attr->ic_id_validity = EFADV_MR_ATTR_VALIDITY_RDMA_RECV_IC_ID;
	attr->rdma_recv_ic_id = 2;
	return 0;
}

/* set recv_ic_id id as 0, rdma_read_ic_id as 1 */
int efa_mock_efadv_query_mr_recv_and_rdma_read_ic_id_0_1(struct ibv_mr *ibv_mr, struct efadv_mr_attr *attr, uint32_t inlen)
{
	attr->ic_id_validity = EFADV_MR_ATTR_VALIDITY_RECV_IC_ID;
	attr->recv_ic_id = 0;
	attr->ic_id_validity |= EFADV_MR_ATTR_VALIDITY_RDMA_READ_IC_ID;
	attr->rdma_read_ic_id = 1;
	return 0;
}

#endif /* HAVE_EFADV_QUERY_MR */

#if HAVE_EFA_DATA_IN_ORDER_ALIGNED_128_BYTES
int __wrap_ibv_query_qp_data_in_order(struct ibv_qp *qp, enum ibv_wr_opcode op, uint32_t flags)
{
	return g_efa_unit_test_mocks.ibv_query_qp_data_in_order(qp, op, flags);
}

int efa_mock_ibv_query_qp_data_in_order_return_0(struct ibv_qp *qp, enum ibv_wr_opcode op, uint32_t flags)
{
	return 0;
}

int efa_mock_ibv_query_qp_data_in_order_return_in_order_aligned_128_bytes(struct ibv_qp *qp, enum ibv_wr_opcode op, uint32_t flags)
{
	return IBV_QUERY_QP_DATA_IN_ORDER_ALIGNED_128_BYTES;
}
#endif

#if HAVE_EFADV_QUERY_QP_WQS
int __wrap_efadv_query_qp_wqs(struct ibv_qp *ibvqp, struct efadv_wq_attr *sq_attr,
			      struct efadv_wq_attr *rq_attr, uint32_t inlen)
{
	return g_efa_unit_test_mocks.efadv_query_qp_wqs(ibvqp, sq_attr, rq_attr, inlen);
}

int efa_mock_efadv_query_qp_wqs(struct ibv_qp *ibvqp, struct efadv_wq_attr *sq_attr,
				struct efadv_wq_attr *rq_attr, uint32_t inlen)
{
	sq_attr->buffer = (uint8_t *) 0x12345678;
	sq_attr->doorbell = (uint32_t *) 0x87654321;
	sq_attr->entry_size = 64;
	sq_attr->num_entries = 128;
	sq_attr->max_batch = 16;

	rq_attr->buffer = (uint8_t *) 0x12345678;
	rq_attr->doorbell = (uint32_t *) 0x87654321;
	rq_attr->entry_size = 64;
	rq_attr->num_entries = 128;
	rq_attr->max_batch = 16;
	return 0;
}
#endif /* HAVE_EFADV_QUERY_QP_WQS */

#if HAVE_EFADV_QUERY_CQ
int __wrap_efadv_query_cq(struct ibv_cq *ibvcq, struct efadv_cq_attr *attr, uint32_t inlen)
{
	return g_efa_unit_test_mocks.efadv_query_cq(ibvcq, attr, inlen);
}

int efa_mock_efadv_query_cq(struct ibv_cq *ibvcq, struct efadv_cq_attr *attr, uint32_t inlen)
{
	attr->buffer = (uint8_t *) 0x12345678;
	attr->entry_size = 64;
	attr->num_entries = 128;
	return 0;
}
#endif /* HAVE_EFADV_QUERY_CQ */
 
int efa_mock_ibv_req_notify_cq_return_mock(struct efa_ibv_cq *ibv_cq, int solicited_only)
{
	return 0;
}

int efa_mock_ibv_get_cq_event_return_mock(struct efa_ibv_cq *ibv_cq, void **cq_context)
{
	return mock();
}
