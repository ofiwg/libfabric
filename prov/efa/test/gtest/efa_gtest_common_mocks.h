/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_GTEST_COMMON_MOCKS_H
#define EFA_GTEST_COMMON_MOCKS_H

#include <gmock/gmock.h>
#include <infiniband/efadv.h>
#include <infiniband/verbs.h>

struct efa_av;
struct efa_cur_reverse_av;
struct efa_prv_reverse_av;
struct efa_conn;
struct ofi_mr_map;
struct fi_mr_attr;
struct efa_ibv_cq;
struct efa_rdm_pke;
struct ofi_bufpool;
struct efa_qp;
struct efa_ah;

/*
 * X macro infrastructure for mock generation.
 *
 * To add a new mocked function:
 *   1. Define EFA_MOCK_PARAMS_<name> and EFA_MOCK_ARGS_<name> below
 *   2. Add X(ret, name) to EFA_MOCK_FUNCTIONS
 *   3. Add any needed forward struct declarations above
 *   4. Add -Wl,--wrap=<name> to prov_efa_test_gtest_efa_gtest_LDFLAGS
 *      in prov/efa/Makefile.include
 */

/* --- Per-function parameter definitions --- */

#define EFA_MOCK_PARAMS_ibv_create_ah \
	struct ibv_pd *pd, struct ibv_ah_attr *attr
#define EFA_MOCK_ARGS_ibv_create_ah pd, attr

#define EFA_MOCK_PARAMS_ibv_destroy_ah struct ibv_ah *ibv_ah
#define EFA_MOCK_ARGS_ibv_destroy_ah   ibv_ah

#define EFA_MOCK_PARAMS_efadv_query_ah \
	struct ibv_ah *ibv_ah, struct efadv_ah_attr *attr, uint32_t inlen
#define EFA_MOCK_ARGS_efadv_query_ah ibv_ah, attr, inlen

#define EFA_MOCK_PARAMS_efa_av_reverse_av_add                          \
	struct efa_av *av, struct efa_cur_reverse_av **cur_reverse_av, \
		struct efa_prv_reverse_av **prv_reverse_av,            \
		struct efa_conn *conn
#define EFA_MOCK_ARGS_efa_av_reverse_av_add \
	av, cur_reverse_av, prv_reverse_av, conn

#define EFA_MOCK_PARAMS_ofi_mr_map_insert                                  \
	struct ofi_mr_map *map, const struct fi_mr_attr *attr,             \
		uint64_t *key, void *context, uint64_t flags
#define EFA_MOCK_ARGS_ofi_mr_map_insert map, attr, key, context, flags
#define EFA_MOCK_PARAMS_efa_rdm_pke_clone                              \
	struct efa_rdm_pke *src, struct ofi_bufpool *pkt_pool, int alloc_type
#define EFA_MOCK_ARGS_efa_rdm_pke_clone src, pkt_pool, alloc_type

#define EFA_MOCK_PARAMS_efa_qp_post_recv \
	struct efa_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad
#define EFA_MOCK_ARGS_efa_qp_post_recv qp, wr, bad

#define EFA_MOCK_PARAMS_efa_qp_post_send                                        \
	struct efa_qp *qp, const struct ibv_sge *sge_list,                     \
		const struct ibv_data_buf *inline_data_list, size_t iov_count, \
		bool use_inline, uintptr_t wr_id, uint64_t data,               \
		uint64_t flags, struct efa_ah *ah, uint32_t qpn, uint32_t qkey
#define EFA_MOCK_ARGS_efa_qp_post_send                                        \
	qp, sge_list, inline_data_list, iov_count, use_inline, wr_id, data,   \
		flags, ah, qpn, qkey

#define EFA_MOCK_PARAMS_efa_qp_post_read                                       \
	struct efa_qp *qp, const struct ibv_sge *sge_list, size_t sge_count,  \
		uint32_t remote_key, uint64_t remote_addr, uintptr_t wr_id,   \
		uint64_t flags, struct efa_ah *ah, uint32_t qpn, uint32_t qkey
#define EFA_MOCK_ARGS_efa_qp_post_read \
	qp, sge_list, sge_count, remote_key, remote_addr, wr_id, flags, ah, qpn, qkey

#define EFA_MOCK_PARAMS_efa_qp_post_write                                      \
	struct efa_qp *qp, const struct ibv_sge *sge_list, size_t sge_count,  \
		const struct ibv_data_buf *inline_data_list, bool use_inline, \
		uint32_t remote_key, uint64_t remote_addr, uintptr_t wr_id,   \
		uint64_t data, uint64_t flags, struct efa_ah *ah,             \
		uint32_t qpn, uint32_t qkey
#define EFA_MOCK_ARGS_efa_qp_post_write                                  \
	qp, sge_list, sge_count, inline_data_list, use_inline, remote_key, \
		remote_addr, wr_id, data, flags, ah, qpn, qkey

#define EFA_MOCK_PARAMS_efa_ibv_cq_start_poll \
	struct efa_ibv_cq *ibv_cq, struct ibv_poll_cq_attr *attr
#define EFA_MOCK_ARGS_efa_ibv_cq_start_poll ibv_cq, attr

#define EFA_MOCK_PARAMS_efa_ibv_cq_end_poll struct efa_ibv_cq *ibv_cq
#define EFA_MOCK_ARGS_efa_ibv_cq_end_poll   ibv_cq

#define EFA_MOCK_PARAMS_efa_ibv_cq_wc_read_opcode struct efa_ibv_cq *ibv_cq
#define EFA_MOCK_ARGS_efa_ibv_cq_wc_read_opcode	  ibv_cq

#define EFA_MOCK_PARAMS_efa_ibv_cq_wc_read_vendor_err struct efa_ibv_cq *ibv_cq
#define EFA_MOCK_ARGS_efa_ibv_cq_wc_read_vendor_err   ibv_cq

#define EFA_MOCK_PARAMS_efa_ibv_cq_wc_read_qp_num struct efa_ibv_cq *ibv_cq
#define EFA_MOCK_ARGS_efa_ibv_cq_wc_read_qp_num	  ibv_cq

#define EFA_MOCK_PARAMS_efa_ibv_cq_wc_read_wc_flags struct efa_ibv_cq *ibv_cq
#define EFA_MOCK_ARGS_efa_ibv_cq_wc_read_wc_flags   ibv_cq

#define EFA_MOCK_PARAMS_efa_ibv_cq_wc_read_byte_len struct efa_ibv_cq *ibv_cq
#define EFA_MOCK_ARGS_efa_ibv_cq_wc_read_byte_len   ibv_cq

/* --- Function list --- */

#define EFA_MOCK_FUNCTIONS(X)             \
	X(struct ibv_ah *, ibv_create_ah) \
	X(int, ibv_destroy_ah)            \
	X(int, efadv_query_ah)            \
	X(int, efa_av_reverse_av_add)     \
	X(int, efa_qp_post_recv)          \
	X(int, efa_qp_post_send)          \
	X(int, efa_qp_post_read)          \
	X(int, efa_qp_post_write)		\
	X(int, efa_ibv_cq_start_poll)                    \
	X(void, efa_ibv_cq_end_poll)                     \
	X(enum ibv_wc_opcode, efa_ibv_cq_wc_read_opcode) \
	X(uint32_t, efa_ibv_cq_wc_read_vendor_err)       \
	X(uint32_t, efa_ibv_cq_wc_read_qp_num)           \
	X(unsigned int, efa_ibv_cq_wc_read_wc_flags)     \
	X(uint32_t, efa_ibv_cq_wc_read_byte_len)		\
	X(int, ofi_mr_map_insert)						\
	X(struct efa_rdm_pke *, efa_rdm_pke_clone)

/* --- Generator macros --- */

#define EFA_MOCK_GEN_METHOD(ret, name) \
	MOCK_METHOD(ret, name, (EFA_MOCK_PARAMS_##name));

#define EFA_MOCK_GEN_REAL_DECL(ret, name) \
	ret __real_##name(EFA_MOCK_PARAMS_##name);

// Generate the definitions of the MOCK_METHOD's in the MockEfa class
class MockEfa
{
	public:
	EFA_MOCK_FUNCTIONS(EFA_MOCK_GEN_METHOD)

	static MockEfa *get();
	static void set(MockEfa *instance);
};

// Generate the declarations of the real function prototypes
extern "C" {
EFA_MOCK_FUNCTIONS(EFA_MOCK_GEN_REAL_DECL)
}

#endif /* EFA_GTEST_COMMON_MOCKS_H */
