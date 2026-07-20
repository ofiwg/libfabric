/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#ifndef EFA_GTEST_COMMON_MOCKS_H
#define EFA_GTEST_COMMON_MOCKS_H

#include <bitset>
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
 * X-macro list of every mocked function. Each row is
 *   X(ret, name, (param decls), (arg names))
 * 
 * To add a new mocked function:
 *   1. Add a row to EFA_MOCK_FUNCTIONS
 *   2. Add any needed forward struct declarations above
 *   3. Add -Wl,--wrap=<name> to prov_efa_test_gtest_efa_gtest_LDFLAGS
 *      in prov/efa/Makefile.include
 */
#define EFA_MOCK_FUNCTIONS(X)                                                  \
	X(struct ibv_ah *, ibv_create_ah,                                      \
	  (struct ibv_pd * pd, struct ibv_ah_attr * attr), (pd, attr))         \
	X(int, ibv_destroy_ah, (struct ibv_ah * ibv_ah), (ibv_ah))             \
	X(int, efadv_query_ah,                                                 \
	  (struct ibv_ah * ibv_ah, struct efadv_ah_attr * attr,                \
	   uint32_t inlen),                                                    \
	  (ibv_ah, attr, inlen))                                               \
	X(int, efa_av_reverse_av_add,                                          \
	  (struct efa_av * av, struct efa_cur_reverse_av * *cur_reverse_av,    \
	   struct efa_prv_reverse_av * *prv_reverse_av,                        \
	   struct efa_conn * conn),                                            \
	  (av, cur_reverse_av, prv_reverse_av, conn))                          \
	X(int, efa_ibv_cq_start_poll,                                          \
	  (struct efa_ibv_cq * ibv_cq, struct ibv_poll_cq_attr * attr),        \
	  (ibv_cq, attr))                                                      \
	X(void, efa_ibv_cq_end_poll, (struct efa_ibv_cq * ibv_cq), (ibv_cq))   \
	X(enum ibv_wc_opcode, efa_ibv_cq_wc_read_opcode,                       \
	  (struct efa_ibv_cq * ibv_cq), (ibv_cq))                              \
	X(uint32_t, efa_ibv_cq_wc_read_vendor_err,                             \
	  (struct efa_ibv_cq * ibv_cq), (ibv_cq))                              \
	X(uint32_t, efa_ibv_cq_wc_read_qp_num, (struct efa_ibv_cq * ibv_cq),   \
	  (ibv_cq))                                                            \
	X(unsigned int, efa_ibv_cq_wc_read_wc_flags,                           \
	  (struct efa_ibv_cq * ibv_cq), (ibv_cq))                              \
	X(uint32_t, efa_ibv_cq_wc_read_byte_len, (struct efa_ibv_cq * ibv_cq), \
	  (ibv_cq))                                                            \
	X(int, ofi_mr_map_insert,                                              \
	  (struct ofi_mr_map * map, const struct fi_mr_attr *attr,             \
	   uint64_t *key, void *context, uint64_t flags),                      \
	  (map, attr, key, context, flags))                                    \
	X(struct efa_rdm_pke *, efa_rdm_pke_clone,                             \
	  (struct efa_rdm_pke * src, struct ofi_bufpool * pkt_pool,            \
	   int alloc_type),                                                    \
	  (src, pkt_pool, alloc_type))                                         \
	X(int, efa_qp_post_recv,                                               \
	  (struct efa_qp * qp, struct ibv_recv_wr * wr,                        \
	   struct ibv_recv_wr * *bad),                                         \
	  (qp, wr, bad))                                                       \
	X(int, efa_qp_post_send,                                               \
	  (struct efa_qp * qp, const struct ibv_sge *sge_list,                 \
	   const struct ibv_data_buf *inline_data_list, size_t iov_count,      \
	   bool use_inline, uintptr_t wr_id, uint64_t data, uint64_t flags,    \
	   struct efa_ah *ah, uint32_t qpn, uint32_t qkey),                    \
	  (qp, sge_list, inline_data_list, iov_count, use_inline, wr_id, data, \
	   flags, ah, qpn, qkey))                                              \
	X(int, efa_qp_post_read,                                               \
	  (struct efa_qp * qp, const struct ibv_sge *sge_list,                 \
	   size_t sge_count, uint32_t remote_key, uint64_t remote_addr,        \
	   uintptr_t wr_id, uint64_t flags, struct efa_ah *ah, uint32_t qpn,   \
	   uint32_t qkey),                                                     \
	  (qp, sge_list, sge_count, remote_key, remote_addr, wr_id, flags, ah, \
	   qpn, qkey))                                                         \
	X(int, efa_qp_post_write,                                              \
	  (struct efa_qp * qp, const struct ibv_sge *sge_list,                 \
	   size_t sge_count, const struct ibv_data_buf *inline_data_list,      \
	   bool use_inline, uint32_t remote_key, uint64_t remote_addr,         \
	   uintptr_t wr_id, uint64_t data, uint64_t flags, struct efa_ah *ah,  \
	   uint32_t qpn, uint32_t qkey),                                       \
	  (qp, sge_list, sge_count, inline_data_list, use_inline, remote_key,  \
	   remote_addr, wr_id, data, flags, ah, qpn, qkey))

/* --- Generator macros --- */

#define EFA_MOCK_GEN_METHOD(ret, name, params, args) \
	MOCK_METHOD(ret, name, params);

#define EFA_MOCK_GEN_REAL_DECL(ret, name, params, args) \
	ret __real_##name params;

#define EFA_MOCK_GEN_ENUM(ret, name, params, args) FN_##name,

class MockEfa
{
	public:
	EFA_MOCK_FUNCTIONS(EFA_MOCK_GEN_METHOD)

	/* FN_COUNT is the last member of the enum, hence the number of wrapped fns */
	enum Fn { EFA_MOCK_FUNCTIONS(EFA_MOCK_GEN_ENUM) FN_COUNT };

	void arm(Fn fn) { armed_.set(fn); }
	/* test whether a wrapped function should be mocked in the current test */
	bool is_armed(Fn fn) const { return armed_.test(fn); }

	static MockEfa *get();
	static void set(MockEfa *instance);

	private:
	std::bitset<FN_COUNT> armed_;
};

#define EFA_EXPECT_CALL(obj, name, ...)          \
	((obj).arm(MockEfa::FN_##name),          \
	 EXPECT_CALL(obj, name __VA_OPT__((__VA_ARGS__))))

// Generate the declarations of the real function prototypes
extern "C" {
EFA_MOCK_FUNCTIONS(EFA_MOCK_GEN_REAL_DECL)
}

#endif /* EFA_GTEST_COMMON_MOCKS_H */
