/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include "efa_gtest_rdm_pke_utils.h"
#include <cstring>
#include <gtest/gtest.h>
#include <rdma/fi_errno.h>

using testing::_;
using testing::Invoke;
using testing::NotNull;
using testing::Range;
using testing::Return;
using testing::StrictMock;
using testing::Test;
using testing::WithParamInterface;

class EfaRdmPkeTest : public Test
{
	protected:
	struct efa_resource resource = {};

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));
		efa_test_resource_construct(
			&resource,
			efa_test_alloc_default_hints(FI_EP_RDM, EFA_FABRIC_NAME));
		ASSERT_NE(resource.ep, nullptr);
	}

	void TearDown() override
	{
		efa_test_resource_destruct(&resource);
	}
};

/* pke_release_cloned must release arbitrary length packet list correctly */
class EfaRdmPkeChainTest : public EfaRdmPkeTest,
			   public WithParamInterface<size_t>
{
};

TEST_P(EfaRdmPkeChainTest, release_cloned_frees_whole_chain)
{
	size_t n = GetParam();
	struct efa_rdm_pke *head;

	head = efa_test_pke_build_unexp_chain(resource.ep, n);
	ASSERT_EQ(head == nullptr, n == 0);
	ASSERT_EQ(efa_test_ep_unexp_pool_outstanding(resource.ep), n);

	efa_test_pke_release_cloned(head);

	EXPECT_EQ(efa_test_ep_unexp_pool_outstanding(resource.ep), 0u);
}

INSTANTIATE_TEST_SUITE_P(ChainLengths, EfaRdmPkeChainTest, Range<size_t>(0, 6));

class EfaRdmPkeRtaTest : public Test
{
	protected:
	struct efa_resource resource = {};
	StrictMock<MockEfa> mock_efa;
	fi_addr_t peer_addr = FI_ADDR_NOTAVAIL;

	void SetUp() override
	{
		struct fi_info *hints;

		memset(&resource, 0, sizeof(resource));
		hints = efa_test_alloc_default_hints(FI_EP_RDM,
						     EFA_FABRIC_NAME);
		ASSERT_NE(hints, nullptr);
		hints->caps |= FI_ATOMIC;
		efa_test_resource_construct(&resource, hints);
		ASSERT_NE(resource.ep, nullptr);
		ASSERT_EQ(efa_test_av_insert_self(resource.ep, resource.av,
						  &peer_addr),
			  1);
		MockEfa::set(&mock_efa);
	}

	void TearDown() override
	{
		MockEfa::set(nullptr);
		efa_test_resource_destruct(&resource);
	}

	void expect_clean(const struct efa_test_rta_responder_result &res)
	{
		EXPECT_EQ(res.ope_count, 0u);
		EXPECT_EQ(res.tx_pkt_pool_used, 0u);
		EXPECT_EQ(res.rxe_pool_used, 0u);
		EXPECT_EQ(res.atomrsp_pool_used, 0u);
		EXPECT_EQ(res.unexp_pool_used, 0u);
		EXPECT_EQ(res.outstanding_tx_ops, 0u);
		EXPECT_EQ(res.peer_outstanding_tx_ops, 0u);
	}

	void expect_hmem_copies()
	{
		EFA_EXPECT_CALL(mock_efa, efa_copy_from_hmem, NotNull(), _, _,
				1u)
			.WillOnce(Invoke([](void *, void *dest, const void *src,
					    size_t size) {
				std::memcpy(dest, src, size);
				return 0;
			}));
		EFA_EXPECT_CALL(mock_efa, efa_copy_to_hmem, NotNull(), _, _, 1u)
			.WillOnce(Invoke([](void *, void *dest, const void *src,
					    size_t size) {
				std::memcpy(dest, src, size);
				return 0;
			}));
	}
};

TEST_F(EfaRdmPkeRtaTest, invalid_key_rejects_all_responder_paths)
{
	const struct {
		enum efa_test_rta_path path;
		enum fi_op op;
	} cases[] = {
		{EFA_TEST_RTA_WRITE, FI_ATOMIC_WRITE},
		{EFA_TEST_RTA_DC_WRITE, FI_ATOMIC_WRITE},
		{EFA_TEST_RTA_FETCH, FI_SUM},
		{EFA_TEST_RTA_COMPARE, FI_CSWAP},
	};

	EFA_EXPECT_CALL(mock_efa, efa_qp_post_send).Times(0);
	for (const auto &test_case : cases) {
		struct efa_test_rta_responder_result res = {};

		SCOPED_TRACE(testing::Message() << "path=" << test_case.path);
		ASSERT_EQ(efa_test_rta_responder(
				  resource.ep, resource.domain, peer_addr,
				  test_case.path, test_case.op,
				  FI_REMOTE_READ | FI_REMOTE_WRITE, 1, 1, &res),
			  0);
		EXPECT_EQ(res.ret, -FI_EINVAL);
		EXPECT_GT(res.eq_read_ret, 0);
		EXPECT_EQ(res.eq_err, FI_EINVAL);
		EXPECT_EQ(res.eq_prov_errno,
			  efa_test_rta_rma_addr_prov_errno());
		EXPECT_EQ(res.target[0], 1);
		EXPECT_EQ(res.target[1], 4);
		EXPECT_EQ(res.response_valid, 0);
		expect_clean(res);
	}
}

TEST_F(EfaRdmPkeRtaTest, read_only_mr_rejects_mutating_fetch_and_compare)
{
	const struct {
		enum efa_test_rta_path path;
		enum fi_op op;
	} cases[] = {
		{EFA_TEST_RTA_FETCH, FI_SUM},
		{EFA_TEST_RTA_COMPARE, FI_CSWAP},
	};

	EFA_EXPECT_CALL(mock_efa, efa_qp_post_send).Times(0);
	for (const auto &test_case : cases) {
		struct efa_test_rta_responder_result res = {};

		SCOPED_TRACE(testing::Message() << "path=" << test_case.path);
		ASSERT_EQ(efa_test_rta_responder(resource.ep, resource.domain,
						 peer_addr, test_case.path,
						 test_case.op, FI_REMOTE_READ,
						 0, 1, &res),
			  0);
		EXPECT_EQ(res.ret, -FI_EACCES);
		EXPECT_GT(res.eq_read_ret, 0);
		EXPECT_EQ(res.eq_err, FI_EACCES);
		EXPECT_EQ(res.eq_prov_errno,
			  efa_test_rta_rma_addr_prov_errno());
		EXPECT_EQ(res.target[0], 1);
		EXPECT_EQ(res.target[1], 4);
		EXPECT_EQ(res.response_valid, 0);
		expect_clean(res);
	}
}

TEST_F(EfaRdmPkeRtaTest, atomic_read_accepts_read_only_mr)
{
	struct efa_test_rta_responder_result res = {};

	EFA_EXPECT_CALL(mock_efa, efa_qp_post_send).WillOnce(Return(0));
	ASSERT_EQ(efa_test_rta_responder(resource.ep, resource.domain,
					 peer_addr, EFA_TEST_RTA_FETCH,
					 FI_ATOMIC_READ, FI_REMOTE_READ, 0, 1,
					 &res),
		  0);
	EXPECT_EQ(res.ret, 0);
	EXPECT_EQ(res.eq_read_ret, -FI_EAGAIN);
	EXPECT_EQ(res.target[0], 1);
	EXPECT_EQ(res.target[1], 4);
	EXPECT_EQ(res.response_valid, 1);
	EXPECT_EQ(res.response, 1);
	expect_clean(res);
}

TEST_F(EfaRdmPkeRtaTest, multi_iov_write_is_all_or_nothing)
{
	struct efa_test_rta_responder_result res = {};

	EFA_EXPECT_CALL(mock_efa, efa_qp_post_send).Times(0);
	ASSERT_EQ(efa_test_rta_responder(
			  resource.ep, resource.domain, peer_addr,
			  EFA_TEST_RTA_WRITE, FI_ATOMIC_WRITE,
			  FI_REMOTE_READ | FI_REMOTE_WRITE, 1, 2, &res),
		  0);
	EXPECT_EQ(res.ret, -FI_EINVAL);
	EXPECT_GT(res.eq_read_ret, 0);
	EXPECT_EQ(res.eq_err, FI_EINVAL);
	EXPECT_EQ(res.eq_prov_errno, efa_test_rta_rma_addr_prov_errno());
	EXPECT_EQ(res.target[0], 1);
	EXPECT_EQ(res.target[1], 4);
	expect_clean(res);
}

class EfaRdmPkeRtaVerifiedDescTest :
	public EfaRdmPkeRtaTest,
	public WithParamInterface<enum fi_op>
{
};

TEST_P(EfaRdmPkeRtaVerifiedDescTest, responder_uses_verified_mr_desc)
{
	enum fi_op op = GetParam();
	struct efa_test_rta_responder_result res = {};

	expect_hmem_copies();
	if (op == FI_ATOMIC_WRITE)
		EFA_EXPECT_CALL(mock_efa, efa_qp_post_send).Times(0);
	else
		EFA_EXPECT_CALL(mock_efa, efa_qp_post_send).WillOnce(Return(0));

	ASSERT_EQ(efa_test_rta_responder_uses_verified_mr_desc(
			  resource.ep, resource.domain, peer_addr, op, &res),
		  0);
	EXPECT_EQ(res.registered_iface, FI_HMEM_SYSTEM);
	EXPECT_EQ(res.ret, 0);
	EXPECT_EQ(res.target[0], op == FI_SUM ? 3 : 2);
	EXPECT_EQ(res.target[1], 4);
	EXPECT_EQ(res.response_valid, op != FI_ATOMIC_WRITE);
	if (op != FI_ATOMIC_WRITE) {
		EXPECT_EQ(res.response, 1);
		EXPECT_EQ(res.response_recv_id, EFA_TEST_RTA_RECV_ID);
	}
	EXPECT_EQ(res.eq_read_ret, -FI_EAGAIN);
	expect_clean(res);
}

INSTANTIATE_TEST_SUITE_P(VerifiedMrDesc, EfaRdmPkeRtaVerifiedDescTest,
			 testing::Values(FI_ATOMIC_WRITE, FI_SUM, FI_CSWAP),
			 [](const testing::TestParamInfo<enum fi_op> &info) {
				 switch (info.param) {
				 case FI_ATOMIC_WRITE:
					 return "write";
				 case FI_SUM:
					 return "fetch";
				 default:
					 return "compare";
				 }
			 });
