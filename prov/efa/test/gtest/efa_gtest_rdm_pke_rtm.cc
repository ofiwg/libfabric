/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include "efa_gtest_rdm_pke_utils.h"
#include <gtest/gtest.h>
#include <string>

using testing::Bool;
using testing::StrictMock;
using testing::Test;
using testing::TestWithParam;
using testing::ValuesIn;
using testing::WithParamInterface;

class EfaRtmRecvTest : public TestWithParam<bool>
{
	protected:
	struct efa_resource resource = {};
	StrictMock<MockEfa> mock_efa;
	fi_addr_t peer_addr = FI_ADDR_UNSPEC;

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));
		efa_test_resource_construct(
			&resource, efa_test_alloc_default_hints(
					   FI_EP_RDM, EFA_FABRIC_NAME));
		ASSERT_NE(resource.ep, nullptr);

		ASSERT_EQ(efa_test_av_insert_self(resource.ep, resource.av,
						  &peer_addr),
			  1);

		MockEfa::set(&mock_efa);
		efa_test_arm_inert_data_path(mock_efa);
	}

	void TearDown() override
	{
		MockEfa::set(nullptr);
		efa_test_resource_destruct(&resource);
	}
};

/**
 * @brief Asserts that a READ_NACK RTM whose msg_id misses the peer's rxe_map is
 * handled (not a NULL deref) in both efa_rdm_pke_proc_{msg/tag}rtm
 */
TEST_P(EfaRtmRecvTest, read_nack_missing_rxe_no_null_deref)
{
	ssize_t ret = 0;

	ASSERT_EQ(efa_test_rtm_read_nack_missing_rxe(resource.ep, peer_addr,
						     GetParam(), &ret),
		  1);
	/* Reached only if the handler did not crash; the missing entry is an
	 * error, not silent success. */
	EXPECT_NE(ret, 0);
}

INSTANTIATE_TEST_SUITE_P(, EfaRtmRecvTest, Bool(),
			 [](const testing::TestParamInfo<bool> &info) {
				 return info.param ? "tagrtm" : "msgrtm";
			 });

// 16 init functions in efa_rdm_pke_rtm.c plus the eager/zero-header branch
static const enum efa_test_rtm_variant kRtmInitVariants[] = {
	EFA_TEST_RTM_EAGER_MSG,		 EFA_TEST_RTM_EAGER_TAG,
	EFA_TEST_RTM_DC_EAGER_MSG,	 EFA_TEST_RTM_DC_EAGER_TAG,
	EFA_TEST_RTM_MEDIUM_MSG,	 EFA_TEST_RTM_MEDIUM_TAG,
	EFA_TEST_RTM_DC_MEDIUM_MSG,	 EFA_TEST_RTM_DC_MEDIUM_TAG,
	EFA_TEST_RTM_LONGCTS_MSG,	 EFA_TEST_RTM_LONGCTS_TAG,
	EFA_TEST_RTM_DC_LONGCTS_MSG,	 EFA_TEST_RTM_DC_LONGCTS_TAG,
	EFA_TEST_RTM_LONGREAD_MSG,	 EFA_TEST_RTM_LONGREAD_TAG,
	EFA_TEST_RTM_RUNTREAD_MSG,	 EFA_TEST_RTM_RUNTREAD_TAG,
	EFA_TEST_RTM_EAGER_MSG_ZERO_HDR,
};

/* gtest test-name suffix, reconstructed from the variant bits. */
static std::string rtm_variant_name(enum efa_test_rtm_variant v)
{
	static const char *family[] = {"eager", "medium", "longcts", "longread",
				       "runtread"};
	std::string s;
	if (EFA_TEST_RTM_IS_DC(v))
		s += "dc_";
	s += family[EFA_TEST_RTM_FAMILY(v)];
	s += EFA_TEST_RTM_IS_TAGGED(v) ? "_tag" : "_msg";
	if (EFA_TEST_RTM_IS_ZERO_HDR(v))
		s += "_zero_hdr";
	return s;
}

class EfaRtmTxFixture : public Test
{
	protected:
	struct efa_resource resource = {};

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));
		efa_test_resource_construct(
			&resource, efa_test_alloc_default_hints(
					   FI_EP_RDM, EFA_FABRIC_NAME));
		ASSERT_NE(resource.ep, nullptr);
	}

	void TearDown() override
	{
		efa_test_resource_destruct(&resource);
	}
};

/**
 * @brief Covers the RTM TX-init functions (efa_rdm_pke_init_*rtm), which stamp
 * a packet header and copy/describe payload from a txe.
 */
class EfaRtmTxInitTest :
	public EfaRtmTxFixture,
	public WithParamInterface<enum efa_test_rtm_variant>
{
};

/**
 * @brief Each init function stamps the header fields and payload its protocol
 * family is responsible for; the variant bits select which fields are asserted
 * and to which sentinel.
 */
TEST_P(EfaRtmTxInitTest, stamps_header_and_payload)
{
	enum efa_test_rtm_variant v = GetParam();
	bool dc = EFA_TEST_RTM_IS_DC(v);
	bool tagged = EFA_TEST_RTM_IS_TAGGED(v);
	struct efa_test_rtm_init_result res;

	efa_test_rtm_init_build(resource.ep, resource.av, v, &res);
	ASSERT_EQ(res.ret, 0) << "init failed for " << rtm_variant_name(v);

	if (EFA_TEST_RTM_IS_ZERO_HDR(v)) {
		EXPECT_EQ(res.payload_size, EFA_TEST_RTM_EAGER_LEN);
		return;
	}

	EXPECT_EQ(res.base_type, efa_test_rtm_pkt_type(v));
	EXPECT_TRUE(res.has_msg_flag);
	EXPECT_EQ(res.msg_id, EFA_TEST_RTM_MSG_ID);

	EXPECT_EQ(res.has_tagged_flag, tagged);
	EXPECT_EQ(res.tag, tagged ? EFA_TEST_RTM_TAG : 0u);

	EXPECT_EQ(res.dc_requested, dc);

	switch (EFA_TEST_RTM_FAMILY(v)) {
	case EFA_TEST_RTM_FAM_EAGER:
		EXPECT_EQ(res.payload_size, EFA_TEST_RTM_EAGER_LEN);
		EXPECT_EQ(res.send_id, dc ? EFA_TEST_RTM_SEND_ID : 0u);
		break;
	case EFA_TEST_RTM_FAM_MEDIUM:
		EXPECT_EQ(res.payload_size, EFA_TEST_RTM_MEDIUM_DATA);
		EXPECT_EQ(res.msg_length, EFA_TEST_RTM_LONG_LEN);
		EXPECT_EQ(res.seg_offset, EFA_TEST_RTM_MEDIUM_SEG);
		EXPECT_EQ(res.send_id, dc ? EFA_TEST_RTM_SEND_ID : 0u);
		break;
	case EFA_TEST_RTM_FAM_LONGCTS:
		// longcts payload size is mtu_size - req_hdr_size
		// so it depends on the device
		EXPECT_GT(res.payload_size, 0u);
		EXPECT_LT(res.payload_size, EFA_TEST_RTM_LONG_LEN);
		EXPECT_EQ(res.msg_length, EFA_TEST_RTM_LONG_LEN);
		EXPECT_EQ(res.send_id, EFA_TEST_RTM_SEND_ID);
		EXPECT_EQ(res.credit_request,
			  (uint32_t) efa_test_get_tx_min_credits());
		break;
	case EFA_TEST_RTM_FAM_LONGREAD:
		EXPECT_EQ(res.payload_size, 0u);
		EXPECT_EQ(res.msg_length, EFA_TEST_RTM_LONG_LEN);
		EXPECT_EQ(res.send_id, EFA_TEST_RTM_SEND_ID);
		EXPECT_EQ(res.read_iov_count, 1u);
		break;
	case EFA_TEST_RTM_FAM_RUNTREAD:
		EXPECT_EQ(res.payload_size, EFA_TEST_RTM_RUNT_DATA);
		EXPECT_EQ(res.msg_length, EFA_TEST_RTM_LONG_LEN);
		EXPECT_EQ(res.send_id, EFA_TEST_RTM_SEND_ID);
		EXPECT_EQ(res.seg_offset, EFA_TEST_RTM_RUNT_SEG);
		EXPECT_EQ(res.runt_length, EFA_TEST_RTM_RUNT_LEN);
		EXPECT_EQ(res.read_iov_count, 1u);
		break;
	}
}

INSTANTIATE_TEST_SUITE_P(
	, EfaRtmTxInitTest, ValuesIn(kRtmInitVariants),
	[](const testing::TestParamInfo<enum efa_test_rtm_variant> &info) {
		return rtm_variant_name(info.param);
	});

/**
 * @brief Covers the TX sent and send-completion handlers 
 */
class EfaRtmTxSentTest : public EfaRtmTxFixture
{
};

/* A non-boundary chunk */
static constexpr size_t kChunk = 4096;

TEST_F(EfaRtmTxSentTest, medium_sent_and_completion_boundary)
{
	struct efa_test_rtm_sent_result res;

	/* sent: bytes_sent advances by the payload, no completion. */
	efa_test_rtm_sent_build(resource.ep, resource.av,
				EFA_TEST_RTM_MEDIUM_MSG, EFA_TEST_RTM_OP_SENT,
				/*payload_size=*/kChunk,
				/*bytes_already=*/kChunk, 0, &res);
	EXPECT_EQ(res.bytes_sent, 2 * kChunk);

	/* request completion mid-transfer: bytes_acked advances but no
	 * completion. */
	efa_test_rtm_sent_build(
		resource.ep, resource.av, EFA_TEST_RTM_MEDIUM_MSG,
		EFA_TEST_RTM_OP_COMPLETION, /*payload_size=*/kChunk,
		/*bytes_already=*/0, 0, &res);
	EXPECT_EQ(res.bytes_acked, kChunk);
	EXPECT_FALSE(res.send_completed);
	EXPECT_TRUE(res.txe_on_ope_list);
	EXPECT_FALSE(res.cq_has_completion);

	/* completion at the boundary: the final byte completes the op, frees
	 * the txe, and writes the SEND completion to the CQ. */
	efa_test_rtm_sent_build(resource.ep, resource.av,
				EFA_TEST_RTM_MEDIUM_MSG,
				EFA_TEST_RTM_OP_COMPLETION,
				/*payload_size=*/EFA_TEST_RTM_LONG_LEN,
				/*bytes_already=*/0, 0, &res);
	EXPECT_TRUE(res.send_completed);
	EXPECT_FALSE(res.txe_on_ope_list);
	EXPECT_TRUE(res.cq_has_completion);
}

TEST_F(EfaRtmTxSentTest, eager_completion_single_shot)
{
	struct efa_test_rtm_sent_result res;

	/* payload_size must equal total_len */
	efa_test_rtm_sent_build(resource.ep, resource.av, EFA_TEST_RTM_EAGER_MSG,
				EFA_TEST_RTM_OP_COMPLETION,
				/*payload_size=*/EFA_TEST_RTM_LONG_LEN,
				/*bytes_already=*/0, 0, &res);
	EXPECT_TRUE(res.send_completed);
	EXPECT_FALSE(res.txe_on_ope_list);
	EXPECT_TRUE(res.cq_has_completion);
}

TEST_F(EfaRtmTxSentTest, longcts_sent_and_completion_boundary)
{
	struct efa_test_rtm_sent_result res;

	efa_test_rtm_sent_build(resource.ep, resource.av,
				EFA_TEST_RTM_LONGCTS_MSG, EFA_TEST_RTM_OP_SENT,
				/*payload_size=*/kChunk,
				/*bytes_already=*/kChunk, 0, &res);
	EXPECT_EQ(res.bytes_sent, 2 * kChunk);

	efa_test_rtm_sent_build(resource.ep, resource.av,
				EFA_TEST_RTM_LONGCTS_MSG,
				EFA_TEST_RTM_OP_COMPLETION,
				/*payload_size=*/EFA_TEST_RTM_LONG_LEN,
				/*bytes_already=*/0, 0, &res);
	EXPECT_TRUE(res.send_completed);
	EXPECT_TRUE(res.cq_has_completion);
}

/**
 * @brief longread "sent" bumps the domain's in-flight read count
 */
TEST_F(EfaRtmTxSentTest, longread_sent_bumps_in_flight_read)
{
	struct efa_test_rtm_sent_result res;

	efa_test_rtm_sent_build(resource.ep, resource.av,
				EFA_TEST_RTM_LONGREAD_MSG, EFA_TEST_RTM_OP_SENT,
				/*payload_size=*/0, /*bytes_already=*/0, 0,
				&res);
	EXPECT_EQ(res.num_read_msg_in_flight, 1u);
}

/**
 * @brief runtread sent accrues bytes_sent and the peer's runt-in-flight
 * bytes, and bumps the in-flight read count only for the first segment
 */
TEST_F(EfaRtmTxSentTest, runtread_sent_first_segment)
{
	struct efa_test_rtm_sent_result res;

	efa_test_rtm_sent_build(resource.ep, resource.av,
				EFA_TEST_RTM_RUNTREAD_MSG, EFA_TEST_RTM_OP_SENT,
				/*payload_size=*/kChunk, /*bytes_already=*/0,
				/*seg_offset=*/0, &res);
	EXPECT_EQ(res.bytes_sent, kChunk);
	EXPECT_EQ(res.num_runt_bytes_in_flight, (int64_t) kChunk);
	EXPECT_EQ(res.num_read_msg_in_flight, 1u);

	efa_test_rtm_sent_build(resource.ep, resource.av,
				EFA_TEST_RTM_RUNTREAD_MSG, EFA_TEST_RTM_OP_SENT,
				/*payload_size=*/kChunk, /*bytes_already=*/0,
				/*seg_offset=*/kChunk, &res);
	EXPECT_EQ(res.num_read_msg_in_flight, 0u);
}

TEST_F(EfaRtmTxSentTest, runtread_completion_boundary)
{
	struct efa_test_rtm_sent_result res;

	/* num_runt_bytes_in_flight is set to payload_size, so a
	 * completion drains it back to zero and completes the op. */
	efa_test_rtm_sent_build(resource.ep, resource.av,
				EFA_TEST_RTM_RUNTREAD_MSG,
				EFA_TEST_RTM_OP_COMPLETION,
				/*payload_size=*/EFA_TEST_RTM_LONG_LEN,
				/*bytes_already=*/0, 0, &res);
	EXPECT_TRUE(res.send_completed);
	EXPECT_EQ(res.num_runt_bytes_in_flight, 0);
	EXPECT_TRUE(res.cq_has_completion);
}
