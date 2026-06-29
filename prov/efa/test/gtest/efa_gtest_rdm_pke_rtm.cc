/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_resource.h"
#include "efa_gtest_rdm_pke_utils.h"
#include <gtest/gtest.h>
#include <string>

using testing::TestWithParam;
using testing::ValuesIn;

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

/**
 * @brief Covers the RTM TX-init functions (efa_rdm_pke_init_*rtm), which stamp
 * a packet header and copy/describe payload from a txe.
 */
class EfaRdmPkeRtmInitTest : public TestWithParam<enum efa_test_rtm_variant>
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
 * @brief Each init function stamps the header fields and payload its protocol
 * family is responsible for; the variant bits select which fields are asserted
 * and to which sentinel.
 */
TEST_P(EfaRdmPkeRtmInitTest, stamps_header_and_payload)
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
	, EfaRdmPkeRtmInitTest, ValuesIn(kRtmInitVariants),
	[](const testing::TestParamInfo<enum efa_test_rtm_variant> &info) {
		return rtm_variant_name(info.param);
	});
