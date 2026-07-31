/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include "efa_gtest_tclass_utils.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>

using testing::AtLeast;
using testing::Invoke;
using testing::StrictMock;
using testing::Test;
using testing::TestWithParam;
using testing::Values;

/* An unsupported label and an unsupported DSCP value, per efa_tclass_supported */
#define EFA_TEST_TC_UNSUPPORTED	     FI_TC_BULK_DATA
#define EFA_TEST_TC_UNSUPPORTED_DSCP fi_tc_dscp_set(5)

/*
 * fi_getinfo() must echo a supported tclass hint back in the info it returns,
 * and must fall back to FI_TC_UNSPEC for anything EFA cannot honor.
 */
class EfaTclassInfoTest : public Test
{
	protected:
	struct fi_info *hints = nullptr;
	struct fi_info *info = nullptr;

	void TearDown() override
	{
		if (info)
			fi_freeinfo(info);
		if (hints)
			fi_freeinfo(hints);
	}

	/**
	 * @brief Run fi_getinfo with the given tclass hints.
	 *
	 * @param ep_type	endpoint type to request
	 * @param fabric_name	fabric to request
	 * @param domain_tclass	hints->domain_attr->tclass
	 * @param tx_tclass	hints->tx_attr->tclass
	 */
	void getinfo(enum fi_ep_type ep_type, const char *fabric_name,
		     uint32_t domain_tclass, uint32_t tx_tclass)
	{
		uint32_t version = !strcmp(fabric_name, EFA_DIRECT_FABRIC_NAME) ?
					   FI_VERSION(2, 0) :
					   FI_VERSION(1, 14);

		hints = efa_test_alloc_default_hints(ep_type, fabric_name);
		ASSERT_NE(hints, nullptr);
		hints->domain_attr->tclass = domain_tclass;
		hints->tx_attr->tclass = tx_tclass;

		int ret = fi_getinfo(version, NULL, NULL, 0ULL, hints, &info);
		ASSERT_EQ(ret, 0) << "fi_getinfo failed: " << fi_strerror(-ret);
		ASSERT_NE(info, nullptr);
	}
};

TEST_F(EfaTclassInfoTest, unset_hint_returns_unspec)
{
	ASSERT_NO_FATAL_FAILURE(getinfo(FI_EP_RDM, EFA_FABRIC_NAME,
					FI_TC_UNSPEC, FI_TC_UNSPEC));

	EXPECT_EQ(info->domain_attr->tclass, (uint32_t) FI_TC_UNSPEC);
	EXPECT_EQ(info->tx_attr->tclass, (uint32_t) FI_TC_UNSPEC);
}

TEST_F(EfaTclassInfoTest, low_latency_hint_propagates_to_both_fields)
{
	if (!efa_test_have_efadv_sl())
		GTEST_SKIP() << "build lacks efadv_qp_init_attr::sl, so "
				"FI_TC_LOW_LATENCY is not a supported tclass";

	ASSERT_NO_FATAL_FAILURE(getinfo(FI_EP_RDM, EFA_FABRIC_NAME,
					FI_TC_LOW_LATENCY,
					FI_TC_LOW_LATENCY));

	EXPECT_EQ(info->domain_attr->tclass, (uint32_t) FI_TC_LOW_LATENCY);
	EXPECT_EQ(info->tx_attr->tclass, (uint32_t) FI_TC_LOW_LATENCY);
}

/*
 * The two fields carry independent meanings (domain default vs. per-endpoint
 * override), so a hint on one must not bleed into the other.
 */
TEST_F(EfaTclassInfoTest, domain_and_tx_hints_propagate_independently)
{
	if (!efa_test_have_efadv_sl())
		GTEST_SKIP() << "build lacks efadv_qp_init_attr::sl, so "
				"FI_TC_LOW_LATENCY is not a supported tclass";

	ASSERT_NO_FATAL_FAILURE(getinfo(FI_EP_RDM, EFA_FABRIC_NAME,
					FI_TC_LOW_LATENCY, FI_TC_UNSPEC));

	EXPECT_EQ(info->domain_attr->tclass, (uint32_t) FI_TC_LOW_LATENCY);
	EXPECT_EQ(info->tx_attr->tclass, (uint32_t) FI_TC_UNSPEC);
}

TEST_F(EfaTclassInfoTest, best_effort_hint_propagates_to_both_fields)
{
	ASSERT_NO_FATAL_FAILURE(getinfo(FI_EP_RDM, EFA_FABRIC_NAME,
					FI_TC_BEST_EFFORT,
					FI_TC_BEST_EFFORT));

	EXPECT_EQ(info->domain_attr->tclass, (uint32_t) FI_TC_BEST_EFFORT);
	EXPECT_EQ(info->tx_attr->tclass, (uint32_t) FI_TC_BEST_EFFORT);
}

TEST_F(EfaTclassInfoTest, unsupported_label_hint_falls_back_to_unspec)
{
	ASSERT_NO_FATAL_FAILURE(getinfo(FI_EP_RDM, EFA_FABRIC_NAME,
					EFA_TEST_TC_UNSUPPORTED,
					EFA_TEST_TC_UNSUPPORTED));

	EXPECT_EQ(info->domain_attr->tclass, (uint32_t) FI_TC_UNSPEC);
	EXPECT_EQ(info->tx_attr->tclass, (uint32_t) FI_TC_UNSPEC);
}

TEST_F(EfaTclassInfoTest, unsupported_dscp_hint_falls_back_to_unspec)
{
	ASSERT_NO_FATAL_FAILURE(getinfo(FI_EP_RDM, EFA_FABRIC_NAME,
					EFA_TEST_TC_UNSUPPORTED_DSCP,
					EFA_TEST_TC_UNSUPPORTED_DSCP));

	EXPECT_EQ(info->domain_attr->tclass, (uint32_t) FI_TC_UNSPEC);
	EXPECT_EQ(info->tx_attr->tclass, (uint32_t) FI_TC_UNSPEC);
}

/* An unsupported hint on one field must not discard a supported hint on the other. */
TEST_F(EfaTclassInfoTest, unsupported_tx_hint_leaves_domain_hint_intact)
{
	if (!efa_test_have_efadv_sl())
		GTEST_SKIP() << "build lacks efadv_qp_init_attr::sl, so "
				"FI_TC_LOW_LATENCY is not a supported tclass";

	ASSERT_NO_FATAL_FAILURE(getinfo(FI_EP_RDM, EFA_FABRIC_NAME,
					FI_TC_LOW_LATENCY,
					EFA_TEST_TC_UNSUPPORTED));

	EXPECT_EQ(info->domain_attr->tclass, (uint32_t) FI_TC_LOW_LATENCY);
	EXPECT_EQ(info->tx_attr->tclass, (uint32_t) FI_TC_UNSPEC);
}

/* dgram has no alter function of its own, so it exercises the shared hook. */
TEST_F(EfaTclassInfoTest, dgram_hint_propagates)
{
	if (!efa_test_have_efadv_sl())
		GTEST_SKIP() << "build lacks efadv_qp_init_attr::sl, so "
				"FI_TC_LOW_LATENCY is not a supported tclass";

	ASSERT_NO_FATAL_FAILURE(getinfo(FI_EP_DGRAM, EFA_FABRIC_NAME,
					FI_TC_LOW_LATENCY,
					FI_TC_LOW_LATENCY));

	EXPECT_EQ(info->domain_attr->tclass, (uint32_t) FI_TC_LOW_LATENCY);
	EXPECT_EQ(info->tx_attr->tclass, (uint32_t) FI_TC_LOW_LATENCY);
}

TEST_F(EfaTclassInfoTest, efa_direct_hint_propagates)
{
	if (!efa_test_have_efadv_sl())
		GTEST_SKIP() << "build lacks efadv_qp_init_attr::sl, so "
				"FI_TC_LOW_LATENCY is not a supported tclass";

	ASSERT_NO_FATAL_FAILURE(getinfo(FI_EP_RDM, EFA_DIRECT_FABRIC_NAME,
					FI_TC_LOW_LATENCY,
					FI_TC_LOW_LATENCY));

	EXPECT_EQ(info->domain_attr->tclass, (uint32_t) FI_TC_LOW_LATENCY);
	EXPECT_EQ(info->tx_attr->tclass, (uint32_t) FI_TC_LOW_LATENCY);
}

struct EfaTclassQpCase {
	uint32_t domain_tclass;
	uint32_t ep_tclass;
	/* whether the resulting QP should ask for the low latency service level */
	bool expect_low_latency;
	const char *name;
};

/*
 * The service level the provider requests when creating the endpoint's QP is
 * the observable effect of the domain/endpoint tclass resolution. Read it off
 * efadv_qp_init_attr::sl at the efadv_create_qp_ex seam, whose only caller in
 * the provider is efa_qp_create.
 */
class EfaTclassQpTest : public TestWithParam<EfaTclassQpCase>
{
	protected:
	struct efa_resource resource = {};
	StrictMock<MockEfa> mock_efa;

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));

		/*
		 * Without efadv_qp_init_attr::sl the provider has no way to
		 * request a service level, so there is nothing to assert.
		 */
		if (!efa_test_have_efadv_sl())
			GTEST_SKIP() << "build lacks efadv_qp_init_attr::sl";
	}

	void TearDown() override
	{
		MockEfa::set(nullptr);
		efa_test_resource_destruct(&resource);
	}

	/* Service level of each QP creation attempt, in order. */
	std::vector<uint8_t> qp_sls;

	/**
	 * @brief Bring up an endpoint with the given tclass hints, recording the
	 * service level of every QP creation attempt into qp_sls.
	 *
	 * @param expect_low_latency	whether the resolved tclass should be
	 *				FI_TC_LOW_LATENCY. Only that case may
	 *				create the QP more than once, because
	 *				efa_qp_create retries with the default
	 *				service level if the device rejects the
	 *				low latency one.
	 */
	void enable_ep_recording_qp_sl(const char *fabric_name,
				       uint32_t domain_tclass,
				       uint32_t ep_tclass,
				       bool expect_low_latency)
	{
		struct fi_info *hints;
		int ret;

		hints = efa_test_alloc_default_hints(FI_EP_RDM, fabric_name);
		ASSERT_NE(hints, nullptr);
		hints->domain_attr->tclass = domain_tclass;
		hints->tx_attr->tclass = ep_tclass;

		/*
		 * Stop before fi_enable so the QP is not created until the mock
		 * that observes its service level is installed.
		 */
		ASSERT_NO_FATAL_FAILURE(
			efa_test_resource_construct_no_enable(&resource, hints));

		/* The hints must have survived getinfo into the domain and ep. */
		ASSERT_EQ(efa_test_get_domain_tclass(resource.domain),
			  domain_tclass);
		ASSERT_EQ(efa_test_get_base_ep_tclass(resource.ep), ep_tclass);

		MockEfa::set(&mock_efa);
		/*
		 * Armed to read efa_attr->sl in flight, then delegated to the
		 * real call so a usable QP is still created.
		 */
		EFA_EXPECT_CALL(mock_efa, efadv_create_qp_ex)
			.Times(AtLeast(1))
			.WillRepeatedly(
				Invoke([this](struct ibv_context *ibvctx,
					      struct ibv_qp_init_attr_ex *attr_ex,
					      struct efadv_qp_init_attr *efa_attr,
					      uint32_t inlen) {
					qp_sls.push_back(
						efa_test_efadv_attr_sl(efa_attr));
					return __real_efadv_create_qp_ex(
						ibvctx, attr_ex, efa_attr, inlen);
				}));

		ret = fi_enable(resource.ep);
		ASSERT_EQ(ret, 0) << "fi_enable failed: " << fi_strerror(-ret);

		/*
		 * Only a low latency request may be attempted twice, because
		 * efa_qp_create retries with the default service level if the
		 * device rejects the low latency one.
		 */
		ASSERT_GE(qp_sls.size(), 1u);
		ASSERT_LE(qp_sls.size(), expect_low_latency ? 2u : 1u);
	}

	/* The service level the provider asked for is the first attempt's. */
	void expect_requested_sl(bool low_latency)
	{
		EXPECT_EQ(qp_sls.front(), low_latency ?
						  efa_test_qp_low_latency_sl :
						  efa_test_qp_default_sl);

		/* A retry, if it happened, must have fallen back to default. */
		if (qp_sls.size() > 1)
			EXPECT_EQ(qp_sls.back(), efa_test_qp_default_sl);
	}
};

TEST_P(EfaTclassQpTest, resolves_qp_service_level)
{
	const EfaTclassQpCase &p = GetParam();

	ASSERT_NO_FATAL_FAILURE(enable_ep_recording_qp_sl(
		EFA_DIRECT_FABRIC_NAME, p.domain_tclass, p.ep_tclass,
		p.expect_low_latency));

	expect_requested_sl(p.expect_low_latency);
}

/*
 * efa-rdm reaches the same resolution through its own ep implementation, so
 * confirm the resolution still lands on the QP there.
 */
TEST_P(EfaTclassQpTest, resolves_qp_service_level_rdm)
{
	const EfaTclassQpCase &p = GetParam();

	ASSERT_NO_FATAL_FAILURE(enable_ep_recording_qp_sl(
		EFA_FABRIC_NAME, p.domain_tclass, p.ep_tclass,
		p.expect_low_latency));

	expect_requested_sl(p.expect_low_latency);
}

INSTANTIATE_TEST_SUITE_P(
	, EfaTclassQpTest,
	Values(
		EfaTclassQpCase{FI_TC_UNSPEC, FI_TC_UNSPEC, false,
				"both_unspec"},
		/* An unspecified ep tclass inherits the domain default. */
		EfaTclassQpCase{FI_TC_LOW_LATENCY, FI_TC_UNSPEC, true,
				"domain_low_latency_ep_unspec"},
		EfaTclassQpCase{FI_TC_UNSPEC, FI_TC_LOW_LATENCY, true,
				"domain_unspec_ep_low_latency"},
		EfaTclassQpCase{FI_TC_LOW_LATENCY, FI_TC_LOW_LATENCY, true,
				"both_low_latency"},
		EfaTclassQpCase{FI_TC_UNSPEC, FI_TC_BEST_EFFORT, false,
				"domain_unspec_ep_best_effort"},
		/*
		 * A tclass set on the endpoint overrides the domain default, so
		 * best effort on the ep must not inherit the domain's low
		 * latency.
		 */
		EfaTclassQpCase{FI_TC_LOW_LATENCY, FI_TC_BEST_EFFORT, false,
				"domain_low_latency_ep_best_effort"},
		EfaTclassQpCase{FI_TC_BEST_EFFORT, FI_TC_UNSPEC, false,
				"domain_best_effort_ep_unspec"}),
	[](const testing::TestParamInfo<EfaTclassQpCase> &i) {
		return std::string(i.param.name);
	});
