/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_abi.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include "efa_gtest_domain_utils.h"
#include "fi_ext_efa.h"
#include <gtest/gtest.h>

using testing::_;
using testing::Invoke;
using testing::Return;
using testing::StrictMock;
using testing::Test;
using testing::Truly;
using testing::Values;
using testing::WithParamInterface;

/* fi_efa.7: ep_attr->qkey must be below the privileged queue key range */
static const uint32_t kPrivilegedQkey = 0x80000000;
static const uint32_t kTestQkey = 0x0badf00d;

class EfaModifyEpTest : public Test
{
	protected:
	struct efa_resource resource = {};
	StrictMock<MockEfa> mock_efa;
	struct fi_efa_ops_modify_ep *modify_ep_ops = nullptr;

	void construct(enum fi_ep_type ep_type, const char *fabric_name,
		       bool enable)
	{
		struct fi_info *hints =
			efa_test_alloc_default_hints(ep_type, fabric_name);
		ASSERT_NE(hints, nullptr);

		if (enable)
			efa_test_resource_construct(&resource, hints);
		else
			efa_test_resource_construct_no_enable(&resource, hints);
		ASSERT_NE(resource.ep, nullptr);
	}

	void open_ops()
	{
		ASSERT_EQ(fi_open_ops(&resource.domain->fid,
				      FI_EFA_MODIFY_EP_OPS, 0,
				      (void **) &modify_ep_ops, nullptr),
			  0);
		ASSERT_NE(modify_ep_ops, nullptr);
		ASSERT_NE(modify_ep_ops->modify_ep, nullptr);
	}

	void construct_direct()
	{
		ASSERT_NO_FATAL_FAILURE(
			construct(FI_EP_RDM, EFA_DIRECT_FABRIC_NAME, true));
		ASSERT_NO_FATAL_FAILURE(open_ops());
		MockEfa::set(&mock_efa);
	}

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));
	}

	void TearDown() override
	{
		MockEfa::set(nullptr);
		efa_test_resource_destruct(&resource);
	}
};

TEST_F(EfaModifyEpTest, qkey_updates_qp_and_ep_addr)
{
	struct fi_efa_ep_attr ep_attr = {};
	uint32_t old_qkey, new_qkey, name_qkey = 0;

	ASSERT_NO_FATAL_FAILURE(construct_direct());

	old_qkey = efa_test_get_qp_qkey(resource.ep);
	new_qkey = (old_qkey + 1) & ~kPrivilegedQkey;
	ep_attr.qkey = new_qkey;

	auto sets_qkey = Truly([new_qkey](const struct ibv_qp_attr *attr) {
		return attr->qkey == new_qkey;
	});
	EFA_EXPECT_CALL(mock_efa, ibv_modify_qp, _, sets_qkey,
			(int) IBV_QP_QKEY)
		.WillOnce(Return(0));

	EXPECT_EQ(modify_ep_ops->modify_ep(resource.ep, &ep_attr,
					   FI_EFA_EP_ATTR_QKEY),
		  0);

	EXPECT_EQ(efa_test_get_qp_qkey(resource.ep), new_qkey);
	EXPECT_EQ(efa_test_getname_qkey(resource.ep, &name_qkey), 0);
	EXPECT_EQ(name_qkey, new_qkey);
}

TEST_F(EfaModifyEpTest, qkey_ibv_failure_leaves_qkey_unchanged)
{
	struct fi_efa_ep_attr ep_attr = {};
	uint32_t old_qkey, name_qkey = 0;

	ASSERT_NO_FATAL_FAILURE(construct_direct());

	old_qkey = efa_test_get_qp_qkey(resource.ep);
	ep_attr.qkey = (old_qkey + 1) & ~kPrivilegedQkey;

	EFA_EXPECT_CALL(mock_efa, ibv_modify_qp, _, _, _)
		.WillOnce(Return(EPERM));

	EXPECT_EQ(modify_ep_ops->modify_ep(resource.ep, &ep_attr,
					   FI_EFA_EP_ATTR_QKEY),
		  -FI_EPERM);

	EXPECT_EQ(efa_test_get_qp_qkey(resource.ep), old_qkey);
	EXPECT_EQ(efa_test_getname_qkey(resource.ep, &name_qkey), 0);
	EXPECT_EQ(name_qkey, old_qkey);
}

TEST_F(EfaModifyEpTest, invalid_args_rejected)
{
	struct fi_efa_ep_attr ep_attr = {};
	uint32_t old_qkey, name_qkey = 0;

	ASSERT_NO_FATAL_FAILURE(construct_direct());

	old_qkey = efa_test_get_qp_qkey(resource.ep);
	ep_attr.qkey = (old_qkey + 1) & ~kPrivilegedQkey;

	EFA_EXPECT_CALL(mock_efa, ibv_modify_qp).Times(0);

	EXPECT_EQ(modify_ep_ops->modify_ep(nullptr, &ep_attr,
					   FI_EFA_EP_ATTR_QKEY),
		  -FI_EINVAL);
	EXPECT_EQ(modify_ep_ops->modify_ep(resource.ep, nullptr,
					   FI_EFA_EP_ATTR_QKEY),
		  -FI_EINVAL);
	EXPECT_EQ(modify_ep_ops->modify_ep((struct fid_ep *) resource.domain,
					   &ep_attr, FI_EFA_EP_ATTR_QKEY),
		  -FI_EINVAL);

	EXPECT_EQ(efa_test_get_qp_qkey(resource.ep), old_qkey);
	EXPECT_EQ(efa_test_getname_qkey(resource.ep, &name_qkey), 0);
	EXPECT_EQ(name_qkey, old_qkey);
}

TEST_F(EfaModifyEpTest, qkey_rejected_before_ep_enabled)
{
	struct fi_efa_ep_attr ep_attr = {};

	ASSERT_NO_FATAL_FAILURE(
		construct(FI_EP_RDM, EFA_DIRECT_FABRIC_NAME, false));
	ASSERT_NO_FATAL_FAILURE(open_ops());
	MockEfa::set(&mock_efa);

	ep_attr.qkey = kTestQkey;

	EFA_EXPECT_CALL(mock_efa, ibv_modify_qp).Times(0);

	EXPECT_EQ(modify_ep_ops->modify_ep(resource.ep, &ep_attr,
					   FI_EFA_EP_ATTR_QKEY),
		  -FI_EINVAL);
}

TEST_F(EfaModifyEpTest, ops_rejected_for_rdm)
{
	ASSERT_NO_FATAL_FAILURE(construct(FI_EP_RDM, EFA_FABRIC_NAME, true));

	EXPECT_EQ(fi_open_ops(&resource.domain->fid, FI_EFA_MODIFY_EP_OPS, 0,
			      (void **) &modify_ep_ops, nullptr),
		  -FI_EOPNOTSUPP);
}

TEST_F(EfaModifyEpTest, ops_rejected_for_dgram)
{
	ASSERT_NO_FATAL_FAILURE(construct(FI_EP_DGRAM, EFA_FABRIC_NAME, true));

	EXPECT_EQ(fi_open_ops(&resource.domain->fid, FI_EFA_MODIFY_EP_OPS, 0,
			      (void **) &modify_ep_ops, nullptr),
		  -FI_EOPNOTSUPP);
}

struct ModifyEpNoDeviceCase {
	const char *name;
	uint32_t qkey;
	int attr_mask;
	int expected_ret;
};

class EfaModifyEpNoDeviceTest : public EfaModifyEpTest,
				public WithParamInterface<ModifyEpNoDeviceCase>
{
};

TEST_P(EfaModifyEpNoDeviceTest, leaves_qkey_unchanged)
{
	const ModifyEpNoDeviceCase &param = GetParam();
	struct fi_efa_ep_attr ep_attr = {};
	uint32_t old_qkey, name_qkey = 0;

	ASSERT_NO_FATAL_FAILURE(construct_direct());

	old_qkey = efa_test_get_qp_qkey(resource.ep);
	ep_attr.qkey = param.qkey;

	EFA_EXPECT_CALL(mock_efa, ibv_modify_qp).Times(0);

	EXPECT_EQ(modify_ep_ops->modify_ep(resource.ep, &ep_attr,
					   param.attr_mask),
		  param.expected_ret);

	EXPECT_EQ(efa_test_get_qp_qkey(resource.ep), old_qkey);
	EXPECT_EQ(efa_test_getname_qkey(resource.ep, &name_qkey), 0);
	EXPECT_EQ(name_qkey, old_qkey);
}

INSTANTIATE_TEST_SUITE_P(
	, EfaModifyEpNoDeviceTest,
	Values(ModifyEpNoDeviceCase{"privileged_qkey", kPrivilegedQkey,
				    FI_EFA_EP_ATTR_QKEY, -FI_EINVAL},
	       ModifyEpNoDeviceCase{"unsupported_flag", kTestQkey,
				    FI_EFA_EP_ATTR_QKEY << 1, -FI_EOPNOTSUPP},
	       ModifyEpNoDeviceCase{"empty_mask", kTestQkey, 0, 0}),
	[](const testing::TestParamInfo<ModifyEpNoDeviceCase> &info) {
		return std::string(info.param.name);
	});

TEST(EfaWqAttrAbiTest, size_is_the_shape_the_callers_version_published)
{
	EXPECT_EQ(efa_wq_attr_size(FI_VERSION(2, 0)),
		  sizeof(struct fi_efa_wq_attr_2_3));
	EXPECT_EQ(efa_wq_attr_size(FI_VERSION(2, 6)),
		  sizeof(struct fi_efa_wq_attr_2_3));
	EXPECT_EQ(efa_wq_attr_size(FI_VERSION(2, 7)),
		  sizeof(struct fi_efa_wq_attr));
	EXPECT_EQ(efa_wq_attr_size(FI_VERSION(3, 0)),
		  sizeof(struct fi_efa_wq_attr));
}

TEST(EfaWqAttrAbiTest, published_shape_is_a_prefix_of_the_current_struct)
{
	EXPECT_EQ(offsetof(struct fi_efa_wq_attr, buffer),
		  offsetof(struct fi_efa_wq_attr_2_3, buffer));
	EXPECT_EQ(offsetof(struct fi_efa_wq_attr, entry_size),
		  offsetof(struct fi_efa_wq_attr_2_3, entry_size));
	EXPECT_EQ(offsetof(struct fi_efa_wq_attr, num_entries),
		  offsetof(struct fi_efa_wq_attr_2_3, num_entries));
	EXPECT_EQ(offsetof(struct fi_efa_wq_attr, doorbell),
		  offsetof(struct fi_efa_wq_attr_2_3, doorbell));
	EXPECT_EQ(offsetof(struct fi_efa_wq_attr, max_batch),
		  offsetof(struct fi_efa_wq_attr_2_3, max_batch));

	EXPECT_LE(sizeof(struct fi_efa_wq_attr_2_3),
		  sizeof(struct fi_efa_wq_attr));
}

#if HAVE_EFADV_QUERY_QP_WQS

struct QueryQpWqsCase {
	const char *name;
	uint32_t api_version;
	bool defines_caps;
};

class EfaGdaQueryQpWqsTest : public Test,
			     public WithParamInterface<QueryQpWqsCase>
{
	protected:
	struct efa_resource resource = {};
	StrictMock<MockEfa> mock_efa;
	struct fi_efa_ops_gda *gda_ops = nullptr;

	void construct(uint32_t api_version)
	{
		struct fi_info *hints = efa_test_alloc_default_hints(
			FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
		ASSERT_NE(hints, nullptr);

		ASSERT_NO_FATAL_FAILURE(efa_test_resource_construct_api_version(
			&resource, hints, api_version));
		ASSERT_NE(resource.ep, nullptr);

		ASSERT_EQ(fi_open_ops(&resource.domain->fid, FI_EFA_GDA_OPS, 0,
				      (void **) &gda_ops, nullptr),
			  0);
		ASSERT_NE(gda_ops, nullptr);
		ASSERT_NE(gda_ops->query_qp_wqs, nullptr);

		MockEfa::set(&mock_efa);
	}

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));
	}

	void TearDown() override
	{
		MockEfa::set(nullptr);
		efa_test_resource_destruct(&resource);
	}
};

TEST_P(EfaGdaQueryQpWqsTest, caps_reported_only_to_a_caller_that_defines_it)
{
	const QueryQpWqsCase &param = GetParam();
	struct fi_efa_wq_attr sq_attr = {};
	struct fi_efa_wq_attr rq_attr = {};

	ASSERT_NO_FATAL_FAILURE(construct(param.api_version));

	EFA_EXPECT_CALL(mock_efa, efadv_query_qp_wqs, _, _, _, _)
		.WillOnce(Invoke(efa_test_mock_efadv_query_qp_wqs));

	EXPECT_EQ(gda_ops->query_qp_wqs(resource.ep, &sq_attr, &rq_attr), 0);

	EXPECT_EQ(sq_attr.caps,
		  param.defines_caps ? efa_test_mock_efadv_sq_caps() : 0);
	EXPECT_EQ(rq_attr.caps, 0);

	EXPECT_NE(sq_attr.buffer, nullptr);
	EXPECT_GT(sq_attr.entry_size, 0u);
	EXPECT_NE(rq_attr.buffer, nullptr);
	EXPECT_GT(rq_attr.entry_size, 0u);
}

INSTANTIATE_TEST_SUITE_P(
	, EfaGdaQueryQpWqsTest,
	Values(QueryQpWqsCase{"api_2_0", FI_VERSION(2, 0), false},
	       QueryQpWqsCase{"api_2_6", FI_VERSION(2, 6), false},
	       QueryQpWqsCase{"api_2_7", FI_VERSION(2, 7), true}),
	[](const testing::TestParamInfo<QueryQpWqsCase> &info) {
		return std::string(info.param.name);
	});

#endif /* HAVE_EFADV_QUERY_QP_WQS */
