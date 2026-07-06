/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include "efa_gtest_domain_utils.h"
#include "fi_ext_efa.h"
#include <gtest/gtest.h>

using testing::_;
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
