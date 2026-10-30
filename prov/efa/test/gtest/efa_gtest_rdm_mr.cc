/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include "fi_ext_efa.h"
#include <cstring>
#include <functional>
#include <gtest/gtest.h>

using testing::AtLeast;
using testing::Invoke;
using testing::Return;
using testing::StrictMock;
using testing::Test;

class EfaRdmMrTest : public Test
{
	protected:
	struct efa_resource resource = {};
	StrictMock<MockEfa> mock_efa;

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));

		efa_test_resource_construct(
			&resource, efa_test_alloc_default_hints(
					   FI_EP_RDM, EFA_FABRIC_NAME));
		ASSERT_NE(resource.domain, nullptr);

		MockEfa::set(&mock_efa);
	}

	void TearDown() override
	{
		MockEfa::set(nullptr);
		efa_test_resource_destruct(&resource);
	}
};

/**
 * @brief A failed domain mr_map insert error must propagate out of registration.
 */
TEST_F(EfaRdmMrTest, reg_map_insert_failure_propagates_error)
{
	char buf[64];
	struct iovec iov = {};
	struct fi_mr_attr attr = {};
	struct fid_mr *mr = nullptr;
	struct fid_domain *saved_shm_domain;
	int ret;

	iov.iov_base = buf;
	iov.iov_len = sizeof(buf);
	attr.mr_iov = &iov;
	attr.iov_count = 1;
	attr.access = FI_SEND | FI_RECV;
	attr.iface = FI_HMEM_SYSTEM;

	EFA_EXPECT_CALL(mock_efa, ofi_mr_map_insert)
		.Times(1)
		.WillOnce(Return(-FI_ENOMEM));

	/* Detach shm to skip the irrelevant shm block in mr_regattr. */
	saved_shm_domain = efa_test_get_shm_domain(resource.domain);
	efa_test_set_shm_domain(resource.domain, nullptr);
	ret = fi_mr_regattr(resource.domain, &attr, 0, &mr);
	efa_test_set_shm_domain(resource.domain, saved_shm_domain);
	EXPECT_EQ(ret, -FI_ENOMEM);

	if (mr)
		fi_close(&mr->fid);
}


/*
 * Regression coverage for the FI_EFA_MR_RELAXED_ORDERING bit assignment.
 *
 * FI_EFA_MR_RELAXED_ORDERING (an EFA-specific hint stripped before the shm MR
 * registration) must not share a bit with OFI_HMEM_DATA_DEV_REG_HANDLE (the
 * gdrcopy device-registration handle flag stored in efa_rdm_mr->flags). If they
 * ever alias, deriving the shm flags would clear the gdrcopy handle bit, which
 * on the intra-node (shm) CUDA path silently drops the fast GDRCopy copy in
 * favor of the slower cudaMemcpy fallback.
 */

/**
 * @brief The two flags must occupy distinct bits. This is the root-cause guard.
 */
TEST_F(EfaRdmMrTest, relaxed_ordering_bit_distinct_from_gdrcopy_handle_bit)
{
	EXPECT_EQ(FI_EFA_MR_RELAXED_ORDERING &
			  efa_test_ofi_hmem_data_dev_reg_handle(),
		  0u)
		<< "FI_EFA_MR_RELAXED_ORDERING aliases the gdrcopy handle bit "
		   "OFI_HMEM_DATA_DEV_REG_HANDLE";
}

/**
 * @brief Deriving the shm MR flags for a CUDA region that has the gdrcopy
 * handle bit set must strip FI_EFA_MR_RELAXED_ORDERING while preserving the
 * gdrcopy handle bit (and adding FI_HMEM_DEVICE_ONLY for device memory).
 */
TEST_F(EfaRdmMrTest, shm_flags_preserve_gdrcopy_handle_when_relaxed_ordering_set)
{
	uint64_t gdrcopy_bit = efa_test_ofi_hmem_data_dev_reg_handle();
	uint64_t mr_flags = gdrcopy_bit | FI_EFA_MR_RELAXED_ORDERING;

	uint64_t shm_flags =
		efa_test_rdm_mr_shm_flags(mr_flags, FI_HMEM_CUDA);

	/* The gdrcopy handle bit survives into the shm flags. */
	EXPECT_NE(shm_flags & gdrcopy_bit, 0u)
		<< "shm flag derivation erased the gdrcopy handle bit";
	/* The EFA-specific relaxed-ordering hint is stripped. */
	EXPECT_EQ(shm_flags & FI_EFA_MR_RELAXED_ORDERING, 0u)
		<< "FI_EFA_MR_RELAXED_ORDERING leaked into shm flags";
	/* Device memory is marked device-only for shm. */
	EXPECT_NE(shm_flags & FI_HMEM_DEVICE_ONLY, 0u);
}

/**
 * @brief Without the gdrcopy handle bit, deriving shm flags must not
 * spuriously introduce it, and must still strip the relaxed-ordering hint.
 */
TEST_F(EfaRdmMrTest, shm_flags_no_gdrcopy_handle_when_bit_absent)
{
	uint64_t gdrcopy_bit = efa_test_ofi_hmem_data_dev_reg_handle();

	uint64_t shm_flags = efa_test_rdm_mr_shm_flags(
		FI_EFA_MR_RELAXED_ORDERING, FI_HMEM_CUDA);

	EXPECT_EQ(shm_flags & gdrcopy_bit, 0u);
	EXPECT_EQ(shm_flags & FI_EFA_MR_RELAXED_ORDERING, 0u);
}

/**
 * @brief For host (system) memory, shm flags must not set FI_HMEM_DEVICE_ONLY
 * and must still strip the relaxed-ordering hint.
 */
TEST_F(EfaRdmMrTest, shm_flags_host_memory_not_device_only)
{
	uint64_t shm_flags = efa_test_rdm_mr_shm_flags(
		FI_EFA_MR_RELAXED_ORDERING, FI_HMEM_SYSTEM);

	EXPECT_EQ(shm_flags & FI_HMEM_DEVICE_ONLY, 0u);
	EXPECT_EQ(shm_flags & FI_EFA_MR_RELAXED_ORDERING, 0u);
}

class EfaRdmMrNonP2pTest : public Test
{
	protected:
	struct efa_resource resource = {};
	StrictMock<MockEfa> mock_efa;
	uint64_t saved_caps = 0;
	bool saved_initialized = false;
	bool saved_p2p = false;
	struct fid_domain *saved_shm_domain = nullptr;
	char buf[64] = {};
	struct iovec iov = {};
	struct fi_mr_attr attr = {};

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));

		efa_test_resource_construct(
			&resource, efa_test_alloc_default_hints(
					   FI_EP_RDM, EFA_FABRIC_NAME));
		ASSERT_NE(resource.domain, nullptr);

		saved_caps = efa_test_get_info_domain_caps(resource.domain);
		efa_test_set_info_domain_caps(resource.domain,
					      saved_caps | FI_HMEM);
		saved_initialized =
			efa_test_set_hmem_initialized(FI_HMEM_ROCR, true);
		saved_p2p =
			efa_test_set_hmem_p2p_supported(FI_HMEM_ROCR, false);

		saved_shm_domain = efa_test_get_shm_domain(resource.domain);
		efa_test_set_shm_domain(resource.domain, nullptr);

		iov.iov_base = buf;
		iov.iov_len = sizeof(buf);
		attr.mr_iov = &iov;
		attr.iov_count = 1;
		attr.access = FI_SEND | FI_RECV;
		attr.iface = FI_HMEM_ROCR;

		MockEfa::set(&mock_efa);
	}

	void TearDown() override
	{
		MockEfa::set(nullptr);
		efa_test_set_shm_domain(resource.domain, saved_shm_domain);
		efa_test_set_hmem_p2p_supported(FI_HMEM_ROCR, saved_p2p);
		efa_test_set_hmem_initialized(FI_HMEM_ROCR, saved_initialized);
		efa_test_set_info_domain_caps(resource.domain, saved_caps);
		efa_test_resource_destruct(&resource);
	}

	static uint64_t key_of(uint64_t drawn)
	{
		return (drawn & ~(1ULL << 63)) | (1ULL << 32);
	}

	static std::function<int(struct efa_device *, void *, size_t)>
	draws(uint64_t drawn)
	{
		return [drawn](struct efa_device *, void *out, size_t len) {
			EXPECT_EQ(len, sizeof(drawn));
			memcpy(out, &drawn, sizeof(drawn));
			return 0;
		};
	}
};

TEST_F(EfaRdmMrNonP2pTest, key_is_drawn_from_random_source)
{
	struct fid_mr *mr = nullptr;
	uint64_t drawn = 0xfffffffefffffffeULL;

	EFA_EXPECT_CALL(mock_efa, efa_device_read_random)
		.Times(1)
		.WillOnce(Invoke(draws(drawn)));

	ASSERT_EQ(fi_mr_regattr(resource.domain, &attr, 0, &mr), 0);
	ASSERT_NE(mr, nullptr);

	EXPECT_EQ(fi_mr_key(mr), key_of(drawn));
	EXPECT_GT(fi_mr_key(mr), UINT32_MAX);
	EXPECT_NE(fi_mr_key(mr), FI_KEY_NOTAVAIL);

	EXPECT_EQ(fi_close(&mr->fid), 0);
}

TEST_F(EfaRdmMrNonP2pTest, key_already_in_map_is_redrawn)
{
	struct fid_mr *mr = nullptr, *mr2 = nullptr;
	uint64_t drawn = 0x1234, drawn2 = 0x5678;

	EFA_EXPECT_CALL(mock_efa, efa_device_read_random)
		.Times(3)
		.WillOnce(Invoke(draws(drawn)))
		.WillOnce(Invoke(draws(drawn)))
		.WillOnce(Invoke(draws(drawn2)));

	ASSERT_EQ(fi_mr_regattr(resource.domain, &attr, 0, &mr), 0);
	ASSERT_EQ(fi_mr_regattr(resource.domain, &attr, 0, &mr2), 0);

	EXPECT_EQ(fi_mr_key(mr), key_of(drawn));
	EXPECT_EQ(fi_mr_key(mr2), key_of(drawn2));

	EXPECT_EQ(fi_close(&mr2->fid), 0);
	EXPECT_EQ(fi_close(&mr->fid), 0);
}

TEST_F(EfaRdmMrNonP2pTest, key_is_reusable_after_mr_is_closed)
{
	struct fid_mr *mr = nullptr, *mr2 = nullptr;
	uint64_t drawn = 0x1234;

	EFA_EXPECT_CALL(mock_efa, efa_device_read_random)
		.Times(2)
		.WillRepeatedly(Invoke(draws(drawn)));

	ASSERT_EQ(fi_mr_regattr(resource.domain, &attr, 0, &mr), 0);
	EXPECT_EQ(fi_close(&mr->fid), 0);

	ASSERT_EQ(fi_mr_regattr(resource.domain, &attr, 0, &mr2), 0);
	EXPECT_EQ(fi_mr_key(mr2), key_of(drawn));

	EXPECT_EQ(fi_close(&mr2->fid), 0);
}

TEST_F(EfaRdmMrNonP2pTest, registration_fails_when_every_draw_is_in_map)
{
	struct fid_mr *mr = nullptr, *mr2 = nullptr;
	uint64_t drawn = 0x1234;

	EFA_EXPECT_CALL(mock_efa, efa_device_read_random)
		.Times(AtLeast(2))
		.WillRepeatedly(Invoke(draws(drawn)));

	ASSERT_EQ(fi_mr_regattr(resource.domain, &attr, 0, &mr), 0);

	EXPECT_EQ(fi_mr_regattr(resource.domain, &attr, 0, &mr2), -FI_ENOKEY);

	EXPECT_EQ(fi_close(&mr->fid), 0);
}

TEST_F(EfaRdmMrNonP2pTest, registration_fails_when_random_source_fails)
{
	struct fid_mr *mr = nullptr;

	EFA_EXPECT_CALL(mock_efa, efa_device_read_random)
		.Times(1)
		.WillOnce(Return(-FI_EIO));

	EXPECT_EQ(fi_mr_regattr(resource.domain, &attr, 0, &mr), -FI_EIO);
}
