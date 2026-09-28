/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_gtest_common_mocks.h"

#include <gtest/gtest.h>
#include <rdma/fi_errno.h>
#include <sys/uio.h>
#include <cstdint>
#include <cstring>

extern "C" {
ssize_t efa_copy_to_hmem_iov(void **desc, struct iovec *hmem_iov,
			     size_t iov_count, char *buff, size_t buff_size);
int efa_hmem_set_sync_memops(void *ptr, uint64_t device);
}

class EfaHmemTest : public testing::Test
{
};

/**
 * @brief Assert that efa_copy_to_hmem_iov copies correctly when
 * iov_count > 1
 */
TEST_F(EfaHmemTest, scatter_to_multi_iov_advances_source_cursor)
{
	uint8_t src[16];
	uint8_t r0[8], r1[8];
	memset(src, 0xAA, 8);
	memset(src + 8, 0xBB, 8);
	memset(r0, 0xCC, sizeof(r0));
	memset(r1, 0xCC, sizeof(r1));

	struct iovec hmem_iov[2] = {{r0, sizeof(r0)}, {r1, sizeof(r1)}};
	/* null desc goes to FI_HMEM_SYSTEM, so uses plain memcpy */
	void *desc[2] = {nullptr, nullptr};

	ssize_t ret = efa_copy_to_hmem_iov(desc, hmem_iov, 2, (char *)src,
					   sizeof(src));

	EXPECT_EQ(ret, (ssize_t)sizeof(src));
	for (size_t i = 0; i < sizeof(r0); i++)
		EXPECT_EQ(r0[i], 0xAA);
	for (size_t i = 0; i < sizeof(r1); i++)
		EXPECT_EQ(r1[i], 0xBB);
}

/**
 * @brief Same as above, but with differently-sized iov entries
 */
TEST_F(EfaHmemTest, scatter_to_uneven_iov_advances_by_copied_size)
{
	uint8_t src[16];
	uint8_t r0[4], r1[12];
	memset(src, 0xAA, 4);
	memset(src + 4, 0xBB, 12);
	memset(r0, 0xCC, sizeof(r0));
	memset(r1, 0xCC, sizeof(r1));

	struct iovec hmem_iov[2] = {{r0, sizeof(r0)}, {r1, sizeof(r1)}};
	void *desc[2] = {nullptr, nullptr};

	ssize_t ret = efa_copy_to_hmem_iov(desc, hmem_iov, 2, (char *)src,
					   sizeof(src));

	EXPECT_EQ(ret, (ssize_t)sizeof(src));
	for (size_t i = 0; i < sizeof(r0); i++)
		EXPECT_EQ(r0[i], 0xAA);
	for (size_t i = 0; i < sizeof(r1); i++)
		EXPECT_EQ(r1[i], 0xBB);
}

/** 
 * @brief Assert that buff_size is resepcted and the destination
 * hmem_iov beyond buff_size isn't written to
 */
TEST_F(EfaHmemTest, scatter_clamps_last_copy_to_remaining_bytes)
{
	uint8_t src[12];
	uint8_t r0[8], r1[8];
	memset(src, 0xAA, 8);
	memset(src + 8, 0xBB, 4);
	memset(r0, 0xCC, sizeof(r0));
	memset(r1, 0xCC, sizeof(r1));

	struct iovec hmem_iov[2] = {{r0, sizeof(r0)}, {r1, sizeof(r1)}};
	void *desc[2] = {nullptr, nullptr};

	ssize_t ret = efa_copy_to_hmem_iov(desc, hmem_iov, 2, (char *)src,
					   sizeof(src));

	EXPECT_EQ(ret, (ssize_t)sizeof(src));
	for (size_t i = 0; i < sizeof(r0); i++)
		EXPECT_EQ(r0[i], 0xAA);
	for (size_t i = 0; i < 4; i++)
		EXPECT_EQ(r1[i], 0xBB);
	for (size_t i = 4; i < sizeof(r1); i++)
		EXPECT_EQ(r1[i], 0xCC);
}

/**
 * @brief Assert that -FI_ETRUNC is returned if buff_size is larger
 * than what the iov can accomodate
 */
TEST_F(EfaHmemTest, scatter_source_larger_than_iov_returns_etrunc)
{
	uint8_t src[16];
	uint8_t r0[4], r1[4];
	memset(src, 0xAA, sizeof(src));
	memset(r0, 0xCC, sizeof(r0));
	memset(r1, 0xCC, sizeof(r1));

	struct iovec hmem_iov[2] = {{r0, sizeof(r0)}, {r1, sizeof(r1)}};
	void *desc[2] = {nullptr, nullptr};

	ssize_t ret = efa_copy_to_hmem_iov(desc, hmem_iov, 2, (char *)src,
					   sizeof(src));

	EXPECT_EQ(ret, -FI_ETRUNC);
}

#if HAVE_CUDA && HAVE_CUDA_CTX_SYNC_MEMOPS
class EfaHmemSyncMemopsTest : public testing::Test
{
	protected:
	testing::StrictMock<MockEfa> mock_efa;

	void SetUp() override
	{
		MockEfa::set(&mock_efa);
	}

	void TearDown() override
	{
		MockEfa::set(nullptr);
	}
};

class EfaHmemPointerSyncMemopsTest :
	public EfaHmemSyncMemopsTest,
	public testing::WithParamInterface<int>
{
};

TEST_P(EfaHmemPointerSyncMemopsTest, returns_pointer_sync_result)
{
	uint8_t buffer;
	void *ptr = &buffer;
	CUdevice device = 1;
	int expected_result = GetParam();

	EFA_EXPECT_CALL(mock_efa, cuda_set_sync_memops, ptr)
		.WillOnce(testing::Return(expected_result));
	EFA_EXPECT_CALL(mock_efa, ofi_cuDevicePrimaryCtxGetState,
			testing::_, testing::_, testing::_)
		.Times(0);
	EFA_EXPECT_CALL(mock_efa, ofi_cuDevicePrimaryCtxSetFlags,
			testing::_, testing::_)
		.Times(0);

	EXPECT_EQ(efa_hmem_set_sync_memops(ptr, device), expected_result);
}

INSTANTIATE_TEST_SUITE_P(
	, EfaHmemPointerSyncMemopsTest,
	testing::Values(FI_SUCCESS, -FI_EINVAL),
	[](const testing::TestParamInfo<int> &info) {
		return info.param == FI_SUCCESS ? "success" : "failure";
	});

TEST_F(EfaHmemSyncMemopsTest, falls_back_to_primary_context)
{
	uint8_t buffer;
	void *ptr = &buffer;
	CUdevice device = 1;
	unsigned int ctx_flags = CU_CTX_SCHED_YIELD;

	EFA_EXPECT_CALL(mock_efa, cuda_set_sync_memops, ptr)
		.WillOnce(testing::Return(-FI_EOPNOTSUPP));
	EFA_EXPECT_CALL(mock_efa, ofi_cuDevicePrimaryCtxGetState, device,
			testing::_, testing::_)
		.WillOnce(testing::DoAll(
			testing::SetArgPointee<1>(ctx_flags),
			testing::SetArgPointee<2>(1),
			testing::Return(CUDA_SUCCESS)));
	EFA_EXPECT_CALL(mock_efa, ofi_cuDevicePrimaryCtxSetFlags, device,
			ctx_flags | CU_CTX_SYNC_MEMOPS)
		.WillOnce(testing::Return(CUDA_SUCCESS));

	EXPECT_EQ(efa_hmem_set_sync_memops(ptr, device), FI_SUCCESS);
}

TEST_F(EfaHmemSyncMemopsTest, keeps_enabled_primary_context)
{
	uint8_t buffer;
	void *ptr = &buffer;
	CUdevice device = 2;
	unsigned int ctx_flags = CU_CTX_SCHED_YIELD | CU_CTX_SYNC_MEMOPS;

	EFA_EXPECT_CALL(mock_efa, cuda_set_sync_memops, ptr)
		.WillOnce(testing::Return(-FI_EOPNOTSUPP));
	EFA_EXPECT_CALL(mock_efa, ofi_cuDevicePrimaryCtxGetState, device,
			testing::_, testing::_)
		.WillOnce(testing::DoAll(
			testing::SetArgPointee<1>(ctx_flags),
			testing::SetArgPointee<2>(1),
			testing::Return(CUDA_SUCCESS)));
	EFA_EXPECT_CALL(mock_efa, ofi_cuDevicePrimaryCtxSetFlags,
			testing::_, testing::_)
		.Times(0);

	EXPECT_EQ(efa_hmem_set_sync_memops(ptr, device), FI_SUCCESS);
}

TEST_F(EfaHmemSyncMemopsTest, returns_einval_when_getting_context_state_fails)
{
	uint8_t buffer;
	void *ptr = &buffer;
	CUdevice device = 3;

	EFA_EXPECT_CALL(mock_efa, cuda_set_sync_memops, ptr)
		.WillOnce(testing::Return(-FI_EOPNOTSUPP));
	EFA_EXPECT_CALL(mock_efa, ofi_cuDevicePrimaryCtxGetState, device,
			testing::_, testing::_)
		.WillOnce(testing::Return(CUDA_ERROR_INVALID_DEVICE));
	EFA_EXPECT_CALL(mock_efa, ofi_cuDevicePrimaryCtxSetFlags,
			testing::_, testing::_)
		.Times(0);

	EXPECT_EQ(efa_hmem_set_sync_memops(ptr, device), -FI_EINVAL);
}

TEST_F(EfaHmemSyncMemopsTest, returns_einval_when_setting_context_flags_fails)
{
	uint8_t buffer;
	void *ptr = &buffer;
	CUdevice device = 4;
	unsigned int ctx_flags = 0;

	EFA_EXPECT_CALL(mock_efa, cuda_set_sync_memops, ptr)
		.WillOnce(testing::Return(-FI_EOPNOTSUPP));
	EFA_EXPECT_CALL(mock_efa, ofi_cuDevicePrimaryCtxGetState, device,
			testing::_, testing::_)
		.WillOnce(testing::DoAll(
			testing::SetArgPointee<1>(ctx_flags),
			testing::SetArgPointee<2>(1),
			testing::Return(CUDA_SUCCESS)));
	EFA_EXPECT_CALL(mock_efa, ofi_cuDevicePrimaryCtxSetFlags, device,
			CU_CTX_SYNC_MEMOPS)
		.WillOnce(testing::Return(CUDA_ERROR_NOT_SUPPORTED));

	EXPECT_EQ(efa_hmem_set_sync_memops(ptr, device), -FI_EINVAL);
}
#endif /* HAVE_CUDA && HAVE_CUDA_CTX_SYNC_MEMOPS */
