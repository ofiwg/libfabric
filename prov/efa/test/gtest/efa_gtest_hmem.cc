/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include <gtest/gtest.h>
#include <rdma/fi_errno.h>
#include <sys/uio.h>
#include <cstdint>
#include <cstring>

extern "C" {
ssize_t efa_copy_to_hmem_iov(void **desc, struct iovec *hmem_iov,
			     size_t iov_count, char *buff, size_t buff_size);
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
