/*
 * Copyright (c) 2017 Intel Corporation, Inc.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "rxd_mock.h"
#include "gtest/gtest.h"

using namespace ::testing;

TEST(GetInfoTest, Negative) {
	FI_GetInfoMock mock;

	struct fi_info *hints = NULL, **fi = NULL;

	EXPECT_FUNCTION_CALL(mock,
			     (FI_VERSION(1, 5), NULL,
			      NULL, 0, hints, fi)).WillOnce(::testing::Return(0));
	ASSERT_EQ(0, fi_getinfo(FI_VERSION(1, 5), NULL,
			      NULL, 0, hints, fi));
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);

	fi_freeinfo(NULL);

	return RUN_ALL_TESTS();
}
