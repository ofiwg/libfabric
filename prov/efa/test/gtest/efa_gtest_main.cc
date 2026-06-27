/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

// Harmless suppression: persistent allocation from glibc outside of EFA provider
extern "C" const char *__lsan_default_suppressions(void)
{
	return "leak:_dlerror_run\n";
}

int main(int argc, char **argv)
{
	::testing::InitGoogleMock(&argc, argv);
	return RUN_ALL_TESTS();
}
