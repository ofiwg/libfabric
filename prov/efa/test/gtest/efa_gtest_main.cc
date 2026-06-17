/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

/*
 * Suppress LeakSanitizer reports for rdma-core's load-time driver
 * registration (verbs_register_driver) and device discovery cache, which
 * are intentionally never freed. ibv_close_device() properly frees all
 * per-context state; only these one-time framework allocations persist.
 *
 * The cmocka suite doesn't need this because its --wrap=calloc (for OOM
 * fault injection) accidentally hides rdma-core allocations from LSan.
 * We don't wrap calloc here to preserve leak detection for genuine leaks.
 */
#if __has_include(<sanitizer/lsan_interface.h>)
#include <sanitizer/lsan_interface.h>

extern "C" const char *__lsan_default_suppressions(void)
{
	return "leak:libefa.so\n";
}
#endif

int main(int argc, char **argv)
{
	::testing::InitGoogleMock(&argc, argv);
	return RUN_ALL_TESTS();
}
