/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_mocks.h"

static MockEfa *g_mock_efa = nullptr;

MockEfa *MockEfa::get()
{
	return g_mock_efa;
}

void MockEfa::set(MockEfa *instance)
{
	g_mock_efa = instance;
}

// Generate the definitions of the __wrap_* functions
#define EFA_MOCK_GEN_WRAP(ret, name, params, args)               \
	ret __wrap_##name params                                 \
	{                                                        \
		auto *mock = MockEfa::get();                     \
		if (mock)                                        \
			return mock->name args;                  \
		return __real_##name args;                       \
	}

extern "C" {
EFA_MOCK_FUNCTIONS(EFA_MOCK_GEN_WRAP)
}
