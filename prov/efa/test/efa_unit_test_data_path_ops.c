/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

/*
 * Unit test definitions of the EFA data path operations.
 *
 * efa_data_path_ops.h declares these extern under EFA_UNIT_TEST rather than
 * defining them static inline, so that `ld --wrap` can intercept them. This is
 * the single translation unit that supplies the definitions, and they are the
 * real implementations - the same text a production build compiles.
 *
 * An unmocked call therefore reaches the device. A test that must not touch the
 * device installs a mock that does nothing; both suites default the data path
 * ops to such mocks, so only a test that deliberately wants the real thing gets
 * it. See prov/efa/test/gtest/AGENTS.md.
 */

#include <errno.h>
#include <infiniband/verbs.h>
#include "efa_cq.h"
#include "efa_base_ep.h"
#include "efa_data_path_ops.h"
#include "efa_data_path_ops_body.h"
