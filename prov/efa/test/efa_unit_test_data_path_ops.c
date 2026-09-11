/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

/*
 * efa_data_path_ops.h gives these functions external linkage under
 * EFA_UNIT_TEST rather than defining them static inline, so that they can be
 * mocked. Only one translation unit may define them by defining the macro
 * EFA_DATA_PATH_OPS_EMIT_BODIES. This TU defines EFA_DATA_PATH_OPS_EMIT_BODIES
 *
 * The macro _has_ to be defined before the first include that could reach
 * efa_data_path_ops.h. Otherwise, linking will fail with undefined references
 * to efa_qp_post_*.
 */

#define EFA_DATA_PATH_OPS_EMIT_BODIES

#include <errno.h>
#include <infiniband/verbs.h>
#include "efa_cq.h"
#include "efa_base_ep.h"
#include "efa_data_path_ops.h"
