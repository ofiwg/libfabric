/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_thread_annotations.h"

OFI_TSA_LOCK_SYMBOL_DEFINE(efa_qp_table_lock_sym);
OFI_TSA_LOCK_SYMBOL_DEFINE(efa_implicit_av_lock_sym);
OFI_TSA_LOCK_SYMBOL_DEFINE(efa_util_ep_lock_sym);
OFI_TSA_LOCK_SYMBOL_DEFINE(efa_util_av_lock_sym);
OFI_TSA_LOCK_SYMBOL_DEFINE(efa_util_domain_lock_sym);
