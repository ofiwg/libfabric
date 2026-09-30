/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_ABI_H
#define EFA_ABI_H

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <rdma/fabric.h>

#include "fi_ext_efa.h"

/*
 * Frozen copies of the shapes the EFA provider's public structs have published,
 * and the rules for deciding how much of one a caller actually owns.
 *
 * Adding a member to a public struct means adding a shape here and extending
 * the matching efa_*_size(); the asserts fail if the member was inserted
 * rather than appended.
 */

/*
 * struct fi_efa_wq_attr, filled by query_qp_wqs in FI_EFA_GDA_OPS.
 *
 * 2.3 published the struct.
 */
struct fi_efa_wq_attr_2_3 {
	uint8_t *buffer;
	uint32_t entry_size;
	uint32_t num_entries;
	uint32_t *doorbell;
	uint32_t max_batch;
};

/* 2.7 added caps. */

static_assert(offsetof(struct fi_efa_wq_attr, max_batch) ==
		      offsetof(struct fi_efa_wq_attr_2_3, max_batch),
	      "struct fi_efa_wq_attr diverged from its 2.3 shape");
static_assert(sizeof(struct fi_efa_wq_attr_2_3) <=
		      sizeof(struct fi_efa_wq_attr),
	      "struct fi_efa_wq_attr shapes must not shrink");

/**
 * @brief Size of the struct fi_efa_wq_attr a caller of this version allocated
 *
 * @param api_version version the caller negotiated at fi_getinfo
 * @return size of the shape that version published
 */
static inline size_t efa_wq_attr_size(uint32_t api_version)
{
	if (FI_VERSION_GE(api_version, FI_VERSION(2, 7)))
		return sizeof(struct fi_efa_wq_attr);

	return sizeof(struct fi_efa_wq_attr_2_3);
}

#endif /* EFA_ABI_H */
