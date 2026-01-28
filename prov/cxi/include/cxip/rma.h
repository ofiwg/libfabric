/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_RMA_H_
#define _CXIP_RMA_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Forward declarations */
struct cxip_cntr;
struct cxip_txc;

/* Function declarations */
ssize_t cxip_rma_common(enum fi_op_type op, struct cxip_txc *txc,
			const void *buf, size_t len, void *desc,
			fi_addr_t tgt_addr, uint64_t addr, uint64_t key,
			uint64_t data, uint64_t flags, uint32_t tclass,
			uint64_t msg_order, void *context, bool triggered,
			uint64_t trig_thresh, struct cxip_cntr *trig_cntr,
			struct cxip_cntr *comp_cntr);

#endif /* _CXIP_RMA_H_ */
