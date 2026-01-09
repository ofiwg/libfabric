/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_ATOMIC_H_
#define _CXIP_ATOMIC_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Forward declarations */
struct cxip_cntr;
struct cxip_txc;

/* Function declarations */
int cxip_amo_common(enum cxip_amo_req_type req_type, struct cxip_txc *txc,
		    uint32_t tclass, const struct fi_msg_atomic *msg,
		    const struct fi_ioc *comparev, void **comparedesc,
		    size_t compare_count, const struct fi_ioc *resultv,
		    void **resultdesc, size_t result_count, uint64_t flags,
		    bool triggered, uint64_t trig_thresh,
		    struct cxip_cntr *trig_cntr, struct cxip_cntr *comp_cntr);

int _cxip_atomic_opcode(enum cxip_amo_req_type req_type, enum fi_datatype dt,
			enum fi_op op, int amo_remap_to_pcie_fadd,
			enum c_atomic_op *cop, enum c_atomic_type *cdt,
			enum c_cswap_op *copswp, unsigned int *cdtlen);

#endif /* _CXIP_ATOMIC_H_ */
