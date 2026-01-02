/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_PORTALS_TABLE_H_
#define _CXIP_PORTALS_TABLE_H_


#include <stdint.h>
#include <stddef.h>

/* Forward declarations */
struct cxip_lni;

/* Type definitions */
struct cxip_portals_table {
	struct cxip_lni *lni;
	uint32_t pid;
	struct cxil_domain **doms;
	size_t doms_count;
};

/* Function declarations */
int cxip_portals_table_alloc(struct cxip_lni *lni, uint16_t *vni,
			     size_t vni_count, uint32_t pid,
			     struct cxip_portals_table **ptable);

void cxip_portals_table_free(struct cxip_portals_table *ptable);

#endif /* _CXIP_PORTALS_TABLE_H_ */
