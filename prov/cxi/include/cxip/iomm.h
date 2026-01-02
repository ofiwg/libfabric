/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_IOMM_H_
#define _CXIP_IOMM_H_


#include <stdint.h>

/* Forward declarations */
struct cxip_domain;
struct cxip_md;

/* Function declarations */
int cxip_iomm_init(struct cxip_domain *dom);

void cxip_iomm_fini(struct cxip_domain *dom);

int cxip_map(struct cxip_domain *dom, const void *buf, unsigned long len,
	     uint64_t access, uint64_t flags, struct cxip_md **md);

void cxip_unmap(struct cxip_md *md);

#endif /* _CXIP_IOMM_H_ */
