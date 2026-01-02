/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_NIC_H_
#define _CXIP_NIC_H_


/* Forward declarations */
struct cxip_if;

/* Function declarations */
int cxip_nic_alloc(struct cxip_if *nic_if, struct fid_nic **fid_nic);

#endif /* _CXIP_NIC_H_ */
