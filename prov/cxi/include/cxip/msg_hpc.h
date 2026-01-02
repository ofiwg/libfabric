/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_MSG_HPC_H_
#define _CXIP_MSG_HPC_H_

#include <stddef.h>

/* Forward declarations */
struct cxip_ep;
struct cxip_rxc_hpc;

/* Function declarations */
int cxip_oflow_bufpool_init(struct cxip_rxc_hpc *rxc);

void cxip_oflow_bufpool_fini(struct cxip_rxc_hpc *rxc);

int cxip_build_ux_entry_info(struct cxip_ep *ep,
			     struct fi_cq_tagged_entry *entry, size_t count,
			     fi_addr_t *src_addr, size_t *ux_count);

int cxip_unexp_start(struct fi_peer_rx_entry *entry);

#endif /* _CXIP_MSG_HPC_H_ */
