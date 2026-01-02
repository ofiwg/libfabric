/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_REQ_BUF_H_
#define _CXIP_REQ_BUF_H_

/* Forward declarations */
struct cxip_rxc_hpc;
struct cxip_ux_send;

/* Macros */
#define CXIP_REQ_BUF_SIZE (12 * 1024 * 1024)

#define CXIP_REQ_BUF_MIN_POSTED 6

#define CXIP_REQ_BUF_MAX_CACHED 0

#define CXIP_REQ_BUF_HEADER_MAX_SIZE \
	(sizeof(struct c_port_fab_hdr) + sizeof(struct c_port_unrestricted_hdr))

#define CXIP_REQ_BUF_HEADER_MIN_SIZE \
	(sizeof(struct c_port_fab_hdr) + sizeof(struct c_port_small_msg_hdr))

/* Function declarations */
int cxip_req_bufpool_init(struct cxip_rxc_hpc *rxc);

void cxip_req_bufpool_fini(struct cxip_rxc_hpc *rxc);

void cxip_req_buf_ux_free(struct cxip_ux_send *ux);

#endif /* _CXIP_REQ_BUF_H_ */
