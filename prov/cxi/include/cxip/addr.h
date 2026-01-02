/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_ADDR_H_
#define _CXIP_ADDR_H_

#include <stdint.h>

/* Macros */
#define CXIP_ADDR_EQUAL(a, b) ((a).nic == (b).nic && (a).pid == (b).pid)

#define CXIP_ADDR_VNI_EQUAL(a, b) (CXIP_ADDR_EQUAL(a, b) && (a).vni == (b).vni)

#define CXIP_ADDR_PORT_BITS 6

#define CXIP_ADDR_SWITCH_BITS 5

#define CXIP_ADDR_GROUP_BITS 9

#define CXIP_ADDR_FATTREE_PORT_BITS 6

#define CXIP_ADDR_FATTREE_SWITCH_BITS 14

/* Type definitions */
struct cxip_addr {
	uint32_t pid : C_DFA_PID_BITS_MAX;
	uint32_t nic : C_DFA_NIC_BITS;
	uint32_t pad : 3;
	uint16_t vni;
};

#endif /* _CXIP_ADDR_H_ */
