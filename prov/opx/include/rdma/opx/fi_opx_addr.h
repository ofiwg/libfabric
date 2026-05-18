/*
 * Copyright (C) 2016 by Argonne National Laboratory.
 * Copyright (C) 2021,2024-2026 Cornelis Networks.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef _FI_PROV_OPX_ADDR_H_
#define _FI_PROV_OPX_ADDR_H_

#include <stdio.h>
#include <stdlib.h>
#include <arpa/inet.h> /* only for fi_opx_addr_dump ... */

#include "rdma/fabric.h" /* only for 'fi_addr_t' ... which is a typedef to uint64_t */

/* Index constants for the planes[] array in struct fi_opx_addr.
 * OPX_PRIMARY_PLANE is the default / first fabric plane. */
#define OPX_PRIMARY_PLANE 0

/* Get macro for subctxt/rx from combined field.
 * WFR and JKR: bits 10:8 is subctxt and 7:0 is rx id
 * CYR: bits 10:9 is subctxt and 8:0 is rx id */
#define OPX_HFI1_SUBCTXT(_subctxt_rx, _hw_hfi1_type)                                     \
	(OPX_IS_EXTENDED_RX_TYPE(_hw_hfi1_type) ? (0x600 & __be16_to_cpu(_subctxt_rx)) : \
						  (0x700 & __be16_to_cpu(_subctxt_rx)))

#define OPX_HFI1_RX(_subctxt_rx, _hw_hfi1_type)                                          \
	(OPX_IS_EXTENDED_RX_TYPE(_hw_hfi1_type) ? (0x1ff & __be16_to_cpu(_subctxt_rx)) : \
						  (0xff & __be16_to_cpu(_subctxt_rx)))

struct fi_opx_addr {
	struct {
		uint8_t	 hfi1_unit;
		uint8_t	 unused;
		uint16_t hfi1_subctxt_rx; /* (Stored big endian)
					   * WFR and JKR: bits 10:8 is subctxt and 7:0 is rx id
					   * CYR: bits 10:9 is subctxt and 8:0 is rx id */
		opx_lid_t lid;		  /* fabric-scoped node identifier */
		uint64_t  gid_hi;	  /* GID high 64 bits (identifies fabric plane) */
	} __attribute__((__packed__)) planes[OPX_MAX_TX_CONTEXTS];
	uint8_t			      tx_index;
	uint8_t			      unused2[7];
} __attribute__((__packed__));

struct fi_opx_extended_addr {
	struct fi_opx_addr addr;
	uint32_t	   rank;
	uint32_t	   rank_inst;
} __attribute__((__packed__));

extern struct fi_opx_addr opx_default_addr;

static inline void fi_opx_addr_dump(char *prefix, const struct fi_opx_addr *const addr)
{
	fprintf(stderr, "%s opx addr dump at %p\n", prefix, (void *) addr);
	fprintf(stderr, "%s   .tx_index ...................................... %u\n", prefix, addr->tx_index);
	for (int i = 0; i < OPX_MAX_TX_CONTEXTS; i++) {
		fprintf(stderr, "%s   .planes[%d].hfi1_unit ........................... %u\n", prefix, i,
			addr->planes[i].hfi1_unit);
		fprintf(stderr, "%s   .planes[%d].unused .............................. %u\n", prefix, i,
			addr->planes[i].unused);
		fprintf(stderr, "%s   .planes[%d].hfi1_subctxt_rx ..................... %u\n", prefix, i,
			addr->planes[i].hfi1_subctxt_rx);
		fprintf(stderr, "%s   .planes[%d].lid ................................. %d (le: %#x, be16: %#x)\n",
			prefix, i, addr->planes[i].lid, __cpu_to_le24(addr->planes[i].lid),
			__cpu24_to_be16(addr->planes[i].lid));
		fprintf(stderr, "%s   .planes[%d].gid_hi .............................. 0x%016lx\n", prefix, i,
			addr->planes[i].gid_hi);
	}
	fflush(stderr);
}

#define FI_OPX_ADDR_DUMP(addr)                                                      \
	({                                                                          \
		char prefix[1024];                                                  \
		snprintf(prefix, 1023, "%s:%s():%d", __FILE__, __func__, __LINE__); \
		fi_opx_addr_dump(prefix, (addr));                                   \
	})

#define FI_OPX_ADDR_TO_HFI1_LRH_DLID_9B(lid) ((uint64_t) __cpu24_to_be16(lid) << 16)

#endif /* _FI_PROV_OPX_ADDR_H_ */
