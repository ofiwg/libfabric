/*
 * Copyright (c) Amazon.com, Inc. or its affiliates.
 * All rights reserved.
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

#ifndef EFA_RDM_OPE_RECVMAP
#define EFA_RDM_OPE_RECVMAP

#include <stdint.h>
#include <rdma/fi_endpoint.h>
#include "uthash.h"


struct efa_rdm_rxe_map_key {
	uint64_t msg_id;
	fi_addr_t addr;
};


struct efa_rdm_rxe_map_entry;

/**
 * @brief a hashmap between sender address + msg_id to RX entry
 * 
 * @details
 * This hash map is used on the receiver side to implement
 * medium and runting protocols. Such protocol will send
 * multiple RTM packets at the same time. The first RTM
 * will be matched with an RX entry and will be inserted
 * to this map, the later arriving RTM packet will use
 * this hashmap to find the RX entry.
 */
struct efa_rdm_rxe_map {
	struct efa_rdm_rxe_map_entry *head;
};

struct efa_rdm_ope;

struct efa_rdm_rxe_map_entry {
	struct efa_rdm_rxe_map_key key;
	struct efa_rdm_ope *rxe;
	UT_hash_handle hh;
};

static inline
void efa_rdm_rxe_map_construct(struct efa_rdm_rxe_map *rxe_map)
{
	rxe_map->head = NULL;
}

struct efa_rdm_pke;

struct efa_rdm_ope *efa_rdm_rxe_map_lookup(struct efa_rdm_rxe_map *rxe_map,
					   struct efa_rdm_pke *pkt_entry);

void efa_rdm_rxe_map_insert(struct efa_rdm_rxe_map *rxe_map,
			    struct efa_rdm_pke *pkt_entry,
			    struct efa_rdm_ope *rxe);

void efa_rdm_rxe_map_remove(struct efa_rdm_rxe_map *rxe_map,
			   struct efa_rdm_pke *pkt_entry,
			   struct efa_rdm_ope *rxe);
#endif