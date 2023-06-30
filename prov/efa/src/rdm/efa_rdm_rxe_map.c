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
#include "efa.h"
#include "efa_rdm_pke.h"
#include "efa_rdm_rxe_map.h"
#include "efa_rdm_pke_rtm.h"

/**
 * @brief find an RX entry for a received RTM packet entry's sender address and msg_id
 * 
 * @param[in]		rxe_map		RX entry map
 * @param[in]		pkt_entry	received packet entry
 * @returns
 * pointer to an RX entry. If such RX entry does not exist, return NULL
*/
struct efa_rdm_ope *efa_rdm_rxe_map_lookup(struct efa_rdm_rxe_map *rxe_map,
					   struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_rxe_map_entry *entry = NULL;
	struct efa_rdm_rxe_map_key key;

	memset(&key, 0, sizeof(key));
	key.msg_id = efa_rdm_pke_get_rtm_msg_id(pkt_entry);
	key.addr = pkt_entry->addr;
	HASH_FIND(hh, rxe_map->head, &key, sizeof(struct efa_rdm_rxe_map_key), entry);
	return entry ? entry->rxe : NULL;
}

/**
 * @brief insert an RX entry into an RX entry map
 *
 * @details
 * the insertion will use the combination of packet entry sender address and msg_id as key.
 * Caller is responsible to make sure the key does not exist in the map.
 * 
 * @param[in,out]	rxe_map		RX entry map
 * @param[in]		pkt_entry	received RTM packet
 * @param[in]		rxe		RX entry
*/
void efa_rdm_rxe_map_insert(struct efa_rdm_rxe_map *rxe_map,
			    struct efa_rdm_pke *pkt_entry,
			    struct efa_rdm_ope *rxe)
{
	struct efa_rdm_rxe_map_entry *entry;

	entry = ofi_buf_alloc(pkt_entry->ep->map_entry_pool);
	if (OFI_UNLIKELY(!entry)) {
		EFA_WARN(FI_LOG_CQ,
			"Map entries for medium size message exhausted.\n");
		efa_base_ep_write_eq_error(&pkt_entry->ep->base_ep, FI_ENOBUFS, FI_EFA_ERR_RXE_POOL_EXHAUSTED);
		return;
	}

	memset(&entry->key, 0, sizeof(entry->key));
	entry->key.msg_id = efa_rdm_pke_get_rtm_msg_id(pkt_entry);
	entry->key.addr = pkt_entry->addr;

#if ENABLE_DEBUG
	{
		struct efa_rdm_rxe_map_entry *existing_entry = NULL;

		HASH_FIND(hh, rxe_map->head, &entry->key, sizeof(struct efa_rdm_rxe_map_key), existing_entry);
		assert(!existing_entry);
	}
#endif

	entry->rxe = rxe;
	HASH_ADD(hh, rxe_map->head, key, sizeof(struct efa_rdm_rxe_map_key), entry);
}

/**
 * @brief remove an RX entry from the RX entry map
 * 
 * @details
 * the removal will use the combination of packet entry sender address and msg_id as key.
 * Caller is responsible to make sure the key does exist in the map.
 * 
 * @param[in,out]	rxe_map		RX entry map
 * @param[in]		pkt_entry	received RTM packet
 * @param[in]		rxe		RX entry
 */
void efa_rdm_rxe_map_remove(struct efa_rdm_rxe_map *rxe_map,
			    struct efa_rdm_pke *pkt_entry,
			    struct efa_rdm_ope *rxe)
{
	struct efa_rdm_rxe_map_entry *entry;
	struct efa_rdm_rxe_map_key key;

	memset(&key, 0, sizeof(key));
	key.msg_id = efa_rdm_pke_get_rtm_msg_id(pkt_entry);
	key.addr = pkt_entry->addr;

	HASH_FIND(hh, rxe_map->head, &key, sizeof(key), entry);
	assert(entry && entry->rxe == rxe);
	HASH_DEL(rxe_map->head, entry);
	ofi_buf_free(entry);
}