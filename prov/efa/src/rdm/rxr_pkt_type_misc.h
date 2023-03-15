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

#ifndef _RXR_PKT_TYPE_MISC_H
#define _RXR_PKT_TYPE_MISC_H

#include "rxr.h"

/**
 * @brief Return a pointer to the optional host id header in a handshake packet
 *
 * @param[in]	pkt_entry	A packet entry
 * @return	If the input has the optional host id header, return the pointer to
 *	host id value; otherwise, return NULL
 */
static inline
uint64_t *rxr_pkt_handshake_host_id_ptr(struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_base_hdr *base_hdr = rxr_get_base_hdr(pkt_entry->wiredata);

	if (base_hdr->type != RXR_HANDSHAKE_PKT || !(base_hdr->flags & RXR_HANDSHAKE_HOST_ID_HDR))
		return NULL;

	return &(rxr_get_handshake_opt_host_id_hdr(pkt_entry->wiredata)->host_id);
}

#endif
