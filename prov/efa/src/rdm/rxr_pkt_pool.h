/*
 * Copyright (c) 2019-2022 Amazon.com, Inc. or its affiliates.
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

#ifndef _RXR_PKT_POOL_H
#define _RXR_PKT_POOL_H

#include <stddef.h>
#include "rxr_pkt_entry.h"

/* Forward declaration to avoid circular dependency */
struct rxr_ep;

struct rxr_pkt_pool {
	struct ofi_bufpool *entry_pool;
	struct ofi_bufpool *sendv_pool;
	struct efa_send_wr *efa_send_wr_pool;
};

int rxr_pkt_pool_create(struct rxr_ep *ep,
			enum rxr_pkt_entry_alloc_type pkt_pool_type,
			size_t chunk_cnt, size_t max_cnt,
			struct rxr_pkt_pool **pkt_pool);

int rxr_pkt_pool_grow(struct rxr_pkt_pool *rxr_pkt_pool);

void rxr_pkt_pool_destroy(struct rxr_pkt_pool *pkt_pool);

#endif
