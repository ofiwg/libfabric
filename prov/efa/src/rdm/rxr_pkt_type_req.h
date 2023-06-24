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

#ifndef _RXR_PKT_TYPE_REQ_H
#define _RXR_PKT_TYPE_REQ_H

bool rxr_pkt_req_supported_by_peer(int req_type, struct efa_rdm_peer *peer);

void rxr_pkt_init_req_hdr(struct efa_rdm_pke *pkt_entry,
			  int pkt_type,
			  struct efa_rdm_ope *txe);

void *rxr_pkt_req_raw_addr(struct efa_rdm_pke *pkt_entry);

int64_t rxr_pkt_req_cq_data(struct efa_rdm_pke *pkt_entry);

uint32_t *rxr_pkt_req_connid_ptr(struct efa_rdm_pke *pkt_entry);

size_t rxr_pkt_req_hdr_size_from_pkt_entry(struct efa_rdm_pke *pkt_entry);

size_t rxr_pkt_req_base_hdr_size(struct efa_rdm_pke *pkt_entry);

size_t rxr_pkt_req_data_size(struct efa_rdm_pke *pkt_entry);

size_t rxr_pkt_req_hdr_size(int pkt_type, uint16_t flags, size_t rma_iov_count);

uint32_t rxr_pkt_hdr_rma_iov_count(struct efa_rdm_pke *pkt_entry);

size_t rxr_pkt_req_max_hdr_size(int pkt_type);

size_t rxr_pkt_max_hdr_size(void);

size_t rxr_pkt_req_data_offset(struct efa_rdm_pke *pkt_entry);
#endif
