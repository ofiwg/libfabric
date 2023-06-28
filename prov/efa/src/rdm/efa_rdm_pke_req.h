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

#ifndef _EFA_RDM_PKE_REQ_H
#define _EFA_RDM_PKE_REQ_H

#include <stdint.h>
#include <stdlib.h>

struct efa_rdm_ope;

struct efa_rdm_pke;

void efa_rdm_pke_init_req_hdr_common(struct efa_rdm_pke *pkt_entry,
			      int pkt_type,
			      struct efa_rdm_ope *txe);

void *efa_rdm_pke_get_req_raw_addr(struct efa_rdm_pke *pkt_entry);

int64_t efa_rdm_pke_get_req_cq_data(struct efa_rdm_pke *pkt_entry);

uint32_t *efa_rdm_pke_get_req_connid_ptr(struct efa_rdm_pke *pkt_entry);

size_t efa_rdm_pke_get_req_base_hdr_size(struct efa_rdm_pke *pkt_entry);

size_t efa_rdm_pke_get_req_hdr_size(struct efa_rdm_pke *pkt_entry);

#endif
