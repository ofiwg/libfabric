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

#ifndef EFA_RDM_PKE_RTW_H
#define EFA_RDM_PKE_RTW_H

#include "efa_rdm_pke.h"
#include "efa_rdm_protocol.h"

static inline
struct efa_rdm_rtw_base_hdr *efa_rdm_pke_get_rtw_base_hdr(struct efa_rdm_pke *pkt_entry)
{
	return (struct efa_rdm_rtw_base_hdr *)pkt_entry->wiredata;
}

static inline
struct efa_rdm_dc_eager_rtw_hdr *efa_rdm_pke_dc_eager_rtw_hdr(struct efa_rdm_pke *pkt_entry)
{
	return (struct efa_rdm_dc_eager_rtw_hdr *)pkt_entry->wiredata;
}

ssize_t efa_rdm_pke_init_eager_rtw(struct efa_rdm_pke *pkt_entry,
				   struct efa_rdm_ope *txe);

void efa_rdm_pke_handle_eager_rtw_send_completion(struct efa_rdm_pke *pkt_entry);

void efa_rdm_pke_handle_eager_rtw_recv(struct efa_rdm_pke *pkt_entry);

ssize_t efa_rdm_pke_init_dc_eager_rtw(struct efa_rdm_pke *pkt_entry,
				      struct efa_rdm_ope *txe);

void efa_rdm_pke_handle_dc_eager_rtw_recv(struct efa_rdm_pke *pkt_entry);

ssize_t efa_rdm_pke_init_longcts_rtw(struct efa_rdm_pke *pkt_entry,
				     struct efa_rdm_ope *txe);

void efa_rdm_pke_handle_longcts_rtw_sent(struct efa_rdm_pke *pkt_entry);

void efa_rdm_pke_handle_longcts_rtw_send_completion(struct efa_rdm_pke *pkt_entry);

ssize_t efa_rdm_pke_init_dc_longcts_rtw(struct efa_rdm_pke *pkt_entry,
					struct efa_rdm_ope *txe);

void efa_rdm_pke_handle_longcts_rtw_recv(struct efa_rdm_pke *pkt_entry);

ssize_t efa_rdm_pke_init_longread_rtw(struct efa_rdm_pke *pkt_entry,
				      struct efa_rdm_ope *txe);

void efa_rdm_pke_handle_longread_rtw_recv(struct efa_rdm_pke *pkt_entry);

#endif
