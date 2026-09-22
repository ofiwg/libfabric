/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_RDM_PKE_RTW_H
#define EFA_RDM_PKE_RTW_H

#include "efa_rdm_pke.h"
#include "efa_rdm_protocol.h"

static inline
struct efa_rdm_rtw_base_hdr *efa_rdm_pke_get_rtw_base_hdr(struct efa_rdm_pke *pkt_entry)
{
	return (struct efa_rdm_rtw_base_hdr *)pkt_entry->wiredata;
}

struct efa_rdm_ope *efa_rdm_pke_alloc_rtw_rxe(struct efa_rdm_pke *pkt_entry);

#endif
