/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef _EFA_RDM_UTIL_H
#define _EFA_RDM_UTIL_H

#include "efa.h"
#include "efa_rdm_protocol.h"
#include "efa_rdm_pke.h"

/*
 * The application-visible FI_MSG_PREFIX size is kept byte-identical to what
 * the EFA RDM provider historically advertised. It is intentionally pinned to
 * EFA_RDM_PKE_ALIGNMENT (the pke metadata's former in-pool size) rather than
 * sizeof(struct efa_rdm_pke): splitting the wiredata bounce buffer out of the
 * pke struct shrank the struct, but the advertised prefix must not change.
 */
#define EFA_RDM_MSG_PREFIX_SIZE (EFA_RDM_PKE_ALIGNMENT + sizeof(struct efa_rdm_eager_msgrtm_hdr) + EFA_RDM_REQ_OPT_RAW_ADDR_HDR_SIZE)

#if defined(static_assert)
static_assert(EFA_RDM_MSG_PREFIX_SIZE % 8 == 0, "message prefix size alignment check");
#endif


bool efa_rdm_get_use_device_rdma(uint32_t fabric_api_version);

void efa_rdm_get_desc_for_shm(int numdesc, void **efa_desc, void **shm_desc);

int efa_rdm_construct_msg_with_local_and_peer_information(struct efa_rdm_ep *ep, struct efa_rdm_peer *peer, char *msg, const char *base_msg, size_t msg_len);

int efa_rdm_write_error_msg(struct efa_rdm_ep *ep, struct efa_rdm_peer *peer, int prov_errno, char *err_msg, size_t *buflen);

#ifdef ENABLE_EFA_POISONING
static inline void efa_rdm_poison_mem_region(void *ptr, size_t size)
{
	uint32_t efa_rdm_poison_value = 0xdeadbeef;
	for (int i = 0; i < size / sizeof(efa_rdm_poison_value); i++)
		memcpy((uint32_t *)ptr + i, &efa_rdm_poison_value, sizeof(efa_rdm_poison_value));
}
#endif


#endif /* _EFA_RDM_UTIL_H */
