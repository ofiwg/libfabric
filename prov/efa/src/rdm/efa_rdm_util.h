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

#ifndef _EFA_RDM_UTIL_H
#define _EFA_RDM_UTIL_H

#include "efa.h"
#include "efa_rdm_protocol.h"
#include "efa_rdm_pke.h"

#define EFA_RDM_MSG_PREFIX_SIZE (sizeof(struct efa_rdm_pke) + sizeof(struct efa_rdm_eager_msgrtm_hdr) + EFA_RDM_REQ_OPT_RAW_ADDR_HDR_SIZE)

#if defined(static_assert) && defined(__x86_64__)
static_assert(EFA_RDM_MSG_PREFIX_SIZE % 8 == 0, "message prefix size alignment check");
#endif


bool efa_rdm_get_use_device_rdma(uint32_t fabric_api_version);

void efa_rdm_get_desc_for_shm(int numdesc, void **efa_desc, void **shm_desc);

int efa_rdm_write_error_msg(struct efa_rdm_ep *ep, fi_addr_t addr, int err, int prov_errno, void **buf, size_t *buflen);

#ifdef ENABLE_EFA_POISONING
static inline void efa_rdm_poison_mem_region(void *ptr, size_t size)
{
	uint32_t efa_rdm_poison_value = 0xdeadbeef;
	for (int i = 0; i < size / sizeof(efa_rdm_poison_value); i++)
		memcpy((uint32_t *)ptr + i, &efa_rdm_poison_value, sizeof(efa_rdm_poison_value));
}
#endif

#endif /* _EFA_RDM_UTIL_H */
