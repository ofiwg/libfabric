/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_av.h"
#include "rdm/efa_rdm_ep.h"
#include "efa_gtest_common_helpers.h"

void efa_test_fabricate_addr(struct fid_ep *ep, struct efa_ep_addr *addr)
{
	size_t addr_len = sizeof(*addr);
	static uint8_t gid_suffix = 1;

	memset(addr, 0, addr_len);
	// Get ep's actual address
	fi_getname(&ep->fid, addr, &addr_len);
	// Flip the first bit
	addr->raw[0] ^= 0xFF;
	addr->raw[15] = gid_suffix++;
	addr->qpn = 1;
	addr->qkey = 0x5678;
}

int efa_test_explicit_av_insert(struct fid_ep *ep, struct fid_av *av,
				fi_addr_t *addr)
{
	struct efa_ep_addr raw_addr;

	efa_test_fabricate_addr(ep, &raw_addr);
	return fi_av_insert(av, &raw_addr, 1, addr, 0, NULL);
}

fi_addr_t efa_test_insert_peer_new_gid(struct fid_ep *ep, struct fid_av *av)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_av *efa_av;
	struct efa_ep_addr raw_addr;
	fi_addr_t fi_addr = FI_ADDR_NOTAVAIL;
	int err;

	efa_rdm_ep =
		container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	efa_av = container_of(av, struct efa_av, util_av.av_fid);

	efa_test_fabricate_addr(ep, &raw_addr);

	ofi_genlock_lock(&efa_rdm_ep->base_ep.domain->srx_lock);
	err = efa_av_insert_one(efa_av, &raw_addr, &fi_addr, 0, NULL, true,
				true);
	ofi_genlock_unlock(&efa_rdm_ep->base_ep.domain->srx_lock);

	if (err)
		return FI_ADDR_NOTAVAIL;

	return fi_addr;
}
