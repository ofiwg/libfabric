/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_av.h"
#include "efa_env.h"
#include "rdm/efa_rdm_av.h"
#include "rdm/efa_rdm_ep.h"
#include "rdm/efa_rdm_peer.h"
#include "efa_gtest_common_helpers.h"
#include "efa_gtest_rdm_av_utils.h"

int efa_test_get_shm_av_size(void)
{
	return efa_env.shm_av_size;
}

int efa_test_set_shm_av_size(int value)
{
	int prev = efa_env.shm_av_size;

	efa_env.shm_av_size = value;
	return prev;
}

int efa_test_av_insert_local_peer(struct fid_ep *ep, struct fid_av *av,
				  uint32_t qpn, fi_addr_t *addr)
{
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(raw_addr);
	int ret;

	ret = fi_getname(&ep->fid, &raw_addr, &raw_addr_len);
	if (ret)
		return ret;
	raw_addr.qpn = qpn;
	raw_addr.qkey = 0x1234;

	return fi_av_insert(av, &raw_addr, 1, addr, 0, NULL);
}

bool efa_test_av_peer_uses_shm(struct fid_ep *ep, fi_addr_t addr)
{
	struct efa_rdm_ep *efa_rdm_ep =
		container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	return efa_rdm_ep_get_explicit_shm_fi_addr(efa_rdm_ep, addr) !=
	       FI_ADDR_NOTAVAIL;
}
