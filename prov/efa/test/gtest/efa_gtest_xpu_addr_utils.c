/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_av.h"
#include "efa_gtest_xpu_addr_utils.h"

int efa_test_xpu_av_insert_peer(struct fid_ep *ep, struct fid_av *av,
				uint16_t qpn, uint32_t qkey, fi_addr_t *addr)
{
	struct efa_ep_addr raw_addr = { 0 };
	size_t raw_addr_len = sizeof(raw_addr);
	int ret;

	ret = fi_getname(&ep->fid, &raw_addr, &raw_addr_len);
	if (ret)
		return ret;

	raw_addr.qpn = qpn;
	raw_addr.qkey = qkey;

	return fi_av_insert(av, &raw_addr, 1, addr, 0, NULL);
}

int efa_test_xpu_av_ahn(struct fid_av *av_fid, fi_addr_t fi_addr)
{
	struct efa_av *av = container_of(av_fid, struct efa_av, util_av.av_fid);
	struct efa_av_entry *av_entry;

	av_entry = (struct efa_av_entry *) ofi_av_get_addr(&av->util_av,
							  fi_addr);
	if (!av_entry || !av_entry->ah)
		return -1;

	return (int) av_entry->ah->ahn;
}

uint32_t efa_test_xpu_mr_lkey(struct fid_mr *mr)
{
	struct efa_mr *efa_mr =
		container_of(&mr->fid, struct efa_mr, mr_fid.fid);

	return efa_mr->lkey;
}
