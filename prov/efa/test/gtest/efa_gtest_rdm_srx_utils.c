/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_rdm_srx_utils.h"
#include "efa.h"
#include "rdm/efa_rdm_ep.h"
#include "rdm/efa_rdm_srx.h"

int efa_test_srx_directed_match_honors_post_order(
	struct fid_ep *ep_fid, struct fid_av *av, int tagged,
	struct efa_test_srx_directed_match_result *out)
{
	struct efa_rdm_ep *ep = container_of(
		ep_fid, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	struct util_srx_ctx *srx = efa_rdm_ep_get_peer_srx_ctx(ep);
	struct fid_peer_srx *peer_srx = util_get_peer_srx(ep->peer_srx_ep);
	struct fi_peer_match_attr attr = {0};
	struct fi_peer_rx_entry *peer_rxe;
	struct fi_context2 ctx[2];
	struct iovec iov = {0};
	void *desc = NULL;
	fi_addr_t peer_addr;
	uint64_t tag = 0x1234;
	int i, ret;

	memset(out, 0, sizeof *out);
	if (efa_test_av_insert_self(ep_fid, av, &peer_addr) != 1)
		return -FI_EINVAL;
	out->dir_recv = srx->dir_recv;

	for (i = 0; i < 2; i++) {
		fi_addr_t addr = i ? peer_addr : FI_ADDR_UNSPEC;

		ret = tagged ? util_srx_generic_trecv(ep->peer_srx_ep, &iov,
						      &desc, 1, addr, &ctx[i],
						      tag, 0, 0)
			     : util_srx_generic_recv(ep->peer_srx_ep, &iov,
						     &desc, 1, addr, &ctx[i],
						     0);
		if (ret)
			return ret;
	}

	attr.addr = peer_addr;
	attr.tag = tag;
	ofi_genlock_lock(srx->lock);
	for (i = 0; i < 2; i++) {
		ret = tagged ? peer_srx->owner_ops->get_tag(peer_srx, &attr,
							    &peer_rxe)
			     : peer_srx->owner_ops->get_msg(peer_srx, &attr,
							    &peer_rxe);
		if (ret)
			break;
		out->matched_context[i] = peer_rxe->context;
		if (peer_rxe->context != &ctx[i]) {
			ret = -FI_EOTHER;
			break;
		}
		peer_srx->owner_ops->free_entry(peer_rxe);
	}
	if (!ret) {
		out->queues_empty =
			tagged ? slist_empty(&srx->tag_queue) &&
					 slist_empty(ofi_array_at(
						 &srx->src_trecv_queues,
						 peer_addr))
			       : slist_empty(&srx->msg_queue) &&
					 slist_empty(ofi_array_at(
						 &srx->src_recv_queues,
						 peer_addr));
	}
	ofi_genlock_unlock(srx->lock);
	out->posted_context[0] = &ctx[0];
	out->posted_context[1] = &ctx[1];
	return ret == -FI_EOTHER ? 0 : ret;
}
