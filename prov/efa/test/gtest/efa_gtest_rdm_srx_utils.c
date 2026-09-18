/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_rdm_srx_utils.h"
#include "efa.h"
#include "efa_env.h"
#include "rdm/efa_rdm_ep.h"
#include "rdm/efa_rdm_msg.h"
#include "rdm/efa_rdm_pke_rtm.h"
#include "rdm/efa_rdm_srx.h"
#include "rdm/protocols/efa_rdm_proto_eager.h"

static int efa_test_srx_callback_invocations;

static ssize_t efa_test_srx_callback(struct efa_rdm_pke *pke)
{
	(void) pke;
	efa_test_srx_callback_invocations++;
	return 0;
}

static void efa_test_srx_construct_eager(struct efa_rdm_pke *pke)
{
	struct efa_rdm_eager_msgrtm_hdr hdr = {0};
	struct efa_rdm_req_opt_connid_hdr connid = {0};

	hdr.hdr.version = EFA_RDM_PROTOCOL_VERSION;
	hdr.hdr.type = EFA_RDM_EAGER_MSGRTM_PKT;
	hdr.hdr.flags = EFA_RDM_PKT_CONNID_HDR | EFA_RDM_REQ_MSG;
	connid.connid = 0x1234;
	memcpy(pke->wiredata, &hdr, sizeof hdr);
	memcpy(pke->wiredata + sizeof hdr, &connid, sizeof connid);
	pke->pkt_size = sizeof hdr + sizeof connid;
}

int efa_test_srx_dispatches_receive_callback(
	struct fid_ep *ep_fid, struct fid_av *av, int unexpected,
	struct efa_test_srx_dispatch_result *out)
{
	struct efa_rdm_ep *ep = container_of(
		ep_fid, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	struct util_srx_ctx *srx = efa_rdm_ep_get_peer_srx_ctx(ep);
	struct efa_rdm_proto *proto = &efa_rdm_proto_eager;
	efa_rdm_pke_callback saved_callback = proto->handle_unexp_pke_match;
	struct efa_rdm_ope *rxe;
	struct iovec iov = {0};
	void *desc = NULL;
	fi_addr_t peer_addr;
	int saved_rx_copy_unexp = efa_env.rx_copy_unexp;
	int ret;

	memset(out, 0, sizeof *out);
	if (efa_test_av_insert_self(ep_fid, av, &peer_addr) != 1)
		return -FI_EINVAL;
	struct efa_rdm_peer *peer =
		efa_rdm_ep_get_peer_explicit(ep, peer_addr);

	if (!peer)
		return -FI_EINVAL;

	if (!unexpected) {
		ret = util_srx_generic_recv(ep->peer_srx_ep, &iov, &desc, 1, 0,
					    NULL, 0);
		if (ret)
			return ret;
	}

	struct efa_rdm_pke *pke =
		efa_rdm_pke_alloc(ep, ep->efa_rx_pkt_pool,
				  EFA_RDM_PKE_FROM_EFA_RX_POOL);

	if (!pke)
		return -FI_ENOMEM;
	ep->efa_rx_pkts_posted = efa_base_ep_get_rx_pool_size(&ep->base_ep);
	pke->peer = peer;
	efa_test_srx_construct_eager(pke);
	efa_test_srx_callback_invocations = 0;

	ofi_genlock_lock(srx->lock);
	if (unexpected) {
		efa_env.rx_copy_unexp = 0;
		ret = efa_rdm_pke_proc_msgrtm(pke);
		rxe = pke->ope;
		out->was_unexpected = rxe && rxe->state == EFA_RDM_RXE_UNEXP;
		out->callback_set = pke->handle_pke != NULL;
		pke->handle_pke = efa_test_srx_callback;
	} else {
		proto->handle_unexp_pke_match = efa_test_srx_callback;
		ret = efa_rdm_pke_proc_msgrtm(pke);
		proto->handle_unexp_pke_match = saved_callback;
		rxe = pke->ope;
	}
	ofi_genlock_unlock(srx->lock);
	efa_env.rx_copy_unexp = saved_rx_copy_unexp;
	if (ret)
		return ret;

	if (unexpected) {
		ret = util_srx_generic_recv(ep->peer_srx_ep, &iov, &desc, 1, 0,
					    NULL, 0);
		if (ret)
			return ret;
	}

	out->callback_invocations = efa_test_srx_callback_invocations;
	out->matched = rxe->state == EFA_RDM_RXE_MATCHED;
	out->unexpected_packet_cleared = rxe->unexp_pkt == NULL;

	ofi_genlock_lock(srx->lock);
	efa_rdm_pke_release_rx(pke);
	efa_rdm_rxe_release(rxe);
	ofi_genlock_unlock(srx->lock);
	return 0;
}
