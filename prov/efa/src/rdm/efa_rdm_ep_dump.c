/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

/*
 * EFA RDM endpoint state dumper.
 *
 * Triggered by a user-configured signal (FI_EFA_STATE_DUMP_SIGNAL),
 * dumps the current state of all EFA RDM endpoints to stderr at WARN
 * level. This is useful for diagnosing hangs where the application is
 * busy-polling with no forward progress.
 *
 * Usage:
 *   export FI_EFA_STATE_DUMP_SIGNAL=12   # SIGUSR2
 *   kill -12 <pid>
 */

#include "efa_rdm_ep.h"
#include "efa_rdm_peer.h"
#include "efa_rdm_ope.h"
#include "efa.h"
#include "efa_env.h"
#if ENABLE_DEBUG
#include "efa_rdm_pke.h"
#include "efa_rdm_pke_print.h"
#endif

#include <signal.h>

volatile sig_atomic_t g_efa_rdm_dump_requested = 0;

static void efa_rdm_dump_signal_handler(int sig)
{
	g_efa_rdm_dump_requested = 1;
}

/**
 * @brief Install signal handler for state dumping.
 * Called once during provider initialization.
 * Only installs the handler if FI_EFA_STATE_DUMP_SIGNAL is set to a nonzero value.
 */
void efa_rdm_dump_init(void)
{
	struct sigaction sa;

	if (!efa_env.state_dump_signal)
		return;

	sa.sa_handler = efa_rdm_dump_signal_handler;
	sa.sa_flags = 0;
	sigemptyset(&sa.sa_mask);
	sigaction(efa_env.state_dump_signal, &sa, NULL);
}

static void efa_rdm_dump_peer(struct efa_rdm_peer *peer)
{
	struct efa_rdm_ope *ope;
	struct dlist_entry *tmp;
#if ENABLE_DEBUG
	struct efa_rdm_pke *pke;
	struct dlist_entry *tmp2;
#endif
	int txe_cnt = 0, rxe_cnt = 0;
	int overflow_cnt = 0;
	struct efa_rdm_peer_overflow_pke_list_entry *overflow_entry;

	if (!peer->conn)
		return;

	/* Count txe/rxe */
	dlist_foreach_container_safe(&peer->txe_list,
				    struct efa_rdm_ope, ope, peer_entry, tmp)
		txe_cnt++;
	dlist_foreach_container_safe(&peer->rxe_list,
				    struct efa_rdm_ope, ope, peer_entry, tmp)
		rxe_cnt++;

	/* Count overflow list */
	dlist_foreach_container_safe(&peer->overflow_pke_list,
				    struct efa_rdm_peer_overflow_pke_list_entry,
				    overflow_entry, entry, tmp)
		overflow_cnt++;

	EFA_WARN(FI_LOG_EP_CTRL,
		 "  PEER %p fi_addr=%lu flags=0x%x "
		 "exp_msg_id=%u next_msg_id=%u "
		 "txe_cnt=%d rxe_cnt=%d overflow_cnt=%d "
		 "rnr_queued=%d backoff_wait=%lu us "
		 "outstanding_tx=%zu\n",
		 (void *)peer,
		 peer->conn->fi_addr,
		 peer->flags,
		 ofi_recvwin_next_exp_id((&peer->robuf)),
		 peer->next_msg_id,
		 txe_cnt, rxe_cnt, overflow_cnt,
		 peer->rnr_queued_pkt_cnt,
		 (unsigned long)peer->rnr_backoff_wait_time,
		 peer->efa_outstanding_tx_ops);

	/* Dump outstanding txe details */
	dlist_foreach_container_safe(&peer->txe_list,
				    struct efa_rdm_ope, ope, peer_entry, tmp) {
		EFA_WARN(FI_LOG_EP_CTRL,
			 "    TXE peer=%p fi_addr=%lu msg_id=%u op=%u total_len=%lu "
			 "bytes_sent=%lu window=%ld "
			 "internal_flags=0x%x%s%s%s%s\n",
			 (void *)peer,
			 peer->conn->fi_addr,
			 ope->msg_id, ope->op,
			 (unsigned long)ope->total_len,
			 (unsigned long)ope->bytes_sent,
			 (long)ope->window,
			 ope->internal_flags,
			 (ope->internal_flags & EFA_RDM_OPE_QUEUED_RNR) ? " [QUEUED_RNR]" : "",
			 (ope->internal_flags & EFA_RDM_OPE_QUEUED_CTRL) ? " [QUEUED_CTRL]" : "",
			 (ope->internal_flags & EFA_RDM_OPE_QUEUED_READ) ? " [QUEUED_READ]" : "",
			 (ope->internal_flags & EFA_RDM_OPE_QUEUED_BEFORE_HANDSHAKE) ? " [QUEUED_HANDSHAKE]" : "");

		#if ENABLE_DEBUG
		dlist_foreach_container_safe(&ope->queued_pkts,
					    struct efa_rdm_pke, pke, entry, tmp2) {
			efa_rdm_pke_print(pke, "      queued_pkt");
		}
		#endif
	}

	/* Dump outstanding rxe details */
	dlist_foreach_container_safe(&peer->rxe_list,
				    struct efa_rdm_ope, ope, peer_entry, tmp) {
		EFA_WARN(FI_LOG_EP_CTRL,
			 "    RXE peer=%p fi_addr=%lu msg_id=%u op=%u total_len=%lu "
			 "bytes_sent=%lu window=%ld "
			 "internal_flags=0x%x\n",
			 (void *)peer,
			 peer->conn->fi_addr,
			 ope->msg_id, ope->op,
			 (unsigned long)ope->total_len,
			 (unsigned long)ope->bytes_sent,
			 (long)ope->window,
			 ope->internal_flags);

		#if ENABLE_DEBUG
		dlist_foreach_container_safe(&ope->queued_pkts,
					    struct efa_rdm_pke, pke, entry, tmp2) {
			efa_rdm_pke_print(pke, "      queued_pkt");
		}
		#endif
	}
}

/**
 * @brief Dump the state of an EFA RDM endpoint.
 *
 * Prints endpoint-level counters and per-peer state for all
 * peers with outstanding operations.
 */
void efa_rdm_dump_ep_state(struct efa_rdm_ep *ep)
{
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *ope;
	struct dlist_entry *tmp;
	int ep_txe_cnt = 0, ep_rxe_cnt = 0;

	/* Count EP-level TXE/RXE totals */
	dlist_foreach_container_safe(&ep->txe_list,
				    struct efa_rdm_ope, ope, ep_entry, tmp)
		ep_txe_cnt++;
	dlist_foreach_container_safe(&ep->rxe_list,
				    struct efa_rdm_ope, ope, ep_entry, tmp)
		ep_rxe_cnt++;

	EFA_WARN(FI_LOG_EP_CTRL,
		 "=== EFA RDM EP STATE DUMP ===\n");

	EFA_WARN(FI_LOG_EP_CTRL,
		 "--- EP-level counters ---\n");
	EFA_WARN(FI_LOG_EP_CTRL,
		 "efa_outstanding_tx_ops=%zu "
		 "efa_rnr_queued_pkt_cnt=%zu "
		 "efa_rx_pkts_posted=%zu "
		 "efa_rx_pkts_held=%zu "
		 "efa_max_outstanding_tx_ops=%zu "
		 "efa_max_outstanding_rx_ops=%zu\n",
		 ep->efa_outstanding_tx_ops,
		 ep->efa_rnr_queued_pkt_cnt,
		 ep->efa_rx_pkts_posted,
		 ep->efa_rx_pkts_held,
		 ep->efa_max_outstanding_tx_ops,
		 ep->efa_max_outstanding_rx_ops);
	EFA_WARN(FI_LOG_EP_CTRL,
		 "queued_copy_num=%d "
		 "ope_queued_before_handshake_cnt=%zu "
		 "txe_cnt=%d rxe_cnt=%d\n",
		 ep->queued_copy_num,
		 ep->ope_queued_before_handshake_cnt,
		 ep_txe_cnt, ep_rxe_cnt);

	EFA_WARN(FI_LOG_EP_CTRL,
		 "--- Peer-level state ---\n");

	/* Iterate all peers */
	dlist_foreach_container_safe(&ep->ep_peer_list,
				    struct efa_rdm_peer, peer,
				    ep_peer_list_entry, tmp) {
		efa_rdm_dump_peer(peer);
	}

	EFA_WARN(FI_LOG_EP_CTRL,
		 "=== END EFA RDM EP STATE DUMP ===\n");
}

