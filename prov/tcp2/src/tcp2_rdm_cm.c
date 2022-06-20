/*
 * Copyright (c) 2016-2022 Intel Corporation, Inc.  All rights reserved.
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

#include <stdlib.h>
#include <string.h>
#include <poll.h>

#include <ofi.h>
#include <ofi_util.h>
#include "tcp2.h"


/* Include cm_msg with connect() data.  If the connection is accepted,
 * return the version.  The returned version must be <= the requested
 * version, and is used by the active side to fallback to an older
 * protocol version.
 */
struct tcp2_rdm_cm {
	uint8_t version;
	uint8_t resv;
	uint16_t port;
	uint32_t pid;
};

static int tcp2_match_event(struct slist_entry *item, const void *arg)
{
	struct tcp2_event *event;
	event = container_of(item, struct tcp2_event, list_entry);
	return event->cm_entry.fid == arg;
}

static void tcp2_close_conn(struct tcp2_conn *conn)
{
	struct tcp2_event *event;
	struct slist_entry *item;

	FI_DBG(&tcp2_prov, FI_LOG_EP_CTRL, "closing conn %p\n", conn);
	assert(tcp2_progress_locked(tcp2_rdm2_progress(conn->rdm)));
	dlist_remove_init(&conn->loopback_entry);

	if (conn->ep) {
		fi_close(&conn->ep->util_ep.ep_fid.fid);
		do {
			item = slist_remove_first_match(&conn->rdm->event_list,
				tcp2_match_event, &conn->ep->util_ep.ep_fid.fid);
			if (!item)
				break;

			tcp2_rdm2_progress(conn->rdm)->event_cnt--;
			event = container_of(item, struct tcp2_event, list_entry);
			free(event);
		} while (item);
	}

	conn->ep = NULL;
}

/* MSG EPs under an RDM EP do not write events to the EQ. */
static int tcp2_bind_conn(struct tcp2_rdm *rdm, struct tcp2_ep *ep)
{
	int ret;

	assert(tcp2_progress_locked(tcp2_rdm2_progress(rdm)));

	ret = fi_ep_bind(&ep->util_ep.ep_fid, &rdm->srx->rx_fid.fid, 0);
	if (ret)
		return ret;

	ret = fi_ep_bind(&ep->util_ep.ep_fid,
			 &rdm->util_ep.rx_cq->cq_fid.fid, FI_RECV);
	if (ret)
		return ret;

	ret = fi_ep_bind(&ep->util_ep.ep_fid,
			 &rdm->util_ep.tx_cq->cq_fid.fid, FI_SEND);
	if (ret)
		return ret;

	if (rdm->util_ep.rx_cntr) {
		ret = fi_ep_bind(&ep->util_ep.ep_fid,
				 &rdm->util_ep.rx_cntr->cntr_fid.fid, FI_RECV);
		if (ret)
			return ret;
	}

	if (rdm->util_ep.tx_cntr) {
		ret = fi_ep_bind(&ep->util_ep.ep_fid,
				 &rdm->util_ep.tx_cntr->cntr_fid.fid, FI_SEND);
		if (ret)
			return ret;
	}

	if (rdm->util_ep.rd_cntr) {
		ret = fi_ep_bind(&ep->util_ep.ep_fid,
				 &rdm->util_ep.rd_cntr->cntr_fid.fid, FI_READ);
		if (ret)
			return ret;
	}

	if (rdm->util_ep.wr_cntr) {
		ret = fi_ep_bind(&ep->util_ep.ep_fid,
				 &rdm->util_ep.wr_cntr->cntr_fid.fid, FI_WRITE);
		if (ret)
			return ret;
	}

	if (rdm->util_ep.rem_rd_cntr) {
		ret = fi_ep_bind(&ep->util_ep.ep_fid,
				 &rdm->util_ep.rem_rd_cntr->cntr_fid.fid,
				 FI_REMOTE_READ);
		if (ret)
			return ret;
	}

	if (rdm->util_ep.rem_wr_cntr) {
		ret = fi_ep_bind(&ep->util_ep.ep_fid,
				 &rdm->util_ep.rem_wr_cntr->cntr_fid.fid,
				 FI_REMOTE_WRITE);
		if (ret)
			return ret;
	}

	return 0;
}

static int tcp2_open_conn(struct tcp2_conn *conn, struct fi_info *info)
{
	struct fid_ep *ep_fid;
	int ret;

	assert(tcp2_progress_locked(tcp2_rdm2_progress(conn->rdm)));
	ret = fi_endpoint(&conn->rdm->util_ep.domain->domain_fid, info,
			  &ep_fid, conn);
	if (ret) {
		TCP2_WARN_ERR(FI_LOG_EP_CTRL, "fi_endpoint", ret);
		return ret;
	}

	conn->ep = container_of(ep_fid, struct tcp2_ep, util_ep.ep_fid);
	ret = tcp2_bind_conn(conn->rdm, conn->ep);
	if (ret)
		goto err;

	ret = fi_enable(&conn->ep->util_ep.ep_fid);
	if (ret) {
		TCP2_WARN_ERR(FI_LOG_EP_CTRL, "fi_enable", ret);
		goto err;
	}

	return 0;

err:
	fi_close(&conn->ep->util_ep.ep_fid.fid);
	conn->ep = NULL;
	return ret;
}

static int tcp2_rdm_connect(struct tcp2_conn *conn)
{
	struct tcp2_rdm_cm msg;
	struct fi_info *info;
	int ret;

	FI_DBG(&tcp2_prov, FI_LOG_EP_CTRL, "connecting %p\n", conn);
	assert(tcp2_progress_locked(tcp2_rdm2_progress(conn->rdm)));

	info = conn->rdm->pep->info;
	info->dest_addrlen = info->src_addrlen;

	free(info->dest_addr);
	info->dest_addr = mem_dup(&conn->peer->addr, info->dest_addrlen);
	if (!info->dest_addr)
		return -FI_ENOMEM;

	ret = tcp2_open_conn(conn, info);
	if (ret)
		return ret;

	msg.version = TCP2_RDM_VERSION;
	msg.pid = htonl((uint32_t) getpid());
	msg.resv = 0;
	msg.port = htons(ofi_addr_get_port(info->src_addr));

	ret = fi_connect(&conn->ep->util_ep.ep_fid, info->dest_addr,
			 &msg, sizeof msg);
	if (ret) {
		TCP2_WARN_ERR(FI_LOG_EP_CTRL, "fi_connect", ret);
		goto err;
	}
	return 0;

err:
	tcp2_close_conn(conn);
	return ret;
}

static void tcp2_free_conn(struct tcp2_conn *conn)
{
	struct rxm_av *av;

	FI_DBG(&tcp2_prov, FI_LOG_EP_CTRL, "free conn %p\n", conn);
	assert(tcp2_progress_locked(tcp2_rdm2_progress(conn->rdm)));

	if (conn->flags & TCP2_CONN_INDEXED)
		ofi_idm_clear(&conn->rdm->conn_idx_map, conn->peer->index);

	util_put_peer(conn->peer);
	av = container_of(conn->rdm->util_ep.av, struct rxm_av, util_av);
	rxm_av_free_conn(av, conn);
}

void tcp2_freeall_conns(struct tcp2_rdm *rdm)
{
	struct tcp2_conn *conn;
	struct dlist_entry *tmp;
	struct rxm_av *av;
	int i, cnt;

	av = container_of(rdm->util_ep.av, struct rxm_av, util_av);
	assert(tcp2_progress_locked(tcp2_rdm2_progress(rdm)));

	/* We can't have more connections than the current number of
	 * possible peers.
	 */
	cnt = (int) rxm_av_max_peers(av);
	for (i = 0; i < cnt; i++) {
		conn = ofi_idm_lookup(&rdm->conn_idx_map, i);
		if (!conn)
			continue;

		tcp2_close_conn(conn);
		tcp2_free_conn(conn);
	}

	dlist_foreach_container_safe(&rdm->loopback_list, struct tcp2_conn,
				     conn, loopback_entry, tmp) {
		tcp2_close_conn(conn);
		tcp2_free_conn(conn);
	}
}

static struct tcp2_conn *
tcp2_alloc_conn(struct tcp2_rdm *rdm, struct util_peer_addr *peer)
{
	struct tcp2_conn *conn;
	struct rxm_av *av;

	assert(tcp2_progress_locked(tcp2_rdm2_progress(rdm)));
	av = container_of(rdm->util_ep.av, struct rxm_av, util_av);
	conn = rxm_av_alloc_conn(av);
	if (!conn) {
		TCP2_WARN_ERR(FI_LOG_EP_CTRL, "rxm_av_alloc_conn", -FI_ENOMEM);
		return NULL;
	}

	conn->rdm = rdm;
	conn->flags = 0;
	dlist_init(&conn->loopback_entry);

	conn->peer = peer;
	rxm_ref_peer(peer);

	FI_DBG(&tcp2_prov, FI_LOG_EP_CTRL, "allocated conn %p\n", conn);
	return conn;
}

static struct tcp2_conn *
tcp2_add_conn(struct tcp2_rdm *rdm, struct util_peer_addr *peer)
{
	struct tcp2_conn *conn;

	assert(tcp2_progress_locked(tcp2_rdm2_progress(rdm)));
	conn = ofi_idm_lookup(&rdm->conn_idx_map, peer->index);
	if (conn)
		return conn;

	conn = tcp2_alloc_conn(rdm, peer);
	if (!conn)
		return NULL;

	if (ofi_idm_set(&rdm->conn_idx_map, peer->index, conn) < 0) {
		tcp2_free_conn(conn);
		TCP2_WARN_ERR(FI_LOG_EP_CTRL, "ofi_idm_set", -FI_ENOMEM);
		return NULL;
	}

	conn->flags |= TCP2_CONN_INDEXED;
	return conn;
}

/* The returned conn is only valid if the function returns success.
 * This is called from data transfer ops, which return ssize_t, so
 * we return that rather than int.
 */
ssize_t tcp2_get_conn(struct tcp2_rdm *rdm, fi_addr_t addr,
		      struct tcp2_conn **conn)
{
	struct util_peer_addr **peer;

	assert(tcp2_progress_locked(tcp2_rdm2_progress(rdm)));
	peer = ofi_av_addr_context(rdm->util_ep.av, addr);
	*conn = tcp2_add_conn(rdm, *peer);
	if (!*conn)
		return -FI_ENOMEM;

	if (!(*conn)->ep)
		return tcp2_rdm_connect(*conn);
	else if((*conn)->ep->state != TCP2_CONNECTED)
		return -FI_EAGAIN;

	return 0;
}

void tcp2_process_connect(struct fi_eq_cm_entry *cm_entry)
{
	struct tcp2_rdm_cm *msg;
	struct tcp2_conn *conn;

	assert(cm_entry->fid->fclass == TCP2_CLASS_CM);
	conn = cm_entry->fid->context;

	assert(tcp2_progress_locked(tcp2_rdm2_progress(conn->rdm)));
	msg = (struct tcp2_rdm_cm *) cm_entry->data;
	conn->remote_pid = ntohl(msg->pid);
}

static void tcp2_process_connreq(struct fi_eq_cm_entry *cm_entry)
{
	struct tcp2_rdm *rdm;
	struct tcp2_rdm_cm *msg;
	union ofi_sock_ip peer_addr;
	struct util_peer_addr *peer;
	struct tcp2_conn *conn;
	struct rxm_av *av;
	int ret, cmp;

	assert(cm_entry->fid->fclass == FI_CLASS_PEP);
	rdm = cm_entry->fid->context;
	assert(tcp2_progress_locked(tcp2_rdm2_progress(rdm)));
	msg = (struct tcp2_rdm_cm *) cm_entry->data;

	memcpy(&peer_addr, cm_entry->info->dest_addr,
	       cm_entry->info->dest_addrlen);
	ofi_addr_set_port(&peer_addr.sa, ntohs(msg->port));

	av = container_of(rdm->util_ep.av, struct rxm_av, util_av);
	peer = util_get_peer(av, &peer_addr);
	if (!peer) {
		TCP2_WARN_ERR(FI_LOG_EP_CTRL, "util_get_peer", -FI_ENOMEM);
		goto reject;
	}

	conn = tcp2_add_conn(rdm, peer);
	if (!conn)
		goto put;

	FI_INFO(&tcp2_prov, FI_LOG_EP_CTRL, "connreq for %p\n", conn);
	if (!conn->ep)
		goto accept;

	switch (conn->ep->state) {
	case TCP2_CONNECTING:
	case TCP2_REQ_SENT:
		/* simultaneous connections */
		cmp = ofi_addr_cmp(&tcp2_prov, &peer_addr.sa, &rdm->addr.sa);
		if (cmp < 0) {
			/* let our request finish */
			FI_INFO(&tcp2_prov, FI_LOG_EP_CTRL,
				"simultaneous, reject peer %p\n", conn);
			goto put;
		} else if (cmp > 0) {
			/* accept peer's request */
			FI_INFO(&tcp2_prov, FI_LOG_EP_CTRL,
				"simultaneous, accept peer %p\n", conn);
			tcp2_close_conn(conn);
		} else {
			/* connecting to ourself, create loopback conn */
			FI_INFO(&tcp2_prov, FI_LOG_EP_CTRL,
				"loopback conn %p\n", conn);
			conn = tcp2_alloc_conn(rdm, peer);
			if (!conn)
				goto put;

			dlist_insert_tail(&conn->loopback_entry,
					  &rdm->loopback_list);
		}
		break;
	case TCP2_ACCEPTING:
	case TCP2_CONNECTED:
		if (conn->remote_pid == ntohl(msg->pid)) {
			FI_INFO(&tcp2_prov, FI_LOG_EP_CTRL,
				"simultaneous, reject peer\n");
			goto put;
		} else {
			FI_INFO(&tcp2_prov, FI_LOG_EP_CTRL,
				"old connection exists, replacing %p\n", conn);
			tcp2_close_conn(conn);
		}
		break;
	default:
		assert(0);
		tcp2_close_conn(conn);
		break;
	}

accept:
	conn->remote_pid = ntohl(msg->pid);
	ret = tcp2_open_conn(conn, cm_entry->info);
	if (ret)
		goto free;

	msg->pid = htonl((uint32_t) getpid());
	ret = fi_accept(&conn->ep->util_ep.ep_fid, msg, sizeof(*msg));
	if (ret)
		goto close;

	return;

close:
	tcp2_close_conn(conn);
free:
	tcp2_free_conn(conn);
put:
	util_put_peer(peer);
reject:
	(void) fi_reject(&rdm->pep->util_pep.pep_fid, cm_entry->info->handle,
			 msg, sizeof(*msg));
	fi_freeinfo(cm_entry->info);
}

void tcp2_handle_events(struct tcp2_progress *progress)
{
	struct tcp2_event *event;
	struct slist_entry *item;
	struct tcp2_rdm_cm *msg;
	struct tcp2_conn *conn;

	struct tcp2_rdm *rdm;

	ofi_genlock_held(&progress->rdm_lock);
	if (!progress->event_cnt)
		return;

	dlist_foreach_container(&progress->event_list, struct tcp2_rdm, rdm,
				progress_entry) {
		while (!slist_empty(&rdm->event_list)) {
			item = slist_remove_head(&rdm->event_list);
			progress->event_cnt--;
			event = container_of(item, struct tcp2_event, list_entry);

			FI_INFO(&tcp2_prov, FI_LOG_EP_CTRL, "event %s\n",
				fi_tostr(&event->event, FI_TYPE_EQ_EVENT));

			switch (event->event) {
			case FI_CONNREQ:
				tcp2_process_connreq(&event->cm_entry);
				break;
			case FI_CONNECTED:
				conn = event->cm_entry.fid->context;
				msg = (struct tcp2_rdm_cm *) event->cm_entry.data;
				conn->remote_pid = ntohl(msg->pid);
				break;
			case FI_SHUTDOWN:
				conn = event->cm_entry.fid->context;
				tcp2_close_conn(conn);
				tcp2_free_conn(conn);
				break;
			default:
				assert(0);
				break;
			}
			free(event);
		};
	}
}
