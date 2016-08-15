/*
 * Copyright (c) 2016 Intel Corporation, Inc.  All rights reserved.
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

#include <fi_util.h>
#include "rxm.h"

int rxm_msg_ep_open(struct rxm_ep *rxm_ep, struct fi_info *msg_info,
		struct rxm_conn *rxm_conn)
{
	struct rxm_domain *rxm_domain;
	struct rxm_fabric *rxm_fabric;
	struct rxm_cq *rxm_cq;
	struct fid_ep *msg_ep;
	int ret;

	rxm_domain = container_of(rxm_ep->util_ep.domain, struct rxm_domain,
			util_domain);
	rxm_fabric = container_of(rxm_domain->util_domain.fabric, struct rxm_fabric,
			util_fabric);
	ret = fi_endpoint(rxm_domain->msg_domain, msg_info, &msg_ep, rxm_conn);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Unable to create msg_ep\n");
		return ret;
	}

	ret = fi_ep_bind(msg_ep, &rxm_fabric->msg_eq->fid, 0);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_FABRIC, "Unable to bind msg EP to EQ\n");
		goto err;
	}

	ret = fi_ep_bind(msg_ep, &rxm_ep->srx_ctx->fid, 0);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_FABRIC, "Unable to bind msg EP to shared RX ctx\n");
		goto err;
	}

	// TODO add other completion flags
	if (rxm_ep->util_ep.tx_cq) {
		rxm_cq = container_of(rxm_ep->util_ep.tx_cq, struct rxm_cq, util_cq);
		ret = fi_ep_bind(msg_ep, &rxm_cq->msg_cq->fid, FI_TRANSMIT);
		if (ret) {
			FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
					"Unable to bind msg_ep to tx_cq\n");
			goto err;
		}
	}

	if (rxm_ep->util_ep.rx_cq) {
		rxm_cq = container_of(rxm_ep->util_ep.rx_cq, struct rxm_cq, util_cq);
		ret = fi_ep_bind(msg_ep, &rxm_cq->msg_cq->fid, FI_RECV);
		if (ret) {
			FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
					"Unable to bind msg_ep to rx_cq\n");
			goto err;
		}
	}

	ret = fi_enable(msg_ep);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Unable to enable msg_ep\n");
		goto err;
	}

	rxm_conn->msg_ep = msg_ep;
	return 0;
err:
	fi_close(&msg_ep->fid);
	return ret;
}

void rxm_conn_close(void *arg)
{
	struct util_cmap_handle *handle = (struct util_cmap_handle *)arg;
	struct rxm_conn *rxm_conn = container_of(handle, struct rxm_conn, handle);
	int ret;

	if ((rxm_conn->handle.state == CMAP_UNSPEC) || !rxm_conn->msg_ep)
		goto out;

	if (rxm_conn->handle.state == CMAP_CONNECTED) {
		ret = fi_shutdown(rxm_conn->msg_ep, 0);
		if (ret)
			FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
					"Unable to close connection\n");
	}
	ret = fi_close(&rxm_conn->msg_ep->fid);
	if (ret)
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
				"Unable to close msg_ep\n");
out:
	free(rxm_conn);
}

int rxm_msg_process_connreq(struct rxm_ep *rxm_ep, struct fi_info *msg_info,
		void *data, ssize_t datalen)
{
	struct rxm_conn *rxm_conn;
	int ret;

	if (!datalen) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "No addr info received in cm_data\n");
		ret = -FI_EINVAL;
		goto err1;
	}

	if  (!(rxm_conn = calloc(1, sizeof(*rxm_conn)))) {
		ret = -FI_ENOMEM;
		goto err1;
	}

	ret = ofi_cmap_add_handle(rxm_ep->cmap, &rxm_conn->handle, CMAP_CONNECTING,
			FI_ADDR_UNSPEC, data, datalen);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_FABRIC, "Unable to add handle/peer\n");
		goto err2;
	}

	ret = rxm_msg_ep_open(rxm_ep, msg_info, rxm_conn);
	if (ret)
		goto err2;

	ret = fi_accept(rxm_conn->msg_ep, NULL, 0);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_FABRIC,
				"Unable to accept incoming connection\n");
		goto err2;
	}
	return ret;
err2:
	ofi_cmap_del_handle(&rxm_conn->handle);
err1:
	if (fi_reject(rxm_ep->msg_pep, msg_info->handle, NULL, 0))
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
				"Unable to reject incoming connection\n");
	return ret;
}

static void rxm_msg_process_connect_event(fid_t fid)
{
	struct rxm_conn *rxm_conn = (struct rxm_conn *)fid->context;
	ofi_cmap_update_state(&rxm_conn->handle, CMAP_CONNECTED);
}
static void rxm_msg_process_shutdown_event(fid_t fid)
{
	// TODO process shutdown - need to diff local and remote shutdown
	return;
}

void *rxm_msg_listener(void *arg)
{
	struct fi_eq_cm_entry *entry;
	struct fi_eq_err_entry err_entry;
	size_t datalen = sizeof(struct sockaddr);
	size_t len = sizeof(*entry) + datalen;
	struct rxm_fabric *rxm_fabric = (struct rxm_fabric *)arg;
	uint32_t event;
	ssize_t rd;
	int ret;

	entry = calloc(1, len);
	if (!entry) {
		FI_WARN(&rxm_prov, FI_LOG_FABRIC, "Unable to allocate memory!\n");
		return NULL;
	}

	FI_DBG(&rxm_prov, FI_LOG_FABRIC, "Starting MSG listener thread\n");
	while (1) {
		rd = fi_eq_sread(rxm_fabric->msg_eq, &event, entry, len, -1, 0);
		/* We would receive more bytes than sizeof *entry during CONNREQ */
		if (rd < 0) {
			if (rd == -FI_EAVAIL)
				OFI_EQ_READERR(&rxm_prov, FI_LOG_FABRIC,
						rxm_fabric->msg_eq, rd, err_entry);
			else
				FI_WARN(&rxm_prov, FI_LOG_FABRIC,
						"msg: unable to fi_eq_sread\n");
			continue;
		}

		switch(event) {
		case FI_NOTIFY:
			FI_WARN(&rxm_prov, FI_LOG_FABRIC, "Closing rxm msg listener\n");
			return NULL;
		case FI_CONNREQ:
			if (rd != len)
				goto err;
			FI_DBG(&rxm_prov, FI_LOG_FABRIC, "Got new connection\n");
			ret = rxm_msg_process_connreq(entry->fid->context, entry->info,
					entry->data, rd - sizeof(*entry));
			if (ret)
				FI_WARN(&rxm_prov, FI_LOG_FABRIC,
						"Unable to process connection request\n");
			break;
		case FI_CONNECTED:
			FI_DBG(&rxm_prov, FI_LOG_FABRIC, "Connected\n");
			rxm_msg_process_connect_event(entry->fid);
			break;
		case FI_SHUTDOWN:
			FI_DBG(&rxm_prov, FI_LOG_FABRIC, "Received connection shutdown\n");
			rxm_msg_process_shutdown_event(entry->fid);
			break;
		default:
			FI_WARN(&rxm_prov, FI_LOG_FABRIC, "Unknown event: %u\n", event);
		}
		continue;
err:
		FI_WARN(&rxm_prov, FI_LOG_FABRIC,
				"Received size (%d) not matching expected (%d)\n", rd, len);
	}
}

int rxm_msg_connect(struct rxm_ep *rxm_ep, fi_addr_t fi_addr,
		struct fi_info *msg_hints)
{
	struct rxm_conn *rxm_conn;
	struct fi_info *msg_info;
	struct sockaddr name;
	size_t name_len;
	int ret;

	assert(!msg_hints->dest_addr);

	msg_hints->dest_addrlen = rxm_ep->util_ep.av->addrlen;
	msg_hints->dest_addr = mem_dup(ofi_av_get_addr(rxm_ep->util_ep.av,
				fi_addr), msg_hints->dest_addrlen);

	ret = fi_getinfo(rxm_prov.version, NULL, NULL, 0, msg_hints, &msg_info);
	if (ret)
		return ret;

	if  (!(rxm_conn = calloc(1, sizeof(*rxm_conn)))) {
		ret = -FI_ENOMEM;
		goto err1;
	}

	ret = ofi_cmap_add_handle(rxm_ep->cmap, &rxm_conn->handle,
			CMAP_CONNECTING, fi_addr, NULL, 0);
	if (ret)
		goto err2;

	ret = rxm_msg_ep_open(rxm_ep, msg_info, rxm_conn);
	if (ret)
		goto err2;

	ret = fi_getname(&rxm_ep->msg_pep->fid, &name, &name_len);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Unable to get msg pep name\n");
		goto err2;
	}

	ret = fi_connect(rxm_conn->msg_ep, msg_info->dest_addr, &name, name_len);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Unable to connect msg_ep\n");
		goto err2;
	}
	fi_freeinfo(msg_info);
	return 0;
err2:
	ofi_cmap_del_handle(&rxm_conn->handle);
err1:
	fi_freeinfo(msg_info);
	return ret;
}

int rxm_get_msg_ep(struct rxm_ep *rxm_ep, fi_addr_t fi_addr,
		struct fid_ep **msg_ep)
{
	struct util_cmap_handle *handle;
	struct rxm_conn *rxm_conn;

	if (fi_addr > rxm_ep->util_ep.av->count) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Invalid fi_addr\n");
		return -FI_EINVAL;
	}

	handle = ofi_cmap_get_handle(rxm_ep->cmap, fi_addr);
	if (!handle)
		goto connect;

	switch (handle->state) {
	case CMAP_CONNECTING:
		return -FI_EAGAIN;
	case CMAP_CONNECTED:
		rxm_conn = container_of(handle, struct rxm_conn, handle);
		*msg_ep = rxm_conn->msg_ep;
		return 0;
	default:
		/* We shouldn't be here */
		assert(0);
	}

connect:
	if (rxm_msg_connect(rxm_ep, fi_addr, rxm_ep->msg_info)) {
		FI_WARN(&rxm_prov, FI_LOG_EP_DATA, "Unable to connect\n");
		return -FI_EOTHER;
	}
	return -FI_EAGAIN;
}


