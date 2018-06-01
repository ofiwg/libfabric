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

#include <ofi.h>
#include <ofi_util.h>
#include "rxm.h"

static int rxm_msg_ep_open(struct rxm_ep *rxm_ep, struct fi_info *msg_info,
			   struct rxm_conn *rxm_conn, void *context)
{
	struct rxm_domain *rxm_domain;
	struct fid_ep *msg_ep;
	int ret;

	rxm_domain = container_of(rxm_ep->util_ep.domain, struct rxm_domain,
			util_domain);
	ret = fi_endpoint(rxm_domain->msg_domain, msg_info, &msg_ep, context);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Unable to create msg_ep\n");
		return ret;
	}

	ret = fi_ep_bind(msg_ep, &rxm_ep->msg_eq->fid, 0);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_FABRIC, "Unable to bind msg EP to EQ\n");
		goto err;
	}

	if (rxm_ep->srx_ctx) {
		ret = fi_ep_bind(msg_ep, &rxm_ep->srx_ctx->fid, 0);
		if (ret) {
			FI_WARN(&rxm_prov, FI_LOG_FABRIC,
				"Unable to bind msg EP to shared RX ctx\n");
			goto err;
		}
	}

	// TODO add other completion flags
	ret = fi_ep_bind(msg_ep, &rxm_ep->msg_cq->fid, FI_TRANSMIT | FI_RECV);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
				"Unable to bind msg_ep to msg_cq\n");
		goto err;
	}

	ret = fi_enable(msg_ep);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Unable to enable msg_ep\n");
		goto err;
	}

	if (!rxm_ep->srx_ctx) {
		ret = rxm_ep_prepost_buf(rxm_ep, msg_ep);
		if (ret)
			goto err;
	}

	rxm_conn->msg_ep = msg_ep;
	return 0;
err:
	fi_close(&msg_ep->fid);
	return ret;
}

void rxm_conn_close(struct util_cmap_handle *handle)
{
	struct rxm_conn *rxm_conn = container_of(handle, struct rxm_conn, handle);
	if (!rxm_conn->msg_ep)
		return;

	/* Assuming fi_close also shuts down the connection gracefully if the
	 * endpoint is in connected state */
	if (fi_close(&rxm_conn->msg_ep->fid))
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Unable to close msg_ep\n");
	FI_DBG(&rxm_prov, FI_LOG_EP_CTRL, "Closed msg_ep\n");
	rxm_conn->msg_ep = NULL;
}

static void rxm_conn_free(struct util_cmap_handle *handle)
{
	rxm_conn_close(handle);
	free(container_of(handle, struct rxm_conn, handle));
}

static struct util_cmap_handle *rxm_conn_alloc(void)
{
	struct rxm_conn *rxm_conn = calloc(1, sizeof(*rxm_conn));
	if (OFI_UNLIKELY(!rxm_conn))
		return NULL;

	dlist_init(&rxm_conn->postponed_tx_list);
	return &rxm_conn->handle;
}

static int
rxm_msg_process_connreq(struct rxm_ep *rxm_ep, struct fi_info *msg_info,
			void *data)
{
	struct rxm_conn *rxm_conn;
	struct rxm_cm_data *remote_cm_data = data;
	struct rxm_cm_data cm_data;
	struct util_cmap_handle *handle;
	int ret;

	ret = ofi_cmap_process_connreq(rxm_ep->util_ep.cmap,
				       &remote_cm_data->name, &handle);
	if (ret)
		goto err1;

	rxm_conn = container_of(handle, struct rxm_conn, handle);

	rxm_conn->handle.remote_key = remote_cm_data->conn_id;

	ret = rxm_msg_ep_open(rxm_ep, msg_info, rxm_conn, handle);
	if (ret)
		goto err2;

	cm_data.conn_id = rxm_conn->handle.key;

	ret = fi_accept(rxm_conn->msg_ep, &cm_data, sizeof(cm_data));
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_FABRIC,
				"Unable to accept incoming connection\n");
		goto err2;
	}
	return ret;
err2:
	ofi_cmap_del_handle(&rxm_conn->handle);
err1:
	FI_DBG(&rxm_prov, FI_LOG_EP_CTRL,
		"Rejecting incoming connection request\n");
	if (fi_reject(rxm_ep->msg_pep, msg_info->handle, NULL, 0))
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
				"Unable to reject incoming connection\n");
	return ret;
}

static int rxm_conn_handle_notify(struct fi_eq_entry *eq_entry)
{
	switch((enum ofi_cmap_signal)eq_entry->data) {
	case OFI_CMAP_FREE:
		FI_DBG(&rxm_prov, FI_LOG_FABRIC, "Freeing handle\n");
		rxm_conn_free((struct util_cmap_handle *)eq_entry->context);
		return 0;
	case OFI_CMAP_EXIT:
		FI_TRACE(&rxm_prov, FI_LOG_FABRIC, "Closing event handler\n");
		return 1;
	default:
		FI_WARN(&rxm_prov, FI_LOG_FABRIC, "Unknown cmap signal\n");
		return 1;
	}
}

static void rxm_conn_handle_eq_err(struct rxm_ep *rxm_ep, ssize_t rd)
{
	struct fi_eq_err_entry err_entry = {0};

	if (rd != -FI_EAVAIL) {
		FI_WARN(&rxm_prov, FI_LOG_FABRIC, "Unable to fi_eq_sread\n");
		return;
	}
	OFI_EQ_READERR(&rxm_prov, FI_LOG_FABRIC, rxm_ep->msg_eq, rd, err_entry);
	if (err_entry.err == ECONNREFUSED) {
		FI_DBG(&rxm_prov, FI_LOG_FABRIC, "Connection refused\n");
		ofi_cmap_process_reject(rxm_ep->util_ep.cmap,
					err_entry.fid->context);
	}
}

static void rxm_conn_handle_postponed_op(struct rxm_ep *rxm_ep,
					 struct util_cmap_handle *handle)
{
	struct rxm_tx_entry *tx_entry;
	struct rxm_conn *rxm_conn = container_of(handle, struct rxm_conn, handle);

	while (!dlist_empty(&rxm_conn->postponed_tx_list)) {
		dlist_pop_front(&rxm_conn->postponed_tx_list, struct rxm_tx_entry,
				tx_entry, postponed_entry);
		if (!(tx_entry->comp_flags & FI_RMA))
			rxm_ep_handle_postponed_tx_op(rxm_ep, rxm_conn, tx_entry);
		else
			rxm_ep_handle_postponed_rma_op(rxm_ep, rxm_conn, tx_entry);
	}
}

static void *rxm_conn_event_handler(void *arg)
{
	struct fi_eq_cm_entry *entry;
	size_t datalen = sizeof(struct rxm_cm_data);
	size_t len = sizeof(*entry) + datalen;
	struct rxm_ep *rxm_ep = container_of(arg, struct rxm_ep, util_ep);
	struct rxm_cm_data *cm_data;
	uint32_t event;
	ssize_t rd;

	entry = calloc(1, len);
	if (!entry) {
		FI_WARN(&rxm_prov, FI_LOG_FABRIC, "Unable to allocate memory!\n");
		return NULL;
	}

	FI_DBG(&rxm_prov, FI_LOG_FABRIC, "Starting conn event handler\n");
	while (1) {
		rd = fi_eq_sread(rxm_ep->msg_eq, &event, entry, len, -1, 0);
		/* We would receive more bytes than sizeof *entry during CONNREQ */
		if (rd < 0) {
			rxm_conn_handle_eq_err(rxm_ep, rd);
			continue;
		}

		switch(event) {
		case FI_NOTIFY:
			if (rxm_conn_handle_notify((struct fi_eq_entry *)entry))
				goto exit;
			break;
		case FI_CONNREQ:
			FI_DBG(&rxm_prov, FI_LOG_FABRIC, "Got new connection\n");
			if ((size_t)rd != len) {
				FI_WARN(&rxm_prov, FI_LOG_FABRIC,
					"Received size (%zd) not matching "
					"expected (%zu)\n", rd, len);
				goto exit;
			}
			rxm_msg_process_connreq(rxm_ep, entry->info, entry->data);
			break;
		case FI_CONNECTED:
			FI_DBG(&rxm_prov, FI_LOG_FABRIC,
			       "Connection successful\n");
			fastlock_acquire(&rxm_ep->util_ep.cmap->lock);
			cm_data = (void *)entry->data;
			ofi_cmap_process_connect(rxm_ep->util_ep.cmap,
						 entry->fid->context,
						 ((rd - sizeof(*entry)) ?
						  &cm_data->conn_id : NULL));
			rxm_conn_handle_postponed_op(rxm_ep, entry->fid->context);
			fastlock_release(&rxm_ep->util_ep.cmap->lock);
			break;
		case FI_SHUTDOWN:
			FI_DBG(&rxm_prov, FI_LOG_FABRIC,
			       "Received connection shutdown\n");
			ofi_cmap_process_shutdown(rxm_ep->util_ep.cmap,
						  entry->fid->context);
			break;
		default:
			FI_WARN(&rxm_prov, FI_LOG_FABRIC,
				"Unknown event: %u\n", event);
			goto exit;
		}
	}
exit:
	free(entry);
	return NULL;
}

static int rxm_prepare_cm_data(struct fid_pep *pep, struct util_cmap_handle *handle,
		struct rxm_cm_data *cm_data)
{
	size_t cm_data_size = 0;
	size_t name_size = sizeof(cm_data->name);
	size_t opt_size = sizeof(cm_data_size);
	int ret;

	ret = fi_getopt(&pep->fid, FI_OPT_ENDPOINT, FI_OPT_CM_DATA_SIZE,
			&cm_data_size, &opt_size);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "fi_getopt failed\n");
		return ret;
	}

	if (cm_data_size < sizeof(*cm_data)) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "MSG EP CM data size too small\n");
		return -FI_EOTHER;
	}

	ret = fi_getname(&pep->fid, &cm_data->name, &name_size);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Unable to get msg pep name\n");
		return ret;
	}

	cm_data->conn_id = handle->key;
	return 0;
}

static int
rxm_conn_connect(struct util_ep *util_ep, struct util_cmap_handle *handle,
		 const void *addr, size_t addrlen)
{
	struct rxm_ep *rxm_ep;
	struct rxm_conn *rxm_conn;
	struct fi_info *msg_info;
	struct rxm_cm_data cm_data;
	int ret;

	rxm_ep = container_of(util_ep, struct rxm_ep, util_ep);

	rxm_conn = container_of(handle, struct rxm_conn, handle);

	free(rxm_ep->msg_info->dest_addr);
	rxm_ep->msg_info->dest_addrlen = addrlen;

	rxm_ep->msg_info->dest_addr = mem_dup(addr, rxm_ep->msg_info->dest_addrlen);
	if (!rxm_ep->msg_info->dest_addr)
		return -FI_ENOMEM;

	ret = fi_getinfo(rxm_ep->util_ep.domain->fabric->fabric_fid.api_version,
			 NULL, NULL, 0, rxm_ep->msg_info, &msg_info);
	if (ret)
		return ret;

	ret = rxm_msg_ep_open(rxm_ep, msg_info, rxm_conn, &rxm_conn->handle);
	if (ret)
		goto err1;

	/* We have to send passive endpoint's address to the server since the
	 * address from which connection request would be sent would have a
	 * different port. */
	ret = rxm_prepare_cm_data(rxm_ep->msg_pep, &rxm_conn->handle, &cm_data);
	if (ret)
		goto err2;

	ret = fi_connect(rxm_conn->msg_ep, msg_info->dest_addr, &cm_data, sizeof(cm_data));
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Unable to connect msg_ep\n");
		goto err2;
	}
	fi_freeinfo(msg_info);
	return 0;
err2:
	fi_close(&rxm_conn->msg_ep->fid);
	rxm_conn->msg_ep = NULL;
err1:
	fi_freeinfo(msg_info);
	return ret;
}

static int rxm_conn_signal(struct util_ep *util_ep, void *context,
			   enum ofi_cmap_signal signal)
{
	struct rxm_ep *rxm_ep = container_of(util_ep, struct rxm_ep, util_ep);
	struct fi_eq_entry entry = {0};
	ssize_t rd;

	entry.context = context;
	entry.data = (uint64_t)signal;

	rd = fi_eq_write(rxm_ep->msg_eq, FI_NOTIFY, &entry, sizeof(entry), 0);
	if (rd != sizeof(entry)) {
		FI_WARN(&rxm_prov, FI_LOG_FABRIC, "Unable to signal\n");
		return (int)rd;
	}
	return 0;
}

struct util_cmap *rxm_conn_cmap_alloc(struct rxm_ep *rxm_ep)
{
	struct util_cmap_attr attr;
	struct util_cmap *cmap = NULL;
	void *name;
	size_t len;
	int ret;

	len = rxm_ep->msg_info->src_addrlen;
	name = calloc(1, len);
	if (!name) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
			"Unable to allocate memory for EP name\n");
		return NULL;
	}

	/* Passive endpoint should already have fi_setname or fi_listen
	 * called on it for this to work */
	ret = fi_getname(&rxm_ep->msg_pep->fid, name, &len);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
			"Unable to fi_getname on msg_ep\n");
		goto fn;
	}
	ofi_straddr_dbg(&rxm_prov, FI_LOG_EP_CTRL, "local_name", name);

	attr.name		= name;
	attr.alloc 		= rxm_conn_alloc;
	attr.close 		= rxm_conn_close;
	attr.free 		= rxm_conn_free;
	attr.connect 		= rxm_conn_connect;
	attr.event_handler	= rxm_conn_event_handler;
	attr.signal		= rxm_conn_signal;

	cmap = ofi_cmap_alloc(&rxm_ep->util_ep, &attr);
	if (!cmap)
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
			"Unable to allocate CMAP\n");
fn:
	free(name);
	return cmap;
}
