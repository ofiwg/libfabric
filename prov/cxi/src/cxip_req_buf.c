/*
 * (C) Copyright 2021 Hewlett Packard Enterprise Development LP
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
#include "config.h"
#include "cxip.h"

static bool cxip_req_buf_is_head(struct cxip_req_buf *buf)
{
	struct cxip_req_buf *head_buf =
		container_of(buf->rxc->active_req_bufs.next,
			     struct cxip_req_buf, req_buf_entry);

	return head_buf == buf;
}

static bool cxip_req_buf_is_consumed(struct cxip_req_buf *buf)
{
	return buf->unlink_length && buf->unlink_length == buf->cur_offset &&
		dlist_empty(&buf->pending_ux_list);
}

static bool cxip_req_buf_is_next_put(struct cxip_req_buf *buf,
					    const union c_event *event)
{
	return (CXI_VA_TO_IOVA(buf->md->md, buf->req_buf) + buf->cur_offset) ==
		event->tgt_long.start;
}

static void cxip_req_buf_put(struct cxip_req_buf *buf, bool repost)
{
	int ret;
	int refcount = ofi_atomic_dec32(&buf->refcount);

	RXC_DBG(buf->rxc, "rbuf=%p refcount=%u\n", buf, refcount);

	if (refcount < 0) {
		RXC_FATAL(buf->rxc, "Request buffer refcount underflow: %d\n",
			  refcount);
	} else if (refcount == 0 && repost) {
		do {
			ret = cxip_req_buf_link(buf, false);
		} while (ret == -FI_EAGAIN);

		if (ret != FI_SUCCESS)
			RXC_FATAL(buf->rxc,
				  "Unhandled request buffer link error: %d",
				  ret);
	}
}

static void cxip_req_buf_get(struct cxip_req_buf *buf)
{
	ofi_atomic_inc32(&buf->refcount);

	RXC_DBG(buf->rxc, "rbuf=%p refcount=%u\n", buf,
		ofi_atomic_get32(&buf->refcount));
}

static void cxip_req_buf_get_header_info(struct cxip_req_buf *buf,
					 struct cxip_ux_send *ux,
					 size_t *header_length,
					 uint64_t *remote_offset)
{
	struct c_port_fab_hdr *fab_hdr =
		(void *)CXI_IOVA_TO_VA(buf->md->md, ux->put_ev.tgt_long.start);
	struct c_port_unrestricted_hdr *unres_hdr =
		(void *)((char *)fab_hdr + sizeof(*fab_hdr));

	if (fab_hdr->ver != 4)
		RXC_FATAL(buf->rxc, "Unsupported fabric header version: %u\n",
			  fab_hdr->ver);

	switch (unres_hdr->ver_pkt_type) {
	case C_V4_PKT_UNRESTRICTED:
		*header_length = sizeof(*fab_hdr) +
			sizeof(struct c_port_unrestricted_hdr);
		*remote_offset =
			c_port_unrestricted_hdr_get_remote_offset(unres_hdr);
		break;
	case C_V4_PKT_SMALLMSG:
		*header_length = sizeof(*fab_hdr) +
			sizeof(struct c_port_small_msg_hdr);
		*remote_offset = 0;
		break;
	default:
		RXC_FATAL(buf->rxc, "Unsupported packet type: %u\n",
			  unres_hdr->ver_pkt_type);
	}
}

static void _cxip_req_buf_ux_free(struct cxip_ux_send *ux, bool repost)
{
	struct cxip_req_buf *buf = ux->req->req_ctx;

	assert(ux->req->type == CXIP_REQ_RBUF);

	cxip_req_buf_put(buf, repost);
	free(ux);

	RXC_DBG(buf->rxc, "rbuf=%p ux=%p\n", buf, ux);
}

void cxip_req_buf_ux_free(struct cxip_ux_send *ux)
{
	_cxip_req_buf_ux_free(ux, true);
}

static struct cxip_ux_send *cxip_req_buf_ux_alloc(struct cxip_req_buf *buf,
						  const union c_event *event)
{
	struct cxip_ux_send *ux;

	ux = calloc(1, sizeof(*ux));
	if (!ux)
		return NULL;

	ux->put_ev = *event;
	ux->req = buf->req;
	dlist_init(&ux->rxc_entry);
	cxip_req_buf_get(buf);

	RXC_DBG(buf->rxc, "rbuf=%p ux=%p\n", buf, ux);

	return ux;
}

/* Caller must hold rxc->lock */
static int cxip_req_buf_process_ux(struct cxip_req_buf *buf,
				   struct cxip_ux_send *ux)
{
	struct cxip_rxc *rxc = buf->rxc;
	size_t header_length;
	uint64_t remote_offset;
	int ret;
	size_t unlink_length;
	bool unlinked = ux->put_ev.tgt_long.auto_unlinked;

	/* Pre-processing of unlink events. */
	if (unlinked)
		unlink_length = ux->put_ev.tgt_long.start -
			CXI_VA_TO_IOVA(buf->md->md, buf->req_buf) +
			ux->put_ev.tgt_long.mlength;

	buf->cur_offset += ux->put_ev.tgt_long.mlength;

	/* Fixed the put event to point to where the payload resides in the
	 * request buffer. In addition, extract the remote offset needed for
	 * rendezvous.
	 */
	cxip_req_buf_get_header_info(buf, ux, &header_length, &remote_offset);
	assert((ssize_t)ux->put_ev.tgt_long.mlength -
	       (ssize_t)header_length >= 0);

	ux->put_ev.tgt_long.start += header_length;
	ux->put_ev.tgt_long.mlength -= header_length;
	ux->put_ev.tgt_long.remote_offset = remote_offset +
		ux->put_ev.tgt_long.mlength;

	/* If making a transition from hardware to software managed
	 * PTLTE, queue request list entries to be appended to
	 * onloaded unexpected list; the software receive list is empty.
	 *
	 * Note: For FC to software transitions, onloading is complete
	 * once flow control has completed. The check for RXC_FLOW_CONTROL
	 * handles any potential race between hardware enabling the PtlTE
	 * and software handling the PtlTE state change event.
	 */
	if (rxc->state != RXC_ENABLED_SOFTWARE &&
	    rxc->state != RXC_FLOW_CONTROL) {
		dlist_insert_tail(&ux->rxc_entry, &rxc->sw_pending_ux_list);
		rxc->sw_pending_ux_list_len++;

		RXC_DBG(buf->rxc, "rbuf=%p ux=%p sw_pending_ux_list_len=%u\n",
			buf, ux, buf->rxc->sw_pending_ux_list_len);
		goto check_unlinked;
	}

	rxc->sw_ux_list_len++;

	ret = cxip_recv_ux_sw_matcher(ux);
	switch (ret) {
	/* Unexpected message needs to be processed again. Put event fields
	 * need to be reset.
	 */
	case -FI_EAGAIN:
		ux->put_ev.tgt_long.mlength += header_length;
		ux->put_ev.tgt_long.start -= header_length;
		buf->cur_offset -= ux->put_ev.tgt_long.mlength;

		rxc->sw_ux_list_len--;
		return -FI_EAGAIN;

	/* Unexpected message failed to match a user posted request. Need to
	 * queue the unexpected message for future processing.
	 */
	case -FI_ENOMSG:
		dlist_insert_tail(&ux->rxc_entry, &rxc->sw_ux_list);

		RXC_DBG(buf->rxc, "rbuf=%p ux=%p sw_ux_list_len=%u\n",
			buf, ux, buf->rxc->sw_ux_list_len);
		break;

	/* Unexpected message successfully matched a user posted request. */
	case FI_SUCCESS:
		break;

	default:
		RXC_FATAL(rxc, "Unexpected cxip_recv_ux_sw_matcher() rc: %d\n",
			  ret);
	}

check_unlinked:
	/* Once unexpected send has been accepted, complete processing of the
	 * unlink.
	 */
	if (unlinked) {
		buf->unlink_length = unlink_length;
		ofi_atomic_dec32(&rxc->req_bufs_linked);

		RXC_DBG(rxc, "rbuf=%p rxc_rbuf_linked=%u\n", buf,
			ofi_atomic_get32(&rxc->req_bufs_linked));

		/* Replenish to keep minimum linked */
		ret = cxip_req_buf_replenish(rxc, false);
		if (ret)
			RXC_WARN(rxc, "Request replenish failed: %d\n", ret);
	}

	RXC_DBG(rxc, "rbuf=%p processed ux_send=%p\n", buf, ux);

	return FI_SUCCESS;
}

static void cxip_req_buf_progress_pending_ux(struct cxip_req_buf *buf)
{
	struct cxip_ux_send *ux;
	struct dlist_entry *tmp;
	int ret;

again:
	dlist_foreach_container_safe(&buf->pending_ux_list, struct cxip_ux_send,
				     ux, rxc_entry, tmp) {
		if (cxip_req_buf_is_next_put(buf, &ux->put_ev)) {
			dlist_remove(&ux->rxc_entry);

			/* The corresponding event from the completion queue has
			 * already been consumed. Thus, -FI_EAGAIN cannot be
			 * returned.
			 */
			do {
				ret = cxip_req_buf_process_ux(buf, ux);
			} while (ret == -FI_EAGAIN);

			/* Previously processed unexpected messages may now be
			 * valid. Need to reprocess the entire list.
			 */
			goto again;
		}
	}
}

static int cxip_req_buf_process_put_event(struct cxip_req_buf *buf,
					  const union c_event *event)
{
	struct cxip_ux_send *ux;
	int ret = FI_SUCCESS;
	struct cxip_rxc *rxc = buf->rxc;

	assert(event->tgt_long.mlength >= CXIP_REQ_BUF_HEADER_MIN_SIZE);

	fastlock_acquire(&rxc->lock);

	ux = cxip_req_buf_ux_alloc(buf, event);
	if (!ux) {
		RXC_WARN(rxc, "Memory allocation error\n");
		ret = -FI_EAGAIN;
		goto unlock;
	}

	/* Target events can be out-of-order with respect to how they were
	 * matched on the PtlTE request list. To maintain the hardware matched
	 * order, software unexpected entries are only processed in the order in
	 * which they land in the request buffer.
	 */
	if (cxip_req_buf_is_head(buf) && cxip_req_buf_is_next_put(buf, event)) {
		ret = cxip_req_buf_process_ux(buf, ux);
		if (ret == -FI_EAGAIN) {
			_cxip_req_buf_ux_free(ux, false);
			goto unlock;
		}

		/* Since events arrive out-of-order, it is possible that a
		 * non-head request buffer receive an event. Scrub all request
		 * buffers processing their pending unexpected lists until a
		 * request buffer is not consumed.
		 */
		while ((buf = dlist_first_entry_or_null(&rxc->active_req_bufs,
							struct cxip_req_buf,
							req_buf_entry))) {
			cxip_req_buf_progress_pending_ux(buf);

			if (cxip_req_buf_is_consumed(buf)) {
				RXC_DBG(rxc, "rbuf=%p consumed\n", buf);

				dlist_remove(&buf->req_buf_entry);
				dlist_insert_tail(&buf->req_buf_entry,
						  &rxc->consumed_req_bufs);

				/* Since buffer is consumed, return reference
				 * taken during the initial linking.
				 */
				cxip_req_buf_put(buf, true);
			} else {
				break;
			}
		}
	} else {
		/* Out-of-order target event. Queue unexpected message on
		 * pending list until these addition events occur.
		 */
		dlist_insert_tail(&ux->rxc_entry, &buf->pending_ux_list);

		RXC_DBG(rxc, "rbuf=%p pend ux_send=%p\n", buf, ux);
	}

unlock:
	fastlock_release(&rxc->lock);

	return ret;
}

static int cxip_req_buf_cb(struct cxip_req *req, const union c_event *event)
{
	struct cxip_req_buf *buf = req->req_ctx;

	switch (event->hdr.event_type) {
	case C_EVENT_LINK:
		RXC_WARN(buf->rxc, "Request LINK error %d\n",
			 cxi_event_rc(event));

		assert(cxi_event_rc(event) == C_RC_NO_SPACE);
		cxip_req_buf_put(buf, false);
		ofi_atomic_dec32(&buf->rxc->req_bufs_linked);

		/* We are running out of LE resources, do not
		 * repost immediately.
		 */
		assert(ofi_atomic_get32(&buf->refcount) == 0);
		dlist_remove(&buf->req_buf_entry);
		dlist_insert_tail(&buf->req_buf_entry,
				  &buf->rxc->free_req_bufs);

		return FI_SUCCESS;

	case C_EVENT_UNLINK:
		assert(!event->tgt_long.auto_unlinked);
		cxip_req_buf_put(buf, false);
		ofi_atomic_dec32(&buf->rxc->req_bufs_linked);

		RXC_DBG(buf->rxc, "rbuf=%p rxc_rbuf_linked=%u\n", buf,
			ofi_atomic_get32(&buf->rxc->req_bufs_linked));

		return FI_SUCCESS;

	case C_EVENT_PUT:
		return cxip_req_buf_process_put_event(buf, event);

	default:
		RXC_FATAL(buf->rxc, "Unexpected event type: %d\n",
			  event->hdr.event_type);
	}
}

int cxip_req_buf_unlink(struct cxip_req_buf *buf)
{
	struct cxip_rxc *rxc = buf->rxc;
	int ret;

	ret = cxip_pte_unlink(rxc->rx_pte, C_PTL_LIST_REQUEST, buf->req->req_id,
			      rxc->rx_cmdq);
	if (ret)
		RXC_DBG(rxc, "Failed to write Unlink command: %d\n", ret);

	return ret;
}

int cxip_req_buf_link(struct cxip_req_buf *buf, bool seq_restart)
{
	struct cxip_rxc *rxc = buf->rxc;
	uint32_t le_flags = C_LE_MANAGE_LOCAL | C_LE_NO_TRUNCATE |
			    C_LE_UNRESTRICTED_BODY_RO | C_LE_OP_PUT |
			    C_LE_UNRESTRICTED_END_RO | C_LE_EVENT_LINK_DISABLE |
			    C_LE_EVENT_UNLINK_DISABLE;
	size_t min_free = CXIP_REQ_BUF_HEADER_MAX_SIZE + rxc->max_eager_size;
	int ret;

	/* Match all eager, long sends */
	union cxip_match_bits mb = {
		.le_type = CXIP_LE_TYPE_RX
	};
	union cxip_match_bits ib = {
		.tag = ~0,
		.tx_id = ~0,
		.cq_data = 1,
		.tagged = 1,
		.match_comp = 1,
	};

	if (seq_restart)
		le_flags |= C_LE_RESTART_SEQ;

	/* Reset request buffer stats used to know when the buffer is consumed.
	 */
	assert(dlist_empty(&buf->pending_ux_list));
	buf->unlink_length = 0;
	buf->cur_offset = 0;

	/* Take a request buffer reference for the link. */
	ret = cxip_pte_append(rxc->rx_pte,
			      CXI_VA_TO_IOVA(buf->md->md, buf->req_buf),
			      cxip_env.req_buf_size, buf->md->md->lac,
			      C_PTL_LIST_REQUEST, buf->req->req_id, mb.raw,
			      ib.raw, CXI_MATCH_ID_ANY, min_free,
			      le_flags, NULL, rxc->rx_cmdq, true);
	if (ret) {
		RXC_DBG(rxc, "Failed to write Append command: %d\n", ret);
	} else {
		dlist_remove(&buf->req_buf_entry);
		dlist_insert_tail(&buf->req_buf_entry,
				  &buf->rxc->active_req_bufs);
		ofi_atomic_inc32(&buf->rxc->req_bufs_linked);

		/* Reference taken until buffer is consumed or manually
		 * unlinked.
		 */
		cxip_req_buf_get(buf);

		RXC_DBG(rxc, "rbuf=%p rxc_rbuf_linked=%u\n", buf,
			ofi_atomic_get32(&buf->rxc->req_bufs_linked));
	}

	return ret;
}

/*
 * cxip_req_buf_alloc() - Allocate a request buffer against an RX context.
 */
struct cxip_req_buf *cxip_req_buf_alloc(struct cxip_rxc *rxc)
{
	struct cxip_req_buf *buf;
	size_t req_buf_size = sizeof(*buf) + rxc->req_buf_size;
	int ret;

	buf = calloc(1, req_buf_size);
	if (!buf)
		goto err;

	if (rxc->hmem) {
		ret = ofi_hmem_host_register(buf, req_buf_size);
		if (ret)
			goto err_free_buf;
	}

	ret = cxip_map(rxc->domain, buf->req_buf, rxc->req_buf_size, &buf->md);
	if (ret)
		goto err_unreg_buf;

	buf->req = cxip_cq_req_alloc(rxc->recv_cq, true, buf);
	if (!buf->req) {
		ret = -FI_ENOMEM;
		goto err_unmap_buf;
	}

	buf->req->cb = cxip_req_buf_cb;
	buf->req->type = CXIP_REQ_RBUF;

	ofi_atomic_initialize32(&buf->refcount, 0);
	dlist_init(&buf->pending_ux_list);
	dlist_init(&buf->req_buf_entry);

	ofi_atomic_inc32(&rxc->req_bufs_allocated);
	buf->rxc = rxc;

	RXC_DBG(rxc, "rbuf=%p rxc_rbuf_cnt=%u\n", buf,
		ofi_atomic_get32(&rxc->req_bufs_allocated));

	return buf;

err_unmap_buf:
	cxip_unmap(buf->md);
err_unreg_buf:
	if (rxc->hmem)
		ofi_hmem_host_unregister(buf);
err_free_buf:
	free(buf);
err:
	return NULL;
}

void cxip_req_buf_free(struct cxip_req_buf *buf)
{
	struct cxip_ux_send *ux;
	struct dlist_entry *tmp;
	struct cxip_rxc *rxc = buf->rxc;

	/* Sanity check making sure the buffer was properly removed before
	 * freeing.
	 */
	assert(dlist_empty(&buf->req_buf_entry));

	dlist_foreach_container_safe(&buf->pending_ux_list, struct cxip_ux_send,
				     ux, rxc_entry, tmp) {
		dlist_remove(&ux->rxc_entry);
		_cxip_req_buf_ux_free(ux, false);
	}

	if (ofi_atomic_get32(&buf->refcount) != 0)
		RXC_FATAL(rxc, "rbuf=%p non-zero refcount: %d\n", buf,
			  ofi_atomic_get32(&buf->refcount));

	cxip_cq_req_free(buf->req);
	cxip_unmap(buf->md);
	if (rxc->hmem)
		ofi_hmem_host_unregister(buf);
	free(buf);

	ofi_atomic_dec32(&rxc->req_bufs_allocated);

	RXC_DBG(rxc, "rbuf=%p rxc_rbuf_cnt=%u\n", buf,
		ofi_atomic_get32(&rxc->req_bufs_allocated));
}

/*
 * cxip_req_buf_replenish() - Replenish RXC request list eager buffers.
 *
 * Caller must hold rxc->rx_lock.
 */
int cxip_req_buf_replenish(struct cxip_rxc *rxc, bool seq_restart)
{
	struct cxip_req_buf *buf;
	struct dlist_entry *tmp;
	int bufs_added = 0;
	int ret = FI_SUCCESS;

	if (rxc->msg_offload)
		return FI_SUCCESS;

	/* Append any buffers that failed to be previously
	 * appended, then replenish up to the minimum that
	 * should be posted.
	 */
	dlist_foreach_container_safe(&rxc->free_req_bufs, struct cxip_req_buf,
				     buf, req_buf_entry, tmp) {

		RXC_DBG(rxc, "Append previous link error req buf entry %p\n",
			buf);

		/* Link call removes from list */
		ret = cxip_req_buf_link(buf, !bufs_added);
		if (ret)
			RXC_WARN(rxc, "Request append failure %d\n", ret);

		bufs_added++;
	}

	while ((ofi_atomic_get32(&rxc->req_bufs_linked) <
		rxc->req_buf_min_posted) &&
	       (!rxc->req_buf_max_count ||
	       (ofi_atomic_get32(&rxc->req_bufs_allocated) <
		rxc->req_buf_max_count))) {

		RXC_DBG(rxc, "Allocate new req buf entry %p\n", buf);

		buf = cxip_req_buf_alloc(rxc);
		if (!buf) {
			RXC_WARN(rxc, "Buffer allocation/registration err\n");
			return -FI_ENOMEM;
		}

		RXC_DBG(rxc, "Link req buf entry %p\n", buf);

		ret = cxip_req_buf_link(buf, !bufs_added);
		if (ret) {
			RXC_WARN(rxc, "Request append failure %d\n", ret);
			dlist_insert_tail(&buf->req_buf_entry,
					  &rxc->free_req_bufs);
			break;
		}
		bufs_added++;
	}

	/* If no buffer appended, check for fatal conditions. */
	if (!bufs_added) {
		if (rxc->req_buf_max_count &&
		    (ofi_atomic_get32(&rxc->req_bufs_allocated) >=
		     rxc->req_buf_max_count))
			RXC_FATAL(rxc,
				  "Request buffer max exceeded: %ld, increase"
				  " or set FI_CXI_REQ_BUF_MAX_COUNT=0\n",
				  rxc->req_buf_max_count);

		if (ofi_atomic_get32(&rxc->req_bufs_linked) < 1)
			RXC_FATAL(rxc, "Request buffer list exhausted\n");
	}

	RXC_DBG(rxc, "req_bufs_allocated=%u, req_bufs_linked=%u\n",
		ofi_atomic_get32(&rxc->req_bufs_allocated),
		ofi_atomic_get32(&rxc->req_bufs_linked));

	return ret;
}
