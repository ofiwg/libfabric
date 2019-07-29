/*
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
 * Copyright (c) 2018 Cray Inc. All rights reserved.
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

#include <ofi_util.h>
#include "fi_verbs.h"

const struct fi_info *
fi_ibv_get_verbs_info(const struct fi_info *ilist, const char *domain_name)
{
	const struct fi_info *fi;

	for (fi = ilist; fi; fi = fi->next) {
		if (!strcmp(fi->domain_attr->name, domain_name))
			return fi;
	}

	return NULL;
}

static ssize_t
fi_ibv_eq_readerr(struct fid_eq *eq, struct fi_eq_err_entry *entry,
		  uint64_t flags)
{
	struct fi_ibv_eq *_eq =
		container_of(eq, struct fi_ibv_eq, eq_fid.fid);
	ssize_t rd = -FI_EAGAIN;
	fastlock_acquire(&_eq->lock);
	if (!_eq->err.err)
		goto unlock;

	ofi_eq_handle_err_entry(_eq->fab->util_fabric.fabric_fid.api_version,
				flags, &_eq->err, entry);
	rd = sizeof(*entry);
unlock:
	fastlock_release(&_eq->lock);
	return rd;
}

/* Caller must hold eq:lock */
void fi_ibv_eq_set_xrc_conn_tag(struct fi_ibv_xrc_ep *ep)
{
	struct fi_ibv_eq *eq = ep->base_ep.eq;

	assert(ep->conn_setup);
	assert(ep->conn_setup->conn_tag == VERBS_CONN_TAG_INVALID);
	ep->conn_setup->conn_tag =
		(uint32_t)ofi_idx2key(&eq->xrc.conn_key_idx,
				ofi_idx_insert(eq->xrc.conn_key_map, ep));
	ep->conn_setup->created_conn_tag = true;
}

/* Caller must hold eq:lock */
void fi_ibv_eq_clear_xrc_conn_tag(struct fi_ibv_xrc_ep *ep)
{
	struct fi_ibv_eq *eq = ep->base_ep.eq;
	int index;

	assert(ep->conn_setup);
	if (!ep->conn_setup->created_conn_tag)
		return;

	index = ofi_key2idx(&eq->xrc.conn_key_idx,
			    (uint64_t)ep->conn_setup->conn_tag);
	if (!ofi_idx_is_valid(eq->xrc.conn_key_map, index))
	    VERBS_WARN(FI_LOG_EP_CTRL,
		       "Invalid XRC connection connection tag\n");
	else
		ofi_idx_remove(eq->xrc.conn_key_map, index);
	ep->conn_setup->conn_tag = VERBS_CONN_TAG_INVALID;
}

/* Caller must hold eq:lock */
struct fi_ibv_xrc_ep *fi_ibv_eq_xrc_conn_tag2ep(struct fi_ibv_eq *eq,
						uint32_t conn_tag)
{
	struct fi_ibv_xrc_ep *ep;
	int index;

	index = ofi_key2idx(&eq->xrc.conn_key_idx, (uint64_t)conn_tag);
	ep = ofi_idx_lookup(eq->xrc.conn_key_map, index);
	if (!ep || ep->magic != VERBS_XRC_EP_MAGIC) {
		VERBS_WARN(FI_LOG_EP_CTRL, "XRC EP is not valid\n");
		return NULL;
	}
	if (!ep->conn_setup) {
		VERBS_WARN(FI_LOG_EP_CTRL,
			   "Bad state, no connection data\n");
		return NULL;
	}
	if (ep->conn_setup->conn_tag != conn_tag) {
		VERBS_WARN(FI_LOG_EP_CTRL, "Connection tag mismatch\n");
		return NULL;
	}

	ofi_idx_remove(eq->xrc.conn_key_map, index);
	ep->conn_setup->conn_tag = VERBS_CONN_TAG_INVALID;

	return ep;
}

static int fi_ibv_eq_set_xrc_info(struct rdma_cm_event *event,
				  struct fi_ibv_xrc_conn_info *info)
{
	struct fi_ibv_xrc_cm_data *remote = (struct fi_ibv_xrc_cm_data *)
						event->param.conn.private_data;
	int ret;

	ret = fi_ibv_verify_xrc_cm_data(remote,
					event->param.conn.private_data_len);
	if (ret)
		return ret;

	info->is_reciprocal = remote->reciprocal;
	info->conn_tag = ntohl(remote->conn_tag);
	info->port = ntohs(remote->port);
	info->conn_data = ntohl(remote->param);
	info->conn_param = event->param.conn;
	info->conn_param.private_data = NULL;
	info->conn_param.private_data_len = 0;

	return FI_SUCCESS;
}

static int
fi_ibv_pep_dev_domain_match(struct fi_info *hints, const char *devname)
{
	int ret;

	if ((FI_IBV_EP_PROTO(hints)) == FI_PROTO_RDMA_CM_IB_XRC)
		ret = fi_ibv_cmp_xrc_domain_name(hints->domain_attr->name,
						 devname);
	else
		ret = strcmp(hints->domain_attr->name, devname);

	return ret;
}

static int
fi_ibv_eq_cm_getinfo(struct rdma_cm_event *event, struct fi_info *pep_info,
		     struct fi_info **info)
{
	struct fi_info *hints;
	struct fi_ibv_connreq *connreq;
	const char *devname = ibv_get_device_name(event->id->verbs->device);
	int ret = -FI_ENOMEM;

	if (!(hints = fi_dupinfo(pep_info))) {
		VERBS_WARN(FI_LOG_EP_CTRL, "dupinfo failure\n");
		return -FI_ENOMEM;
	}

	/* Free src_addr info from pep to avoid addr reuse errors */
	free(hints->src_addr);
	hints->src_addr = NULL;
	hints->src_addrlen = 0;

	if (!strcmp(hints->domain_attr->name, VERBS_ANY_DOMAIN)) {
		free(hints->domain_attr->name);
		if (!(hints->domain_attr->name = strdup(devname)))
			goto err1;
	} else {
		if (fi_ibv_pep_dev_domain_match(hints, devname)) {
			VERBS_WARN(FI_LOG_EQ, "Passive endpoint domain: %s does"
				   " not match device: %s where we got a "
				   "connection request\n",
				   hints->domain_attr->name, devname);
			ret = -FI_ENODATA;
			goto err1;
		}
	}

	if (!strcmp(hints->domain_attr->name, VERBS_ANY_FABRIC)) {
		free(hints->fabric_attr->name);
		hints->fabric_attr->name = NULL;
	}

	if (fi_ibv_getinfo(hints->fabric_attr->api_version, NULL, NULL, 0,
			   hints, info))
		goto err1;

	assert(!(*info)->dest_addr);

	free((*info)->src_addr);

	(*info)->src_addrlen = fi_ibv_sockaddr_len(rdma_get_local_addr(event->id));
	if (!((*info)->src_addr = malloc((*info)->src_addrlen)))
		goto err2;
	memcpy((*info)->src_addr, rdma_get_local_addr(event->id), (*info)->src_addrlen);

	(*info)->dest_addrlen = fi_ibv_sockaddr_len(rdma_get_peer_addr(event->id));
	if (!((*info)->dest_addr = malloc((*info)->dest_addrlen)))
		goto err2;
	memcpy((*info)->dest_addr, rdma_get_peer_addr(event->id), (*info)->dest_addrlen);

	ofi_straddr_dbg(&fi_ibv_prov, FI_LOG_EQ, "src", (*info)->src_addr);
	ofi_straddr_dbg(&fi_ibv_prov, FI_LOG_EQ, "dst", (*info)->dest_addr);

	connreq = calloc(1, sizeof *connreq);
	if (!connreq) {
		VERBS_WARN(FI_LOG_EP_CTRL,
			   "Unable to allocate connreq memory\n");
		goto err2;
	}

	connreq->handle.fclass = FI_CLASS_CONNREQ;
	connreq->id = event->id;

	if (fi_ibv_is_xrc(*info)) {
		connreq->is_xrc = 1;
		ret = fi_ibv_eq_set_xrc_info(event, &connreq->xrc);
		if (ret)
			goto err3;
	}

	(*info)->handle = &connreq->handle;
	fi_freeinfo(hints);
	return 0;

err3:
	free(connreq);
err2:
	fi_freeinfo(*info);
err1:
	fi_freeinfo(hints);
	return ret;
}

static inline int fi_ibv_eq_copy_event_data(struct fi_eq_cm_entry *entry,
				size_t max_dest_len, const void *priv_data,
				size_t priv_datalen)
{
	const struct fi_ibv_cm_data_hdr *cm_hdr = priv_data;

	size_t datalen = MIN(max_dest_len - sizeof(*entry), cm_hdr->size);
	if (datalen)
		memcpy(entry->data, cm_hdr->data, datalen);

	return datalen;
}

static void fi_ibv_eq_skip_xrc_cm_data(const void **priv_data,
				       size_t *priv_data_len)
{
	const struct fi_ibv_xrc_cm_data *cm_data = *priv_data;

	if (*priv_data_len > sizeof(*cm_data)) {
		*priv_data = (cm_data + 1);
		*priv_data_len -= sizeof(*cm_data);
	}
}

static int
fi_ibv_eq_xrc_connreq_event(struct fi_ibv_eq *eq, struct fi_eq_cm_entry *entry,
			    const void **priv_data, size_t *priv_datalen)
{
	struct fi_ibv_connreq *connreq = container_of(entry->info->handle,
						struct fi_ibv_connreq, handle);
	struct fi_ibv_xrc_ep *ep;
	struct fi_ibv_xrc_cm_data cm_data;
	int ret;

	if (!connreq->xrc.is_reciprocal) {
		fi_ibv_eq_skip_xrc_cm_data(priv_data, priv_datalen);
		return FI_SUCCESS;
	}

	fastlock_acquire(&eq->lock);
	/*
	 * Reciprocal connections are initiated and handled internally by
	 * the provider, get the endpoint that issued the original connection
	 * request.
	 */
	ep = fi_ibv_eq_xrc_conn_tag2ep(eq, connreq->xrc.conn_tag);
	if (!ep) {
		VERBS_WARN(FI_LOG_EP_CTRL,
			   "Reciprocal XRC connection tag 0x%x not found\n",
			   connreq->xrc.conn_tag);
		goto done;
	}
	assert(ep->conn_state == FI_IBV_XRC_ORIG_CONNECTED);

	ep->tgt_id = connreq->id;
	ep->tgt_id->context = &ep->base_ep.util_ep.ep_fid.fid;
	ep->base_ep.info->handle = entry->info->handle;

	ret = rdma_migrate_id(ep->tgt_id, ep->base_ep.eq->channel);
	if (ret) {
		VERBS_WARN(FI_LOG_EP_CTRL, "Could not migrate CM ID\n");
		goto send_reject;
	}

	ret = fi_ibv_accept_xrc(ep, FI_IBV_RECIP_CONN, &cm_data,
				sizeof(cm_data));
	if (ret) {
		VERBS_WARN(FI_LOG_EP_CTRL,
			   "Reciprocal XRC Accept failed %d\n", ret);
		goto send_reject;
	}
done:
	fastlock_release(&eq->lock);

	/* Event is handled internally and not passed to the application */
	return -FI_EAGAIN;

send_reject:
	if (rdma_reject(connreq->id, *priv_data, *priv_datalen))
		VERBS_WARN(FI_LOG_EP_CTRL, "rdma_reject %d\n", -errno);
	fastlock_release(&eq->lock);

	return -FI_EAGAIN;
}

static int
fi_ibv_eq_xrc_conn_event(struct fi_ibv_xrc_ep *ep,
			 struct rdma_cm_event *cma_event,
			 struct fi_eq_cm_entry *entry)
{
	struct fi_ibv_xrc_conn_info xrc_info;
	struct fi_ibv_xrc_cm_data cm_data;
	const void *priv_data = cma_event->param.conn.private_data;
	size_t priv_datalen = cma_event->param.conn.private_data_len;
	int ret;

	VERBS_DBG(FI_LOG_EP_CTRL, "EP %p INITIAL CONNECTION DONE state %d\n",
		  ep, ep->conn_state);
	fi_ibv_next_xrc_conn_state(ep);

	/*
	 * Original application initiated connect is done, if the passive
	 * side of that connection initiate the reciprocal connection request
	 * to create bidirectional connectivity.
	 */
	if (priv_data) {
		ret = fi_ibv_eq_set_xrc_info(cma_event, &xrc_info);
		if (ret) {
			fi_ibv_prev_xrc_conn_state(ep);
			rdma_disconnect(ep->base_ep.id);
			goto err;
		}
		ep->peer_srqn = xrc_info.conn_data;
		fi_ibv_ep_ini_conn_done(ep, xrc_info.conn_data,
					xrc_info.conn_param.qp_num);
		fi_ibv_eq_skip_xrc_cm_data(&priv_data, &priv_datalen);
		fi_ibv_save_priv_data(ep, priv_data, priv_datalen);
	} else {
		fi_ibv_ep_tgt_conn_done(ep);
		ret = fi_ibv_connect_xrc(ep, NULL, FI_IBV_RECIP_CONN, &cm_data,
					 sizeof(cm_data));
		if (ret) {
			fi_ibv_prev_xrc_conn_state(ep);
			rdma_disconnect(ep->tgt_id);
			goto err;
		}
	}
err:
	entry->info = NULL;
	/* Event is handled internally and not passed to the application */
	return -FI_EAGAIN;
}

static size_t
fi_ibv_eq_xrc_recip_conn_event(struct fi_ibv_eq *eq,
			       struct fi_ibv_xrc_ep *ep,
			       struct rdma_cm_event *cma_event,
			       struct fi_eq_cm_entry *entry, size_t len)
{
	fid_t fid = cma_event->id->context;
	struct fi_ibv_xrc_conn_info xrc_info;
	int ret;

	fi_ibv_next_xrc_conn_state(ep);
	VERBS_DBG(FI_LOG_EP_CTRL, "EP %p RECIPROCAL CONNECTION DONE state %d\n",
		  ep, ep->conn_state);

	/* If this is the reciprocal active side notification */
	if (cma_event->param.conn.private_data) {
		ret = fi_ibv_eq_set_xrc_info(cma_event, &xrc_info);
		if (ret) {
			VERBS_WARN(FI_LOG_EP_CTRL,
				   "Reciprocal connection protocol mismatch\n");
			eq->err.err = -ret;
			eq->err.prov_errno = ret;
			eq->err.fid = fid;
			return -FI_EAVAIL;
		}

		ep->peer_srqn = xrc_info.conn_data;
		fi_ibv_ep_ini_conn_done(ep, xrc_info.conn_data,
					xrc_info.conn_param.qp_num);
	} else {
		fi_ibv_ep_tgt_conn_done(ep);
	}

	/* The internal reciprocal XRC connection has completed. Return the
	 * CONNECTED event application data associated with the original
	 * connection. */
	entry->fid = fid;
	len = fi_ibv_eq_copy_event_data(entry, len,
					ep->conn_setup->event_data,
					ep->conn_setup->event_len);
	entry->info = NULL;
	return sizeof(*entry) + len;
}

/* Caller must hold eq:lock */
static int
fi_ibv_eq_xrc_rej_event(struct fi_ibv_eq *eq, struct rdma_cm_event *cma_event)
{
	struct fi_ibv_xrc_ep *ep;
	fid_t fid = cma_event->id->context;
	struct fi_ibv_xrc_conn_info xrc_info;
	enum fi_ibv_xrc_ep_conn_state state;

	ep = container_of(fid, struct fi_ibv_xrc_ep, base_ep.util_ep.ep_fid);
	if (ep->magic != VERBS_XRC_EP_MAGIC) {
		VERBS_WARN(FI_LOG_EP_CTRL,
			   "CM ID context not valid\n");
		return -FI_EAGAIN;
	}

	state = ep->conn_state;
	if (ep->base_ep.id != cma_event->id ||
	    (state != FI_IBV_XRC_ORIG_CONNECTING &&
	     state != FI_IBV_XRC_RECIP_CONNECTING)) {
		VERBS_WARN(FI_LOG_EP_CTRL,
			   "Stale/invalid CM reject %d received\n", cma_event->status);
		return -FI_EAGAIN;
	}

	/* If reject comes from remote provider peer */
	if (cma_event->status == FI_IBV_CM_REJ_CONSUMER_DEFINED) {
		if (cma_event->param.conn.private_data_len &&
		    fi_ibv_eq_set_xrc_info(cma_event, &xrc_info)) {
			VERBS_WARN(FI_LOG_EP_CTRL,
				   "CM REJ private data not valid\n");
			return -FI_EAGAIN;
		}

		fi_ibv_ep_ini_conn_rejected(ep);
		return FI_SUCCESS;
	}

	VERBS_WARN(FI_LOG_EP_CTRL, "Non-application generated CM Reject %d\n",
		   cma_event->status);
	if (cma_event->param.conn.private_data_len)
		VERBS_WARN(FI_LOG_EP_CTRL, "Unexpected CM Reject priv_data\n");

	fi_ibv_ep_ini_conn_rejected(ep);

	return state == FI_IBV_XRC_ORIG_CONNECTING ? FI_SUCCESS : -FI_EAGAIN;
}

/* Caller must hold eq:lock */                                                                                  
static inline int
fi_ibv_eq_xrc_cm_err_event(struct fi_ibv_eq *eq,
                           struct rdma_cm_event *cma_event, int *acked)
{
	struct fi_ibv_xrc_ep *ep;
	fid_t fid = cma_event->id->context;

	ep = container_of(fid, struct fi_ibv_xrc_ep, base_ep.util_ep.ep_fid);
	if (ep->magic != VERBS_XRC_EP_MAGIC) {
		VERBS_WARN(FI_LOG_EP_CTRL, "CM ID context invalid\n");
		return -FI_EAGAIN;
	}

	/* Connect errors can be reported on active or passive side, all other
	 * errors considered are reported on the active side only */
	if ((ep->base_ep.id != cma_event->id) &&
	    (cma_event->event == RDMA_CM_EVENT_CONNECT_ERROR &&
	     ep->tgt_id != cma_event->id)) {
		VERBS_WARN(FI_LOG_EP_CTRL, "CM error not valid for EP\n");
		return -FI_EAGAIN;
	}

	VERBS_WARN(FI_LOG_EP_CTRL, "CM error event %s, status %d\n",
		   rdma_event_str(cma_event->event), cma_event->status);
	if (ep->base_ep.info->src_addr)
		ofi_straddr_log(&fi_ibv_prov, FI_LOG_WARN, FI_LOG_EP_CTRL,
				"Src ", ep->base_ep.info->src_addr);
	if (ep->base_ep.info->dest_addr)
		ofi_straddr_log(&fi_ibv_prov, FI_LOG_WARN, FI_LOG_EP_CTRL,
				"Dest ", ep->base_ep.info->dest_addr);
        ep->conn_state = FI_IBV_XRC_ERROR;
        return FI_SUCCESS;
}

/* Caller must hold eq:lock */
static inline int
fi_ibv_eq_xrc_connected_event(struct fi_ibv_eq *eq,
			      struct rdma_cm_event *cma_event,
			      struct fi_eq_cm_entry *entry, size_t len,
			      int *acked)
{
	struct fi_ibv_xrc_ep *ep;
	fid_t fid = cma_event->id->context;
	int ret;

	ep = container_of(fid, struct fi_ibv_xrc_ep, base_ep.util_ep.ep_fid);

	assert(ep->conn_state == FI_IBV_XRC_ORIG_CONNECTING ||
	       ep->conn_state == FI_IBV_XRC_RECIP_CONNECTING);

	if (ep->conn_state == FI_IBV_XRC_ORIG_CONNECTING)
		return fi_ibv_eq_xrc_conn_event(ep, cma_event, entry);

	ret = fi_ibv_eq_xrc_recip_conn_event(eq, ep, cma_event, entry, len);

	/* Bidirectional connection setup is complete, disconnect RDMA CM
	 * ID(s) and release shared QP reservations/hardware resources
	 * that were needed for shared connection setup only. */
	*acked = 1;
	rdma_ack_cm_event(cma_event);
	fi_ibv_free_xrc_conn_setup(ep, 1);

	return ret;
}

/* Caller must hold eq:lock */
static inline void
fi_ibv_eq_xrc_timewait_event(struct fi_ibv_eq *eq,
			     struct rdma_cm_event *cma_event, int *acked)
{
	fid_t fid = cma_event->id->context;
	struct fi_ibv_xrc_ep *ep = container_of(fid, struct fi_ibv_xrc_ep,
						base_ep.util_ep.ep_fid);
	assert(ep->magic == VERBS_XRC_EP_MAGIC);
	assert(ep->conn_setup);

	if (cma_event->id == ep->tgt_id && ep->conn_setup->rsvd_tgt_qpn) {
		*acked = 1;
		rdma_ack_cm_event(cma_event);
		ibv_destroy_qp(ep->conn_setup->rsvd_tgt_qpn);
		ep->conn_setup->rsvd_tgt_qpn = NULL;
		rdma_destroy_id(ep->tgt_id);
		ep->tgt_id = NULL;
	} else if (cma_event->id == ep->base_ep.id &&
		   ep->conn_setup->rsvd_ini_qpn) {
		*acked = 1;
		rdma_ack_cm_event(cma_event);
		ibv_destroy_qp(ep->conn_setup->rsvd_ini_qpn);
		ep->conn_setup->rsvd_ini_qpn = NULL;
		rdma_destroy_id(ep->base_ep.id);
		ep->base_ep.id = NULL;
	}
	if (!ep->conn_setup->rsvd_ini_qpn && !ep->conn_setup->rsvd_tgt_qpn)
		fi_ibv_free_xrc_conn_setup(ep, 0);
}

/* Caller must hold eq:lock */
static inline void
fi_ibv_eq_xrc_disconnect_event(struct fi_ibv_eq *eq,
			       struct rdma_cm_event *cma_event, int *acked)
{
	fid_t fid = cma_event->id->context;
	struct fi_ibv_xrc_ep *ep = container_of(fid, struct fi_ibv_xrc_ep,
						base_ep.util_ep.ep_fid);
	assert(ep->magic == VERBS_XRC_EP_MAGIC);

	if (ep->conn_setup && cma_event->id == ep->base_ep.id &&
	    ep->conn_setup->rsvd_ini_qpn) {
		*acked = 1;
		rdma_ack_cm_event(cma_event);
		rdma_disconnect(ep->base_ep.id);
		ep->conn_setup->ini_connected = 0;
	}
}

static ssize_t
fi_ibv_eq_cm_process_event(struct fi_ibv_eq *eq,
	struct rdma_cm_event *cma_event, uint32_t *event,
	struct fi_eq_cm_entry *entry, size_t len, int *acked)
{
	const struct fi_ibv_cm_data_hdr *cm_hdr;
	size_t datalen = 0;
	size_t priv_datalen = cma_event->param.conn.private_data_len;
	const void *priv_data = cma_event->param.conn.private_data;
	int ret;
	fid_t fid = cma_event->id->context;
	struct fi_ibv_pep *pep =
		container_of(fid, struct fi_ibv_pep, pep_fid);
	struct fi_ibv_ep *ep;
	struct fi_ibv_xrc_ep *xrc_ep;
	struct fi_ibv_domain *domain;

	*acked = 0;

	switch (cma_event->event) {
	case RDMA_CM_EVENT_ROUTE_RESOLVED:
		ep = container_of(fid, struct fi_ibv_ep, util_ep.ep_fid);
		if (rdma_connect(ep->id, &ep->conn_param)) {
			ret = -errno;
			FI_WARN(&fi_ibv_prov, FI_LOG_EP_CTRL,
				"rdma_connect failed: %s (%d)\n",
				strerror(-ret), -ret);
			if (fi_ibv_is_xrc(ep->info)) {
				xrc_ep = container_of(fid, struct fi_ibv_xrc_ep,
						      base_ep.util_ep.ep_fid);
				domain = fi_ibv_ep_to_domain(ep);
				fastlock_acquire(&domain->xrc.ini_mgmt_lock);
				fi_ibv_put_shared_ini_conn(xrc_ep);
				fastlock_release(&domain->xrc.ini_mgmt_lock);
			}
			return ret;
		}
		return -FI_EAGAIN;
	case RDMA_CM_EVENT_CONNECT_REQUEST:
		*event = FI_CONNREQ;

		ret = fi_ibv_eq_cm_getinfo(cma_event, pep->info, &entry->info);
		if (ret) {
			fastlock_acquire(&eq->lock);
			rdma_destroy_id(cma_event->id);
			eq->err.err = -ret;
			eq->err.prov_errno = ret;
			goto err;
		}

		if (fi_ibv_is_xrc(entry->info)) {
			ret = fi_ibv_eq_xrc_connreq_event(eq, entry, &priv_data,
							  &priv_datalen);
			if (ret == -FI_EAGAIN)
				return ret;
		}
		break;
	case RDMA_CM_EVENT_ESTABLISHED:
		*event = FI_CONNECTED;

		if (cma_event->id->qp &&
		    cma_event->id->qp->context->device->transport_type !=
		    IBV_TRANSPORT_IWARP) {
			ret = fi_ibv_set_rnr_timer(cma_event->id->qp);
			if (ret)
				return ret;
		}
		ep = container_of(fid, struct fi_ibv_ep, util_ep.ep_fid);
		if (fi_ibv_is_xrc(ep->info)) {
			fastlock_acquire(&eq->lock);
			ret = fi_ibv_eq_xrc_connected_event(eq, cma_event,
							    entry, len, acked);
			fastlock_release(&eq->lock);
			return ret;
		}
		entry->info = NULL;
		break;
	case RDMA_CM_EVENT_DISCONNECTED:
		ep = container_of(fid, struct fi_ibv_ep, util_ep.ep_fid);
		if (fi_ibv_is_xrc(ep->info)) {
			fastlock_acquire(&eq->lock);
			fi_ibv_eq_xrc_disconnect_event(eq, cma_event, acked);
			fastlock_release(&eq->lock);
			return -FI_EAGAIN;
		}
		*event = FI_SHUTDOWN;
		entry->info = NULL;
		break;
	case RDMA_CM_EVENT_TIMEWAIT_EXIT:
		ep = container_of(fid, struct fi_ibv_ep, util_ep.ep_fid);
		if (fi_ibv_is_xrc(ep->info)) {
			fastlock_acquire(&eq->lock);
			fi_ibv_eq_xrc_timewait_event(eq, cma_event, acked);
			fastlock_release(&eq->lock);
		}
		return -FI_EAGAIN;
	case RDMA_CM_EVENT_ADDR_ERROR:
	case RDMA_CM_EVENT_ROUTE_ERROR:
	case RDMA_CM_EVENT_CONNECT_ERROR:
	case RDMA_CM_EVENT_UNREACHABLE:
		fastlock_acquire(&eq->lock);
		ep = container_of(fid, struct fi_ibv_ep, util_ep.ep_fid);
		assert(ep->info);
		if (fi_ibv_is_xrc(ep->info)) {
			ret = fi_ibv_eq_xrc_cm_err_event(eq, cma_event, acked);
			if (ret == -FI_EAGAIN) {
				fastlock_release(&eq->lock);
				return ret;
			}
		}
		eq->err.err = ETIMEDOUT;
		eq->err.prov_errno = -cma_event->status;
		if (eq->err.err_data) {
			free(eq->err.err_data);
			eq->err.err_data = NULL;
			eq->err.err_data_size = 0;
		}
		goto err;
	case RDMA_CM_EVENT_REJECTED:
		ep = container_of(fid, struct fi_ibv_ep, util_ep.ep_fid);
		if (fi_ibv_is_xrc(ep->info)) {
			fastlock_acquire(&eq->lock);
			ret = fi_ibv_eq_xrc_rej_event(eq, cma_event);
			fastlock_release(&eq->lock);
			if (ret == -FI_EAGAIN)
				return ret;
			fi_ibv_eq_skip_xrc_cm_data(&priv_data, &priv_datalen);
		}
		fastlock_acquire(&eq->lock);
		eq->err.err = ECONNREFUSED;
		eq->err.prov_errno = -cma_event->status;
		if (eq->err.err_data) {
			free(eq->err.err_data);
			eq->err.err_data = NULL;
			eq->err.err_data_size = 0;
		}
		if (priv_datalen) {
			cm_hdr = priv_data;
			eq->err.err_data = calloc(1, cm_hdr->size);
			assert(eq->err.err_data);
			memcpy(eq->err.err_data, cm_hdr->data,
			       cm_hdr->size);
			eq->err.err_data_size = cm_hdr->size;
		}
		goto err;
	case RDMA_CM_EVENT_DEVICE_REMOVAL:
		fastlock_acquire(&eq->lock);
		eq->err.err = ENODEV;
		goto err;
	case RDMA_CM_EVENT_ADDR_CHANGE:
		fastlock_acquire(&eq->lock);
		eq->err.err = EADDRNOTAVAIL;
		goto err;
	default:
		return 0;
	}

	entry->fid = fid;

	/* rdmacm has no way to track how much data is sent by peer */
	if (priv_datalen)
		datalen = fi_ibv_eq_copy_event_data(entry, len, priv_data,
						    priv_datalen);
	return sizeof(*entry) + datalen;
err:
	eq->err.fid = fid;
	fastlock_release(&eq->lock);
	return -FI_EAVAIL;
}

int fi_ibv_eq_trywait(struct fi_ibv_eq *eq)
{
	int ret;
	fastlock_acquire(&eq->lock);
	ret = dlistfd_empty(&eq->list_head);
	fastlock_release(&eq->lock);
	return ret ? 0 : -FI_EAGAIN;
}

ssize_t fi_ibv_eq_write_event(struct fi_ibv_eq *eq, uint32_t event,
			      const void *buf, size_t len)
{
	struct fi_ibv_eq_entry *entry;

	entry = calloc(1, sizeof(struct fi_ibv_eq_entry) + len);
	if (!entry) {
		VERBS_WARN(FI_LOG_EP_CTRL, "Unable to allocate EQ entry\n");
		return -FI_ENOMEM;
	}

	entry->event = event;
	entry->len = len;
	memcpy(entry->eq_entry, buf, len);

	fastlock_acquire(&eq->lock);
	dlistfd_insert_tail(&entry->item, &eq->list_head);
	fastlock_release(&eq->lock);

	return len;
}

static ssize_t fi_ibv_eq_write(struct fid_eq *eq_fid, uint32_t event,
			       const void *buf, size_t len, uint64_t flags)
{
	struct fi_ibv_eq *eq;

	eq = container_of(eq_fid, struct fi_ibv_eq, eq_fid.fid);
	if (!(eq->flags & FI_WRITE))
		return -FI_EINVAL;

	return fi_ibv_eq_write_event(eq, event, buf, len);
}

static size_t fi_ibv_eq_read_event(struct fi_ibv_eq *eq, uint32_t *event,
		void *buf, size_t len, uint64_t flags)
{
	struct fi_ibv_eq_entry *entry;
	ssize_t ret = 0;

	fastlock_acquire(&eq->lock);

	if (eq->err.err) {
		ret = -FI_EAVAIL;
		goto out;
	}

	if (dlistfd_empty(&eq->list_head))
		goto out;

	entry = container_of(eq->list_head.list.next, struct fi_ibv_eq_entry, item);
	if (entry->len > len) {
		ret = -FI_ETOOSMALL;
		goto out;
	}

	ret = entry->len;
	*event = entry->event;
	memcpy(buf, entry->eq_entry, entry->len);

	if (!(flags & FI_PEEK)) {
		dlistfd_remove(eq->list_head.list.next, &eq->list_head);
		free(entry);
	}

out:
	fastlock_release(&eq->lock);
	return ret;
}

static ssize_t
fi_ibv_eq_read(struct fid_eq *eq_fid, uint32_t *event,
	       void *buf, size_t len, uint64_t flags)
{
	struct fi_ibv_eq *eq;
	struct rdma_cm_event *cma_event;
	ssize_t ret = 0;
	int acked;

	eq = container_of(eq_fid, struct fi_ibv_eq, eq_fid.fid);

	if ((ret = fi_ibv_eq_read_event(eq, event, buf, len, flags)))
		return ret;

	if (eq->channel) {
		fastlock_acquire(&eq->lock);
		ret = rdma_get_cm_event(eq->channel, &cma_event);
		fastlock_release(&eq->lock);

		if (ret)
			return -errno;

		acked = 0;
		if (len < sizeof(struct fi_eq_cm_entry)) {
			ret = -FI_ETOOSMALL;
			goto ack;
		}

		ret = fi_ibv_eq_cm_process_event(eq, cma_event, event,
				(struct fi_eq_cm_entry *)buf, len, &acked);
		if (ret < 0)
			goto ack;

		if (flags & FI_PEEK)
			ret = fi_ibv_eq_write_event(eq, *event, buf, ret);
ack:
		if (!acked)
			rdma_ack_cm_event(cma_event);

		return ret;
	}

	return -FI_EAGAIN;
}

static ssize_t
fi_ibv_eq_sread(struct fid_eq *eq_fid, uint32_t *event,
		void *buf, size_t len, int timeout, uint64_t flags)
{
	struct fi_ibv_eq *eq;
	struct epoll_event events[2];
	ssize_t ret;

	eq = container_of(eq_fid, struct fi_ibv_eq, eq_fid.fid);

	while (1) {
		ret = fi_ibv_eq_read(eq_fid, event, buf, len, flags);
		if (ret && (ret != -FI_EAGAIN))
			return ret;

		ret = epoll_wait(eq->epfd, events, 2, timeout);
		if (ret == 0)
			return -FI_EAGAIN;
		else if (ret < 0)
			return -errno;
	};
}

static const char *
fi_ibv_eq_strerror(struct fid_eq *eq, int prov_errno, const void *err_data,
		   char *buf, size_t len)
{
	if (buf && len)
		strncpy(buf, strerror(prov_errno), len);
	return strerror(prov_errno);
}

static struct fi_ops_eq fi_ibv_eq_ops = {
	.size = sizeof(struct fi_ops_eq),
	.read = fi_ibv_eq_read,
	.readerr = fi_ibv_eq_readerr,
	.write = fi_ibv_eq_write,
	.sread = fi_ibv_eq_sread,
	.strerror = fi_ibv_eq_strerror
};

static int fi_ibv_eq_control(fid_t fid, int command, void *arg)
{
	struct fi_ibv_eq *eq;
	int ret = 0;

	eq = container_of(fid, struct fi_ibv_eq, eq_fid.fid);
	switch (command) {
	case FI_GETWAIT:
		if (!eq->epfd) {
			ret = -FI_ENODATA;
			break;
		}
		*(int *) arg = eq->epfd;
		break;
	default:
		ret = -FI_ENOSYS;
		break;
	}

	return ret;
}

static int fi_ibv_eq_close(fid_t fid)
{
	struct fi_ibv_eq *eq;
	struct fi_ibv_eq_entry *entry;

	eq = container_of(fid, struct fi_ibv_eq, eq_fid.fid);
	/* TODO: use util code, if possible, and add ref counting */

	free(eq->err.err_data);

	if (eq->channel)
		rdma_destroy_event_channel(eq->channel);

	close(eq->epfd);

	while (!dlistfd_empty(&eq->list_head)) {
		entry = container_of(eq->list_head.list.next,
				     struct fi_ibv_eq_entry, item);
		dlistfd_remove(eq->list_head.list.next, &eq->list_head);
		free(entry);
	}

	dlistfd_head_free(&eq->list_head);

	ofi_idx_reset(eq->xrc.conn_key_map);
	free(eq->xrc.conn_key_map);
	fastlock_destroy(&eq->lock);
	free(eq);

	return 0;
}

static struct fi_ops fi_ibv_eq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_eq_close,
	.bind = fi_no_bind,
	.control = fi_ibv_eq_control,
	.ops_open = fi_no_ops_open,
};

int fi_ibv_eq_open(struct fid_fabric *fabric, struct fi_eq_attr *attr,
		   struct fid_eq **eq, void *context)
{
	struct fi_ibv_eq *_eq;
	struct epoll_event event;
	int ret;

	_eq = calloc(1, sizeof *_eq);
	if (!_eq)
		return -ENOMEM;

	_eq->fab = container_of(fabric, struct fi_ibv_fabric,
				util_fabric.fabric_fid);

	ofi_key_idx_init(&_eq->xrc.conn_key_idx, VERBS_CONN_TAG_INDEX_BITS);
	_eq->xrc.conn_key_map = calloc(1, sizeof(*_eq->xrc.conn_key_map));
	if (!_eq->xrc.conn_key_map) {
		ret = -ENOMEM;
		goto err0;
	}
	fastlock_init(&_eq->lock);
	ret = dlistfd_head_init(&_eq->list_head);
	if (ret) {
		VERBS_INFO(FI_LOG_EQ, "Unable to initialize dlistfd\n");
		goto err1;
	}

	_eq->epfd = epoll_create1(0);
	if (_eq->epfd < 0) {
		ret = -errno;
		goto err2;
	}

	memset(&event, 0, sizeof(event));
	event.events = EPOLLIN;

	if (epoll_ctl(_eq->epfd, EPOLL_CTL_ADD,
		      _eq->list_head.signal.fd[FI_READ_FD], &event)) {
		ret = -errno;
		goto err3;
	}

	switch (attr->wait_obj) {
	case FI_WAIT_NONE:
	case FI_WAIT_UNSPEC:
	case FI_WAIT_FD:
		_eq->channel = rdma_create_event_channel();
		if (!_eq->channel) {
			ret = -errno;
			goto err3;
		}

		ret = fi_fd_nonblock(_eq->channel->fd);
		if (ret)
			goto err4;

		if (epoll_ctl(_eq->epfd, EPOLL_CTL_ADD, _eq->channel->fd, &event)) {
			ret = -errno;
			goto err4;
		}

		break;
	default:
		ret = -FI_ENOSYS;
		goto err1;
	}

	_eq->flags = attr->flags;
	_eq->eq_fid.fid.fclass = FI_CLASS_EQ;
	_eq->eq_fid.fid.context = context;
	_eq->eq_fid.fid.ops = &fi_ibv_eq_fi_ops;
	_eq->eq_fid.ops = &fi_ibv_eq_ops;

	*eq = &_eq->eq_fid;
	return 0;
err4:
	if (_eq->channel)
		rdma_destroy_event_channel(_eq->channel);
err3:
	close(_eq->epfd);
err2:
	dlistfd_head_free(&_eq->list_head);
err1:
	fastlock_destroy(&_eq->lock);
	free(_eq->xrc.conn_key_map);
err0:
	free(_eq);
	return ret;
}

