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

/* XRC SIDR connection map RBTree key */
struct fi_ibv_sidr_conn_key {
	struct sockaddr		*addr;
	uint16_t		pep_port;
	bool			recip;
};

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
}

/* Caller must hold eq:lock */
void fi_ibv_eq_clear_xrc_conn_tag(struct fi_ibv_xrc_ep *ep)
{
	struct fi_ibv_eq *eq = ep->base_ep.eq;
	int index;

	assert(ep->conn_setup);
	if (ep->conn_setup->conn_tag == VERBS_CONN_TAG_INVALID)
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
	info->tgt_qpn = ntohl(remote->tgt_qpn);
	info->peer_srqn = ntohl(remote->srqn);
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
			VERBS_WARN(FI_LOG_EQ, "passive endpoint domain: %s does"
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

	ret = fi_ibv_get_matching_info(hints->fabric_attr->api_version, hints,
				       info, fi_ibv_util_prov.info, 0);
	if (ret)
		goto err1;

	assert(!(*info)->dest_addr);

	ofi_alter_info(*info, hints, hints->fabric_attr->api_version);
	fi_ibv_alter_info(hints, *info);

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

static inline void fi_ibv_set_sidr_conn_key(struct sockaddr *addr,
					    uint16_t pep_port, bool recip,
					    struct fi_ibv_sidr_conn_key *key)
{
	key->addr = addr;
	key->pep_port = pep_port;
	key->recip = recip;
}

static int fi_ibv_sidr_conn_compare(struct ofi_rbmap *map,
				    void *key, void *data)
{
	struct fi_ibv_sidr_conn_key *_key = key;
	struct fi_ibv_xrc_ep *ep = data;
	int ret;

	assert(_key->addr->sa_family ==
	       ofi_sa_family(ep->base_ep.info->dest_addr));

	/* The interface address and the passive endpoint port define
	 * the unique connection to a peer */
	switch(_key->addr->sa_family) {
	case AF_INET:
		ret = memcmp(&ofi_sin_addr(_key->addr),
			     &ofi_sin_addr(ep->base_ep.info->dest_addr),
			     sizeof(ofi_sin_addr(_key->addr)));
		break;
	case AF_INET6:
		ret = memcmp(&ofi_sin6_addr(_key->addr),
			     &ofi_sin6_addr(ep->base_ep.info->dest_addr),
			     sizeof(ofi_sin6_addr(_key->addr)));
		break;
	default:
		VERBS_WARN(FI_LOG_EP_CTRL, "Unsuuported address format\n");
		assert(0);
		ret = -FI_EINVAL;
	}

	if (ret)
		return ret;

	if (_key->pep_port != ep->remote_pep_port)
		return _key->pep_port < ep->remote_pep_port ? -1 : 1;

	return _key->recip < ep->recip_accept ?
		-1 : _key->recip > ep->recip_accept;
}

/* Caller must hold eq:lock */
struct fi_ibv_xrc_ep *fi_ibv_eq_get_sidr_conn(struct fi_ibv_eq *eq,
					      struct sockaddr *peer,
					      uint16_t pep_port, bool recip)
{
	struct ofi_rbnode *node;
	struct fi_ibv_sidr_conn_key key;

	fi_ibv_set_sidr_conn_key(peer, pep_port, recip, &key);
	node = ofi_rbmap_find(&eq->xrc.sidr_conn_rbmap, &key);
	if (OFI_LIKELY(!node))
		return NULL;

	return (struct fi_ibv_xrc_ep *) node->data;
}

/* Caller must hold eq:lock */
int fi_ibv_eq_add_sidr_conn(struct fi_ibv_xrc_ep *ep,
			    void *param_data, size_t param_len)
{
	int ret;
	struct fi_ibv_sidr_conn_key key;

	assert(!ep->accept_param_data);
	assert(param_len);
	assert(ep->tgt_id && ep->tgt_id->ps == RDMA_PS_UDP);

	fi_ibv_set_sidr_conn_key(ep->base_ep.info->dest_addr,
				 ep->remote_pep_port, ep->recip_accept, &key);
	ep->accept_param_data = calloc(1, param_len);
	if (!ep->accept_param_data) {
		VERBS_WARN(FI_LOG_EP_CTRL,
			   "SIDR alloc conn param memory failure\n");
		return -FI_ENOMEM;
	}
	memcpy(ep->accept_param_data, param_data, param_len);
	ep->accept_param_len = param_len;

	ret = ofi_rbmap_insert(&ep->base_ep.eq->xrc.sidr_conn_rbmap,
			       &key, (void *) ep, &ep->conn_map_node);
	assert(ret != -FI_EALREADY);
	if (OFI_UNLIKELY(ret)) {
		VERBS_WARN(FI_LOG_EP_CTRL,
			   "SIDR conn map entry insert error %d\n", ret);
		free(ep->accept_param_data);
		ep->accept_param_data = NULL;
		return ret;
	}

	return FI_SUCCESS;
}

/* Caller must hold eq:lock */
void fi_ibv_eq_remove_sidr_conn(struct fi_ibv_xrc_ep *ep)
{
	assert(ep->conn_map_node);

	ofi_rbmap_delete(&ep->base_ep.eq->xrc.sidr_conn_rbmap,
			 ep->conn_map_node);
	ep->conn_map_node = NULL;
	free(ep->accept_param_data);
	ep->accept_param_data = NULL;
}

static int
fi_ibv_eq_accept_recip_conn(struct fi_ibv_xrc_ep *ep,
			    struct fi_eq_cm_entry *entry, size_t len,
			    uint32_t *event, struct rdma_cm_event *cma_event,
			    int *acked)
{
	struct fi_ibv_xrc_cm_data cm_data;
	int ret;

	assert(ep->conn_state == FI_IBV_XRC_ORIG_CONNECTED);

	ret = fi_ibv_accept_xrc(ep, FI_IBV_RECIP_CONN, &cm_data,
				sizeof(cm_data));
	if (ret) {
		VERBS_WARN(FI_LOG_EP_CTRL,
			   "Reciprocal XRC Accept failed %d\n", ret);
		return ret;
	}

	/* SIDR based shared reciprocal connections are complete at
	 * this point, generate the connection established event. */
	if (ep->tgt_id->ps == RDMA_PS_UDP) {
		fi_ibv_next_xrc_conn_state(ep);
		fi_ibv_ep_tgt_conn_done(ep);
		entry->fid = &ep->base_ep.util_ep.ep_fid.fid;
		*event = FI_CONNECTED;
		len = fi_ibv_eq_copy_event_data(entry, len,
						ep->conn_setup->event_data,
						ep->conn_setup->event_len);
		*acked = 1;
		rdma_ack_cm_event(cma_event);
		fi_ibv_free_xrc_conn_setup(ep, 1);

		return sizeof(*entry) + len;
	}

	/* Event is handled internally and not passed to the application */
	return -FI_EAGAIN;
}

static int
fi_ibv_eq_xrc_connreq_event(struct fi_ibv_eq *eq, struct fi_eq_cm_entry *entry,
			    size_t len, uint32_t *event,
			    struct rdma_cm_event *cma_event, int *acked,
			    const void **priv_data, size_t *priv_datalen)
{
	struct fi_ibv_connreq *connreq = container_of(entry->info->handle,
						struct fi_ibv_connreq, handle);
	struct fi_ibv_xrc_ep *ep;
	int ret;

	/*
	 * If this is a retransmitted SIDR request for a previously accepted
	 * connection then the shared SIDR response message was lost and must
	 * be retransmitted. Note that a lost SIDR reject response message will
	 * be rejected again by the application.
	 */
	assert(entry->info->dest_addr);
	if (cma_event->id->ps == RDMA_PS_UDP) {
		ep = fi_ibv_eq_get_sidr_conn(eq, entry->info->dest_addr,
					     connreq->xrc.port,
					     connreq->xrc.is_reciprocal);
		if (ep) {
			VERBS_DBG(FI_LOG_EP_CTRL,
				  "SIDR %s request retry received\n",
				  connreq->xrc.is_reciprocal ?
				  "reciprocal" : "original");
			ret = fi_ibv_resend_shared_accept_xrc(ep, connreq,
							      cma_event->id);
			if (ret)
				VERBS_WARN(FI_LOG_EP_CTRL,
					   "SIDR accept resend failure %d\n",
					   -errno);
			rdma_destroy_id(cma_event->id);
			return -FI_EAGAIN;
		}
	}

	if (!connreq->xrc.is_reciprocal) {
		fi_ibv_eq_skip_xrc_cm_data(priv_data, priv_datalen);
		return FI_SUCCESS;
	}

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
		return -FI_EAGAIN;
	}
	ep->recip_req_received = 1;

	assert(ep->conn_state == FI_IBV_XRC_ORIG_CONNECTED ||
	       ep->conn_state == FI_IBV_XRC_ORIG_CONNECTING);

	ep->tgt_id = connreq->id;
	ep->tgt_id->context = &ep->base_ep.util_ep.ep_fid.fid;
	ep->base_ep.info->handle = entry->info->handle;

	ret = rdma_migrate_id(ep->tgt_id, ep->base_ep.eq->channel);
	if (ret) {
		VERBS_WARN(FI_LOG_EP_CTRL, "Could not migrate CM ID\n");
		goto send_reject;
	}

	/* If the initial connection has completed proceed with accepting
	 * the reciprocal; otherwise wait until it has before proceeding */
	if (ep->conn_state == FI_IBV_XRC_ORIG_CONNECTED)
		return fi_ibv_eq_accept_recip_conn(ep, entry, len, event,
						   cma_event, acked);

	return -FI_EAGAIN;

send_reject:
	if (rdma_reject(connreq->id, *priv_data, *priv_datalen))
		VERBS_WARN(FI_LOG_EP_CTRL, "rdma_reject %d\n", -errno);

	return -FI_EAGAIN;
}

static void
fi_ibv_eq_xrc_establish(struct rdma_cm_event *cma_event)
{
	/* For newer rdma-core, active side  must complete the
	 * connect if rdma_cm is not managing the QP */
	if (cma_event->event == RDMA_CM_EVENT_CONNECT_RESPONSE &&
	    !cma_event->id->qp)
		rdma_establish(cma_event->id);
}

static int
fi_ibv_eq_xrc_conn_event(struct fi_ibv_xrc_ep *ep,
			 struct rdma_cm_event *cma_event, int *acked,
			 struct fi_eq_cm_entry *entry, size_t len,
			 uint32_t *event)
{
	struct fi_ibv_xrc_conn_info xrc_info;
	struct fi_ibv_xrc_cm_data cm_data;
	const void *priv_data = cma_event->param.conn.private_data;
	size_t priv_datalen = cma_event->param.conn.private_data_len;
	int ret;

	VERBS_DBG(FI_LOG_EP_CTRL, "EP %p INITIAL CONNECTION DONE state %d, ps %d\n",
		  ep, ep->conn_state, cma_event->id->ps);
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
		ep->peer_srqn = xrc_info.peer_srqn;
		fi_ibv_eq_skip_xrc_cm_data(&priv_data, &priv_datalen);
		fi_ibv_save_priv_data(ep, priv_data, priv_datalen);
		fi_ibv_ep_ini_conn_done(ep, xrc_info.conn_param.qp_num);
		fi_ibv_eq_xrc_establish(cma_event);

		/* If we have received the reciprocal connect request,
		 * process it now */
		if (ep->recip_req_received)
			return fi_ibv_eq_accept_recip_conn(ep, entry,
							   len, event,
							   cma_event, acked);
	} else {
		fi_ibv_ep_tgt_conn_done(ep);
		ret = fi_ibv_connect_xrc(ep, NULL, FI_IBV_RECIP_CONN, &cm_data,
					 sizeof(cm_data));
		if (ret) {
			fi_ibv_prev_xrc_conn_state(ep);
			ep->tgt_id->qp = NULL;
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

		ep->peer_srqn = xrc_info.peer_srqn;
		fi_ibv_ep_ini_conn_done(ep, xrc_info.conn_param.qp_num);
		fi_ibv_eq_xrc_establish(cma_event);
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
	if (cma_event->status == FI_IBV_CM_REJ_CONSUMER_DEFINED ||
	    cma_event->status == FI_IBV_CM_REJ_SIDR_CONSUMER_DEFINED) {
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
                           struct rdma_cm_event *cma_event)
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
			      struct rdma_cm_event *cma_event, int *acked,
			      struct fi_eq_cm_entry *entry, size_t len,
			      uint32_t *event)
{
	struct fi_ibv_xrc_ep *ep;
	fid_t fid = cma_event->id->context;
	int ret;

	ep = container_of(fid, struct fi_ibv_xrc_ep, base_ep.util_ep.ep_fid);

	assert(ep->conn_state == FI_IBV_XRC_ORIG_CONNECTING ||
	       ep->conn_state == FI_IBV_XRC_RECIP_CONNECTING);

	if (ep->conn_state == FI_IBV_XRC_ORIG_CONNECTING)
		return fi_ibv_eq_xrc_conn_event(ep, cma_event, acked,
						entry, len, event);

	ret = fi_ibv_eq_xrc_recip_conn_event(eq, ep, cma_event, entry, len);

	/* Bidirectional connection setup is complete, release RDMA CM ID
	 * resources. */
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

	if (cma_event->id == ep->tgt_id) {
		*acked = 1;
		rdma_ack_cm_event(cma_event);
		rdma_destroy_id(ep->tgt_id);
		ep->tgt_id = NULL;
	} else if (cma_event->id == ep->base_ep.id) {
		*acked = 1;
		rdma_ack_cm_event(cma_event);
		rdma_destroy_id(ep->base_ep.id);
		ep->base_ep.id = NULL;
	}
	if (!ep->base_ep.id && !ep->tgt_id)
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

	if (ep->conn_setup && cma_event->id == ep->base_ep.id) {
		*acked = 1;
		rdma_ack_cm_event(cma_event);
		rdma_disconnect(ep->base_ep.id);
	}
}

static ssize_t
fi_ibv_eq_cm_process_event(struct fi_ibv_eq *eq,
	struct rdma_cm_event *cma_event, uint32_t *event,
	struct fi_eq_cm_entry *entry, size_t len)
{
	const struct fi_ibv_cm_data_hdr *cm_hdr;
	size_t datalen = 0;
	size_t priv_datalen = cma_event->param.conn.private_data_len;
	const void *priv_data = cma_event->param.conn.private_data;
	int ret, acked = 0;;
	fid_t fid = cma_event->id->context;
	struct fi_ibv_pep *pep =
		container_of(fid, struct fi_ibv_pep, pep_fid);
	struct fi_ibv_ep *ep;
	struct fi_ibv_xrc_ep *xrc_ep;

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
				fi_ibv_put_shared_ini_conn(xrc_ep);
			}
		} else {
			ret = -FI_EAGAIN;
		}
		goto ack;
	case RDMA_CM_EVENT_CONNECT_REQUEST:
		*event = FI_CONNREQ;

		ret = fi_ibv_eq_cm_getinfo(cma_event, pep->info, &entry->info);
		if (ret) {
			VERBS_WARN(FI_LOG_EP_CTRL,
				   "CM getinfo error %d\n", ret);
			rdma_destroy_id(cma_event->id);
			eq->err.err = -ret;
			eq->err.prov_errno = ret;
			goto err;
		}

		if (fi_ibv_is_xrc(entry->info)) {
			ret = fi_ibv_eq_xrc_connreq_event(eq, entry, len, event,
							  cma_event, &acked,
							  &priv_data, &priv_datalen);
			if (ret == -FI_EAGAIN || *event == FI_CONNECTED)
				goto ack;
		}
		break;
	case RDMA_CM_EVENT_CONNECT_RESPONSE:
	case RDMA_CM_EVENT_ESTABLISHED:
		*event = FI_CONNECTED;

		if (cma_event->id->qp &&
		    cma_event->id->qp->context->device->transport_type !=
		    IBV_TRANSPORT_IWARP) {
			ret = fi_ibv_set_rnr_timer(cma_event->id->qp);
			if (ret)
				goto ack;
		}
		ep = container_of(fid, struct fi_ibv_ep, util_ep.ep_fid);
		if (fi_ibv_is_xrc(ep->info)) {
			ret = fi_ibv_eq_xrc_connected_event(eq, cma_event,
							    &acked, entry, len,
							    event);
			goto ack;
		}
		entry->info = NULL;
		break;
	case RDMA_CM_EVENT_DISCONNECTED:
		ep = container_of(fid, struct fi_ibv_ep, util_ep.ep_fid);
		if (fi_ibv_is_xrc(ep->info)) {
			fi_ibv_eq_xrc_disconnect_event(eq, cma_event, &acked);
			ret = -FI_EAGAIN;
			goto ack;
		}
		*event = FI_SHUTDOWN;
		entry->info = NULL;
		break;
	case RDMA_CM_EVENT_TIMEWAIT_EXIT:
		ep = container_of(fid, struct fi_ibv_ep, util_ep.ep_fid);
		if (fi_ibv_is_xrc(ep->info))
			fi_ibv_eq_xrc_timewait_event(eq, cma_event, &acked);
		ret = -FI_EAGAIN;
		goto ack;
	case RDMA_CM_EVENT_ADDR_ERROR:
	case RDMA_CM_EVENT_ROUTE_ERROR:
	case RDMA_CM_EVENT_CONNECT_ERROR:
	case RDMA_CM_EVENT_UNREACHABLE:
		ep = container_of(fid, struct fi_ibv_ep, util_ep.ep_fid);
		assert(ep->info);
		if (fi_ibv_is_xrc(ep->info)) {
			/* SIDR Reject is reported as UNREACHABLE */
			if (cma_event->id->ps == RDMA_PS_UDP &&
			    cma_event->event == RDMA_CM_EVENT_UNREACHABLE)
				goto xrc_shared_reject;

			ret = fi_ibv_eq_xrc_cm_err_event(eq, cma_event);
			if (ret == -FI_EAGAIN)
				goto ack;
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
xrc_shared_reject:
			ret = fi_ibv_eq_xrc_rej_event(eq, cma_event);
			if (ret == -FI_EAGAIN)
				goto ack;
			fi_ibv_eq_skip_xrc_cm_data(&priv_data, &priv_datalen);
		}
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
		eq->err.err = ENODEV;
		goto err;
	case RDMA_CM_EVENT_ADDR_CHANGE:
		eq->err.err = EADDRNOTAVAIL;
		goto err;
	default:
		VERBS_WARN(FI_LOG_EP_CTRL, "unknown rdmacm event received: %d\n",
			   cma_event->event);
		ret = -FI_EAGAIN;
		goto ack;
	}

	entry->fid = fid;

	/* rdmacm has no way to track how much data is sent by peer */
	if (priv_datalen)
		datalen = fi_ibv_eq_copy_event_data(entry, len, priv_data,
						    priv_datalen);
	if (!acked)
		rdma_ack_cm_event(cma_event);
	return sizeof(*entry) + datalen;
err:
	ret = -FI_EAVAIL;
	eq->err.fid = fid;
ack:
	if (!acked)
		rdma_ack_cm_event(cma_event);
	return ret;
}

int fi_ibv_eq_trywait(struct fi_ibv_eq *eq)
{
	int ret;
	fastlock_acquire(&eq->lock);
	ret = dlistfd_empty(&eq->list_head);
	fastlock_release(&eq->lock);
	return ret ? 0 : -FI_EAGAIN;
}

int fi_ibv_eq_match_event(struct dlist_entry *item, const void *arg)
{
	struct fi_ibv_eq_entry *entry =
		container_of(item, struct fi_ibv_eq_entry, item);
	const struct fid *fid = arg;
	return entry->eq_entry->fid == fid;
}

/* Caller must hold eq->lock */
void fi_ibv_eq_remove_events(struct fi_ibv_eq *eq, struct fid *fid)
{
	struct dlist_entry *item;
	struct fi_ibv_eq_entry *entry;

	while ((item =
		dlistfd_remove_first_match(&eq->list_head,
					   fi_ibv_eq_match_event, fid))) {
		entry = container_of(item, struct fi_ibv_eq_entry, item);
		if (entry->event == FI_CONNREQ)
			fi_freeinfo(entry->cm_entry->info);
		free(entry);
	}
}

struct fi_ibv_eq_entry  *
fi_ibv_eq_alloc_entry(uint32_t event, const void *buf, size_t len)
{
	struct fi_ibv_eq_entry *entry;

	entry = calloc(1, sizeof(struct fi_ibv_eq_entry) + len);
	if (!entry) {
		VERBS_WARN(FI_LOG_EP_CTRL, "Unable to allocate EQ entry\n");
		return NULL;
	}

	entry->event = event;
	entry->len = len;
	memcpy(entry->entry, buf, len);

	return entry;
}

ssize_t fi_ibv_eq_write_event(struct fi_ibv_eq *eq, uint32_t event,
			      const void *buf, size_t len)
{
	struct fi_ibv_eq_entry *entry;

	entry = fi_ibv_eq_alloc_entry(event, buf, len);
	if (!entry)
		return -FI_ENOMEM;

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
	memcpy(buf, entry->entry, entry->len);

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

	if (len < sizeof(struct fi_eq_cm_entry))
		return -FI_ETOOSMALL;

	eq = container_of(eq_fid, struct fi_ibv_eq, eq_fid.fid);

	if ((ret = fi_ibv_eq_read_event(eq, event, buf, len, flags)))
		return ret;

	if (eq->channel) {
next_event:
		fastlock_acquire(&eq->lock);
		ret = rdma_get_cm_event(eq->channel, &cma_event);
		if (ret) {
			fastlock_release(&eq->lock);
			return -errno;
		}

		ret = fi_ibv_eq_cm_process_event(eq, cma_event, event,
						 (struct fi_eq_cm_entry *)buf,
						 len);
		fastlock_release(&eq->lock);
		/* If the CM event was handled internally (e.g. XRC), continue
		 * to process events. */
		if (ret == -FI_EAGAIN)
			goto next_event;

		if (flags & FI_PEEK)
			ret = fi_ibv_eq_write_event(eq, *event, buf, ret);

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

	if (!ofi_rbmap_empty(&eq->xrc.sidr_conn_rbmap))
		VERBS_WARN(FI_LOG_EP_CTRL, "SIDR connection RBmap not empty\n");

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

	ofi_rbmap_cleanup(&eq->xrc.sidr_conn_rbmap);
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
	ofi_rbmap_init(&_eq->xrc.sidr_conn_rbmap, fi_ibv_sidr_conn_compare);

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

