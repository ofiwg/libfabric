#include "verbs_ofi.h"

static int vrb_rdmacm_eq_open(struct vrb_eq *eq);

int vrb_xrc_migrade_rdma_id(struct vrb_xrc_ep *ep)
{
	struct vrb_rdmacm_eq_ctx *eq_ctx = ep->base_ep.eq ? ep->base_ep.eq->cm_ctx : NULL;
	if (!eq_ctx) {
		struct vrb_rdmacm_domain_ctx *domain_ctx = vrb_ep2_domain(&ep->base_ep)->cm_ctx;
		return rdma_migrate_id(ep->tgt_id, domain_ctx->cm_channel);
	}
	return rdma_migrate_id(ep->tgt_id, eq_ctx->cm_channel);
}

static void
vrb_msg_ep_prepare_rdma_cm_hdr(void *priv_data,
				const struct rdma_cm_id *id)
{
	struct vrb_rdma_cm_hdr *rdma_cm_hdr = priv_data;

	/* ip_version=6 would requires IPoIB to be installed and the IP link
	 * to be UP, which we don't want. As a work-around, we set ip_version to 0,
	 * which let the CMA kernel code to skip any requirement for IPoIB. */
	rdma_cm_hdr->ip_version = 0;
	rdma_cm_hdr->port = htons(ofi_addr_get_port(&id->route.addr.src_addr));

	/* Record the GIDs */
	memcpy(rdma_cm_hdr->src_addr,
		   &((struct ofi_sockaddr_ib *)&id->route.addr.src_addr)->sib_addr, 16);
	memcpy(rdma_cm_hdr->dst_addr,
		   &((struct ofi_sockaddr_ib *)&id->route.addr.dst_addr)->sib_addr, 16);
}

static int vrb_rdmacm_listen(struct vrb_pep *pep)
{
	struct sockaddr *addr;
	int ret;
	struct vrb_rdmacm_pep_ctx *ctx = pep->cm_ctx;
	struct vrb_rdmacm_eq_ctx *eq_ctx = pep->eq->cm_ctx;

	addr = rdma_get_local_addr(ctx->id);
	ofi_straddr_log(&vrb_prov, FI_LOG_INFO,
			FI_LOG_EP_CTRL, "listening on", addr);

	VRB_INFO(FI_LOG_EP_CTRL,
		 "listen: id->channel=%p eq_channel=%p eq_channel_fd=%d\n",
		 (void *)ctx->id->channel, (void *)eq_ctx->cm_channel,
		 eq_ctx->cm_channel->fd);

	ret = rdma_listen(ctx->id, pep->backlog);
	if (ret) {
		VRB_WARN(FI_LOG_EP_CTRL, "rdma_listen failed: %s\n",
			 strerror(errno));
		return -errno;
	}

	VRB_INFO(FI_LOG_EP_CTRL,
		 "listen: success, id->channel after listen=%p\n",
		 (void *)ctx->id->channel);

	if (vrb_is_xrc_info(pep->info)) {
		ret = rdma_listen(ctx->xrc_ps_udp_id, pep->backlog);
		if (ret)
			ret = -errno;
	}
	return ret;
}

static inline void
vrb_ep_prepare_rdma_cm_param(struct rdma_conn_param *conn_param,
				void *priv_data, size_t priv_data_size)
{
	conn_param->private_data = priv_data;
	conn_param->private_data_len = (uint8_t)priv_data_size;
	conn_param->responder_resources = RDMA_MAX_RESP_RES;
	conn_param->initiator_depth = RDMA_MAX_INIT_DEPTH;
	conn_param->flow_control = 1;
	conn_param->rnr_retry_count = 7;
}

static int vrb_rdmacm_connect(struct vrb_ep *ep, const void *addr,
			      const void *param, size_t paramlen)
{
	struct vrb_rdmacm_ep_ctx *ctx = ep->cm_ctx;
	struct vrb_cm_data_hdr *cm_hdr;
	int ret;
	size_t priv_data_len = sizeof(*cm_hdr) + paramlen + sizeof(struct vrb_rdma_cm_hdr);
	assert(ctx);
	ep->cm_priv_data = malloc(priv_data_len);
	if (!ep->cm_priv_data)
		return -FI_ENOMEM;

	cm_hdr = (void *)((char *)ep->cm_priv_data + sizeof(struct vrb_rdma_cm_hdr));
	vrb_msg_ep_prepare_cm_data(param, paramlen, cm_hdr);
	vrb_ep_prepare_rdma_cm_param(&ep->conn_param, ep->cm_priv_data,
					priv_data_len);
	ep->conn_param.retry_count = 15;

	if (ep->srx)
		ep->conn_param.srq = 1;

	if (addr) {
		free(ep->info_attr.dest_addr);
		ep->info_attr.dest_addr = mem_dup(addr, ofi_sizeofaddr(addr));
		if (!ep->info_attr.dest_addr) {
			ret = -FI_ENOMEM;
			goto err1;
		}
		ep->info_attr.dest_addrlen = ofi_sizeofaddr(addr);
	}

	ofi_genlock_lock(&vrb_ep2_progress(ep)->ep_lock);
	assert(ep->state == VRB_IDLE);
	ep->state = VRB_RESOLVE_ADDR;
	vrb_prof_func_start("rdma_resolve_addr");
	if (rdma_resolve_addr(ctx->id, ep->info_attr.src_addr,
			      ep->info_attr.dest_addr, VERBS_RESOLVE_TIMEOUT)) {
		ret = -errno;
		VRB_WARN_ERRNO(FI_LOG_EP_CTRL, "rdma_resolve_addr");
		ofi_straddr_log(&vrb_prov, FI_LOG_WARN, FI_LOG_EP_CTRL,
				"src addr", ep->info_attr.src_addr);
		ofi_straddr_log(&vrb_prov, FI_LOG_WARN, FI_LOG_EP_CTRL,
				"dst addr", ep->info_attr.dest_addr);
		ep->state = VRB_IDLE;
		goto err2;
	}
	vrb_prof_func_end("rdma_resolve_addr");
	ofi_genlock_unlock(&vrb_ep2_progress(ep)->ep_lock);
	return 0;
err2:
	vrb_prof_func_end("rdma_resolve_addr");
	ofi_genlock_unlock(&vrb_ep2_progress(ep)->ep_lock);
	free(ep->info_attr.dest_addr);
	ep->info_attr.dest_addr = NULL;
	ep->info_attr.dest_addrlen = 0;
err1:
	free(ep->cm_priv_data);
	ep->cm_priv_data = NULL;
	return ret;
}

static int vrb_rdmacm_accept(struct vrb_ep *ep, const void *param,
			     size_t paramlen)
{
	struct vrb_cm_data_hdr *cm_hdr;
	struct vrb_rdmacm_ep_ctx *ctx = ep->cm_ctx;
	struct rdma_conn_param conn_param = {0};
	struct vrb_connreq *connreq;
	int ret;

	cm_hdr = alloca(sizeof(*cm_hdr) + paramlen);
	vrb_msg_ep_prepare_cm_data(param, paramlen, cm_hdr);
	vrb_ep_prepare_rdma_cm_param(&conn_param, cm_hdr,
					sizeof(*cm_hdr) + paramlen);

	if (ep->srx)
		conn_param.srq = 1;

	ofi_genlock_lock(&vrb_ep2_progress(ep)->ep_lock);
	assert(ep->state == VRB_REQ_RCVD);
	ep->state = VRB_ACCEPTING;
	vrb_prof_func_start("rdma_accept");
	ret = rdma_accept(ctx->id, &conn_param);
	vrb_prof_func_end("rdma_accept");
	if (ret) {
		VRB_WARN_ERRNO(FI_LOG_EP_CTRL, "rdma_accept");
		ep->state = VRB_DISCONNECTED;
		ret = -errno;
	} else {
		connreq = container_of(ep->info_attr.handle,
				       struct vrb_connreq, handle);
		free(connreq);
	}
	if (!ret && ep->profile)
		vrb_prof_cntr_inc(ep->profile, FI_VAR_CONN_ACCEPT);

	ofi_genlock_unlock(&vrb_ep2_progress(ep)->ep_lock);
	return 0;
}

static int vrb_rdmacm_shutdown(struct vrb_ep *ep, uint64_t flags)
{
	struct vrb_rdmacm_ep_ctx *ctx = ep->cm_ctx;

	if (!ctx || !ctx->id)
		return 0;

	if (rdma_disconnect(ctx->id)) {
		VRB_WARN_ERRNO(FI_LOG_EP_CTRL, "rdma_disconnect");
		return -errno;
	}

	return 0;
}

static int vrb_rdmacm_ep_init(struct vrb_ep *ep, struct fi_info *info)
{
	struct vrb_rdmacm_ep_ctx *ep_ctx;
	struct vrb_domain *domain = vrb_ep2_domain(ep);
	struct vrb_pep *pep;
	struct vrb_connreq *connreq;
	struct vrb_rdmacm_domain_ctx *domain_ctx = domain->cm_ctx;
	struct vrb_rdmacm_pep_ctx *pep_ctx;
	int ret;

	ep_ctx = calloc(1, sizeof(*ep_ctx));
	if (!ep_ctx)
		return -FI_ENOMEM;

	ep->cm_ctx = ep_ctx;

	if (domain->ext_flags & VRB_USE_XRC) {
		if (domain->util_domain.threading == FI_THREAD_SAFE) {
			*ep->util_ep.ep_fid.msg = vrb_msg_xrc_ep_msg_ops_ts;
			ep->util_ep.ep_fid.rma = &vrb_msg_xrc_ep_rma_ops_ts;
		} else {
			*ep->util_ep.ep_fid.msg = vrb_msg_xrc_ep_msg_ops;
			ep->util_ep.ep_fid.rma = &vrb_msg_xrc_ep_rma_ops;
		}
		ep->util_ep.ep_fid.cm = &vrb_msg_xrc_ep_cm_ops;
		ep->util_ep.ep_fid.atomic = &vrb_msg_xrc_ep_atomic_ops;
	} else {
		if (domain->util_domain.threading == FI_THREAD_SAFE) {
			*ep->util_ep.ep_fid.msg = vrb_msg_ep_msg_ops_ts;
			ep->util_ep.ep_fid.rma = &vrb_msg_ep_rma_ops_ts;
		} else {
			*ep->util_ep.ep_fid.msg = vrb_msg_ep_msg_ops;
			ep->util_ep.ep_fid.rma = &vrb_msg_ep_rma_ops;
		}
		ep->util_ep.ep_fid.cm = &vrb_msg_ep_cm_ops;
		ep->util_ep.ep_fid.atomic = &vrb_msg_ep_atomic_ops;
	}

	if (!info->handle) {
		/* Only RC, XRC active RDMA CM ID is created at connect */
		if (!(domain->ext_flags & VRB_USE_XRC)) {
			ret = vrb_create_ep(ep,
				vrb_get_port_space(info->addr_format), &ep_ctx->id);
			if (ret)
				goto err;
			ep_ctx->id->context = &ep->util_ep.ep_fid.fid;
		}
	} else if (info->handle->fclass == FI_CLASS_CONNREQ) {
		connreq = container_of(info->handle,
					struct vrb_connreq, handle);
		if (domain->ext_flags & VRB_USE_XRC) {
			assert(connreq->is_xrc);

			if (!connreq->xrc.is_reciprocal) {
				ret = vrb_process_xrc_connreq(ep, connreq);
				if (ret)
					goto err;
			}
		} else {
			/* ep now owns this rdma cm id, prevent trying to access
				* it outside of ep operations to avoid possible use-after-
				* free bugs in case the ep is closed
				*/
			ofi_genlock_lock(&vrb_ep2_progress(ep)->ep_lock);
			ep->state = VRB_REQ_RCVD;
			ep_ctx->id = connreq->id;
			connreq->id = NULL;
			ep->ibv_qp = ep_ctx->id->qp;
			ep_ctx->id->context = &ep->util_ep.ep_fid.fid;
			ofi_genlock_unlock(&vrb_ep2_progress(ep)->ep_lock);
		}
	} else if (info->handle->fclass == FI_CLASS_PEP) {
		/* Opening an active EP seeded from a PEP's address.
		 * Create a new ID on the domain channel and resolve
		 * the address; the PEP itself keeps its own ID. */
		pep = container_of(info->handle, struct vrb_pep, pep_fid.fid);
		pep_ctx = pep->cm_ctx;
		(void)pep_ctx; /* pep_ctx available for future use */
		ret = rdma_create_id(domain_ctx->cm_channel, &ep_ctx->id,
				     &ep->util_ep.ep_fid.fid,
				     vrb_get_port_space(info->addr_format));
		if (ret) {
			VRB_WARN_ERRNO(FI_LOG_EP_CTRL, "rdma_create_id");
			goto err;
		}
		vrb_prof_func_start("rdma_resolve_addr");
		if (rdma_resolve_addr(ep_ctx->id, info->src_addr, info->dest_addr,
				      VERBS_RESOLVE_TIMEOUT)) {
			ret = -errno;
			VRB_WARN_ERRNO(FI_LOG_EP_CTRL, "rdma_resolve_addr");
			rdma_destroy_id(ep_ctx->id);
			ep_ctx->id = NULL;
			vrb_prof_func_end("rdma_resolve_addr");
			goto err;
		}
		vrb_prof_func_end("rdma_resolve_addr");
		ep_ctx->id->context = &ep->util_ep.ep_fid.fid;
	} else {
		ret = -FI_ENOSYS;
		goto err;
	}

	return 0;
err:
	free(ep_ctx);
	ep->cm_ctx = NULL;
	return ret;
}

static int vrb_rdmacm_ep_close(struct vrb_ep *ep)
{
	struct vrb_rdmacm_ep_ctx *ctx = ep->cm_ctx;

	if (ctx) {
		if (ctx->id)
			rdma_destroy_ep(ctx->id);
		free(ctx);
		ep->cm_ctx = NULL;
	}

	return 0;
}

static int vrb_rdmacm_ep_bind(struct vrb_ep *ep, struct vrb_eq *eq)
{
	struct vrb_rdmacm_ep_ctx *ctx = ep->cm_ctx;
	struct vrb_rdmacm_eq_ctx *eq_ctx;
	int ret;

	if (!eq->cm_ctx) {
		ret = ep->cm_ops->eq_open(eq);
		if (ret)
			return ret;
	}
	eq_ctx = eq->cm_ctx;
	ret = rdma_migrate_id(ctx->id, eq_ctx->cm_channel);
	if (ret) {
		VRB_WARN_ERRNO(FI_LOG_EP_CTRL, "rdma_migrate_id");
		return -errno;
	}

	return 0;
}

static int vrb_rdmacm_pep_init(struct vrb_pep *pep)
{
	int ret;
	struct vrb_rdmacm_pep_ctx *ctx = calloc(1, sizeof(struct vrb_rdmacm_pep_ctx));
	if (!ctx)
		return -FI_ENOMEM;

	ret = rdma_create_id(NULL, &ctx->id, &pep->pep_fid.fid,
			     vrb_get_port_space(pep->info->addr_format));
	if (ret) {
		VRB_WARN_ERRNO(FI_LOG_EP_CTRL, "rdma_create_id");
		goto err1;
	}

	if (pep->info->src_addr) {
		ret = rdma_bind_addr(ctx->id, (struct sockaddr *) pep->info->src_addr);
		if (ret) {
			VRB_WARN_ERRNO(FI_LOG_EP_CTRL, "rdma_bind_addr");
			ret = -errno;
			goto err2;
		}
		pep->bound = 1;
	}

	/* XRC listens on both RDMA_PS_TCP and RDMA_PS_UDP */
	if (vrb_is_xrc_info(pep->info)) {
		ret = rdma_create_id(NULL, &ctx->xrc_ps_udp_id,
				     &pep->pep_fid.fid, RDMA_PS_UDP);
		if (ret) {
			VRB_WARN_ERRNO(FI_LOG_EP_CTRL, "rdma_create_id");
			goto err2;
		}
		/* Currently both listens must be bound to same port number */
		ofi_addr_set_port(pep->info->src_addr,
				  ntohs(rdma_get_src_port(ctx->id)));
		ret = rdma_bind_addr(ctx->xrc_ps_udp_id,
				     (struct sockaddr *)pep->info->src_addr);
		if (ret) {
			VRB_WARN_ERRNO(FI_LOG_EP_CTRL, "rdma_bind_addr");
			goto err3;
		}
	}

	pep->cm_ctx = ctx;
	return 0;

err3:
	/* Only possible for XRC code path */
	rdma_destroy_id(ctx->xrc_ps_udp_id);
err2:
	rdma_destroy_id(ctx->id);
err1:
	free(ctx);
	return ret;
}

static int vrb_rdmacm_pep_close(struct vrb_pep *pep)
{
	struct vrb_rdmacm_pep_ctx *ctx = pep->cm_ctx;

	if (ctx) {
		if (ctx->id)
			rdma_destroy_id(ctx->id);
		if (ctx->xrc_ps_udp_id)
			rdma_destroy_id(ctx->xrc_ps_udp_id);
		free(ctx);
		pep->cm_ctx = NULL;
	}

	return 0;
}

static int vrb_rdmacm_pep_bind(struct vrb_pep *pep, struct vrb_eq *eq)
{
	struct vrb_rdmacm_pep_ctx *pep_ctx = pep->cm_ctx;
	struct vrb_rdmacm_eq_ctx *eq_ctx;
	int ret;

	/*
	 * This is a restrictive solution that enables an XRC EP to
	 * inform it's peer the port that should be used in making the
	 * reciprocal connection request. While it meets RXM requirements
	 * it limits an EQ to a single passive endpoint. TODO: implement
	 * a more general solution.
	 */
	if (vrb_is_xrc_info(pep->info)) {
		if (pep->eq->xrc.pep_port) {
			VRB_WARN(FI_LOG_EP_CTRL,
				   "XRC limits EQ binding to a single PEP\n");
			return -FI_EINVAL;
		}
		pep->eq->xrc.pep_port = ntohs(rdma_get_src_port(pep_ctx->id));
	}

	if (!eq->cm_ctx) {
		ret = vrb_rdmacm_eq_open(eq);
		if (ret)
			return ret;
	}
	eq_ctx = eq->cm_ctx;

	ret = rdma_migrate_id(pep_ctx->id, eq_ctx->cm_channel);
	if (ret) {
		VRB_WARN_ERRNO(FI_LOG_EP_CTRL, "rdma_migrate_id");
		return -errno;
	}

	if (vrb_is_xrc_info(pep->info) && pep_ctx->xrc_ps_udp_id) {
		ret = rdma_migrate_id(pep_ctx->xrc_ps_udp_id,
				      eq_ctx->cm_channel);
		if (ret) {
			VRB_WARN_ERRNO(FI_LOG_EP_CTRL, "rdma_migrate_id xrc");
			return -errno;
		}
		ofi_addr_set_port(pep->info->src_addr,
				  ntohs(rdma_get_src_port(pep_ctx->id)));
		ret = rdma_bind_addr(pep_ctx->xrc_ps_udp_id,
				     (struct sockaddr *)pep->info->src_addr);
		if (ret) {
			VRB_WARN_ERRNO(FI_LOG_EP_CTRL, "rdma_bind_addr");
			return -errno;
		}
	}

	VRB_INFO(FI_LOG_EP_CTRL,
		 "rdmacm pep_bind: recreated on eq channel fd=%d\n",
		 eq_ctx->cm_channel->fd);
	return 0;
}

static int vrb_rdmacm_domain_init(struct vrb_domain *domain)
{
	struct vrb_rdmacm_domain_ctx *ctx;

	ctx = calloc(1, sizeof(*ctx));
	if (!ctx)
		return -FI_ENOMEM;

	/* Create a bootstrap channel for creating rdmacm IDs before an EQ
	 * is available.  This channel is migrated away from in ep_bind and
	 * is never directly polled (not added to any epollfd). */
	ctx->cm_channel = rdma_create_event_channel();
	if (!ctx->cm_channel) {
		free(ctx);
		return -errno;
	}

	domain->cm_ctx = ctx;
	return 0;
}

static int vrb_rdmacm_domain_close(struct vrb_domain *domain)
{
	struct vrb_rdmacm_domain_ctx *ctx = domain->cm_ctx;

	if (ctx) {
		if (ctx->cm_channel)
			rdma_destroy_event_channel(ctx->cm_channel);
		free(ctx);
		domain->cm_ctx = NULL;
	}

	return 0;
}

static int
vrb_eq_addr_resolved_event(struct vrb_ep *ep)
{
	struct vrb_rdmacm_ep_ctx *ctx = ep->cm_ctx;
	struct vrb_recv_wr *wr;
	struct slist_entry *entry;
	struct ibv_qp_init_attr attr = { 0 };
	int ret;

	assert(ofi_genlock_held(&vrb_ep2_progress(ep)->ep_lock));
	assert(ep->state == VRB_RESOLVE_ADDR);
	if (ep->util_ep.type == FI_EP_MSG) {
		vrb_msg_ep_get_qp_attr(ep, &attr);

		/* Client-side QP creation */
		vrb_prof_func_start("rdma_create_qp");
		if (rdma_create_qp(ctx->id, vrb_ep2_domain(ep)->pd, &attr)) {
			ep->state = VRB_DISCONNECTED;
			ret = -errno;
			VRB_WARN(FI_LOG_EP_CTRL,
				 "rdma_create_qp failed: %d\n", -ret);
			return ret;
		}
		vrb_prof_func_end("rdma_create_qp");
		if (ep->profile)
			vrb_prof_cntr_inc(ep->profile, FI_VAR_MSG_QUEUE_CNT);

		/* Allow shared XRC INI QP not controlled by RDMA CM
		 * to share same post functions as RC QP. */
		ep->ibv_qp = ctx->id->qp;
	}

	assert(ep->ibv_qp);
	while (!slist_empty(&ep->prepost_wr_list)) {
		entry = ep->prepost_wr_list.head;
		wr = container_of(entry, struct vrb_recv_wr, entry);

		ret = vrb_post_recv_internal(ep, &wr->wr);
		if (ret) {
			VRB_WARN(FI_LOG_EP_CTRL,
			         "Failed to post receive buffers: %d\n", -ret);

			return ret;
		}
		vrb_free_recv_wr(vrb_ep2_progress(ep), wr);
		slist_remove_head(&ep->prepost_wr_list);
	}

	ep->state = VRB_RESOLVE_ROUTE;
	vrb_prof_func_start("rdma_resolve_route");
	if (rdma_resolve_route(ctx->id, VERBS_RESOLVE_TIMEOUT)) {
		ep->state = VRB_DISCONNECTED;
		ret = -errno;
		VRB_WARN(FI_LOG_EP_CTRL,
			"rdma_resolve_route failed: %d\n",
			-ret);
		return ret;
	}
	vrb_prof_func_end("rdma_resolve_route");

	return -FI_EAGAIN;
}

static int vrb_eq_set_xrc_info(struct rdma_cm_event *event,
				  struct vrb_xrc_conn_info *info)
{
	struct vrb_xrc_cm_data *remote = (struct vrb_xrc_cm_data *)
						event->param.conn.private_data;
	int ret;

	ret = vrb_verify_xrc_cm_data(remote,
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
vrb_pep_dev_domain_match(struct fi_info *hints, const char *devname)
{
	int ret;

	if ((VRB_EP_PROTO(hints)) == FI_PROTO_RDMA_CM_IB_XRC)
		ret = vrb_cmp_xrc_domain_name(hints->domain_attr->name,
						 devname);
	else
		ret = strcmp(hints->domain_attr->name, devname);

	return ret;
}

static int
vrb_eq_cm_getinfo(struct rdma_cm_event *event, struct fi_info *pep_info,
		     struct fi_info **info)
{
	struct fi_info *hints;
	struct vrb_connreq *connreq;
	const char *devname = ibv_get_device_name(event->id->verbs->device);
	int ret = -FI_ENOMEM;

	if (!(hints = fi_dupinfo(pep_info))) {
		VRB_WARN(FI_LOG_EP_CTRL, "dupinfo failure\n");
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
		if (vrb_pep_dev_domain_match(hints, devname)) {
			VRB_WARN(FI_LOG_EQ, "passive endpoint domain: %s does"
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

	ofi_mutex_lock(&vrb_info_mutex);
	ret = vrb_get_matching_info(hints->fabric_attr->api_version, hints,
				    info, vrb_util_prov.info, 0);
	ofi_mutex_unlock(&vrb_info_mutex);
	if (ret)
		goto err1;

	ofi_alter_info(*info, hints, hints->fabric_attr->api_version);
	vrb_alter_info(hints, *info);
	(*info)->fabric_attr->api_version = pep_info->fabric_attr->api_version;
	(*info)->fabric_attr->prov_name = strdup(pep_info->fabric_attr->prov_name);
	if (!(*info)->fabric_attr->prov_name)
		goto err2;

	free((*info)->src_addr);
	(*info)->src_addrlen = ofi_sizeofaddr(rdma_get_local_addr(event->id));
	(*info)->src_addr = malloc((*info)->src_addrlen);
	if (!((*info)->src_addr))
		goto err2;
	memcpy((*info)->src_addr, rdma_get_local_addr(event->id), (*info)->src_addrlen);

	assert(!(*info)->dest_addr);
	(*info)->dest_addrlen = ofi_sizeofaddr(rdma_get_peer_addr(event->id));
	(*info)->dest_addr = malloc((*info)->dest_addrlen);
	if (!((*info)->dest_addr))
		goto err2;
	memcpy((*info)->dest_addr, rdma_get_peer_addr(event->id), (*info)->dest_addrlen);

	ofi_straddr_dbg(&vrb_prov, FI_LOG_EQ, "src", (*info)->src_addr);
	ofi_straddr_dbg(&vrb_prov, FI_LOG_EQ, "dst", (*info)->dest_addr);

	connreq = calloc(1, sizeof *connreq);
	if (!connreq) {
		VRB_WARN(FI_LOG_EP_CTRL,
			   "Unable to allocate connreq memory\n");
		goto err2;
	}

	connreq->handle.fclass = FI_CLASS_CONNREQ;
	connreq->id = event->id;

	if (vrb_is_xrc_info(*info)) {
		connreq->is_xrc = 1;
		ret = vrb_eq_set_xrc_info(event, &connreq->xrc);
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

static void vrb_eq_skip_rdma_cm_hdr(const void **priv_data,
						size_t *priv_data_len)
{
	size_t rdma_cm_hdr_len = sizeof(struct vrb_rdma_cm_hdr);

	if (*priv_data_len > rdma_cm_hdr_len) {
		*priv_data = (void*)((char *)*priv_data + rdma_cm_hdr_len);
		*priv_data_len -= rdma_cm_hdr_len;
	}
}

static ssize_t
vrb_eq_cm_process_event(struct vrb_eq *eq,
	struct rdma_cm_event *cma_event, uint32_t *event,
	struct fi_eq_cm_entry *entry, size_t len)
{
	const struct vrb_cm_data_hdr *cm_hdr;
	size_t datalen = 0;
	size_t priv_datalen = cma_event->param.conn.private_data_len;
	const void *priv_data = cma_event->param.conn.private_data;
	int ret, acked = 0;;
	fid_t fid = cma_event->id->context;
	struct vrb_pep *pep =
		container_of(fid, struct vrb_pep, pep_fid);
	struct vrb_ep *ep;
	struct vrb_xrc_ep *xrc_ep;

	assert(ofi_mutex_held(&eq->event_lock));
	switch (cma_event->event) {
	case RDMA_CM_EVENT_ADDR_RESOLVED:
		ep = container_of(fid, struct vrb_ep, util_ep.ep_fid);
		if (ep->profile)
			vrb_prof_set_st_time(ep->profile, (ofi_gettime_ns()),
					VRB_RESOLVE_ADDR);

		ofi_genlock_lock(&vrb_ep2_progress(ep)->ep_lock);
		ret = vrb_eq_addr_resolved_event(ep);
		ofi_genlock_unlock(&vrb_ep2_progress(ep)->ep_lock);
		if (ret != -FI_EAGAIN) {
			eq->err.err = -ret;
			eq->err.prov_errno = ret;
			goto err;
		}
		goto ack;

	case RDMA_CM_EVENT_ROUTE_RESOLVED:
		ep = container_of(fid, struct vrb_ep, util_ep.ep_fid);
		if (ep->profile)
			vrb_prof_set_st_time(ep->profile, (ofi_gettime_ns()),
					VRB_RESOLVE_ROUTE);
		ofi_genlock_lock(&vrb_ep2_progress(ep)->ep_lock);
		assert(ep->state == VRB_RESOLVE_ROUTE);
		ep->state = VRB_CONNECTING;
		{
			struct vrb_rdmacm_ep_ctx *ep_ctx = ep->cm_ctx;
			if (cma_event->id->route.addr.src_addr.sa_family != AF_IB) {
				vrb_eq_skip_rdma_cm_hdr((const void **)&ep->conn_param.private_data,
							(size_t *)&ep->conn_param.private_data_len);
			} else {
				vrb_msg_ep_prepare_rdma_cm_hdr(ep->cm_priv_data, ep_ctx->id);
			}
			vrb_prof_func_start("rdma_connect");
			ret = rdma_connect(ep_ctx->id, &ep->conn_param);
			vrb_prof_func_end("rdma_connect");
		}
		if (!ret && ep->profile)
			vrb_prof_cntr_inc(ep->profile, FI_VAR_CONN_REQUEST);

		if (ret) {
			ep->state = VRB_DISCONNECTED;
			ret = -errno;
			FI_WARN(&vrb_prov, FI_LOG_EP_CTRL,
				"rdma_connect failed: %s (%d)\n",
				strerror(-ret), -ret);
			if (vrb_is_xrc_ep(ep)) {
				xrc_ep = container_of(fid, struct vrb_xrc_ep,
						      base_ep.util_ep.ep_fid);
				vrb_put_shared_ini_conn(xrc_ep);
			}
		} else {
			ret = -FI_EAGAIN;
		}
		ofi_genlock_unlock(&vrb_ep2_progress(ep)->ep_lock);
		if (ret != -FI_EAGAIN) {
			eq->err.err = -ret;
			eq->err.prov_errno = ret;
			goto err;
		}
		goto ack;
	case RDMA_CM_EVENT_CONNECT_REQUEST:
		*event = FI_CONNREQ;
		ret = vrb_eq_cm_getinfo(cma_event, pep->info, &entry->info);
		if (ret) {
			VRB_WARN(FI_LOG_EP_CTRL,
				   "CM getinfo error %d\n", ret);
			rdma_destroy_id(cma_event->id);
			eq->err.err = -ret;
			eq->err.prov_errno = ret;
			goto err;
		}

		if (vrb_is_xrc_info(entry->info)) {
			ret = vrb_eq_xrc_connreq_event(eq, entry, len, event,
							  cma_event, &acked,
							  &priv_data, &priv_datalen);
			if (ret == -FI_EAGAIN) {
				fi_freeinfo(entry->info);
				entry->info = NULL;
				goto ack;
			}
			if (*event == FI_CONNECTED)
				goto ack;
		} else if (cma_event->id->route.addr.src_addr.sa_family == AF_IB) {
			vrb_eq_skip_rdma_cm_hdr(&priv_data, &priv_datalen);
		}
		break;
	case RDMA_CM_EVENT_CONNECT_RESPONSE:
	case RDMA_CM_EVENT_ESTABLISHED:
		*event = FI_CONNECTED;
		ep = container_of(fid, struct vrb_ep, util_ep.ep_fid);
		if (ep->profile) {
			vrb_prof_set_st_time(ep->profile, (ofi_gettime_ns()),
					VRB_CONNECTED);
			vrb_prof_cntr_inc(ep->profile,
					FI_VAR_CONNECTION_CNT);
		}
		if (cma_event->id->qp &&
		    cma_event->id->qp->context->device->transport_type !=
		    IBV_TRANSPORT_IWARP) {
			vrb_set_rnr_timer(cma_event->id->qp);
		}
		if (vrb_is_xrc_ep(ep)) {
			ret = vrb_eq_xrc_connected_event(eq, cma_event,
							    &acked, entry, len,
							    event);
			goto ack;
		}
		ofi_genlock_lock(&vrb_ep2_progress(ep)->ep_lock);
		assert(ep->state == VRB_CONNECTING || ep->state == VRB_ACCEPTING);
		ep->state = VRB_CONNECTED;
		ofi_genlock_unlock(&vrb_ep2_progress(ep)->ep_lock);
		entry->info = NULL;
		break;
	case RDMA_CM_EVENT_DISCONNECTED:
		ep = container_of(fid, struct vrb_ep, util_ep.ep_fid);
		if (ep->profile)
			vrb_prof_set_st_time(ep->profile, (ofi_gettime_ns()),
                                            VRB_DISCONNECTED);
		ofi_genlock_lock(&vrb_ep2_progress(ep)->ep_lock);
		if (ep->state == VRB_DISCONNECTED) {
			/* If we saw a transfer error, we already generated
			 * a shutdown event.
			 */
			ret = -FI_EAGAIN;
			ofi_genlock_unlock(&vrb_ep2_progress(ep)->ep_lock);
			goto ack;
		}
		ep->state = VRB_DISCONNECTED;
		ofi_genlock_unlock(&vrb_ep2_progress(ep)->ep_lock);
		if (vrb_is_xrc_ep(ep)) {
			vrb_eq_xrc_disconnect_event(eq, cma_event, &acked);
			ret = -FI_EAGAIN;
			goto ack;
		}
		*event = FI_SHUTDOWN;
		entry->info = NULL;
		break;
	case RDMA_CM_EVENT_TIMEWAIT_EXIT:
		ep = container_of(fid, struct vrb_ep, util_ep.ep_fid);
		if (vrb_is_xrc_ep(ep))
			vrb_eq_xrc_timewait_event(eq, cma_event, &acked);
		ret = -FI_EAGAIN;
		goto ack;
	case RDMA_CM_EVENT_ADDR_ERROR:
	case RDMA_CM_EVENT_ROUTE_ERROR:
	case RDMA_CM_EVENT_CONNECT_ERROR:
	case RDMA_CM_EVENT_UNREACHABLE:
		ep = container_of(fid, struct vrb_ep, util_ep.ep_fid);
		ofi_genlock_lock(&vrb_ep2_progress(ep)->ep_lock);
		assert(ep->state != VRB_DISCONNECTED);
		ep->state = VRB_DISCONNECTED;
		ofi_genlock_unlock(&vrb_ep2_progress(ep)->ep_lock);
		if (vrb_is_xrc_ep(ep)) {
			/* SIDR Reject is reported as UNREACHABLE unless
			 * status is negative */
			if (cma_event->id->ps == RDMA_PS_UDP &&
			    (cma_event->event == RDMA_CM_EVENT_UNREACHABLE &&
			     cma_event->status >= 0))
				goto xrc_shared_reject;

			ret = vrb_eq_xrc_cm_err_event(eq, cma_event, &acked);
			if (ret == -FI_EAGAIN)
				goto ack;

			*event = FI_SHUTDOWN;
			entry->info = NULL;
			break;
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
		ep = container_of(fid, struct vrb_ep, util_ep.ep_fid);
		ofi_genlock_lock(&vrb_ep2_progress(ep)->ep_lock);
		assert(ep->state != VRB_DISCONNECTED);
		ep->state = VRB_DISCONNECTED;
		ofi_genlock_unlock(&vrb_ep2_progress(ep)->ep_lock);
		if (vrb_is_xrc_ep(ep)) {
xrc_shared_reject:
			ret = vrb_eq_xrc_rej_event(eq, cma_event);
			if (ret == -FI_EAGAIN)
				goto ack;
			vrb_eq_skip_xrc_cm_data(&priv_data, &priv_datalen);
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
		VRB_WARN(FI_LOG_EP_CTRL, "unknown rdmacm event received: %d\n",
			   cma_event->event);
		ret = -FI_EAGAIN;
		goto ack;
	}

	entry->fid = fid;

	/* rdmacm has no way to track how much data is sent by peer */
	if (priv_datalen)
		datalen = vrb_eq_copy_event_data(entry, len, priv_data,
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

static ssize_t vrb_rdmacm_progress(struct vrb_eq *eq, uint32_t *event,
				   void *buf, size_t len)
{
	struct vrb_rdmacm_eq_ctx *ctx = eq->cm_ctx;
	struct rdma_cm_event *cma_event;
	ssize_t ret;

	assert(ctx && ctx->cm_channel);


	do {
		ofi_mutex_lock(&eq->event_lock);
		vrb_prof_func_start("rdma_get_cm_event");
		ret = rdma_get_cm_event(ctx->cm_channel, &cma_event);
		vrb_prof_func_end("rdma_get_cm_event");
		if (ret) {
			ofi_mutex_unlock(&eq->event_lock);
			if (errno == EAGAIN || errno == EWOULDBLOCK)
				return 0;
			VRB_INFO(FI_LOG_EP_CTRL,
				 "rdmacm_progress: rdma_get_cm_event err=%d\n",
				 errno);
			return -errno;
		}

		VRB_INFO(FI_LOG_EP_CTRL,
			 "rdmacm_progress: got cm_event type=%d\n",
			 cma_event->event);
		vrb_prof_func_start("vrb_eq_cm_process_event");
		ret = vrb_eq_cm_process_event(eq, cma_event, event, buf, len);
		vrb_prof_func_end("vrb_eq_cm_process_event");
		VRB_INFO(FI_LOG_EP_CTRL,
			 "rdmacm_progress: process_event ret=%zd\n", ret);
		ofi_mutex_unlock(&eq->event_lock);
	} while (ret == -FI_EAGAIN);

	return ret;
}


static int vrb_rdmacm_setname(struct vrb_pep *pep, void *addr, size_t addrlen)
{
	struct vrb_rdmacm_pep_ctx *ctx;
	int ret;

	ctx = pep->cm_ctx;

	if (pep->src_addrlen && (addrlen != pep->src_addrlen)) {
		VRB_INFO(FI_LOG_FABRIC, "addrlen expected: %zu, got: %zu.\n",
			   pep->src_addrlen, addrlen);
		return -FI_EINVAL;
	}

	/* Re-create id if already bound */
	if (pep->bound) {
		ret = rdma_destroy_id(ctx->id);
		if (ret) {
			VRB_WARN_ERRNO(FI_LOG_FABRIC, "rdma_destroy_id");
			return -errno;
		}
		ret = rdma_create_id(NULL, &ctx->id,
				     &pep->pep_fid.fid, RDMA_PS_TCP);
		if (ret) {
			VRB_WARN_ERRNO(FI_LOG_FABRIC, "rdma_cm_id\n");
			return -errno;
		}
	}

	ret = rdma_bind_addr(ctx->id, (struct sockaddr *)addr);
	if (ret) {
		VRB_WARN_ERRNO(FI_LOG_FABRIC, "rdma_bind_addr");
		return -errno;
	}

	return 0;
}

static int vrb_rdmacm_getname(struct vrb_pep *pep, void *addr, size_t *addrlen)
{
	struct sockaddr *sa;
	struct vrb_rdmacm_pep_ctx *ctx = pep->cm_ctx;

	sa = rdma_get_local_addr(ctx->id);
	return vrb_copy_addr(addr, addrlen, sa);
}

static int vrb_rdmacm_eq_open(struct vrb_eq *eq)
{
	struct vrb_rdmacm_eq_ctx *ctx;
	int ret;

	ctx = calloc(1, sizeof(*ctx));
	if (!ctx)
		return -FI_ENOMEM;

	ctx->cm_channel = rdma_create_event_channel();
	if (!ctx->cm_channel) {
		ret = -errno;
		goto err_free;
	}

	ret = fi_fd_nonblock(ctx->cm_channel->fd);
	if (ret) {
		ret = -errno;
		goto err_chan;
	}

	if (ofi_epoll_add(eq->epollfd, ctx->cm_channel->fd,
			  OFI_EPOLL_IN, NULL)) {
		ret = -errno;
		goto err_chan;
	}

	eq->cm_ctx = ctx;
	eq->cm_ops = &vrb_rdmacm_ops;
	return 0;

err_chan:
	rdma_destroy_event_channel(ctx->cm_channel);
err_free:
	free(ctx);
	return ret;
}

static int vrb_rdmacm_eq_close(struct vrb_eq *eq)
{
	struct vrb_rdmacm_eq_ctx *ctx = eq->cm_ctx;

	if (!ctx)
		return 0;

	ofi_epoll_del(eq->epollfd, ctx->cm_channel->fd);
	rdma_destroy_event_channel(ctx->cm_channel);
	free(ctx);
	eq->cm_ctx = NULL;
	return 0;
}

static int vrb_rdmacm_disconnect(struct vrb_ep *ep)
{
	/* rdma_disconnect is called as part of shutdown; nothing extra needed. */
	return 0;
}

static int vrb_rdmacm_reject(struct vrb_pep *pep, struct vrb_connreq *connreq,
			     const void *param, size_t paramlen)
{
	int ret;

	if (!connreq->id)
		return -FI_EBUSY;

	vrb_prof_func_start("rdma_reject");
	ret = rdma_reject(connreq->id, param, (uint8_t)paramlen) ? -errno : 0;
	vrb_prof_func_end("rdma_reject");
	if (rdma_destroy_id(connreq->id))
		VRB_WARN_ERRNO(FI_LOG_EP_CTRL, "rdma_destroy_id");
	connreq->id = NULL;

	if (ret)
		VRB_WARN_ERR(FI_LOG_EP_CTRL, "rdma_reject", ret);
	return ret;
}

static int vrb_rdmacm_getpeer(struct vrb_ep *ep, void *addr, size_t *addrlen)
{
	struct vrb_rdmacm_ep_ctx *ctx = ep->cm_ctx;
	struct sockaddr *sa;

	if (!ctx || !ctx->id)
		return -FI_EOPBADSTATE;

	sa = rdma_get_peer_addr(ctx->id);
	return vrb_copy_addr(addr, addrlen, sa);
}

static int vrb_rdmacm_ep_enable(struct vrb_ep *ep,
				struct ibv_qp_init_attr *attr)
{
	struct vrb_rdmacm_ep_ctx *ctx = ep->cm_ctx;
	struct vrb_domain *domain = vrb_ep2_domain(ep);

	if (!ctx || !ctx->id || !ctx->id->verbs || ep->ibv_qp != NULL)
		return 0;

	/* Server-side QP creation after RDMA_CM_EVENT_CONNECT_REQUEST */
	vrb_prof_func_start("rdma_create_qp");
	if (rdma_create_qp(ctx->id, domain->pd, attr)) {
		VRB_WARN_ERRNO(FI_LOG_EP_CTRL, "rdma_create_qp");
		return -errno;
	}
	vrb_prof_func_end("rdma_create_qp");
	if (ep->profile)
		vrb_prof_cntr_inc(ep->profile, FI_VAR_MSG_QUEUE_CNT);
	ep->ibv_qp = ctx->id->qp;
	return 0;
}

static int vrb_rdmacm_ep_setname(struct vrb_ep *ep, void *addr, size_t addrlen)
{
	struct vrb_rdmacm_ep_ctx *ctx = ep->cm_ctx;
	void *save_addr;
	struct rdma_cm_id *id;
	int ret;

	save_addr = ep->info_attr.src_addr;
	ep->info_attr.src_addr = malloc(ep->info_attr.src_addrlen);
	if (!ep->info_attr.src_addr) {
		ep->info_attr.src_addr = save_addr;
		return -FI_ENOMEM;
	}
	memcpy(ep->info_attr.src_addr, addr, ep->info_attr.src_addrlen);

	ret = vrb_create_ep(ep, RDMA_PS_TCP, &id);
	if (ret) {
		free(ep->info_attr.src_addr);
		ep->info_attr.src_addr = save_addr;
		return ret;
	}

	if (ctx->id)
		rdma_destroy_ep(ctx->id);
	ctx->id = id;
	ep->ibv_qp = ctx->id->qp;
	free(save_addr);
	return 0;
}

static int vrb_rdmacm_ep_getname(struct vrb_ep *ep, void *addr, size_t *addrlen)
{
	struct vrb_rdmacm_ep_ctx *ctx = ep->cm_ctx;
	struct sockaddr *sa;

	if (!ctx || !ctx->id)
		return -FI_EOPBADSTATE;

	sa = rdma_get_local_addr(ctx->id);
	return vrb_copy_addr(addr, addrlen, sa);
}

// RDMA-CM ops structure
struct vrb_cm_ops vrb_rdmacm_ops = {
	.connect = vrb_rdmacm_connect,
	.accept = vrb_rdmacm_accept,
	.shutdown = vrb_rdmacm_shutdown,
	.disconnect = vrb_rdmacm_disconnect,
	.listen = vrb_rdmacm_listen,
	.reject = vrb_rdmacm_reject,
	.progress = vrb_rdmacm_progress,
	.ep_init = vrb_rdmacm_ep_init,
	.ep_close = vrb_rdmacm_ep_close,
	.ep_bind = vrb_rdmacm_ep_bind,
	.ep_enable = vrb_rdmacm_ep_enable,
	.ep_setname = vrb_rdmacm_ep_setname,
	.ep_getname = vrb_rdmacm_ep_getname,
	.pep_init = vrb_rdmacm_pep_init,
	.pep_close = vrb_rdmacm_pep_close,
	.pep_bind = vrb_rdmacm_pep_bind,
	.domain_init = vrb_rdmacm_domain_init,
	.domain_close = vrb_rdmacm_domain_close,
	.eq_open = vrb_rdmacm_eq_open,
	.eq_close = vrb_rdmacm_eq_close,
	.setname = vrb_rdmacm_setname,
	.getname = vrb_rdmacm_getname,
	.getpeer = vrb_rdmacm_getpeer,
	.cm_backend = VRB_CM_RDMACM,
};

/* Caller must hold domain:xrc.ini_lock */
void vrb_sched_ini_conn(struct vrb_ini_shared_conn *ini_conn)
{
	struct vrb_xrc_ep *ep;
	enum vrb_ini_qp_state last_state;
	struct vrb_rdmacm_eq_ctx *eq_ctx;
	struct vrb_rdmacm_ep_ctx *ep_ctx;
	int ret;

	/* Continue to schedule shared connections if the physical connection
	 * has completed and there are connection requests pending. We could
	 * implement a throttle here if it is determined that it is better to
	 * limit the number of outstanding connections. */
	while (1) {
		if (dlist_empty(&ini_conn->pending_list) ||
				ini_conn->state == VRB_INI_QP_CONNECTING)
			return;

		dlist_pop_front(&ini_conn->pending_list,
				struct vrb_xrc_ep, ep, ini_conn_entry);

		ep_ctx = ep->base_ep.cm_ctx;

		dlist_insert_tail(&ep->ini_conn_entry,
				  &ep->ini_conn->active_list);
		last_state = ep->ini_conn->state;

		ret = vrb_create_ep(&ep->base_ep,
				       last_state == VRB_INI_QP_UNCONNECTED ?
				       RDMA_PS_TCP : RDMA_PS_UDP,
				       &ep_ctx->id);
		if (ret) {
			VRB_WARN(FI_LOG_EP_CTRL,
				   "Failed to create active CM ID %d\n",
				   ret);
			goto err;
		}

		if (last_state == VRB_INI_QP_UNCONNECTED) {
			assert(!ep->ini_conn->phys_conn_id && ep_ctx->id);

			if (ep->ini_conn->ini_qp &&
			    ibv_destroy_qp(ep->ini_conn->ini_qp)) {
				VRB_WARN(FI_LOG_EP_CTRL, "Failed to destroy "
					   "physical INI QP %d\n", errno);
			}
			ret = vrb_create_ini_qp(ep);
			if (ret) {
				VRB_WARN(FI_LOG_EP_CTRL, "Failed to create "
					   "physical INI QP %d\n", ret);
				goto err;
			}
			ep->ini_conn->ini_qp = ep_ctx->id->qp;
			ep->ini_conn->state = VRB_INI_QP_CONNECTING;
			ep->ini_conn->phys_conn_id = ep_ctx->id;
		} else {
			assert(!ep_ctx->id->qp);
			VRB_DBG(FI_LOG_EP_CTRL, "Sharing XRC INI QPN %d\n",
				  ep->ini_conn->ini_qp->qp_num);
		}

		assert(ep->ini_conn->ini_qp);
		eq_ctx = ep->base_ep.eq ? ep->base_ep.eq->cm_ctx : NULL;
		ep_ctx->id->context = &ep->base_ep.util_ep.ep_fid.fid;
		/* Migrate to EQ channel if available, else fall back to domain
		 * bootstrap channel (eq_ctx is NULL before ep_bind for XRC). */
		ret = rdma_migrate_id(ep_ctx->id,
				      eq_ctx ? eq_ctx->cm_channel :
				      ((struct vrb_rdmacm_domain_ctx *)
					vrb_ep2_domain(&ep->base_ep)->cm_ctx)->cm_channel);
		if (ret) {
			VRB_WARN(FI_LOG_EP_CTRL,
				   "Failed to migrate active CM ID %d\n", ret);
			goto err;
		}

		ofi_straddr_dbg(&vrb_prov, FI_LOG_EP_CTRL, "XRC connect src_addr",
				rdma_get_local_addr(ep_ctx->id));
		ofi_straddr_dbg(&vrb_prov, FI_LOG_EP_CTRL, "XRC connect dest_addr",
				rdma_get_peer_addr(ep_ctx->id));

		ep->base_ep.ibv_qp = ep->ini_conn->ini_qp;
		ret = vrb_process_ini_conn(ep, ep->conn_setup->pending_recip,
					      ep->conn_setup->pending_param,
					      ep->conn_setup->pending_paramlen);
err:
		if (ret) {
			ep->ini_conn->state = last_state;
			_vrb_put_shared_ini_conn(ep);

			/* We need to let the application know that the
			 * connect request has failed. */
			vrb_create_shutdown_event(ep);
			break;
		}
	}
}
