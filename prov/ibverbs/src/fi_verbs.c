/*
 * Copyright (c) 2013-2014 Intel Corporation, Inc.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
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

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_prov.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_errno.h>
#include "fi.h"


struct ibv_domain {
	struct fid_domain	domain_fid;
	struct ibv_context	*verbs;
	struct ibv_pd		*pd;
};

struct ibv_eq {
	struct fid_eq		fid;
	enum fi_eq_domain	eq_domain;
	struct ibv_domain	*domain;
};

struct ibv_eq_comp {
	struct ibv_eq		eq;
	struct ibv_comp_channel	*channel;
	struct ibv_cq		*cq;
	uint64_t		flags;
	struct ibv_wc		wc;
};

struct ibv_eq_cm {
	struct ibv_eq		eq;
	struct rdma_event_channel *channel;
	uint64_t		flags;
	struct fi_eq_err_entry	err;
};

struct ibv_mem_desc {
	struct fid_mr		mr_fid;
	struct ibv_mr		*mr;
	struct ibv_domain	*domain;
};

struct ibv_msg_ep {
	struct fid_ep		ep_fid;
	struct rdma_cm_id	*id;
	struct ibv_eq_cm	*cm_eq;
	struct ibv_eq_comp	*req;
	struct ibv_eq_comp	*seq;
	uint32_t		inline_size;
};

static char def_send_wr[16] = "384";
static char def_recv_wr[16] = "384";
static char def_send_sge[16] = "4";
static char def_recv_sge[16] = "4";
static char def_inline_data[16] = "64";

static int ibv_check_hints(struct fi_info *hints)
{
	switch (hints->type) {
	case FID_UNSPEC:
	case FID_MSG:
	case FID_DGRAM:
		break;
	default:
		return -FI_ENODATA;
	}

	switch (hints->protocol) {
	case FI_PROTO_UNSPEC:
	case FI_PROTO_IB_RC:
	case FI_PROTO_IWARP:
	case FI_PROTO_IB_UC:
	case FI_PROTO_IB_UD:
		break;
	default:
		return -FI_ENODATA;
	}

	if ((hints->protocol_cap & (FI_PROTO_CAP_MSG | FI_PROTO_CAP_RMA)) !=
	    hints->protocol_cap)
		return -FI_ENODATA;

	if (hints->fabric_name && !strcmp(hints->fabric_name, "RDMA"))
		return -FI_ENODATA;

	return 0;
}

/*
 * TODO: this is not the full set of checks which are needed
 */
static int ibv_fi_to_rai(struct fi_info *fi, struct rdma_addrinfo *rai)
{
	memset(rai, 0, sizeof *rai);
	if (fi->flags & FI_PASSIVE)
		rai->ai_flags = RAI_PASSIVE;
	if (fi->flags & FI_NUMERICHOST)
		rai->ai_flags |= RAI_NUMERICHOST;
//	if (fi->flags & FI_FAMILY)
//		rai->ai_flags |= RAI_FAMILY;

//	rai->ai_family = fi->sa_family;
	if (fi->type == FID_MSG || fi->protocol_cap & FI_PROTO_CAP_RMA ||
	    fi->protocol == FI_PROTO_IB_RC || fi->protocol == FI_PROTO_IWARP) {
		rai->ai_qp_type = IBV_QPT_RC;
		rai->ai_port_space = RDMA_PS_TCP;
	} else if (fi->type == FID_DGRAM || fi->protocol == FI_PROTO_IB_UD) {
		rai->ai_qp_type = IBV_QPT_UD;
		rai->ai_port_space = RDMA_PS_UDP;
	}

	if (fi->src_addrlen) {
		if (!(rai->ai_src_addr = malloc(fi->src_addrlen)))
			return ENOMEM;
		memcpy(rai->ai_src_addr, fi->src_addr, fi->src_addrlen);
		rai->ai_src_len = fi->src_addrlen;
	}
	if (fi->dest_addrlen) {
		if (!(rai->ai_dst_addr = malloc(fi->dest_addrlen)))
			return ENOMEM;
		memcpy(rai->ai_dst_addr, fi->dest_addr, fi->dest_addrlen);
		rai->ai_dst_len = fi->dest_addrlen;
	}

	return 0;
}

 static int ibv_rai_to_fi(struct rdma_addrinfo *rai, struct fi_info *fi)
 {
 	memset(fi, 0, sizeof *fi);
 	if (rai->ai_flags & RAI_PASSIVE)
 		fi->flags = RAI_PASSIVE;

 //	fi->sa_family = rai->ai_family;
	if (rai->ai_qp_type == IBV_QPT_RC || rai->ai_port_space == RDMA_PS_TCP) {
		fi->protocol_cap = FI_PROTO_CAP_MSG | FI_PROTO_CAP_RMA;
		fi->type = FID_MSG;
	} else if (rai->ai_qp_type == IBV_QPT_UD ||
		   rai->ai_port_space == RDMA_PS_UDP) {
		fi->protocol = FI_PROTO_IB_UD;
		fi->protocol_cap = FI_PROTO_CAP_MSG;
		fi->type = FID_DGRAM;
	}

 	if (rai->ai_src_len) {
 		if (!(fi->src_addr = malloc(rai->ai_src_len)))
 			return ENOMEM;
 		memcpy(fi->src_addr, rai->ai_src_addr, rai->ai_src_len);
 		fi->src_addrlen = rai->ai_src_len;
 	}
 	if (rai->ai_dst_len) {
 		if (!(fi->dest_addr = malloc(rai->ai_dst_len)))
 			return ENOMEM;
 		memcpy(fi->dest_addr, rai->ai_dst_addr, rai->ai_dst_len);
 		fi->dest_addrlen = rai->ai_dst_len;
 	}

 	return 0;
 }

static int ibv_getinfo(const char *node, const char *service,
		       struct fi_info *hints, struct fi_info **info)
{
	struct rdma_addrinfo rai_hints, *rai;
	struct fi_info *fi;
	struct rdma_cm_id *id;
	int ret;

	if (hints) {
		ret = ibv_check_hints(hints);
		if (ret)
			return ret;

		ret = ibv_fi_to_rai(hints, &rai_hints);
		if (ret)
			return ret;

		ret = rdma_getaddrinfo((char *) node, (char *) service,
					&rai_hints, &rai);
	} else {
		ret = rdma_getaddrinfo((char *) node, (char *) service,
					NULL, &rai);
	}
	if (ret)
		return -errno;

	if (!(fi = malloc(sizeof *fi))) {
		ret = FI_ENOMEM;
		goto err1;
	}

	ret = ibv_rai_to_fi(rai, fi);
	if (ret)
		goto err2;

	ret = rdma_create_ep(&id, rai, NULL, NULL);
	if (ret) {
		ret = -errno;
		goto err2;
	}
	rdma_freeaddrinfo(rai);

	if (!fi->src_addr) {
		fi->src_addrlen = fi_sockaddr_len(rdma_get_local_addr(id));
		if (!(fi->src_addr = malloc(fi->src_addrlen))) {
			ret = -FI_ENOMEM;
			goto err3;
		}
		memcpy(fi->src_addr, rdma_get_local_addr(id), fi->src_addrlen);
	}

	if (id->verbs) {
		if (!(fi->domain_name = strdup(id->verbs->device->name))) {
			ret = -FI_ENOMEM;
			goto err3;
		}
	}

	// TODO: Get a real name here
	if (!(fi->fabric_name = strdup("RDMA"))) {
		ret = -FI_ENOMEM;
		goto err3;
	}

	fi->data = id;
	fi->datalen = sizeof id;
	*info = fi;
	return 0;

err3:
	rdma_destroy_ep(id);
err2:
	__fi_freeinfo(fi);
err1:
	rdma_freeaddrinfo(rai);
	return ret;
}

static int ibv_freeinfo(struct fi_info *info)
{
	if (!strcmp(info->fabric_name, "RDMA"))
		return -FI_ENODATA;

	if (info->data) {
		rdma_destroy_ep(info->data);
		info->data = NULL;
	}
	__fi_freeinfo(info);
	return 0;
}

static int ibv_msg_ep_create_qp(struct ibv_msg_ep *ep)
{
	struct ibv_qp_init_attr attr;

	/* TODO: serialize access to string buffers */
	fi_read_file(FI_CONF_DIR, "def_send_wr",
			def_send_wr, sizeof def_send_wr);
	fi_read_file(FI_CONF_DIR, "def_recv_wr",
			def_recv_wr, sizeof def_recv_wr);
	fi_read_file(FI_CONF_DIR, "def_send_sge",
			def_send_sge, sizeof def_send_sge);
	fi_read_file(FI_CONF_DIR, "def_recv_sge",
			def_recv_sge, sizeof def_recv_sge);
	fi_read_file(FI_CONF_DIR, "def_inline_data",
			def_inline_data, sizeof def_inline_data);

	attr.cap.max_send_wr = atoi(def_send_wr);
	attr.cap.max_recv_wr = atoi(def_recv_wr);
	attr.cap.max_send_sge = atoi(def_send_sge);
	attr.cap.max_recv_sge = atoi(def_recv_sge);
	if (!ep->inline_size)
		ep->inline_size = atoi(def_inline_data);
	attr.cap.max_inline_data = ep->inline_size;
	attr.qp_context = ep;
	attr.send_cq = ep->seq->cq;
	attr.recv_cq = ep->req->cq;
	attr.srq = NULL;
	attr.qp_type = IBV_QPT_RC;
	attr.sq_sig_all = 1;

	return rdma_create_qp(ep->id, ep->req->eq.domain->pd, &attr) ? -errno : 0;
}

static int ibv_msg_ep_bind(fid_t fid, struct fi_resource *fids, int nfids)
{
	struct ibv_msg_ep *ep;
	struct ibv_eq *eq;
	int i, ret;

	ep = container_of(fid, struct ibv_msg_ep, ep_fid.fid);
	for (i = 0; i < nfids; i++) {
		if (fids[i].fid->fclass != FID_CLASS_EQ)
			return -EINVAL;

		eq = container_of(fids[i].fid, struct ibv_eq, fid.fid);
		if (fids[i].flags & FI_RECV) {
			if (ep->req)
				return -EINVAL;
			ep->req = container_of(eq, struct ibv_eq_comp, eq);
		}
		if (fids[i].flags & FI_SEND) {
			if (ep->seq)
				return -EINVAL;
			ep->seq = container_of(eq, struct ibv_eq_comp, eq);
		}
		if (eq->eq_domain == FI_EQ_DOMAIN_CM) {
			ep->cm_eq = container_of(eq, struct ibv_eq_cm, eq);
			ret = rdma_migrate_id(ep->id, ep->cm_eq->channel);
			if (ret)
				return -errno;
		}
	}

	if (ep->seq && ep->req && !ep->id->qp) {
		ret = ibv_msg_ep_create_qp(ep);
		if (ret)
			return ret;
	}

	return 0;
}

static ssize_t
ibv_msg_ep_recv(struct fid_ep *ep, void *buf, size_t len,
		void *desc, void *context)
{
	struct ibv_msg_ep *_ep;
	struct ibv_recv_wr wr, *bad;
	struct ibv_sge sge;

	sge.addr = (uintptr_t) buf;
	sge.length = (uint32_t) len;
	sge.lkey = (uint32_t) (uintptr_t) desc;

	wr.wr_id = (uintptr_t) context;
	wr.next = NULL;
	wr.sg_list = &sge;
	wr.num_sge = 1;

	_ep = container_of(ep, struct ibv_msg_ep, ep_fid);
	return -ibv_post_recv(_ep->id->qp, &wr, &bad);
}

static ssize_t
ibv_msg_ep_send(struct fid_ep *ep, const void *buf, size_t len,
		void *desc, void *context)
{
	struct ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge sge;

	_ep = container_of(ep, struct ibv_msg_ep, ep_fid);
	sge.addr = (uintptr_t) buf;
	sge.length = (uint32_t) len;
	sge.lkey = (uint32_t) (uintptr_t) desc;

	wr.wr_id = (uintptr_t) context;
	wr.next = NULL;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.opcode = IBV_WR_SEND;
	wr.send_flags = (len <= _ep->inline_size) ? IBV_SEND_INLINE : 0;

	return -ibv_post_send(_ep->id->qp, &wr, &bad);
}

static ssize_t
ibv_msg_ep_sendmsg(struct fid_ep *ep, const struct fi_msg *msg, uint64_t flags)
{
	struct ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge *sge;
	size_t i, len;

	_ep = container_of(ep, struct ibv_msg_ep, ep_fid);
	wr.num_sge = msg->iov_count;
	if (msg->iov_count) {
		sge = alloca(sizeof(*sge) * msg->iov_count);
		for (len = 0, i = 0; i < msg->iov_count; i++) {
			sge[i].addr = (uintptr_t) msg->msg_iov[i].iov_base;
			sge[i].length = (uint32_t) msg->msg_iov[i].iov_len;
			sge[i].lkey = (uint32_t) (uintptr_t) (msg->desc + i);
			len += sge[i].length;
		}

		wr.sg_list = sge;
		wr.send_flags = (len <= _ep->inline_size) ? IBV_SEND_INLINE : 0;
	} else {
		wr.send_flags = 0;
	}

	wr.wr_id = (uintptr_t) msg->context;
	wr.next = NULL;
	if (flags & FI_IMM) {
		wr.opcode = IBV_WR_SEND_WITH_IMM;
		wr.imm_data = (uint32_t) msg->data;
	} else {
		wr.opcode = IBV_WR_SEND;
	}

	return -ibv_post_send(_ep->id->qp, &wr, &bad);
}

static struct fi_ops_msg ibv_msg_ep_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = ibv_msg_ep_recv,
	.send = ibv_msg_ep_send,
	.sendmsg = ibv_msg_ep_sendmsg,
};

static ssize_t
ibv_msg_ep_rma_write(struct fid_ep *ep, const void *buf, size_t len,
		     void *desc, uint64_t addr, uint64_t tag, void *context)
{
	struct ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge sge;

	_ep = container_of(ep, struct ibv_msg_ep, ep_fid);
	sge.addr = (uintptr_t) buf;
	sge.length = (uint32_t) len;
	sge.lkey = (uint32_t) (uintptr_t) desc;

	wr.wr_id = (uintptr_t) context;
	wr.next = NULL;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.opcode = IBV_WR_RDMA_WRITE;
	wr.send_flags = (len <= _ep->inline_size) ? IBV_SEND_INLINE : 0;
	wr.wr.rdma.remote_addr = addr;
	wr.wr.rdma.rkey = (uint32_t) tag;

	return -ibv_post_send(_ep->id->qp, &wr, &bad);
}

static ssize_t
ibv_msg_ep_rma_read(struct fid_ep *ep, void *buf, size_t len,
		    void *desc, uint64_t addr, uint64_t tag, void *context)
{
	struct ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge sge;

	sge.addr = (uintptr_t) buf;
	sge.length = (uint32_t) len;
	sge.lkey = (uint32_t) (uintptr_t) desc;

	wr.wr_id = (uintptr_t) context;
	wr.next = NULL;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.opcode = IBV_WR_RDMA_READ;
	wr.send_flags = 0;
	wr.wr.rdma.remote_addr = addr;
	wr.wr.rdma.rkey = (uint32_t) tag;

	_ep = container_of(ep, struct ibv_msg_ep, ep_fid);
	return -ibv_post_send(_ep->id->qp, &wr, &bad);
}

static struct fi_ops_rma ibv_msg_ep_rma_ops = {
	.size = sizeof(struct fi_ops_rma),
	.write = ibv_msg_ep_rma_write,
	.read = ibv_msg_ep_rma_read
};

static int
ibv_msg_ep_connect(struct fid_ep *ep, const void *addr,
		   const void *param, size_t paramlen)
{
	struct ibv_msg_ep *_ep;
	struct rdma_conn_param conn_param;

	_ep = container_of(ep, struct ibv_msg_ep, ep_fid);
	memset(&conn_param, 0, sizeof conn_param);
	conn_param.private_data = param;
	conn_param.private_data_len = paramlen;
	conn_param.responder_resources = RDMA_MAX_RESP_RES;
	conn_param.initiator_depth = RDMA_MAX_INIT_DEPTH;
	conn_param.flow_control = 1;
	conn_param.retry_count = 15;
	conn_param.rnr_retry_count = 7;

	return rdma_connect(_ep->id, &conn_param) ? -errno : 0;
}

static int
ibv_msg_ep_accept(struct fid_ep *ep, const void *param, size_t paramlen)
{
	struct ibv_msg_ep *_ep;
	struct rdma_conn_param conn_param;

	_ep = container_of(ep, struct ibv_msg_ep, ep_fid);
	memset(&conn_param, 0, sizeof conn_param);
	conn_param.private_data = param;
	conn_param.private_data_len = paramlen;
	conn_param.responder_resources = RDMA_MAX_RESP_RES;
	conn_param.initiator_depth = RDMA_MAX_INIT_DEPTH;
	conn_param.flow_control = 1;
	conn_param.rnr_retry_count = 7;

	return rdma_accept(_ep->id, &conn_param) ? -errno : 0;
}

static int
ibv_msg_ep_reject(struct fid_pep *pep, struct fi_info *info,
		  const void *param, size_t paramlen)
{
	return rdma_reject(info->data, param, (uint8_t) paramlen) ? -errno : 0;
}

static int ibv_msg_ep_shutdown(struct fid_ep *ep, uint64_t flags)
{
	struct ibv_msg_ep *_ep;
	_ep = container_of(ep, struct ibv_msg_ep, ep_fid);
	return rdma_disconnect(_ep->id) ? -errno : 0;
}

static struct fi_ops_cm ibv_msg_ep_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.connect = ibv_msg_ep_connect,
	.accept = ibv_msg_ep_accept,
	.reject = ibv_msg_ep_reject,
	.shutdown = ibv_msg_ep_shutdown,
};

static int
ibv_msg_ep_getopt(fid_t fid, int level, int optname,
		  void *optval, size_t *optlen)
{
	struct ibv_msg_ep *ep;
	ep = container_of(fid, struct ibv_msg_ep, ep_fid.fid);

	switch (level) {
	case FI_OPT_ENDPOINT:
		switch (optname) {
		case FI_OPT_MAX_BUFFERED_SEND:
			if (*optlen < sizeof(size_t)) {
				*optlen = sizeof(size_t);
				return -FI_ETOOSMALL;
			}
			*((size_t *) optval) = (size_t) ep->inline_size;
			*optlen = sizeof(size_t);
			break;
		default:
			return -FI_ENOPROTOOPT;
		}
	default:
		return -FI_ENOPROTOOPT;
	}
	return 0;
}

static int
ibv_msg_ep_setopt(fid_t fid, int level, int optname,
		  const void *optval, size_t optlen)
{
	struct ibv_msg_ep *ep;
	ep = container_of(fid, struct ibv_msg_ep, ep_fid.fid);

	switch (level) {
	case FI_OPT_ENDPOINT:
		switch (optname) {
		case FI_OPT_MAX_BUFFERED_SEND:
			if (optlen != sizeof(size_t))
				return -FI_EINVAL;
			if (ep->id->qp)
				return -FI_EOPBADSTATE;
			ep->inline_size = (uint32_t) *(size_t *) optval;
			break;
		default:
			return -FI_ENOPROTOOPT;
		}
	default:
		return -FI_ENOPROTOOPT;
	}
	return 0;
}

static int ibv_msg_ep_enable(struct fid_ep *ep)
{
	return 0;
}

static struct fi_ops_ep ibv_msg_ep_base_ops = {
	.size = sizeof(struct fi_ops_ep),
	.enable = ibv_msg_ep_enable,
	.getopt = ibv_msg_ep_getopt,
	.setopt = ibv_msg_ep_setopt,
};

static int ibv_msg_ep_close(fid_t fid)
{
	struct ibv_msg_ep *ep;

	ep = container_of(fid, struct ibv_msg_ep, ep_fid.fid);
	if (ep->id)
		rdma_destroy_ep(ep->id);

	free(ep);
	return 0;
}

static struct fi_ops ibv_msg_ep_ops = {
	.size = sizeof(struct fi_ops),
	.close = ibv_msg_ep_close,
	.bind = ibv_msg_ep_bind
};

static int
ibv_open_ep(struct fid_domain *domain, struct fi_info *info,
	    struct fid_ep **ep, void *context)
{
	struct ibv_domain *_domain;
	struct ibv_msg_ep *_ep;

	_domain = container_of(domain, struct ibv_domain, domain_fid);
	if (strcmp(info->fabric_name, "RDMA") ||
	    strcmp(_domain->verbs->device->name, info->domain_name))
		return -FI_EINVAL;

	if (!info->data || info->datalen != sizeof(_ep->id))
		return -FI_ENOSYS;

	_ep = calloc(1, sizeof *_ep);
	if (!_ep)
		return -FI_ENOMEM;

	_ep->id = info->data;
	_ep->id->context = &_ep->ep_fid.fid;
	info->data = NULL;
	info->datalen = 0;

	_ep->ep_fid.fid.fclass = FID_CLASS_EP;
	_ep->ep_fid.fid.size = sizeof(struct fid_ep);
	_ep->ep_fid.fid.context = context;
	_ep->ep_fid.fid.ops = &ibv_msg_ep_ops;
	_ep->ep_fid.ops = &ibv_msg_ep_base_ops;
	_ep->ep_fid.msg = &ibv_msg_ep_msg_ops;
	_ep->ep_fid.cm = &ibv_msg_ep_cm_ops;
	_ep->ep_fid.rma = &ibv_msg_ep_rma_ops;

	*ep = &_ep->ep_fid;
	return 0;
}

static ssize_t
ibv_eq_cm_readerr(struct fid_eq *eq, struct fi_eq_err_entry *entry, size_t len,
		  uint64_t flags)
{
	struct ibv_eq_cm *_eq;

	_eq = container_of(eq, struct ibv_eq_cm, eq.fid);
	if (!_eq->err.err)
		return 0;

	if (len < sizeof(*entry))
		return -FI_EINVAL;

	*entry = _eq->err;
	_eq->err.err = 0;
	_eq->err.prov_errno = 0;
	return sizeof(*entry);
}

static struct fi_info * ibv_eq_cm_getinfo(struct rdma_cm_event *event)
{
	struct fi_info *fi;

	fi = calloc(1, sizeof *fi);
	if (!fi)
		return NULL;

	fi->size = sizeof *fi;
	fi->type = FID_MSG;
	fi->protocol_cap  = FI_PROTO_CAP_MSG | FI_PROTO_CAP_RMA;
	if (event->id->verbs->device->transport_type == IBV_TRANSPORT_IWARP) {
		fi->protocol = FI_PROTO_IWARP;
	} else {
		fi->protocol = FI_PROTO_IB_RC;
	}
//	fi->sa_family = rdma_get_local_addr(event->id)->sa_family;

	fi->src_addrlen = fi_sockaddr_len(rdma_get_local_addr(event->id));
	if (!(fi->src_addr = malloc(fi->src_addrlen)))
		goto err;
	memcpy(fi->src_addr, rdma_get_local_addr(event->id), fi->src_addrlen);

	fi->dest_addrlen = fi_sockaddr_len(rdma_get_peer_addr(event->id));
	if (!(fi->dest_addr = malloc(fi->dest_addrlen)))
		goto err;
	memcpy(fi->dest_addr, rdma_get_peer_addr(event->id), fi->dest_addrlen);

	if (!(fi->fabric_name = strdup("RDMA")))
		goto err;
	if (!(fi->domain_name = strdup(event->id->verbs->device->name)))
		goto err;

	fi->datalen = sizeof event->id;
	fi->data = event->id;
	return fi;
err:
	fi_freeinfo(fi);
	return NULL;
}

static ssize_t
ibv_eq_cm_process_event(struct ibv_eq_cm *eq, struct rdma_cm_event *event,
			struct fi_eq_cm_entry *entry, size_t len)
{
	fid_t fid;
	size_t datalen;

	fid = event->id->context;
	switch (event->event) {
//	case RDMA_CM_EVENT_ADDR_RESOLVED:
//		return 0;
//	case RDMA_CM_EVENT_ROUTE_RESOLVED:
//		return 0;
	case RDMA_CM_EVENT_CONNECT_REQUEST:
		rdma_migrate_id(event->id, NULL);
		entry->event = FI_CONNREQ;
		entry->info = ibv_eq_cm_getinfo(event);
		if (!entry->info) {
			rdma_destroy_id(event->id);
			return 0;
		}
		break;
	case RDMA_CM_EVENT_ESTABLISHED:
		entry->event = FI_CONNECTED;
		entry->info = NULL;
		break;
	case RDMA_CM_EVENT_DISCONNECTED:
		entry->event = FI_SHUTDOWN;
		entry->info = NULL;
		break;
	case RDMA_CM_EVENT_ADDR_ERROR:
	case RDMA_CM_EVENT_ROUTE_ERROR:
	case RDMA_CM_EVENT_CONNECT_ERROR:
	case RDMA_CM_EVENT_UNREACHABLE:
		eq->err.fid_context = fid->context;
		eq->err.err = event->status;
		return -EIO;
	case RDMA_CM_EVENT_REJECTED:
		eq->err.fid_context = fid->context;
		eq->err.err = ECONNREFUSED;
		eq->err.prov_errno = event->status;
		return -EIO;
	case RDMA_CM_EVENT_DEVICE_REMOVAL:
		eq->err.fid_context = fid->context;
		eq->err.err = ENODEV;
		return -EIO;
	case RDMA_CM_EVENT_ADDR_CHANGE:
		eq->err.fid_context = fid->context;
		eq->err.err = EADDRNOTAVAIL;
		return -EIO;
	default:
		return 0;
	}

	entry->fid_context = fid->context;
	entry->flags = 0;
	datalen = min(len - sizeof(*entry), event->param.conn.private_data_len);
	if (datalen)
		memcpy(entry->data, event->param.conn.private_data, datalen);
	return sizeof(*entry) + datalen;
}

static ssize_t ibv_eq_cm_read_data(struct fid_eq *eq, void *buf, size_t len)
{
	struct ibv_eq_cm *_eq;
	struct fi_eq_cm_entry *entry;
	struct rdma_cm_event *event;
	size_t left;
	ssize_t ret = -EINVAL;

	_eq = container_of(eq, struct ibv_eq_cm, eq.fid);
	entry = (struct fi_eq_cm_entry *) buf;
	if (_eq->err.err)
		return -EIO;

	for (left = len; left >= sizeof(*entry); ) {
		ret = rdma_get_cm_event(_eq->channel, &event);
		if (!ret) {
			ret = ibv_eq_cm_process_event(_eq, event, entry, left);
			rdma_ack_cm_event(event);
			if (ret < 0)
				break;
			else if (!ret)
				continue;

			left -= ret;
			entry = ((void *) entry) + ret;
		} else if (errno == EAGAIN) {
			if (left < len)
				return len - left;

			if (!(_eq->flags & FI_BLOCK))
				return 0;

			fi_poll_fd(_eq->channel->fd);
		} else {
			ret = -errno;
			break;
		}
	}

	return (left < len) ? len - left : ret;
}

static const char *
ibv_eq_cm_strerror(struct fid_eq *eq, int prov_errno, const void *prov_data,
		   void *buf, size_t len)
{
	if (buf && len)
		strncpy(buf, strerror(prov_errno), len);
	return strerror(prov_errno);
}

static struct fi_ops_eq ibv_eq_cm_data_ops = {
	.size = sizeof(struct fi_ops_eq),
	.read = ibv_eq_cm_read_data,
	.readerr = ibv_eq_cm_readerr,
	.strerror = ibv_eq_cm_strerror
};

static int ibv_eq_cm_control(fid_t fid, int command, void *arg)
{
	struct ibv_eq_cm *eq;
	int ret = 0;

	eq = container_of(fid, struct ibv_eq_cm, eq.fid.fid);
	switch(command) {
	case FI_GETWAIT:
		if (!eq->channel) {
			ret = -FI_ENODATA;
			break;
		}
		*(void **) arg = &eq->channel->fd;
		break;
	default:
		ret = -FI_ENOSYS;
		break;
	}

	return ret;
}

static int ibv_eq_cm_close(fid_t fid)
{
	struct ibv_eq_cm *eq;

	eq = container_of(fid, struct ibv_eq_cm, eq.fid.fid);
	if (eq->channel)
		rdma_destroy_event_channel(eq->channel);

	free(eq);
	return 0;
}

static struct fi_ops ibv_eq_cm_ops = {
	.size = sizeof(struct fi_ops),
	.close = ibv_eq_cm_close,
	.control = ibv_eq_cm_control,
};

static int
ibv_eq_cm_open(struct fid_domain *domain, struct fi_eq_attr *attr,
	       struct fid_eq **eq, void *context)
{
	struct ibv_eq_cm *_eq;
	long flags = 0;
	int ret;

	if (attr->format != FI_EQ_FORMAT_CM)
		return -ENOSYS;

	_eq = calloc(1, sizeof *_eq);
	if (!_eq)
		return -ENOMEM;

	_eq->eq.domain = container_of(domain, struct ibv_domain, domain_fid);

	switch (attr->wait_obj) {
	case FI_EQ_WAIT_FD:
		_eq->channel = rdma_create_event_channel();
		if (!_eq->channel) {
			ret = -errno;
			goto err1;
		}
		fcntl(_eq->channel->fd, F_GETFL, &flags);
		ret = fcntl(_eq->channel->fd, F_SETFL, flags | O_NONBLOCK);
		if (ret) {
			ret = -errno;
			goto err2;
		}
		break;
	case FI_EQ_WAIT_NONE:
		break;
	default:
		return -ENOSYS;
	}

	_eq->flags = attr->flags;
	_eq->eq.fid.fid.fclass = FID_CLASS_EQ;
	_eq->eq.fid.fid.size = sizeof(struct fid_eq);
	_eq->eq.fid.fid.context = context;
	_eq->eq.fid.fid.ops = &ibv_eq_cm_ops;
	_eq->eq.fid.ops = &ibv_eq_cm_data_ops;

	*eq = &_eq->eq.fid;
	return 0;
err2:
	if (_eq->channel)
		rdma_destroy_event_channel(_eq->channel);
err1:
	free(_eq);
	return ret;
}

static int ibv_eq_comp_reset(struct fid_eq *eq, const void *cond)
{
	struct ibv_eq_comp *_eq;
	struct ibv_cq *cq;
	void *context;
	int ret;

	_eq = container_of(eq, struct ibv_eq_comp, eq.fid);
	ret = ibv_get_cq_event(_eq->channel, &cq, &context);
	if (!ret)
		ibv_ack_cq_events(cq, 1);

	return -ibv_req_notify_cq(_eq->cq, (_eq->flags & FI_REMOTE_SIGNAL) ? 1 : 0);
}

static ssize_t
ibv_eq_comp_readerr(struct fid_eq *eq, struct fi_eq_err_entry *entry,
		    size_t len, uint64_t flags)
{
	struct ibv_eq_comp *_eq;

	_eq = container_of(eq, struct ibv_eq_comp, eq.fid);
	if (!_eq->wc.status)
		return 0;

	if (len < sizeof(*entry))
		return -EINVAL;

	entry->fid_context = NULL;	/* TODO: return qp context from wc */
	entry->op_context = (void *) (uintptr_t) _eq->wc.wr_id;
	entry->flags = 0;
	entry->err = EIO;
	entry->prov_errno = _eq->wc.status;
	entry->data = _eq->wc.vendor_err;
	entry->prov_data = NULL;

	_eq->wc.status = 0;
	return sizeof(*entry);
}

static ssize_t ibv_eq_comp_read_context(struct fid_eq *eq, void *buf, size_t len)
{
	struct ibv_eq_comp *_eq;
	struct fi_eq_entry *entry;
	size_t left;
	int reset = 1, ret = -EINVAL;

	_eq = container_of(eq, struct ibv_eq_comp, eq.fid);
	entry = (struct fi_eq_entry *) buf;
	if (_eq->wc.status)
		return -EIO;

	for (left = len; left >= sizeof(*entry); ) {
		ret = ibv_poll_cq(_eq->cq, 1, &_eq->wc);
		if (ret > 0) {
			if (_eq->wc.status) {
				ret = -EIO;
				break;
			}

			entry->op_context = (void *) (uintptr_t) _eq->wc.wr_id;
			left -= sizeof(*entry);
			entry = entry + 1;
		} else if (ret == 0) {
			if (left < len)
				return len - left;

			if (reset && (_eq->flags & FI_AUTO_RESET)) {
				ibv_eq_comp_reset(eq, NULL);
				reset = 0;
				continue;
			}

			if (!(_eq->flags & FI_BLOCK))
				return 0;

			fi_poll_fd(_eq->channel->fd);
		} else {
			break;
		}
	}

	return (left < len) ? len - left : ret;
}

static ssize_t ibv_eq_comp_read_comp(struct fid_eq *eq, void *buf, size_t len)
{
	struct ibv_eq_comp *_eq;
	struct fi_eq_comp_entry *entry;
	size_t left;
	int reset = 1, ret = -EINVAL;

	_eq = container_of(eq, struct ibv_eq_comp, eq.fid);
	entry = (struct fi_eq_comp_entry *) buf;
	if (_eq->wc.status)
		return -EIO;

	for (left = len; left >= sizeof(*entry); ) {
		ret = ibv_poll_cq(_eq->cq, 1, &_eq->wc);
		if (ret > 0) {
			if (_eq->wc.status) {
				ret = -EIO;
				break;
			}

			entry->op_context = (void *) (uintptr_t) _eq->wc.wr_id;
			entry->flags = (uint64_t) _eq->wc.wc_flags;
			entry->len = (uint64_t) _eq->wc.byte_len;
			left -= sizeof(*entry);
			entry = entry + 1;
		} else if (ret == 0) {
			if (left < len)
				return len - left;

			if (reset && (_eq->flags & FI_AUTO_RESET)) {
				ibv_eq_comp_reset(eq, NULL);
				reset = 0;
				continue;
			}

			if (!(_eq->flags & FI_BLOCK))
				return 0;

			fi_poll_fd(_eq->channel->fd);
		} else {
			break;
		}
	}

	return (left < len) ? len - left : ret;
}

static ssize_t ibv_eq_comp_read_data(struct fid_eq *eq, void *buf, size_t len)
{
	struct ibv_eq_comp *_eq;
	struct fi_eq_data_entry *entry;
	size_t left;
	int reset = 1, ret = -EINVAL;

	_eq = container_of(eq, struct ibv_eq_comp, eq.fid);
	entry = (struct fi_eq_data_entry *) buf;
	if (_eq->wc.status)
		return -EIO;

	for (left = len; left >= sizeof(*entry); ) {
		ret = ibv_poll_cq(_eq->cq, 1, &_eq->wc);
		if (ret > 0) {
			if (_eq->wc.status) {
				ret = -EIO;
				break;
			}

			entry->op_context = (void *) (uintptr_t) _eq->wc.wr_id;
			if (_eq->wc.wc_flags & IBV_WC_WITH_IMM) {
				entry->flags = FI_IMM;
				entry->data = _eq->wc.imm_data;
			}
			if (_eq->wc.opcode & IBV_WC_RECV)
				entry->len = _eq->wc.byte_len;
			left -= sizeof(*entry);
			entry = entry + 1;
		} else if (ret == 0) {
			if (left < len)
				return len - left;

			if (reset && (_eq->flags & FI_AUTO_RESET)) {
				ibv_eq_comp_reset(eq, NULL);
				reset = 0;
				continue;
			}

			if (!(_eq->flags & FI_BLOCK))
				return 0;

			fi_poll_fd(_eq->channel->fd);
		} else {
			break;
		}
	}

	return (left < len) ? len - left : ret;
}

static const char *
ibv_eq_comp_strerror(struct fid_eq *eq, int prov_errno, const void *prov_data,
		     void *buf, size_t len)
{
	if (buf && len)
		strncpy(buf, ibv_wc_status_str(prov_errno), len);
	return ibv_wc_status_str(prov_errno);
}

static struct fi_ops_eq ibv_eq_comp_context_ops = {
	.size = sizeof(struct fi_ops_eq),
	.read = ibv_eq_comp_read_context,
	.readerr = ibv_eq_comp_readerr,
	.reset = ibv_eq_comp_reset,
	.strerror = ibv_eq_comp_strerror
};

static struct fi_ops_eq ibv_eq_comp_comp_ops = {
	.size = sizeof(struct fi_ops_eq),
	.read = ibv_eq_comp_read_comp,
	.readerr = ibv_eq_comp_readerr,
	.reset = ibv_eq_comp_reset,
	.strerror = ibv_eq_comp_strerror
};

static struct fi_ops_eq ibv_eq_comp_data_ops = {
	.size = sizeof(struct fi_ops_eq),
	.read = ibv_eq_comp_read_data,
	.readerr = ibv_eq_comp_readerr,
	.reset = ibv_eq_comp_reset,
	.strerror = ibv_eq_comp_strerror
};

static int ibv_eq_comp_control(fid_t fid, int command, void *arg)
{
	struct ibv_eq_comp *eq;
	int ret = 0;

	eq = container_of(fid, struct ibv_eq_comp, eq.fid.fid);
	switch(command) {
	case FI_GETWAIT:
		if (!eq->channel) {
			ret = -FI_ENODATA;
			break;
		}
		*(void **) arg = &eq->channel->fd;
		break;
	default:
		ret = -FI_ENOSYS;
		break;
	}

	return ret;
}

static int ibv_eq_comp_close(fid_t fid)
{
	struct ibv_eq_comp *eq;
	int ret;

	eq = container_of(fid, struct ibv_eq_comp, eq.fid.fid);
	if (eq->cq) {
		ret = ibv_destroy_cq(eq->cq);
		if (ret)
			return -ret;
		eq->cq = NULL;
	}
	if (eq->channel)
		ibv_destroy_comp_channel(eq->channel);

	free(eq);
	return 0;
}

static struct fi_ops ibv_eq_comp_ops = {
	.size = sizeof(struct fi_ops),
	.close = ibv_eq_comp_close,
	.control = ibv_eq_comp_control,
};

static int
ibv_eq_comp_open(struct fid_domain *domain, struct fi_eq_attr *attr,
		 struct fid_eq **eq, void *context)
{
	struct ibv_eq_comp *_eq;
	long flags = 0;
	int ret;

	if (attr->wait_cond != FI_EQ_COND_NONE)
		return -ENOSYS;

	_eq = calloc(1, sizeof *_eq);
	if (!_eq)
		return -ENOMEM;

	_eq->eq.domain = container_of(domain, struct ibv_domain, domain_fid);

	switch (attr->wait_obj) {
	case FI_EQ_WAIT_FD:
		_eq->channel = ibv_create_comp_channel(_eq->eq.domain->verbs);
		if (!_eq->channel) {
			ret = -errno;
			goto err1;
		}
		fcntl(_eq->channel->fd, F_GETFL, &flags);
		ret = fcntl(_eq->channel->fd, F_SETFL, flags | O_NONBLOCK);
		if (ret) {
			ret = -errno;
			goto err1;
		}
		break;
	case FI_EQ_WAIT_NONE:
		break;
	default:
		return -ENOSYS;
	}

	_eq->cq = ibv_create_cq(_eq->eq.domain->verbs, attr->size, _eq,
				_eq->channel, attr->signaling_vector);
	if (!_eq->cq) {
		ret = -errno;
		goto err2;
	}

	_eq->flags |= attr->flags;
	_eq->eq.fid.fid.fclass = FID_CLASS_EQ;
	_eq->eq.fid.fid.size = sizeof(struct fid_eq);
	_eq->eq.fid.fid.context = context;
	_eq->eq.fid.fid.ops = &ibv_eq_comp_ops;

	switch (attr->format) {
	case FI_EQ_FORMAT_CONTEXT:
		_eq->eq.fid.ops = &ibv_eq_comp_context_ops;
		break;
	case FI_EQ_FORMAT_COMP:
		_eq->eq.fid.ops = &ibv_eq_comp_comp_ops;
		break;
	case FI_EQ_FORMAT_DATA:
		_eq->eq.fid.ops = &ibv_eq_comp_data_ops;
		break;
	default:
		ret = -ENOSYS;
		goto err3;
	}

	*eq = &_eq->eq.fid;
	return 0;

err3:
	ibv_destroy_cq(_eq->cq);
err2:
	if (_eq->channel)
		ibv_destroy_comp_channel(_eq->channel);
err1:
	free(_eq);
	return ret;
}

static int
ibv_eq_open(struct fid_domain *domain, struct fi_eq_attr *attr,
	    struct fid_eq **eq, void *context)
{
	struct ibv_eq *_eq;
	int ret;

	switch (attr->domain) {
	case FI_EQ_DOMAIN_GENERAL:
		return -ENOSYS;
	case FI_EQ_DOMAIN_COMP:
		ret = ibv_eq_comp_open(domain, attr, eq, context);
		break;
	case FI_EQ_DOMAIN_CM:
		ret  = ibv_eq_cm_open(domain, attr, eq, context);
		break;
	case FI_EQ_DOMAIN_AV:
		return -ENOSYS;
	default:
		return -ENOSYS;
	}
	if (ret)
		return ret;

	_eq = container_of(*eq, struct ibv_eq, fid);
	_eq->eq_domain = attr->domain;

	if (attr->flags & FI_AUTO_RESET && _eq->fid.ops->reset)
		fi_eq_reset(*eq, attr->cond);

	return 0;
}

static int ibv_mr_close(fid_t fid)
{
	struct ibv_mem_desc *mr;
	int ret;

	mr = container_of(fid, struct ibv_mem_desc, mr_fid.fid);
	ret = -ibv_dereg_mr(mr->mr);
	if (!ret)
		free(mr);
	return ret;
}

static struct fi_ops ibv_mr_ops = {
	.size = sizeof(struct fi_ops),
	.close = ibv_mr_close
};

static int
ibv_mr_check_reg(uint64_t access, uint64_t requested_key, uint64_t flags)
{
	if (!(flags & FI_BLOCK))
		return -FI_EBADFLAGS;

	if (requested_key)
		return -FI_ENOSYS;

	return 0;
}

static int
ibv_mr_reg(struct fid_domain *domain, const void *buf, size_t len,
	   uint64_t access, uint64_t requested_key,
	   uint64_t flags, struct fid_mr **mr, void *context)
{
	struct ibv_mem_desc *md;
	int ibv_access, ret;

	ret = ibv_mr_check_reg(access, requested_key, flags);
	if (ret)
		return ret;

	md = calloc(1, sizeof *md);
	if (!md)
		return -FI_ENOMEM;

	md->domain = container_of(domain, struct ibv_domain, domain_fid);
	md->mr_fid.fid.fclass = FID_CLASS_MR;
	md->mr_fid.fid.size = sizeof(struct fid_mr);
	md->mr_fid.fid.context = context;
	md->mr_fid.fid.ops = &ibv_mr_ops;

	ibv_access = IBV_ACCESS_LOCAL_WRITE;
	if (access & FI_REMOTE_READ)
		ibv_access |= IBV_ACCESS_REMOTE_READ;
	if (access & FI_REMOTE_WRITE)
		ibv_access |= IBV_ACCESS_REMOTE_WRITE;

	md->mr = ibv_reg_mr(md->domain->pd, (void *) buf, len, ibv_access);
	if (!md->mr)
		goto err;

	md->mr_fid.mem_desc = (void *) (uintptr_t) md->mr->lkey;
	md->mr_fid.key = md->mr->rkey;
	*mr = &md->mr_fid;
	return 0;

err:
	free(md);
	return -errno;
}

static int ibv_close(fid_t fid)
{
	struct ibv_domain *domain;
	int ret;

	domain = container_of(fid, struct ibv_domain, domain_fid.fid);
	if (domain->pd) {
		ret = ibv_dealloc_pd(domain->pd);
		if (ret)
			return -ret;
		domain->pd = NULL;
	}

	free(domain);
	return 0;
}

static int ibv_open_device_by_name(struct ibv_domain *domain, const char *name)
{
	struct ibv_context **dev_list;
	int i, ret = -FI_ENODEV;

	dev_list = rdma_get_devices(NULL);
	if (!dev_list)
		return -errno;

	for (i = 0; dev_list[i]; i++) {
		if (!strcmp(name, ibv_get_device_name(dev_list[i]->device))) {
			domain->verbs = dev_list[i];
			ret = 0;
			break;
		}
	}
	rdma_free_devices(dev_list);
	return ret;
}

static struct fi_ops ibv_fid_ops = {
	.size = sizeof(struct fi_ops),
	.close = ibv_close,
};

static struct fi_ops_mr ibv_domain_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = ibv_mr_reg,
};

static struct fi_ops_domain ibv_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.eq_open = ibv_eq_open,
	.endpoint = ibv_open_ep,
};

static int
ibv_domain(struct fid_fabric *fabric, struct fi_info *info,
	   struct fid_domain **domain, void *context)
{
	struct ibv_domain *_domain;
	int ret;

	if (strcmp(info->fabric_name, "RDMA"))
		return -FI_EINVAL;

	_domain = calloc(1, sizeof *_domain);
	if (!_domain)
		return -FI_ENOMEM;

	ret = ibv_open_device_by_name(_domain, info->domain_name);
	if (ret)
		goto err;

	_domain->pd = ibv_alloc_pd(_domain->verbs);
	if (!_domain->pd) {
		ret = -errno;
		goto err;
	}

	_domain->domain_fid.fid.fclass = FID_CLASS_DOMAIN;
	_domain->domain_fid.fid.size = sizeof(struct fid_domain);
	_domain->domain_fid.fid.context = context;
	_domain->domain_fid.fid.ops = &ibv_fid_ops;
	_domain->domain_fid.ops = &ibv_domain_ops;
	_domain->domain_fid.mr = &ibv_domain_mr_ops;

	*domain = &_domain->domain_fid;
	return 0;
err:
	free(_domain);
	return ret;
}

static struct fi_ops_prov ibv_ops = {
	.size = sizeof(struct fi_ops_prov),
	.getinfo = ibv_getinfo,
	.freeinfo = ibv_freeinfo,
	.domain = ibv_domain,
};


void ibv_ini(void)
{
	fi_register(&ibv_ops);
}

void ibv_fini(void)
{
}
