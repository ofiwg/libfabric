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

#include <asm/types.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <infiniband/ib.h>
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


struct ibv_fabric {
	struct fid_fabric	fabric_fid;
	uint64_t		flags;
	char			name[FI_NAME_MAX];
};

struct ibv_feq_cm {
	struct fid_eq		eq_fid;
	struct ibv_fabric	*fab;
	struct rdma_event_channel *channel;
	uint64_t		flags;
	struct fi_eq_err_entry	err;
};

struct ibv_pep {
	struct fid_pep		pep_fid;
	struct ibv_feq_cm	*cm_eq;
	struct rdma_cm_id	*id;
};

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

static int ibv_sockaddr_len(struct sockaddr *addr)
{
	if (!addr)
		return 0;

	switch (addr->sa_family) {
	case AF_INET:
		return sizeof(struct sockaddr_in);
	case AF_INET6:
		return sizeof(struct sockaddr_in6);
#ifdef HAVE_VERBS
	case AF_IB:
		return sizeof(struct sockaddr_ib);
#endif // HAVE_VERBS
	default:
		return 0;
	}
}

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

	if ( !(hints->ep_cap & (FI_MSG | FI_RMA)) )
		return -FI_ENODATA;

	if (hints->fabric_name && strcmp(hints->fabric_name, "RDMA"))
		return -FI_ENODATA;

	return 0;
}

/*
 * TODO: this is not the full set of checks which are needed
 */
static int ibv_fi_to_rai(struct fi_info *fi, uint64_t flags, struct rdma_addrinfo *rai)
{
	memset(rai, 0, sizeof *rai);
	if ((fi->ep_cap & FI_PASSIVE) || (flags & FI_SOURCE))
		rai->ai_flags = RAI_PASSIVE;
	if (flags & FI_NUMERICHOST)
		rai->ai_flags |= RAI_NUMERICHOST;
//	if (fi->flags & FI_FAMILY)
//		rai->ai_flags |= RAI_FAMILY;

//	rai->ai_family = fi->sa_family;
	if (fi->type == FID_MSG || fi->ep_cap & FI_RMA ||
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

 static int ibv_rai_to_fi(struct rdma_addrinfo *rai, struct fi_info *hints,
		 	  struct fi_info *fi)
 {
 	memset(fi, 0, sizeof *fi);
 	if ((hints->ep_cap & FI_PASSIVE) && (rai->ai_flags & RAI_PASSIVE))
 		fi->ep_cap = FI_PASSIVE;

 //	fi->sa_family = rai->ai_family;
	if (rai->ai_qp_type == IBV_QPT_RC || rai->ai_port_space == RDMA_PS_TCP) {
		fi->ep_cap = FI_MSG | FI_RMA;
		fi->type = FID_MSG;
	} else if (rai->ai_qp_type == IBV_QPT_UD ||
		   rai->ai_port_space == RDMA_PS_UDP) {
		fi->protocol = FI_PROTO_IB_UD;
		fi->ep_cap = FI_MSG;
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

static int ibv_getinfo(int version, const char *node, const char *service,
		       uint64_t flags, struct fi_info *hints, struct fi_info **info)
{
	struct rdma_addrinfo rai_hints, *rai;
	struct fi_info *fi;
	struct rdma_cm_id *id;
	int ret;

	if (hints) {
		ret = ibv_check_hints(hints);
		if (ret)
			return ret;

		ret = ibv_fi_to_rai(hints, flags, &rai_hints);
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

	ret = ibv_rai_to_fi(rai, hints, fi);
	if (ret)
		goto err2;

	ret = rdma_create_ep(&id, rai, NULL, NULL);
	if (ret) {
		ret = -errno;
		goto err2;
	}
	rdma_freeaddrinfo(rai);

	if (!fi->src_addr) {
		fi->src_addrlen = ibv_sockaddr_len(rdma_get_local_addr(id));
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
	if (info->fabric_name && strcmp(info->fabric_name, "RDMA"))
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

static int ibv_msg_ep_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct ibv_msg_ep *ep;
	struct ibv_eq *eq;
	int ret;

	ep = container_of(fid, struct ibv_msg_ep, ep_fid.fid);
	if (bfid->fclass != FID_CLASS_EQ)
		return -EINVAL;

	eq = container_of(bfid, struct ibv_eq, fid.fid);
	if (flags & FI_RECV) {
		if (ep->req)
			return -EINVAL;
		ep->req = container_of(eq, struct ibv_eq_comp, eq);
	}
	if (flags & FI_SEND) {
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
ibv_msg_ep_recvv(struct fid_ep *ep, const struct iovec *iov, void **desc,
                 size_t count, void *context)
{
	struct ibv_msg_ep *_ep;
	struct ibv_recv_wr wr, *bad;
	struct ibv_sge *sge;
	size_t i;

	sge = alloca(count * sizeof(struct ibv_sge));
	wr.wr_id = (uintptr_t) context;
	wr.next = NULL;
	wr.sg_list = sge;
	wr.num_sge = (int) count;

	for (i = 0; i < count; i++) {
		sge[i].addr = (uintptr_t) iov[i].iov_base;
		sge[i].length = (uint32_t) iov[i].iov_len;
		sge[i].lkey = (uint32_t) (uintptr_t) desc[i];
	}

	_ep = container_of(ep, struct ibv_msg_ep, ep_fid);
	return -ibv_post_recv(_ep->id->qp, &wr, &bad);
}

static ssize_t
ibv_msg_ep_recvfrom(struct fid_ep *ep, void *buf, size_t len, void *desc,
		    const void *src_addr, void *context)
{
	return -FI_ENOSYS;
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
ibv_msg_ep_senddata(struct fid_ep *ep, const void *buf, size_t len,
		    void *desc, uint64_t data, void *context)
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
	wr.opcode = IBV_WR_SEND_WITH_IMM;
	wr.send_flags = (len <= _ep->inline_size) ? IBV_SEND_INLINE : 0;
	wr.imm_data = (uint32_t) data;

	return -ibv_post_send(_ep->id->qp, &wr, &bad);
}

static ssize_t
ibv_msg_ep_sendv(struct fid_ep *ep, const struct iovec *iov, void **desc,
                 size_t count, void *context)
{
	struct ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge *sge;
	size_t bytes = 0, i;

	sge = alloca(count * sizeof(struct ibv_sge));
	wr.wr_id = (uintptr_t) context;
	wr.next = NULL;
	wr.sg_list = sge;
	wr.num_sge = (int) count;
	wr.opcode = IBV_WR_SEND;

	_ep = container_of(ep, struct ibv_msg_ep, ep_fid);
	for (i = 0; i < count; i++) {
		sge[i].addr = (uintptr_t) iov[i].iov_base;
		sge[i].length = (uint32_t) iov[i].iov_len;
		bytes += iov[i].iov_len;
		sge[i].lkey = (uint32_t) (uintptr_t) desc[i];
	}
	wr.send_flags = (bytes <= _ep->inline_size) ? IBV_SEND_INLINE : 0;

	return -ibv_post_send(_ep->id->qp, &wr, &bad);
}

static ssize_t
ibv_msg_ep_sendto(struct fid_ep *ep, const void *buf, size_t len, void *desc,
		  const void *dest_addr, void *context)
{
	return -FI_ENOSYS;
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
			sge[i].lkey = (uint32_t) (uintptr_t) (msg->desc[i]);
			len += sge[i].length;
		}

		wr.sg_list = sge;
		wr.send_flags = (len <= _ep->inline_size) ? IBV_SEND_INLINE : 0;
	} else {
		wr.send_flags = 0;
	}

	wr.wr_id = (uintptr_t) msg->context;
	wr.next = NULL;
	if (flags & FI_REMOTE_EQ_DATA) {
		wr.opcode = IBV_WR_SEND_WITH_IMM;
		wr.imm_data = (uint32_t) msg->data;
	} else {
		wr.opcode = IBV_WR_SEND;
	}

	return -ibv_post_send(_ep->id->qp, &wr, &bad);
}

static ssize_t
ibv_msg_ep_inject(struct fid_ep *ep, const void *buf, size_t len)
{
	return -FI_ENOSYS;
}

static ssize_t
ibv_msg_ep_injectto(struct fid_ep *ep, const void *buf, size_t len,
		    const void *dest_addr)
{
	return -FI_ENOSYS;
}

static ssize_t
ibv_msg_ep_senddatato(struct fid_ep *ep, const void *buf, size_t len,
		      void *desc, uint64_t data, const void *dest_addr,
		      void *context)
{
	return -FI_ENOSYS;
}

static ssize_t
ibv_msg_ep_recvmsg(struct fid_ep *ep, const struct fi_msg *msg, uint64_t flags)
{
	struct ibv_msg_ep *_ep;
	struct ibv_recv_wr wr, *bad;
	struct ibv_sge *sge=NULL;
	size_t i;

	_ep = container_of(ep, struct ibv_msg_ep, ep_fid);
	wr.wr_id = (uintptr_t) msg->context;
	wr.next = NULL;
	if (msg->iov_count) {
		sge = alloca(sizeof(*sge) * msg->iov_count);
		for (i = 0; i < msg->iov_count; i++) {
			sge[i].addr = (uintptr_t) msg->msg_iov[i].iov_base;
			sge[i].length = (uint32_t) msg->msg_iov[i].iov_len;
			sge[i].lkey = (uint32_t) (uintptr_t) (msg->desc[i]);
		}

	}
	wr.sg_list = sge;
	wr.num_sge = msg->iov_count;

	return -ibv_post_recv(_ep->id->qp, &wr, &bad);
}

static struct fi_ops_msg ibv_msg_ep_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = ibv_msg_ep_recv,
	.recvv = ibv_msg_ep_recvv,
	.recvfrom = ibv_msg_ep_recvfrom,
	.recvmsg = ibv_msg_ep_recvmsg,
	.send = ibv_msg_ep_send,
	.sendv = ibv_msg_ep_sendv,
	.sendto = ibv_msg_ep_sendto,
	.sendmsg = ibv_msg_ep_sendmsg,
	.inject = ibv_msg_ep_inject,
	.injectto = ibv_msg_ep_injectto,
	.senddata = ibv_msg_ep_senddata,
	.senddatato = ibv_msg_ep_senddatato
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
ibv_msg_ep_rma_writev(struct fid_ep *ep, const struct iovec *iov, void **desc,
		      size_t count, uint64_t addr, uint64_t tag, void *context)
{
	struct ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge *sge;
	size_t bytes = 0, i;

	_ep = container_of(ep, struct ibv_msg_ep, ep_fid);
	sge = alloca(count * sizeof(struct ibv_sge));

	wr.wr_id = (uintptr_t) context;
	wr.next = NULL;
	wr.sg_list = sge;
	wr.num_sge = count;
	wr.opcode = IBV_WR_RDMA_WRITE;
	wr.wr.rdma.remote_addr = addr;
	wr.wr.rdma.rkey = (uint32_t) tag;

	for (i = 0; i < count; i++) {
		sge[i].addr = (uintptr_t) iov[i].iov_base;
		sge[i].length = (uint32_t) iov[i].iov_len;
		bytes += iov[i].iov_len;
		sge[i].lkey = (uint32_t) (uintptr_t) desc[i];
	}
	wr.send_flags = (bytes <= _ep->inline_size) ? IBV_SEND_INLINE : 0;

	return -ibv_post_send(_ep->id->qp, &wr, &bad);
}

static ssize_t
ibv_msg_ep_rma_writeto(struct fid_ep *ep, const void *buf, size_t len,
			void *desc, const void *dest_addr, uint64_t addr,
			uint64_t key, void *context)
{
	return -FI_ENOSYS;
}

static ssize_t
ibv_msg_ep_rma_writemsg(struct fid_ep *ep, const struct fi_msg_rma *msg,
			uint64_t flags)
{
	struct ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge *sge=NULL;
	size_t i, len;

	_ep = container_of(ep, struct ibv_msg_ep, ep_fid);
	wr.wr_id = (uintptr_t) msg->context;
	wr.next = NULL;
	wr.num_sge = msg->iov_count;
	wr.send_flags = 0;
	if (msg->iov_count) {
		sge = alloca(sizeof(*sge) * msg->iov_count);
		for (len = 0, i = 0; i < msg->iov_count; i++) {
			sge[i].addr = (uintptr_t) msg->msg_iov[i].iov_base;
			sge[i].length = (uint32_t) msg->msg_iov[i].iov_len;
			sge[i].lkey = (uint32_t) (uintptr_t) (msg->desc[i]);
			len += sge[i].length;
		}

		wr.send_flags = (len <= _ep->inline_size) ? IBV_SEND_INLINE : 0;
	}
	wr.sg_list = sge;

	wr.opcode = IBV_WR_RDMA_WRITE;
	if (flags & FI_REMOTE_EQ_DATA) {
		wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
		wr.imm_data = (uint32_t) msg->data;
	}

	wr.wr.rdma.remote_addr = msg->rma_iov->addr;
	wr.wr.rdma.rkey = (uint32_t) msg->rma_iov->key;

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

static ssize_t
ibv_msg_ep_rma_readv(struct fid_ep *ep, const struct iovec *iov, void **desc,
		     size_t count, uint64_t addr, uint64_t tag, void *context)
{
	struct ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge *sge;
	size_t i;

	sge = alloca(count * sizeof(struct ibv_sge));
	wr.wr_id = (uintptr_t) context;
	wr.next = NULL;
	wr.sg_list = sge;
	wr.num_sge = count;
	wr.opcode = IBV_WR_RDMA_READ;
	wr.send_flags = 0;
	wr.wr.rdma.remote_addr = addr;
	wr.wr.rdma.rkey = (uint32_t) tag;

	for (i = 0; i < count; i++) {
		sge[i].addr = (uintptr_t) iov[i].iov_base;
		sge[i].length = (uint32_t) iov[i].iov_len;
		sge[i].lkey = (uint32_t) (uintptr_t) desc[i];
	}

	_ep = container_of(ep, struct ibv_msg_ep, ep_fid);
	return -ibv_post_send(_ep->id->qp, &wr, &bad);
}

static ssize_t
ibv_msg_ep_rma_readfrom(struct fid_ep *ep, void *buf, size_t len, void *desc,
			const void *src_addr, uint64_t addr, uint64_t key,
			void *context)
{
	return -FI_ENOSYS;
}

static ssize_t
ibv_msg_ep_rma_readmsg(struct fid_ep *ep, const struct fi_msg_rma *msg,
			uint64_t flags)
{
	struct ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge *sge;
	size_t i;

	wr.wr_id = (uintptr_t) msg->context;
	wr.next = NULL;
	wr.sg_list = NULL;
	if (msg->iov_count) {
		sge = alloca(sizeof(*sge) * msg->iov_count);
		for (i = 0; i < msg->iov_count; i++) {
			sge[i].addr = (uintptr_t) msg->msg_iov[i].iov_base;
			sge[i].length = (uint32_t) msg->msg_iov[i].iov_len;
			sge[i].lkey = (uint32_t) (uintptr_t) (msg->desc[i]);
		}
		wr.sg_list = sge;
	}
	wr.num_sge = msg->iov_count;
	wr.opcode = IBV_WR_RDMA_READ;
	wr.send_flags = 0;

	wr.wr.rdma.remote_addr = msg->rma_iov->addr;
	wr.wr.rdma.rkey = (uint32_t) msg->rma_iov->key;

	_ep = container_of(ep, struct ibv_msg_ep, ep_fid);
	return -ibv_post_send(_ep->id->qp, &wr, &bad);
}

static ssize_t
ibv_msg_ep_rma_inject(struct fid_ep *ep, const void *buf, size_t len,
			uint64_t addr, uint64_t key)
{
	return -FI_ENOSYS;
}

static ssize_t
ibv_msg_ep_rma_injectto(struct fid_ep *ep, const void *buf, size_t len,
                        const void *dest_addr, uint64_t addr, uint64_t key)
{
	return -FI_ENOSYS;
}

static ssize_t
ibv_msg_ep_rma_writedata(struct fid_ep *ep, const void *buf, size_t len,
			void *desc, uint64_t data, uint64_t addr, uint64_t key,
			void *context)
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
	wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
	wr.send_flags = (len <= _ep->inline_size) ? IBV_SEND_INLINE : 0;
	wr.imm_data = (uint32_t) data;

	wr.wr.rdma.remote_addr = addr;
	wr.wr.rdma.rkey = (uint32_t) key;

	return -ibv_post_send(_ep->id->qp, &wr, &bad);
}

static ssize_t
ibv_msg_ep_rma_writedatato(struct fid_ep *ep, const void *buf, size_t len,
			void *desc, uint64_t data, const void *dest_addr,
			uint64_t addr, uint64_t key, void *context)
{
	return -FI_ENOSYS;
}


static struct fi_ops_rma ibv_msg_ep_rma_ops = {
	.size = sizeof(struct fi_ops_rma),
	.read = ibv_msg_ep_rma_read,
	.readv = ibv_msg_ep_rma_readv,
	.readfrom = ibv_msg_ep_rma_readfrom,
	.readmsg = ibv_msg_ep_rma_readmsg,
	.write = ibv_msg_ep_rma_write,
	.writev = ibv_msg_ep_rma_writev,
	.writeto = ibv_msg_ep_rma_writeto,
	.writemsg = ibv_msg_ep_rma_writemsg,
	.inject = ibv_msg_ep_rma_inject,
	.injectto = ibv_msg_ep_rma_injectto,
	.writedata = ibv_msg_ep_rma_writedata,
	.writedatato = ibv_msg_ep_rma_writedatato
};

static ssize_t
ibv_msg_ep_atomic_write(struct fid_ep *ep, const void *buf, size_t count,
			void *desc, uint64_t addr, uint64_t key,
			enum fi_datatype datatype, enum fi_op op, void *context)
{
	struct ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge sge;

	if (count != 1)
		return -FI_E2BIG;

	switch (datatype) {
	case FI_INT64:
	case FI_UINT64:
#if __BITS_PER_LONG == 64
	case FI_DOUBLE:
	case FI_FLOAT:
#endif
		break;
	default:
		return -FI_EINVAL;
	}

	switch (op) {
	case FI_ATOMIC_WRITE:
		wr.opcode = IBV_WR_RDMA_WRITE;
		wr.wr.rdma.remote_addr = addr;
		wr.wr.rdma.rkey = (uint32_t) (uintptr_t) key;
		break;
	default:
		return -ENOSYS;
	}
	_ep = container_of(ep, struct ibv_msg_ep, ep_fid);

	sge.addr = (uintptr_t) buf;
	sge.length = (uint32_t) sizeof(uint64_t);
	sge.lkey = (uint32_t) (uintptr_t) desc;

	wr.wr_id = (uintptr_t) context;
	wr.next = NULL;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.send_flags = (sge.length <= _ep->inline_size) ? IBV_SEND_INLINE : 0;
	wr.send_flags |= IBV_SEND_FENCE; 

	return -ibv_post_send(_ep->id->qp, &wr, &bad);
}

static ssize_t
ibv_msg_ep_atomic_writev(struct fid_ep *ep,
                        const struct fi_ioc *iov, void **desc, size_t count,
                        uint64_t addr, uint64_t key,
                        enum fi_datatype datatype, enum fi_op op, void *context)
{
	if (iov->count != 1)
		return -FI_E2BIG;

	return ibv_msg_ep_atomic_write(ep, iov->addr, count, desc[0], addr,
					key, datatype, op, context);
}

static ssize_t
ibv_msg_ep_atomic_writeto(struct fid_ep *ep,
                        const void *buf, size_t count, void *desc,
                        const void *dest_addr,
                        uint64_t addr, uint64_t key,
                        enum fi_datatype datatype, enum fi_op op, void *context)
{
	return -FI_ENOSYS;
}

static ssize_t
ibv_msg_ep_atomic_writemsg(struct fid_ep *ep,
                        const struct fi_msg_atomic *msg, uint64_t flags)
{
	struct ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge sge;

	if (msg->iov_count != 1 || msg->msg_iov->count != 1)
		return -FI_E2BIG;

	switch (msg->datatype) {
	case FI_INT64:
	case FI_UINT64:
#if __BITS_PER_LONG == 64
	case FI_DOUBLE:
	case FI_FLOAT:
#endif
		break;
	default:
		return -FI_EINVAL;
	}

	switch (msg->op) {
	case FI_ATOMIC_WRITE:
		if (flags & FI_REMOTE_EQ_DATA) {
			wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
			wr.imm_data = (uint32_t) msg->data;
		} else {
			wr.opcode = IBV_WR_RDMA_WRITE;
		}
		wr.wr.rdma.remote_addr = msg->rma_iov->addr;
		wr.wr.rdma.rkey = (uint32_t) (uintptr_t) msg->rma_iov->key;
		break;
	default:
		return -ENOSYS;
	}
	_ep = container_of(ep, struct ibv_msg_ep, ep_fid);

	sge.addr = (uintptr_t) msg->msg_iov->addr;
	sge.length = (uint32_t) sizeof(uint64_t);
	sge.lkey = (uint32_t) (uintptr_t) msg->desc[0];

	wr.wr_id = (uintptr_t) msg->context;
	wr.next = NULL;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.send_flags = (sge.length <= _ep->inline_size) ? IBV_SEND_INLINE : 0;
	wr.send_flags |= IBV_SEND_FENCE; 

	return -ibv_post_send(_ep->id->qp, &wr, &bad);
}

static ssize_t
ibv_msg_ep_atomic_readwrite(struct fid_ep *ep, const void *buf, size_t count,
			void *desc, void *result, void *result_desc,
			uint64_t addr, uint64_t key, enum fi_datatype datatype,
			enum fi_op op, void *context)
{
	struct ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge sge;

	if (count != 1)
		return -FI_E2BIG;

	switch (datatype) {
	case FI_INT64:
	case FI_UINT64:
#if __BITS_PER_LONG == 64
	case FI_DOUBLE:
	case FI_FLOAT:
#endif
		break;
	default:
		return -FI_EINVAL;
	}

	switch (op) {
	case FI_ATOMIC_READ:
		wr.opcode = IBV_WR_RDMA_READ;
		wr.wr.rdma.remote_addr = addr;
		wr.wr.rdma.rkey = (uint32_t) (uintptr_t) key;
		break;
	case FI_SUM:
		wr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
		wr.wr.atomic.remote_addr = addr;
		wr.wr.atomic.compare_add = (uintptr_t) buf;
		wr.wr.atomic.swap = 0;
		wr.wr.atomic.rkey = (uint32_t) (uintptr_t) key;
		break;
	default:
		return -ENOSYS;
	}

	sge.addr = (uintptr_t) result;
	sge.length = (uint32_t) sizeof(uint64_t);
	sge.lkey = (uint32_t) (uintptr_t) result_desc;

	_ep = container_of(ep, struct ibv_msg_ep, ep_fid);

	wr.wr_id = (uintptr_t) context;
	wr.next = NULL;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.send_flags = IBV_SEND_FENCE; 

	return -ibv_post_send(_ep->id->qp, &wr, &bad);
}

static ssize_t
ibv_msg_ep_atomic_readwritev(struct fid_ep *ep, const struct fi_ioc *iov,
			void **desc, size_t count,
			struct fi_ioc *resultv, void **result_desc,
			size_t result_count, uint64_t addr,
			uint64_t key, enum fi_datatype datatype,
			enum fi_op op, void *context)
{
	if (iov->count != 1)
		return -FI_E2BIG;

        return ibv_msg_ep_atomic_readwrite(ep, iov->addr, count,
			desc[0], resultv->addr, result_desc[0],
			addr, key, datatype, op, context);
}

static ssize_t
ibv_msg_ep_atomic_readwriteto(struct fid_ep *ep,
                        const void *buf, size_t count, void *desc,
                        void *result, void *result_desc,
                        const void *dest_addr,
                        uint64_t addr, uint64_t key,
                        enum fi_datatype datatype, enum fi_op op, void *context)
{
	return -FI_ENOSYS;
}

static ssize_t
ibv_msg_ep_atomic_readwritemsg(struct fid_ep *ep,
				const struct fi_msg_atomic *msg,
				struct fi_ioc *resultv, void **result_desc,
				size_t result_count, uint64_t flags)
{
	struct ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge sge;

	if (msg->iov_count != 1 || msg->msg_iov->count != 1)
		return -FI_E2BIG;

	switch (msg->datatype) {
	case FI_INT64:
	case FI_UINT64:
#if __BITS_PER_LONG == 64
	case FI_DOUBLE:
	case FI_FLOAT:
#endif
		break;
	default:
		return -FI_EINVAL;
	}

	switch (msg->op) {
	case FI_ATOMIC_READ:
		wr.opcode = IBV_WR_RDMA_READ;
		wr.wr.rdma.remote_addr = msg->rma_iov->addr;
		wr.wr.rdma.rkey = (uint32_t) (uintptr_t) msg->rma_iov->key;
		break;
	case FI_SUM:
		wr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
		wr.wr.atomic.remote_addr = msg->rma_iov->addr;
		wr.wr.atomic.compare_add = (uintptr_t) msg->addr;
		wr.wr.atomic.swap = 0;
		wr.wr.atomic.rkey = (uint32_t) (uintptr_t) msg->rma_iov->key;
		break;
	default:
		return -ENOSYS;
	}

	sge.addr = (uintptr_t) resultv->addr;
	sge.length = (uint32_t) sizeof(uint64_t);
	sge.lkey = (uint32_t) (uintptr_t) result_desc[0];

	_ep = container_of(ep, struct ibv_msg_ep, ep_fid);

	wr.wr_id = (uintptr_t) msg->context;
	wr.next = NULL;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.send_flags = IBV_SEND_FENCE; 
	if (flags & FI_REMOTE_EQ_DATA)
		wr.imm_data = (uint32_t) msg->data;

	return -ibv_post_send(_ep->id->qp, &wr, &bad);
}

static ssize_t
ibv_msg_ep_atomic_compwrite(struct fid_ep *ep, const void *buf, size_t count,
			void *desc, const void *compare,
			void *compare_desc, void *result,
			void *result_desc, uint64_t addr, uint64_t key,
			enum fi_datatype datatype,
			enum fi_op op, void *context)
{
	struct ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge sge;

	if (op != FI_CSWAP)
		return -ENOSYS;

	if (count != 1)
		return -FI_E2BIG;

	switch (datatype) {
	case FI_INT64:
	case FI_UINT64:
#if __BITS_PER_LONG == 64
	case FI_DOUBLE:
	case FI_FLOAT:
#endif
		break;
	default:
		return -FI_EINVAL;
	}

	wr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
	wr.wr.atomic.remote_addr = addr;
	wr.wr.atomic.compare_add = (uintptr_t) compare;
	wr.wr.atomic.swap = (uintptr_t) buf;
	wr.wr.atomic.rkey = (uint32_t) (uintptr_t) key;

	sge.addr = (uintptr_t) result;
	sge.length = (uint32_t) sizeof(uint64_t);
	sge.lkey = (uint32_t) (uintptr_t) result_desc;

	_ep = container_of(ep, struct ibv_msg_ep, ep_fid);

	wr.wr_id = (uintptr_t) context;
	wr.next = NULL;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.send_flags = IBV_SEND_FENCE; 

	return -ibv_post_send(_ep->id->qp, &wr, &bad);
}

static ssize_t
ibv_msg_ep_atomic_compwritev(struct fid_ep *ep, const struct fi_ioc *iov,
				void **desc, size_t count,
				const struct fi_ioc *comparev,
				void **compare_desc, size_t compare_count,
				struct fi_ioc *resultv, void **result_desc,
				size_t result_count, uint64_t addr,
				uint64_t key, enum fi_datatype datatype,
				enum fi_op op, void *context)
{
	if (iov->count != 1)
		return -FI_E2BIG;

	return ibv_msg_ep_atomic_compwrite(ep, iov->addr, count, desc[0],
				comparev->addr, compare_desc[0], resultv->addr,
				result_desc[0], addr, key,
                        	datatype, op, context);
}

static ssize_t
ibv_msg_ep_atomic_compwriteto(struct fid_ep *ep, const void *buf, size_t count,
				void *desc, const void *compare,
				void *compare_desc, void *result,
				void *result_desc, const void *dest_addr,
                        	uint64_t addr, uint64_t key,
                        	enum fi_datatype datatype,
				enum fi_op op, void *context)
{
	return -FI_ENOSYS;
}

static ssize_t
ibv_msg_ep_atomic_compwritemsg(struct fid_ep *ep,
				const struct fi_msg_atomic *msg,
				const struct fi_ioc *comparev,
				void **compare_desc, size_t compare_count,
				struct fi_ioc *resultv,
				void **result_desc, size_t result_count,
				uint64_t flags)
{
	struct ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge sge;

	if (msg->op != FI_CSWAP)
		return -ENOSYS;

	if (msg->iov_count != 1 || msg->msg_iov->count != 1)
		return -FI_E2BIG;

	switch(msg->datatype) {
	case FI_INT64:
	case FI_UINT64:
#if __BITS_PER_LONG == 64
	case FI_DOUBLE:
	case FI_FLOAT:
#endif
		break;
	default:
		return -FI_EINVAL;
	}

	wr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
	wr.wr.atomic.remote_addr = msg->rma_iov->addr;
	wr.wr.atomic.compare_add = (uintptr_t) comparev->addr;
	wr.wr.atomic.swap = (uintptr_t) msg->addr;
	wr.wr.atomic.rkey = (uint32_t) (uintptr_t) msg->rma_iov->key;

	sge.addr = (uintptr_t) resultv->addr;
	sge.length = (uint32_t) sizeof(uint64_t);
	sge.lkey = (uint32_t) (uintptr_t) result_desc[0];

	_ep = container_of(ep, struct ibv_msg_ep, ep_fid);

	wr.wr_id = (uintptr_t) msg->context;
	wr.next = NULL;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.send_flags = IBV_SEND_FENCE; 
	if (flags & FI_REMOTE_EQ_DATA)
		wr.imm_data = (uint32_t) msg->data;

	return -ibv_post_send(_ep->id->qp, &wr, &bad);
}

static int
ibv_msg_ep_atomic_writevalid(struct fid_ep *ep, enum fi_datatype datatype,
				enum fi_op op, size_t *count)
{
	switch (op) {
	case FI_ATOMIC_WRITE:
		break;
	default:
		return -FI_ENOSYS;
	}

	switch (datatype) {
	case FI_INT64:
	case FI_UINT64:
#if __BITS_PER_LONG == 64
	case FI_DOUBLE:
	case FI_FLOAT:
#endif
		break;
	default:
		return -FI_EINVAL;
	}

	if (count)
		*count = 1;
	return 0;
}

static int
ibv_msg_ep_atomic_readwritevalid(struct fid_ep *ep, enum fi_datatype datatype,
				enum fi_op op, size_t *count)
{
	switch (op) {
	case FI_ATOMIC_READ:
	case FI_SUM:
		break;
	default:
		return -FI_ENOSYS;
	}

	switch (datatype) {
	case FI_INT64:
	case FI_UINT64:
#if __BITS_PER_LONG == 64
	case FI_DOUBLE:
	case FI_FLOAT:
#endif
		break;
	default:
		return -FI_EINVAL;
	}

	if (count)
		*count = 1;
	return 0;
}

static int
ibv_msg_ep_atomic_compwritevalid(struct fid_ep *ep, enum fi_datatype datatype,
				enum fi_op op, size_t *count)
{
	if (op != FI_CSWAP)
		return -FI_ENOSYS;

	switch (datatype) {
	case FI_INT64:
	case FI_UINT64:
#if __BITS_PER_LONG == 64
	case FI_DOUBLE:
	case FI_FLOAT:
#endif
		break;
	default:
		return -FI_EINVAL;
	}

	if (count)
		*count = 1;
	return 0;
}


static struct fi_ops_atomic ibv_msg_ep_atomic_ops = {
	.size		= sizeof(struct fi_ops_atomic),
	.write		= ibv_msg_ep_atomic_write,
	.writev		= ibv_msg_ep_atomic_writev,
	.writeto	= ibv_msg_ep_atomic_writeto,
	.writemsg	= ibv_msg_ep_atomic_writemsg,
	.readwrite	= ibv_msg_ep_atomic_readwrite,
	.readwritev	= ibv_msg_ep_atomic_readwritev,
	.readwriteto	= ibv_msg_ep_atomic_readwriteto,
	.readwritemsg	= ibv_msg_ep_atomic_readwritemsg,
	.compwrite	= ibv_msg_ep_atomic_compwrite,
	.compwritev	= ibv_msg_ep_atomic_compwritev,
	.compwriteto	= ibv_msg_ep_atomic_compwriteto,
	.compwritemsg	= ibv_msg_ep_atomic_compwritemsg,
	.writevalid	= ibv_msg_ep_atomic_writevalid,
	.readwritevalid	= ibv_msg_ep_atomic_readwritevalid,
	.compwritevalid = ibv_msg_ep_atomic_compwritevalid
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
		case FI_OPT_MAX_INJECTED_SEND:
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
		case FI_OPT_MAX_INJECTED_SEND:
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

static ssize_t ibv_msg_ep_cancel(fid_t fid, void *context)
{
	return 0;
}

static struct fi_ops_ep ibv_msg_ep_base_ops = {
	.size = sizeof(struct fi_ops_ep),
	.enable = ibv_msg_ep_enable,
	.cancel = ibv_msg_ep_cancel,
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
	_ep->ep_fid.fid.context = context;
	_ep->ep_fid.fid.ops = &ibv_msg_ep_ops;
	_ep->ep_fid.ops = &ibv_msg_ep_base_ops;
	_ep->ep_fid.msg = &ibv_msg_ep_msg_ops;
	_ep->ep_fid.cm = &ibv_msg_ep_cm_ops;
	_ep->ep_fid.rma = &ibv_msg_ep_rma_ops;
	_ep->ep_fid.atomic = &ibv_msg_ep_atomic_ops;

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

	fi->type = FID_MSG;
	fi->ep_cap  = FI_MSG | FI_RMA;
	if (event->id->verbs->device->transport_type == IBV_TRANSPORT_IWARP) {
		fi->protocol = FI_PROTO_IWARP;
	} else {
		fi->protocol = FI_PROTO_IB_RC;
	}
//	fi->sa_family = rdma_get_local_addr(event->id)->sa_family;

	fi->src_addrlen = ibv_sockaddr_len(rdma_get_local_addr(event->id));
	if (!(fi->src_addr = malloc(fi->src_addrlen)))
		goto err;
	memcpy(fi->src_addr, rdma_get_local_addr(event->id), fi->src_addrlen);

	fi->dest_addrlen = ibv_sockaddr_len(rdma_get_peer_addr(event->id));
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

			if (_eq->channel == NULL)
				return 0;

			fi_poll_fd(_eq->channel->fd);
		} else {
			ret = -errno;
			break;
		}
	}

	return (left < len) ? len - left : ret;
}

static ssize_t
ibv_eq_cm_condread_data(struct fid_eq *eq, void *buf, size_t len, const void *cond)
{
	ssize_t rc, cur, left;
	ssize_t  threshold;
	struct ibv_eq_cm *_eq;
	struct fi_eq_cm_entry *entry = (struct fi_eq_cm_entry *) buf;

	_eq = container_of(eq, struct ibv_eq_cm, eq.fid);
	threshold = cond ? (long) cond : sizeof(*entry);

	for (cur = 0, left = len; cur < threshold && left > 0; ) {
		rc = ibv_eq_cm_read_data(eq, (void*)entry, left);
		if (rc < 0)
			return rc;
		if (rc > 0) {
			left -= rc;
			entry += (rc / sizeof(*entry));
			cur += rc;
		}

		if (cur >= threshold || left <= 0)
			break;

		fi_poll_fd(_eq->channel->fd);
	}
	
	return cur;
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
	.condread = ibv_eq_cm_condread_data,
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
	case FI_WAIT_FD:
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
	case FI_WAIT_NONE:
		break;
	default:
		return -ENOSYS;
	}

	_eq->flags = attr->flags;
	_eq->eq.fid.fid.fclass = FID_CLASS_EQ;
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

        return -ibv_req_notify_cq(_eq->cq, (_eq->flags & FI_REMOTE_SIGNAL) ? 1:0);
}

static ssize_t
ibv_eq_comp_condread(struct fid_eq *eq, void *buf, size_t len, const void *cond,
		     size_t entry_len)
{
	ssize_t rc, cur, left;
	ssize_t  threshold;
	int reset=1;
	struct ibv_eq_comp *_eq;
	char *entry = (char *) buf;

	_eq = container_of(eq, struct ibv_eq_comp, eq.fid);

	threshold = cond ? (ssize_t) cond : entry_len;

	for (cur = 0, left = len; cur < threshold && left > 0; ) {
		rc = _eq->eq.fid.ops->read(eq, (void*)entry, left);
		if (rc < 0)
			return rc;

		if (rc > 0) {
			left -= rc;
			entry += rc;
			cur += rc;
		}

		if (cur >= threshold || left <= 0)
			break;

		if (_eq->channel == NULL)
			return cur;	// .wait_obj = FI_WAIT_NONE

		if (reset) {
			ibv_eq_comp_reset(eq, NULL);
			reset = 0;
			continue;
		}
		fi_poll_fd(_eq->channel->fd);
	}
	return cur;
}

static ssize_t ibv_eq_comp_read_context(struct fid_eq *eq, void *buf, size_t len)
{
	struct ibv_eq_comp *_eq;
	struct fi_eq_entry *entry;
	size_t left;
	int ret = -FI_EINVAL;
	struct ibv_wc *wc, *wc_base;
	int num_wc = len / sizeof(*entry);

	_eq = container_of(eq, struct ibv_eq_comp, eq.fid);
	entry = (struct fi_eq_entry *) buf;
	if (_eq->wc.status)
		return -FI_EAVAIL;

	wc_base = alloca(num_wc * sizeof(struct ibv_wc));

	for (left = len; left >= sizeof(*entry); ) {
		ret = ibv_poll_cq(_eq->cq, num_wc, wc_base);
		if (ret <= 0)
			break;

		for (wc = wc_base; wc < wc_base + ret; wc++) {
			if (wc->status) {
				_eq->wc = *wc;
				return (left < len) ? len - left : -FI_EAVAIL;
			}
			entry->op_context = (void *) (uintptr_t) wc->wr_id;
			left -= sizeof(*entry);
			entry = entry + 1;
		}
	}

	return (left < len) ? len - left : ret;
}

static ssize_t
ibv_eq_comp_condread_context(struct fid_eq *eq, void *buf, size_t len, const void *cond)
{
	return ibv_eq_comp_condread(eq, buf, len, cond, sizeof( struct fi_eq_entry));
}

static ssize_t ibv_eq_comp_read_comp(struct fid_eq *eq, void *buf, size_t len)
{
	struct ibv_eq_comp *_eq;
	struct fi_eq_comp_entry *entry;
	size_t left;
	int ret = -FI_EINVAL;
	struct ibv_wc *wc, *wc_base;
	int num_wc = len / sizeof(*entry);

	_eq = container_of(eq, struct ibv_eq_comp, eq.fid);
	entry = (struct fi_eq_comp_entry *) buf;
	if (_eq->wc.status)
		return -FI_EAVAIL;

	wc_base = alloca(num_wc * sizeof(struct ibv_wc));

	for (left = len; left >= sizeof(*entry); ) {
		ret = ibv_poll_cq(_eq->cq, num_wc, wc_base);
		if (ret <= 0)
			break;

		for (wc = wc_base; wc < wc_base + ret; wc++) {
			if (wc->status) {
				_eq->wc = *wc;
				return (left < len) ? len - left : -FI_EAVAIL;
			}
			entry->op_context = (void *) (uintptr_t) wc->wr_id;
			entry->flags = (uint64_t) wc->wc_flags;
			entry->len = (uint64_t) wc->byte_len;
			left -= sizeof(*entry);
			entry = entry + 1;
		}
	}

	return (left < len) ? len - left : ret;
}

static ssize_t
ibv_eq_comp_condread_comp(struct fid_eq *eq, void *buf, size_t len, const void *cond)
{
	return ibv_eq_comp_condread(eq, buf, len, cond, sizeof(struct fi_eq_comp_entry));
}

static ssize_t ibv_eq_comp_read_data(struct fid_eq *eq, void *buf, size_t len)
{
	struct ibv_eq_comp *_eq;
	struct fi_eq_data_entry *entry;
	size_t left;
	int ret = -FI_EINVAL;
	struct ibv_wc *wc, *wc_base;
	int num_wc = len / sizeof(*entry);

	_eq = container_of(eq, struct ibv_eq_comp, eq.fid);
	entry = (struct fi_eq_data_entry *) buf;
	if (_eq->wc.status)
		return -FI_EAVAIL;

	wc_base = alloca(num_wc * sizeof(struct ibv_wc));

	for (left = len; left >= sizeof(*entry); ) {
		ret = ibv_poll_cq(_eq->cq, num_wc, wc_base);
		if (ret <= 0)
			break;

		for (wc = wc_base; wc < wc_base + ret; wc++) {
			if (wc->status) {
				_eq->wc = *wc;
				return (left < len) ? len - left : -FI_EAVAIL;
			}

			entry->op_context = (void *) (uintptr_t) wc->wr_id;
			entry->flags = 0;
			entry->data = 0;
			if (wc->wc_flags & IBV_WC_WITH_IMM) {
				entry->flags = FI_REMOTE_EQ_DATA;
				entry->data = wc->imm_data;
			}
			entry->len = 0;
			if (wc->opcode & (IBV_WC_RECV | IBV_WC_RECV_RDMA_WITH_IMM))
				entry->len = wc->byte_len;
			left -= sizeof(*entry);
			entry = entry + 1;
		}
	}

	return (left < len) ? len - left : ret;
}

static ssize_t
ibv_eq_comp_condread_data(struct fid_eq *eq, void *buf, size_t len, const void *cond)
{
	return ibv_eq_comp_condread(eq, buf, len, cond, sizeof(struct fi_eq_data_entry));
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
	.condread = ibv_eq_comp_condread_context,
	.readerr = ibv_eq_comp_readerr,
	.strerror = ibv_eq_comp_strerror
};

static struct fi_ops_eq ibv_eq_comp_comp_ops = {
	.size = sizeof(struct fi_ops_eq),
	.read = ibv_eq_comp_read_comp,
	.condread = ibv_eq_comp_condread_comp,
	.readerr = ibv_eq_comp_readerr,
	.strerror = ibv_eq_comp_strerror
};

static struct fi_ops_eq ibv_eq_comp_data_ops = {
	.size = sizeof(struct fi_ops_eq),
	.read = ibv_eq_comp_read_data,
	.condread = ibv_eq_comp_condread_data,
	.readerr = ibv_eq_comp_readerr,
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

	_eq = calloc(1, sizeof *_eq);
	if (!_eq)
		return -FI_ENOMEM;

	_eq->eq.domain = container_of(domain, struct ibv_domain, domain_fid);

	switch (attr->wait_obj) {
	case FI_WAIT_FD:
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
	case FI_WAIT_NONE:
		break;
	default:
		return -FI_ENOSYS;
	}

	_eq->cq = ibv_create_cq(_eq->eq.domain->verbs, attr->size, _eq,
				_eq->channel, attr->signaling_vector);
	if (!_eq->cq) {
		ret = -errno;
		goto err2;
	}

	_eq->flags |= attr->flags;
	_eq->eq.fid.fid.fclass = FID_CLASS_EQ;
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
		ret = -FI_ENOSYS;
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
ibv_mr_reg(struct fid_domain *domain, const void *buf, size_t len,
	   uint64_t access, uint64_t offset, uint64_t requested_key,
	   uint64_t flags, struct fid_mr **mr, void *context)
{
	struct ibv_mem_desc *md;
	int ibv_access;

	if (flags)
		return -FI_EBADFLAGS;

	md = calloc(1, sizeof *md);
	if (!md)
		return -FI_ENOMEM;

	md->domain = container_of(domain, struct ibv_domain, domain_fid);
	md->mr_fid.fid.fclass = FID_CLASS_MR;
	md->mr_fid.fid.context = context;
	md->mr_fid.fid.ops = &ibv_mr_ops;

	ibv_access = IBV_ACCESS_LOCAL_WRITE;
	if (access & FI_REMOTE_READ)
		ibv_access |= IBV_ACCESS_REMOTE_READ;
	if (access & FI_REMOTE_WRITE)
		ibv_access |= IBV_ACCESS_REMOTE_WRITE;
	if ((access & FI_READ) || (access & FI_WRITE))
		ibv_access |= IBV_ACCESS_REMOTE_ATOMIC;

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

static ssize_t
ibv_feq_cm_readerr(struct fid_eq *eq, struct fi_eq_err_entry *entry,
		   size_t len, uint64_t flags)
{
	struct ibv_feq_cm *_eq;

	_eq = container_of(eq, struct ibv_feq_cm, eq_fid);
	if (!_eq->err.err)
		return 0;

	if (len < sizeof(*entry))
		return -EINVAL;

	*entry = _eq->err;
	_eq->err.err = 0;
	_eq->err.prov_errno = 0;
	return sizeof(*entry);
}

static struct fi_info *
ibv_feq_cm_getinfo(struct ibv_fabric *fab, struct rdma_cm_event *event)
{
	struct fi_info *fi;

	fi = calloc(1, sizeof *fi);
	if (!fi)
		return NULL;

	fi->type = FID_MSG;
	if (event->id->verbs->device->transport_type == IBV_TRANSPORT_IWARP) {
		fi->protocol = FI_PROTO_IWARP;
	} else {
		fi->protocol = FI_PROTO_IB_RC;
	}
	fi->ep_cap = FI_MSG | FI_RMA;

	fi->src_addrlen = ibv_sockaddr_len(rdma_get_local_addr(event->id));
	if (!(fi->src_addr = malloc(fi->src_addrlen)))
		goto err;
	memcpy(fi->src_addr, rdma_get_local_addr(event->id), fi->src_addrlen);

	fi->dest_addrlen = ibv_sockaddr_len(rdma_get_peer_addr(event->id));
	if (!(fi->dest_addr = malloc(fi->dest_addrlen)))
		goto err;
	memcpy(fi->dest_addr, rdma_get_peer_addr(event->id), fi->dest_addrlen);

	if (!(fi->fabric_name = strdup(fab->name)))
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

static ssize_t ibv_feq_cm_process_event(struct ibv_feq_cm *eq,
	struct rdma_cm_event *event, struct fi_eq_cm_entry *entry, size_t len)
{
	fid_t fid;
	size_t datalen;

	fid = event->id->context;
	switch (event->event) {
	case RDMA_CM_EVENT_CONNECT_REQUEST:
		rdma_migrate_id(event->id, NULL);
		entry->event = FI_CONNREQ;
		entry->info = ibv_feq_cm_getinfo(eq->fab, event);
		if (!entry->info) {
			rdma_destroy_id(event->id);
			return 0;
		}
		break;
	case RDMA_CM_EVENT_DEVICE_REMOVAL:
		eq->err.fid_context = fid->context;
		eq->err.err = ENODEV;
		return -FI_EAVAIL;
	case RDMA_CM_EVENT_ADDR_CHANGE:
		eq->err.fid_context = fid->context;
		eq->err.err = EADDRNOTAVAIL;
		return -FI_EAVAIL;
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

static ssize_t ibv_feq_cm_read_data(struct fid_eq *eq, void *buf, size_t len)
{
	struct ibv_feq_cm *_eq;
	struct fi_eq_cm_entry *entry;
	struct rdma_cm_event *event;
	size_t left;
	ssize_t ret = -FI_EINVAL;

	_eq = container_of(eq, struct ibv_feq_cm, eq_fid);
	entry = (struct fi_eq_cm_entry *) buf;
	if (_eq->err.err)
		return -FI_EAVAIL;

	for (left = len; left >= sizeof(*entry); ) {
		ret = rdma_get_cm_event(_eq->channel, &event);
		if (!ret) {
			ret = ibv_feq_cm_process_event(_eq, event, entry, left);
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

static ssize_t
ibv_feq_cm_condread_data(struct fid_eq *eq, void *buf, size_t len, const void *cond)
{
	ssize_t rc, cur, left;
	ssize_t  threshold;
	struct ibv_feq_cm *_eq;
	struct fi_eq_cm_entry *entry = (struct fi_eq_cm_entry *) buf;

	_eq = container_of(eq, struct ibv_feq_cm, eq_fid);
	threshold = cond ? (long) cond : sizeof(*entry);

	for(cur = 0, left = len; cur < threshold && left > 0; ) {
		rc = ibv_feq_cm_read_data(eq, (void*)entry, left);
		if (rc < 0)
			return rc;
		if (rc > 0) {
			left -= rc;
			entry += (rc / sizeof(*entry));
			cur += rc;
		}

		if (cur >= threshold || left <= 0)
			break;

		fi_poll_fd(_eq->channel->fd);
	}

	return cur;
}

struct fi_ops_eq ibv_feq_cm_data_ops = {
	.size = sizeof(struct fi_ops_eq),
	.read = ibv_feq_cm_read_data,
	.condread = ibv_feq_cm_condread_data,
	.readerr = ibv_feq_cm_readerr,
	.strerror = ibv_eq_cm_strerror
};

static int ibv_feq_cm_close(fid_t fid)
{
	struct ibv_feq_cm *eq;

	eq = container_of(fid, struct ibv_feq_cm, eq_fid.fid);
	if (eq->channel)
		rdma_destroy_event_channel(eq->channel);

	free(eq);
	return 0;
}

static int ibv_feq_cm_control(fid_t fid, int command, void *arg)
{
	struct ibv_feq_cm *eq;
	int ret = 0;

	eq = container_of(fid, struct ibv_feq_cm, eq_fid.fid);
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

struct fi_ops ibv_feq_cm_ops = {
	.size = sizeof(struct fi_ops),
	.close = ibv_feq_cm_close,
	.control = ibv_feq_cm_control,
};

static int
ibv_feq_open(struct fid_fabric *fabric, const struct fi_eq_attr *attr,
	     struct fid_eq **eq, void *context)
{
	struct ibv_feq_cm *_eq;
	long flags = 0;
	int ret;

	if (attr->format != FI_EQ_FORMAT_CM)
		return -FI_ENOSYS;

	_eq = calloc(1, sizeof *_eq);
	if (!_eq)
		return -FI_ENOMEM;

	_eq->fab = container_of(fabric, struct ibv_fabric, fabric_fid);

	switch (attr->wait_obj) {
	case FI_WAIT_FD:
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
	case FI_WAIT_NONE:
		break;
	default:
		return -FI_ENOSYS;
	}

	_eq->flags = attr->flags;
	_eq->eq_fid.fid.fclass = FID_CLASS_EQ;
	_eq->eq_fid.fid.context = context;
	_eq->eq_fid.fid.ops = &ibv_feq_cm_ops;
	_eq->eq_fid.ops = &ibv_feq_cm_data_ops;

	*eq = &_eq->eq_fid;
	return 0;
err2:
	if (_eq->channel)
		rdma_destroy_event_channel(_eq->channel);
err1:
	free(_eq);
	return ret;
}

static int ibv_pep_listen(struct fid_pep *pep)
{
	struct ibv_pep *_pep;

	_pep = container_of(pep, struct ibv_pep, pep_fid);
	return rdma_listen(_pep->id, 0) ? -errno : 0;
}

static struct fi_ops_cm ibv_pep_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.listen = ibv_pep_listen,
};

static int ibv_pep_bind(fid_t fid, struct fid *bfid, uint64_t flags)
{
	struct ibv_pep *pep;
	struct ibv_feq_cm *eq;
	int ret;

	pep = container_of(fid, struct ibv_pep, pep_fid.fid);
	if (bfid->fclass != FID_CLASS_EQ)
		return -FI_EINVAL;

	eq = container_of(bfid, struct ibv_feq_cm, eq_fid.fid);
	pep->cm_eq = eq;
	ret = rdma_migrate_id(pep->id, pep->cm_eq->channel);
	if (ret)
		return -errno;

	return 0;
}

static int ibv_pep_close(fid_t fid)
{
	struct ibv_pep *pep;

	pep = container_of(fid, struct ibv_pep, pep_fid.fid);
	if (pep->id)
		rdma_destroy_ep(pep->id);

	free(pep);
	return 0;
}

static struct fi_ops ibv_pep_ops = {
	.size = sizeof(struct fi_ops),
	.close = ibv_pep_close,
	.bind = ibv_pep_bind
};

static int
ibv_pendpoint(struct fid_fabric *fabric, struct fi_info *info,
	      struct fid_pep **pep, void *context)
{
	struct ibv_fabric *fab;
	struct ibv_pep *_pep;

	fab = container_of(fabric, struct ibv_fabric, fabric_fid);
	if (strcmp(fab->name, info->fabric_name))
		return -FI_EINVAL;

	if (!info->data || info->datalen != sizeof(_pep->id))
		return -FI_ENOSYS;

	_pep = calloc(1, sizeof *_pep);
	if (!_pep)
		return -FI_ENOMEM;

	_pep->id = info->data;
	_pep->id->context = &_pep->pep_fid.fid;
	info->data = NULL;
	info->datalen = 0;

	_pep->pep_fid.fid.fclass = FID_CLASS_PEP;
	_pep->pep_fid.fid.context = context;
	_pep->pep_fid.fid.ops = &ibv_pep_ops;
	_pep->pep_fid.cm = &ibv_pep_cm_ops;

	*pep = &_pep->pep_fid;
	return 0;
}

static int ibv_fabric_close(fid_t fid)
{
	free(fid);
	return 0;
}

static struct fi_ops ibv_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = ibv_fabric_close,
};

static struct fi_ops_fabric ibv_ops_fabric = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = ibv_domain,
	.endpoint = ibv_pendpoint,
	.eq_open = ibv_feq_open,
};

int ibv_fabric(const char *name, uint64_t flags, struct fid_fabric **fabric,
	       void *context)
{
	struct ibv_fabric *fab;

	fab = calloc(1, sizeof(*fab));
	if (!fab)
		return -FI_ENOMEM;

	fab->fabric_fid.fid.fclass = FID_CLASS_FABRIC;
	fab->fabric_fid.fid.context = context;
	fab->fabric_fid.fid.ops = &ibv_fi_ops;
	fab->fabric_fid.ops = &ibv_ops_fabric;
	strncpy(fab->name, name, FI_NAME_MAX);
	fab->flags = flags;
	*fabric = &fab->fabric_fid;
	return 0;
}

static struct fi_ops_prov ibv_ops = {
	.size = sizeof(struct fi_ops_prov),
	.getinfo = ibv_getinfo,
	.freeinfo = ibv_freeinfo,
	.domain = ibv_domain,
	.fabric = ibv_fabric,
};

void ibv_ini(void)
{
	(void) fi_register(&ibv_ops);
}

void ibv_fini(void)
{
}
