/*
 * Copyright (c) 2013 Intel Corporation, Inc.  All rights reserved.
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
#include <rdma/fi_socket.h>
#include <rdma/fi_rdma.h>
#include <rdma/fi_errno.h>

#include "ibverbs.h"


struct ibv_domain {
	struct fid_domain	domain_fid;
	struct ibv_context	*verbs;
	struct ibv_pd		*pd;
};

struct ibv_ec {
	struct fid_ec		fid;
	enum fi_ec_domain	ec_domain;
	struct ibv_domain	*domain;
};

struct ibv_ec_comp {
	struct ibv_ec		ec;
	struct ibv_comp_channel	*channel;
	struct ibv_cq		*cq;
	uint64_t		flags;
	struct ibv_wc		wc;
};

struct ibv_ec_cm {
	struct ibv_ec		ec;
	struct rdma_event_channel *channel;
	uint64_t		flags;
	struct fi_ec_err_entry	err;
};

struct ibv_mem_desc {
	struct fid_mr		mr_fid;
	struct ibv_mr		*mr;
	struct ibv_domain	*domain;
};

struct ibv_msg_socket {
	struct fid_socket	socket_fid;
	struct rdma_cm_id	*id;
	struct ibv_ec_cm	*cm_ec;
	struct ibv_ec_comp	*rec;
	struct ibv_ec_comp	*sec;
	uint32_t		inline_size;
};

static char def_send_wr[16] = "384";
static char def_recv_wr[16] = "384";
static char def_send_sge[16] = "4";
static char def_recv_sge[16] = "4";
static char def_inline_data[16] = "64";

static int ibv_check_domain(const char *name)
{
	return (!name || !strncmp(name, IBV_PREFIX "/", sizeof(IBV_PREFIX))) ?
		0 : -ENODATA;
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
	if (fi->type == FID_MSG || fi->protocol_cap & FI_PROTO_CAP_RDMA ||
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
	if (fi->dst_addrlen) {
		if (!(rai->ai_dst_addr = malloc(fi->dst_addrlen)))
			return ENOMEM;
		memcpy(rai->ai_dst_addr, fi->dst_addr, fi->dst_addrlen);
		rai->ai_dst_len = fi->dst_addrlen;
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
		fi->protocol_cap = FI_PROTO_CAP_MSG | FI_PROTO_CAP_RDMA;
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
 		if (!(fi->dst_addr = malloc(rai->ai_dst_len)))
 			return ENOMEM;
 		memcpy(fi->dst_addr, rai->ai_dst_addr, rai->ai_dst_len);
 		fi->dst_addrlen = rai->ai_dst_len;
 	}

 	return 0;
 }

static int ibv_getinfo(char *node, char *service, struct fi_info *hints,
		      struct fi_info **info)
{
	struct rdma_addrinfo rai_hints, *rai;
	struct fi_info *fi;
	struct rdma_cm_id *id;
	int ret;

	if (hints) {
		ret = ibv_check_domain(hints->domain_name);
		if (ret)
			return ret;

		ret = ibv_fi_to_rai(hints, &rai_hints);
		if (ret)
			return ret;

		ret = rdma_getaddrinfo(node, service, &rai_hints, &rai);
	} else {
		ret = rdma_getaddrinfo(node, service, NULL, &rai);
	}
	if (ret)
		return -errno;

	if (!(fi = malloc(sizeof *fi))) {
		ret = ENOMEM;
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
		fi->src_addrlen = rdma_addrlen(rdma_get_local_addr(id));
		if (!(fi->src_addr = malloc(fi->src_addrlen))) {
			ret = -ENOMEM;
			goto err3;
		}
		memcpy(fi->src_addr, rdma_get_local_addr(id), fi->src_addrlen);
	}

	if (id->verbs) {
		if (!(fi->domain_name = malloc(FI_NAME_MAX))) {
			ret = -ENOMEM;
			goto err3;
		}
		strcpy(fi->domain_name, IBV_PREFIX "/");
		strcpy(&fi->domain_name[sizeof(IBV_PREFIX)], id->verbs->device->name);
	} else {
		fi->domain_name = strdup(IBV_PREFIX "/" FI_UNBOUND_NAME);
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
	int ret;

	ret = ibv_check_domain(info->domain_name);
	if (ret)
		return ret;

	if (info->data) {
		rdma_destroy_ep(info->data);
		info->data = NULL;
	}
	__fi_freeinfo(info);
	return 0;
}

static int ibv_msg_socket_create_qp(struct ibv_msg_socket *sock)
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
	if (!sock->inline_size)
		sock->inline_size = atoi(def_inline_data);
	attr.cap.max_inline_data = sock->inline_size;
	attr.qp_context = sock;
	attr.send_cq = sock->sec->cq;
	attr.recv_cq = sock->rec->cq;
	attr.srq = NULL;
	attr.qp_type = IBV_QPT_RC;
	attr.sq_sig_all = 1;

	return rdma_create_qp(sock->id, sock->rec->ec.domain->pd, &attr) ? -errno : 0;
}

static int ibv_msg_socket_bind(fid_t fid, struct fi_resource *fids, int nfids)
{
	struct ibv_msg_socket *sock;
	struct ibv_ec *ec;
	int i, ret;

	sock = container_of(fid, struct ibv_msg_socket, socket_fid.fid);
	for (i = 0; i < nfids; i++) {
		if (fids[i].fid->fclass != FID_CLASS_EC)
			return -EINVAL;

		ec = container_of(fids[i].fid, struct ibv_ec, fid.fid);
		if (fids[i].flags & FI_RECV) {
			if (sock->rec)
				return -EINVAL;
			sock->rec = container_of(ec, struct ibv_ec_comp, ec);
		}
		if (fids[i].flags & FI_SEND) {
			if (sock->sec)
				return -EINVAL;
			sock->sec = container_of(ec, struct ibv_ec_comp, ec);
		}
		if (ec->ec_domain == FI_EC_DOMAIN_CM) {
			sock->cm_ec = container_of(ec, struct ibv_ec_cm, ec);
			ret = rdma_migrate_id(sock->id, sock->cm_ec->channel);
			if (ret)
				return -errno;
		}
	}

	if (sock->sec && sock->rec && !sock->id->qp) {
		ret = ibv_msg_socket_create_qp(sock);
		if (ret)
			return ret;
	}

	return 0;
}

static ssize_t ibv_msg_socket_recvmem(fid_t fid, void *buf, size_t len,
				      uint64_t mem_desc, void *context)
{
	struct ibv_msg_socket *sock;
	struct ibv_recv_wr wr, *bad;
	struct ibv_sge sge;

	sge.addr = (uintptr_t) buf;
	sge.length = (uint32_t) len;
	sge.lkey = (uint32_t) mem_desc;

	wr.wr_id = (uintptr_t) context;
	wr.next = NULL;
	wr.sg_list = &sge;
	wr.num_sge = 1;

	sock = container_of(fid, struct ibv_msg_socket, socket_fid.fid);
	return -ibv_post_recv(sock->id->qp, &wr, &bad);
}

static ssize_t ibv_msg_socket_sendmem(fid_t fid, const void *buf, size_t len,
				      uint64_t mem_desc, void *context)
{
	struct ibv_msg_socket *sock;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge sge;

	sock = container_of(fid, struct ibv_msg_socket, socket_fid.fid);
	sge.addr = (uintptr_t) buf;
	sge.length = (uint32_t) len;
	sge.lkey = (uint32_t) mem_desc;

	wr.wr_id = (uintptr_t) context;
	wr.next = NULL;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.opcode = IBV_WR_SEND;
	wr.send_flags = (len <= sock->inline_size) ? IBV_SEND_INLINE : 0;

	return -ibv_post_send(sock->id->qp, &wr, &bad);
}

static struct fi_ops_msg ibv_msg_socket_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recvmem = ibv_msg_socket_recvmem,
	.sendmem = ibv_msg_socket_sendmem,
};

static int ibv_msg_socket_rdma_writemem(fid_t fid, const void *buf, size_t len,
	uint64_t mem_desc, uint64_t addr, be64_t tag, void *context)
{
	struct ibv_msg_socket *sock;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge sge;

	sock = container_of(fid, struct ibv_msg_socket, socket_fid.fid);
	sge.addr = (uintptr_t) buf;
	sge.length = (uint32_t) len;
	sge.lkey = (uint32_t) mem_desc;

	wr.wr_id = (uintptr_t) context;
	wr.next = NULL;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.opcode = IBV_WR_RDMA_WRITE;
	wr.send_flags = (len <= sock->inline_size) ? IBV_SEND_INLINE : 0;
	wr.wr.rdma.remote_addr = addr;
	wr.wr.rdma.rkey = (uint32_t) tag;

	return -ibv_post_send(sock->id->qp, &wr, &bad);
}

static int ibv_msg_socket_rdma_readmem(fid_t fid, void *buf, size_t len,
	uint64_t mem_desc, uint64_t addr, be64_t tag, void *context)
{
	struct ibv_msg_socket *sock;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge sge;

	sge.addr = (uintptr_t) buf;
	sge.length = (uint32_t) len;
	sge.lkey = (uint32_t) mem_desc;

	wr.wr_id = (uintptr_t) context;
	wr.next = NULL;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.opcode = IBV_WR_RDMA_READ;
	wr.send_flags = 0;
	wr.wr.rdma.remote_addr = addr;
	wr.wr.rdma.rkey = (uint32_t) tag;

	sock = container_of(fid, struct ibv_msg_socket, socket_fid.fid);
	return -ibv_post_send(sock->id->qp, &wr, &bad);
}

static struct fi_ops_rdma ibv_msg_socket_rdma_ops = {
	.size = sizeof(struct fi_ops_rdma),
	.writemem = ibv_msg_socket_rdma_writemem,
	.readmem = ibv_msg_socket_rdma_readmem
};

static int ibv_msg_socket_connect(fid_t fid, const void *param, size_t paramlen)
{
	struct ibv_msg_socket *sock;
	struct rdma_conn_param conn_param;

	sock = container_of(fid, struct ibv_msg_socket, socket_fid.fid);
	memset(&conn_param, 0, sizeof conn_param);
	conn_param.private_data = param;
	conn_param.private_data_len = paramlen;
	conn_param.responder_resources = RDMA_MAX_RESP_RES;
	conn_param.initiator_depth = RDMA_MAX_INIT_DEPTH;
	conn_param.flow_control = 1;
	conn_param.retry_count = 15;
	conn_param.rnr_retry_count = 7;

	return rdma_connect(sock->id, &conn_param) ? -errno : 0;
}

static int ibv_msg_socket_listen(fid_t fid)
{
	struct ibv_msg_socket *sock;

	sock = container_of(fid, struct ibv_msg_socket, socket_fid.fid);
	return rdma_listen(sock->id, 0) ? -errno : 0;
}

static int ibv_msg_socket_accept(fid_t fid, const void *param, size_t paramlen)
{
	struct ibv_msg_socket *sock;
	struct rdma_conn_param conn_param;

	sock = container_of(fid, struct ibv_msg_socket, socket_fid.fid);
	memset(&conn_param, 0, sizeof conn_param);
	conn_param.private_data = param;
	conn_param.private_data_len = paramlen;
	conn_param.responder_resources = RDMA_MAX_RESP_RES;
	conn_param.initiator_depth = RDMA_MAX_INIT_DEPTH;
	conn_param.flow_control = 1;
	conn_param.rnr_retry_count = 7;

	return rdma_accept(sock->id, &conn_param) ? -errno : 0;
}

static int ibv_msg_socket_reject(fid_t fid, struct fi_info *info,
				 const void *param, size_t paramlen)
{
	return rdma_reject(info->data, param, (uint8_t) paramlen) ? -errno : 0;
}

static int ibv_msg_socket_shutdown(fid_t fid, uint64_t flags)
{
	struct ibv_msg_socket *sock;
	sock = container_of(fid, struct ibv_msg_socket, socket_fid.fid);
	return rdma_disconnect(sock->id) ? -errno : 0;
}

struct fi_ops_cm ibv_msg_socket_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.connect = ibv_msg_socket_connect,
	.listen = ibv_msg_socket_listen,
	.accept = ibv_msg_socket_accept,
	.reject = ibv_msg_socket_reject,
	.shutdown = ibv_msg_socket_shutdown,
};

static int ibv_msg_socket_getopt(fid_t fid, int level, int optname,
				 void *optval, size_t *optlen)
{
	struct ibv_msg_socket *sock;
	sock = container_of(fid, struct ibv_msg_socket, socket_fid.fid);

	switch (level) {
	case FI_OPT_SOCKET:
		switch (optname) {
		case FI_OPT_MAX_BUFFERED_SEND:
			if (*optlen < sizeof(size_t)) {
				*optlen = sizeof(size_t);
				return -FI_ETOOSMALL;
			}
			*((size_t *) optval) = (size_t) sock->inline_size;
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

static int ibv_msg_socket_setopt(fid_t fid, int level, int optname,
				 const void *optval, size_t optlen)
{
	struct ibv_msg_socket *sock;
	sock = container_of(fid, struct ibv_msg_socket, socket_fid.fid);

	switch (level) {
	case FI_OPT_SOCKET:
		switch (optname) {
		case FI_OPT_MAX_BUFFERED_SEND:
			if (optlen != sizeof(size_t))
				return -FI_EINVAL;
			if (sock->id->qp)
				return -FI_EOPBADSTATE;
			sock->inline_size = (uint32_t) *(size_t *) optval;
			break;
		default:
			return -FI_ENOPROTOOPT;
		}
	default:
		return -FI_ENOPROTOOPT;
	}
	return 0;
}

struct fi_ops_sock ibv_msg_socket_base_ops = {
	.size = sizeof(struct fi_ops_sock),
	.getopt = ibv_msg_socket_getopt,
	.setopt = ibv_msg_socket_setopt,
};

static int ibv_msg_socket_close(fid_t fid)
{
	struct ibv_msg_socket *sock;

	sock = container_of(fid, struct ibv_msg_socket, socket_fid.fid);
	if (sock->id)
		rdma_destroy_ep(sock->id);

	free(sock);
	return 0;
}

struct fi_ops ibv_msg_socket_ops = {
	.size = sizeof(struct fi_ops),
	.close = ibv_msg_socket_close,
	.bind = ibv_msg_socket_bind
};

static int ibv_socket(struct fi_info *info, fid_t *fid, void *context)
{
	struct ibv_msg_socket *sock;
	int ret;

	ret = ibv_check_domain(info->domain_name);
	if (ret)
		return ret;

	if (!info->data || info->datalen != sizeof(sock->id))
		return -ENOSYS;

	sock = calloc(1, sizeof *sock);
	if (!sock)
		return -ENOMEM;

	sock->id = info->data;
	sock->id->context = &sock->socket_fid.fid;
	info->data = NULL;
	info->datalen = 0;

	sock->socket_fid.fid.fclass = FID_CLASS_SOCKET;
	sock->socket_fid.fid.size = sizeof(struct fid_socket);
	sock->socket_fid.fid.context = context;
	sock->socket_fid.fid.ops = &ibv_msg_socket_ops;
	sock->socket_fid.ops = &ibv_msg_socket_base_ops;
	sock->socket_fid.msg = &ibv_msg_socket_msg_ops;
	sock->socket_fid.cm = &ibv_msg_socket_cm_ops;
	sock->socket_fid.rdma = &ibv_msg_socket_rdma_ops;

	*fid = &sock->socket_fid.fid;
	return 0;
}

static int ibv_poll_fd(int fd)
{
	struct pollfd fds;

	fds.fd = fd;
	fds.events = POLLIN;
	return poll(&fds, 1, -1) < 0 ? -errno : 0;
}

static ssize_t ibv_ec_cm_readerr(fid_t fid, void *buf, size_t len, uint64_t flags)
{
	struct ibv_ec_cm *ec;
	struct fi_ec_err_entry *entry;

	ec = container_of(fid, struct ibv_ec_cm, ec.fid.fid);
	if (!ec->err.err)
		return 0;

	if (len < sizeof(*entry))
		return -EINVAL;

	entry = (struct fi_ec_err_entry *) buf;
	*entry = ec->err;
	ec->err.err = 0;
	ec->err.prov_errno = 0;
	return sizeof(*entry);
}

static struct fi_info * ibv_ec_cm_getinfo(struct rdma_cm_event *event)
{
	struct fi_info *fi;

	fi = calloc(1, sizeof *fi);
	if (!fi)
		return NULL;

	fi->size = sizeof *fi;
	fi->type = FID_MSG;
	if (event->id->verbs->device->transport_type == IBV_TRANSPORT_IWARP) {
		fi->protocol = FI_PROTO_IWARP;
		fi->protocol_cap = FI_PROTO_CAP_RDMA;
	} else {
		fi->protocol = FI_PROTO_IB_RC;
		fi->protocol_cap = FI_PROTO_CAP_RDMA;
	}
//	fi->sa_family = rdma_get_local_addr(event->id)->sa_family;

	fi->src_addrlen = rdma_addrlen(rdma_get_local_addr(event->id));
	if (!(fi->src_addr = malloc(fi->src_addrlen)))
		goto err;
	memcpy(fi->src_addr, rdma_get_local_addr(event->id), fi->src_addrlen);

	fi->dst_addrlen = rdma_addrlen(rdma_get_peer_addr(event->id));
	if (!(fi->dst_addr = malloc(fi->dst_addrlen)))
		goto err;
	memcpy(fi->dst_addr, rdma_get_peer_addr(event->id), fi->dst_addrlen);

	if (!(fi->domain_name = malloc(FI_NAME_MAX)))
		goto err;
	strcpy(fi->domain_name, IBV_PREFIX "/");
	strcpy(&fi->domain_name[sizeof(IBV_PREFIX)], event->id->verbs->device->name);

	fi->datalen = sizeof event->id;
	fi->data = event->id;
	return fi;
err:
	fi_freeinfo(fi);
	return NULL;
}

static ssize_t ibv_ec_cm_process_event(struct ibv_ec_cm *ec,
	struct rdma_cm_event *event, struct fi_ec_cm_entry *entry, size_t len)
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
		entry->info = ibv_ec_cm_getinfo(event);
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
		ec->err.fid_context = fid->context;
		ec->err.err = event->status;
		return -EIO;
	case RDMA_CM_EVENT_REJECTED:
		ec->err.fid_context = fid->context;
		ec->err.err = ECONNREFUSED;
		ec->err.prov_errno = event->status;
		return -EIO;
	case RDMA_CM_EVENT_DEVICE_REMOVAL:
		ec->err.fid_context = fid->context;
		ec->err.err = ENODEV;
		return -EIO;
	case RDMA_CM_EVENT_ADDR_CHANGE:
		ec->err.fid_context = fid->context;
		ec->err.err = EADDRNOTAVAIL;
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

static ssize_t ibv_ec_cm_read_data(fid_t fid, void *buf, size_t len)
{
	struct ibv_ec_cm *ec;
	struct fi_ec_cm_entry *entry;
	struct rdma_cm_event *event;
	size_t left;
	ssize_t ret = -EINVAL;

	ec = container_of(fid, struct ibv_ec_cm, ec.fid.fid);
	entry = (struct fi_ec_cm_entry *) buf;
	if (ec->err.err)
		return -EIO;

	for (left = len; left >= sizeof(*entry); ) {
		ret = rdma_get_cm_event(ec->channel, &event);
		if (!ret) {
			ret = ibv_ec_cm_process_event(ec, event, entry, left);
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

			if (ec->flags & FI_NONBLOCK)
				return 0;

			ibv_poll_fd(ec->channel->fd);
		} else {
			ret = -errno;
			break;
		}
	}

	return (left < len) ? len - left : ret;
}

static const char * ibv_ec_cm_strerror(fid_t fid, int prov_errno, void *prov_data,
					 void *buf, size_t len)
{
	if (buf && len)
		strncpy(buf, strerror(prov_errno), len);
	return strerror(prov_errno);
}

struct fi_ops_ec ibv_ec_cm_data_ops = {
	.size = sizeof(struct fi_ops_ec),
	.read = ibv_ec_cm_read_data,
	.readfrom = NULL,
	.readerr = ibv_ec_cm_readerr,
	.write = NULL,
	.reset = NULL,
	.strerror = ibv_ec_cm_strerror
};

static int ibv_ec_cm_close(fid_t fid)
{
	struct ibv_ec_cm *ec;

	ec = container_of(fid, struct ibv_ec_cm, ec.fid.fid);
	if (ec->channel)
		rdma_destroy_event_channel(ec->channel);

	free(ec);
	return 0;
}

struct fi_ops ibv_ec_cm_ops = {
	.size = sizeof(struct fi_ops),
	.close = ibv_ec_cm_close,
};

static int ibv_ec_cm_open(fid_t fid, struct fi_ec_attr *attr, fid_t *ec, void *context)
{
	struct ibv_ec_cm *vec;
	long flags = 0;
	int ret;

	if (attr->type != FI_EC_QUEUE || attr->format != FI_EC_FORMAT_CM)
		return -ENOSYS;

	vec = calloc(1, sizeof *vec);
	if (!vec)
		return -ENOMEM;

	vec->ec.domain = container_of(fid, struct ibv_domain, domain_fid.fid);

	switch (attr->wait_obj) {
	case FI_EC_WAIT_FD:
		vec->channel = rdma_create_event_channel();
		if (!vec->channel) {
			ret = -errno;
			goto err1;
		}
		fcntl(vec->channel->fd, F_GETFL, &flags);
		ret = fcntl(vec->channel->fd, F_SETFL, flags | O_NONBLOCK);
		if (ret) {
			ret = -errno;
			goto err2;
		}
		break;
	case FI_EC_WAIT_NONE:
		vec->flags = O_NONBLOCK;
		break;
	default:
		return -ENOSYS;
	}

	vec->flags = attr->flags;
	vec->ec.fid.fid.fclass = FID_CLASS_EC;
	vec->ec.fid.fid.size = sizeof(struct fid_ec);
	vec->ec.fid.fid.context = context;
	vec->ec.fid.fid.ops = &ibv_ec_cm_ops;
	vec->ec.fid.ops = &ibv_ec_cm_data_ops;

	*ec = &vec->ec.fid.fid;
	return 0;
err2:
	if (vec->channel)
		rdma_destroy_event_channel(vec->channel);
err1:
	free(vec);
	return ret;
}

static int ibv_ec_comp_reset(fid_t fid, void *cond)
{
	struct ibv_ec_comp *ec;
	struct ibv_cq *cq;
	void *context;
	int ret;

	ec = container_of(fid, struct ibv_ec_comp, ec.fid.fid);
	ret = ibv_get_cq_event(ec->channel, &cq	, &context);
	if (!ret)
		ibv_ack_cq_events(cq, 1);

	return -ibv_req_notify_cq(ec->cq, (ec->flags & FI_SIGNAL) ? 1 : 0);
}

static ssize_t ibv_ec_comp_readerr(fid_t fid, void *buf, size_t len, uint64_t flags)
{
	struct ibv_ec_comp *ec;
	struct fi_ec_err_entry *entry;

	ec = container_of(fid, struct ibv_ec_comp, ec.fid.fid);
	if (!ec->wc.status)
		return 0;

	if (len < sizeof(*entry))
		return -EINVAL;

	entry = (struct fi_ec_err_entry *) buf;
	entry->fid_context = NULL;	/* TODO: return qp context from wc */
	entry->op_context = (void *) (uintptr_t) ec->wc.wr_id;
	entry->flags = 0;
	entry->err = EIO;
	entry->prov_errno = ec->wc.status;
	entry->data = ec->wc.vendor_err;
	entry->prov_data = NULL;

	ec->wc.status = 0;
	return sizeof(*entry);
}

static ssize_t ibv_ec_comp_read(fid_t fid, void *buf, size_t len)
{
	struct ibv_ec_comp *ec;
	struct fi_ec_entry *entry;
	size_t left;
	int reset = 1, ret = -EINVAL;

	ec = container_of(fid, struct ibv_ec_comp, ec.fid.fid);
	entry = (struct fi_ec_entry *) buf;
	if (ec->wc.status)
		return -EIO;

	for (left = len; left >= sizeof(*entry); ) {
		ret = ibv_poll_cq(ec->cq, 1, &ec->wc);
		if (ret > 0) {
			if (ec->wc.status) {
				ret = -EIO;
				break;
			}

			entry->op_context = (void *) (uintptr_t) ec->wc.wr_id;
			left -= sizeof(*entry);
			entry = entry + 1;
		} else if (ret == 0) {
			if (left < len)
				return len - left;

			if (reset && (ec->flags & FI_AUTO_RESET)) {
				ibv_ec_comp_reset(fid, NULL);
				reset = 0;
				continue;
			}

			if (ec->flags & FI_NONBLOCK)
				return 0;

			ibv_poll_fd(ec->channel->fd);
		} else {
			break;
		}
	}

	return (left < len) ? len - left : ret;
}

static ssize_t ibv_ec_comp_read_data(fid_t fid, void *buf, size_t len)
{
	struct ibv_ec_comp *ec;
	struct fi_ec_data_entry *entry;
	size_t left;
	int reset = 1, ret = -EINVAL;

	ec = container_of(fid, struct ibv_ec_comp, ec.fid.fid);
	entry = (struct fi_ec_data_entry *) buf;
	if (ec->wc.status)
		return -EIO;

	for (left = len; left >= sizeof(*entry); ) {
		ret = ibv_poll_cq(ec->cq, 1, &ec->wc);
		if (ret > 0) {
			if (ec->wc.status) {
				ret = -EIO;
				break;
			}

			entry->op_context = (void *) (uintptr_t) ec->wc.wr_id;
			if (ec->wc.wc_flags & IBV_WC_WITH_IMM) {
				entry->flags = FI_IMM;
				entry->data = ec->wc.imm_data;
			}
			if (ec->wc.opcode & IBV_WC_RECV)
				entry->len = ec->wc.byte_len;
			left -= sizeof(*entry);
			entry = entry + 1;
		} else if (ret == 0) {
			if (left < len)
				return len - left;

			if (reset && (ec->flags & FI_AUTO_RESET)) {
				ibv_ec_comp_reset(fid, NULL);
				reset = 0;
				continue;
			}

			if (ec->flags & FI_NONBLOCK)
				return 0;

			ibv_poll_fd(ec->channel->fd);
		} else {
			break;
		}
	}

	return (left < len) ? len - left : ret;
}

static const char * ibv_ec_comp_strerror(fid_t fid, int prov_errno, void *prov_data,
					 void *buf, size_t len)
{
	if (buf && len)
		strncpy(buf, ibv_wc_status_str(prov_errno), len);
	return ibv_wc_status_str(prov_errno);
}

struct fi_ops_ec ibv_ec_comp_context_ops = {
	.size = sizeof(struct fi_ops_ec),
	.read = ibv_ec_comp_read,
	.readerr = ibv_ec_comp_readerr,
	.reset = ibv_ec_comp_reset,
	.strerror = ibv_ec_comp_strerror
};

struct fi_ops_ec ibv_ec_comp_data_ops = {
	.size = sizeof(struct fi_ops_ec),
	.read = ibv_ec_comp_read_data,
	.readerr = ibv_ec_comp_readerr,
	.reset = ibv_ec_comp_reset,
	.strerror = ibv_ec_comp_strerror
};

static int ibv_ec_comp_close(fid_t fid)
{
	struct ibv_ec_comp *ec;
	int ret;

	ec = container_of(fid, struct ibv_ec_comp, ec.fid.fid);
	if (ec->cq) {
		ret = ibv_destroy_cq(ec->cq);
		if (ret)
			return -ret;
		ec->cq = NULL;
	}
	if (ec->channel)
		ibv_destroy_comp_channel(ec->channel);

	free(ec);
	return 0;
}

struct fi_ops ibv_ec_comp_ops = {
	.size = sizeof(struct fi_ops),
	.close = ibv_ec_comp_close,
};

static int ibv_ec_comp_open(fid_t fid, struct fi_ec_attr *attr, fid_t *ec, void *context)
{
	struct ibv_ec_comp *vec;
	long flags = 0;
	int ret;

	if (attr->type != FI_EC_QUEUE || attr->wait_cond != FI_EC_COND_NONE)
		return -ENOSYS;

	vec = calloc(1, sizeof *vec);
	if (!vec)
		return -ENOMEM;

	vec->ec.domain = container_of(fid, struct ibv_domain, domain_fid.fid);

	switch (attr->wait_obj) {
	case FI_EC_WAIT_FD:
		vec->channel = ibv_create_comp_channel(vec->ec.domain->verbs);
		if (!vec->channel) {
			ret = -errno;
			goto err1;
		}
		fcntl(vec->channel->fd, F_GETFL, &flags);
		ret = fcntl(vec->channel->fd, F_SETFL, flags | O_NONBLOCK);
		if (ret) {
			ret = -errno;
			goto err1;
		}
		break;
	case FI_EC_WAIT_NONE:
		vec->flags = FI_NONBLOCK;
		break;
	default:
		return -ENOSYS;
	}

	vec->cq = ibv_create_cq(vec->ec.domain->verbs, attr->size, vec,
				vec->channel, attr->signaling_vector);
	if (!vec->cq) {
		ret = -errno;
		goto err2;
	}

	vec->flags |= attr->flags;
	vec->ec.fid.fid.fclass = FID_CLASS_EC;
	vec->ec.fid.fid.size = sizeof(struct fid_ec);
	vec->ec.fid.fid.context = context;
	vec->ec.fid.fid.ops = &ibv_ec_comp_ops;

	switch (attr->format) {
	case FI_EC_FORMAT_CONTEXT:
		vec->ec.fid.ops = &ibv_ec_comp_context_ops;
		break;
	case FI_EC_FORMAT_DATA:
		vec->ec.fid.ops = &ibv_ec_comp_data_ops;
		break;
	default:
		ret = -ENOSYS;
		goto err3;
	}

	*ec = &vec->ec.fid.fid;
	return 0;

err3:
	ibv_destroy_cq(vec->cq);
err2:
	if (vec->channel)
		ibv_destroy_comp_channel(vec->channel);
err1:
	free(vec);
	return ret;
}

static int ibv_ec_open(fid_t fid, struct fi_ec_attr *attr, fid_t *ec, void *context)
{
	struct ibv_ec *vec;
	int ret;

	switch (attr->domain) {
	case FI_EC_DOMAIN_GENERAL:
		return -ENOSYS;
	case FI_EC_DOMAIN_COMP:
		ret = ibv_ec_comp_open(fid, attr, ec, context);
		break;
	case FI_EC_DOMAIN_CM:
		ret  = ibv_ec_cm_open(fid, attr, ec, context);
		break;
	case FI_EC_DOMAIN_AV:
		return -ENOSYS;
	default:
		return -ENOSYS;
	}
	if (ret)
		return ret;

	vec = container_of(*ec, struct ibv_ec, fid);
	vec->ec_domain = attr->domain;

	if (attr->flags & FI_AUTO_RESET && vec->fid.ops->reset)
		fi_ec_reset(*ec, attr->cond);

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

struct fi_ops ibv_mr_ops = {
	.size = sizeof(struct fi_ops),
	.close = ibv_mr_close
};

static int ibv_mr_reg(fid_t fid, const void *buf, size_t len,
		      struct fi_mr_attr *attr, fid_t *mr, void *context)
{
	struct ibv_mem_desc *md;
	int access;

	md = calloc(1, sizeof *md);
	if (!md)
		return -ENOMEM;

	md->domain = container_of(fid, struct ibv_domain, domain_fid.fid);
	md->mr_fid.fid.fclass = FID_CLASS_MR;
	md->mr_fid.fid.size = sizeof(struct fid_mr);
	md->mr_fid.fid.context = context;
	md->mr_fid.fid.ops = &ibv_mr_ops;

	access = IBV_ACCESS_LOCAL_WRITE;
	if (attr) {
		if (attr->access & FI_READ)
			access |= IBV_ACCESS_REMOTE_READ;
		if (attr->access & FI_WRITE)
			access |= IBV_ACCESS_REMOTE_WRITE;
	}
	md->mr = ibv_reg_mr(md->domain->pd, (void *) buf, len, access);
	if (!md->mr)
		goto err;

	md->mr_fid.mem_desc = md->mr->lkey;
	md->mr_fid.key = md->mr->rkey;
	*mr = &md->mr_fid.fid;
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
	int i, ret = -ENODEV;

	name = name + sizeof(IBV_PREFIX);
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

struct fi_ops ibv_fid_ops = {
	.size = sizeof(struct fi_ops),
	.close = ibv_close,
};

struct fi_ops_domain ibv_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.mr_reg = ibv_mr_reg,
	.ec_open = ibv_ec_open
};

static int ibv_open(const char *name, struct fi_info *info,
		    fid_t *fid, void *context)
{
	struct ibv_domain *domain;
	const char *domain_name;
	int ret;

	domain_name = name ? name : info->domain_name;
	ret = ibv_check_domain(domain_name);
	if (ret)
		return ret;

	domain = calloc(1, sizeof *domain);
	if (!domain)
		return -ENOMEM;

	if (strcmp(domain_name + sizeof(IBV_PREFIX), "local")) {
		ret = ibv_open_device_by_name(domain, domain_name);
		if (ret)
			goto err;

		domain->pd = ibv_alloc_pd(domain->verbs);
		if (!domain->pd) {
			ret = -errno;
			goto err;
		}
	}

	domain->domain_fid.fid.fclass = FID_CLASS_RESOURCE_DOMAIN;
	domain->domain_fid.fid.size = sizeof(struct fid_domain);
	domain->domain_fid.fid.context = context;
	domain->domain_fid.fid.ops = &ibv_fid_ops;
	domain->domain_fid.ops = &ibv_domain_ops;

	*fid = &domain->domain_fid.fid;
	return 0;
err:
	free(domain);
	return ret;
}

struct fi_ops_prov ibv_ops = {
	.size = sizeof(struct fi_ops_prov),
	.getinfo = ibv_getinfo,
	.freeinfo = ibv_freeinfo,
	.socket = ibv_socket,
	.open = ibv_open
};


void ibv_ini(void)
{
	fi_register(&ibv_ops);
}

void ibv_fini(void)
{
}
