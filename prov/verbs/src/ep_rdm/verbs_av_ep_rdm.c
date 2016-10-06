/*
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
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

#include <arpa/inet.h>

#include "verbs_rdm.h"


extern struct fi_provider fi_ibv_prov;

ssize_t
fi_ibv_rdm_start_connection(struct fi_ibv_rdm_ep *ep, 
			    struct fi_ibv_rdm_conn *conn)
{
	struct rdma_cm_id *id = NULL;
	assert(ep->domain->rdm_cm->listener);

	if (conn->state != FI_VERBS_CONN_ALLOCATED) {
		return FI_SUCCESS;
	}

	if (ep->is_closing) {
		VERBS_INFO(FI_LOG_AV, "Attempt to start connection with addr %s:%u when ep is closing\n",
			inet_ntoa(conn->addr.sin_addr),
			ntohs(conn->addr.sin_port));
		return -FI_EOTHER;
	}

	conn->state = FI_VERBS_CONN_STARTED;

	if (rdma_create_id(ep->domain->rdm_cm->ec, &id, conn, RDMA_PS_TCP)) {
		VERBS_INFO_ERRNO(FI_LOG_AV, "rdma_create_id\n", errno);
		return -errno;
	}

	if (conn->cm_role == FI_VERBS_CM_ACTIVE || 
	    conn->cm_role == FI_VERBS_CM_SELF)
	{
		conn->id[0] = id;
	}

	if (rdma_resolve_addr(id, NULL, (struct sockaddr *)&conn->addr, 30000)) {
		VERBS_INFO_ERRNO(FI_LOG_AV, "rdma_resolve_addr\n", errno);
		return -errno;
	}
	return FI_SUCCESS;
}

static inline ssize_t
fi_ibv_rdm_disconnect_request(struct fi_ibv_rdm_ep *ep,
			      struct fi_ibv_rdm_conn *conn)
{
	struct fi_ibv_rdm_buf *sbuf = NULL;
	ssize_t ret = FI_SUCCESS;

	if (!ep->is_closing) {
		pthread_mutex_lock(&ep->cm_lock);
	}
	/* The latest control message MUST be in order */
	while (conn->postponed_entry) {
		fi_ibv_rdm_tagged_poll(ep);
	}

	/*
	 * Wait untill resources will be available.
	 * (Remote side should process all recvs then release local buffer)
	 */
	while ( !(sbuf = fi_ibv_rdm_prepare_send_resources(conn, ep)) ) {
		if (conn->state != FI_VERBS_CONN_ESTABLISHED) {
			goto out;
		}
	};

	/* To make sure it's still in order */
	assert(!conn->postponed_entry);

	struct ibv_sge sge = {0};
	struct ibv_send_wr wr = {0};
	struct ibv_send_wr *bad_wr = NULL;

	sge.addr = (uintptr_t)(void*)sbuf;
	sge.length = sizeof(*sbuf);
	sge.lkey = conn->s_mr->lkey;

	wr.wr_id = FI_IBV_RDM_PACK_SERVICE_WR(conn);
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.wr.rdma.remote_addr =
		(uintptr_t)fi_ibv_rdm_get_remote_addr(conn, sbuf);
	wr.wr.rdma.rkey = conn->remote_rbuf_rkey;
	wr.send_flags = (sge.length < ep->max_inline_rc)
		? IBV_SEND_INLINE : 0;
	wr.imm_data = 0;
	wr.opcode = ep->eopcode;

	sbuf->service_data.pkt_len = 0;
	sbuf->header.tag = 0;
	sbuf->header.service_tag = 0;

	FI_IBV_RDM_SET_PKTTYPE(sbuf->header.service_tag,
			       FI_IBV_RDM_DISCONNECT_PKT);

	FI_IBV_RDM_INC_SIG_POST_COUNTERS(conn, ep, wr.send_flags);
	if (ibv_post_send(conn->qp[0], &wr, &bad_wr)) {
		assert(0);
		ret =  -errno;
	} else {
		conn->state = FI_VERBS_CONN_DISCONNECT_REQUESTED;

		VERBS_INFO(FI_LOG_EP_DATA, "posted %d bytes, conn %p\n",
			  sge.length, conn);
	}
out:
	if (!ep->is_closing) {
		pthread_mutex_unlock(&ep->cm_lock);
	}
	return ret;
}

ssize_t
fi_ibv_rdm_start_disconnect(struct fi_ibv_rdm_ep *ep,
			    struct fi_ibv_rdm_conn *conn)
{
	ssize_t ret = FI_SUCCESS;

	FI_INFO(&fi_ibv_prov, FI_LOG_AV,
		"Closing connection %p, state %d\n", conn, conn->state);

	if (conn->state == FI_VERBS_CONN_ESTABLISHED) {
		assert(ep);
		ret = fi_ibv_rdm_disconnect_request(ep, conn);
	} else if (conn->id[0]) {
		if (rdma_disconnect(conn->id[0])) {
			VERBS_INFO_ERRNO(FI_LOG_AV, "rdma_disconnect\n", errno);
			ret = -errno;
		}
		conn->state = FI_VERBS_CONN_DISCONNECT_STARTED;
	} else if (conn->state != FI_VERBS_CONN_DISCONNECT_REQUESTED) {
		ret = fi_ibv_rdm_conn_cleanup(conn);
	}

	return ret;
}

static inline int fi_ibv_rdm_av_is_valid_address(struct sockaddr_in *addr)
{
	return addr->sin_family == AF_INET ? 1 : 0;
}


static int fi_ibv_rdm_av_insert(struct fid_av *av_fid, const void *addr,
                                size_t count, fi_addr_t * fi_addr,
                                uint64_t flags, void *context)
{
	struct fi_ibv_av *av = container_of(av_fid, struct fi_ibv_av, av_fid);
	struct fi_ibv_rdm_ep *ep = av->ep;
	size_t i;
	int ret = 0;

	if (ep) {
		pthread_mutex_lock(&ep->cm_lock);
	}

	if (av->used + count > av->count) {
		const size_t new_av_count = av->used + count;
		if (av->type == FI_AV_TABLE) {
			void *p = realloc(av->domain->rdm_cm->conn_table,
					  (new_av_count *
					  sizeof(*av->domain->rdm_cm->conn_table)));
			if (p) {
				av->domain->rdm_cm->conn_table = p;
			} else {
				ret = -FI_ENOMEM;
				goto out;
			}
		}
		av->count = new_av_count;
	}

	for (i = 0; i < count; i++) {
		struct fi_ibv_rdm_conn *conn = NULL;
		void *addr_i = (uint8_t *) addr +
			i * (ep ? ep->addrlen : FI_IBV_RDM_DFLT_ADDRLEN);

		if (!fi_ibv_rdm_av_is_valid_address(addr_i)) {
			if (fi_addr) {
				fi_addr[i] = FI_ADDR_NOTAVAIL;
			}

			FI_INFO(&fi_ibv_prov, FI_LOG_AV,
				"fi_av_insert: bad addr #%i\n", i);

			continue;
		}

		HASH_FIND(hh, av->domain->rdm_cm->conn_hash, addr_i,
			  FI_IBV_RDM_DFLT_ADDRLEN, conn);

		if (!conn) {
			/* If addr_i is not found in HASH then we malloc it.
			 * It could be found if the connection was initiated by the remote
			 * side.
			 */
			conn = memalign(FI_IBV_RDM_MEM_ALIGNMENT, sizeof *conn);
			if (!conn) {
				ret = -FI_ENOMEM;
				goto out;
			}

			memset(conn, 0, sizeof *conn);
			dlist_init(&conn->postponed_requests_head);
			conn->state = FI_VERBS_CONN_ALLOCATED;
			memcpy(&conn->addr, addr_i, FI_IBV_RDM_DFLT_ADDRLEN);
			HASH_ADD(hh, av->domain->rdm_cm->conn_hash, addr,
				 FI_IBV_RDM_DFLT_ADDRLEN, conn);
		}

		if (ep) {
			fi_ibv_rdm_conn_init_cm_role(conn, ep);
		}


		switch (av->type) {
		case FI_AV_MAP:
			if (fi_addr) {
				fi_addr[i] = (uintptr_t) (void *) conn;
			}
			break;
		case FI_AV_TABLE:
			if (fi_addr) {
				fi_addr[i] = av->used;
			}
			av->domain->rdm_cm->conn_table[av->used] = conn;
			break;
		default:
			assert(0);
			break;
		}

		FI_INFO(&fi_ibv_prov, FI_LOG_AV, "fi_av_insert: addr %s:%u conn %p %d\n",
			inet_ntoa(conn->addr.sin_addr),
			ntohs(conn->addr.sin_port), conn, conn->cm_role);

		av->used++;
		ret++;
	}

out:
	if (ep) {
		pthread_mutex_unlock(&ep->cm_lock);
	}
	return ret;
}

static int fi_ibv_rdm_av_remove(struct fid_av *av_fid, fi_addr_t * fi_addr,
                                size_t count, uint64_t flags)
{
	struct fi_ibv_av *av = container_of(av_fid, struct fi_ibv_av, av_fid);
	struct fi_ibv_rdm_conn *conn = NULL;
	int ret = FI_SUCCESS;
	int err = FI_SUCCESS;
	int i;

	if (!fi_addr || (av->type != FI_AV_MAP && av->type != FI_AV_TABLE)) {
		return -FI_EINVAL;
	}

	if (av->ep) {
		pthread_mutex_lock(&av->ep->cm_lock);
	}

	for (i = 0; i < count; i++) {

		if (fi_addr[i] == FI_ADDR_NOTAVAIL) {
			continue;
		}

		if (av->type == FI_AV_MAP) {
			conn = (struct fi_ibv_rdm_conn *) fi_addr[i];
		} else { /* (av->type == FI_AV_TABLE) */
			conn = av->domain->rdm_cm->conn_table[fi_addr[i]];
		}

		FI_INFO(&fi_ibv_prov, FI_LOG_AV, "av_remove conn %p, addr %s:%u\n",
			conn, inet_ntoa(conn->addr.sin_addr),
			ntohs(conn->addr.sin_port));
		HASH_DEL(av->domain->rdm_cm->conn_hash, conn);
		err = fi_ibv_rdm_start_disconnect(av->ep, conn);
		ret = (ret == FI_SUCCESS) ? err : ret;
	}

	if (av->ep) {
		pthread_mutex_unlock(&av->ep->cm_lock);
	}
	return ret;
}

static struct fi_ops_av fi_ibv_rdm_av_ops = {
	.size = sizeof(struct fi_ops_av),
	.insert = fi_ibv_rdm_av_insert,
	.remove = fi_ibv_rdm_av_remove,
};

struct fi_ops_av *fi_ibv_rdm_set_av_ops(void)
{
	return &fi_ibv_rdm_av_ops;
}
