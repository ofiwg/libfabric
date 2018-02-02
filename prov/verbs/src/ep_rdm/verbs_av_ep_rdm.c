/*
 * Copyright (c) 2013-2016 Intel Corporation, Inc.  All rights reserved.
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
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <ctype.h>

#include "ofi.h"

#include "verbs_rdm.h"


extern struct fi_provider fi_ibv_prov;

ssize_t
fi_ibv_rdm_start_connection(struct fi_ibv_rdm_ep *ep, 
			    struct fi_ibv_rdm_conn *conn)
{
	struct rdma_cm_id *id = NULL;
	assert(ep->domain->rdm_cm->listener);

	if (conn->state != FI_VERBS_CONN_ALLOCATED)
		return FI_SUCCESS;

	if (ep->is_closing) {
		VERBS_INFO(FI_LOG_AV,
			   "Attempt start connection with %s:%u when ep is closing\n",
			   inet_ntoa(conn->addr.sin_addr),
			   ntohs(conn->addr.sin_port));
		return -FI_EOTHER;
	}

	conn->state = FI_VERBS_CONN_STARTED;
	fi_ibv_rdm_conn_init_cm_role(conn, ep);

	if (rdma_create_id(ep->domain->rdm_cm->ec, &id, conn, RDMA_PS_TCP)) {
		VERBS_INFO_ERRNO(FI_LOG_AV, "rdma_create_id\n", errno);
		return -errno;
	}

	if (conn->cm_role == FI_VERBS_CM_ACTIVE || 
	    conn->cm_role == FI_VERBS_CM_SELF)
		conn->id[0] = id;

	if (rdma_resolve_addr(id, NULL, (struct sockaddr *)&conn->addr, 30000)) {
		VERBS_INFO_ERRNO(FI_LOG_AV, "rdma_resolve_addr\n", errno);
		return -errno;
	}
	return FI_SUCCESS;
}

/* Must call with `rdm_cm::cm_lock` held */
ssize_t fi_ibv_rdm_start_overall_disconnection(struct fi_ibv_rdm_av_entry *av_entry)
{
	struct fi_ibv_rdm_conn *conn = NULL, *tmp = NULL;
	ssize_t ret = FI_SUCCESS;
	ssize_t err = FI_SUCCESS;

	HASH_ITER(hh, av_entry->conn_hash, conn, tmp) {
		ret = fi_ibv_rdm_start_disconnection(conn);
		if (ret) {
			VERBS_INFO(FI_LOG_AV, "Disconnection failed "
				   "(%zd) for %p\n", ret, conn);
			err = ret;
		}
		/*
		 * do NOT remove entry of connection from HASH.
		 * We will refer to the connection during
		 * cleanup of the connections.
		 */
	}

	return err;
}

ssize_t fi_ibv_rdm_start_disconnection(struct fi_ibv_rdm_conn *conn)
{
	ssize_t ret = FI_SUCCESS;

	VERBS_INFO(FI_LOG_AV, "Closing connection %p, state %d\n",
		   conn, conn->state);

	if (conn->id[0]) {
		if (rdma_disconnect(conn->id[0])) {
			VERBS_INFO_ERRNO(FI_LOG_AV, "rdma_disconnect\n", errno);
			ret = -errno;
		}
	}

	switch (conn->state) {
	case FI_VERBS_CONN_ESTABLISHED:
		conn->state = FI_VERBS_CONN_LOCAL_DISCONNECT;
		break;
	case FI_VERBS_CONN_REJECTED:
		conn->state = FI_VERBS_CONN_CLOSED;
		break;
	case FI_VERBS_CONN_ALLOCATED:
	case FI_VERBS_CONN_CLOSED:
		break;
	default:
		VERBS_WARN(FI_LOG_EP_CTRL, "Unknown connection state: %d\n",
			  (int)conn->state);
		ret = -FI_EOTHER;
	}

	return ret;
}

static inline int fi_ibv_rdm_av_is_valid_address(struct sockaddr_in *addr)
{
	return addr->sin_family == AF_INET ? 1 : 0;
}

int fi_ibv_av_entry_alloc(struct fi_ibv_domain *domain,
			  struct fi_ibv_rdm_av_entry **av_entry,
			  void *addr)
{
	int ret = ofi_memalign((void**)av_entry,
			       FI_IBV_MEM_ALIGNMENT,
			       sizeof (**av_entry));
	if (ret)
		return -ret;
	memset((*av_entry), 0, sizeof(**av_entry));
	memcpy(&(*av_entry)->addr, addr, FI_IBV_RDM_DFLT_ADDRLEN);
	HASH_ADD(hh, domain->rdm_cm->av_hash, addr,
		 FI_IBV_RDM_DFLT_ADDRLEN, (*av_entry));
	(*av_entry)->sends_outgoing = 0;
	(*av_entry)->recv_preposted = 0;

	return ret;
}

static int fi_ibv_rdm_av_insert(struct fid_av *av_fid, const void *addr,
                                size_t count, fi_addr_t * fi_addr,
                                uint64_t flags, void *context)
{
	struct fi_ibv_av *av = container_of(av_fid, struct fi_ibv_av, av_fid);
	size_t i;
	int failed = 0;
	int ret = 0;
	int *fi_errors = context;

	if((av->flags & FI_EVENT) && !av->eq)
		return -FI_ENOEQ;

	if ((flags & FI_SYNC_ERR) && ((!context) || (flags & FI_EVENT)))
		return -FI_EINVAL;
	else if (flags & FI_SYNC_ERR)
		memset(context, 0, sizeof(int) * count);

	pthread_mutex_lock(&av->domain->rdm_cm->cm_lock);

	if (av->used + count > av->count) {
		const size_t new_av_count = av->used + count;
		if (av->type == FI_AV_TABLE) {
			void *p = realloc(av->domain->rdm_cm->av_table,
					  (new_av_count *
					  sizeof(*av->domain->rdm_cm->av_table)));
			if (p) {
				av->domain->rdm_cm->av_table = p;
			}
			else {
				ret = -FI_ENOMEM;
				goto out;
			}
		}
		av->count = new_av_count;
	}

	for (i = 0; i < count; i++) {
		struct fi_ibv_rdm_av_entry *av_entry = NULL;
		void *addr_i = (uint8_t *) addr +
			i * FI_IBV_RDM_DFLT_ADDRLEN;

		if (flags & FI_SYNC_ERR)
			fi_errors[i] = FI_SUCCESS;

		if (!fi_ibv_rdm_av_is_valid_address(addr_i)) {
			if (fi_addr)
				fi_addr[i] = FI_ADDR_NOTAVAIL;

			VERBS_INFO(FI_LOG_AV,
				   "fi_av_insert: bad addr #%zu\n", i);

			if (av->flags & FI_EVENT) {
				/* due to limited functionality of
				 * verbs EQ notify last failed element
				 * only. */
				/* TODO: what about utils EQ? */
				struct fi_eq_err_entry err = {
					.fid = &av->av_fid.fid,
					.context = context,
					.data = i,
					.err = FI_EINVAL,
					.prov_errno = FI_EINVAL
				};
				av->eq->err = err;
			} else if (flags & FI_SYNC_ERR) {
				fi_errors[i] = -FI_EADDRNOTAVAIL;
			}

			failed++;
			continue;
		}

		HASH_FIND(hh, av->domain->rdm_cm->av_hash, addr_i,
			  FI_IBV_RDM_DFLT_ADDRLEN, av_entry);

		if (!av_entry) {
			/* If addr_i is not found in HASH then we malloc it.
			 * It could be found if the connection was initiated
			 * by the remote side.
			 */
			ret = fi_ibv_av_entry_alloc(av->domain, &av_entry, addr_i);
			if (ret)
				goto out;
		}

		switch (av->type) {
		case FI_AV_MAP:
			if (fi_addr)
				fi_addr[i] = (uintptr_t) (void *) av_entry;
			break;
		case FI_AV_TABLE:
			if (fi_addr)
				fi_addr[i] = av->used;
			av->domain->rdm_cm->av_table[av->used] = av_entry;
			break;
		default:
			assert(0);
			break;
		}

		VERBS_INFO(FI_LOG_AV,
			   "fi_av_insert: addr %s:%u; av_entry - %p\n",
			   inet_ntoa(av_entry->addr.sin_addr),
			   ntohs(av_entry->addr.sin_port), av_entry);

		av->used++;
	}
	ret = count - failed;

	if (av->flags & FI_EVENT) {
		struct fi_eq_entry entry = {
			.fid = &av->av_fid.fid,
			.context = context,
			.data = count - failed
		};
		fi_ibv_eq_write_event(
			av->eq, FI_AV_COMPLETE, &entry, sizeof(entry));
	}

out:
	pthread_mutex_unlock(&av->domain->rdm_cm->cm_lock);
	return (av->flags & FI_EVENT) ? FI_SUCCESS : ret;
}

static int fi_ibv_rdm_av_insertsvc(struct fid_av *av_fid, const char *node,
				   const char *service, fi_addr_t *fi_addr,
				   uint64_t flags, void *context)
{
	struct addrinfo addrinfo_hints;
	struct addrinfo *result = NULL;
	int ret;

	if (!node || !service) {
		VERBS_WARN(FI_LOG_AV, "fi_av_insertsvc: %s provided\n",
			   (!node ? (!service ? "node and service weren't" :
						"node wasn't") :
				    ("service wasn't")));
		return -FI_EINVAL;
	}

	struct fi_ibv_av *av = container_of(av_fid, struct fi_ibv_av, av_fid);

	memset(&addrinfo_hints, 0, sizeof(addrinfo_hints));
	addrinfo_hints.ai_family = AF_INET;
	ret = getaddrinfo(node, service, &addrinfo_hints, &result);
	if (ret) {
		if ((av->flags & FI_EVENT) && (av->eq)) {
			struct fi_eq_entry entry = {
				.fid = &av->av_fid.fid,
				.context = context,
				.data = 0
			};
			struct fi_eq_err_entry err = {
				.fid = &av->av_fid.fid,
				.context = context,
				.data = 0,
				.err = FI_EINVAL,
				.prov_errno = FI_EINVAL
			};
			av->eq->err = err;

			fi_ibv_eq_write_event(
				av->eq, FI_AV_COMPLETE,
				&entry, sizeof(entry));
		}
		return -ret;
	}

	ret = fi_ibv_rdm_av_insert(av_fid, (struct sockaddr_in *)result->ai_addr,
				   1, fi_addr, flags, context);
	freeaddrinfo(result);
	return ret;
}

static int fi_ibv_rdm_av_insertsym(struct fid_av *av, const char *node,
				   size_t nodecnt, const char *service,
				   size_t svccnt, fi_addr_t *fi_addr,
				   uint64_t flags, void *context)
{
	int ret = 0, success = 0, err_code = 0;
	int var_port, var_host, len_port, len_host;
	char base_host[FI_NAME_MAX] = {0};
	char tmp_host[FI_NAME_MAX] = {0};
	char tmp_port[FI_NAME_MAX] = {0};
	int hostlen, offset = 0, fmt;
	size_t i, j;

	if (!node || !service || node[0] == '\0') {
		VERBS_WARN(FI_LOG_AV, "fi_av_insertsym: %s provided\n",
			   (!service ? (!node ? "node and service weren't" :
						"service wasn't") :
				    ("node wasn't")));
		return -FI_EINVAL;
	}

	hostlen = strlen(node);
	while (isdigit(*(node + hostlen - (offset + 1))))
		offset++;

	if (*(node + hostlen - offset) == '.')
		fmt = 0;
	else
		fmt = offset;

	assert((hostlen-offset) < FI_NAME_MAX);
	strncpy(base_host, node, hostlen - (offset));
	var_port = atoi(service);
	var_host = atoi(node + hostlen - offset);

	for (i = 0; i < nodecnt; i++) {
		for (j = 0; j < svccnt; j++) {
			int check_host = 0, check_port = 0;

			len_host = snprintf(tmp_host, FI_NAME_MAX, "%s%0*d",
					    base_host, fmt,
					    var_host + (int)i);
			len_port = snprintf(tmp_port, FI_NAME_MAX,  "%d",
					    var_port + (int)j);

			check_host = (len_host > 0 && len_host < FI_NAME_MAX);
			check_port = (len_port > 0 && len_port < FI_NAME_MAX);

			if (check_port && check_host) {
				ret = fi_ibv_rdm_av_insertsvc(av, tmp_host,
							      tmp_port, fi_addr,
							      flags, context);
				if (ret == 1)
					success++;
				else
					err_code = ret;
			} else {
				VERBS_WARN(FI_LOG_AV,
					   "fi_av_insertsym: %s is invalid\n",
					   (!check_port ?
					    (!check_host ?
					     "node and service weren't" :
					     "service wasn't") :
					    ("node wasn't")));
				err_code = FI_ETOOSMALL;
			}
		}
	}
	return ((success > 0) ? success : err_code);
}

static int fi_ibv_rdm_av_lookup(struct fid_av *av_fid, fi_addr_t fi_addr,
				void *addr, size_t *addrlen)
{
	struct fi_ibv_av *av = container_of(av_fid, struct fi_ibv_av, av_fid);
	struct fi_ibv_rdm_av_entry *av_entry = NULL;

	if (fi_addr == FI_ADDR_NOTAVAIL)
		return -FI_EINVAL;

	if (av->type == FI_AV_MAP)
		av_entry = (struct fi_ibv_rdm_av_entry *) fi_addr;
	else /* (av->type == FI_AV_TABLE) */
		av_entry = av->domain->rdm_cm->av_table[fi_addr];

	memcpy(addr, &av_entry->addr, MIN(*addrlen, sizeof(av_entry->addr)));
	*addrlen = sizeof(av_entry->addr);

	return 0;
}

static int fi_ibv_rdm_av_remove(struct fid_av *av_fid, fi_addr_t *fi_addr,
                                size_t count, uint64_t flags)
{
	struct fi_ibv_av *av = container_of(av_fid, struct fi_ibv_av, av_fid);
	struct fi_ibv_rdm_av_entry *av_entry = NULL;
	int ret = FI_SUCCESS;
	int err = FI_SUCCESS;
	size_t i;

	if(av->flags & FI_EVENT && !av->eq)
		return -FI_ENOEQ;

	if (!fi_addr || (av->type != FI_AV_MAP && av->type != FI_AV_TABLE))
		return -FI_EINVAL;

	pthread_mutex_lock(&av->domain->rdm_cm->cm_lock);
	for (i = 0; i < count; i++) {

		if (fi_addr[i] == FI_ADDR_NOTAVAIL)
			continue;

		if (av->type == FI_AV_MAP)
			av_entry = (struct fi_ibv_rdm_av_entry *)fi_addr[i];
		else /* (av->type == FI_AV_TABLE) */
			av_entry = av->domain->rdm_cm->av_table[fi_addr[i]];

		VERBS_INFO(FI_LOG_AV, "av_remove conn - %p; addr %s:%u\n",
			   av_entry, inet_ntoa(av_entry->addr.sin_addr),
			   ntohs(av_entry->addr.sin_port));

		err = fi_ibv_rdm_start_overall_disconnection(av_entry);
		ret = (ret == FI_SUCCESS) ? err : ret;
		/* do not destroy connection here because we may
		 * get WC for this connection. just move connection
		 * to list of av-removed objects to clean later */

		/* TODO: add cleaning into av_insert */
		HASH_DEL(av->domain->rdm_cm->av_hash, av_entry);
		slist_insert_tail(&av_entry->removed_next,
				  &av->domain->rdm_cm->av_removed_entry_head);
	}
	pthread_mutex_unlock(&av->domain->rdm_cm->cm_lock);
	return ret;
}

static const char *fi_ibv_rdm_av_straddr(struct fid_av *av, const void *addr,
					 char *buf, size_t *len)
{
	return ofi_straddr(buf, len, FI_SOCKADDR, addr);
}

static struct fi_ops_av fi_ibv_rdm_av_ops = {
	.size = sizeof(struct fi_ops_av),
	.insert = fi_ibv_rdm_av_insert,
	.insertsvc = fi_ibv_rdm_av_insertsvc,
	.insertsym = fi_ibv_rdm_av_insertsym,
	.remove = fi_ibv_rdm_av_remove,
	.lookup = fi_ibv_rdm_av_lookup,
	.straddr = fi_ibv_rdm_av_straddr,
};

struct fi_ops_av *fi_ibv_rdm_set_av_ops(void)
{
	return &fi_ibv_rdm_av_ops;
}

static int fi_ibv_rdm_av_close(fid_t fid)
{
	struct fi_ibv_av *av = container_of(fid, struct fi_ibv_av, av_fid.fid);
	free(av);
	return 0;
}

static struct fi_ops fi_ibv_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_rdm_av_close,
	.bind = fi_no_bind,
};

static inline struct fi_ibv_rdm_av_entry *
fi_ibv_rdm_av_tbl_idx_to_av_entry(struct fi_ibv_rdm_ep *ep, fi_addr_t addr)
{
	return (addr == FI_ADDR_UNSPEC) ? NULL :
		ep->domain->rdm_cm->av_table[addr];
}

static inline struct fi_ibv_rdm_av_entry *
fi_ibv_rdm_av_map_addr_to_av_entry(struct fi_ibv_rdm_ep *ep, fi_addr_t addr)
{
	return (struct fi_ibv_rdm_av_entry *)
		(addr == FI_ADDR_UNSPEC ? NULL : (void *)(uintptr_t)addr);
}

static inline fi_addr_t
fi_ibv_rdm_av_entry_to_av_tbl_idx(struct fi_ibv_rdm_ep *ep,
				  struct fi_ibv_rdm_av_entry *av_entry)
{
	size_t i;

	for (i = 0; i < ep->av->used; i++) {
		if (ep->domain->rdm_cm->av_table[i] == av_entry) {
			return i;
		}
	}

	return FI_ADDR_UNSPEC;
}

static inline fi_addr_t
fi_ibv_rdm_av_entry_to_av_map_addr(struct fi_ibv_rdm_ep *ep,
				   struct fi_ibv_rdm_av_entry *av_entry)
{
	return (av_entry == NULL) ? FI_ADDR_UNSPEC :
	       (fi_addr_t)(uintptr_t)av_entry;
}

static inline fi_addr_t
fi_ibv_rdm_conn_to_av_tbl_idx(struct fi_ibv_rdm_ep *ep,
			      struct fi_ibv_rdm_conn *conn)
{
	if (conn == NULL)
		return FI_ADDR_UNSPEC;
	return fi_ibv_rdm_av_entry_to_av_tbl_idx(ep, conn->av_entry);
}

static inline fi_addr_t
fi_ibv_rdm_conn_to_av_map_addr(struct fi_ibv_rdm_ep *ep,
			       struct fi_ibv_rdm_conn *conn)
{
    return fi_ibv_rdm_av_entry_to_av_map_addr(ep, conn->av_entry);
}

/* Must call with `rdm_cm::cm_lock` held */
static inline struct fi_ibv_rdm_conn *
fi_ibv_rdm_conn_entry_alloc(struct fi_ibv_rdm_av_entry *av_entry,
			    struct fi_ibv_rdm_ep *ep)
{
	struct fi_ibv_rdm_conn *conn;

	if (ofi_memalign((void**) &conn,
			 FI_IBV_MEM_ALIGNMENT,
			 sizeof(*conn)))
		return NULL;
	memset(conn, 0, sizeof(*conn));
	memcpy(&conn->addr, &av_entry->addr,
	       FI_IBV_RDM_DFLT_ADDRLEN);
	conn->ep = ep;
	conn->av_entry = av_entry;
	conn->state = FI_VERBS_CONN_ALLOCATED;
	dlist_init(&conn->postponed_requests_head);
	HASH_ADD(hh, av_entry->conn_hash, ep,
		 sizeof(struct fi_ibv_rdm_ep *), conn);

	/* Initiates connection to the peer */
	fi_ibv_rdm_start_connection(ep, conn);

	return conn;
}

static inline struct fi_ibv_rdm_conn *
fi_ibv_rdm_av_map_addr_to_conn_add_new_conn(struct fi_ibv_rdm_ep *ep,
					    fi_addr_t addr)
{
	struct fi_ibv_rdm_av_entry *av_entry =
		fi_ibv_rdm_av_map_addr_to_av_entry(ep, addr);
	if (av_entry) {
		struct fi_ibv_rdm_conn *conn;
		pthread_mutex_lock(&ep->domain->rdm_cm->cm_lock);
		HASH_FIND(hh, av_entry->conn_hash,
			  &ep, sizeof(struct fi_ibv_rdm_ep *), conn);
		if (OFI_UNLIKELY(!conn))
			conn = fi_ibv_rdm_conn_entry_alloc(av_entry, ep);
		pthread_mutex_unlock(&ep->domain->rdm_cm->cm_lock);
		return conn;
	}

	return NULL;
}

static inline struct fi_ibv_rdm_conn *
fi_ibv_rdm_av_tbl_idx_to_conn_add_new_conn(struct fi_ibv_rdm_ep *ep,
					   fi_addr_t addr)
{
	struct fi_ibv_rdm_av_entry *av_entry =
		fi_ibv_rdm_av_tbl_idx_to_av_entry(ep, addr);
	if (av_entry) {
		struct fi_ibv_rdm_conn *conn;
		pthread_mutex_lock(&ep->domain->rdm_cm->cm_lock);
		HASH_FIND(hh, av_entry->conn_hash,
			  &ep, sizeof(struct fi_ibv_rdm_ep *), conn);
		if (OFI_UNLIKELY(!conn))
			conn = fi_ibv_rdm_conn_entry_alloc(av_entry, ep);
		pthread_mutex_unlock(&ep->domain->rdm_cm->cm_lock);
		return conn;
	}

	return NULL;
}

int fi_ibv_rdm_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
			struct fid_av **av_fid, void *context)
{
	struct fi_ibv_domain *fid_domain;
	struct fi_ibv_av *av;
	size_t count = 64;

	fid_domain = container_of(domain, struct fi_ibv_domain, util_domain.domain_fid);

	if (!attr)
		return -FI_EINVAL;

	if (attr->name) {
		VERBS_WARN(FI_LOG_AV,
			   "Shared AV is not implemented\n");
		return -FI_ENOSYS;
	}

	switch (attr->type) {
	case FI_AV_UNSPEC:
		attr->type = FI_AV_MAP;
	case FI_AV_MAP:
	case FI_AV_TABLE:
		break;
	default:
		return -EINVAL;
	}

	if (attr->count)
		count = attr->count;

	av = calloc(1, sizeof *av);
	if (!av)
		return -ENOMEM;

	assert(fid_domain->ep_type == FI_EP_RDM);
	av->domain = fid_domain;
	av->type = attr->type;
	av->count = count;
	av->flags = attr->flags;
	av->used = 0;

	if (av->type == FI_AV_TABLE && av->count > 0) {
		av->domain->rdm_cm->av_table =
			calloc(av->count,
			       sizeof(*av->domain->rdm_cm->av_table));
		if (!av->domain->rdm_cm->av_table) {
			free(av);
			return -ENOMEM;
		}
	}

	if (av->type == FI_AV_MAP) {
		av->addr_to_av_entry = fi_ibv_rdm_av_map_addr_to_av_entry;
		av->av_entry_to_addr = fi_ibv_rdm_av_entry_to_av_map_addr;
		av->addr_to_conn = fi_ibv_rdm_av_map_addr_to_conn_add_new_conn;
		av->conn_to_addr = fi_ibv_rdm_conn_to_av_map_addr;
	} else /* if (av->type == FI_AV_TABLE) */ {
		av->addr_to_av_entry = fi_ibv_rdm_av_tbl_idx_to_av_entry;
		av->av_entry_to_addr = fi_ibv_rdm_av_entry_to_av_tbl_idx;
		av->addr_to_conn = fi_ibv_rdm_av_tbl_idx_to_conn_add_new_conn;
		av->conn_to_addr = fi_ibv_rdm_conn_to_av_tbl_idx;
	}

	av->av_fid.fid.fclass = FI_CLASS_AV;
	av->av_fid.fid.context = context;
	av->av_fid.fid.ops = &fi_ibv_fi_ops;

	av->av_fid.ops = fi_ibv_rdm_set_av_ops();

	*av_fid = &av->av_fid;
	return 0;
}
