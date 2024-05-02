/*
 * Copyright (c) 2022 ORNL. All rights reserved.
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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <rdma/fi_errno.h>
#include "ofi_util.h"
#include "ofi.h"
#include "ofi_str.h"
#include "ofi_prov.h"
#include "ofi_perf.h"
#include "ofi_hmem.h"
#include "rdma/fi_ext.h"
#include "lnx.h"

static void lnx_free_peer(struct lnx_peer *lp)
{
	struct lnx_peer_prov *lpp;
	struct dlist_entry *tmp, *tmp2;
	struct lnx_local2peer_map *lpm;

	dlist_foreach_container_safe(&lp->lp_provs,
		struct lnx_peer_prov, lpp, entry, tmp) {
		dlist_foreach_container_safe(&lpp->lpp_map,
			struct lnx_local2peer_map, lpm, entry, tmp2) {
			dlist_remove(&lpm->entry);
			free(lpm);
		}
		dlist_remove(&lpp->entry);
		free(lpp);
	}

	free(lp);
}

#if ENABLE_DEBUG
static void lnx_print_peer(int idx, struct lnx_peer *lp)
{
	int k;
	struct lnx_peer_prov *lpp;
	struct lnx_local2peer_map *lpm;

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "%d: lnx_peer[%d] is %s\n", getpid(), idx,
	       (lp->lp_local) ? "local" : "remote");
	dlist_foreach_container(&lp->lp_provs,
			struct lnx_peer_prov, lpp, entry) {
		FI_DBG(&lnx_prov, FI_LOG_CORE,
		       "%d: peer[%p] provider %s\n", getpid(), lpp,
		       lpp->lpp_prov_name);
		dlist_foreach_container(&lpp->lpp_map,
			struct lnx_local2peer_map, lpm, entry) {
			FI_DBG(&lnx_prov, FI_LOG_CORE,
			       "   %d: peer has %d mapped addrs\n",
			       getpid(), lpm->addr_count);
			for (k = 0; k < lpm->addr_count; k++)
				FI_DBG(&lnx_prov, FI_LOG_CORE,
				       "        %d: addr = %lu\n",
				       getpid(), lpm->peer_addrs[k]);
		}
	}
}
#endif /* ENABLE_DEBUG */

static int lnx_peer_insert(struct lnx_peer_table *tbl,
			   struct lnx_peer *lp)
{
	int i;

	if (tbl->lpt_max_count == 0 ||
	    tbl->lpt_count >= tbl->lpt_max_count)
		return -FI_ENOENT;

	for (i = 0; i < tbl->lpt_max_count; i++) {
		if (!tbl->lpt_entries[i]) {
			tbl->lpt_entries[i] = lp;
#if ENABLE_DEBUG
			lnx_print_peer(i, lp);
#endif
			tbl->lpt_count++;
			return i;
		}
	}

	return -FI_ENOENT;
}

static int lnx_peer_av_remove(struct lnx_peer *lp)
{
	int rc, frc = 0;
	struct lnx_peer_prov *lpp;
	struct lnx_local2peer_map *lpm;

	dlist_foreach_container(&lp->lp_provs,
			struct lnx_peer_prov, lpp, entry) {
		/* if this is a remote peer then we didn't insert its shm address
		 * into our local shm endpoint, so no need to remove it
		 */
		if (!strncasecmp(lpp->lpp_prov_name, "shm", 3) &&
			!lp->lp_local)
			continue;

		/* remove these address from all local providers */
		dlist_foreach_container(&lpp->lpp_map,
			struct lnx_local2peer_map, lpm, entry) {
			if (lpm->addr_count > 0) {
				rc = fi_av_remove(lpm->local_ep->lpe_av, lpm->peer_addrs,
						  lpm->addr_count, lpp->lpp_flags);
				if (rc)
					frc = rc;
			}
		}
	}

	return frc;
}

static int lnx_peer_remove(struct lnx_peer_table *tbl, int idx)
{
	struct lnx_peer *lp = tbl->lpt_entries[idx];
	int rc = 0;

	if (!lp)
		return 0;

	rc = lnx_peer_av_remove(lp);

	tbl->lpt_entries[idx] = NULL;
	tbl->lpt_count--;

	return rc;
}

static int lnx_cleanup_avs(struct local_prov *prov)
{
	int rc, frc = 0;
	struct local_prov_ep *ep;

	dlist_foreach_container(&prov->lpv_prov_eps,
		struct local_prov_ep, ep, entry) {
		rc = fi_close(&ep->lpe_av->fid);
		if (rc)
			frc = rc;
	}

	return frc;
}

static inline void lnx_free_peer_tbl(struct lnx_peer_table *peer_tbl)
{
	free(peer_tbl->lpt_entries);
	free(peer_tbl);
}

int lnx_av_close(struct fid *fid)
{
	int rc;
	struct local_prov *entry;
	struct lnx_fabric *fabric;
	struct lnx_peer_table *peer_tbl;

	peer_tbl = container_of(fid, struct lnx_peer_table, lpt_av.av_fid.fid);
	fabric = peer_tbl->lpt_domain->ld_fabric;

	/* walk through the rest of the core providers and open their
	 * respective address vector tables
	 */
	dlist_foreach_container(&fabric->local_prov_table, struct local_prov,
				entry, lpv_entry) {
		rc = lnx_cleanup_avs(entry);
		if (rc) {
			FI_INFO(&lnx_prov, FI_LOG_CORE, "Failed to close av for %s\n",
					entry->lpv_prov_name);
		}
	}

	ofi_av_close_lightweight(&peer_tbl->lpt_av);

	free(peer_tbl);

	return 0;
}

static struct fi_ops lnx_av_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = lnx_av_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static int lnx_get_or_create_peer_prov(struct dlist_entry *prov_table,
				       struct lnx_peer *lp, char *prov_name,
				       struct lnx_peer_prov **lpp)
{
	bool shm = false;
	struct local_prov *entry;
	struct lnx_peer_prov *peer_prov;

	if (!strcmp(prov_name, "shm")) {
		if (lp->lp_shm_prov)
			return -FI_ENOENT;
		shm = true;
		goto insert_prov;
	}

	/* check if we already have a peer provider */
	dlist_foreach_container(&lp->lp_provs,
			struct lnx_peer_prov, peer_prov, entry) {
		if (!strncasecmp(peer_prov->lpp_prov_name, prov_name, FI_NAME_MAX)) {
			*lpp = peer_prov;
			return 0;
		}
	}

insert_prov:
	dlist_foreach_container(prov_table, struct local_prov,
				entry, lpv_entry) {
		if (!strncasecmp(entry->lpv_prov_name, prov_name, FI_NAME_MAX)) {
			peer_prov = calloc(sizeof(*peer_prov), 1);
			if (!peer_prov)
				return -FI_ENOMEM;

			dlist_init(&peer_prov->entry);
			dlist_init(&peer_prov->lpp_map);

			strncpy(peer_prov->lpp_prov_name, prov_name, FI_NAME_MAX);

			peer_prov->lpp_prov = entry;

			if (shm)
				lp->lp_shm_prov = peer_prov;
			else
				dlist_insert_tail(&peer_prov->entry, &lp->lp_provs);

			*lpp = peer_prov;
			return 0;
		}
	}

	return -FI_ENOENT;
}

static inline struct lnx_address_prov *
next_prov(struct lnx_address_prov *prov)
{
	uint8_t *ptr;

	ptr = (uint8_t*) prov;

	ptr += (sizeof(*prov) + (prov->lap_addr_count * prov->lap_addr_size));

	return (struct lnx_address_prov*)ptr;
}

static inline size_t
get_lnx_addresses_size(struct lnx_addresses *addrs)
{
	int i;
	size_t s = sizeof(*addrs);
	struct lnx_address_prov *prov;

	prov = addrs->la_addr_prov;
	for (i = 0; i < addrs->la_prov_count; i++) {
		s += sizeof(*prov) + (prov->lap_addr_count * prov->lap_addr_size);
		prov = next_prov(prov);
	}

	return s;
}

static inline struct lnx_addresses *
next_peer(struct lnx_addresses *addrs)
{
	uint8_t *ptr;

	ptr = (uint8_t*)addrs + get_lnx_addresses_size(addrs);

	return (struct lnx_addresses *)ptr;
}

static struct lnx_address_prov *
lnx_get_peer_shm_addr(struct lnx_addresses *addrs)
{
	int i;
	struct lnx_address_prov *prov;

	prov = addrs->la_addr_prov;
	for (i = 0; i < addrs->la_prov_count; i++) {
		if (!strcmp(prov->lap_prov, "shm"))
			return prov;
		prov = next_prov(prov);
	}

	return NULL;
}

static int is_local_addr(struct local_prov **shm_prov, struct lnx_addresses *la)
{
	int rc;
	char hostname[FI_NAME_MAX];
	struct lnx_address_prov *lap_shm;

	/* check the hostname and compare it to mine
	 * TODO: Is this good enough? or do we need a better way of
	 * determining if the address is local?
	 */
	rc = gethostname(hostname, FI_NAME_MAX);
	if (rc == -1) {
		FI_INFO(&lnx_prov, FI_LOG_CORE, "failed to get hostname\n");
		return -FI_EPERM;
	}

	lap_shm = lnx_get_peer_shm_addr(la);
	if (!lap_shm)
		return -FI_EOPNOTSUPP;

	/* Shared memory address not provided or not local*/
	if ((lap_shm->lap_addr_count == 0) ||
		strncasecmp(hostname, la->la_hostname, FI_NAME_MAX))
		return -FI_EOPNOTSUPP;

	/* badly formed address */
	if (*shm_prov && (lap_shm->lap_addr_count > 1 ||
			  lap_shm->lap_addr_count < 0))
		return -FI_EPROTO;

	return 0;
}

static void
lnx_update_msg_entries(struct lnx_qpair *qp,
		       fi_addr_t (*get_addr)(struct fi_peer_rx_entry *))
{
	struct lnx_queue *q = &qp->lqp_unexq;
	struct lnx_rx_entry *rx_entry;
	struct dlist_entry *item;

	ofi_spin_lock(&q->lq_qlock);
	dlist_foreach(&q->lq_queue, item) {
		rx_entry = (struct lnx_rx_entry *) item;
		if (rx_entry->rx_entry.addr == FI_ADDR_UNSPEC)
			rx_entry->rx_entry.addr = get_addr(&rx_entry->rx_entry);
	}
	ofi_spin_unlock(&q->lq_qlock);
}

void
lnx_foreach_unspec_addr(struct fid_peer_srx *srx,
			fi_addr_t (*get_addr)(struct fi_peer_rx_entry *))
{
	struct lnx_srx_context *ctxt;

	ctxt = (struct lnx_srx_context *) srx->ep_fid.fid.context;

	lnx_update_msg_entries(&ctxt->srx_lep->le_srq.lps_trecv, get_addr);
	lnx_update_msg_entries(&ctxt->srx_lep->le_srq.lps_recv, get_addr);
}

static int lnx_peer_map_addrs(struct dlist_entry *prov_table,
			      struct lnx_peer *lp, struct lnx_addresses *la,
			      uint64_t flags, void *context)
{
	int i, j, rc;
	struct lnx_peer_prov *lpp;
	struct lnx_address_prov *lap;
	struct local_prov_ep *lpe;
	struct dlist_entry *eps;

	lap = &la->la_addr_prov[0];

	for (i = 0; i < la->la_prov_count; i++) {
		if (lap->lap_addr_count > LNX_MAX_LOCAL_EPS)
			return -FI_EPROTO;

		rc = lnx_get_or_create_peer_prov(prov_table, lp, lap->lap_prov,
						 &lpp);
		if (rc)
			return rc;

		lpp->lpp_flags = flags;

		eps = &lpp->lpp_prov->lpv_prov_eps;
		dlist_foreach_container(eps, struct local_prov_ep, lpe,
					entry) {
			struct lnx_local2peer_map *lpm;

			/* if this is a remote peer, don't insert the shm address
			 * since we will never talk to that peer over shm
			 */
			if (!strncasecmp(lpe->lpe_fabric_name, "shm", 3) &&
				!lp->lp_local)
				continue;

			lpm = calloc(sizeof(*lpm), 1);
			if (!lpm)
				return -FI_ENOMEM;

			dlist_init(&lpm->entry);
			dlist_insert_tail(&lpm->entry, &lpp->lpp_map);

			lpm->local_ep = lpe;
			lpm->addr_count = lap->lap_addr_count;
			for (j = 0; j < LNX_MAX_LOCAL_EPS; j++)
				lpm->peer_addrs[j] = FI_ADDR_NOTAVAIL;
			/* fi_av_insert returns the number of addresses inserted */
			rc = fi_av_insert(lpe->lpe_av, (void*)lap->lap_addrs,
					  lap->lap_addr_count,
					  lpm->peer_addrs, flags, context);
			if (rc < 0)
				return rc;

			/* should only insert the number of addresses indicated */
			assert(rc == lap->lap_addr_count);
		}

		lap = next_prov(lap);
	}

	return 0;
}

/*
 * count: number of LNX addresses
 * addr: an array of addresses
 * fi_addr: an out array of fi_addr)t
 *
 * Each LNX address can have multiple core provider addresses
 * Check the hostname provided in each address to see if it's the same as
 * me. If so, then we'll use the SHM address if available.
 *
 * ASSUMPTION: fi_av_insert() is called exactly once per peer.
 * We're not handling multiple av_inserts on the same peer. If that
 * happens then we will create multiple peers entries.
 */
int lnx_av_insert(struct fid_av *av, const void *addr, size_t count,
		  fi_addr_t *fi_addr, uint64_t flags, void *context)
{
	int i, rc, idx;
	int disable_shm = 0;
	struct lnx_peer *lp;
	struct dlist_entry *prov_table;
	struct lnx_peer_table *peer_tbl;
	struct lnx_addresses *la = (struct lnx_addresses *)addr;

	fi_param_get_bool(&lnx_prov, "disable_shm", &disable_shm);

	peer_tbl = container_of(av, struct lnx_peer_table, lpt_av.av_fid.fid);
	prov_table = &peer_tbl->lpt_domain->ld_fabric->local_prov_table;

	/* each entry represents a separate peer */
	for (i = 0; i < count; i++) {
		/* can't have more providers than LNX_MAX_LOCAL_EPS */
		if (la->la_prov_count >= LNX_MAX_LOCAL_EPS ||
			la->la_prov_count <= 0)
			return -FI_EPROTO;

		/* this is a local peer */
		lp = calloc(sizeof(*lp), 1);
		if (!lp)
			return -FI_ENOMEM;

		dlist_init(&lp->lp_provs);

		rc = is_local_addr(&peer_tbl->lpt_domain->ld_fabric->shm_prov,
				   la);
		if (!rc) {
			lp->lp_local = !disable_shm;
		} else if (rc == -FI_EOPNOTSUPP) {
			lp->lp_local = false;
		} else if (rc) {
			FI_INFO(&lnx_prov, FI_LOG_CORE, "failed to identify address\n");
			return rc;
		}

		rc = lnx_peer_map_addrs(prov_table, lp, la, flags, context);
		if (rc) {
			free(lp);
			return rc;
		}

		idx = lnx_peer_insert(peer_tbl, lp);
		if (idx == -1) {
			rc = lnx_peer_av_remove(lp);
			lnx_free_peer(lp);
			FI_INFO(&lnx_prov, FI_LOG_CORE,
					"Peer table size exceeded. Removed = %d\n", rc);
			return -FI_ENOENT;
		}

		fi_addr[i] = (fi_addr_t) idx;

		la = next_peer(la);
	}

	return i;
}

int lnx_av_remove(struct fid_av *av, fi_addr_t *fi_addr, size_t count,
		  uint64_t flags)
{
	struct lnx_peer_table *peer_tbl;
	int frc = 0, rc, i;

	peer_tbl = container_of(av, struct lnx_peer_table, lpt_av.av_fid.fid);

	for (i = 0; i < count; i++) {
		rc = lnx_peer_remove(peer_tbl, (int)fi_addr[i]);
		if (rc)
			frc = rc;
	}

	return frc;
}

static const char *
lnx_av_straddr(struct fid_av *av, const void *addr,
	       char *buf, size_t *len)
{
	/* TODO: implement */
	return NULL;
}

static int
lnx_av_lookup(struct fid_av *av, fi_addr_t fi_addr, void *addr,
	      size_t *addrlen)
{
	/* TODO: implement */
	return -FI_EOPNOTSUPP;
}

static struct fi_ops_av lnx_av_ops = {
	.size = sizeof(struct fi_ops_av),
	.insert = lnx_av_insert,
	.remove = lnx_av_remove,
	.insertsvc = fi_no_av_insertsvc,
	.insertsym = fi_no_av_insertsym,
	.lookup = lnx_av_lookup,
	.straddr = lnx_av_straddr,
};

static void lnx_get_core_av_attr(struct local_prov_ep *ep,
				 struct fi_av_attr *attr)
{
	memset(attr, 0, sizeof(*attr));
	attr->type = ep->lpe_fi_info->domain_attr->av_type;
}

static int lnx_open_avs(struct local_prov *prov, struct fi_av_attr *attr,
			void *context)
{
	int rc = 0;
	struct local_prov_ep *ep;
	struct fi_av_attr core_attr;

	dlist_foreach_container(&prov->lpv_prov_eps,
		struct local_prov_ep, ep, entry) {
		lnx_get_core_av_attr(ep, &core_attr);
		if (ep->lpe_local)
			core_attr.count = ep->lpe_fi_info->domain_attr->ep_cnt;
		else
			core_attr.count = attr->count;
		rc = fi_av_open(ep->lpe_domain, &core_attr,
					    &ep->lpe_av, context);
		if (rc)
			return rc;
	}

	return 0;
}

int lnx_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		struct fid_av **av, void *context)
{
	struct lnx_fabric *fabric;
	struct lnx_domain *lnx_domain;
	struct lnx_peer_table *peer_tbl;
	struct local_prov *entry;
	size_t table_sz = LNX_DEF_AV_SIZE;
	int rc = 0;

	if (!attr)
		return -FI_EINVAL;

	if (attr->name)
		return -FI_ENOSYS;

	if (attr->type != FI_AV_UNSPEC &&
	    attr->type != FI_AV_TABLE)
		return -FI_ENOSYS;

	if (attr->type == FI_AV_UNSPEC)
		attr->type = FI_AV_TABLE;

	peer_tbl = calloc(sizeof(*peer_tbl), 1);
	if (!peer_tbl)
		return -FI_ENOMEM;

	if (attr->count != 0)
		table_sz = attr->count;

	peer_tbl->lpt_entries =
	  calloc(sizeof(struct lnx_peer *) * table_sz, 1);
	if (!peer_tbl->lpt_entries) {
		rc = -FI_ENOMEM;
		goto failed;
	}

	lnx_domain = container_of(domain, struct lnx_domain,
				  ld_domain.domain_fid.fid);
	fabric = lnx_domain->ld_fabric;

	rc = ofi_av_init_lightweight(&lnx_domain->ld_domain, attr,
				     &peer_tbl->lpt_av, context);
	if (rc) {
		FI_WARN(&lnx_prov, FI_LOG_CORE,
			"failed to initialize AV: %d\n", rc);
		goto failed;
	}

	peer_tbl->lpt_max_count = table_sz;
	peer_tbl->lpt_domain = lnx_domain;
	peer_tbl->lpt_av.av_fid.fid.ops = &lnx_av_fi_ops;
	peer_tbl->lpt_av.av_fid.ops = &lnx_av_ops;

	assert(fabric->lnx_peer_tbl == NULL);

	/* need this to handle memory registration vi fi_mr_regattr(). We need
	 * to be able to access the peer table to determine which endpoint
	 * we'll be using based on the source/destination address */
	fabric->lnx_peer_tbl = peer_tbl;

	/* walk through the rest of the core providers and open their
	 * respective address vector tables
	 */
	dlist_foreach_container(&fabric->local_prov_table, struct local_prov,
				entry, lpv_entry) {
		rc = lnx_open_avs(entry, attr, context);
		if (rc) {
			FI_INFO(&lnx_prov, FI_LOG_CORE, "Failed to initialize domain for %s\n",
					entry->lpv_prov_name);
			goto close;
		}
	}

	*av = &peer_tbl->lpt_av.av_fid;

	return 0;

close:
	ofi_av_close_lightweight(&peer_tbl->lpt_av);
failed:
	lnx_free_peer_tbl(peer_tbl);
	return rc;
}


