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
#include "shared/ofi_str.h"
#include "ofi_prov.h"
#include "ofi_perf.h"
#include "ofi_hmem.h"
#include "rdma/fi_ext.h"
#include "linkx.h"

static void lnx_free_peer(struct lnx_peer *lp)
{
	int i, j;

	for (i = 0; i < LNX_MAX_LOCAL_EPS; i++) {
		struct lnx_peer_prov *lpp = lp->lp_provs[i];
		if (!lpp)
			continue;
		for (j = 0; j < LNX_MAX_LOCAL_EPS; j++) {
			struct lnx_local2peer_map *lpm = lpp->lpp_map[j];
			if (!lpm)
				continue;
			free(lpm);
			lpp->lpp_map[j] = NULL;
		}
		free(lpp);
		lp->lp_provs[i] = NULL;
	}

	free(lp);
}

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
			tbl->lpt_count++;
			return i;
		}
	}

	return -FI_ENOENT;
}

static int lnx_peer_av_remove(struct lnx_peer *lp)
{
	int i;
	int rc, frc = 0;

	for (i = 0; i < LNX_MAX_LOCAL_EPS; i++) {
		struct lnx_peer_prov *lpp = lp->lp_provs[i];
		struct lnx_local2peer_map *lpm;
		int j;

		if (!lpp)
			continue;

		/* if this is a remote peer then we didn't insert its shm address
		 * into our local shm endpoint, so no need to remove it
		 */
		if (!strncasecmp(lpp->lpp_prov_name, "shm", 3) &&
			!lp->lp_local)
			continue;

		/* remove these address from all local providers */
		for (j = 0; j < LNX_MAX_LOCAL_EPS; j++) {
			lpm = lpp->lpp_map[j];

			if (!lpm)
				continue;

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
	int i;
	int rc, frc = 0;
	struct local_prov_ep *ep;

	for (i = 0; i < LNX_MAX_LOCAL_EPS; i++) {
		ep = prov->lpv_prov_eps[i];
		if (!ep)
			continue;
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
	struct lnx_peer_table *peer_tbl;

	peer_tbl = container_of(fid, struct lnx_peer_table, lpt_av.av_fid.fid);

	/* walk through the rest of the core providers and open their
	 * respective address vector tables
	 */
	dlist_foreach_container(&local_prov_table, struct local_prov,
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

static int lnx_get_or_create_peer_prov(struct lnx_peer *lp, char *prov_name,
									   struct lnx_peer_prov **lpp)
{
	int i;
	struct local_prov *entry;
	struct lnx_peer_prov *peer_prov;
	bool shm = false;

	/* shm provider should always be in index 0.
	 * We can't have more than one shm provider so if slot 0 is already
	 * busy, then fail.
	 */
	if (!strcmp(prov_name, "shm")) {
		if (lp->lp_provs[0])
			return -FI_ENOENT;
		shm = true;
	}

	/* check if we already have a peer provider */
	for (i = 0; i < LNX_MAX_LOCAL_EPS; i++) {
		peer_prov = lp->lp_provs[i];
		if (!peer_prov)
			continue;

		/* it's enough to check slot 0 if this is the shm provider */
		if (shm)
			break;

		if (!strncasecmp(peer_prov->lpp_prov_name, prov_name, FI_NAME_MAX)) {
			*lpp = peer_prov;
			return 0;
		}
	}

	/* reserve slot 0 for shm provider */
	i = 0;
	if (shm)
		goto insert_prov;

	for (i = 1; i < LNX_MAX_LOCAL_EPS; i++) {
		if (!lp->lp_provs[i])
			break;
	}

	if (i == LNX_MAX_LOCAL_EPS)
		return -FI_EPROTO;

insert_prov:
	dlist_foreach_container(&local_prov_table, struct local_prov,
							entry, lpv_entry) {
		if (!strncasecmp(entry->lpv_prov_name, prov_name, FI_NAME_MAX)) {
			peer_prov = calloc(sizeof(*peer_prov), 1);
			if (!peer_prov)
				return -FI_ENOMEM;

			strncpy(peer_prov->lpp_prov_name, prov_name, FI_NAME_MAX);

			peer_prov->lpp_prov = entry;

			lp->lp_provs[i] = peer_prov;

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

static int is_local_addr(struct lnx_addresses *la)
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

	/* Shared memory address not provided or not local*/
	if ((lap_shm->lap_addr_count == 0) ||
		strncasecmp(hostname, la->la_hostname, FI_NAME_MAX))
		return -FI_EOPNOTSUPP;

	/* badly formed address */
	if (shm_prov && (lap_shm->lap_addr_count > 1 ||
					 lap_shm->lap_addr_count < 0))
		return -FI_EPROTO;

	return 0;
}

static int lnx_peer_map_addrs(struct lnx_peer *lp, struct lnx_addresses *la,
							  uint64_t flags, void *context)
{
	int i, k, rc;
	struct lnx_peer_prov *lpp;
	struct lnx_address_prov *lap;

	lap = &la->la_addr_prov[0];

	for (i = 0; i < la->la_prov_count; i++) {
		int num_eps;

		if (lap->lap_addr_count > LNX_MAX_LOCAL_EPS)
			return -FI_EPROTO;

		rc = lnx_get_or_create_peer_prov(lp, lap->lap_prov, &lpp);
		if (rc)
			return rc;

		lpp->lpp_flags = flags;

		/* insert these addresses in all our local endpoints for this
		 * provider
		 */
		if (lp->lp_local)
			num_eps = 1;
		else
			num_eps = LNX_MAX_LOCAL_EPS;

		for (k = 0; k < num_eps; k++) {
			struct local_prov_ep *lpe;
			struct lnx_local2peer_map *lpm;

			lpe = lpp->lpp_prov->lpv_prov_eps[k];
			if (!lpe)
				continue;

			/* if this is a remote peer, don't insert the shm address
			 * since we will never talk to that peer over shm
			 */
			if (!strncasecmp(lpe->lpe_fabric_name, "shm", 3) &&
				!lp->lp_local)
				continue;

			lpm = calloc(sizeof(*lpm), 1);
			if (!lpm)
				return -FI_ENOMEM;

			lpp->lpp_map[k] = lpm;

			lpm->local_ep = lpe;
			lpm->addr_count = lap->lap_addr_count;
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
 * count: number of LINKx addresses
 * addr: an array of addresses
 * fi_addr: an out array of fi_addr)t
 *
 * Each LINKx address can have multiple core provider addresses
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
	struct lnx_peer *lp;
	struct lnx_peer_table *peer_tbl;
	struct lnx_addresses *addrs = (struct lnx_addresses *)addr;

	peer_tbl = container_of(av, struct lnx_peer_table, lpt_av.av_fid.fid);

	/* each entry represents a separate peer */
	struct lnx_addresses *la = addrs;
	for (i = 0; i < count; i++) {
		/* can't have more providers than LNX_MAX_LOCAL_EPS */
		if (la->la_prov_count >= LNX_MAX_LOCAL_EPS ||
			la->la_prov_count <= 0)
			return -FI_EPROTO;

		/* this is a local peer */
		lp = calloc(sizeof(*lp), 1);
		if (!lp)
			return -FI_ENOMEM;

		rc = is_local_addr(la);
		if (!rc) {
			lp->lp_local = true;
		} else if (rc == -FI_EOPNOTSUPP) {
			lp->lp_local = false;
		} else if (rc) {
			FI_INFO(&lnx_prov, FI_LOG_CORE, "failed to identify address\n");
			return rc;
		}

		rc = lnx_peer_map_addrs(lp, la, flags, context);
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

static int lnx_open_avs(struct local_prov *prov, struct fi_av_attr *attr,
						void *context)
{
	int i;
	int rc = 0;
	struct local_prov_ep *ep;

	for (i = 0; i < LNX_MAX_LOCAL_EPS; i++) {
		ep = prov->lpv_prov_eps[i];
		if (!ep)
			continue;
		rc = fi_av_open(ep->lpe_domain, attr,
					    &ep->lpe_av, context);
		if (rc)
			return rc;
	}

	return 0;
}

int lnx_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
				struct fid_av **av, void *context)
{
	struct util_domain *util_domain;
	struct lnx_peer_table *peer_tbl;
	struct local_prov *entry;
	int rc = 0;

	if (!attr)
		return -FI_EINVAL;

	if (attr->name)
		return -FI_ENOSYS;

	if (attr->type == FI_AV_UNSPEC)
		attr->type = FI_AV_TABLE;

	peer_tbl = calloc(sizeof(*peer_tbl), 1);
	if (!peer_tbl)
		return -FI_ENOMEM;

	peer_tbl->lpt_entries = calloc(sizeof(struct lnx_peer *) * attr->count, 1);
	if (!peer_tbl->lpt_entries) {
		rc = -FI_ENOMEM;
		goto failed;
	}

	util_domain = container_of(domain, struct util_domain, domain_fid.fid);

	rc = ofi_av_init_lightweight(util_domain, attr, &peer_tbl->lpt_av, context);
	if (rc) {
		FI_WARN(&lnx_prov, FI_LOG_CORE, "failed to initialize AV: %d\n", rc);
		goto failed;
	}

	peer_tbl->lpt_max_count = attr->count;
	peer_tbl->lpt_domain = util_domain;
	peer_tbl->lpt_av.av_fid.fid.ops = &lnx_av_fi_ops;
	peer_tbl->lpt_av.av_fid.ops = &lnx_av_ops;

	/* walk through the rest of the core providers and open their
	 * respective address vector tables
	 */
	dlist_foreach_container(&local_prov_table, struct local_prov,
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


