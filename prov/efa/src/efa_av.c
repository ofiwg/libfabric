/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright (c) 2016, Cisco Systems, Inc. All rights reserved. */
/* SPDX-FileCopyrightText: Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved. */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include <malloc.h>
#include <stdio.h>

#include <infiniband/efadv.h>
#include <ofi_enosys.h>

#include "efa.h"
#include "efa_av.h"
#include "rdm/efa_rdm_domain.h"
#include "rdm/efa_rdm_fabric.h"
#include "rdm/efa_rdm_ep.h"
#include "rdm/efa_rdm_pke_utils.h"


struct efa_av_entry *efa_av_addr_to_entry_impl(struct efa_av_array *entry_map,
					       fi_addr_t fi_addr)
{
	if (OFI_UNLIKELY(fi_addr == FI_ADDR_UNSPEC || fi_addr == FI_ADDR_NOTAVAIL))
		return NULL;
	return efa_av_array_at(entry_map, fi_addr);
}


/**
 * @brief find the efa_av_entry using fi_addr in the explicit AV
 */
struct efa_av_entry *efa_av_addr_to_entry(struct efa_av *av, fi_addr_t fi_addr)
{
	return efa_av_addr_to_entry_impl(av->addr_to_entry_map, fi_addr);
}


/**
 * @brief find fi_addr for an efa endpoint via the explicit cur reverse AV
 *
 * @param[in]	av	address vector
 * @param[in]	ahn	address handle number
 * @param[in]	qpn	QP number
 * @return	On success, return fi_addr to the peer who sent the packet.
 * 		If no such peer exists, return FI_ADDR_NOTAVAIL
 */
fi_addr_t efa_av_reverse_lookup(struct efa_av *av, uint16_t ahn, uint16_t qpn)
	OFI_TSA_NO_ANALYSIS // DGRAM uses FI_THREAD_DOMAIN, efa direct doesn't acquire the lock
{
	struct efa_cur_reverse_av *cur_entry;
	struct efa_cur_reverse_av_key cur_key;

	memset(&cur_key, 0, sizeof(cur_key));
	cur_key.ahn = ahn;
	cur_key.qpn = qpn;
	/* coverity[overflow_const : FALSE] - intentional unsigned wraparound in uthash Jenkins hash */
	HASH_FIND(hh, av->cur_reverse_av, &cur_key, sizeof(cur_key), cur_entry);

	return (OFI_LIKELY(!!cur_entry)) ? cur_entry->entry->fi_addr : FI_ADDR_NOTAVAIL;
}


int efa_av_is_valid_address(struct efa_ep_addr *addr)
{
	struct efa_ep_addr all_zeros = { 0 };

	return memcmp(addr->raw, all_zeros.raw, sizeof(addr->raw));
}


/*
 * @brief base reverse-AV add: add/replace the entry in cur_reverse_av
 *
 * @param[in,out]	cur_reverse_av	Reverse AV with AHN and QPN as key
 * @param[in]		entry		efa_av_entry object
 * @return		On success, return 0.
 * 			Otherwise, return a negative libfabric error code
 */
int efa_av_reverse_av_add(struct efa_cur_reverse_av **cur_reverse_av,
				 struct efa_av_entry *entry)
{
	struct efa_cur_reverse_av *cur_entry;
	struct efa_cur_reverse_av_key cur_key;

	memset(&cur_key, 0, sizeof(cur_key));
	cur_key.ahn = entry->ah->ahn;
	cur_key.qpn = efa_av_entry_ep_addr(entry)->qpn;
	cur_entry = NULL;

	/* coverity[overflow_const : FALSE] - intentional unsigned wraparound in uthash Jenkins hash */
	HASH_FIND(hh, *cur_reverse_av, &cur_key, sizeof(cur_key), cur_entry);
	if (!cur_entry) {
		cur_entry = malloc(sizeof(*cur_entry));
		if (!cur_entry) {
			EFA_WARN(FI_LOG_AV, "Cannot allocate memory for cur_reverse_av entry\n");
			return -FI_ENOMEM;
		}

		cur_entry->key.ahn = cur_key.ahn;
		cur_entry->key.qpn = cur_key.qpn;
		cur_entry->entry = entry;
		HASH_ADD(hh, *cur_reverse_av, key, sizeof(cur_key), cur_entry);

		return 0;
	}

	cur_entry->entry = entry;
	return 0;
}


/*
 * @brief base reverse-AV remove: drop the entry from cur_reverse_av if current
 *
 * @param[in,out]	cur_reverse_av	Reverse AV with AHN and QPN as key
 * @param[in]		entry		efa_av_entry object
 * @return		true if the entry was the current one for its (ahn, qpn)
 *			and was removed from cur_reverse_av; false otherwise.
 */
bool efa_av_reverse_av_remove(struct efa_cur_reverse_av **cur_reverse_av,
				    struct efa_av_entry *entry)
{
	struct efa_cur_reverse_av *cur_reverse_av_entry;
	struct efa_cur_reverse_av_key cur_key;

	memset(&cur_key, 0, sizeof(cur_key));
	cur_key.ahn = entry->ah->ahn;
	cur_key.qpn = efa_av_entry_ep_addr(entry)->qpn;
	/* coverity[overflow_const : FALSE] - intentional unsigned wraparound in uthash Jenkins hash */
	HASH_FIND(hh, *cur_reverse_av, &cur_key, sizeof(cur_key),
		  cur_reverse_av_entry);
	if (cur_reverse_av_entry && cur_reverse_av_entry->entry == entry) {
		HASH_DEL(*cur_reverse_av, cur_reverse_av_entry);
		free(cur_reverse_av_entry);
		return true;
	}
	return false;
}


/**
 * @brief remove an efa_av_entry from an addr map + util AV and clear its address
 *
 * Shared by the base explicit release path and the RDM implicit release paths
 * (each supplies its own addr map and util AV).
 */
void efa_av_entry_remove_from_util_av(struct efa_av_array *entry_map,
				      struct util_av *util_av,
				      struct efa_av_entry *entry,
				      fi_addr_t fi_addr)
{
	struct efa_ep_addr *ep_addr = efa_av_entry_ep_addr(entry);
	char gidstr[INET6_ADDRSTRLEN];
	int err;

	err = efa_av_array_insert(entry_map, fi_addr, NULL);
	if (err) {
		EFA_WARN(FI_LOG_AV, "Failed to remove entry for fi_addr %" PRIu64
			 " from array: %s\n", fi_addr, fi_strerror(-err));
	}

	err = ofi_av_remove_addr(util_av, fi_addr);
	if (err) {
		EFA_WARN(FI_LOG_AV, "ofi_av_remove_addr failed for fi_addr %" PRIu64
			 ": %s\n", fi_addr, fi_strerror(-err));
	}

	inet_ntop(AF_INET6, ep_addr->raw, gidstr, INET6_ADDRSTRLEN);
	EFA_INFO(FI_LOG_AV, "efa_av_entry released! entry[%p] GID[%s] QP[%u]\n",
		 entry, gidstr, ep_addr->qpn);

	memset(entry->ep_addr, 0, EFA_EP_ADDR_LEN);
}


/**
 * @brief create a base explicit AV entry: insert into the explicit util AV,
 * set the base fields (fi_addr, base AH) and register in the addr map.
 * Reverse-AV indexing and RDM-only state are layered on by the caller.
 * Caller must hold util_domain.lock and util_av.lock.
 */
struct efa_av_entry *efa_av_entry_alloc_explicit(struct efa_av *av,
						 struct efa_ep_addr *raw_addr,
						 fi_addr_t *fi_addr_out)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym)
{
	struct util_av *util_av = &av->util_av;
	struct util_av_entry *util_av_entry;
	struct efa_av_entry *efa_av_entry;
	fi_addr_t fi_addr;
	int err;

	err = ofi_av_insert_addr(util_av, raw_addr, &fi_addr);
	if (err) {
		EFA_WARN(FI_LOG_AV, "ofi_av_insert_addr failed! Error message: %s\n", fi_strerror(-err));
		return NULL;
	}

	util_av_entry = ofi_bufpool_get_ibuf(util_av->av_entry_pool, fi_addr);
	efa_av_entry = (struct efa_av_entry *)util_av_entry->data;
	assert(efa_is_same_addr(raw_addr, efa_av_entry_ep_addr(efa_av_entry)));
	assert(av->type == FI_AV_TABLE);
	efa_av_entry->fi_addr = fi_addr;

	efa_av_entry->ah = efa_ah_alloc(av->domain, raw_addr->raw, false, sizeof(struct efa_ah));
	if (!efa_av_entry->ah)
		goto err_remove_addr;

	err = efa_av_array_insert(av->addr_to_entry_map, fi_addr, efa_av_entry);
	if (err) {
		EFA_WARN(FI_LOG_AV, "Failed to insert entry for fi_addr %" PRIu64
			" into array: %s\n", fi_addr, fi_strerror(-err));
		goto err_release_ah;
	}

	*fi_addr_out = fi_addr;
	return efa_av_entry;

err_release_ah:
	efa_ah_release(av->domain, efa_av_entry->ah, false);
err_remove_addr:
	err = ofi_av_remove_addr(util_av, fi_addr);
	if (err)
		EFA_WARN(FI_LOG_AV, "While processing previous failure, ofi_av_remove_addr failed for fi_addr %" PRIu64
			": %s\n", fi_addr, fi_strerror(-err));
	return NULL;
}


/**
 * @brief release the base resources of an explicit AV entry
 *
 * Release the entry's base AH and remove it from the explicit addr map and
 * util AV (clearing its raw address). Reverse-AV removal and RDM-only teardown
 * are handled by the caller before calling this. Caller must hold
 * util_domain.lock and util_av.lock.
 */
void efa_av_entry_release_explicit(struct efa_av *av, struct efa_av_entry *entry,
				   fi_addr_t fi_addr)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym)
{
	efa_av_reverse_av_remove(&av->cur_reverse_av, entry);
	efa_ah_release(av->domain, entry->ah, false);
	efa_av_entry_remove_from_util_av(av->addr_to_entry_map, &av->util_av,
					 entry, fi_addr);
}


int efa_av_insert_one_validate(struct efa_ep_addr *addr, fi_addr_t *fi_addr,
			       char *raw_gid_str)
{
	if (!efa_av_is_valid_address(addr)) {
		EFA_WARN(FI_LOG_AV, "Failed to insert bad addr\n");
		*fi_addr = FI_ADDR_NOTAVAIL;
		return -FI_EADDRNOTAVAIL;
	}

	memset(raw_gid_str, 0, INET6_ADDRSTRLEN);
	if (!inet_ntop(AF_INET6, addr->raw, raw_gid_str, INET6_ADDRSTRLEN)) {
		EFA_WARN(FI_LOG_AV, "cannot convert address to string. errno: %d\n", errno);
		*fi_addr = FI_ADDR_NOTAVAIL;
		return -FI_EINVAL;
	}

	return 0;
}


/**
 * @brief insert one address into the efa-direct explicit address vector
 *
 * If the address already exists, return the existing fi_addr. Otherwise
 * allocate a new base explicit entry. The efa-direct path has no implicit AV
 * and no shm AV.
 */
static int efa_av_insert_one_explicit(struct efa_av *av, struct efa_ep_addr *addr,
				      fi_addr_t *fi_addr, uint64_t flags,
				      void *context)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym)
{
	char raw_gid_str[INET6_ADDRSTRLEN];
	struct efa_av_entry *entry;
	fi_addr_t efa_fiaddr;
	fi_addr_t new_fi_addr;
	int ret;

	ret = efa_av_insert_one_validate(addr, fi_addr, raw_gid_str);
	if (ret)
		return ret;

	if (flags & FI_SYNC_ERR)
		memset(context, 0, sizeof(int));

	EFA_INFO(FI_LOG_AV,
		 "Inserting address GID[%s] QP[%u] QKEY[%u] to explicit AV\n",
		 raw_gid_str, addr->qpn, addr->qkey);

	EFA_GENLOCK_LOCK(&av->util_av.lock, efa_util_av_lock_sym);

	efa_fiaddr = ofi_av_lookup_fi_addr_unsafe(&av->util_av, addr);
	if (efa_fiaddr != FI_ADDR_NOTAVAIL) {
		EFA_INFO(FI_LOG_AV,
			 "Found existing AV entry pointing to this "
			 "address! fi_addr: %" PRId64 "\n",
			 efa_fiaddr);
		*fi_addr = efa_fiaddr;
		EFA_GENLOCK_UNLOCK(&av->util_av.lock, efa_util_av_lock_sym);
		return 0;
	}

	entry = efa_av_entry_alloc_explicit(av, addr, &new_fi_addr);
	if (!entry) {
		*fi_addr = FI_ADDR_NOTAVAIL;
		EFA_GENLOCK_UNLOCK(&av->util_av.lock, efa_util_av_lock_sym);
		return -FI_EADDRNOTAVAIL;
	}

	if (efa_av_reverse_av_add(&av->cur_reverse_av, entry)) {
		efa_av_entry_release_explicit(av, entry, new_fi_addr);
		*fi_addr = FI_ADDR_NOTAVAIL;
		EFA_GENLOCK_UNLOCK(&av->util_av.lock, efa_util_av_lock_sym);
		return -FI_EADDRNOTAVAIL;
	}

	*fi_addr = new_fi_addr;
	EFA_GENLOCK_UNLOCK(&av->util_av.lock, efa_util_av_lock_sym);

	EFA_INFO(FI_LOG_AV,
		 "Successfully inserted address GID[%s] QP[%u] "
		 "QKEY[%u] to explicit AV. fi_addr: %" PRId64 "\n",
		 raw_gid_str, addr->qpn, addr->qkey, *fi_addr);

	return 0;
}


static int efa_av_insert(struct fid_av *av_fid, const void *addr,
			 size_t count, fi_addr_t *fi_addr,
			 uint64_t flags, void *context)
{
	struct efa_av *av = container_of(av_fid, struct efa_av, util_av.av_fid);
	int ret = 0, success_cnt = 0;
	size_t i = 0;
	struct efa_ep_addr *addr_i;
	fi_addr_t fi_addr_res;

	if (av->util_av.flags & FI_EVENT)
		return -FI_ENOEQ;

	if ((flags & FI_SYNC_ERR) && (!context || (flags & FI_EVENT)))
		return -FI_EINVAL;

	/*
	 * Providers are allowed to ignore FI_MORE.
	 */
	flags &= ~FI_MORE;
	if (flags)
		return -FI_ENOSYS;

	/*
	 * Acquire domain lock because AH is a domain-level resource whose fields
	 * are modified during av insert.
	 * The order in which the util domain and av locks are acquired must be
	 * util_domain.lock -> util_av.lock in the AV insertion and removal
	 * paths to prevent deadlocks */
	EFA_GENLOCK_LOCK(&av->domain->util_domain.lock, efa_util_domain_lock_sym);

	for (i = 0; i < count; i++) {
		addr_i = (struct efa_ep_addr *) ((uint8_t *)addr + i * EFA_EP_ADDR_LEN);

		ret = efa_av_insert_one_explicit(av, addr_i, &fi_addr_res, flags, context);
		if (ret) {
			EFA_WARN(FI_LOG_AV, "insert raw_addr to av failed! ret=%d\n",
				 ret);
			break;
		}

		if (fi_addr)
			fi_addr[i] = fi_addr_res;
		success_cnt++;
	}

	EFA_GENLOCK_UNLOCK(&av->domain->util_domain.lock, efa_util_domain_lock_sym);

	/* cancel remaining request and log to event queue */
	for (; i < count ; i++) {
		if (fi_addr)
			fi_addr[i] = FI_ADDR_NOTAVAIL;
	}

	return success_cnt;
}


int efa_av_lookup(struct fid_av *av_fid, fi_addr_t fi_addr,
		  void *addr, size_t *addrlen)
{
	struct efa_av *av = container_of(av_fid, struct efa_av, util_av.av_fid);
	struct efa_av_entry *entry = NULL;

	if (av->type != FI_AV_TABLE)
		return -FI_EINVAL;

	if (fi_addr == FI_ADDR_NOTAVAIL)
		return -FI_EINVAL;

	EFA_GENLOCK_LOCK(&av->util_av.lock, efa_util_av_lock_sym);
	entry = efa_av_addr_to_entry(av, fi_addr);
	if (!entry) {
		EFA_GENLOCK_UNLOCK(&av->util_av.lock, efa_util_av_lock_sym);
		return -FI_EINVAL;
	}

	memcpy(addr, (void *)efa_av_entry_ep_addr(entry), MIN(EFA_EP_ADDR_LEN, *addrlen));
	EFA_GENLOCK_UNLOCK(&av->util_av.lock, efa_util_av_lock_sym);
	if (*addrlen > EFA_EP_ADDR_LEN)
		*addrlen = EFA_EP_ADDR_LEN;
	return 0;
}


/*
 * @brief remove a set of addresses from AV and release its resources
 *
 * This function implements fi_av_remove() for EFA provider.
 *
 * Note that even after an address was removed from AV, it is still
 * possible to get TX and RX completion for the address. Per libfabric
 * standard, these completions should be ignored.
 *
 * To help TX completion handler to identify such a TX completion,
 * when removing an address, all its outstanding TX packet's addr
 * was set to FI_ADDR_NOTAVAIL. The TX completion handler will
 * ignore TX packet whose address is FI_ADDR_NOTAVAIL.
 *
 * Meanwhile, lower provider  will set a packet's address to
 * FI_ADDR_NOTAVAIL from it is from a removed address. RX completion
 * handler will ignore such packets.
 *
 * @param[in]	av_fid	fid of AV (address vector)
 * @param[in]	fi_addr pointer to an array of libfabric addresses
 * @param[in]	count	number of libfabric addresses in the array
 * @param[in]	flags	flags
 * @return	0 if all addresses have been removed successfully,
 * 		negative libfabric error code if error was encountered.
 */
static int efa_av_remove(struct fid_av *av_fid, fi_addr_t *fi_addr,
			 size_t count, uint64_t flags)
{
	int err = 0;
	size_t i;
	struct efa_av *av;
	struct efa_av_entry *entry;

	if (!fi_addr)
		return -FI_EINVAL;

	av = container_of(av_fid, struct efa_av, util_av.av_fid);
	if (av->type != FI_AV_TABLE)
		return -FI_EINVAL;

	/* The order in which the util domain and av locks are acquired must be
	 * util_domain.lock -> util_av.lock in the AV insertion and removal
	 * paths to prevent deadlocks */
	EFA_GENLOCK_LOCK(&av->domain->util_domain.lock, efa_util_domain_lock_sym);
	EFA_GENLOCK_LOCK(&av->util_av.lock, efa_util_av_lock_sym);
	for (i = 0; i < count; i++) {
		entry = efa_av_addr_to_entry(av, fi_addr[i]);
		if (!entry) {
			err = -FI_EINVAL;
			break;
		}

		efa_av_entry_release_explicit(av, entry, entry->fi_addr);
	}

	if (i < count) {
		/* something went wrong, so err cannot be zero */
		assert(err);
	}

	EFA_GENLOCK_UNLOCK(&av->util_av.lock, efa_util_av_lock_sym);
	EFA_GENLOCK_UNLOCK(&av->domain->util_domain.lock, efa_util_domain_lock_sym);
	return err;
}


const char *efa_av_straddr(struct fid_av *av_fid, const void *addr,
			   char *buf, size_t *len)
{
	return ofi_straddr(buf, len, FI_ADDR_EFA, addr);
}


static struct fi_ops_av efa_av_ops = {
	.size = sizeof(struct fi_ops_av),
	.insert = efa_av_insert,
	.insertsvc = fi_no_av_insertsvc,
	.insertsym = fi_no_av_insertsym,
	.remove = efa_av_remove,
	.lookup = efa_av_lookup,
	.straddr = efa_av_straddr,
	.lookup2 = ofi_av_lookup2,
};


/**
 * @brief per-entry callback for the base close path
 *
 * Release every live entry through the forward map. efa-direct does not
 * populate prv_reverse_av, so an entry displaced from cur_reverse_av by a
 * reused (ahn, qpn) would be missed if we iterated the reverse AV here and
 * would leak its AH.
 */
static int efa_av_destruct_release_entry(struct efa_av_array *arr, void *entry,
					 void *context)
	OFI_TSA_NO_ANALYSIS
{
	struct efa_av *av = context;
	struct efa_av_entry *av_entry = entry;
	fi_addr_t fi_addr = av_entry->fi_addr;

	efa_av_entry_release_explicit(av, av_entry, fi_addr);
	return 0;
}


/**
 * @brief release the base explicit state of an efa_av and free its containers
 *
 * Release every entry in the explicit AV (via the forward map), close the
 * explicit util AV, and destroy the explicit addr map. Used by both the
 * efa-direct and RDM close paths for the base teardown.
 */
static int efa_av_close(struct fid *fid)
	OFI_TSA_NO_ANALYSIS
{
	struct efa_av *av = container_of(fid, struct efa_av, util_av.av_fid.fid);
	int err;

	EFA_GENLOCK_LOCK(&av->domain->util_domain.lock, efa_util_domain_lock_sym);
	EFA_GENLOCK_LOCK(&av->util_av.lock, efa_util_av_lock_sym);
	/*
	 * Release every live entry through the forward map. efa-direct does not
	 * populate prv_reverse_av, so an entry displaced from cur_reverse_av by a
	 * reused (ahn, qpn) would be missed if we iterated the reverse AV here and
	 * would leak its AH.
	 */
	efa_av_array_iter(av->addr_to_entry_map, av,
			  efa_av_destruct_release_entry);
	EFA_GENLOCK_UNLOCK(&av->util_av.lock, efa_util_av_lock_sym);
	EFA_GENLOCK_UNLOCK(&av->domain->util_domain.lock, efa_util_domain_lock_sym);

	err = ofi_av_close(&av->util_av);
	if (OFI_UNLIKELY(err))
		EFA_WARN(FI_LOG_AV, "Failed to close util av: %s\n", fi_strerror(-err));

	efa_av_array_destroy(av->addr_to_entry_map);

	free(av);
	return 0;
}


static struct fi_ops efa_av_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = efa_av_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};


/**
 * @brief initialize the util_av field in efa_av
 *
 * @param[in]	efa_domain	efa_domain which owns the util_domain
 * @param[in]	attr		AV attr application passed to fi_av_open
 * @param[out]	util_av		util_av field to initialize
 * @param[in]	context		context application passed to fi_av_open
 * @param[in]	context_len	util_av entry context length (path dependent)
 * @return	On success, return 0.
 *		On failure, return a negative libfabric error code.
 */
int efa_av_init_util_av(struct efa_domain *efa_domain,
			struct fi_av_attr *attr,
			struct util_av *util_av,
			void *context,
			size_t context_len)
{
	struct util_av_attr util_attr;

	util_attr.addrlen = EFA_EP_ADDR_LEN;
	util_attr.context_len = context_len;
	util_attr.flags = 0;
	return ofi_av_init(&efa_domain->util_domain, attr, &util_attr,
			   util_av, context);
}


/**
 * @brief initialize the shared (base) fields of an efa_av
 *
 * Initialize the explicit forward AV array, the explicit util AV (sized by
 * the caller-supplied entry_size), and the owning domain and AV type. The
 * cur_reverse_av map starts empty (NULL) and is populated on insert. This is
 * the shared base of both the efa-direct and RDM open paths.
 *
 * @param[out]	av		efa address vector
 * @param[in]	efa_domain	owning domain
 * @param[in]	attr		AV attr application passed to fi_av_open
 * @param[in]	context		context application passed to fi_av_open
 * @param[in]	entry_size	util_av entry context length (path dependent)
 * @return	On success, return 0. On failure, a negative libfabric error code.
 */
int efa_av_init_base(struct efa_av *av, struct efa_domain *efa_domain,
		     struct fi_av_attr *attr, void *context, size_t entry_size)
{
	int ret;

	ret = efa_av_array_init(&av->addr_to_entry_map);
	if (ret)
		return ret;

	ret = efa_av_init_util_av(efa_domain, attr, &av->util_av, context, entry_size);
	if (ret) {
		efa_av_array_destroy(av->addr_to_entry_map);
		return ret;
	}

	av->domain = efa_domain;
	av->type = attr->type;
	return 0;
}


int efa_av_open_prepare_attr(struct fid_domain *domain_fid,
			     struct fi_av_attr *attr,
			     struct efa_domain **efa_domain_out)
{
	struct efa_domain *efa_domain;
	size_t universe_size;

	if (!attr)
		return -FI_EINVAL;

	if (attr->name)
		return -FI_ENOSYS;

	/* FI_EVENT, FI_READ, and FI_SYMMETRIC are not supported */
	if (attr->flags)
		return -FI_ENOSYS;

	/*
	 * TODO: remove me once EFA RDM endpoint supports resizing members tied to the AV
	 * size.
	 */
	if (!attr->count)
		attr->count = EFA_MIN_AV_SIZE;
	else
		attr->count = MAX(attr->count, EFA_MIN_AV_SIZE);

	if (attr->type == FI_AV_MAP) {
		EFA_INFO(FI_LOG_AV, "FI_AV_MAP is deprecated in Libfabric 2.x. Please use FI_AV_TABLE. "
					"EFA provider will now switch to using FI_AV_TABLE.\n");
	}
	attr->type = FI_AV_TABLE;

	efa_domain = container_of(domain_fid, struct efa_domain, util_domain.domain_fid);

	if (fi_param_get_size_t(NULL, "universe_size",
				&universe_size) == FI_SUCCESS)
		attr->count = MAX(attr->count, universe_size);

	*efa_domain_out = efa_domain;
	return 0;
}


int efa_av_open(struct fid_domain *domain_fid, struct fi_av_attr *attr,
		struct fid_av **av_fid, void *context)
{
	struct efa_domain *efa_domain;
	struct efa_av *av;
	int ret;

	ret = efa_av_open_prepare_attr(domain_fid, attr, &efa_domain);
	if (ret)
		return ret;

	av = calloc(1, sizeof(*av));
	if (!av)
		return -FI_ENOMEM;

	ret = efa_av_init_base(av, efa_domain, attr, context,
			       sizeof(struct efa_av_entry) - EFA_EP_ADDR_LEN);
	if (ret)
		goto err_free;

	EFA_INFO(FI_LOG_AV, "fi_av_attr:%" PRId64 "\n", attr->flags);

	*av_fid = &av->util_av.av_fid;
	(*av_fid)->fid.fclass = FI_CLASS_AV;
	(*av_fid)->fid.context = context;
	(*av_fid)->fid.ops = &efa_av_fi_ops;
	(*av_fid)->ops = &efa_av_ops;

	return 0;

err_free:
	free(av);
	return ret;
}
