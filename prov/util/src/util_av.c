/*
 * Copyright (c) 2015-2016 Intel Corporation. All rights reserved.
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

#include <arpa/inet.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netdb.h>
#include <netinet/in.h>

#if HAVE_GETIFADDRS
#include <net/if.h>
#include <ifaddrs.h>
#endif

#include <fi_util.h>


enum {
	UTIL_NO_ENTRY = -1,
	UTIL_DEFAULT_AV_SIZE = 1024,
};


static int fi_get_src_sockaddr(const struct sockaddr *dest_addr, size_t dest_addrlen,
			       struct sockaddr **src_addr, size_t *src_addrlen)
{
	socklen_t len; /* needed for OS compatability */
	int sock, ret;

	sock = socket(dest_addr->sa_family, SOCK_DGRAM, 0);
	if (sock < 0)
		return -errno;

	ret = connect(sock, dest_addr, dest_addrlen);
	if (ret)
		goto out;

	*src_addr = calloc(dest_addrlen, 1);
	if (!*src_addr) {
		ret = -FI_ENOMEM;
		goto out;
	}

	len = (socklen_t) dest_addrlen;
	ret = getsockname(sock, *src_addr, &len);
	if (ret) {
		ret = -errno;
		goto out;
	}
	*src_addrlen = len;

	switch ((*src_addr)->sa_family) {
	case AF_INET:
		((struct sockaddr_in *) (*src_addr))->sin_port = 0;
		break;
	case AF_INET6:
		((struct sockaddr_in6 *) (*src_addr))->sin6_port = 0;
		break;
	default:
		ret = -FI_ENOSYS;
		break;
	}

out:
	close(sock);
	return ret;

}

void ofi_getnodename(char *buf, int buflen)
{
	int ret;
	struct addrinfo ai, *rai = NULL;
	struct ifaddrs *ifaddrs, *ifa;

	ret = gethostname(buf, buflen);
	if (ret == 0) {
		memset(&ai, 0, sizeof(ai));
		ai.ai_family = AF_INET;
		ret = getaddrinfo(buf, NULL, &ai, &rai);
		if (!ret) {
			freeaddrinfo(rai);
			return;
		}
	}

#if HAVE_GETIFADDRS
	ret = getifaddrs(&ifaddrs);
	if (!ret) {
		for (ifa = ifaddrs; ifa != NULL; ifa = ifa->ifa_next) {
			if (ifa->ifa_addr == NULL || !(ifa->ifa_flags & IFF_UP) ||
			     (ifa->ifa_addr->sa_family != AF_INET))
				continue;

			ret = getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in),
				  	  buf, buflen, NULL, 0, NI_NUMERICHOST);
			if (ret == 0) {
				freeifaddrs(ifaddrs);
				return;
			}
		}
		freeifaddrs(ifaddrs);
	}
#endif
	/* no reasonable address found, try loopback */
	strncpy(buf, "127.0.0.1", buflen);
}

int ofi_get_src_addr(uint32_t addr_format,
		    const void *dest_addr, size_t dest_addrlen,
		    void **src_addr, size_t *src_addrlen)
{
	switch (addr_format) {
	case FI_SOCKADDR:
	case FI_SOCKADDR_IN:
	case FI_SOCKADDR_IN6:
		return fi_get_src_sockaddr(dest_addr, dest_addrlen,
					   (struct sockaddr **) src_addr,
					   src_addrlen);
	default:
		return -FI_ENOSYS;
	}
}

static int fi_get_sockaddr(int sa_family, uint64_t flags,
			   const char *node, const char *service,
			   struct sockaddr **addr, size_t *addrlen)
{
	struct addrinfo hints, *ai;
	int ret;

	memset(&hints, 0, sizeof hints);
	hints.ai_family = sa_family;
	hints.ai_socktype = SOCK_STREAM;
	if (flags & FI_SOURCE)
		hints.ai_flags = AI_PASSIVE;

	ret = getaddrinfo(node, service, &hints, &ai);
	if (ret)
		return -FI_ENODATA;

	*addr = mem_dup(ai->ai_addr, ai->ai_addrlen);
	if (!*addr) {
		ret = -FI_ENOMEM;
		goto out;
	}

	*addrlen = ai->ai_addrlen;
out:
	freeaddrinfo(ai);
	return ret;
}

int ofi_get_addr(uint32_t addr_format, uint64_t flags,
		const char *node, const char *service,
		void **addr, size_t *addrlen)
{
	switch (addr_format) {
	case FI_SOCKADDR:
		return fi_get_sockaddr(0, flags, node, service,
				       (struct sockaddr **) addr, addrlen);
	case FI_SOCKADDR_IN:
		return fi_get_sockaddr(AF_INET, flags, node, service,
				       (struct sockaddr **) addr, addrlen);
	case FI_SOCKADDR_IN6:
		return fi_get_sockaddr(AF_INET6, flags, node, service,
				       (struct sockaddr **) addr, addrlen);
	default:
		return -FI_ENOSYS;
	}
}

static void *util_av_get_data(struct util_av *av, int index)
{
	return (char *) av->data + (index * av->addrlen);
}

void *ofi_av_get_addr(struct util_av *av, int index)
{
	return util_av_get_data(av, index);
}

static void util_av_set_data(struct util_av *av, int index,
			     const void *data, size_t len)
{
	memcpy(util_av_get_data(av, index), data, len);
}

static int fi_verify_av_insert(struct util_av *av, uint64_t flags)
{
	if ((av->flags & FI_EVENT) && !av->eq) {
		FI_WARN(av->prov, FI_LOG_AV, "no EQ bound to AV\n");
		return -FI_ENOEQ;
	}

	if (flags & ~(FI_MORE)) {
		FI_WARN(av->prov, FI_LOG_AV, "unsupported flags\n");
		return -FI_ENOEQ;
	}

	return 0;
}

/*
 * Must hold AV lock
 */
static int util_av_hash_insert(struct util_av_hash *hash, int slot, int index)
{
	int entry, i;

	if (slot < 0 || slot >= hash->slots)
		return -FI_EINVAL;

	if (hash->table[slot].index == UTIL_NO_ENTRY) {
		hash->table[slot].index = index;
		return 0;
	}

	if (hash->free_list == UTIL_NO_ENTRY)
		return -FI_ENOSPC;

	entry = hash->free_list;
	hash->free_list = hash->table[hash->free_list].next;

	for (i = slot; hash->table[i].next != UTIL_NO_ENTRY; )
		i = hash->table[i].next;

	hash->table[i].next = entry;
	hash->table[entry].index = index;
	hash->table[entry].next = UTIL_NO_ENTRY;
	return 0;
}

int ofi_av_insert_addr(struct util_av *av, const void *addr, int slot, int *index)
{
	int ret = 0;

	fastlock_acquire(&av->lock);
	if (av->free_list == UTIL_NO_ENTRY) {
		FI_WARN(av->prov, FI_LOG_AV, "AV is full\n");
		ret = -FI_ENOSPC;
		goto out;
	}

	if (av->flags & FI_SOURCE) {
		ret = util_av_hash_insert(&av->hash, slot, av->free_list);
		if (ret) {
			FI_WARN(av->prov, FI_LOG_AV,
				"failed to insert addr into hash table\n");
			goto out;
		}
	}

	*index = av->free_list;
	av->free_list = *(int *) util_av_get_data(av, av->free_list);
	util_av_set_data(av, *index, addr, av->addrlen);
out:
	fastlock_release(&av->lock);
	return ret;
}

/*
 * Must hold AV lock
 */
static void util_av_hash_remove(struct util_av_hash *hash, int slot, int index)
{
	int i;

	if (slot < 0 || slot >= hash->slots)
		return;

	if (slot == index) {
		if (hash->table[slot].next == UTIL_NO_ENTRY) {
			hash->table[slot].index = UTIL_NO_ENTRY;
			return;
		} else {
			index = hash->table[slot].next;
			hash->table[slot] = hash->table[index];
		}
	} else {
		for (i = slot; hash->table[i].next != index; )
			i = hash->table[i].next;

		hash->table[i].next = hash->table[index].next;
	}
	hash->table[index].next = hash->free_list;
	hash->free_list = index;
}

static int fi_av_remove_addr(struct util_av *av, int slot, int index)
{
	int *entry, *next, i;

	if (index < 0 || index > av->count) {
		FI_WARN(av->prov, FI_LOG_AV, "index out of range\n");
		return -FI_EINVAL;
	}

	fastlock_acquire(&av->lock);
	if (av->flags & FI_SOURCE)
		util_av_hash_remove(&av->hash, slot, index);

	entry = util_av_get_data(av, index);
	if (av->free_list == UTIL_NO_ENTRY || index < av->free_list) {
		*entry = av->free_list;
		av->free_list = index;
	} else {
		i = av->free_list;
		for (next = util_av_get_data(av, i); index > *next;) {
			i = *next;
			next = util_av_get_data(av, i);
		}
		util_av_set_data(av, index, next, sizeof index);
		*next = index;
	}

	fastlock_release(&av->lock);
	return 0;
}

int ofi_av_lookup_index(struct util_av *av, const void *addr, int slot)
{
	int i, ret = -FI_ENODATA;

	if (slot < 0 || slot >= av->hash.slots) {
		FI_WARN(av->prov, FI_LOG_AV, "invalid slot (%d)\n", slot);
		return -FI_EINVAL;
	}

	fastlock_acquire(&av->lock);
	if (av->hash.table[slot].index == UTIL_NO_ENTRY) {
		FI_DBG(av->prov, FI_LOG_AV, "no entry at slot (%d)\n", slot);
		goto out;
	}

	for (i = slot; i != UTIL_NO_ENTRY; i = av->hash.table[i].next) {
		if (!memcmp(ofi_av_get_addr(av, av->hash.table[i].index), addr,
			    av->addrlen)) {
			ret = av->hash.table[i].index;
			FI_DBG(av->prov, FI_LOG_AV, "entry at index (%d)\n", ret);
			break;
		}
	}
out:
	FI_DBG(av->prov, FI_LOG_AV, "%d\n", ret);
	fastlock_release(&av->lock);
	return ret;
}

int ofi_av_bind(struct fid *av_fid, struct fid *eq_fid, uint64_t flags)
{
	struct util_av *av;
	struct util_eq *eq;

	av = container_of(av_fid, struct util_av, av_fid.fid);
	if (eq_fid->fclass != FI_CLASS_EQ) {
		FI_WARN(av->prov, FI_LOG_AV, "invalid fid class\n");
		return -FI_EINVAL;
	}

	if (flags) {
		FI_WARN(av->prov, FI_LOG_AV, "invalid flags\n");
		return -FI_EINVAL;
	}

	eq = container_of(eq_fid, struct util_eq, eq_fid.fid);
	av->eq = eq;
	atomic_inc(&eq->ref);
	return 0;
}

int ofi_av_close(struct util_av *av)
{
	if (atomic_get(&av->ref)) {
		FI_WARN(av->prov, FI_LOG_AV, "AV is busy\n");
		return -FI_EBUSY;
	}

	if (av->eq)
		atomic_dec(&av->eq->ref);

	atomic_dec(&av->domain->ref);
	fastlock_destroy(&av->lock);
	/* TODO: unmap data? */
	free(av->data);
	return 0;
}

static void util_av_hash_init(struct util_av_hash *hash)
{
	int i;

	for (i = 0; i < hash->slots; i++) {
		hash->table[i].index = UTIL_NO_ENTRY;
		hash->table[i].next = UTIL_NO_ENTRY;
	}

	hash->free_list = hash->slots;
	for (i = hash->slots; i < hash->total_count; i++) {
		hash->table[i].index = UTIL_NO_ENTRY;
		hash->table[i].next = i + 1;
	}
	hash->table[hash->total_count - 1].next = UTIL_NO_ENTRY;
}

static int util_av_init(struct util_av *av, const struct fi_av_attr *attr,
			const struct util_av_attr *util_attr)
{
	int *entry, i, ret = 0;

	atomic_initialize(&av->ref, 0);
	fastlock_init(&av->lock);
	av->count = attr->count ? attr->count : UTIL_DEFAULT_AV_SIZE;
	av->count = roundup_power_of_two(av->count);
	av->addrlen = util_attr->addrlen;
	av->flags = util_attr->flags | attr->flags;

	FI_INFO(av->prov, FI_LOG_AV, "AV size %zu\n", av->count);

	/* TODO: Handle FI_READ */
	/* TODO: Handle mmap - shared AV */

	if (util_attr->flags & FI_SOURCE) {
		av->hash.slots = av->count;
		av->hash.total_count = av->count + util_attr->overhead;
		FI_INFO(av->prov, FI_LOG_AV,
		       "FI_SOURCE requested, hash size %zu\n", av->hash.total_count);
	}

	av->data = malloc((av->count * util_attr->addrlen) +
			  (av->hash.total_count * sizeof(*av->hash.table)));
	if (!av->data)
		return -FI_ENOMEM;

	for (i = 0; i < av->count - 1; i++) {
		entry = util_av_get_data(av, i);
		*entry = i + 1;
	}
	entry = util_av_get_data(av, av->count - 1);
	*entry = UTIL_NO_ENTRY;

	if (util_attr->flags & FI_SOURCE) {
		av->hash.table = util_av_get_data(av, av->count);
		util_av_hash_init(&av->hash);
	}

	return ret;
}

static int util_verify_av_attr(struct util_domain *domain,
			       const struct fi_av_attr *attr,
			       const struct util_av_attr *util_attr)
{
	switch (attr->type) {
	case FI_AV_MAP:
	case FI_AV_TABLE:
		if ((domain->av_type != FI_AV_UNSPEC) &&
		    (attr->type != domain->av_type)) {
			FI_INFO(domain->prov, FI_LOG_AV, "Invalid AV type\n");
		   	return -FI_EINVAL;
		}
		break;
	default:
		FI_WARN(domain->prov, FI_LOG_AV, "invalid av type\n");
		return -FI_EINVAL;
	}

	if (attr->flags & ~(FI_EVENT | FI_READ | FI_SYMMETRIC)) {
		FI_WARN(domain->prov, FI_LOG_AV, "invalid flags\n");
		return -FI_EINVAL;
	}

	if (util_attr->flags & ~(FI_SOURCE)) {
		FI_WARN(domain->prov, FI_LOG_AV, "invalid internal flags\n");
		return -FI_EINVAL;
	}

	if (util_attr->addrlen < sizeof(int)) {
		FI_WARN(domain->prov, FI_LOG_AV, "unsupported address size\n");
		return -FI_ENOSYS;
	}

	return 0;
}

int ofi_av_init(struct util_domain *domain, const struct fi_av_attr *attr,
	       const struct util_av_attr *util_attr,
	       struct util_av *av, void *context)
{
	int ret;

	ret = util_verify_av_attr(domain, attr, util_attr);
	if (ret)
		return ret;

	av->prov = domain->prov;
	ret = util_av_init(av, attr, util_attr);
	if (ret)
		return ret;

	av->av_fid.fid.fclass = FI_CLASS_AV;
	/*
	 * ops set by provider
	 * av->av_fid.fid.ops = &prov_av_fi_ops;
	 * av->av_fid.ops = &prov_av_ops;
	 */
	av->context = context;
	av->domain = domain;
	atomic_inc(&domain->ref);
	return 0;
}


/*************************************************************************
 *
 * AV for IP addressing
 *
 *************************************************************************/

static int ip_av_slot(struct util_av *av, const struct sockaddr *sa)
{
	uint32_t host;
	uint16_t port;

	if (!sa)
		return UTIL_NO_ENTRY;

	switch (((struct sockaddr *) sa)->sa_family) {
	case AF_INET:
		host = (uint16_t) ntohl(((struct sockaddr_in *) sa)->
					sin_addr.s_addr);
		port = ntohs(((struct sockaddr_in *) sa)->sin_port);
		break;
	case AF_INET6:
		host = (uint16_t) ((struct sockaddr_in6 *) sa)->
					sin6_addr.s6_addr[15];
		port = ntohs(((struct sockaddr_in6 *) sa)->sin6_port);
		break;
	default:
		assert(0);
		return UTIL_NO_ENTRY;
	}

	/* TODO: Find a good hash function */
	FI_DBG(av->prov, FI_LOG_AV, "slot %d\n",
		((host << 16) | port) % av->hash.slots);
	return ((host << 16) | port) % av->hash.slots;
}

int ip_av_get_index(struct util_av *av, const void *addr)
{
	return ofi_av_lookup_index(av, addr, ip_av_slot(av, addr));
}

void ofi_av_write_event(struct util_av *av, uint64_t data,
			int err, void *context)
{
	struct fi_eq_err_entry entry;
	size_t size;
	ssize_t ret;
	uint64_t flags;

	entry.fid = &av->av_fid.fid;
	entry.context = context;
	entry.data = data;

	if (err) {
		FI_INFO(av->prov, FI_LOG_AV, "writing error entry to EQ\n");
		entry.err = err;
		size = sizeof(struct fi_eq_err_entry);
		flags = UTIL_FLAG_ERROR;
	} else {
		FI_DBG(av->prov, FI_LOG_AV, "writing entry to EQ\n");
		size = sizeof(struct fi_eq_entry);
		flags = 0;
	}

	ret = fi_eq_write(&av->eq->eq_fid, FI_AV_COMPLETE, &entry,
			  size, flags);
	if (ret != size)
		FI_WARN(av->prov, FI_LOG_AV, "error writing to EQ\n");
}

static int ip_av_valid_addr(struct util_av *av, const void *addr)
{
	const struct sockaddr_in *sin = addr;
	const struct sockaddr_in6 *sin6 = addr;

	switch (sin->sin_family) {
	case AF_INET:
		return sin->sin_port && sin->sin_addr.s_addr;
	case AF_INET6:
		return sin6->sin6_port &&
		      memcmp(&in6addr_any, &sin6->sin6_addr, sizeof(in6addr_any));
	default:
		return 0;
	}
}

static int ip_av_insert_addr(struct util_av *av, const void *addr,
			     fi_addr_t *fi_addr, void *context)
{
	int ret, index = -1;

	if (ip_av_valid_addr(av, addr)) {
		ret = ofi_av_insert_addr(av, addr, ip_av_slot(av, addr), &index);
	} else {
		ret = -FI_EADDRNOTAVAIL;
		FI_WARN(av->prov, FI_LOG_AV, "invalid address\n");
	}

	if (fi_addr)
		*fi_addr = !ret ? index : FI_ADDR_NOTAVAIL;
	return ret;
}

static int ip_av_insert(struct fid_av *av_fid, const void *addr, size_t count,
			fi_addr_t *fi_addr, uint64_t flags, void *context)
{
	struct util_av *av;
	int i, ret, success_cnt = 0;
	size_t addrlen;

	av = container_of(av_fid, struct util_av, av_fid);
	ret = fi_verify_av_insert(av, flags);
	if (ret)
		return ret;

	addrlen = ((struct sockaddr *) addr)->sa_family == AF_INET ?
		  sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
	FI_DBG(av->prov, FI_LOG_AV, "inserting %d addresses\n", count);
	for (i = 0; i < count; i++) {
		ret = ip_av_insert_addr(av, (const char *) addr + i * addrlen,
					fi_addr ? &fi_addr[i] : NULL, context);
		if (!ret)
			success_cnt++;
		else if (av->eq)
			ofi_av_write_event(av, i, -ret, context);
	}

	FI_DBG(av->prov, FI_LOG_AV, "%d addresses successful\n", success_cnt);
	if (av->eq) {
		ofi_av_write_event(av, success_cnt, 0, context);
		ret = 0;
	} else {
		ret = success_cnt;
	}
	return ret;
}

static int ip_av_insert_svc(struct util_av *av, const char *node,
			    const char *service, fi_addr_t *fi_addr,
			    void *context)
{
	struct addrinfo hints, *ai;
	int ret;

	FI_INFO(av->prov, FI_LOG_AV, "inserting %s-%s\n", node, service);

	memset(&hints, 0, sizeof hints);
	hints.ai_socktype = SOCK_DGRAM;
	switch (av->domain->addr_format) {
	case FI_SOCKADDR_IN:
		hints.ai_family = AF_INET;
		break;
	case FI_SOCKADDR_IN6:
		hints.ai_family = AF_INET6;
		break;
	default:
		break;
	}

	ret = getaddrinfo(node, service, &hints, &ai);
	if (ret)
		return ret;

	ret = ip_av_insert_addr(av, ai->ai_addr, fi_addr, context);
	freeaddrinfo(ai);
	return ret;
}

static int ip_av_insertsvc(struct fid_av *av, const char *node,
			   const char *service, fi_addr_t *fi_addr,
			   uint64_t flags, void *context)
{
	return fi_av_insertsym(av, node, 1, service, 1, fi_addr, flags, context);
}

static int ip_av_insert_ip4sym(struct util_av *av,
			       struct in_addr ip, size_t ipcnt,
			       uint16_t port, size_t portcnt,
			       fi_addr_t *fi_addr, void *context)
{
	struct sockaddr_in sin;
	int i, p, fi, ret, success_cnt = 0;

	memset(&sin, 0, sizeof sin);
	sin.sin_family = AF_INET;

	for (i = 0, fi = 0; i < ipcnt; i++) {
		/* TODO: should we skip addresses x.x.x.0 and x.x.x.255? */
		sin.sin_addr.s_addr = htonl(ntohl(ip.s_addr) + i);

		for (p = 0; p < portcnt; p++, fi++) {
			sin.sin_port = htons(port + p);
			ret = ip_av_insert_addr(av, &sin, fi_addr ?
						&fi_addr[fi] : NULL, context);
			if (!ret)
				success_cnt++;
			else if (av->eq)
				ofi_av_write_event(av, fi, -ret, context);
		}
	}

	return success_cnt;
}

static int ip_av_insert_ip6sym(struct util_av *av,
			       struct in6_addr ip, size_t ipcnt,
			       uint16_t port, size_t portcnt,
			       fi_addr_t *fi_addr, void *context)
{
	struct sockaddr_in6 sin6;
	int i, j, p, fi, ret, success_cnt = 0;

	memset(&sin6, 0, sizeof sin6);
	sin6.sin6_family = AF_INET6;
	sin6.sin6_addr = ip;

	for (i = 0, fi = 0; i < ipcnt; i++) {
		for (p = 0; p < portcnt; p++, fi++) {
			sin6.sin6_port = htons(port + p);
			ret = ip_av_insert_addr(av, &sin6, fi_addr ?
						&fi_addr[fi] : NULL, context);
			if (!ret)
				success_cnt++;
			else if (av->eq)
				ofi_av_write_event(av, fi, -ret, context);
		}

		/* TODO: should we skip addresses x::0 and x::255? */
		for (j = 15; j >= 0; j--) {
			if (++sin6.sin6_addr.s6_addr[j] < 255)
				break;
		}
	}

	return success_cnt;
}

static int ip_av_insert_nodesym(struct util_av *av,
				const char *node, size_t nodecnt,
				const char *service, size_t svccnt,
				fi_addr_t *fi_addr, void *context)
{
	char name[FI_NAME_MAX];
	char svc[FI_NAME_MAX];
	size_t name_len;
	int fi, n, s, ret, name_index, svc_index, success_cnt = 0;

	for (name_len = strlen(node); isdigit(node[name_len - 1]); )
		name_len--;

	memcpy(name, node, name_len);
	name_index = atoi(node + name_len);
	svc_index = atoi(service);

	for (n = 0, fi = 0; n < nodecnt; n++) {
		if (nodecnt == 1) {
			strncpy(name, node, sizeof(name) - 1);
			name[FI_NAME_MAX - 1] = '\0';
		} else {
			snprintf(name + name_len, sizeof(name) - name_len - 1,
				 "%d", name_index + n);
		}

		for (s = 0; s < svccnt; s++, fi++) {
			if (svccnt == 1) {
				strncpy(svc, service, sizeof(svc) - 1);
				svc[FI_NAME_MAX - 1] = '\0';
			} else {
				snprintf(svc, sizeof(svc) - 1,
					 "%d", svc_index + s);
			}

			ret = ip_av_insert_svc(av, name, svc, fi_addr ?
					       &fi_addr[fi] : NULL, context);
			if (!ret)
				success_cnt++;
			else if (av->eq)
				ofi_av_write_event(av, fi, -ret, context);
		}
	}

	return success_cnt;
}

static int ip_av_insertsym(struct fid_av *av_fid, const char *node, size_t nodecnt,
			   const char *service, size_t svccnt, fi_addr_t *fi_addr,
			   uint64_t flags, void *context)
{
	struct util_av *av;
	struct in6_addr ip6;
	struct in_addr ip4;
	int ret;

	av = container_of(av_fid, struct util_av, av_fid);
	ret = fi_verify_av_insert(av, flags);
	if (ret)
		return ret;

	if (strlen(node) >= FI_NAME_MAX || strlen(service) >= FI_NAME_MAX) {
		FI_WARN(av->prov, FI_LOG_AV,
			"node or service name is too long\n");
		return -FI_ENOSYS;
	}

	ret = inet_pton(AF_INET, node, &ip4);
	if (ret == 1) {
		FI_INFO(av->prov, FI_LOG_AV, "insert symmetric IPv4\n");
		ret = ip_av_insert_ip4sym(av, ip4, nodecnt,
					  (uint16_t) strtol(service, NULL, 0),
					  svccnt, fi_addr, context);
		goto out;
	}

	ret = inet_pton(AF_INET6, node, &ip6);
	if (ret == 1) {
		FI_INFO(av->prov, FI_LOG_AV, "insert symmetric IPv6\n");
		ret = ip_av_insert_ip6sym(av, ip6, nodecnt,
					  (uint16_t) strtol(service, NULL, 0),
					  svccnt, fi_addr, context);
		goto out;
	}

	FI_INFO(av->prov, FI_LOG_AV, "insert symmetric host names\n");
	ret = ip_av_insert_nodesym(av, node, nodecnt, service, svccnt,
				  fi_addr, context);

out:
	if (av->eq) {
		ofi_av_write_event(av, ret, 0, context);
		ret = 0;
	}
	return ret;
}

static int ip_av_remove(struct fid_av *av_fid, fi_addr_t *fi_addr, size_t count,
			uint64_t flags)
{
	struct util_av *av;
	int i, slot, index, ret;

	av = container_of(av_fid, struct util_av, av_fid);
	if (flags) {
		FI_WARN(av->prov, FI_LOG_AV, "invalid flags\n");
		return -FI_EINVAL;
	}

	/*
	 * It's more efficient to remove addresses from high to low index.
	 * We assume that addresses are removed in the same order that they were
	 * added -- i.e. fi_addr passed in here was also passed into insert.
	 * Thus, we walk through the array backwards.
	 */
	for (i = count - 1; i >= 0; i--) {
		index = (int) fi_addr[i];
		slot = ip_av_slot(av, ip_av_get_addr(av, index));
		ret = fi_av_remove_addr(av, slot, index);
		if (ret) {
			FI_WARN(av->prov, FI_LOG_AV,
				"removal of fi_addr %d failed\n", index);
		}
	}
	return 0;
}

static int ip_av_lookup(struct fid_av *av_fid, fi_addr_t fi_addr, void *addr,
			size_t *addrlen)
{
	struct util_av *av;
	int index;

	av = container_of(av_fid, struct util_av, av_fid);
	index = (int) fi_addr;
	if (index < 0 || index > av->count) {
		FI_WARN(av->prov, FI_LOG_AV, "unknown address\n");
		return -FI_EINVAL;
	}

	memcpy(addr, ip_av_get_addr(av, index),
	       MIN(*addrlen, av->addrlen));
	*addrlen = av->addrlen;
	return 0;
}

static const char *ip_av_straddr(struct fid_av *av, const void *addr,
				  char *buf, size_t *len)
{
	char str[INET6_ADDRSTRLEN + 8];
	size_t size;

	if (!inet_ntop(((struct sockaddr *) addr)->sa_family, addr,
			str, sizeof str))
		return NULL;

	size = strlen(str);
	size += snprintf(&str[size], sizeof(str) - size, ":%d",
			 ((struct sockaddr_in *) addr)->sin_port);
	memcpy(buf, str, MIN(*len, size));
	*len = size + 1;
	return buf;
}

static struct fi_ops_av ip_av_ops = {
	.size = sizeof(struct fi_ops_av),
	.insert = ip_av_insert,
	.insertsvc = ip_av_insertsvc,
	.insertsym = ip_av_insertsym,
	.remove = ip_av_remove,
	.lookup = ip_av_lookup,
	.straddr = ip_av_straddr,
};

static int ip_av_close(struct fid *av_fid)
{
	int ret;
	struct util_av *av;
	av = container_of(av_fid, struct util_av, av_fid.fid);
	ret = ofi_av_close(av);
	if (ret)
		return ret;
	free(av);
	return 0;
}

static struct fi_ops ip_av_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = ip_av_close,
	.bind = ofi_av_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

int ip_av_create(struct fid_domain *domain_fid, struct fi_av_attr *attr,
		 struct fid_av **av, void *context)
{
	struct util_domain *domain;
	struct util_av_attr util_attr;
	struct util_av *util_av;
	int ret;

	domain = container_of(domain_fid, struct util_domain, domain_fid);
	if (domain->addr_format == FI_SOCKADDR_IN)
		util_attr.addrlen = sizeof(struct sockaddr_in);
	else
		util_attr.addrlen = sizeof(struct sockaddr_in6);

	util_attr.overhead = attr->count >> 1;
	util_attr.flags = domain->caps & FI_SOURCE ? FI_SOURCE : 0;

	if (attr->type == FI_AV_UNSPEC)
		attr->type = FI_AV_MAP;

	util_av = calloc(1, sizeof(*util_av));
	if (!util_av)
		return -FI_ENOMEM;

	ret = ofi_av_init(domain, attr, &util_attr, util_av, context);
	if (ret)
		return ret;

	*av = &util_av->av_fid;
	(*av)->fid.ops = &ip_av_fi_ops;
	(*av)->ops = &ip_av_ops;
	return 0;
}

/*
 * Connection Map
 */
static void ofi_cmap_init_handle(struct util_cmap_handle *handle,
		struct util_cmap *cmap,
		enum util_cmap_state state,
		fi_addr_t fi_addr,
		struct util_cmap_peer *peer)
{
	handle->cmap = cmap;
	handle->state = state;
	handle->key = freestack_pop(cmap->keypool);
	handle->key->handle = handle;
	handle->key_index = util_cmap_keypool_index(cmap->keypool, handle->key);
	handle->fi_addr = fi_addr;
	handle->peer = peer;
}

void ofi_cmap_update_state(struct util_cmap_handle *handle,
		enum util_cmap_state state)
{
	fastlock_acquire(&handle->cmap->lock);
	handle->state = state;
	fastlock_release(&handle->cmap->lock);
}

static int ofi_cmap_match_peer(struct dlist_entry *entry, const void *addr)
{
	struct util_cmap_peer *peer;

	peer = container_of(entry, struct util_cmap_peer, entry);
	return !memcmp(peer->addr, addr, peer->addrlen);
}

static int ofi_cmap_add_peer(struct util_cmap *cmap, struct util_cmap_handle *handle,
		enum util_cmap_state state, void *addr, size_t addrlen)
{
	struct util_cmap_peer *peer;
	int ret = 0;

	fastlock_acquire(&cmap->lock);
	if (dlist_find_first_match(&cmap->peer_list, ofi_cmap_match_peer, addr)) {
		FI_WARN(cmap->av->prov, FI_LOG_EP_CTRL,
				"Peer already present\n");
		goto out;
	}

	// TODO Use util_buf_pool
	peer = calloc(1, sizeof(*peer) + addrlen);
	if (!peer) {
		ret = -FI_ENOMEM;
		goto out;
	}

	ofi_cmap_init_handle(handle, cmap, state, FI_ADDR_UNSPEC, peer);
	peer->handle = handle;
	peer->addrlen = addrlen;
	memcpy(peer->addr, addr, addrlen);
	dlist_insert_tail(&peer->entry, &cmap->peer_list);
out:
	fastlock_release(&cmap->lock);
	return ret;
}

/*
 * Caller must hold cmap->lock. Either fi_addr or
 * addr and addrlen args should be present.
 */
int ofi_cmap_add_handle(struct util_cmap *cmap, struct util_cmap_handle *handle,
		enum util_cmap_state state, fi_addr_t fi_addr, void *addr,
		size_t addrlen)
{
	int index;
	if (fi_addr == FI_ADDR_UNSPEC) {
		index = ip_av_get_index(cmap->av, addr);
		if (index < 0)
			return ofi_cmap_add_peer(cmap, handle, state, addr, addrlen);
		fi_addr = index;
	}

	if (cmap->handles[fi_addr]) {
		FI_WARN(cmap->av->prov, FI_LOG_EP_CTRL,
				"Handle already present\n");
	} else {
		ofi_cmap_init_handle(handle, cmap, state, fi_addr, NULL);
		cmap->handles[fi_addr] = handle;
	}
	return 0;
}

/* Caller must hold cmap->lock */
struct util_cmap_handle *ofi_cmap_get_handle(struct util_cmap *cmap, fi_addr_t fi_addr)
{
	struct util_cmap_peer *peer;
	struct dlist_entry *entry;

	if (cmap->handles[fi_addr])
		return cmap->handles[fi_addr];

	/* Search in peer list */
	entry = dlist_remove_first_match(&cmap->peer_list, ofi_cmap_match_peer,
			ip_av_get_addr(cmap->av, fi_addr));
	if (!entry)
		return NULL;
	peer = container_of(entry, struct util_cmap_peer, entry);

	/* Move handle to cmap */
	peer->handle->peer = NULL;
	peer->handle->fi_addr = fi_addr;

	cmap->handles[fi_addr] = peer->handle;
	free(peer);
	return cmap->handles[fi_addr];
}

void ofi_cmap_del_handle(struct util_cmap_handle *handle)
{
	struct util_cmap *cmap = handle->cmap;

	fastlock_acquire(&cmap->lock);
	if (handle->peer) {
		dlist_remove(&handle->peer->entry);
		free(handle->peer);
	} else {
		cmap->handles[handle->fi_addr] = 0;
	}
	handle->key->handle = NULL;
	freestack_push(cmap->keypool, handle->key);
	cmap->free_handle(handle);
	fastlock_release(&cmap->lock);
}

void ofi_cmap_del_handles(struct util_cmap *cmap)
{
	struct util_cmap_peer *peer;
	struct dlist_entry *entry;
	int i;

	for (i = 0; i < cmap->av->count; i++) {
		if (cmap->handles[i])
			ofi_cmap_del_handle(cmap->handles[i]);
	}
	dlist_foreach(&cmap->peer_list, entry) {
		peer = container_of(entry, struct util_cmap_peer, entry);
		ofi_cmap_del_handle(peer->handle);
	}
}

void ofi_cmap_free(struct util_cmap *cmap)
{
	ofi_cmap_del_handles(cmap);
	fastlock_acquire(&cmap->lock);
	util_cmap_keypool_free(cmap->keypool);
	free(cmap->handles);
	fastlock_release(&cmap->lock);
	free(cmap);
}

struct util_cmap *ofi_cmap_alloc(struct util_av *av,
		ofi_cmap_free_handle_func free_handle)
{
	struct util_cmap *cmap;

	cmap = calloc(1, sizeof *cmap);
	if (!cmap)
		return NULL;

	cmap->av = av;

	cmap->handles = calloc(cmap->av->count, sizeof(*cmap->handles));
	if (!cmap->handles)
		goto err1;

	cmap->keypool = util_cmap_keypool_create(cmap->av->count);
	if (!cmap->keypool)
		goto err2;

	dlist_init(&cmap->peer_list);
	cmap->free_handle = free_handle;
	fastlock_init(&cmap->lock);

	return cmap;
err2:
	free(cmap->handles);
err1:
	free(cmap);
	return NULL;
}
