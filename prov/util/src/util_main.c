/*
 * Copyright (c) 2014-2016 Intel Corporation, Inc.  All rights reserved.
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

#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <fi_util.h>
#include <fi.h>
#include <arpa/inet.h>

#if HAVE_GETIFADDRS
#include <net/if.h>
#include <ifaddrs.h>
#endif

static DEFINE_LIST(fabric_list);
extern struct ofi_common_locks common_locks;

void ofi_fabric_insert(struct util_fabric *fabric)
{
	pthread_mutex_lock(&common_locks.util_fabric_lock);
	dlist_insert_tail(&fabric->list_entry, &fabric_list);
	pthread_mutex_unlock(&common_locks.util_fabric_lock);
}

static int util_match_fabric(struct dlist_entry *item, const void *arg)
{
	struct util_fabric *fabric;
	struct util_fabric_info *fabric_info = (struct util_fabric_info *)arg;

	fabric = container_of(item, struct util_fabric, list_entry);
	return (fabric_info->prov == fabric->prov) &&
		!strcmp(fabric->name, fabric_info->name);
}

struct util_fabric *ofi_fabric_find(struct util_fabric_info *fabric_info)
{
	struct dlist_entry *item;

	pthread_mutex_lock(&common_locks.util_fabric_lock);
	item = dlist_find_first_match(&fabric_list, util_match_fabric, fabric_info);
	pthread_mutex_unlock(&common_locks.util_fabric_lock);

	return item ? container_of(item, struct util_fabric, list_entry) : NULL;
}

void ofi_fabric_remove(struct util_fabric *fabric)
{
	pthread_mutex_lock(&common_locks.util_fabric_lock);
	dlist_remove(&fabric->list_entry);
	pthread_mutex_unlock(&common_locks.util_fabric_lock);
}


static int ofi_fid_match(struct dlist_entry *entry, const void *fid)
{
	struct fid_list_entry *item;
	item = container_of(entry, struct fid_list_entry, entry);
	return (item->fid == fid);
}

int fid_list_insert(struct dlist_entry *fid_list, fastlock_t *lock,
		    struct fid *fid)
{
	int ret = 0;
	struct dlist_entry *entry;
	struct fid_list_entry *item;

	fastlock_acquire(lock);
	entry = dlist_find_first_match(fid_list, ofi_fid_match, fid);
	if (entry)
		goto out;

	item = calloc(1, sizeof(*item));
	if (!item) {
		ret = -FI_ENOMEM;
		goto out;
	}

	item->fid = fid;
	dlist_insert_tail(&item->entry, fid_list);
out:
	fastlock_release(lock);
	return ret;
}

void fid_list_remove(struct dlist_entry *fid_list, fastlock_t *lock,
		     struct fid *fid)
{
	struct fid_list_entry *item;
	struct dlist_entry *entry;

	fastlock_acquire(lock);
	entry = dlist_remove_first_match(fid_list, ofi_fid_match, fid);
	fastlock_release(lock);

	if (entry) {
		item = container_of(entry, struct fid_list_entry, entry);
		free(item);
	}
}

int util_find_domain(struct dlist_entry *item, const void *arg)
{
	const struct util_domain *domain;
	const struct fi_info *info = arg;

	domain = container_of(item, struct util_domain, list_entry);

	return !strcmp(domain->name, info->domain_attr->name) &&
		!((info->caps | info->domain_attr->caps) & ~domain->info_domain_caps) &&
		 (((info->mode | info->domain_attr->mode) &
		   domain->info_domain_mode) == domain->info_domain_mode) &&
		 ((info->domain_attr->mr_mode & domain->mr_mode) == domain->mr_mode);
}
#if HAVE_GETIFADDRS
static int util_get_prefix_len(struct sockaddr *addr)
{
	struct sockaddr_in *in_addr;
	struct sockaddr_in6 *in6_addr;
	int prefix_len = 0, idx;
	unsigned int ip_addr;

	if (AF_INET == addr->sa_family) {
		in_addr = (struct sockaddr_in *) addr;

		ip_addr = ntohl(in_addr->sin_addr.s_addr);
		while (ip_addr) {
			ip_addr <<= 1;
			prefix_len++;
		}
	} else if (AF_INET6 == addr->sa_family){

		in6_addr = (struct sockaddr_in6 *) addr;
		for (idx = 0 ; idx < 16 ; idx++) {
			if (!in6_addr->sin6_addr.s6_addr[idx])
				break;

			switch (in6_addr->sin6_addr.s6_addr[idx]) {
			case 0xff:
				prefix_len += 8;
				break;
			case 0xfe:
				prefix_len += 7;
				break;
			case 0xfc:
				prefix_len += 6;
				break;
			case 0xf8:
				prefix_len += 5;
				break;
			case 0xf0:
				prefix_len += 4;
				break;
			case 0xe0:
				prefix_len += 3;
				break;
			case 0xc0:
				prefix_len += 2;
				break;
			case 0x80:
				prefix_len += 1;
				break;
			default:
				goto out;
			}
		}
	}
out:
	return prefix_len;
}

static int util_equals_ipaddr(struct sockaddr *addr1,
				struct sockaddr *addr2)
{
	struct sockaddr_in *in_addr1, *in_addr2;
	struct sockaddr_in6 *in6_addr1, *in6_addr2;

	if (!addr1 || !addr2 ||
	    (addr1->sa_family != addr2->sa_family))
		return 0;

	if (AF_INET == addr1->sa_family) {
		in_addr1 = (struct sockaddr_in *)addr1;
		in_addr2 = (struct sockaddr_in *)addr2;
		return (!memcmp(&in_addr1->sin_addr,
				&in_addr2->sin_addr,
				sizeof(in_addr1->sin_addr)));
	} else if (AF_INET6 == addr1->sa_family){
		in6_addr1 = (struct sockaddr_in6 *)addr1;
		in6_addr2 = (struct sockaddr_in6 *)addr2;
		return (!memcmp(&in6_addr1->sin6_addr,
				&in6_addr2->sin6_addr,
				sizeof(in6_addr1->sin6_addr)));
	}
	return 0;
}

static char *util_get_domain_name(const struct fi_provider *prov,
				  struct fi_info *info)
{
	struct sockaddr *addr;
	int ret;
	struct ifaddrs *ifaddrs, *ifa;
	char *domain_name = NULL;

	if (!info && !info->src_addr)
		return NULL;

	switch (info->addr_format) {
	case FI_SOCKADDR:
	case FI_SOCKADDR_IN:
	case FI_SOCKADDR_IN6:
		addr = (struct sockaddr *) info->src_addr;
		ret = getifaddrs(&ifaddrs);
		if (ret)
			return NULL;

		for (ifa = ifaddrs; ifa != NULL; ifa = ifa->ifa_next) {
			if (ifa->ifa_addr == NULL || !(ifa->ifa_flags & IFF_UP) ||
			    ((ifa->ifa_addr->sa_family != AF_INET) &&
			     (ifa->ifa_addr->sa_family != AF_INET6)))
				continue;

			if (util_equals_ipaddr((struct sockaddr *)ifa->ifa_addr, addr)) {
				domain_name = strdup(ifa->ifa_name);
				freeifaddrs(ifaddrs);
				return domain_name;
			}
		}
		freeifaddrs(ifaddrs);
		break;
	default:
		FI_DBG(prov, FI_LOG_CORE,
		       "unsupported address format for fabric name\n");
	}
	return NULL;
}

static char *util_get_fabric_name(const struct fi_provider *prov,
				  struct fi_info *info)
{
	struct sockaddr *addr;
	int ret;

	struct ifaddrs *ifaddrs, *ifa;
	char *fabric_name = NULL;
	char netbuf[INET6_ADDRSTRLEN+4];
	int prefix_len, idx;
	struct sockaddr_in *host_addr, *net_mask;
	struct sockaddr_in6 *host6_addr, *net6_mask;
	struct in_addr in_addr;
	struct in6_addr in6_addr;

	if (!info && !info->src_addr)
		return NULL;

	switch (info->addr_format) {
	case FI_SOCKADDR:
	case FI_SOCKADDR_IN:
	case FI_SOCKADDR_IN6:
		addr = (struct sockaddr *) info->src_addr;
		ret = getifaddrs(&ifaddrs);
		if (ret)
			return NULL;

		for (ifa = ifaddrs; ifa != NULL; ifa = ifa->ifa_next) {
			if (ifa->ifa_addr == NULL || !(ifa->ifa_flags & IFF_UP) ||
			    (ifa->ifa_addr->sa_family != AF_INET &&
			     ifa->ifa_addr->sa_family != AF_INET6 ))
				continue;

			if (!util_equals_ipaddr((struct sockaddr *)ifa->ifa_addr, addr))
				continue;

			if (AF_INET == ifa->ifa_addr->sa_family) {
				host_addr = (struct sockaddr_in *)ifa->ifa_addr;
				net_mask = (struct sockaddr_in *)ifa->ifa_netmask;
				in_addr.s_addr = (uint32_t)((uint32_t) host_addr->sin_addr.s_addr &
							    (uint32_t) net_mask->sin_addr.s_addr);

				inet_ntop(host_addr->sin_family, (void *)&(in_addr), netbuf,
					  sizeof(netbuf));
				prefix_len = util_get_prefix_len((struct sockaddr *)net_mask);
			} else {
				host6_addr = (struct sockaddr_in6 *)ifa->ifa_addr;
				net6_mask = (struct sockaddr_in6 *)ifa->ifa_netmask;

				idx = sizeof(in6_addr.s6_addr);
				while (idx--) {
					in6_addr.s6_addr[idx] = host6_addr->sin6_addr.s6_addr[idx] &
								net6_mask->sin6_addr.s6_addr[idx];
				}
				inet_ntop(host6_addr->sin6_family, (void *)&(in6_addr), netbuf,
					  sizeof(netbuf));
				prefix_len = util_get_prefix_len((struct sockaddr *)net6_mask);
			}
			snprintf(netbuf + strlen(netbuf), sizeof(netbuf) - strlen(netbuf),
				 "%s%d", "/", prefix_len);
			fabric_name = strdup(netbuf);
			freeifaddrs(ifaddrs);
			return fabric_name;
		}
	default:
		FI_DBG(prov, FI_LOG_CORE,
		       "unsupported address format for fabric name\n");
	}
	return NULL;
}

#else //HAVE_GETIFADDRS
static char *util_get_fabric_name(const struct fi_provider *prov,
			   struct fi_info *info)
{
	return NULL;
}

static char *util_get_domain_name(const struct fi_provider *prov,
			   struct fi_info *info)
{
	return NULL;
}
#endif //HAVE_GETIFADDRS

void util_set_fabric_domain(const struct fi_provider *prov,
			    struct fi_info *info)
{
	struct util_fabric *fabric;
	struct util_domain *domain;
	struct util_fabric_info fabric_info;
	struct dlist_entry *item;
	char *name = NULL;

	name = util_get_fabric_name(prov,info);
	if (name) {
		if (info->fabric_attr->name)
			free(info->fabric_attr->name);

		info->fabric_attr->name = name;
		name = NULL;
	}

	name = util_get_domain_name(prov, info);
	if (name) {
		if (info->domain_attr->name)
			free(info->domain_attr->name);

		info->domain_attr->name = name;
		name = NULL;
	}

	fabric_info.name =info->fabric_attr->name;
	fabric_info.prov = prov;

	fabric = ofi_fabric_find(&fabric_info);
	if (fabric) {
		FI_DBG(prov, FI_LOG_CORE, "Found opened fabric\n");
		info->fabric_attr->fabric = &fabric->fabric_fid;

		fastlock_acquire(&fabric->lock);
		item = dlist_find_first_match(&fabric->domain_list,
					      util_find_domain, info);
		if (item) {
			FI_DBG(prov, FI_LOG_CORE,
			       "Found open domain\n");
			domain = container_of(item, struct util_domain,
					      list_entry);
			info->domain_attr->domain =
				&domain->domain_fid;
		}
		fastlock_release(&fabric->lock);
	}
}

int util_getinfo(const struct util_prov *util_prov, uint32_t version,
		 const char *node, const char *service, uint64_t flags,
		 const struct fi_info *hints, struct fi_info **info)
{
	const struct fi_provider *prov = util_prov->prov;
	struct fi_info *saved_info;
	int ret, copy_dest;

	FI_DBG(prov, FI_LOG_CORE, "checking info\n");

	if ((flags & FI_SOURCE) && !node && !service) {
		FI_INFO(prov, FI_LOG_CORE,
			"FI_SOURCE set, but no node or service\n");
		return -FI_EINVAL;
	}

	ret = ofi_prov_check_dup_info(util_prov, version, hints, info);
	if (ret)
		return ret;

	ofi_alter_info(*info, hints, version);

	saved_info = *info;

	for (; *info; *info = (*info)->next) {
		if (flags & FI_SOURCE) {
			ret = ofi_get_addr((*info)->addr_format, flags,
					  node, service, &(*info)->src_addr,
					  &(*info)->src_addrlen);
			if (ret) {
				FI_INFO(prov, FI_LOG_CORE,
					"source address not available\n");
				goto err;
			}
			copy_dest = (hints && hints->dest_addr);
		} else {
			if (node || service) {
				copy_dest = 0;
				ret = ofi_get_addr((*info)->addr_format,
						   flags, node, service,
						   &(*info)->dest_addr,
						   &(*info)->dest_addrlen);
				if (ret) {
					FI_INFO(prov, FI_LOG_CORE,
						"cannot resolve dest address\n");
					goto err;
				}
			} else {
				copy_dest = (hints && hints->dest_addr);
			}

			if (hints && hints->src_addr) {
				(*info)->src_addr = mem_dup(hints->src_addr,
						    hints->src_addrlen);
				if (!(*info)->src_addr) {
					ret = -FI_ENOMEM;
					goto err;
				}
				(*info)->src_addrlen = hints->src_addrlen;
			}
		}

		if (copy_dest) {
			(*info)->dest_addr = mem_dup(hints->dest_addr,
						     hints->dest_addrlen);
			if (!(*info)->dest_addr) {
				ret = -FI_ENOMEM;
				goto err;
			}
			(*info)->dest_addrlen = hints->dest_addrlen;
		}

		if ((*info)->dest_addr && !(*info)->src_addr) {
			ret = ofi_get_src_addr((*info)->addr_format,
					       (*info)->dest_addr,
					       (*info)->dest_addrlen,
					       &(*info)->src_addr,
					       &(*info)->src_addrlen);
			if (ret) {
				FI_INFO(prov, FI_LOG_CORE,
					"cannot resolve source address\n");
			}
		}
		util_set_fabric_domain(prov, *info);
	}

	*info = saved_info;

	return 0;

err:
	fi_freeinfo(*info);
	return ret;
}
