/*
 * Copyright (c) 2014 Intel Corporation, Inc.  All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2017 DataDirect Networks, Inc. All rights reserved.
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

#include <stdlib.h>
#include <string.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <limits.h>

#if HAVE_GETIFADDRS
#include <net/if.h>
#include <ifaddrs.h>
#endif

#include "ofi_prov.h"
#include "ofi_osd.h"

#include "sock.h"
#include "sock_util.h"

#define SOCK_LOG_DBG(...) _SOCK_LOG_DBG(FI_LOG_FABRIC, __VA_ARGS__)
#define SOCK_LOG_ERROR(...) _SOCK_LOG_ERROR(FI_LOG_FABRIC, __VA_ARGS__)

int sock_pe_waittime = SOCK_PE_WAITTIME;
const char sock_fab_name[] = "IP";
const char sock_dom_name[] = "sockets";
const char sock_prov_name[] = "sockets";
int sock_conn_retry = SOCK_CM_DEF_RETRY;
int sock_cm_def_map_sz = SOCK_CMAP_DEF_SZ;
int sock_av_def_sz = SOCK_AV_DEF_SZ;
int sock_cq_def_sz = SOCK_CQ_DEF_SZ;
int sock_eq_def_sz = SOCK_EQ_DEF_SZ;
#if ENABLE_DEBUG
int sock_dgram_drop_rate = 0;
#endif
int sock_keepalive_enable;
int sock_keepalive_time = INT_MAX;
int sock_keepalive_intvl = INT_MAX;
int sock_keepalive_probes = INT_MAX;

char *sock_interface_name = NULL;

uint64_t SOCK_EP_RDM_SEC_CAP = SOCK_EP_RDM_SEC_CAP_BASE;
uint64_t SOCK_EP_RDM_CAP = SOCK_EP_RDM_CAP_BASE;
uint64_t SOCK_EP_MSG_SEC_CAP = SOCK_EP_MSG_SEC_CAP_BASE;
uint64_t SOCK_EP_MSG_CAP = SOCK_EP_MSG_CAP_BASE;


const struct fi_fabric_attr sock_fabric_attr = {
	.fabric = NULL,
	.name = NULL,
	.prov_name = NULL,
	.prov_version = FI_VERSION(SOCK_MAJOR_VERSION, SOCK_MINOR_VERSION),
};

static struct dlist_entry sock_fab_list;
static struct dlist_entry sock_dom_list;
static fastlock_t sock_list_lock;
static struct slist sock_addr_list;
static int read_default_params;

void sock_dom_add_to_list(struct sock_domain *domain)
{
	fastlock_acquire(&sock_list_lock);
	dlist_insert_tail(&domain->dom_list_entry, &sock_dom_list);
	fastlock_release(&sock_list_lock);
}

static inline int sock_dom_check_list_internal(struct sock_domain *domain)
{
	struct dlist_entry *entry;
	struct sock_domain *dom_entry;
	for (entry = sock_dom_list.next; entry != &sock_dom_list;
	     entry = entry->next) {
		dom_entry = container_of(entry, struct sock_domain,
					 dom_list_entry);
		if (dom_entry == domain)
			return 1;
	}
	return 0;
}

int sock_dom_check_list(struct sock_domain *domain)
{
	int found;
	fastlock_acquire(&sock_list_lock);
	found = sock_dom_check_list_internal(domain);
	fastlock_release(&sock_list_lock);
	return found;
}

void sock_dom_remove_from_list(struct sock_domain *domain)
{
	fastlock_acquire(&sock_list_lock);
	if (sock_dom_check_list_internal(domain))
		dlist_remove(&domain->dom_list_entry);

	fastlock_release(&sock_list_lock);
}

struct sock_domain *sock_dom_list_head(void)
{
	struct sock_domain *domain;
	fastlock_acquire(&sock_list_lock);
	if (dlist_empty(&sock_dom_list)) {
		domain = NULL;
	} else {
		domain = container_of(sock_dom_list.next,
				      struct sock_domain, dom_list_entry);
	}
	fastlock_release(&sock_list_lock);
	return domain;
}

int sock_dom_check_manual_progress(struct sock_fabric *fabric)
{
	struct dlist_entry *entry;
	struct sock_domain *dom_entry;
	for (entry = sock_dom_list.next; entry != &sock_dom_list;
	     entry = entry->next) {
		dom_entry = container_of(entry, struct sock_domain,
					 dom_list_entry);
		if (dom_entry->fab == fabric &&
		    dom_entry->progress_mode == FI_PROGRESS_MANUAL)
			return 1;
	}
	return 0;
}

void sock_fab_add_to_list(struct sock_fabric *fabric)
{
	fastlock_acquire(&sock_list_lock);
	dlist_insert_tail(&fabric->fab_list_entry, &sock_fab_list);
	fastlock_release(&sock_list_lock);
}

static inline int sock_fab_check_list_internal(struct sock_fabric *fabric)
{
	struct dlist_entry *entry;
	struct sock_fabric *fab_entry;
	for (entry = sock_fab_list.next; entry != &sock_fab_list;
	     entry = entry->next) {
		fab_entry = container_of(entry, struct sock_fabric,
					 fab_list_entry);
		if (fab_entry == fabric)
			return 1;
	}
	return 0;
}

int sock_fab_check_list(struct sock_fabric *fabric)
{
	int found;
	fastlock_acquire(&sock_list_lock);
	found = sock_fab_check_list_internal(fabric);
	fastlock_release(&sock_list_lock);
	return found;
}

void sock_fab_remove_from_list(struct sock_fabric *fabric)
{
	fastlock_acquire(&sock_list_lock);
	if (sock_fab_check_list_internal(fabric))
		dlist_remove(&fabric->fab_list_entry);

	fastlock_release(&sock_list_lock);
}

struct sock_fabric *sock_fab_list_head(void)
{
	struct sock_fabric *fabric;
	fastlock_acquire(&sock_list_lock);
	if (dlist_empty(&sock_fab_list)) {
		fabric = NULL;
	} else {
		fabric = container_of(sock_fab_list.next,
				      struct sock_fabric, fab_list_entry);
	}
	fastlock_release(&sock_list_lock);
	return fabric;
}

int sock_verify_fabric_attr(const struct fi_fabric_attr *attr)
{
	if (!attr)
		return 0;

	if (attr->prov_version) {
		if (attr->prov_version !=
		   FI_VERSION(SOCK_MAJOR_VERSION, SOCK_MINOR_VERSION))
			return -FI_ENODATA;
	}

	return 0;
}

int sock_verify_info(uint32_t version, const struct fi_info *hints)
{
	uint64_t caps;
	enum fi_ep_type ep_type;
	int ret;
	struct sock_domain *domain;
	struct sock_fabric *fabric;

	if (!hints)
		return 0;

	ep_type = hints->ep_attr ? hints->ep_attr->type : FI_EP_UNSPEC;
	switch (ep_type) {
	case FI_EP_UNSPEC:
	case FI_EP_MSG:
		caps = SOCK_EP_MSG_CAP;
		ret = sock_msg_verify_ep_attr(hints->ep_attr,
					      hints->tx_attr,
					      hints->rx_attr);
		break;
	case FI_EP_DGRAM:
		caps = SOCK_EP_DGRAM_CAP;
		ret = sock_dgram_verify_ep_attr(hints->ep_attr,
						hints->tx_attr,
						hints->rx_attr);
		break;
	case FI_EP_RDM:
		caps = SOCK_EP_RDM_CAP;
		ret = sock_rdm_verify_ep_attr(hints->ep_attr,
					      hints->tx_attr,
					      hints->rx_attr);
		break;
	default:
		ret = -FI_ENODATA;
	}
	if (ret)
		return ret;

	if ((caps | hints->caps) != caps) {
		SOCK_LOG_DBG("Unsupported capabilities\n");
		return -FI_ENODATA;
	}

	switch (hints->addr_format) {
	case FI_FORMAT_UNSPEC:
	case FI_SOCKADDR:
	case FI_SOCKADDR_IN:
		break;
	default:
		SOCK_LOG_DBG("Unsupported address format\n");
		return -FI_ENODATA;
	}

	if (hints->domain_attr && hints->domain_attr->domain) {
		domain = container_of(hints->domain_attr->domain,
				      struct sock_domain, dom_fid);
		if (!sock_dom_check_list(domain)) {
			SOCK_LOG_DBG("no matching domain\n");
			return -FI_ENODATA;
		}
	}
	ret = sock_verify_domain_attr(version, hints);
	if (ret)
		return ret;

	if (hints->fabric_attr && hints->fabric_attr->fabric) {
		fabric = container_of(hints->fabric_attr->fabric,
				      struct sock_fabric, fab_fid);
		if (!sock_fab_check_list(fabric)) {
			SOCK_LOG_DBG("no matching fabric\n");
			return -FI_ENODATA;
		}
	}
	ret = sock_verify_fabric_attr(hints->fabric_attr);
	if (ret)
		return ret;

	return 0;
}

static int sock_trywait(struct fid_fabric *fabric, struct fid **fids, int count)
{
	/* we're always ready to wait! */
	return 0;
}

static struct fi_ops_fabric sock_fab_ops = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = sock_domain,
	.passive_ep = sock_msg_passive_ep,
	.eq_open = sock_eq_open,
	.wait_open = sock_wait_open,
	.trywait = sock_trywait
};

static int sock_fabric_close(fid_t fid)
{
	struct sock_fabric *fab;
	fab = container_of(fid, struct sock_fabric, fab_fid);
	if (ofi_atomic_get32(&fab->ref))
		return -FI_EBUSY;

	sock_fab_remove_from_list(fab);
	fastlock_destroy(&fab->lock);
	free(fab);
	return 0;
}

static struct fi_ops sock_fab_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = sock_fabric_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static void sock_read_default_params()
{
	if (!read_default_params) {
		fi_param_get_int(&sock_prov, "pe_waittime", &sock_pe_waittime);
		fi_param_get_int(&sock_prov, "max_conn_retry", &sock_conn_retry);
		fi_param_get_int(&sock_prov, "def_conn_map_sz", &sock_cm_def_map_sz);
		fi_param_get_int(&sock_prov, "def_av_sz", &sock_av_def_sz);
		fi_param_get_int(&sock_prov, "def_cq_sz", &sock_cq_def_sz);
		fi_param_get_int(&sock_prov, "def_eq_sz", &sock_eq_def_sz);
#if ENABLE_DEBUG
		fi_param_get_int(&sock_prov, "dgram_drop_rate", &sock_dgram_drop_rate);
#endif
		fi_param_get_bool(&sock_prov, "keepalive_enable", &sock_keepalive_enable);
		fi_param_get_int(&sock_prov, "keepalive_time", &sock_keepalive_time);
		fi_param_get_int(&sock_prov, "keepalive_intvl", &sock_keepalive_intvl);
		fi_param_get_int(&sock_prov, "keepalive_probes", &sock_keepalive_probes);

		read_default_params = 1;
	}
}

static int sock_fabric(struct fi_fabric_attr *attr,
		       struct fid_fabric **fabric, void *context)
{
	struct sock_fabric *fab;

	fab = calloc(1, sizeof(*fab));
	if (!fab)
		return -FI_ENOMEM;

	sock_read_default_params();

	fastlock_init(&fab->lock);
	dlist_init(&fab->service_list);

	fab->fab_fid.fid.fclass = FI_CLASS_FABRIC;
	fab->fab_fid.fid.context = context;
	fab->fab_fid.fid.ops = &sock_fab_fi_ops;
	fab->fab_fid.ops = &sock_fab_ops;
	*fabric = &fab->fab_fid;
	ofi_atomic_initialize32(&fab->ref, 0);
#if ENABLE_DEBUG
	fab->num_send_msg = 0;
#endif
	sock_fab_add_to_list(fab);
	return 0;
}

int sock_get_src_addr(struct sockaddr_in *dest_addr,
		      struct sockaddr_in *src_addr)
{
	int sock, ret;
	socklen_t len;

	sock = ofi_socket(AF_INET, SOCK_DGRAM, 0);
	if (sock < 0)
		return -ofi_sockerr();

	len = sizeof(*dest_addr);
	ret = connect(sock, (struct sockaddr *) dest_addr, len);
	if (ret) {
		SOCK_LOG_DBG("Failed to connect udp socket\n");
		ret = sock_get_src_addr_from_hostname(src_addr, NULL);
		goto out;
	}

	ret = getsockname(sock, (struct sockaddr *) src_addr, &len);
	src_addr->sin_port = 0;
	if (ret) {
		SOCK_LOG_DBG("getsockname failed\n");
		ret = -ofi_sockerr();
	}
out:
	ofi_close_socket(sock);
	return ret;
}

static int sock_fi_checkinfo(const struct fi_info *info,
			     const struct fi_info *hints)
{
	if (hints && hints->domain_attr && hints->domain_attr->name &&
             strcmp(info->domain_attr->name, hints->domain_attr->name))
		return -FI_ENODATA;

	if (hints && hints->fabric_attr && hints->fabric_attr->name &&
             strcmp(info->fabric_attr->name, hints->fabric_attr->name))
		return -FI_ENODATA;

	return 0;
}

static int sock_ep_getinfo(uint32_t version, const char *node,
			   const char *service, uint64_t flags,
			   const struct fi_info *hints, enum fi_ep_type ep_type,
			   struct fi_info **info)
{
	struct addrinfo ai, *rai = NULL;
	struct sockaddr_in *src_addr = NULL, *dest_addr = NULL;
	struct sockaddr_in sin;
	int ret;

	memset(&ai, 0, sizeof(ai));
	ai.ai_family = AF_INET;
	ai.ai_socktype = SOCK_STREAM;
	if (flags & FI_NUMERICHOST)
		ai.ai_flags |= AI_NUMERICHOST;

	if (flags & FI_SOURCE) {
		ai.ai_flags |= AI_PASSIVE;
		ret = getaddrinfo(node, service, &ai, &rai);
		if (ret) {
			SOCK_LOG_DBG("getaddrinfo failed!\n");
			return -FI_ENODATA;
		}
		src_addr = (struct sockaddr_in *) rai->ai_addr;
		if (hints && hints->dest_addr)
			dest_addr = hints->dest_addr;
	} else {
		if (node || service) {
			ret = getaddrinfo(node, service, &ai, &rai);
			if (ret) {
				SOCK_LOG_DBG("getaddrinfo failed!\n");
				return -FI_ENODATA;
			}
			dest_addr = (struct sockaddr_in *) rai->ai_addr;
		} else if (hints) {
			dest_addr = hints->dest_addr;
		}

		if (hints && hints->src_addr)
			src_addr = hints->src_addr;
	}

	if (dest_addr && !src_addr) {
		ret = sock_get_src_addr(dest_addr, &sin);
		if (!ret)
			src_addr = &sin;
	}

	if (src_addr)
		SOCK_LOG_DBG("src_addr: %s\n", inet_ntoa(src_addr->sin_addr));
	if (dest_addr)
		SOCK_LOG_DBG("dest_addr: %s\n", inet_ntoa(dest_addr->sin_addr));

	switch (ep_type) {
	case FI_EP_MSG:
		ret = sock_msg_fi_info(version, src_addr, dest_addr, hints, info);
		break;
	case FI_EP_DGRAM:
		ret = sock_dgram_fi_info(version, src_addr, dest_addr, hints, info);
		break;
	case FI_EP_RDM:
		ret = sock_rdm_fi_info(version, src_addr, dest_addr, hints, info);
		break;
	default:
		ret = -FI_ENODATA;
		break;
	}

	if (rai)
		freeaddrinfo(rai);

	if (ret == 0)
		return sock_fi_checkinfo(*info, hints);

	return ret;
}

void sock_insert_loopback_addr(struct slist *addr_list)
{
	struct sock_host_list_entry *addr_entry;

	addr_entry = calloc(1, sizeof(struct sock_host_list_entry));
	if (!addr_entry)
		return;
	strncpy(addr_entry->hostname, "127.0.0.1", sizeof(addr_entry->hostname));
	slist_insert_tail(&addr_entry->entry, addr_list);
}

#if HAVE_GETIFADDRS
void sock_get_list_of_addr(struct slist *addr_list)
{
	int ret;
	struct sock_host_list_entry *addr_entry;
	struct ifaddrs *ifaddrs, *ifa;

	fi_param_get_str(&sock_prov, "interface_name", &sock_interface_name);

	ret = ofi_getifaddrs(&ifaddrs);
	if (!ret) {
		if (sock_interface_name) {
			for (ifa = ifaddrs; ifa != NULL; ifa = ifa->ifa_next) {
				if (strncmp(sock_interface_name, ifa->ifa_name,
					    strlen(sock_interface_name)) == 0) {
					break;
				}
			}
			if (ifa == NULL) {
				FI_INFO(&sock_prov, FI_LOG_CORE,
					"Can't set filter to unknown interface: (%s)\n",
					sock_interface_name);
				sock_interface_name = NULL;
			}
		}
		for (ifa = ifaddrs; ifa != NULL; ifa = ifa->ifa_next) {
			if (ifa->ifa_addr == NULL || !(ifa->ifa_flags & IFF_UP) ||
			     (ifa->ifa_addr->sa_family != AF_INET) ||
			     !strcmp(ifa->ifa_name, "lo"))
				continue;
			if (sock_interface_name &&
			    strncmp(sock_interface_name, ifa->ifa_name,
				    strlen(sock_interface_name)) != 0) {
				SOCK_LOG_DBG("Skip (%s) interface\n", ifa->ifa_name);
				continue;
			}
			addr_entry = calloc(1, sizeof(struct sock_host_list_entry));
			if (!addr_entry)
				continue;
			ret = getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in),
					  addr_entry->hostname, sizeof(addr_entry->hostname),
					  NULL, 0, NI_NUMERICHOST);
			if (ret) {
				SOCK_LOG_DBG("getnameinfo failed: %d\n", ret);
				free(addr_entry);
				continue;
			}
			slist_insert_tail(&addr_entry->entry, addr_list);
		}
		freeifaddrs(ifaddrs);
	}
	// Always add loopback address at the end
	sock_insert_loopback_addr(addr_list);
}
#elif defined HAVE_MIB_IPADDRTABLE
void sock_get_list_of_addr(struct slist *addr_list)
{
	sock_get_ip_addr_table(addr_list);
	sock_insert_loopback_addr(addr_list);
}
#else
void sock_get_list_of_addr(struct slist *addr_list)
{
	sock_insert_loopback_addr(addr_list);
}
#endif

int sock_node_getinfo(uint32_t version, const char *node, const char *service,
		      uint64_t flags, const struct fi_info *hints, struct fi_info **info,
		      struct fi_info **tail)
{
	enum fi_ep_type ep_type;
	struct fi_info *cur;
	int ret;

	if (hints && hints->ep_attr) {
		switch (hints->ep_attr->type) {
		case FI_EP_RDM:
		case FI_EP_DGRAM:
		case FI_EP_MSG:
			ret = sock_ep_getinfo(version, node, service, flags,
					      hints, hints->ep_attr->type, &cur);
			if (ret) {
				if (ret == -FI_ENODATA)
					return ret;
				goto err;
			}

			if (!*info)
				*info = cur;
			else
				(*tail)->next = cur;
			(*tail) = cur;
			return 0;
		default:
			break;
		}
	}
	for (ep_type = FI_EP_MSG; ep_type <= FI_EP_RDM; ep_type++) {
		ret = sock_ep_getinfo(version, node, service, flags, hints,
				      ep_type, &cur);
		if (ret) {
			if (ret == -FI_ENODATA)
				continue;
			goto err;
		}

		if (!*info)
			*info = cur;
		else
			(*tail)->next = cur;
		(*tail) = cur;
	}
	if (!*info) {
		ret = -FI_ENODATA;
		goto err_no_free;
	}
	return 0;

err:
	fi_freeinfo(*info);
	*info = NULL;
err_no_free:
	return ret;
}

static int sock_match_src_addr(struct slist_entry *entry, const void *src_addr)
{
	struct sock_host_list_entry *host_entry;
	host_entry = container_of(entry, struct sock_host_list_entry, entry);

        return (strcmp(host_entry->hostname, (char *) src_addr) == 0);
}

static int sock_addr_matches_interface(struct slist *addr_list, struct sockaddr_in *src_addr)
{
	struct slist_entry *entry;

	/* Always match if it's localhost */
	if (ofi_is_loopback_addr((struct sockaddr *)src_addr))
		return 1;

	entry = slist_find_first_match(addr_list, sock_match_src_addr,
					inet_ntoa(src_addr->sin_addr));

	return entry ? 1 : 0;
}

static int sock_node_matches_interface(struct slist *addr_list, const char *node)
{
	struct sockaddr_in addr = { 0 };
	struct addrinfo *rai = NULL, ai = {
		.ai_family = AF_INET,
		.ai_socktype = SOCK_STREAM,
	};

	if (getaddrinfo(node, 0, &ai, &rai)) {
		SOCK_LOG_DBG("getaddrinfo failed!\n");
		return -FI_EINVAL;
	}
	addr = *(struct sockaddr_in *)rai->ai_addr;
	freeaddrinfo(rai);

	return sock_addr_matches_interface(addr_list, &addr);
}

static void sock_free_addr_list(struct slist *addr_list)
{
	struct slist_entry *entry;
	struct sock_host_list_entry *host_entry;

	while (!slist_empty(addr_list)) {
		entry = slist_remove_head(addr_list);
		host_entry = container_of(entry, struct sock_host_list_entry,
					   entry);
		free(host_entry);
	}
}

static int sock_getinfo(uint32_t version, const char *node, const char *service,
			uint64_t flags, const struct fi_info *hints,
			struct fi_info **info)
{
	int ret = 0;
	struct slist_entry *entry, *prev;
	struct sock_host_list_entry *host_entry;
	struct fi_info *tail;

	if (!(flags & FI_SOURCE) && hints && hints->src_addr &&
	    (hints->src_addrlen != sizeof(struct sockaddr_in)))
		return -FI_ENODATA;

	if (((!node && !service) || (flags & FI_SOURCE)) &&
	    hints && hints->dest_addr &&
	    (hints->dest_addrlen != sizeof(struct sockaddr_in)))
		return -FI_ENODATA;

	ret = sock_verify_info(version, hints);
	if (ret)
		return ret;

	ret = 1;
	if ((flags & FI_SOURCE) && node) {
		ret = sock_node_matches_interface(&sock_addr_list, node);
	} else if (hints && hints->src_addr) {
		ret = sock_addr_matches_interface(&sock_addr_list,
						  (struct sockaddr_in *)hints->src_addr);
	}
	if (!ret) {
		SOCK_LOG_ERROR("Couldn't find a match with local interfaces\n");
		return -FI_ENODATA;
	}

	*info = tail = NULL;
	if (node ||
	     (!(flags & FI_SOURCE) && hints && hints->src_addr) ||
	     (!(flags & FI_SOURCE) && hints && hints->dest_addr))
		return sock_node_getinfo(version, node, service, flags,
					 hints, info, &tail);

	(void) prev; /* Makes compiler happy */
	slist_foreach(&sock_addr_list, entry, prev) {
		host_entry = container_of(entry, struct sock_host_list_entry, entry);
		node = host_entry->hostname;
		flags |= FI_SOURCE;
		ret = sock_node_getinfo(version, node, service, flags, hints, info, &tail);
		if (ret) {
			if (ret == -FI_ENODATA)
				continue;
			return ret;
		}
	}

	return (!*info) ? ret : 0;
}

static void fi_sockets_fini(void)
{
	sock_free_addr_list(&sock_addr_list);
	fastlock_destroy(&sock_list_lock);
}

struct fi_provider sock_prov = {
	.name = sock_prov_name,
	.version = FI_VERSION(SOCK_MAJOR_VERSION, SOCK_MINOR_VERSION),
	.fi_version = FI_VERSION(1, 6),
	.getinfo = sock_getinfo,
	.fabric = sock_fabric,
	.cleanup = fi_sockets_fini
};

SOCKETS_INI
{
#if HAVE_SOCKETS_DL
	ofi_pmem_init();
#endif

	fi_param_define(&sock_prov, "pe_waittime", FI_PARAM_INT,
			"How many milliseconds to spin while waiting for progress");

	fi_param_define(&sock_prov, "max_conn_retry", FI_PARAM_INT,
			"Number of connection retries before reporting as failure");

	fi_param_define(&sock_prov, "def_conn_map_sz", FI_PARAM_INT,
			"Default connection map size");

	fi_param_define(&sock_prov, "def_av_sz", FI_PARAM_INT,
			"Default address vector size");

	fi_param_define(&sock_prov, "def_cq_sz", FI_PARAM_INT,
			"Default completion queue size");

	fi_param_define(&sock_prov, "def_eq_sz", FI_PARAM_INT,
			"Default event queue size");

	fi_param_define(&sock_prov, "pe_affinity", FI_PARAM_STRING,
			"If specified, bind the progress thread to the indicated range(s) of Linux virtual processor ID(s). "
			"This option is currently not supported on OS X and Windows. Usage: id_start[-id_end[:stride]][,]");

	fi_param_define(&sock_prov, "keepalive_enable", FI_PARAM_BOOL,
			"Enable keepalive support");

	fi_param_define(&sock_prov, "keepalive_time", FI_PARAM_INT,
			"Idle time in seconds before sending the first keepalive probe");

	fi_param_define(&sock_prov, "keepalive_intvl", FI_PARAM_INT,
			"Time in seconds between individual keepalive probes");

	fi_param_define(&sock_prov, "keepalive_probes", FI_PARAM_INT,
			"Maximum number of keepalive probes sent before dropping the connection");

	fi_param_define(&sock_prov, "interface_name", FI_PARAM_STRING,
			"Specify interface name");

	fastlock_init(&sock_list_lock);
	dlist_init(&sock_fab_list);
	dlist_init(&sock_dom_list);
	slist_init(&sock_addr_list);
	SOCK_EP_RDM_SEC_CAP |= OFI_RMA_PMEM;
	SOCK_EP_RDM_CAP |= OFI_RMA_PMEM;
	SOCK_EP_MSG_SEC_CAP |= OFI_RMA_PMEM;
	SOCK_EP_MSG_CAP |= OFI_RMA_PMEM;
	/* Returns loopback address if no other interfaces are available */
	sock_get_list_of_addr(&sock_addr_list);
#if ENABLE_DEBUG
	fi_param_define(&sock_prov, "dgram_drop_rate", FI_PARAM_INT,
			"Drop every Nth dgram frame (debug only)");
#endif
	return &sock_prov;
}
