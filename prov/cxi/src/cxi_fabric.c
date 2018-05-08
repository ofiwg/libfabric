/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2017 DataDirect Networks, Inc. All rights reserved.
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include "config.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>

#include "ofi_prov.h"
#include "ofi_osd.h"

#include "cxi_prov.h"

#define CXI_LOG_DBG(...) _CXI_LOG_DBG(FI_LOG_FABRIC, __VA_ARGS__)
#define CXI_LOG_ERROR(...) _CXI_LOG_ERROR(FI_LOG_FABRIC, __VA_ARGS__)

const char cxi_fab_fmt[] = "cxi/%d";	/* Provder/Net Name */
const char cxi_dom_fmt[] = "cxi%d:%d";	/* IF/Dom Name */
const char cxi_prov_name[] = "CXI";	/* Provider Name */

int cxi_av_def_sz = CXI_AV_DEF_SZ;
int cxi_cq_def_sz = CXI_CQ_DEF_SZ;
int cxi_eq_def_sz = CXI_EQ_DEF_SZ;

uint64_t CXI_EP_RDM_SEC_CAP = CXI_EP_RDM_SEC_CAP_BASE;
uint64_t CXI_EP_RDM_CAP = CXI_EP_RDM_CAP_BASE;

const struct fi_fabric_attr cxi_fabric_attr = {
	.fabric = NULL,
	.name = NULL,
	.prov_name = NULL,
	.prov_version = FI_VERSION(CXI_MAJOR_VERSION, CXI_MINOR_VERSION),
};

static struct dlist_entry cxi_fab_list;
static struct dlist_entry cxi_dom_list;
static fastlock_t cxi_list_lock;
struct slist cxi_if_list;
static int read_default_params;

struct cxi_if_list_entry *cxi_if_lookup(struct cxi_addr *src_addr)
{
	struct slist_entry *entry, *prev;
	struct cxi_if_list_entry *if_entry;

	(void) prev; /* Makes compiler happy */
	slist_foreach(&cxi_if_list, entry, prev) {
		if_entry = container_of(entry, struct cxi_if_list_entry, entry);
		if (if_entry->if_nic == src_addr->nic) {
			return if_entry;
		}
	}

	return NULL;
}

char *cxi_get_fabric_name(struct cxi_addr *src_addr)
{
	struct cxi_if_list_entry *if_entry;
	char *fab_name;
	int ret;

	if_entry = cxi_if_lookup(src_addr);
	if (!if_entry)
		return NULL;

	ret = asprintf(&fab_name, cxi_fab_fmt, if_entry->if_fabric);
	if (ret == -1)
		return NULL;

	return fab_name;
}

char *cxi_get_domain_name(struct cxi_addr *src_addr)
{
	struct cxi_if_list_entry *if_entry;
	char *dom_name;
	int ret;

	if_entry = cxi_if_lookup(src_addr);
	if (!if_entry)
		return NULL;

	ret = asprintf(&dom_name, cxi_dom_fmt, if_entry->if_idx,
		       src_addr->domain);
	if (ret == -1)
		return NULL;

	return dom_name;
}


void cxi_dom_add_to_list(struct cxi_domain *domain)
{
	fastlock_acquire(&cxi_list_lock);
	dlist_insert_tail(&domain->dom_list_entry, &cxi_dom_list);
	fastlock_release(&cxi_list_lock);
}

static inline int cxi_dom_check_list_internal(struct cxi_domain *domain)
{
	struct dlist_entry *entry;
	struct cxi_domain *dom_entry;

	for (entry = cxi_dom_list.next; entry != &cxi_dom_list;
	     entry = entry->next) {
		dom_entry = container_of(entry, struct cxi_domain,
					 dom_list_entry);
		if (dom_entry == domain)
			return 1;
	}

	return 0;
}

int cxi_dom_check_list(struct cxi_domain *domain)
{
	int found;

	fastlock_acquire(&cxi_list_lock);
	found = cxi_dom_check_list_internal(domain);
	fastlock_release(&cxi_list_lock);

	return found;
}

void cxi_dom_remove_from_list(struct cxi_domain *domain)
{
	fastlock_acquire(&cxi_list_lock);
	if (cxi_dom_check_list_internal(domain))
		dlist_remove(&domain->dom_list_entry);

	fastlock_release(&cxi_list_lock);
}

struct cxi_domain *cxi_dom_list_head(void)
{
	struct cxi_domain *domain;

	fastlock_acquire(&cxi_list_lock);
	if (dlist_empty(&cxi_dom_list)) {
		domain = NULL;
	} else {
		domain = container_of(cxi_dom_list.next,
				      struct cxi_domain, dom_list_entry);
	}
	fastlock_release(&cxi_list_lock);

	return domain;
}

int cxi_dom_check_manual_progress(struct cxi_fabric *fabric)
{
	struct dlist_entry *entry;
	struct cxi_domain *dom_entry;

	for (entry = cxi_dom_list.next; entry != &cxi_dom_list;
	     entry = entry->next) {
		dom_entry = container_of(entry, struct cxi_domain,
					 dom_list_entry);
		if (dom_entry->fab == fabric &&
		    dom_entry->progress_mode == FI_PROGRESS_MANUAL)
			return 1;
	}

	return 0;
}

void cxi_fab_add_to_list(struct cxi_fabric *fabric)
{
	fastlock_acquire(&cxi_list_lock);
	dlist_insert_tail(&fabric->fab_list_entry, &cxi_fab_list);
	fastlock_release(&cxi_list_lock);
}

static inline int cxi_fab_check_list_internal(struct cxi_fabric *fabric)
{
	struct dlist_entry *entry;
	struct cxi_fabric *fab_entry;

	for (entry = cxi_fab_list.next; entry != &cxi_fab_list;
	     entry = entry->next) {
		fab_entry = container_of(entry, struct cxi_fabric,
					 fab_list_entry);
		if (fab_entry == fabric)
			return 1;
	}

	return 0;
}

int cxi_fab_check_list(struct cxi_fabric *fabric)
{
	int found;

	fastlock_acquire(&cxi_list_lock);
	found = cxi_fab_check_list_internal(fabric);
	fastlock_release(&cxi_list_lock);

	return found;
}

void cxi_fab_remove_from_list(struct cxi_fabric *fabric)
{
	fastlock_acquire(&cxi_list_lock);
	if (cxi_fab_check_list_internal(fabric))
		dlist_remove(&fabric->fab_list_entry);

	fastlock_release(&cxi_list_lock);
}

struct cxi_fabric *cxi_fab_list_head(void)
{
	struct cxi_fabric *fabric;

	fastlock_acquire(&cxi_list_lock);
	if (dlist_empty(&cxi_fab_list)) {
		fabric = NULL;
	} else {
		fabric = container_of(cxi_fab_list.next,
				      struct cxi_fabric, fab_list_entry);
	}
	fastlock_release(&cxi_list_lock);

	return fabric;
}

int cxi_verify_fabric_attr(const struct fi_fabric_attr *attr)
{
	if (!attr)
		return 0;

	if (attr->prov_version) {
		if (attr->prov_version !=
		   FI_VERSION(CXI_MAJOR_VERSION, CXI_MINOR_VERSION))
			return -FI_ENODATA;
	}

	return 0;
}

int cxi_verify_info(uint32_t version, const struct fi_info *hints)
{
	uint64_t caps;
	enum fi_ep_type ep_type;
	int ret;
	struct cxi_domain *domain;
	struct cxi_fabric *fabric;

	if (!hints)
		return 0;

	ep_type = hints->ep_attr ? hints->ep_attr->type : FI_EP_UNSPEC;
	switch (ep_type) {
	case FI_EP_UNSPEC:
	case FI_EP_RDM:
		caps = CXI_EP_RDM_CAP;
		ret = cxi_rdm_verify_ep_attr(hints->ep_attr,
					      hints->tx_attr,
					      hints->rx_attr);
		break;
	default:
		ret = -FI_ENODATA;
	}
	if (ret)
		return ret;

	if ((caps | hints->caps) != caps) {
		CXI_LOG_DBG("Unsupported capabilities\n");
		return -FI_ENODATA;
	}

	switch (hints->addr_format) {
	case FI_FORMAT_UNSPEC:
	case FI_ADDR_CXI:
		break;
	default:
		CXI_LOG_DBG("Unsupported address format\n");
		return -FI_ENODATA;
	}

	if (hints->domain_attr && hints->domain_attr->domain) {
		domain = container_of(hints->domain_attr->domain,
				      struct cxi_domain, dom_fid);
		if (!cxi_dom_check_list(domain)) {
			CXI_LOG_DBG("no matching domain\n");
			return -FI_ENODATA;
		}
	}
	ret = cxi_verify_domain_attr(version, hints);
	if (ret)
		return ret;

	if (hints->fabric_attr && hints->fabric_attr->fabric) {
		fabric = container_of(hints->fabric_attr->fabric,
				      struct cxi_fabric, fab_fid);
		if (!cxi_fab_check_list(fabric)) {
			CXI_LOG_DBG("no matching fabric\n");
			return -FI_ENODATA;
		}
	}
	ret = cxi_verify_fabric_attr(hints->fabric_attr);
	if (ret)
		return ret;

	return 0;
}

static int cxi_trywait(struct fid_fabric *fabric, struct fid **fids, int count)
{
	return 0;
}

static struct fi_ops_fabric cxi_fab_ops = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = cxi_domain,
	.passive_ep = fi_no_passive_ep,
	.eq_open = fi_no_eq_open,
	.wait_open = fi_no_wait_open,
	.trywait = cxi_trywait
};

static int cxi_fabric_close(fid_t fid)
{
	struct cxi_fabric *fab;

	fab = container_of(fid, struct cxi_fabric, fab_fid);
	if (ofi_atomic_get32(&fab->ref))
		return -FI_EBUSY;

	cxi_fab_remove_from_list(fab);
	fastlock_destroy(&fab->lock);
	free(fab);

	return 0;
}

static struct fi_ops cxi_fab_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = cxi_fabric_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static void cxi_read_default_params(void)
{
	if (!read_default_params) {
		read_default_params = 1;
	}
}

static int cxi_fabric(struct fi_fabric_attr *attr,
		      struct fid_fabric **fabric, void *context)
{
	struct cxi_fabric *fab;

	if (slist_empty(&cxi_if_list)) {
		CXI_LOG_ERROR("Device not found\n");
		return -FI_ENODATA;
	}

	fab = calloc(1, sizeof(*fab));
	if (!fab)
		return -FI_ENOMEM;

	cxi_read_default_params();

	fastlock_init(&fab->lock);
	dlist_init(&fab->service_list);

	fab->fab_fid.fid.fclass = FI_CLASS_FABRIC;
	fab->fab_fid.fid.context = context;
	fab->fab_fid.fid.ops = &cxi_fab_fi_ops;
	fab->fab_fid.ops = &cxi_fab_ops;
	*fabric = &fab->fab_fid;
	ofi_atomic_initialize32(&fab->ref, 0);

	cxi_fab_add_to_list(fab);

	return 0;
}

int cxi_get_src_addr(struct cxi_addr *dest_addr, struct cxi_addr *src_addr)
{
	struct cxi_if_list_entry *if_entry;

	if_entry = cxi_if_lookup(dest_addr);
	if (if_entry) {
		/* Dest addr is loopback */
		memcpy(src_addr, dest_addr, sizeof(*src_addr));
		return 0;
	}

	/* TODO how to select an address on matching network? */

	/* Just say the first IF matches */
	if_entry = container_of((cxi_if_list.head), struct cxi_if_list_entry,
				entry);
	src_addr->nic = if_entry->if_nic;

	return 0;
}

static int cxi_fi_checkinfo(const struct fi_info *info,
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

static int cxi_parse_node(const char *node, uint32_t *nic)
{
	uint8_t scan_octets[6];
	uint32_t scan_nic;

	if (!node)
		return FI_SUCCESS;

	if (sscanf(node, "%hhx:%hhx:%hhx:%hhx:%hhx:%hhx%*c",
		   &scan_octets[5], &scan_octets[4], &scan_octets[3],
		   &scan_octets[2], &scan_octets[1], &scan_octets[0]) == 6) {
		/* TODO where is NIC addr emebedded in MAC? */
		*nic = scan_octets[0] | (scan_octets[1] << 8) |
				((scan_octets[2] & 0xF) << 16);
		return FI_SUCCESS;
	}

	if (sscanf(node, "%x", &scan_nic) == 1) {
		*nic = scan_nic;
		return FI_SUCCESS;
	}

	return -FI_ENODATA;
}

static int cxi_parse_service(const char *service, uint32_t *domain,
			     uint32_t *port)
{
	uint32_t scan_domain, scan_port;

	if (!service)
		return FI_SUCCESS;

	if (sscanf(service, "%d:%d", &scan_domain, &scan_port) == 2) {
		*domain = scan_domain;
		*port = scan_port;
		return FI_SUCCESS;
	}

	return -FI_ENODATA;
}

int cxi_parse_addr(const char *node, const char *service,
		   struct cxi_addr *addr)
{
	uint32_t nic = 0, domain = 0, port = 0;
	int ret;

	ret = cxi_parse_node(node, &nic);
	if (ret)
		return ret;

	ret = cxi_parse_service(service, &domain, &port);
	if (ret)
		return ret;

	addr->nic = nic;
	addr->domain = domain;
	addr->port = port;

	return FI_SUCCESS;
}

static int cxi_ep_getinfo(uint32_t version, const char *node,
			  const char *service, uint64_t flags,
			  const struct fi_info *hints, enum fi_ep_type ep_type,
			  struct fi_info **info)
{
	struct cxi_addr addr = CXI_ADDR_INIT,
			*src_addr = NULL, *dest_addr = NULL;
	int ret;

	if (flags & FI_SOURCE) {
		if (!node && !service)
			return -FI_ENODATA;

		src_addr = &addr;
		ret = cxi_parse_addr(node, service, src_addr);
		if (ret)
			return ret;

		if (hints && hints->dest_addr)
			dest_addr = hints->dest_addr;
	} else {
		if (node || service) {
			dest_addr = &addr;
			ret = cxi_parse_addr(node, service, dest_addr);
			if (ret)
				return ret;
		} else if (hints) {
			dest_addr = hints->dest_addr;
		}

		if (hints && hints->src_addr)
			src_addr = hints->src_addr;
	}

	if (dest_addr && !src_addr) {
		src_addr = &addr;
		cxi_get_src_addr(dest_addr, src_addr);
	}

	if (src_addr)
		CXI_LOG_DBG("src_addr: 0x%x:%u:%u\n",
			    src_addr->nic, src_addr->domain, src_addr->port);
	if (dest_addr)
		CXI_LOG_DBG("dest_addr: 0x%x:%u:%u\n",
			    dest_addr->nic, dest_addr->domain, dest_addr->port);

	switch (ep_type) {
	case FI_EP_RDM:
		ret = cxi_rdm_fi_info(version, src_addr, dest_addr, hints,
				      info);
		break;
	default:
		ret = -FI_ENODATA;
		break;
	}

	if (ret == 0)
		return cxi_fi_checkinfo(*info, hints);

	return ret;
}

int cxi_node_getinfo(uint32_t version, const char *node, const char *service,
		     uint64_t flags, const struct fi_info *hints,
		     struct fi_info **info, struct fi_info **tail)
{
	enum fi_ep_type ep_type;
	struct fi_info *cur;
	int ret;

	if (hints && hints->ep_attr) {
		switch (hints->ep_attr->type) {
		case FI_EP_RDM:
		case FI_EP_DGRAM:
		case FI_EP_MSG:
			ret = cxi_ep_getinfo(version, node, service, flags,
					      hints, hints->ep_attr->type,
					      &cur);
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
		ret = cxi_ep_getinfo(version, node, service, flags, hints,
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

static int cxi_match_src_addr_if(struct slist_entry *entry,
				const void *src_addr)
{
	struct cxi_if_list_entry *if_entry;

	if_entry = container_of(entry, struct cxi_if_list_entry, entry);

	return if_entry->if_nic == ((struct cxi_addr *)src_addr)->nic;
}

static int cxi_addr_matches_interface(struct slist *addr_list,
				      struct cxi_addr *src_addr)
{
	struct slist_entry *entry;

	entry = slist_find_first_match(addr_list, cxi_match_src_addr_if,
				       src_addr);

	return entry ? 1 : 0;
}

static int cxi_node_matches_interface(struct slist *if_list, const char *node)
{
	uint32_t nic;
	struct cxi_addr addr;
	int ret;

	ret = cxi_parse_node(node, &nic);
	if (ret)
		return 0;

	addr.nic = nic;

	return cxi_addr_matches_interface(if_list, &addr);
}

#define CXI_FAKE_NICS 1
#define NUM_CXI_FAKE_NICS 4
static struct cxi_if_list_entry cxi_fake_nics[] = {
	{ .if_nic = 0xabcd0, .if_idx = 0, .if_fabric = 2 },
	{ .if_nic = 0xabcd1, .if_idx = 1, .if_fabric = 2 },
	{ .if_nic = 0xabcd2, .if_idx = 2, .if_fabric = 3 },
	{ .if_nic = 0xabcd3, .if_idx = 3, .if_fabric = 3 }
};

void cxi_get_list_of_if(struct slist *if_list)
{
	struct cxi_if_list_entry *if_entry;

	/* TODO populate list with all NICs, how do we poll local NICs? */

#if CXI_FAKE_NICS
	/* Make up some NICs for now */
	for (int i = 0; i < NUM_CXI_FAKE_NICS; i++) {
		if_entry = calloc(1, sizeof(struct cxi_if_list_entry));
		if_entry->if_nic = cxi_fake_nics[i].if_nic;
		if_entry->if_idx = cxi_fake_nics[i].if_idx;
		if_entry->if_fabric = cxi_fake_nics[i].if_fabric;
		slist_insert_tail(&if_entry->entry, if_list);
	}
#endif
}

static void cxi_free_if_list(struct slist *if_list)
{
	struct slist_entry *entry;
	struct cxi_if_list_entry *if_entry;

	while (!slist_empty(if_list)) {
		entry = slist_remove_head(if_list);
		if_entry = container_of(entry, struct cxi_if_list_entry,
					   entry);
		free(if_entry);
	}
}

static int cxi_getinfo(uint32_t version, const char *node, const char *service,
		       uint64_t flags, const struct fi_info *hints,
		       struct fi_info **info)
{
	int ret = 0;
	struct slist_entry *entry, *prev;
	struct cxi_if_list_entry *if_entry;
	struct fi_info *tail;

	if (slist_empty(&cxi_if_list)) {
		CXI_LOG_ERROR("Device not found\n");
		return -FI_ENODATA;
	}

	if (!(flags & FI_SOURCE) && hints && hints->src_addr &&
	    (hints->src_addrlen != sizeof(struct cxi_addr)))
		return -FI_ENODATA;

	if (((!node && !service) || (flags & FI_SOURCE)) &&
	    hints && hints->dest_addr &&
	    (hints->dest_addrlen != sizeof(struct cxi_addr)))
		return -FI_ENODATA;

	ret = cxi_verify_info(version, hints);
	if (ret)
		return ret;

	ret = 1;
	if ((flags & FI_SOURCE) && node) {
		ret = cxi_node_matches_interface(&cxi_if_list, node);
	} else if (hints && hints->src_addr) {
		ret = cxi_addr_matches_interface(&cxi_if_list,
				(struct cxi_addr *)hints->src_addr);
	}
	if (!ret) {
		CXI_LOG_ERROR("Couldn't find a match with local interfaces\n");
		return -FI_ENODATA;
	}

	*info = tail = NULL;
	if (node ||
	     (!(flags & FI_SOURCE) && hints && hints->src_addr) ||
	     (!(flags & FI_SOURCE) && hints && hints->dest_addr))
		return cxi_node_getinfo(version, node, service, flags,
					 hints, info, &tail);

	(void) prev; /* Makes compiler happy */
	slist_foreach(&cxi_if_list, entry, prev) {
		char *local_node;

		if_entry = container_of(entry, struct cxi_if_list_entry, entry);
		ret = asprintf(&local_node, "0x%x", if_entry->if_nic);
		if (ret == -1) {
			CXI_LOG_ERROR("asprintf failed: %s\n",
				      strerror(ofi_syserr()));
			local_node = NULL;
		}

		flags |= FI_SOURCE;
		ret = cxi_node_getinfo(version, local_node, service, flags,
				       hints, info, &tail);
		free(local_node);

		if (ret) {
			if (ret == -FI_ENODATA)
				continue;
			return ret;
		}
	}

	return (!*info) ? ret : 0;
}

static void fi_cxi_fini(void)
{
	cxi_free_if_list(&cxi_if_list);
	fastlock_destroy(&cxi_list_lock);
}

struct fi_provider cxi_prov = {
	.name = cxi_prov_name,
	.version = FI_VERSION(CXI_MAJOR_VERSION, CXI_MINOR_VERSION),
	.fi_version = FI_VERSION(1, 6),
	.getinfo = cxi_getinfo,
	.fabric = cxi_fabric,
	.cleanup = fi_cxi_fini
};

CXI_INI
{
	fastlock_init(&cxi_list_lock);
	dlist_init(&cxi_fab_list);
	dlist_init(&cxi_dom_list);
	slist_init(&cxi_if_list);

	cxi_get_list_of_if(&cxi_if_list);

	return &cxi_prov;
}
