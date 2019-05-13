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

#include "cxip.h"

#define CXIP_LOG_DBG(...) _CXIP_LOG_DBG(FI_LOG_FABRIC, __VA_ARGS__)
#define CXIP_LOG_ERROR(...) _CXIP_LOG_ERROR(FI_LOG_FABRIC, __VA_ARGS__)

const char cxip_fab_fmt[] = "cxi/%d"; /* Provder/Net Name */
const char cxip_dom_fmt[] = "cxi%d"; /* IF Index Name */
const char cxip_prov_name[] = "CXI"; /* Provider Name */

int cxip_av_def_sz = CXIP_AV_DEF_SZ;
int cxip_cq_def_sz = CXIP_CQ_DEF_SZ;
int cxip_eq_def_sz = CXIP_EQ_DEF_SZ;

uint64_t CXIP_EP_RDM_SEC_CAP = CXIP_EP_RDM_SEC_CAP_BASE;
uint64_t CXIP_EP_RDM_CAP = CXIP_EP_RDM_CAP_BASE;

const struct fi_fabric_attr cxip_fabric_attr = {
	.fabric = NULL,
	.name = NULL,
	.prov_name = NULL,
	.prov_version = FI_VERSION(CXIP_MAJOR_VERSION, CXIP_MINOR_VERSION),
};

static struct dlist_entry cxip_fab_list;
static struct dlist_entry cxip_dom_list;
static fastlock_t cxip_list_lock;
static int read_default_params;

char *cxip_get_fabric_name(struct cxip_addr *src_addr)
{
	struct cxip_if *if_entry;
	char *fab_name;
	int ret;

	if_entry = cxip_if_lookup(src_addr->nic);
	if (!if_entry)
		return NULL;

	ret = asprintf(&fab_name, cxip_fab_fmt, if_entry->if_fabric);
	if (ret == -1)
		return NULL;

	return fab_name;
}

char *cxip_get_domain_name(struct cxip_addr *src_addr)
{
	struct cxip_if *if_entry;
	char *dom_name;
	int ret;

	if_entry = cxip_if_lookup(src_addr->nic);
	if (!if_entry)
		return NULL;

	ret = asprintf(&dom_name, cxip_dom_fmt, if_entry->if_idx);
	if (ret == -1)
		return NULL;

	return dom_name;
}

void cxip_dom_add_to_list(struct cxip_domain *domain)
{
	fastlock_acquire(&cxip_list_lock);
	dlist_insert_tail(&domain->dom_list_entry, &cxip_dom_list);
	fastlock_release(&cxip_list_lock);
}

static inline int cxip_dom_check_list_internal(struct cxip_domain *domain)
{
	struct dlist_entry *entry;
	struct cxip_domain *dom_entry;

	for (entry = cxip_dom_list.next; entry != &cxip_dom_list;
	     entry = entry->next) {
		dom_entry =
			container_of(entry, struct cxip_domain, dom_list_entry);
		if (dom_entry == domain)
			return 1;
	}

	return 0;
}

int cxip_dom_check_list(struct cxip_domain *domain)
{
	int found;

	fastlock_acquire(&cxip_list_lock);
	found = cxip_dom_check_list_internal(domain);
	fastlock_release(&cxip_list_lock);

	return found;
}

void cxip_dom_remove_from_list(struct cxip_domain *domain)
{
	fastlock_acquire(&cxip_list_lock);
	if (cxip_dom_check_list_internal(domain))
		dlist_remove(&domain->dom_list_entry);

	fastlock_release(&cxip_list_lock);
}

struct cxip_domain *cxip_dom_list_head(void)
{
	struct cxip_domain *domain;

	fastlock_acquire(&cxip_list_lock);
	if (dlist_empty(&cxip_dom_list)) {
		domain = NULL;
	} else {
		domain = container_of(cxip_dom_list.next, struct cxip_domain,
				      dom_list_entry);
	}
	fastlock_release(&cxip_list_lock);

	return domain;
}

int cxip_dom_check_manual_progress(struct cxip_fabric *fabric)
{
	struct dlist_entry *entry;
	struct cxip_domain *dom_entry;

	for (entry = cxip_dom_list.next; entry != &cxip_dom_list;
	     entry = entry->next) {
		dom_entry =
			container_of(entry, struct cxip_domain, dom_list_entry);
		if (dom_entry->fab == fabric &&
		    dom_entry->progress_mode == FI_PROGRESS_MANUAL)
			return 1;
	}

	return 0;
}

void cxip_fab_add_to_list(struct cxip_fabric *fabric)
{
	fastlock_acquire(&cxip_list_lock);
	dlist_insert_tail(&fabric->fab_list_entry, &cxip_fab_list);
	fastlock_release(&cxip_list_lock);
}

static inline int cxip_fab_check_list_internal(struct cxip_fabric *fabric)
{
	struct dlist_entry *entry;
	struct cxip_fabric *fab_entry;

	for (entry = cxip_fab_list.next; entry != &cxip_fab_list;
	     entry = entry->next) {
		fab_entry =
			container_of(entry, struct cxip_fabric, fab_list_entry);
		if (fab_entry == fabric)
			return 1;
	}

	return 0;
}

int cxip_fab_check_list(struct cxip_fabric *fabric)
{
	int found;

	fastlock_acquire(&cxip_list_lock);
	found = cxip_fab_check_list_internal(fabric);
	fastlock_release(&cxip_list_lock);

	return found;
}

void cxip_fab_remove_from_list(struct cxip_fabric *fabric)
{
	fastlock_acquire(&cxip_list_lock);
	if (cxip_fab_check_list_internal(fabric))
		dlist_remove(&fabric->fab_list_entry);

	fastlock_release(&cxip_list_lock);
}

struct cxip_fabric *cxip_fab_list_head(void)
{
	struct cxip_fabric *fabric;

	fastlock_acquire(&cxip_list_lock);
	if (dlist_empty(&cxip_fab_list)) {
		fabric = NULL;
	} else {
		fabric = container_of(cxip_fab_list.next, struct cxip_fabric,
				      fab_list_entry);
	}
	fastlock_release(&cxip_list_lock);

	return fabric;
}

int cxip_verify_fabric_attr(const struct fi_fabric_attr *attr)
{
	if (!attr)
		return 0;

	if (attr->prov_version) {
		if (attr->prov_version !=
		    FI_VERSION(CXIP_MAJOR_VERSION, CXIP_MINOR_VERSION)) {
			CXIP_LOG_DBG("Provider version unsupported\n");
			return -FI_ENODATA;
		}
	}

	return 0;
}

int cxip_verify_info(uint32_t version, const struct fi_info *hints)
{
	uint64_t caps;
	enum fi_ep_type ep_type;
	int ret;
	struct cxip_domain *domain;
	struct cxip_fabric *fabric;

	if (!hints)
		return 0;

	ep_type = hints->ep_attr ? hints->ep_attr->type : FI_EP_UNSPEC;
	switch (ep_type) {
	case FI_EP_UNSPEC:
	case FI_EP_RDM:
		caps = CXIP_EP_RDM_CAP;
		ret = cxip_rdm_verify_ep_attr(hints->ep_attr, hints->tx_attr,
					      hints->rx_attr);
		break;
	default:
		CXIP_LOG_DBG("Unsupported endpoint type\n");
		ret = -FI_ENODATA;
	}
	if (ret)
		return ret;

	if ((caps | hints->caps) != caps) {
		CXIP_LOG_DBG("Unsupported capabilities\n");
		return -FI_ENODATA;
	}

	switch (hints->addr_format) {
	case FI_FORMAT_UNSPEC:
	case FI_ADDR_CXI:
		if (hints->src_addr &&
		    hints->src_addrlen != sizeof(struct cxip_addr))
			return -FI_EINVAL;
		if (hints->dest_addr &&
		    hints->dest_addrlen != sizeof(struct cxip_addr))
			return -FI_EINVAL;
		break;
	default:
		CXIP_LOG_DBG("Unsupported address format\n");
		return -FI_ENODATA;
	}

	if (hints->domain_attr && hints->domain_attr->domain) {
		domain = container_of(hints->domain_attr->domain,
				      struct cxip_domain, dom_fid);
		if (!cxip_dom_check_list(domain)) {
			CXIP_LOG_DBG("no matching domain\n");
			return -FI_ENODATA;
		}
	}
	ret = cxip_verify_domain_attr(version, hints);
	if (ret)
		return ret;

	if (hints->fabric_attr && hints->fabric_attr->fabric) {
		fabric = container_of(hints->fabric_attr->fabric,
				      struct cxip_fabric, fab_fid);
		if (!cxip_fab_check_list(fabric)) {
			CXIP_LOG_DBG("no matching fabric\n");
			return -FI_ENODATA;
		}
	}
	ret = cxip_verify_fabric_attr(hints->fabric_attr);
	if (ret)
		return ret;

	return 0;
}

static int cxip_trywait(struct fid_fabric *fabric, struct fid **fids, int count)
{
	return 0;
}

static struct fi_ops_fabric cxip_fab_ops = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = cxip_domain,
	.passive_ep = fi_no_passive_ep,
	.eq_open = fi_no_eq_open,
	.wait_open = fi_no_wait_open,
	.trywait = cxip_trywait
};

static int cxip_fabric_close(fid_t fid)
{
	struct cxip_fabric *fab;

	fab = container_of(fid, struct cxip_fabric, fab_fid);
	if (ofi_atomic_get32(&fab->ref))
		return -FI_EBUSY;

	cxip_fab_remove_from_list(fab);
	fastlock_destroy(&fab->lock);
	ofi_fabric_close(&fab->util_fabric);
	free(fab);

	return 0;
}

static struct fi_ops cxip_fab_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = cxip_fabric_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static void cxip_read_default_params(void)
{
	if (!read_default_params)
		read_default_params = 1;
}

static int cxip_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
		       void *context)
{
	struct cxip_fabric *fab;
	int ret;

	if (slist_empty(&cxip_if_list)) {
		CXIP_LOG_ERROR("Device not found\n");
		return -FI_ENODATA;
	}

	fab = calloc(1, sizeof(*fab));
	if (!fab)
		return -FI_ENOMEM;

	ret = ofi_fabric_init(&cxip_prov, &cxip_fabric_attr, attr,
			      &fab->util_fabric, context);
	if (ret != FI_SUCCESS)
		goto free_fab;

	cxip_read_default_params();

	fastlock_init(&fab->lock);
	dlist_init(&fab->service_list);

	fab->fab_fid.fid.fclass = FI_CLASS_FABRIC;
	fab->fab_fid.fid.context = context;
	fab->fab_fid.fid.ops = &cxip_fab_fi_ops;
	fab->fab_fid.ops = &cxip_fab_ops;
	*fabric = &fab->fab_fid;
	ofi_atomic_initialize32(&fab->ref, 0);

	cxip_fab_add_to_list(fab);

	return 0;

free_fab:
	free(fab);
	return ret;
}

int cxip_get_src_addr(struct cxip_addr *dest_addr, struct cxip_addr *src_addr)
{
	struct cxip_if *if_entry;

	/* TODO how to select an address on matching network? */

	/* Just say the first IF matches */
	if_entry = container_of((cxip_if_list.head), struct cxip_if, if_entry);
	src_addr->nic = if_entry->if_nic;
	src_addr->pid = C_PID_ANY;

	return 0;
}

static int cxip_fi_checkinfo(const struct fi_info *info,
			     const struct fi_info *hints)
{
	if (hints && hints->domain_attr && hints->domain_attr->name &&
	    strcmp(info->domain_attr->name, hints->domain_attr->name)) {
		CXIP_LOG_DBG("Domain name mismatch\n");
		return -FI_ENODATA;
	}

	if (hints && hints->fabric_attr && hints->fabric_attr->name &&
	    strcmp(info->fabric_attr->name, hints->fabric_attr->name)) {
		CXIP_LOG_DBG("Fabric name mismatch\n");
		return -FI_ENODATA;
	}

	return 0;
}

static int cxip_parse_node(const char *node, uint32_t *nic)
{
	uint8_t scan_octets[6];
	uint32_t scan_nic;

	if (!node)
		return FI_SUCCESS;

	if (sscanf(node, "%hhx:%hhx:%hhx:%hhx:%hhx:%hhx%*c", &scan_octets[5],
		   &scan_octets[4], &scan_octets[3], &scan_octets[2],
		   &scan_octets[1], &scan_octets[0]) == 6) {
		/* TODO where is NIC addr embedded in MAC? */
		*nic = scan_octets[0] | (scan_octets[1] << 8) |
		       ((scan_octets[2] & 0xF) << 16);
		return FI_SUCCESS;
	}

	if (sscanf(node, "%i", &scan_nic) == 1) {
		*nic = scan_nic;
		return FI_SUCCESS;
	}

	return -FI_ENODATA;
}

static int cxip_parse_service(const char *service, uint32_t *pid)
{
	uint32_t scan_pid;

	if (sscanf(service, "%d", &scan_pid) == 1) {
		*pid = scan_pid;
		return FI_SUCCESS;
	}

	return -FI_ENODATA;
}

int cxip_parse_addr(const char *node, const char *service,
		    struct cxip_addr *addr)
{
	uint32_t nic = 0;
	uint32_t pid = C_PID_ANY;
	int ret;

	ret = cxip_parse_node(node, &nic);
	if (ret)
		return ret;

	if (service) {
		ret = cxip_parse_service(service, &pid);
		if (ret)
			return ret;
	}

	addr->nic = nic;
	addr->pid = pid;

	return FI_SUCCESS;
}

static int cxip_ep_getinfo(uint32_t version, const char *node,
			   const char *service, uint64_t flags,
			   const struct fi_info *hints, enum fi_ep_type ep_type,
			   struct fi_info **info)
{
	struct cxip_addr saddr = {};
	struct cxip_addr daddr = {};
	struct cxip_addr *src_addr = NULL;
	struct cxip_addr *dest_addr = NULL;
	int ret;

	if (flags & FI_SOURCE) {
		if (!node && !service) {
			CXIP_LOG_DBG("FI_SOURCE without node and service!\n");
			return -FI_ENODATA;
		}

		src_addr = &saddr;
		ret = cxip_parse_addr(node, service, src_addr);
		if (ret) {
			CXIP_LOG_DBG("Unable to parse src_addr!\n");
			return ret;
		}

		if (hints && hints->dest_addr)
			dest_addr = hints->dest_addr;
	} else {
		if (node || service) {
			dest_addr = &daddr;
			ret = cxip_parse_addr(node, service, dest_addr);
			if (ret) {
				CXIP_LOG_DBG("Unable to parse dest_addr!\n");
				return ret;
			}
		} else if (hints) {
			dest_addr = hints->dest_addr;
		}

		if (hints && hints->src_addr)
			src_addr = hints->src_addr;
	}

	if (dest_addr && !src_addr) {
		src_addr = &saddr;
		cxip_get_src_addr(dest_addr, src_addr);
	}

	CXIP_LOG_DBG("node: %s service: %s\n", node, service);

	if (src_addr)
		CXIP_LOG_DBG("src_addr: 0x%x:%d\n", src_addr->nic,
			     src_addr->pid);
	if (dest_addr)
		CXIP_LOG_DBG("dest_addr: 0x%x:%d\n", dest_addr->nic,
			      dest_addr->pid);

	switch (ep_type) {
	case FI_EP_RDM:
		ret = cxip_rdm_fi_info(version, src_addr, dest_addr, hints,
				       info);
		break;
	default:
		CXIP_LOG_DBG("Invalid ep type %d\n", ep_type);
		ret = -FI_ENODATA;
		break;
	}

	if (ret == 0)
		return cxip_fi_checkinfo(*info, hints);

	return ret;
}

int cxip_node_getinfo(uint32_t version, const char *node, const char *service,
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
			ret = cxip_ep_getinfo(version, node, service, flags,
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
		ret = cxip_ep_getinfo(version, node, service, flags, hints,
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

static int cxip_match_src_addr_if(struct slist_entry *entry,
				  const void *src_addr)
{
	struct cxip_if *if_entry;

	if_entry = container_of(entry, struct cxip_if, if_entry);

	return if_entry->if_nic == ((struct cxip_addr *)src_addr)->nic;
}

static int cxip_addr_matches_interface(struct slist *addr_list,
				       struct cxip_addr *src_addr)
{
	struct slist_entry *entry;

	entry = slist_find_first_match(addr_list, cxip_match_src_addr_if,
				       src_addr);

	return entry ? 1 : 0;
}

static int cxip_node_matches_interface(struct slist *if_list, const char *node)
{
	uint32_t nic;
	struct cxip_addr addr;
	int ret;

	ret = cxip_parse_node(node, &nic);
	if (ret)
		return 0;

	addr.nic = nic;

	return cxip_addr_matches_interface(if_list, &addr);
}

static int cxip_getinfo(uint32_t version, const char *node, const char *service,
			uint64_t flags, const struct fi_info *hints,
			struct fi_info **info)
{
	int ret = 0;
	struct slist_entry *entry, *prev __attribute__((unused));
	struct cxip_if *if_entry;
	struct fi_info *tail;

	if (slist_empty(&cxip_if_list)) {
		CXIP_LOG_ERROR("Device not found\n");
		return -FI_ENODATA;
	}

	if (!(flags & FI_SOURCE) && hints && hints->src_addr &&
	    (hints->src_addrlen != sizeof(struct cxip_addr))) {
		CXIP_LOG_ERROR("Invalid Src Address length\n");
		return -FI_ENODATA;
	}

	if (((!node && !service) || (flags & FI_SOURCE)) && hints &&
	    hints->dest_addr &&
	    (hints->dest_addrlen != sizeof(struct cxip_addr))) {
		CXIP_LOG_ERROR("Invalid parameter set\n");
		return -FI_ENODATA;
	}

	ret = cxip_verify_info(version, hints);
	if (ret)
		return ret;

	ret = 1;
	if ((flags & FI_SOURCE) && node) {
		ret = cxip_node_matches_interface(&cxip_if_list, node);
	} else if (hints && hints->src_addr) {
		ret = cxip_addr_matches_interface(
			&cxip_if_list, (struct cxip_addr *)hints->src_addr);
	}
	if (!ret) {
		CXIP_LOG_ERROR("Couldn't find a match with local interfaces\n");
		return -FI_ENODATA;
	}

	*info = tail = NULL;
	if (node || (!(flags & FI_SOURCE) && hints && hints->src_addr) ||
	    (!(flags & FI_SOURCE) && hints && hints->dest_addr))
		return cxip_node_getinfo(version, node, service, flags, hints,
					 info, &tail);

	slist_foreach(&cxip_if_list, entry, prev) {
		char *local_node;

		if_entry = container_of(entry, struct cxip_if, if_entry);
		ret = asprintf(&local_node, "0x%x", if_entry->if_nic);
		if (ret == -1) {
			CXIP_LOG_ERROR("asprintf failed: %s\n",
				       strerror(ofi_syserr()));
			local_node = NULL;
		}

		flags |= FI_SOURCE;
		if (service) {
			ret = cxip_node_getinfo(version, local_node, service,
						flags, hints, info, &tail);
		} else {
			ret = cxip_node_getinfo(version, local_node,
						NULL, flags,
						hints, info, &tail);

			if (ret && ret != -FI_ENODATA)
				return ret;
		}
		free(local_node);

		if (ret) {
			if (ret == -FI_ENODATA)
				continue;
			return ret;
		}
	}

	return (!*info) ? ret : 0;
}

static void fi_cxip_fini(void)
{
	cxip_fault_inject_fini();

	cxip_if_fini();

	fastlock_destroy(&cxip_list_lock);
}

struct fi_provider cxip_prov = {
	.name = cxip_prov_name,
	.version = FI_VERSION(CXIP_MAJOR_VERSION, CXIP_MINOR_VERSION),
	.fi_version = FI_VERSION(1, 7),
	.getinfo = cxip_getinfo,
	.fabric = cxip_fabric,
	.cleanup = fi_cxip_fini
};

CXI_INI
{
	fastlock_init(&cxip_list_lock);
	dlist_init(&cxip_fab_list);
	dlist_init(&cxip_dom_list);

	cxip_if_init();

	cxip_fault_inject_init();

	return &cxip_prov;
}
