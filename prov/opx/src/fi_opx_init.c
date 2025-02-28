/*
 * Copyright (C) 2016 by Argonne National Laboratory.
 * Copyright (C) 2021-2025 Cornelis Networks.
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
#include <ofi.h>
#include <ofi_mem.h>

#include "rdma/opx/fi_opx.h"
#include "rdma/opx/fi_opx_internal.h"
#include "rdma/opx/fi_opx_hfi1.h"
#include "rdma/opx/fi_opx_domain.h"
#include "rdma/opx/fi_opx_hmem.h"
#include "ofi_prov.h"
#include "opa_service.h"
#include "rdma/opx/fi_opx_hfi1_version.h"

#include "rdma/opx/fi_opx_addr.h"

#include "rdma/opx/opx_tracer.h"

#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>

#include "fi_opx_tid_cache.h"
#include "opx_hmem_cache.h"

union fi_opx_addr opx_default_addr = {
	.hfi1_subctxt_rx = 0xffff,
	.hfi1_unit	 = 0xff,
	.lid		 = 0xffffff,
};

static int fi_opx_init;
static int fi_opx_count;

int fi_opx_check_info(const struct fi_info *info)
{
	int ret;
	/* TODO: check mode */

	/* Checking the general capabilities. OPX will bow out if it cannot support any requested primary or secondary
	 * caps */
	if (info->caps == 0) {
		FI_LOG(fi_opx_global.prov, FI_LOG_DEBUG, FI_LOG_FABRIC,
		       "The application's capability hints are null. OPX is allowed to specify whatever capabilities it wishes\n");
	} else {
		// Check to make sure the hinted capabilites are a subset of what OPX can support
		if ((info->caps & FI_OPX_SUPPORTED_CAPS) != info->caps) {
			FI_WARN(fi_opx_global.prov, FI_LOG_FABRIC,
				"OPX does not support the requested capabilites required by the application\n");
			goto err;
		}
	}

	if (info->domain_attr == NULL) {
		FI_LOG(fi_opx_global.prov, FI_LOG_DEBUG, FI_LOG_FABRIC,
		       "The domain_attr structure is null, there must be an issue in provider initialization\n");
		goto err;
	}

	if (info->caps & FI_HMEM) {
		/* Add FI_MR_HMEM to mr_mode when claiming support of FI_HMEM
		 * because OPX provider's HMEM support performance relies on
		 * application to provide descriptor for device buffer.
		 */
		if (info->domain_attr && !(info->domain_attr->mr_mode & FI_MR_HMEM)) {
			FI_WARN(fi_opx_global.prov, FI_LOG_MR,
				"FI_HMEM capability requires device registrations (FI_MR_HMEM)\n");
			goto err;
		}
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			     "FI_HMEM capability has been successfully enforced by OPX\n");
	}

	switch (info->addr_format) {
	case FI_ADDR_OPX:
	case FI_FORMAT_UNSPEC:
		break;
	default:
		FI_LOG(fi_opx_global.prov, FI_LOG_DEBUG, FI_LOG_FABRIC, "unavailable [bad info->addr_format (%u)]\n",
		       info->addr_format);
		goto err;
	}

	if (info->tx_attr) {
		ret = fi_opx_check_tx_attr(info->tx_attr, info->caps);
		if (ret) {
			return ret;
		}
	}

	if (info->rx_attr) {
		ret = fi_opx_check_rx_attr(info->rx_attr, info->caps);
		if (ret) {
			return ret;
		}
	}

	if (info->ep_attr) {
		ret = fi_opx_check_ep_attr(info->ep_attr);
		if (ret) {
			return ret;
		}
	}

	if (info->domain_attr) {
		ret = fi_opx_check_domain_attr(info->domain_attr);
		if (ret) {
			return ret;
		}
	}

	if (info->fabric_attr) {
		ret = fi_opx_check_fabric_attr(info->fabric_attr);
		if (ret) {
			return ret;
		}
	}

	return 0;

err:

	errno = FI_ENODATA;
	return -errno;
}

static int fi_opx_fillinfo(struct fi_info *fi, const char *node, const char *service, const struct fi_info *hints,
			   uint64_t flags, enum fi_progress progress)
{
	int		   ret;
	union fi_opx_addr *addr;
	uint32_t	   fmt;
	size_t		   len;

	if (!fi) {
		goto err;
	}

	if (!hints && !node && !service) {
		goto err;
	}

	fi->next = NULL;

	/* As a general rule, specifying a non-zero value for input hints indicates that
	   a provider must support the requested value or fail the operation with -FI_ENODATA. */

	if (hints && hints->caps) {
		/*  In the function fi_opx_check_info, we ensure that hints is a subset of what OPX supports.
			We would have already returned a bad return code to the getinfo call if OPX could not support
		   what the app is requesting. Thus, we are safe to assume we can support whatever the cap hints are
		   asking for. */
		fi->caps = hints->caps;
	} else {
		/* A zeroed hint value results in providers either returning a default value or a value that works best
		 * for their implementation */
		/* TODO -> Make sure these Default capabilities are up-to-date and reflect our best performance */
		fi->caps = FI_OPX_DEFAULT_CAPS;
	}

	/* set the mode that we require */
	fi->mode = FI_ASYNC_IOV;

	fi->addr_format	 = FI_ADDR_OPX;
	fi->src_addrlen	 = 0;
	fi->dest_addrlen = 0;
	fi->src_addr	 = NULL;
	fi->dest_addr	 = NULL;

	// Process the node field. Service is treated identically to node.
	if (node) {
		if (!ofi_str_toaddr(node, &fmt, (void **) &addr, &len) && fmt == FI_ADDR_OPX) {
			if (flags & FI_SOURCE) {
				fi->src_addr	= addr;
				fi->src_addrlen = sizeof(union fi_opx_addr);
				FI_INFO(fi_opx_global.prov, FI_LOG_FABRIC, "'%s' is taken as src_addr.\n", node);
			} else {
				fi->dest_addr	 = addr;
				fi->dest_addrlen = sizeof(union fi_opx_addr);
				FI_INFO(fi_opx_global.prov, FI_LOG_FABRIC, "'%s' is taken as dest_addr.\n", node);
			}
			node = NULL;
		} else {
			FI_WARN(fi_opx_global.prov, FI_LOG_FABRIC, "'%s' is not a valid OPX address.\n", node);
			goto err;
		}
	}

	if (hints) {
		if (hints->dest_addr) {
			FI_LOG(fi_opx_global.prov, FI_LOG_DEBUG, FI_LOG_FABRIC,
			       "cannot support dest_addr lookups now\n");
			goto err;
		}

		if (hints->src_addr) {
			fi->src_addr = mem_dup(hints->src_addr, hints->src_addrlen);
			if (!fi->src_addr) {
				FI_WARN(fi_opx_global.prov, FI_LOG_FABRIC, "Failed to alloc memory.\n");
				goto err;
			}
			fi->src_addrlen = sizeof(union fi_opx_addr);
		}
	}

	if (!fi->src_addr) {
		fi->src_addr = mem_dup(&opx_default_addr, sizeof(opx_default_addr));
		if (!fi->src_addr) {
			FI_WARN(fi_opx_global.prov, FI_LOG_FABRIC, "Failed to alloc memory.\n");
			goto err;
		}
		fi->src_addrlen = sizeof(union fi_opx_addr);
	}

	if ((hints != NULL) && (hints->dest_addr != NULL) &&
	    (((node == NULL) && (service == NULL)) || (flags & FI_SOURCE))) {
		/*
		 * man/fi_getinfo.3
		 *
		 * dest_addr - destination address
		 * If specified, indicates the destination address. This field
		 * will be ignored in hints unless the node and service
		 * parameters are NULL or FI_SOURCE flag is set. If FI_SOURCE
		 * is not specified, on output a provider shall return an
		 * address the corresponds to the indicated node and/or service
		 * fields, relative to the fabric and domain. Note that any
		 * returned address is only usable locally.
		 */

		if ((flags & FI_SOURCE) == 0) {
			if ((hints->addr_format != FI_FORMAT_UNSPEC) && (hints->addr_format != FI_ADDR_OPX)) {
				FI_WARN(fi_opx_global.prov, FI_LOG_FABRIC, "invalid addr_format hint (%d)\n",
					hints->addr_format);
				errno = FI_EINVAL;
				goto err;
			}
			fi->dest_addr = mem_dup(hints->dest_addr, hints->dest_addrlen);
			if (!fi->dest_addr) {
				FI_WARN(fi_opx_global.prov, FI_LOG_FABRIC, "Failed to alloc memory.\n");
				goto err;
			}
			fi->dest_addrlen = sizeof(union fi_opx_addr);
		}
	}

	/*
	 * man/fi_fabric.3
	 *
	 * On input to fi_getinfo, a user may set this (fi_fabric_attr::fabric)
	 * to an opened fabric instance to restrict output to the given fabric.
	 * On output from fi_getinfo, if no fabric was specified, but the user
	 * has an opened instance of the named fabric, this (fi_fabric_attr::fabric)
	 * will reference the first opened instance. If no instance has been
	 * opened, this field will be NULL.
	 */

	fi->fabric_attr->name = strdup(FI_OPX_FABRIC_NAME);
	if (!fi->fabric_attr->name) {
		FI_LOG(fi_opx_global.prov, FI_LOG_DEBUG, FI_LOG_FABRIC, "memory allocation failed");
		goto err;
	}

	fi->fabric_attr->prov_version = FI_OPX_PROVIDER_VERSION;

	if (fi_opx_global.default_tx_attr == NULL) {
		if (fi_opx_alloc_default_tx_attr(&fi_opx_global.default_tx_attr)) {
			FI_DBG(fi_opx_global.prov, FI_LOG_DOMAIN,
			       "alloc function could not allocate block of memory\n");
			errno = FI_ENOMEM;
			goto err;
		}
	}

	memcpy(fi->tx_attr, fi_opx_global.default_tx_attr, sizeof(*fi->tx_attr));
	if (hints && hints->tx_attr) {
		/*
		 * man/fi_endpoint.3
		 *
		 *   fi_tx_attr::caps
		 *
		 *   "... If the caps field is 0 on input to fi_getinfo(3), the
		 *   caps value from the fi_info structure will be used."
		 */
		if (hints->tx_attr->caps) {
			fi->tx_attr->caps = hints->tx_attr->caps;
		} else {
			/* fi->caps is either:
			   1) The caps value from the fi_info hints->caps structure
			   2) The OPX defaults, which happens if the fi_info hints->caps is null
			*/
			fi->tx_attr->caps = fi->caps;
		}

		/* adjust parameters down from what requested if required */
		fi->tx_attr->op_flags = hints->tx_attr->op_flags;
	} else if (hints && hints->caps) {
		fi->tx_attr->caps = hints->caps;
	}

	if (fi_opx_global.default_rx_attr == NULL) {
		if (fi_opx_alloc_default_rx_attr(&fi_opx_global.default_rx_attr)) {
			FI_DBG(fi_opx_global.prov, FI_LOG_DOMAIN,
			       "alloc function could not allocate block of memory\n");
			errno = FI_ENOMEM;
			goto err;
		}
	}
	memcpy(fi->rx_attr, fi_opx_global.default_rx_attr, sizeof(*fi->rx_attr));
	if (hints && hints->rx_attr) {
		/*
		 * man/fi_endpoint.3
		 *
		 *   fi_rx_attr::caps
		 *
		 *   "... If the caps field is 0 on input to fi_getinfo(3), the
		 *   caps value from the fi_info structure will be used."
		 */
		if (hints->rx_attr->caps) {
			fi->rx_attr->caps = hints->rx_attr->caps;
		} else {
			/* fi->caps is either:
			   1) The caps value from the fi_info hints->caps structure
			   2) The OPX defaults, which happens if the fi_info hints->caps is null
			*/
			fi->rx_attr->caps = fi->caps;
		}

		/* adjust parameters down from what requested if required */
		fi->rx_attr->op_flags = hints->rx_attr->op_flags;
	} else if (hints && hints->caps) {
		fi->rx_attr->caps = hints->caps;
	}

	/*
	 * man/fi_domain.3
	 *
	 * On input to fi_getinfo, a user may set this (fi_domain_attr::domain)
	 * to an opened domain instance to restrict output to the given domain.
	 * On output from fi_getinfo, if no domain was specified, but the user
	 * has an opened instance of the named domain, this (fi_domain_attr::domain)
	 * will reference the first opened instance. If no instance has been
	 * opened, this field will be NULL.
	 */

	if (fi_opx_global.default_domain_attr == NULL) {
		if (fi_opx_alloc_default_domain_attr(&fi_opx_global.default_domain_attr)) {
			FI_DBG(fi_opx_global.prov, FI_LOG_DOMAIN,
			       "alloc function could not allocate block of memory\n");
			errno = FI_ENOMEM;
			goto err;
		}
	}

	ret = fi_opx_choose_domain(fi->caps, fi->domain_attr, (hints) ? (hints->domain_attr) : NULL, progress);
	if (ret) {
		FI_LOG(fi_opx_global.prov, FI_LOG_DEBUG, FI_LOG_FABRIC, "cannot find appropriate domain\n");
		goto err;
	}

	if (fi_opx_global.default_ep_attr == NULL) {
		if (fi_opx_alloc_default_ep_attr(&fi_opx_global.default_ep_attr)) {
			FI_DBG(fi_opx_global.prov, FI_LOG_DOMAIN,
			       "alloc function could not allocate block of memory\n");
			errno = FI_ENOMEM;
			goto err;
		}
	}

	memcpy(fi->ep_attr, fi_opx_global.default_ep_attr, sizeof(*fi->ep_attr));
	if (hints && hints->ep_attr) {
		/* adjust parameters down from what requested if required */
		fi->ep_attr->type = hints->ep_attr->type;
		if (hints->ep_attr->max_msg_size > 0 &&
		    hints->ep_attr->max_msg_size <= fi_opx_global.default_ep_attr->max_msg_size) {
			fi->ep_attr->max_msg_size = hints->ep_attr->max_msg_size;
		}

		if (0 != hints->ep_attr->tx_ctx_cnt && hints->ep_attr->tx_ctx_cnt <= fi->ep_attr->tx_ctx_cnt) {
			fi->ep_attr->tx_ctx_cnt = hints->ep_attr->tx_ctx_cnt; /* TODO - check */
		}

		if (0 != hints->ep_attr->rx_ctx_cnt && hints->ep_attr->rx_ctx_cnt <= fi->ep_attr->rx_ctx_cnt) {
			fi->ep_attr->rx_ctx_cnt = hints->ep_attr->rx_ctx_cnt; /* TODO - check */
		}
	}

	fi->nic			    = ofi_nic_dup(NULL);
	fi->nic->bus_attr->bus_type = FI_BUS_PCI;

	return 0;

err:
	if (fi) {
		free(fi->domain_attr->name);
		fi->domain_attr->name = NULL;
		free(fi->fabric_attr->name);
		fi->fabric_attr->name = NULL;
		free(fi->fabric_attr->prov_name);
		fi->fabric_attr->prov_name = NULL;
		free(fi->src_addr);
		fi->src_addr	= NULL;
		fi->src_addrlen = 0;
		free(fi->dest_addr);
		fi->dest_addr	 = NULL;
		fi->dest_addrlen = 0;
	}

	if (fi_opx_global.default_ep_attr != NULL) {
		free(fi_opx_global.default_ep_attr);
		fi_opx_global.default_ep_attr = NULL;
	}

	if (fi_opx_global.default_tx_attr != NULL) {
		free(fi_opx_global.default_tx_attr);
		fi_opx_global.default_tx_attr = NULL;
	}

	if (fi_opx_global.default_rx_attr != NULL) {
		free(fi_opx_global.default_rx_attr);
		fi_opx_global.default_rx_attr = NULL;
	}

	if (fi_opx_global.default_domain_attr != NULL) {
		if (fi_opx_global.default_domain_attr->name != NULL) {
			free(fi_opx_global.default_domain_attr->name);
			fi_opx_global.default_domain_attr->name = NULL;
		}
		free(fi_opx_global.default_domain_attr);
		fi_opx_global.default_domain_attr = NULL;
	}

	return -errno;
}

struct fi_opx_global_data fi_opx_global = {.hfi_local_info.type	  = OPX_HFI1_UNDEF,
					   .opx_hfi1_type_strings = {[OPX_HFI1_UNDEF]  = "OPX_HFI1_UNDEF",
								     [OPX_HFI1_JKR_9B] = "OPX_HFI1_JKR_9B",
								     [OPX_HFI1_WFR]    = "OPX_HFI1_WFR",
								     [3]	       = "ERROR",
								     [OPX_HFI1_JKR]    = "OPX_HFI1_JKR"}};
/* ROUTE CONTROL table for each packet type */
int opx_route_control[OPX_HFI1_NUM_PACKET_TYPES];

static int fi_opx_getinfo_hfi(int hfi, uint32_t version, const char *node, const char *service, uint64_t flags,
			      const struct fi_info *hints, struct fi_info **info, struct fi_info **info_tail)
{
	int		ret, ret_auto;
	struct fi_info *fi	= NULL;
	struct fi_info *fi_auto = NULL;

	*info	   = NULL;
	*info_tail = NULL;

	if (hints) {
		ret = fi_opx_check_info(hints);
		if (ret) {
			return ret;
		}
		if (!(fi = fi_allocinfo()) || !(fi_auto = fi_allocinfo())) {
			ret = -FI_ENOMEM;
			goto err;
		}
		ret	 = fi_opx_fillinfo(fi, node, service, hints, flags, FI_PROGRESS_MANUAL);
		ret_auto = fi_opx_fillinfo(fi_auto, node, service, hints, flags, FI_PROGRESS_AUTO);
		if (hints->domain_attr->data_progress != FI_PROGRESS_UNSPEC) {
			fi_opx_global.progress = hints->domain_attr->data_progress;
			if (hints->domain_attr->data_progress == FI_PROGRESS_AUTO) {
				FI_INFO(fi_opx_global.prov, FI_LOG_FABRIC, "Locking is forced in FI_PROGRESS_AUTO\n");
			}
		}
		if (ret || ret_auto) {
			ret = ret ? ret : ret_auto;
			goto err;
		}

	} else if (node || service) {
		if (!(fi = fi_allocinfo()) || !(fi_auto = fi_allocinfo())) {
			ret = -FI_ENOMEM;
			goto err;
		}

		ret	 = fi_opx_fillinfo(fi, node, service, hints, flags, FI_PROGRESS_MANUAL);
		ret_auto = fi_opx_fillinfo(fi_auto, node, service, hints, flags, FI_PROGRESS_AUTO);
		if (ret || ret_auto) {
			ret = ret ? ret : ret_auto;
			goto err;
		}

	} else {
		if (!(fi = fi_dupinfo(fi_opx_global.info)) ||
		    !(fi_opx_global.info->next != NULL && (fi_auto = fi_dupinfo(fi_opx_global.info->next)))) {
			ret = -FI_ENOMEM;
			goto err;
		}
	}

	/* Set the appropriate domain name associated with the HFI */
	char domain_name[128];

	sprintf(domain_name, "%s%d", FI_OPX_DOMAIN_NAME_PREFIX, hfi);
	free(fi->domain_attr->name);
	fi->domain_attr->name = strdup(domain_name);

	if (fi_auto) {
		free(fi_auto->domain_attr->name);
		fi_auto->domain_attr->name = strdup(fi->domain_attr->name);
	}

	fi->next = fi_auto;

	*info	   = fi;
	*info_tail = (fi->next) ? fi->next : fi;

	return 0;

err:
	if (fi) {
		fi_freeinfo(fi);
	}
	if (fi_auto) {
		fi_freeinfo(fi_auto);
	}
	return ret;
}

static int fi_opx_getinfo(uint32_t version, const char *node, const char *service, uint64_t flags,
			  const struct fi_info *hints, struct fi_info **info)
{
	int		ret, i;
	struct fi_info *cur, *cur_tail;
	struct fi_info *tail = NULL;

	*info	     = NULL;
	fi_opx_count = opx_hfi_get_hfi1_count();
	FI_LOG(fi_opx_global.prov, FI_LOG_TRACE, FI_LOG_FABRIC, "Detected %d hfi1(s) in the system\n", fi_opx_count);

	if (!fi_opx_count) {
		return -FI_ENODATA;
	}

	for (i = 0; i < fi_opx_count; i++) {
		ret = fi_opx_getinfo_hfi(i, version, node, service, flags, hints, &cur, &cur_tail);
		if (ret) {
			continue;
		}
		if (!cur) {
			continue;
		}

		FI_LOG(fi_opx_global.prov, FI_LOG_TRACE, FI_LOG_FABRIC, "Successfully got getinfo for HFI %d\n", i);

		if (!*info) {
			*info = cur;
		} else {
			tail->next = cur;
		}

		tail = cur_tail;
	}

	return 0;
}

static void fi_opx_fini()
{
	always_assert(fi_opx_init == 1, "OPX provider finalize called before initialize\n");

	/* If we abnormally exited holding the memory monitor lock, we
	 * want to unlock if we can so we don't hang in flush.
	 * If it's still locked in another thread we can't flush/cleanup,
	 * so do our best and free storage */
	pthread_mutex_trylock(&mm_lock);
	int locked = pthread_mutex_unlock(&mm_lock); /* rc 0 is unlocked */

	struct dlist_entry    *tmp;
	struct opx_tid_domain *tid_domain;

	dlist_foreach_container_safe (&(fi_opx_global.tid_domain_list), struct opx_tid_domain, tid_domain, list_entry,
				      tmp) {
		opx_close_tid_domain(tid_domain, locked);
	}

#ifdef OPX_HMEM
	struct opx_hmem_domain *hmem_domain;

	dlist_foreach_container_safe (&(fi_opx_global.hmem_domain_list), struct opx_hmem_domain, hmem_domain,
				      list_entry, tmp) {
		opx_hmem_close_domain(hmem_domain, locked);
	}
#endif

	fi_freeinfo(fi_opx_global.info);
	OPX_TRACER_EXIT();

	if (fi_opx_global.daos_hfi_rank_hashmap) {
		struct fi_opx_daos_hfi_rank *cur_hfi_rank = NULL;
		struct fi_opx_daos_hfi_rank *tmp_hfi_rank = NULL;

		HASH_ITER(hh, fi_opx_global.daos_hfi_rank_hashmap, cur_hfi_rank, tmp_hfi_rank)
		{
			if (cur_hfi_rank) {
				HASH_DEL(fi_opx_global.daos_hfi_rank_hashmap, cur_hfi_rank);
				free(cur_hfi_rank);
				cur_hfi_rank = NULL;
			}
		}
	}
}

struct fi_provider fi_opx_provider = {.name	  = FI_OPX_PROVIDER_NAME,
				      .version	  = FI_OPX_PROVIDER_VERSION,
				      .fi_version = OFI_VERSION_LATEST,
				      .getinfo	  = fi_opx_getinfo,
				      .fabric	  = fi_opx_fabric,
				      .cleanup	  = fi_opx_fini};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
/*
 * Use this dummy function to do any compile-time validation of data
 * structure sizes needed to ensure performance.
 */
static void do_static_assert_tests()
{
	// Verify that pio_state is exactly one cache-line long.
	OPX_COMPILE_TIME_ASSERT((sizeof(union fi_opx_hfi1_pio_state) == 8), "fi_opx_hfi1_pio_state size error.");
	// Verify that pointers are exactly one cache-line long.
	OPX_COMPILE_TIME_ASSERT((sizeof(union fi_opx_hfi1_pio_state *) == 8),
				"fi_opx_hfi1_pio_state pointer size error.");

	union fi_opx_hfi1_packet_payload *payload = NULL;
	OPX_COMPILE_TIME_ASSERT(sizeof(*payload) == sizeof(payload->tid_cts),
				"Expected TID rendezvous CTS payload size error");

	OPX_COMPILE_TIME_ASSERT(sizeof(*payload) >= sizeof(payload->rendezvous.contiguous),
				"Contiguous rendezvous payload size error");

	OPX_COMPILE_TIME_ASSERT(sizeof(*payload) == sizeof(payload->rendezvous.noncontiguous),
				"Non-contiguous rendezvous payload size error");

	OPX_COMPILE_TIME_ASSERT((sizeof(struct fi_opx_hmem_info) >> 3) == OPX_HMEM_SIZE_QWS,
				"sizeof(fi_opx_hmem_info) >> 3 != OPX_HMEM_SIZE_QWS");
	OPX_COMPILE_TIME_ASSERT(OPX_HFI1_TID_PAGESIZE == 4096, "OPX_HFI1_TID_PAGESIZE must be 4K!");
}
#pragma GCC diagnostic pop

OPX_INI
{
	fi_opx_count	       = 1;
	fi_opx_global.progress = FI_PROGRESS_MANUAL;
	fi_opx_set_default_info(); // TODO: fold into fi_opx_set_defaults

	/* Refrain from allocating memory dynamically in this INI function.
	   That sort of behavior will results in memory leaks for the fi_info
	   executable. */

	fi_opx_global.default_domain_attr = NULL;
	fi_opx_global.default_ep_attr	  = NULL;
	fi_opx_global.default_tx_attr	  = NULL;
	fi_opx_global.default_rx_attr	  = NULL;

	fi_opx_global.prov		    = &fi_opx_provider;
	fi_opx_global.daos_hfi_rank_hashmap = NULL;

	memset(&fi_opx_global.hfi_local_info, 0, sizeof(fi_opx_global.hfi_local_info));

	fi_opx_init = 1;

	fi_param_define(
		&fi_opx_provider, "uuid", FI_PARAM_STRING,
		"Globally unique ID for preventing OPX jobs from conflicting either in shared memory or over the OPX fabric. Defaults to the Slurm job ID if one exists, otherwise defaults to Intel MPI UUID if one exists, otherwise defaults to \"%s\"",
		OPX_DEFAULT_JOB_KEY_STR);
	fi_param_define(&fi_opx_provider, "force_cpuaffinity", FI_PARAM_BOOL,
			"Causes the thread to bind itself to the cpu core it is running on. Defaults to FALSE.");
	fi_param_define(&fi_opx_provider, "reliability_service_usec_max", FI_PARAM_INT,
			"The number of microseconds between pings for un-acknowledged packets. Defaults to 500 usec.");
	fi_param_define(
		&fi_opx_provider, "reliability_max_uncongested_pings", FI_PARAM_INT,
		"The maximum number of reliability pings sent in a single timer iteration when the network link is uncongested. Value must be between %d and %d. Defaults to %d.",
		OPX_RELIABILITY_MAX_UNCONGESTED_PINGS_MIN, OPX_RELIABILITY_MAX_UNCONGESTED_PINGS_MAX,
		OPX_RELIABILITY_MAX_UNCONGESTED_PINGS_DEFAULT);
	fi_param_define(
		&fi_opx_provider, "reliability_max_congested_pings", FI_PARAM_INT,
		"The maximum number of reliability pings sent in a single timer iteration when the network link is congested. Value must be between %d and %d. Defaults to %d.",
		OPX_RELIABILITY_MAX_CONGESTED_PINGS_MIN, OPX_RELIABILITY_MAX_CONGESTED_PINGS_MAX,
		OPX_RELIABILITY_MAX_CONGESTED_PINGS_DEFAULT);
	fi_param_define(
		&fi_opx_provider, "reliability_service_pre_ack_rate", FI_PARAM_INT,
		"The number of packets to receive from a particular sender before preemptively acknowledging them without waiting for a ping. Valid values are powers of 2 in the range of 0-32,768, where 0 indicates no preemptive acking. Defaults to 64.");
	fi_param_define(
		&fi_opx_provider, "selinux", FI_PARAM_BOOL,
		"Set to TRUE if you're running a security-enhanced Linux. This enables updating the Jkey used based on system settings. Defaults to FALSE.");
	fi_param_define(
		&fi_opx_provider, "hfi_select", FI_PARAM_STRING,
		"Overrides the normal algorithm used to choose which HFI a process will use. See the documentation for more information.");
	fi_param_define(&fi_opx_provider, "mp_eager_disable", FI_PARAM_BOOL,
			"Disables tx multi-packet eager use. Defaults to %s.",
			OPX_MP_EGR_DISABLE_DEFAULT ? "TRUE" : "FALSE");
	fi_param_define(
		&fi_opx_provider, "rzv_min_payload_bytes", FI_PARAM_INT,
		"The minimum length in bytes where rendezvous will be used. For messages smaller than this threshold, the send will first try to be completed using eager or multi-packet eager. Defaults to %d.",
		OPX_RZV_MIN_PAYLOAD_BYTES_DEFAULT);
	fi_param_define(&fi_opx_provider, "delivery_completion_threshold", FI_PARAM_INT,
			"Will be deprecated. Please use FI_OPX_SDMA_BOUNCE_BUF_THRESHOLD");
	fi_param_define(
		&fi_opx_provider, "sdma_bounce_buf_threshold", FI_PARAM_INT,
		"The maximum message length in bytes that will be copied to the SDMA bounce buffer. For messages larger than this threshold, the send will not be completed until receiver has ACKed. Value must be between %d and %d. Defaults to %d.",
		OPX_SDMA_BOUNCE_BUF_MIN, OPX_SDMA_BOUNCE_BUF_MAX, OPX_SDMA_BOUNCE_BUF_THRESHOLD);
	fi_param_define(&fi_opx_provider, "sdma_disable", FI_PARAM_BOOL,
			"Disables SDMA offload hardware. Default is FALSE (SDMA Enabled).");
	fi_param_define(
		&fi_opx_provider, "sdma_min_payload_bytes", FI_PARAM_INT,
		"The minimum message length in bytes where SDMA will be used. For messages smaller than this threshold, the send will be completed using PIO. Value must be between %d and %d. Defaults to %d.",
		FI_OPX_SDMA_MIN_PAYLOAD_BYTES_MIN, FI_OPX_SDMA_MIN_PAYLOAD_BYTES_MAX,
		FI_OPX_SDMA_MIN_PAYLOAD_BYTES_DEFAULT);
	fi_param_define(
		&fi_opx_provider, "tid_min_payload_bytes", FI_PARAM_INT,
		"The minimum message length in bytes where TID will be used. Value must be >= %d. Defaults to %d.",
		OPX_TID_MIN_PAYLOAD_BYTES_MIN, OPX_TID_MIN_PAYLOAD_BYTES_DEFAULT);
	fi_param_define(&fi_opx_provider, "tid_disable", FI_PARAM_BOOL,
			"Disables using Token ID (TID). Defaults to FALSE (TID Enabled).");
	fi_param_define(&fi_opx_provider, "expected_receive_enable", FI_PARAM_BOOL,
			"Deprecated. Use FI_OPX_TID_DISABLE instead.");
	fi_param_define(
		&fi_opx_provider, "prog_affinity", FI_PARAM_STRING,
		"When set, specify the set of CPU cores to set the progress thread affinity to. The format is <start>:<end>:<stride> where each triplet <start>:<end>:<stride> defines a block Both <start> and <end> is a core_id.");
	fi_param_define(&fi_opx_provider, "auto_progress_interval_usec", FI_PARAM_INT,
			"Number of usec that the progress thread waits between polling. Default is 1.");
	fi_param_define(
		&fi_opx_provider, "pkey", FI_PARAM_INT,
		"Partition key.  Should be a 2 byte positive integer. Default is the Pkey in the index 0 of the Pkey table of the unit and port on which context is created.");
	fi_param_define(&fi_opx_provider, "sl", FI_PARAM_INT,
			"Service Level.  This will also determine Service Class and Virtual Lane.  Default is %d.",
			FI_OPX_HFI1_SL_DEFAULT);
	fi_param_define(NULL, "opx_tracer_out_path", FI_PARAM_STRING,
			"Specify path to output per-process performance tracing log files (default: none)");
	fi_param_define(
		&fi_opx_provider, "shm_enable", FI_PARAM_BOOL,
		"Enables SHM across all ports and hfi units on the node. Setting it to FALSE disables SHM except peers with same lid and same hfi1 (loopback). Defaults to TRUE.");
	fi_param_define(
		&fi_opx_provider, "port", FI_PARAM_INT,
		"HFI1 port number.  If the specified port is not available, a default active port will be selected. Special value 0 indicates any available port. Defaults to port 1 on OPA100 and any port on CN5000.");
	fi_param_define(&fi_opx_provider, "link_down_wait_time_max_sec", FI_PARAM_INT,
			"The maximum time in seconds to wait for a link to come back up. Default is %d\n",
			OPX_LINK_DOWN_WAIT_TIME_MAX_SEC_DEFAULT);
#ifdef OPX_HMEM
	fi_param_define(
		&fi_opx_provider, "dev_reg_send_threshold", FI_PARAM_INT,
		"The individual packet threshold where lengths above do not use a device registered copy when sending data from GPU. Default is %d.",
		OPX_HMEM_DEV_REG_SEND_THRESHOLD_DEFAULT);
	fi_param_define(
		&fi_opx_provider, "dev_reg_recv_threshold", FI_PARAM_INT,
		"The individual packet threshold where lengths above do not use a device registered copy when receiving data into GPU. Default is %d.",
		OPX_HMEM_DEV_REG_RECV_THRESHOLD_DEFAULT);
#endif
	fi_param_define(
		&fi_opx_provider, "route_control", FI_PARAM_STRING,
		"Specify the route control for each packet type. The format is <inject packet type value>:<eager packet type value>:<multi-packet eager packet type value>:<dput packet type value>:<rendezvous control packet value>:<rendezvous data packet value>. Each value can range from 0-7. 0-3 is used for in-order and 4-7 is used for out-of-order. If Token ID (TID) is enabled the out-of-order route controls are disabled. Default is \"%d:%d:%d:%d:%d:%d\" on OPA100 and \"%d:%d:%d:%d:%d:%d\" on CN5000",
		OPX_RC_IN_ORDER_0, OPX_RC_IN_ORDER_0, OPX_RC_IN_ORDER_0, OPX_RC_IN_ORDER_0, OPX_RC_IN_ORDER_0,
		OPX_RC_IN_ORDER_0, OPX_RC_OUT_OF_ORDER_0, OPX_RC_OUT_OF_ORDER_0, OPX_RC_OUT_OF_ORDER_0,
		OPX_RC_OUT_OF_ORDER_0, OPX_RC_IN_ORDER_0, OPX_RC_OUT_OF_ORDER_0);
	fi_param_define(
		&fi_opx_provider, "mixed_network", FI_PARAM_INT,
		"Indicates a mixed network of OPA100 and CN5000. Needs to be set to 1 when mixed network is used. Default is 0.");

	/* Track TID and HMEM domains so caches can be cleared on exit */
	dlist_init(&fi_opx_global.tid_domain_list);
#ifdef OPX_HMEM
	dlist_init(&fi_opx_global.hmem_domain_list);
#endif

	if (fi_log_enabled(fi_opx_global.prov, FI_LOG_TRACE, FI_LOG_FABRIC)) {
		Dl_info dl_info;
		if (dladdr((void *) fi_opx_ini,
			   &dl_info)) { // Use the OPX_INI function as the symbol to get runtime info
			FI_TRACE(fi_opx_global.prov, FI_LOG_FABRIC,
				 "Using opx Provider: Library file location is %s.\n", dl_info.dli_fname);
		} else {
			FI_TRACE(fi_opx_global.prov, FI_LOG_FABRIC,
				 "Error retrieving library file location for opx Provider.\n");
		}
	}

	OPX_TRACER_INIT();

	return (&fi_opx_provider);
}
