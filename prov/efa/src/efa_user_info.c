/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_prov_info.h"

/**
 * @brief set the desc_addr field of user info
 *
 * @param	node[in]	node from user's call to fi_getinfo()
 * @param	service[in]	service from user's call to fi_getinfo()
 * @param	flags[in]	flags from user's call to fi_getinfo()
 * @param	hints[in]	hints from user's call to fi_getinfo()
 * @param	fi[out]		user_info object to be updated, can be a list of infos
 *
 * @return	0 on success
 * 		negative libfabric error code on failure
 */
int efa_user_info_set_dest_addr(const char *node, const char *service, uint64_t flags,
				const struct fi_info *hints, struct fi_info *fi)
{
	struct efa_ep_addr tmp_addr;
	void *dest_addr = NULL;
	int ret = FI_SUCCESS;
	struct fi_info *cur;

	if (flags & FI_SOURCE) {
		if (hints && hints->dest_addr)
			dest_addr = hints->dest_addr;
	} else {
		if (node || service) {
			ret = efa_str_to_ep_addr(node, service, &tmp_addr);
			if (ret)
				return ret;
			dest_addr = &tmp_addr;
		} else if (hints && hints->dest_addr) {
			dest_addr = hints->dest_addr;
		}
	}

	if (dest_addr) {
		for (cur = fi; cur; cur = cur->next) {
			cur->dest_addr = malloc(EFA_EP_ADDR_LEN);
			if (!cur->dest_addr) {
				for (; fi->dest_addr; fi = fi->next)
					free(fi->dest_addr);
				return -FI_ENOMEM;
			}
			memcpy(cur->dest_addr, dest_addr, EFA_EP_ADDR_LEN);
			cur->dest_addrlen = EFA_EP_ADDR_LEN;
		}
	}
	return ret;
}

/**
 * @brief check the src_addr and desc_addr field of user provided hints
 *
 * @param	node[in]	node from user's call to fi_getinfo()
 * @param	service[in]	service from user's call to fi_getinfo()
 * @param	flags[in]	flags from user's call to fi_getinfo()
 * @param	hints[in]	hints from user's call to fi_getinfo()
 * @param	fi[out]		user_info object to be updated, can be a list of infos
 *
 * @return	0, if hints matches EFA provider's capability
 * 		-FI_ENODATA, otherwise
 */
int efa_user_info_check_hints_addr(const char *node, const char *service,
				   uint64_t flags, const struct fi_info *hints)
{
	if (!(flags & FI_SOURCE) && hints && hints->src_addr &&
	    hints->src_addrlen != EFA_EP_ADDR_LEN)
		return -FI_ENODATA;

	if (((!node && !service) || (flags & FI_SOURCE)) &&
	    hints && hints->dest_addr &&
	    hints->dest_addrlen != EFA_EP_ADDR_LEN)
		return -FI_ENODATA;

	return 0;
}

#if HAVE_CUDA || HAVE_NEURON || HAVE_SYNAPSEAI
/**
 * @brief determine if EFA provider should claim support of FI_HMEM in info
 * @param[in]	version		libfabric API version used by user
 * @return	true, if EFA provider should claim support of FI_HMEM
 * 		false, otherwise
 */
bool efa_user_info_should_support_hmem(int version)
{
	bool any_hmem, rdma_allowed;
	char *extra_info = "";
	int i;

	/* Note that the default behavior of EFA provider is different between
	 * libfabric API version when CUDA is used as HMEM system.
	 *
	 * For libfabric API version 1.17 and earlier, EFA provider does not
	 * support FI_HMEM on CUDA unless GPUDirect RDMA is available.
	 *
	 * For libfabric API version 1.18 and later, EFA provider will claim support
	 * of FI_HMEM on CUDA as long as CUDA library is initialized and a CUDA device is
	 * available. On an system without GPUDirect RDMA, the support of CUDA memory
	 * is implemented by calling CUDA library. If user does not want EFA provider
	 * to use CUDA library, the user can call use fi_setopt to set
	 * FI_OPT_CUDA_API_PERMITTED to false.
	 * On an system without GPUDirect RDMA, such a call would fail.
	 *
	 * For NEURON and SYNAPSEAI HMEM types, use_device_rdma is required no
	 * matter the API version, as is P2P support.
	 */
	if (hmem_ops[FI_HMEM_CUDA].initialized && FI_VERSION_GE(version, FI_VERSION(1, 18))) {
		EFA_INFO(FI_LOG_CORE,
			"User is using API version >= 1.18. CUDA library and "
			"devices are available, claim support of FI_HMEM.\n");
			/* For this API we can support HMEM regardless of
			   use_device_rdma and P2P support, because we can use
			   CUDA api calls.*/
			return true;
	}

	if (hmem_ops[FI_HMEM_CUDA].initialized) {
		extra_info = "For CUDA and libfabric API version <1.18 ";
	}

	any_hmem = false;
	EFA_HMEM_IFACE_FOREACH_NON_SYSTEM(i) {
		enum fi_hmem_iface hmem_iface = efa_hmem_ifaces[i];
		/* Note that .initialized doesn't necessarily indicate there are
		   hardware devices available, only that the libraries are
		   available. */
		if (hmem_ops[hmem_iface].initialized) {
			any_hmem = true;
		}
	}

	if (!any_hmem) {
		EFA_WARN(FI_LOG_CORE,
			"FI_HMEM cannot be supported because no compatible "
			"libraries were found.\n");
		return false;
	}

	/* All HMEM providers require P2P support. */
	if (ofi_hmem_p2p_disabled()) {
		EFA_WARN(FI_LOG_CORE,
			"%sFI_HMEM capability requires peer to peer "
			"support, which is disabled because "
			"FI_HMEM_P2P_DISABLED was set to 1/on/true.\n",
			extra_info);
		return false;
	}

	/* all devices require use_device_rdma. */
	if (!efa_device_support_rdma_read()) {
		EFA_WARN(FI_LOG_CORE,
		"%sEFA cannot support FI_HMEM because the EFA device "
		"does not have RDMA support.\n",
		extra_info);
		return false;
	}

	rdma_allowed = efa_rdm_get_use_device_rdma(version);
	/* not allowed to use rdma, but device supports it... */
	if (!rdma_allowed) {
		EFA_WARN(FI_LOG_CORE,
		"%sEFA cannot support FI_HMEM because the environment "
		"variable FI_EFA_USE_DEVICE_RDMA is 0.\n",
		extra_info);
		return false;
	}

	return true;
}

#else

bool efa_user_info_should_support_hmem(int version)
{
	EFA_WARN(FI_LOG_CORE,
		"EFA cannot support FI_HMEM because it was not compiled "
		"with any supporting FI_HMEM capabilities\n");
	return false;
}

#endif
/**
 * @brief update RDM info to match user hints
 *
 * the input info is a duplicate of prov info, which matches
 * the capability of the EFA device. This function tailor it
 * so it matches user provided hints
 *
 * @param	version[in]	libfabric API version
 * @param	info[in,out]	info to be updated
 * @param	hints[in]	user provided hints
 * @return	0 on success
 * 		negative libfabric error code on failure
 */
static
int efa_user_info_alter_rdm(int version, struct fi_info *info, const struct fi_info *hints)
{
	if (hints && (hints->caps & FI_HMEM)) {
		/*
		 * FI_HMEM is a primary capability, therefore only check
		 * (and cliam) its support when user explicitly requested it.
		 */
		if (!efa_user_info_should_support_hmem(version)) {
			return -FI_ENODATA;
		}

		info->caps |= FI_HMEM;
	} else {
		info->caps &= ~FI_HMEM;
	}

	if (info->caps & FI_HMEM) {
		/* Add FI_MR_HMEM to mr_mode when claiming support of FI_HMEM
		 * because EFA provider's HMEM support rely on
		 * application to provide descriptor for device buffer.
		 */
		if (hints->domain_attr &&
		    !(hints->domain_attr->mr_mode & FI_MR_HMEM)) {
			EFA_WARN(FI_LOG_CORE,
			        "FI_HMEM capability requires device registrations (FI_MR_HMEM)\n");
			return -FI_ENODATA;
		}

		info->domain_attr->mr_mode |= FI_MR_HMEM;
	}

	if (FI_VERSION_LT(version, FI_VERSION(1, 18)) && info->caps & FI_HMEM) {
		/* our HMEM atomic support rely on calls to CUDA API, which
		 * is disabled if user are using libfabric API version 1.17 and earlier.
		 */
		if (hints->caps & FI_ATOMIC) {
			EFA_WARN(FI_LOG_CORE,
			        "FI_ATOMIC capability with FI_HMEM relies on CUDA API, "
				"which is disable for libfabric API version 1.17 and eariler\n");
			return -FI_ENODATA;
		}
		info->caps &= ~FI_ATOMIC;
	}

	/*
	 * Handle user-provided hints and adapt the info object passed back up
	 * based on EFA-specific constraints.
	 */
	if (hints) {
		if (hints->tx_attr) {
			/* efa device doesn't have ordering,
			 * if apps request an ordering that is relaxed than
			 * what provider supports, we should respect that.
			 * If no ordering is specified,
			 * the default message order supported by the provider is returned.
			 */
			info->tx_attr->msg_order &= hints->tx_attr->msg_order;

			/* If no atomic ordering is requested, set the max_order_*_size as 0 */
			if (!(hints->tx_attr->msg_order & FI_ORDER_ATOMIC_RAW))
				info->ep_attr->max_order_raw_size = 0;
			if (!(hints->tx_attr->msg_order & FI_ORDER_ATOMIC_WAR))
				info->ep_attr->max_order_war_size = 0;
			if (!(hints->tx_attr->msg_order & FI_ORDER_ATOMIC_WAW))
				info->ep_attr->max_order_waw_size = 0;
		}

		if (hints->rx_attr) {
			/* efa device doesn't have ordering,
			 * if apps request an ordering that is relaxed than
			 * what provider supports, we should respect that.
			 * If no ordering is specified,
			 * the default message order supported by the provider is returned.
			 */
			info->rx_attr->msg_order &= hints->rx_attr->msg_order;
		}

		if (info->tx_attr->msg_order != info->rx_attr->msg_order)
			EFA_INFO(FI_LOG_EP_CTRL, "Inconsistent tx/rx msg order. Tx msg order: %lu, Rx msg order: %lu. "
						 "Libfabric can proceed but it is recommended to align the tx and rx msg order.\n",
						 info->tx_attr->msg_order, info->rx_attr->msg_order);

		/*
		 * The provider does not force applications to register buffers
		 * with the device, but if an application is able to, reuse
		 * their registrations and avoid the bounce buffers.
		 */
		if (hints->domain_attr && hints->domain_attr->mr_mode & FI_MR_LOCAL)
			info->domain_attr->mr_mode |= FI_MR_LOCAL;

		/*
		 * Same goes for prefix mode, where the protocol does not
		 * absolutely need a prefix before receive buffers, but it can
		 * use it when available to optimize transfers with endpoints
		 * having the following profile:
		 *	- Requires FI_MSG and not FI_TAGGED/FI_ATOMIC/FI_RMA
		 *	- Can handle registrations (FI_MR_LOCAL)
		 *	- No need for FI_DIRECTED_RECV
		 *	- Guaranteed to send msgs smaller than info->nic->link_attr->mtu
		 */
		if (hints->mode & FI_MSG_PREFIX) {
			EFA_INFO(FI_LOG_CORE,
				"FI_MSG_PREFIX supported by application.\n");
			info->mode |= FI_MSG_PREFIX;
			info->tx_attr->mode |= FI_MSG_PREFIX;
			info->rx_attr->mode |= FI_MSG_PREFIX;
			info->ep_attr->msg_prefix_size = EFA_RDM_MSG_PREFIX_SIZE;
			EFA_INFO(FI_LOG_CORE,
				"FI_MSG_PREFIX size = %ld\n", info->ep_attr->msg_prefix_size);
		}
	}

	/* Print a warning and use FI_AV_TABLE if the app requests FI_AV_MAP */
	if (hints && hints->domain_attr && hints->domain_attr->av_type == FI_AV_MAP)
		EFA_WARN(FI_LOG_CORE, "FI_AV_MAP is deprecated in Libfabric 2.x. Please use FI_AV_TABLE. "
					"EFA provider will now switch to using FI_AV_TABLE.\n");
	info->domain_attr->av_type = FI_AV_TABLE;

	if (!hints || !hints->domain_attr ||
	    hints->domain_attr->resource_mgmt == FI_RM_UNSPEC)
		info->domain_attr->resource_mgmt = FI_RM_ENABLED;
	else
		info->domain_attr->resource_mgmt = hints->domain_attr->resource_mgmt;

	return 0;
}

/**
 * @brief update EFA direct info to match user hints
 *
 * the input info is a duplicate of prov info, which matches
 * the capability of the EFA device. This function tailor it
 * so it matches user provided hints
 *
 * @param	version[in]	libfabric API version
 * @param	info[in,out]	info to be updated
 * @param	hints[in]	user provided hints
 * @return	0 on success
 * 		negative libfabric error code on failure
 */
static
int efa_user_info_alter_direct(int version, struct fi_info *info, const struct fi_info *hints)
{
	/*
	 * FI_HMEM is a primary capability, therefore only check
	 * and claim support when explicitly requested
	 */
	if (hints && (hints->caps & FI_HMEM))
		info->caps |= FI_HMEM;
	else
		info->caps &= ~FI_HMEM;

	if (info->caps & FI_HMEM) {
		/* Add FI_MR_HMEM to mr_mode when claiming support of FI_HMEM
		 * because EFA provider's HMEM support rely on
		 * application to provide descriptor for device buffer.
		 */
		if (hints->domain_attr &&
		    !(hints->domain_attr->mr_mode & FI_MR_HMEM)) {
			EFA_WARN(FI_LOG_CORE,
			        "FI_HMEM capability requires device registrations (FI_MR_HMEM)\n");
			return -FI_ENODATA;
		}

		info->domain_attr->mr_mode |= FI_MR_HMEM;
	}

	/*
	 * Handle user-provided hints and adapt the info object passed back up
	 * based on EFA-specific constraints.
	 */
	if (hints) {
		/* EFA direct cannot make use of message prefix */
		if (hints->mode & FI_MSG_PREFIX) {
			EFA_INFO(FI_LOG_CORE,
				"FI_MSG_PREFIX supported by application but EFA direct cannot "
				"use prefix. Setting prefix size to 0.\n");
			info->ep_attr->msg_prefix_size = 0;
			EFA_INFO(FI_LOG_CORE,
				"FI_MSG_PREFIX size = %ld\n", info->ep_attr->msg_prefix_size);
		}
		/* When user requests FI_RMA and it's supported, the max_msg_size should be returned
		 * as the maximum of both MSG and RMA operations
		 */
		if (hints->caps & FI_RMA)
			info->ep_attr->max_msg_size = MAX(g_device_list[0].ibv_port_attr.max_msg_sz, g_device_list[0].max_rdma_size);
	}

	/* Print a warning and use FI_AV_TABLE if the app requests FI_AV_MAP */
	if (hints && hints->domain_attr && hints->domain_attr->av_type == FI_AV_MAP)
		EFA_WARN(FI_LOG_CORE, "FI_AV_MAP is deprecated in Libfabric 2.x. Please use FI_AV_TABLE. "
					"EFA direct provider will now switch to using FI_AV_TABLE.\n");
	info->domain_attr->av_type = FI_AV_TABLE;

	return 0;
}

/**
 * @brief get a list of fi_info objects the fit user's requirements
 *
 * @param	node[in]	node from user's call to fi_getinfo()
 * @param	service[in]	service from user's call to fi_getinfo()
 * @param	flags[in]	flags from user's call to fi_getinfo()
 * @param	hints[in]	hints from user's call to fi_getinfo()
 * @param	info[out]	a linked list of user_info that met user's requirements
 * @return 	0 on success
 * 		negative libfabric error code on failure
 */
static
int efa_get_user_info(uint32_t version, const char *node,
			  const char *service, uint64_t flags,
			  const struct fi_info *hints, struct fi_info **info)
{
	const struct fi_info *prov_info;
	struct fi_info *dupinfo, *tail;
	int ret;

	ret = efa_user_info_check_hints_addr(node, service, flags, hints);
	if (ret) {
		*info = NULL;
		return ret;
	}

	*info = tail = NULL;
	for (prov_info = efa_util_prov.info;
	     prov_info;
	     prov_info = prov_info->next) {

		ret = ofi_check_info(&efa_util_prov, prov_info, version, hints);
		if (ret)
			continue;

		if (!efa_env_allows_nic(prov_info->nic->device_attr->name))
			continue;

		ret = efa_prov_info_compare_src_addr(node, flags, hints, prov_info);
		if (ret)
			continue;

		ret = efa_prov_info_compare_domain_name(hints, prov_info);
		if (ret)
			continue;

		ret = efa_prov_info_compare_pci_bus_id(hints, prov_info);
		if (ret)
			continue;

		dupinfo = fi_dupinfo(prov_info);
		if (!dupinfo) {
			ret = -FI_ENOMEM;
			goto free_info;
		}

		ret = efa_user_info_set_dest_addr(node, service, flags, hints, dupinfo);
		if (ret)
			goto free_info;

		dupinfo->fabric_attr->api_version = version;

		if (EFA_INFO_TYPE_IS_RDM(prov_info)) {
			ret = efa_user_info_alter_rdm(version, dupinfo, hints);
			if (ret)
				goto free_info;

			/* If application asked for FI_REMOTE_COMM but not FI_LOCAL_COMM, it
			 * does not want to use shm. In this case, we honor the request by
			 * unsetting the FI_LOCAL_COMM flag in info. This way efa_rdm_ep_open()
			 * should disable shm transfer for the endpoint
			 */
			if (hints && hints->caps & FI_REMOTE_COMM && !(hints->caps & FI_LOCAL_COMM))
				dupinfo->caps &= ~FI_LOCAL_COMM;
		}

		if (EFA_INFO_TYPE_IS_DIRECT(prov_info)) {
			ret = efa_user_info_alter_direct(version, dupinfo, hints);
			if (ret)
				goto free_info;
		}

		ofi_alter_info(dupinfo, hints, version);

		if (!*info)
			*info = dupinfo;
		else
			tail->next = dupinfo;
		tail = dupinfo;
	}

	return 0;
free_info:
	fi_freeinfo(dupinfo);
	fi_freeinfo(*info);
	*info = NULL;
	return ret;
}

/**
 * @brief get a list of info the fit user's requirements
 *
 * This is EFA provider's implemenation of fi_getinfo() API.
 *
 * @param	node[in]	node from user's call to fi_getinfo()
 * @param	service[in]	service from user's call to fi_getinfo()
 * @param	flags[in]	flags from user's call to fi_getinfo()
 * @param	hints[in]	hints from user's call to fi_getinfo()
 * @param	info[out]	a linked list of user_info that met user's requirements
 * @return 	0 on success
 * 		negative libfabric error code on failure
 */
int efa_getinfo(uint32_t version, const char *node,
		const char *service, uint64_t flags,
		const struct fi_info *hints, struct fi_info **info)
{
	struct fi_info *info_list;
	enum fi_ep_type hints_ep_type;
	int err;

	hints_ep_type = FI_EP_UNSPEC;
	if (hints && hints->ep_attr) {
		hints_ep_type = hints->ep_attr->type;
	}

	if (hints_ep_type != FI_EP_UNSPEC && hints_ep_type != FI_EP_RDM && hints_ep_type != FI_EP_DGRAM) {
		EFA_WARN(FI_LOG_DOMAIN, "unsupported endpoint type: %d\n",
			 hints_ep_type);
		return -FI_ENODATA;
	}

	err = efa_get_user_info(version, node, service, flags, hints, &info_list);
	if (err && err != -FI_ENODATA) {
		return err;
	}

	*info = info_list;
	return FI_SUCCESS;
}
