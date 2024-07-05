/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_hmem.h"
#include "rdm/efa_rdm_pkt_type.h"

#define P2P_PROV_MAX_LEN 32

#if HAVE_CUDA || HAVE_NEURON
static size_t efa_max_eager_msg_size_with_largest_header(struct efa_domain *efa_domain) {
	int mtu_size;

	mtu_size = efa_domain->device->rdm_info->ep_attr->max_msg_size;

	return mtu_size - efa_rdm_pkt_type_get_max_hdr_size();
}
#else
static size_t efa_max_eager_msg_size_with_largest_header(struct efa_domain *efa_domain) {
	return 0;
}
#endif

/**
 * @brief  Initialize the various protocol thresholds tracked in efa_hmem_info
 *         according to the given FI_HMEM interface.
 *
 * @param[in,out]  efa_domain  Pointer to struct efa_domain
 * @param[in]      iface       The FI_HMEM interface to initialize
 *
 * @return  0
 */
static int efa_domain_hmem_info_init_protocol_thresholds(struct efa_domain *efa_domain, enum fi_hmem_iface iface)
{
	struct efa_hmem_info *info = &efa_domain->hmem_info[iface];
	size_t tmp_value;

	/* Fall back to FI_HMEM_SYSTEM initialization logic when p2p is unavailable */
	if (!info->p2p_supported_by_device)
		iface = FI_HMEM_SYSTEM;

	switch (iface) {
	case FI_HMEM_SYSTEM:
		/* We have not yet tested runting with system memory */
		info->runt_size = 0;
		info->max_medium_msg_size = EFA_DEFAULT_INTER_MAX_MEDIUM_MESSAGE_SIZE;
		info->min_read_msg_size = EFA_DEFAULT_INTER_MIN_READ_MESSAGE_SIZE;
		info->min_read_write_size = EFA_DEFAULT_INTER_MIN_READ_WRITE_SIZE;
		fi_param_get_size_t(&efa_prov, "runt_size", &info->runt_size);
		fi_param_get_size_t(&efa_prov, "inter_max_medium_message_size", &info->max_medium_msg_size);
		fi_param_get_size_t(&efa_prov, "inter_min_read_message_size", &info->min_read_msg_size);
		fi_param_get_size_t(&efa_prov, "inter_min_read_write_size", &info->min_read_write_size);
		break;
	case FI_HMEM_CUDA:
		info->runt_size = EFA_DEFAULT_RUNT_SIZE;
		info->max_medium_msg_size = 0;
		info->min_read_msg_size = efa_max_eager_msg_size_with_largest_header(efa_domain) + 1;
		info->min_read_write_size = efa_max_eager_msg_size_with_largest_header(efa_domain) + 1;
		fi_param_get_size_t(&efa_prov, "runt_size", &info->runt_size);
		fi_param_get_size_t(&efa_prov, "inter_min_read_message_size", &info->min_read_msg_size);
		fi_param_get_size_t(&efa_prov, "inter_min_read_write_size", &info->min_read_write_size);
		if (-FI_ENODATA != fi_param_get(&efa_prov, "inter_max_medium_message_size", &tmp_value)) {
			EFA_WARN(FI_LOG_DOMAIN,
			         "The environment variable FI_EFA_INTER_MAX_MEDIUM_MESSAGE_SIZE was set, "
			         "but EFA HMEM via Cuda API only supports eager and runting read protocols. "
			         "The variable will not modify CUDA memory run config.\n");
		}
		break;
	case FI_HMEM_NEURON:
		info->runt_size = EFA_NEURON_RUNT_SIZE;
		info->max_medium_msg_size = 0;
		info->min_read_msg_size = efa_max_eager_msg_size_with_largest_header(efa_domain) + 1;
		info->min_read_write_size = efa_max_eager_msg_size_with_largest_header(efa_domain) + 1;
		fi_param_get_size_t(&efa_prov, "runt_size", &info->runt_size);
		fi_param_get_size_t(&efa_prov, "inter_min_read_message_size", &info->min_read_msg_size);
		fi_param_get_size_t(&efa_prov, "inter_min_read_write_size", &info->min_read_write_size);
		if (-FI_ENODATA != fi_param_get(&efa_prov, "inter_max_medium_message_size", &tmp_value)) {
			EFA_WARN(FI_LOG_DOMAIN,
			         "The environment variable FI_EFA_INTER_MAX_MEDIUM_MESSAGE_SIZE was set, "
			         "but EFA HMEM via Neuron API only supports eager and runting read protocols. "
			         "The variable will not modify CUDA memory run config.\n");
		}
		break;
	case FI_HMEM_SYNAPSEAI:
		info->runt_size = 0;
		info->max_medium_msg_size = 0;
		info->min_read_msg_size = 1;
		info->min_read_write_size = 1;
		if (-FI_ENODATA != fi_param_get_size_t(&efa_prov, "inter_max_medium_message_size", &tmp_value) ||
		    -FI_ENODATA != fi_param_get_size_t(&efa_prov, "inter_min_read_message_size", &tmp_value) ||
		    -FI_ENODATA != fi_param_get_size_t(&efa_prov, "inter_min_read_write_size", &tmp_value) ||
		    -FI_ENODATA != fi_param_get_size_t(&efa_prov, "runt_size", &tmp_value)) {
			EFA_WARN(FI_LOG_DOMAIN,
			        "One or more of the following environment variable(s) were set: ["
			        "FI_EFA_INTER_MAX_MEDIUM_MESSAGE_SIZE, "
			        "FI_EFA_INTER_MIN_READ_MESSAGE_SIZE, "
			        "FI_EFA_INTER_MIN_READ_WRITE_SIZE, "
			        "FI_EFA_RUNT_SIZE"
			        "], but EFA HMEM via Synapse only supports long read protocol. "
			        "The variable(s) will not modify Synapse memory run config.\n");
		}
		break;
	default:
		break;
	}
	return 0;
}

/**
 * @brief Retrieve the P2P capability provider name for the chosen domain
 * 
 * @param[in]		efa_domain	EFA domain
 * @param[in,out]	name		The output string
 * @param[in]		len		The expected provider name length
 * @return		On success, return the (positive) length of the
 * 			provider name; otherwise a negative error code.
 */
static inline ssize_t
efa_domain_hmem_p2p_prov_name(struct efa_domain *efa_domain, char *name, uint16_t len)
{
	FILE *fp = NULL;
	char *ibdev_path, *p2p_path = NULL;
	ssize_t ret = -FI_EINVAL;

	ibdev_path = efa_domain->device->ibv_ctx->device->ibdev_path;
	if (!ibdev_path) {
		EFA_WARN(FI_LOG_DOMAIN, "IB device sysfs is not defined\n");
		ret = -FI_EINVAL;
		goto out;
	}

	p2p_path = malloc(strlen(ibdev_path) + strlen(efa_env.p2p_file_suffix) + 1);
	if (!p2p_path) {
		ret = -FI_ENOMEM;
		goto out;
	}

	strcpy(p2p_path, ibdev_path);
	strcat(p2p_path, efa_env.p2p_file_suffix);

	fp = fopen(p2p_path, "r");
	if (!fp) {
		EFA_WARN(FI_LOG_DOMAIN, "Cannot open P2P file: %s\n", p2p_path);
		ret = -FI_ENOENT;
		goto out;
	}

	ret = (ssize_t) fread(name, 1, (size_t) len, fp);
	if (ret <= 0) {
		EFA_WARN(FI_LOG_DOMAIN, "P2P provider is not available\n");
		ret = -FI_ENOSYS;
		goto out;
	} else if (ret >= (ssize_t) len) {
		EFA_WARN(FI_LOG_DOMAIN, "Truncated P2P provider name\n");
		ret = -FI_ETRUNC;
		goto out;
	}

	name[ret] = '\0';
	EFA_INFO(FI_LOG_DOMAIN, "P2P provider name: %s\n", name);
out:
	if (fp) {
		fclose(fp);
	}
	if (p2p_path) {
		free(p2p_path);
	}
	return ret;
}

/**
 * @brief Determine if peer-to-peer capability is supported on the domain
 *
 * @param[in]	efa_domain	EFA domain
 * @param[in]	iface		HMEM interface to check
 * @return	true if P2P is supported, otherwise false
 */
static inline bool
efa_domain_hmem_support_p2p(struct efa_domain *efa_domain, enum fi_hmem_iface iface)
{
	char p2p_prov[P2P_PROV_MAX_LEN];
	const char *p2p_prov_prefix;

	if (iface == FI_HMEM_SYSTEM || iface == FI_HMEM_SYNAPSEAI) {
		return true;
	}

	assert(iface == FI_HMEM_CUDA || iface == FI_HMEM_NEURON);

	p2p_prov_prefix = iface == FI_HMEM_CUDA ? "NVIDIA" : "NEURON";

	if ((efa_domain_hmem_p2p_prov_name(efa_domain, p2p_prov,
					   P2P_PROV_MAX_LEN)) <= 0) {
		EFA_INFO(FI_LOG_DOMAIN, "Failed to get P2P provider\n");
		return false;
	}

	if (strlen(p2p_prov) < strlen(p2p_prov_prefix) ||
	    strncmp(p2p_prov, p2p_prov_prefix, strlen(p2p_prov_prefix))) {
		EFA_INFO(FI_LOG_DOMAIN, "P2P provider does not support hmem interface: %d\n", iface);
		return false;
	}

	return true;
}

/**
 * @brief Initialize the efa_hmem_info state for iface
 *
 * @param[in,out]  efa_domain  Pointer to struct efa_domain
 * @param[in]      iface       HMEM interface
 */
static void
efa_domain_hmem_info_init_iface(struct efa_domain *efa_domain, enum fi_hmem_iface iface)
{
	struct efa_hmem_info *info = &efa_domain->hmem_info[iface];

	if (!ofi_hmem_is_initialized(iface)) {
		EFA_INFO(FI_LOG_DOMAIN, "%s is not initialized\n",
		         fi_tostr(&iface, FI_TYPE_HMEM_IFACE));
		return;
	}

	info->p2p_disabled_by_user = false;
	info->p2p_supported_by_device = efa_domain_hmem_support_p2p(efa_domain, iface);
	if (!info->p2p_supported_by_device) {
		EFA_INFO(FI_LOG_DOMAIN, "%s P2P support is not available.\n",
		         fi_tostr(&iface, FI_TYPE_HMEM_IFACE));
	}

	switch (iface) {
	case FI_HMEM_CUDA:
		/* If user is using libfabric API 1.18 or later, by default EFA
		 * provider is permitted to use CUDA library to support CUDA
		 * memory, therefore p2p is not required.
		 */
		if (FI_VERSION_GE(efa_domain->util_domain.fabric->fabric_fid.api_version,
				  FI_VERSION(1, 18)))
			info->p2p_required_by_impl = !hmem_ops[iface].initialized;
		else
			info->p2p_required_by_impl = true;
		break;
	case FI_HMEM_NEURON:
	case FI_HMEM_SYNAPSEAI:
		info->p2p_required_by_impl = true;
		break;
	default:
		assert(iface == FI_HMEM_SYSTEM);
		info->p2p_required_by_impl = false;
	}

	efa_domain_hmem_info_init_protocol_thresholds(efa_domain, iface);

	info->initialized = true;
}

/**
 * @brief   Validate an FI_OPT_FI_HMEM_P2P (FI_OPT_ENDPOINT) option for a
 *          specified HMEM interface.
 *          Also update hmem_info[iface]->p2p_disabled_by_user accordingly.
 *
 * @param[in,out]  domain   The efa_domain struct which contains an efa_hmem_info array
 * @param[in]      iface    The fi_hmem_iface enum of the FI_HMEM interface to validate
 * @param[in]      p2p_opt  The P2P option to validate
 *
 * @return  0 if the P2P option is valid for the given interface
 *         -FI_OPNOTSUPP if the P2P option is invalid
 *         -FI_ENODATA if the given HMEM interface was not initialized
 *         -FI_EINVAL if p2p_opt is not a valid FI_OPT_FI_HMEM_P2P option
 */
int efa_domain_hmem_validate_p2p_opt(struct efa_domain *efa_domain, enum fi_hmem_iface iface, int p2p_opt)
{
	struct efa_hmem_info *info = &efa_domain->hmem_info[iface];

	if (OFI_UNLIKELY(!info->initialized))
		return -FI_ENODATA;

	switch (p2p_opt) {
	case FI_HMEM_P2P_REQUIRED:
		if (!info->p2p_supported_by_device)
			return -FI_EOPNOTSUPP;

		info->p2p_disabled_by_user = false;
		return 0;
	/*
	 * According to fi_setopt() document:
	 *
	 *     ENABLED means a provider may use P2P.
	 *     PREFERED means a provider should prefer P2P if it is available.
	 *
	 * These options does not require that p2p is supported by device,
	 * nor do they prohibit that p2p is reqruied by implementation. Therefore
	 * they are always supported.
	 */
	case FI_HMEM_P2P_PREFERRED:
	case FI_HMEM_P2P_ENABLED:
		info->p2p_disabled_by_user = false;
		return 0;

	case FI_HMEM_P2P_DISABLED:
		if (info->p2p_required_by_impl)
			return -FI_EOPNOTSUPP;

		info->p2p_disabled_by_user = true;
		return 0;
	}

	return -FI_EINVAL;
}

/**
 * @brief Initialize the support status for
 * all of the HMEM devices. The device hmem_info
 * struct will be used to determine which efa transfer
 * protocol should be selected.
 *
 * @param[in,out]  efa_domain  Pointer to struct efa_domain to be initialized
 *
 * @return  0 on success
 *          negative libfabric error code on an unexpected error
 */
int efa_domain_hmem_support_init_all(struct efa_domain *efa_domain)
{
	int ret = 0;
	enum fi_hmem_iface ifaces[4] = {FI_HMEM_SYSTEM, FI_HMEM_CUDA,
	                                FI_HMEM_NEURON, FI_HMEM_SYNAPSEAI};

	if(g_device_cnt <= 0) {
		return -FI_ENODEV;
	}

	memset(efa_domain->hmem_info, 0, OFI_HMEM_MAX * sizeof(struct efa_hmem_info));

	for (int i = 0; i < sizeof(ifaces) / sizeof(enum fi_hmem_iface); ++i) {
		efa_domain_hmem_info_init_iface(efa_domain, ifaces[i]);
	}

	return ret;
}

/**
 * @brief Copy data from a hmem IOV to a system buffer
 *
 * @param[in]   desc          Array of memory desc corresponding to IOV buffers
 * @param[out]  buff          Target buffer (system memory)
 * @param[in]   buff_size     The size of the target buffer
 * @param[in]   hmem_iov      IOV data source
 * @param[in]   iov_count     Number of IOV structures in IOV array
 * @return  number of bytes copied on success, or a negative error code
 */
ssize_t efa_copy_from_hmem_iov(void **desc, char *buff, int buff_size,
                               const struct iovec *hmem_iov, int iov_count)
{
	int i, ret = -1;
	size_t data_size = 0;

	for (i = 0; i < iov_count; i++) {
		if (data_size + hmem_iov[i].iov_len > buff_size) {
			EFA_WARN(FI_LOG_CQ, "IOV is larger than the target buffer\n");
			return -FI_ETRUNC;
		}

		ret = efa_copy_from_hmem(desc[i], buff + data_size,
		                         hmem_iov[i].iov_base, hmem_iov[i].iov_len);

		if (ret < 0)
			return ret;

		data_size += hmem_iov[i].iov_len;
	}

	return data_size;
}

/**
 * @brief Copy data from a system buffer to a hmem IOV
 *
 * @param[in]    desc            Array of memory desc corresponding to IOV buffers
 * @param[out]   hmem_iov        Target IOV (HMEM)
 * @param[in]    iov_count       Number of IOV entries in vector
 * @param[in]    buff            System buffer data source
 * @param[in]    buff_size       Size of data to copy
 * @return  number of bytes copied on success, or a negative error code
 */
ssize_t efa_copy_to_hmem_iov(void **desc, struct iovec *hmem_iov,
                             int iov_count, char *buff, int buff_size)
{
	int i, ret, bytes_remaining = buff_size, size;

	for (i = 0; i < iov_count && bytes_remaining; i++) {
		size = hmem_iov[i].iov_len;
		if (bytes_remaining < size) {
			size = bytes_remaining;
		}

		ret = efa_copy_to_hmem(desc[i], hmem_iov[i].iov_base, buff, size);

		if (ret < 0)
			return ret;

		bytes_remaining -= size;
	}

	if (bytes_remaining) {
		EFA_WARN(FI_LOG_CQ, "Source buffer is larger than target IOV\n");
		return -FI_ETRUNC;
	}
	return buff_size;
}
