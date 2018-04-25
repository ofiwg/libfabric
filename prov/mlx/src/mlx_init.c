/*
 * Copyright (c) 2016 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenFabrics.org BSD license below:
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
#include "mlx.h"


int mlx_errcode_translation_table[(-UCS_ERR_LAST)+2] = { -FI_EOTHER };

struct mlx_global_descriptor mlx_descriptor = {
	.config = NULL,
	.use_ns = 0,
	.ns_port = FI_MLX_DEFAULT_NS_PORT,
	.localhost = NULL,
};

static int mlx_init_errcodes()
{
	MLX_TRANSLATE_ERRCODE (UCS_OK)                  = -FI_SUCCESS;
	MLX_TRANSLATE_ERRCODE (UCS_INPROGRESS)          = -FI_EINPROGRESS;
	MLX_TRANSLATE_ERRCODE (UCS_ERR_NO_MESSAGE)      = -FI_ENOMSG;
	MLX_TRANSLATE_ERRCODE (UCS_ERR_NO_RESOURCE)     = -FI_EINVAL;
	MLX_TRANSLATE_ERRCODE (UCS_ERR_IO_ERROR)        = -FI_EIO;
	MLX_TRANSLATE_ERRCODE (UCS_ERR_NO_MEMORY)       = -FI_ENOMEM;
	MLX_TRANSLATE_ERRCODE (UCS_ERR_INVALID_PARAM)   = -FI_EINVAL;
	MLX_TRANSLATE_ERRCODE (UCS_ERR_UNREACHABLE)     = -FI_ENETUNREACH;
	MLX_TRANSLATE_ERRCODE (UCS_ERR_INVALID_ADDR)    = -FI_EINVAL;
	MLX_TRANSLATE_ERRCODE (UCS_ERR_NOT_IMPLEMENTED) = -FI_ENOSYS;
	MLX_TRANSLATE_ERRCODE (UCS_ERR_MESSAGE_TRUNCATED) = -FI_EMSGSIZE;
	MLX_TRANSLATE_ERRCODE (UCS_ERR_NO_PROGRESS)     = -FI_EAGAIN;
	MLX_TRANSLATE_ERRCODE (UCS_ERR_BUFFER_TOO_SMALL)= -FI_ETOOSMALL;
	MLX_TRANSLATE_ERRCODE (UCS_ERR_NO_ELEM)         = -FI_ENOENT;
	MLX_TRANSLATE_ERRCODE (UCS_ERR_SOME_CONNECTS_FAILED)   = -FI_EIO;
	MLX_TRANSLATE_ERRCODE (UCS_ERR_NO_DEVICE)       = -FI_ENODEV;
	MLX_TRANSLATE_ERRCODE (UCS_ERR_BUSY)            = -FI_EBUSY;
	MLX_TRANSLATE_ERRCODE (UCS_ERR_CANCELED)        = -FI_ECANCELED;
	MLX_TRANSLATE_ERRCODE (UCS_ERR_SHMEM_SEGMENT)   = -FI_EINVAL;
	MLX_TRANSLATE_ERRCODE (UCS_ERR_ALREADY_EXISTS)  = -EEXIST;
	MLX_TRANSLATE_ERRCODE (UCS_ERR_OUT_OF_RANGE)    = -FI_EINVAL;
	MLX_TRANSLATE_ERRCODE (UCS_ERR_TIMED_OUT)       = -FI_ETIMEDOUT;
	MLX_TRANSLATE_ERRCODE (UCS_ERR_EXCEEDS_LIMIT)   = -FI_E2BIG;
	MLX_TRANSLATE_ERRCODE (UCS_ERR_UNSUPPORTED)     = -FI_ENOSYS;
	return 0;
}


struct fi_domain_attr mlx_domain_attrs = {
	.domain = NULL,
	.name = FI_MLX_FABRIC_NAME,
	.threading = FI_THREAD_SAFE,
	.control_progress = FI_PROGRESS_AUTO,
	.data_progress = FI_PROGRESS_MANUAL,
	.resource_mgmt = FI_RM_DISABLED,
	.av_type = FI_AV_UNSPEC,
	.mr_mode = OFI_MR_BASIC_MAP,
	.mr_key_size = -1, /*Should be setup after init*/
	.tx_ctx_cnt = 1,
	.rx_ctx_cnt = 1,
	.max_ep_tx_ctx = 1,
	.max_ep_rx_ctx = 1,
	.mr_cnt = FI_MLX_DEF_MR_CNT,
};

struct fi_rx_attr mlx_rx_attrs = {
	.caps = FI_MLX_CAPS,
	.mode = FI_MLX_MODE_REQUIRED,
	.op_flags = FI_MLX_OP_FLAGS,
	.msg_order = FI_ORDER_SAS,
	.comp_order = FI_ORDER_NONE,
	.total_buffered_recv = ~(0ULL),
	.size = UINT64_MAX,
	.iov_limit = 1
};

struct fi_tx_attr mlx_tx_attrs = {
	.caps = FI_MLX_CAPS,
	.mode = FI_MLX_MODE_REQUIRED,
	.op_flags = FI_MLX_OP_FLAGS,
	.msg_order = FI_ORDER_SAS,
	.comp_order = FI_ORDER_NONE,
	.inject_size = FI_MLX_DEFAULT_INJECT_SIZE, /*Should be setup after init*/
	.size = UINT64_MAX,
	.iov_limit = 1,
	.rma_iov_limit = 0
};

struct fi_fabric_attr mlx_fabric_attrs = {
	.name = FI_MLX_FABRIC_NAME,
	.prov_version = FI_MLX_VERSION,
	.fabric = NULL
};

struct fi_ep_attr mlx_ep_attrs = {
	.type = FI_EP_RDM,
	.protocol = FI_PROTO_MLX,
#if defined(UCP_API_RELEASE) && (UCP_API_RELEASE <= 2947)
#warning "HPCX 1.9.7 have an issue with UCP_API_VERSION macro"
	.protocol_version = (((UCP_API_MAJOR) << UCP_VERSION_MAJOR_SHIFT)|
			((UCP_API_MINOR) << UCP_VERSION_MINOR_SHIFT)),
#else
	.protocol_version = (UCP_API_VERSION),
#endif
	.max_msg_size = 0xFFFFFFFF,
	.mem_tag_format = 0x0,
	.tx_ctx_cnt = 1,
	.rx_ctx_cnt = 1,
};


struct fi_info mlx_info = {
	.caps = FI_MLX_CAPS,
	.mode = FI_MLX_MODE_REQUIRED,
	.addr_format = FI_ADDR_MLX,
	.src_addrlen = 0,
	.dest_addr = 0,
	.tx_attr = &mlx_tx_attrs,
	.rx_attr = &mlx_rx_attrs,
	.ep_attr = &mlx_ep_attrs,
	.domain_attr = &mlx_domain_attrs,
	.fabric_attr = &mlx_fabric_attrs
};

struct util_prov mlx_util_prov = {
	.prov = &mlx_prov,
	.info = &mlx_info,
	.flags = 0,
};


static int mlx_getinfo (
			uint32_t version, const char *node,
			const char *service, uint64_t flags,
			const struct fi_info *hints, struct fi_info **info)
{
	int status = -ENODATA;
	char *configfile_name = NULL;
	int inject_thresh = -1;

	mlx_descriptor.config = NULL;

	status = fi_param_get( &mlx_prov,
				"mlx_tinject_limit",
				&inject_thresh);
	if (!status)
		inject_thresh = FI_MLX_DEFAULT_INJECT_SIZE;

	FI_INFO( &mlx_prov, FI_LOG_CORE,
		"used inlect size = %d \n", inject_thresh);

	status = fi_param_get( &mlx_prov, "mlx_config", &configfile_name);
	if (!status) {
		configfile_name = NULL;
	}

	/* NS is disabled by default */
	status = fi_param_get( &mlx_prov, "mlx_ns_enable",
			&mlx_descriptor.use_ns);
	if (!status) {
		mlx_descriptor.use_ns = 0;
	}
	status = fi_param_get( &mlx_prov, "mlx_ns_port",
			&mlx_descriptor.ns_port);
	if (!status) {
		mlx_descriptor.ns_port = FI_MLX_DEFAULT_NS_PORT;
	}



	status = ucp_config_read( NULL,
			status? NULL: configfile_name,
			&mlx_descriptor.config);
	if (status != UCS_OK) {
		FI_WARN( &mlx_prov, FI_LOG_CORE,
			"MLX error: invalid config file\n\t%d (%s)\n",
			status, ucs_status_string(status));
	}

	/*Setup some presets*/
	status = ucm_config_modify("MALLOC_HOOKS", "no");
	if (status != UCS_OK) {
		FI_WARN( &mlx_prov, FI_LOG_CORE,
			"MLX error: failed to switch off UCM memory hooks:\t%d (%s)\n",
			status, ucs_status_string(status));
	}

	FI_INFO( &mlx_prov, FI_LOG_CORE,
		"Loaded MLX version %s\n",
		ucp_get_version_string());

#if ENABLE_DEBUG
	if (mlx_descriptor.config &&
			fi_log_enabled( &mlx_prov, FI_LOG_INFO, FI_LOG_CORE)) {
		ucp_config_print( mlx_descriptor.config,
			stderr, "Used MLX configuration", (1<<4)-1);
	}
#endif

	*info = NULL;
	if (node || service) {
		FI_WARN(&mlx_prov, FI_LOG_CORE,
		"fi_getinfo with \"node != NULL \" or \"service != NULL \" is temporary not supported\n");
		node = service = NULL;
		flags = 0;
	}

	/* Only Pure MLX address and IPv4 are supported */
	if (hints) {
		if (hints->addr_format <= FI_SOCKADDR_IN) {
			mlx_descriptor.use_ns = 1;
			mlx_info.addr_format = FI_SOCKADDR_IN;
		} else {
			mlx_info.addr_format = FI_ADDR_MLX;
		}
	}
	

	status = util_getinfo( &mlx_util_prov, version,
				service, node, flags, hints, info);

	return status;
}

void mlx_cleanup(void)
{
	FI_INFO(&mlx_prov, FI_LOG_CORE, "provider goes cleanup sequence\n");
	if (mlx_descriptor.config) {
		ucp_config_release(mlx_descriptor.config);
		mlx_descriptor.config = NULL;
	}
}


struct fi_provider mlx_prov = {
	.name = FI_MLX_FABRIC_NAME,
	.version = FI_MLX_VERSION,
	.fi_version = FI_VERSION(1, 6),
	.getinfo = mlx_getinfo,
	.fabric = mlx_fabric_open,
	.cleanup = mlx_cleanup,
};


MLX_INI
{
	mlx_init_errcodes();
	fi_param_define( &mlx_prov,
			"mlx_config", FI_PARAM_STRING,
			"MLX configuration file name");

	fi_param_define(&mlx_prov,
			"mlx_tinject_limit", FI_PARAM_INT,
			"Maximal tinject message size");

	fi_param_define(&mlx_prov,
			"mlx_ns_port", FI_PARAM_INT,
			"MLX Name server port");

	fi_param_define(&mlx_prov,
			"mlx_ns_enable",FI_PARAM_BOOL,
			"Enforce usage of name server for MLX provider");

	fi_param_define(&mlx_prov,
			"mlx_ns_iface",FI_PARAM_STRING,
			"Specify IPv4 network interface for MLX provider's name server'");
	return &mlx_prov;
}
