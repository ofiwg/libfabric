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
	.ep_flush = 0,
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
	MLX_TRANSLATE_ERRCODE (UCS_ERR_MESSAGE_TRUNCATED) = -FI_ETRUNC;
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
	.resource_mgmt = FI_RM_ENABLED,
	.av_type = FI_AV_UNSPEC,
	.mr_mode = OFI_MR_BASIC_MAP | FI_MR_BASIC,
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
	.iov_limit = 4,
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
	.rma_iov_limit = 1,
};

struct fi_fabric_attr mlx_fabric_attrs = {
	.name = FI_MLX_FABRIC_NAME,
	.prov_version = FI_MLX_VERSION,
	.fabric = NULL
};

struct fi_ep_attr mlx_ep_attrs = {
	.type = FI_EP_UNSPEC,
	.protocol = FI_PROTO_MLX,
	.protocol_version = (((1) << UCP_VERSION_MAJOR_SHIFT)|
			((5) << UCP_VERSION_MINOR_SHIFT)),
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
	size_t use_cache = 1;

	if (mlx_do_extra_checks() != FI_SUCCESS) {
		return -ENODATA;
	}

	status = fi_param_get( &mlx_prov,
				"inject_limit",
				&inject_thresh);
	if (status != FI_SUCCESS)
		inject_thresh = FI_MLX_DEFAULT_INJECT_SIZE;

	FI_INFO( &mlx_prov, FI_LOG_CORE,
		"used inject size = %d \n", inject_thresh);
	mlx_tx_attrs.inject_size = inject_thresh;

	status = fi_param_get( &mlx_prov, "config", &configfile_name);
	if (status != FI_SUCCESS) {
		configfile_name = NULL;
	}

	status = fi_param_get( &mlx_prov, "tls", &tls);
	if (status != FI_SUCCESS) {
		tls = tls_auto;
	}

	if ((strncmp(tls, tls_auto, strlen(tls_auto)) != 0)
		&& (getenv("UCX_TLS") == NULL)) {
		setenv("UCX_TLS", tls, 0);
	}

	status = fi_param_get( &mlx_prov, "ep_flush",
			&mlx_descriptor.ep_flush);
	if (status != FI_SUCCESS) {
		mlx_descriptor.ep_flush = 0;
	}

	status = ucp_config_read( NULL,
			configfile_name,
			&mlx_descriptor.config);
	if (status != UCS_OK) {
		FI_WARN( &mlx_prov, FI_LOG_CORE,
			"MLX error: invalid config file\n\t%d (%s)\n",
			status, ucs_status_string(status));
	}

	FI_INFO( &mlx_prov, FI_LOG_CORE,
		"Loaded MLX version %s\n",
		ucp_get_version_string());

#if ENABLE_DEBUG
	int extra_debug = 0;
	status = fi_param_get( &mlx_prov, "extra_debug", &extra_debug);
	if (status != FI_SUCCESS) {
		extra_debug = 0;
	}
	if (mlx_descriptor.config && extra_debug &&
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
	mlx_info.addr_format = FI_ADDR_MLX;
	if (hints && !((FI_ADDR_MLX == hints->addr_format)
			|| (FI_FORMAT_UNSPEC == hints->addr_format))) {
			FI_WARN(&mlx_prov, FI_LOG_CORE,
				"invalid addr_format requested\n");
			return -ENODATA;
	}

	status = fi_param_get( &mlx_prov, "enable_spawn", &mlx_descriptor.enable_spawn);
	if (status != FI_SUCCESS) {
		mlx_descriptor.enable_spawn = 0;
	}

	FI_WARN( &mlx_prov, FI_LOG_WARN,
			"MLX: spawn support %d \n", mlx_descriptor.enable_spawn);

	status = util_getinfo( &mlx_util_prov, version,
				service, node, flags, hints, info);

	if (*info)
		(*info)->addr_format = mlx_info.addr_format;

	return status;
}

void mlx_cleanup(void)
{
	FI_DBG(&mlx_prov, FI_LOG_CORE, "provider goes cleanup sequence\n");
	if (mlx_descriptor.config) {
		ucp_config_release(mlx_descriptor.config);
		mlx_descriptor.config = NULL;
	}
}


struct fi_provider mlx_prov = {
	.name = FI_MLX_FABRIC_NAME,
	.version = FI_MLX_VERSION,
	.fi_version = FI_VERSION(1, 8),
	.getinfo = mlx_getinfo,
	.fabric = mlx_fabric_open,
	.cleanup = mlx_cleanup,
};


MLX_INI
{
	mlx_init_errcodes();
	fi_param_define( &mlx_prov,
			"config", FI_PARAM_STRING,
			"MLX configuration file name");

	fi_param_define(&mlx_prov,
			"inject_limit", FI_PARAM_INT,
			"Maximal tinject/inject message size");

	fi_param_define(&mlx_prov,
			"ep_flush",FI_PARAM_BOOL,
			"Use EP flush (Disabled by default)");

	fi_param_define(&mlx_prov,
			"extra_debug",FI_PARAM_BOOL,
			"Output transport-level debug information");

	fi_param_define(&mlx_prov,
			"ep_flush",FI_PARAM_BOOL,
			"Use EP flush (Disabled by default)");

	fi_param_define(&mlx_prov,
			"ns_iface",FI_PARAM_STRING,
			"Specify IPv4 network interface for MLX provider's name server'");

	fi_param_define(&mlx_prov,
			"devices", FI_PARAM_STRING,
			"Specifies devices available for MLX provider (Default: auto)");

	fi_param_define(&mlx_prov,
			"enable_spawn",FI_PARAM_BOOL,
			"Enable dynamic process support (Disabled by default)");

	fi_param_define(&mlx_prov,
			"tls",FI_PARAM_STRING,
			"Specifies transports available for MLX provider (Default: auto)");

	return &mlx_prov;
}
