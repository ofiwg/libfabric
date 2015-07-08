/*
 * Copyright (c) 2013 Intel Corporation. All rights reserved.
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

#include "mlxm.h"
#include "fi.h"
#include "prov.h"


#define MLXM_EP_CAP_BASE (FI_TAGGED          |                          \
                          FI_SEND            | FI_RECV)


#define MLXM_EP_CAP_OPT1 ( FI_MULTI_RECV)



//#define MLXM_EP_CAP	(MLXM_EP_CAP_BASE | MLXM_EP_CAP_OPT1 | MLXM_EP_CAP_OPT2)
#define MLXM_EP_CAP	(MLXM_EP_CAP_BASE)



uint64_t mlxm_mem_tag_format;
struct mlxm_globals mlxm_globals = {0,0};
static int mlxm_getinfo(uint32_t version, const char *node,
                        const char *service, uint64_t flags,
                        struct fi_info *hints, struct fi_info **info)
{
        struct fi_info	*mlxm_info;
        int type = FI_EP_RDM;
        int ep_cap = 0;
        int err = -ENODATA;
        mlxm_mem_tag_format = MLXM_MEM_TAG_FORMAT;
        FI_INFO(&mlxm_prov, FI_LOG_CORE,"\n");

        *info = NULL;

        if (node) {
                FI_WARN(&mlxm_prov, FI_LOG_CORE,
                        "fi_getinfo with \"node != NULL \" is not supported\n");
                /* TODO: clarify "node" parameter usage */
                goto err_out;
        }

        if (hints) {
                if (hints->ep_attr) {
                        switch (hints->ep_attr->type) {
                        case FI_EP_UNSPEC:
                        case FI_EP_RDM:
                                break;
                        default:
                                FI_WARN(&mlxm_prov, FI_LOG_CORE, "unsupported ep type required: %d\n",
                                        (int)hints->ep_attr->type);
                                goto err_out;
                        }

                        if (hints->ep_attr) {
                                switch (hints->ep_attr->protocol) {
                                case FI_PROTO_UNSPEC:
                                        break;
                                default:
                                        FI_WARN(&mlxm_prov, FI_LOG_CORE, "unsupported ep protoclo required: %d\n",
                                                (int)hints->ep_attr->protocol);
                                        goto err_out;

                                }
                                mlxm_mem_tag_format = hints->ep_attr->mem_tag_format > 0 ?
                                        hints->ep_attr->mem_tag_format :
                                        MLXM_MEM_TAG_FORMAT;

                                switch (hints->ep_attr->protocol) {
                                case FI_PROTO_UNSPEC:
                                        break;
                                default:
                                        FI_WARN(&mlxm_prov, FI_LOG_CORE,
                                                "hints->protocol=%d, supported=%d %d\n",
                                                hints->ep_attr->protocol,
                                                FI_PROTO_UNSPEC, FI_PROTO_PSMX);
                                        goto err_out;
                                }

                                if (hints->ep_attr->tx_ctx_cnt > 1) {
                                        FI_WARN(&mlxm_prov, FI_LOG_CORE,
                                                "hints->ep_attr->tx_ctx_cnt=%d, supported=0,1\n",
                                                hints->ep_attr->tx_ctx_cnt);
                                        goto err_out;
                                }

                                if (hints->ep_attr->rx_ctx_cnt > 1) {
                                        FI_WARN(&mlxm_prov, FI_LOG_CORE,
                                                "hints->ep_attr->rx_ctx_cnt=%d, supported=0,1\n",
                                                hints->ep_attr->rx_ctx_cnt);
                                        goto err_out;
                                }
                        }
                }

                if (hints->tx_attr &&
                    (hints->tx_attr->op_flags & MLXM_SUPPORTED_FLAGS) !=
                    hints->tx_attr->op_flags) {
                        FI_WARN(&mlxm_prov, FI_LOG_CORE, "unsupported tx_attr->flags required: 0x%llx, supported: 0x%llx\n",
                                (long long unsigned)hints->tx_attr->op_flags, MLXM_SUPPORTED_FLAGS);
                        goto err_out;

                }

                if (hints->rx_attr &&
                    (hints->rx_attr->op_flags & MLXM_SUPPORTED_FLAGS) !=
                    hints->rx_attr->op_flags) {
                        FI_WARN(&mlxm_prov, FI_LOG_CORE, "unsupported rx_attr->flags required: 0x%llx, supported: 0x%llx\n",
                                (long long unsigned)hints->rx_attr->op_flags, MLXM_SUPPORTED_FLAGS);
                        goto err_out;

                }


                if (hints->domain_attr &&
                    hints->domain_attr->name &&
                    strncmp(hints->domain_attr->name, "mxm", 3)) {
                        FI_WARN(&mlxm_prov, FI_LOG_CORE, "incorrect domain name: %s, correct: mxm\n",
                                hints->domain_attr->name);
                        goto err_out;

                }


                if ((hints->caps & MLXM_EP_CAP) != hints->caps) {
                        FI_WARN(&mlxm_prov, FI_LOG_CORE, "unsupported ep caps: 0x%llx, supported: 0x%llx\n",
                                (long long unsigned)hints->caps, MLXM_EP_CAP);
                        goto err_out;

                }

                if ((hints->mode & FI_CONTEXT) != FI_CONTEXT) {
                        FI_INFO(&mlxm_prov, FI_LOG_CORE,
				"hints->mode=0x%llx, required=0x%llx\n",
                                hints->mode, FI_CONTEXT);
			goto err_out;
		}


                ep_cap = hints->caps;
        }


        mlxm_info = fi_allocinfo();
        if (!mlxm_info)
                return -ENOMEM;


        mlxm_info->ep_attr->protocol             = FI_PROTO_UNSPEC;
        mlxm_info->ep_attr->max_msg_size         = 0xFFFFFFFF;
        mlxm_info->ep_attr->mem_tag_format       = mlxm_mem_tag_format;
        mlxm_info->ep_attr->type                 = type;
        mlxm_info->ep_attr->tx_ctx_cnt           = 1;
        mlxm_info->ep_attr->rx_ctx_cnt           = 1;


        mlxm_info->domain_attr->threading        = FI_THREAD_UNSPEC;
        mlxm_info->domain_attr->control_progress = FI_PROGRESS_MANUAL;
        mlxm_info->domain_attr->data_progress    = FI_PROGRESS_MANUAL;
        mlxm_info->domain_attr->name             = strdup("mxm");

        mlxm_info->next                          = NULL;
        mlxm_info->caps                          = (hints && hints->caps) ? hints->caps : ep_cap;
        mlxm_info->mode                          = FI_CONTEXT;
        mlxm_info->addr_format                   = FI_FORMAT_UNSPEC;
        /* TODO: clarify what are the following 4 members for? */
        mlxm_info->src_addrlen                   = 0;
        mlxm_info->dest_addrlen                  = 0;
        mlxm_info->src_addr                      = NULL;
        mlxm_info->dest_addr                     = NULL;

        mlxm_info->fabric_attr->name             = strdup("mxm");
        mlxm_info->fabric_attr->prov_name        = strdup("mxm");

        mlxm_info->tx_attr->op_flags             = (hints && hints->tx_attr && hints->tx_attr->op_flags) ? hints->tx_attr->op_flags : 0;
        mlxm_info->tx_attr->caps                 = mlxm_info->caps;
        mlxm_info->tx_attr->mode                 = mlxm_info->mode;
        mlxm_info->tx_attr->msg_order            = FI_ORDER_SAS;
        mlxm_info->tx_attr->comp_order           = FI_ORDER_NONE;
        mlxm_info->tx_attr->inject_size          = 32; /* imm data ? */
        mlxm_info->tx_attr->size                 = UINT64_MAX;
        mlxm_info->tx_attr->iov_limit            = 1; /* TODO : mxm supports any*/


        mlxm_info->rx_attr->op_flags             = (hints && hints->rx_attr && hints->rx_attr->op_flags) ? hints->rx_attr->op_flags : 0;
        mlxm_info->rx_attr->caps                 = mlxm_info->caps;
        mlxm_info->rx_attr->mode                 = mlxm_info->mode;
        mlxm_info->rx_attr->msg_order            = FI_ORDER_SAS;
        mlxm_info->rx_attr->comp_order           = FI_ORDER_NONE;
        mlxm_info->rx_attr->total_buffered_recv  = ~(0ULL); /*TODO: clarify  */
	mlxm_info->rx_attr->size = UINT64_MAX;
        mlxm_info->rx_attr->iov_limit = 1; /*TODO */

        *info = mlxm_info;

	return 0;
err_out:
        return err;
}

static int mlxm_fabric_close(fid_t fid)
{
        mxm_ep_powerdown(mlxm_globals.mxm_ep);
        mxm_ep_destroy(mlxm_globals.mxm_ep);
        mxm_cleanup(mlxm_globals.mxm_context);

        free(fid);
	return 0;
}

static struct fi_ops mlxm_fabric_fi_ops = {
        .size  = sizeof(struct fi_ops),
        .close = mlxm_fabric_close    ,
};

static struct fi_ops_fabric mlxm_fabric_ops = {
        .size   = sizeof(struct fi_ops_fabric),
        .domain = mlxm_domain_open            ,
};

static int mlxm_fabric(struct fi_fabric_attr *attr,
		       struct fid_fabric **fabric, void *context)
{
        mlxm_fid_fabric_t *fabric_priv;

        FI_INFO(&mlxm_prov, FI_LOG_CORE, "\n");
        if (strncmp(attr->name, "mxm", 3))
		return -FI_ENODATA;

	fabric_priv = calloc(1, sizeof(*fabric_priv));
	if (!fabric_priv)
		return -FI_ENOMEM;

        fabric_priv->fabric.fid.fclass  = FI_CLASS_FABRIC;
	fabric_priv->fabric.fid.context = context;
        fabric_priv->fabric.fid.ops     = &mlxm_fabric_fi_ops;
        fabric_priv->fabric.ops         = &mlxm_fabric_ops;
        *fabric                         = &fabric_priv->fabric;
	return 0;
}

struct fi_provider mlxm_prov = {
        .name    = "mxm"          ,
        .getinfo = mlxm_getinfo   ,
        .version = FI_VERSION(0, 2),
        .fi_version = FI_VERSION(FI_MAJOR_VERSION,FI_MINOR_VERSION),
        .fabric  = mlxm_fabric    ,
};


MXM_INI
{
    mxm_context_opts_t	*context_opts;
    mxm_ep_opts_t	*ep_opts;
    mxm_error_t		mxm_err;

    mxm_err = mxm_config_read_opts(&context_opts, &ep_opts, NULL, NULL, 0);
    if (mxm_err != MXM_OK) {
            FI_WARN(&mlxm_prov, FI_LOG_DOMAIN,
                    "mxm_config_read_opts returns %d, errno %d\n",
                    mxm_err, errno);
        goto err_out_free_config;
    }
#if 0
    /* Helps to suppress MXM Warnings*/
    extern struct mxm_global_opts mxm_global_opts;
    mxm_global_opts.log_level = MXM_LOG_LEVEL_FATAL;
#endif

    mxm_err = mxm_init(context_opts, &mlxm_globals.mxm_context);
    if (mxm_err != MXM_OK) {
            FI_WARN(&mlxm_prov,FI_LOG_DOMAIN,
                    "mxm_init returns %d, errno %d\n",
                    mxm_err, errno);

        goto err_out_free_config;
    }

    FI_TRACE(&mlxm_prov, FI_LOG_DOMAIN,
             "MXM context initialized, %p\n",
             (void*)mlxm_globals.mxm_context);

    mxm_err = mxm_ep_create(mlxm_globals.mxm_context,
                            ep_opts, &mlxm_globals.mxm_ep);
    if (mxm_err != MXM_OK) {
            FI_WARN(&mlxm_prov, FI_LOG_DOMAIN,
                    "mxm_ep_create returns %d, errno %d\n",
                    mxm_err, errno);

        goto err_out_cleanup_context;
    }

    FI_TRACE(&mlxm_prov,FI_LOG_DOMAIN,
             "MXM endpoint created, %p\n",
             (void*)mlxm_globals.mxm_ep);

    mxm_config_free_ep_opts(ep_opts);
    mxm_config_free_context_opts(context_opts);

    mlxm_mq_storage_init();

    FI_TRACE(&mlxm_prov,FI_LOG_DOMAIN,
             "MLXM MQ storage initialized\n");


    return &mlxm_prov;
err_out_cleanup_context:
    mxm_cleanup(mlxm_globals.mxm_context);
err_out_free_config:
    mxm_config_free_context_opts(context_opts);
    mxm_config_free_ep_opts(ep_opts);
    return NULL;
}

