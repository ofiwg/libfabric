/*
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
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

#include <prov/verbs/src/fi_verbs.h>
#include <prov/verbs/src/verbs_checks.h>

struct fi_info *verbs_info = NULL;
static pthread_mutex_t verbs_info_lock = PTHREAD_MUTEX_INITIALIZER;

static const char *local_node = "localhost";
static char def_tx_ctx_size[16] = "384";
static char def_rx_ctx_size[16] = "384";
static char def_tx_iov_limit[16] = "4";
static char def_rx_iov_limit[16] = "4";
static char def_inject_size[16] = "64";

const struct fi_fabric_attr verbs_fabric_attr = {
    .name = VERBS_PROV_NAME,
    .prov_version = VERBS_PROV_VERS,
};

const struct fi_domain_attr verbs_domain_attr = {
    .threading = FI_THREAD_SAFE,
    .control_progress = FI_PROGRESS_AUTO,
    .data_progress = FI_PROGRESS_AUTO,
    .mr_mode = FI_MR_BASIC,
    .mr_key_size = sizeof_field(struct ibv_sge, lkey),
    .cq_data_size = sizeof_field(struct ibv_send_wr, imm_data),
    .tx_ctx_cnt = 1024,
    .rx_ctx_cnt = 1024,
    .max_ep_tx_ctx = 1,
    .max_ep_rx_ctx = 1,
};

const struct fi_ep_attr verbs_ep_attr = {
    .type = FI_EP_MSG,
    .protocol_version = 1,
    .msg_prefix_size = 0,
    .max_order_war_size = 0,
    .mem_tag_format = 0,
    .tx_ctx_cnt = 1,
    .rx_ctx_cnt = 1,
};

const struct fi_rx_attr verbs_rx_attr = {
    .caps = VERBS_CAPS,
    .mode = VERBS_RX_MODE,
    .msg_order = VERBS_MSG_ORDER,
    .total_buffered_recv = 0,
};

const struct fi_tx_attr verbs_tx_attr = {
    .caps = VERBS_CAPS,
    .mode = VERBS_TX_MODE,
    .op_flags = VERBS_TX_OP_FLAGS,
    .msg_order = VERBS_MSG_ORDER,
    .inject_size = 0,
    .rma_iov_limit = 1,
};

static int fi_ibv_check_hints(const struct fi_info *hints,
                              const struct fi_info *info)
{
    int ret;

    if (hints->caps & ~(info->caps)) {
        FI_INFO(&fi_ibv_prov, FI_LOG_CORE, "Unsupported capabilities\n");
        return -FI_ENODATA;
    }

    if (info->ep_attr->type == FI_EP_RDM && hints->caps & ~(VERBS_EP_RDM_CAPS))
    {
        FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
                "Unsupported capabilities for FI_EP_RDM\n");
        return -FI_ENODATA;
    }

    if (info->ep_attr->type == FI_EP_MSG && hints->caps & ~(VERBS_EP_MSG_CAPS))
    {
        FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
                "Unsupported capabilities for FI_EP_MSG\n");
        return -FI_ENODATA;
    }

    if (info->ep_attr->type == FI_EP_MSG &&
        (hints->mode & info->mode) != info->mode) {
        FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
                "Required hints mode bits not set. Expected:0x%llx"
                " Given:0x%llx\n", info->mode, hints->mode);
        return -FI_ENODATA;
    }

    if (hints->fabric_attr) {
        ret = fi_ibv_check_fabric_attr(hints->fabric_attr, info);
        if (ret)
            return ret;
    }

    if (hints->domain_attr) {
        ret = fi_ibv_check_domain_attr(hints->domain_attr, info);
        if (ret)
            return ret;
    }

    if (hints->ep_attr) {
        ret = fi_ibv_check_ep_attr(hints->ep_attr, info);
        if (ret)
            return ret;
    }

    if (hints->rx_attr) {
        ret = fi_ibv_check_rx_attr(hints->rx_attr, hints, info);
        if (ret)
            return ret;
    }

    if (hints->tx_attr) {
        ret = fi_ibv_check_tx_attr(hints->tx_attr, hints, info);
        if (ret)
            return ret;
    }

    return 0;
}

static int fi_ibv_fi_to_rai(const struct fi_info *fi, uint64_t flags,
                            struct rdma_addrinfo *rai)
{
    fi_ibv_memset(rai, 0, sizeof *rai);
    if (flags & FI_SOURCE)
        rai->ai_flags = RAI_PASSIVE;
    if (flags & FI_NUMERICHOST)
        rai->ai_flags |= RAI_NUMERICHOST;

    rai->ai_qp_type = IBV_QPT_RC;
    rai->ai_port_space = RDMA_PS_TCP;

    if (!fi)
        return 0;

    switch (fi->addr_format) {
    case FI_SOCKADDR_IN:
        rai->ai_family = AF_INET;
        rai->ai_flags |= RAI_FAMILY;
        break;
    case FI_SOCKADDR_IN6:
        rai->ai_family = AF_INET6;
        rai->ai_flags |= RAI_FAMILY;
        break;
    case FI_SOCKADDR_IB:
        rai->ai_family = AF_IB;
        rai->ai_flags |= RAI_FAMILY;
        break;
    case FI_SOCKADDR:
        if (fi->src_addrlen) {
            rai->ai_family = ((struct sockaddr *)fi->src_addr)->sa_family;
            rai->ai_flags |= RAI_FAMILY;
        } else if (fi->dest_addrlen) {
            rai->ai_family = ((struct sockaddr *)fi->dest_addr)->sa_family;
            rai->ai_flags |= RAI_FAMILY;
        }
        break;
    case FI_FORMAT_UNSPEC:
        break;
    default:
        VERBS_INFO(FI_LOG_FABRIC, "Unknown fi->addr_format\n");
    }

    if (fi->src_addrlen) {
        if (!(rai->ai_src_addr = malloc(fi->src_addrlen)))
            return -FI_ENOMEM;
        fi_ibv_memcpy(rai->ai_src_addr, fi->src_addr, fi->src_addrlen);
        rai->ai_src_len = fi->src_addrlen;
    }
    if (fi->dest_addrlen) {
        if (!(rai->ai_dst_addr = malloc(fi->dest_addrlen)))
            return -FI_ENOMEM;
        fi_ibv_memcpy(rai->ai_dst_addr, fi->dest_addr, fi->dest_addrlen);
        rai->ai_dst_len = fi->dest_addrlen;
    }

    return 0;
}

static int fi_ibv_rai_to_fi(struct rdma_addrinfo *rai, struct fi_info *fi)
{
    switch (rai->ai_family) {
    case AF_INET:
        fi->addr_format = FI_SOCKADDR_IN;
        break;
    case AF_INET6:
        fi->addr_format = FI_SOCKADDR_IN6;
        break;
    case AF_IB:
        fi->addr_format = FI_SOCKADDR_IB;
        break;
    default:
        FI_INFO(&fi_ibv_prov, FI_LOG_CORE, "Unknown rai->ai_family\n");
    }

    if (rai->ai_src_len) {
        if (!(fi->src_addr = malloc(rai->ai_src_len)))
            return -FI_ENOMEM;
        fi_ibv_memcpy(fi->src_addr, rai->ai_src_addr, rai->ai_src_len);
        fi->src_addrlen = rai->ai_src_len;
    }
    if (rai->ai_dst_len) {
        if (!(fi->dest_addr = malloc(rai->ai_dst_len)))
            return -FI_ENOMEM;
        fi_ibv_memcpy(fi->dest_addr, rai->ai_dst_addr, rai->ai_dst_len);
        fi->dest_addrlen = rai->ai_dst_len;
    }

    return 0;
}

static inline int fi_ibv_get_qp_cap(struct ibv_context *ctx,
                                    struct ibv_device_attr *device_attr,
                                    struct fi_info *info)
{
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_qp_init_attr init_attr;
    int ret = 0;

    pd = ibv_alloc_pd(ctx);
    if (!pd) {
        VERBS_INFO_ERRNO(FI_LOG_FABRIC, "ibv_alloc_pd", errno);
        return -errno;
    }

    cq = ibv_create_cq(ctx, 1, NULL, NULL, 0);
    if (!cq) {
        VERBS_INFO_ERRNO(FI_LOG_FABRIC, "ibv_create_cq", errno);
        ret = -errno;
        goto err1;
    }

    /* TODO: serialize access to string buffers */
    fi_read_file(FI_CONF_DIR, "def_tx_ctx_size",
                 def_tx_ctx_size, sizeof def_tx_ctx_size);
    fi_read_file(FI_CONF_DIR, "def_rx_ctx_size",
                 def_rx_ctx_size, sizeof def_rx_ctx_size);
    fi_read_file(FI_CONF_DIR, "def_tx_iov_limit",
                 def_tx_iov_limit, sizeof def_tx_iov_limit);
    fi_read_file(FI_CONF_DIR, "def_rx_iov_limit",
                 def_rx_iov_limit, sizeof def_rx_iov_limit);
    fi_read_file(FI_CONF_DIR, "def_inject_size",
                 def_inject_size, sizeof def_inject_size);

    fi_ibv_memset(&init_attr, 0, sizeof init_attr);
    init_attr.send_cq = cq;
    init_attr.recv_cq = cq;
    init_attr.cap.max_send_wr = atoi(def_tx_ctx_size);
    init_attr.cap.max_recv_wr = atoi(def_rx_ctx_size);
    init_attr.cap.max_send_sge =
        MIN(atoi(def_tx_iov_limit), device_attr->max_sge);
    init_attr.cap.max_recv_sge =
        MIN(atoi(def_rx_iov_limit), device_attr->max_sge);
    init_attr.cap.max_inline_data = atoi(def_inject_size);
    init_attr.qp_type = IBV_QPT_RC;

    qp = ibv_create_qp(pd, &init_attr);
    if (!qp) {
        VERBS_INFO_ERRNO(FI_LOG_FABRIC, "ibv_create_qp", errno);
        ret = -errno;
        goto err2;
    }

    info->tx_attr->inject_size = init_attr.cap.max_inline_data;
    info->tx_attr->iov_limit = init_attr.cap.max_send_sge;
    info->tx_attr->size = init_attr.cap.max_send_wr;

    info->rx_attr->iov_limit = init_attr.cap.max_recv_sge;
    info->rx_attr->size = init_attr.cap.max_recv_wr;

    ibv_destroy_qp(qp);
 err2:
    ibv_destroy_cq(cq);
 err1:
    ibv_dealloc_pd(pd);

    return ret;
}

static int fi_ibv_get_device_attrs(struct ibv_context *ctx,
                                   struct fi_info *info)
{
    struct ibv_device_attr device_attr;
    struct ibv_port_attr port_attr;
    int ret = 0;

    ret = ibv_query_device(ctx, &device_attr);
    if (ret) {
        VERBS_INFO_ERRNO(FI_LOG_FABRIC, "ibv_query_device", errno);
        return -errno;
    }

    info->domain_attr->cq_cnt = device_attr.max_cq;
    info->domain_attr->ep_cnt = device_attr.max_qp;
    info->domain_attr->tx_ctx_cnt =
        MIN(info->domain_attr->tx_ctx_cnt, device_attr.max_qp);
    info->domain_attr->rx_ctx_cnt =
        MIN(info->domain_attr->rx_ctx_cnt, device_attr.max_qp);
    info->domain_attr->max_ep_tx_ctx = device_attr.max_qp;
    info->domain_attr->max_ep_rx_ctx = device_attr.max_qp;

    ret = fi_ibv_get_qp_cap(ctx, &device_attr, info);
    if (ret)
        return ret;
    int i;
    int port_up = 0;
    for (i = 1; i <= device_attr.phys_port_cnt; i++) {
        ret = ibv_query_port(ctx, i, &port_attr);
        if (ret) {
            VERBS_INFO_ERRNO(FI_LOG_FABRIC, "ibv_query_port", errno);
            return -errno;
        }

        if (port_attr.state == IBV_PORT_ACTIVE) {
            port_up = i;
            break;
        }
    }

    if (!port_up) {
        return -FI_ENODEV;
    }
    info->ep_attr->max_msg_size = port_attr.max_msg_sz;
    info->ep_attr->max_order_raw_size = port_attr.max_msg_sz;
    info->ep_attr->max_order_waw_size = port_attr.max_msg_sz;

    return 0;
}

/*
 * USNIC plugs into the verbs framework, but is not a usable device.
 * Manually check for devices and fail gracefully if none are present.
 * This avoids the lower libraries (libibverbs and librdmacm) from
 * reporting error messages to stderr.
 */
static int fi_ibv_have_device(void)
{
    struct ibv_device **devs;
    struct ibv_context *verbs;
    int i, ret = 0;

    devs = ibv_get_device_list(NULL);
    if (!devs)
        return 0;

    for (i = 0; devs[i]; i++) {
        verbs = ibv_open_device(devs[i]);
        if (verbs) {
            ibv_close_device(verbs);
            ret = 1;
            break;
        }
    }

    ibv_free_device_list(devs);
    return ret;
}

static int fi_ibv_get_info_ctx(struct ibv_context *ctx, struct fi_info **info,
                               int init_ep_type)
{
    struct fi_info *fi;
    union ibv_gid gid;
    size_t name_len;
    int ret;

    if (!(fi = fi_allocinfo()))
        return -FI_ENOMEM;

    fi->caps = VERBS_CAPS;
    fi->mode = init_ep_type == FI_EP_RDM ? VERBS_EP_RDM_MODE : VERBS_MODE;
    fi->handle = NULL;
    *(fi->tx_attr) = verbs_tx_attr;
    *(fi->rx_attr) = verbs_rx_attr;
    *(fi->ep_attr) = verbs_ep_attr;
    fi->ep_attr->type = init_ep_type;
    *(fi->domain_attr) = verbs_domain_attr;
    *(fi->fabric_attr) = verbs_fabric_attr;
    fi->fabric_attr->name = strdup(verbs_fabric_attr.name);

    ret = fi_ibv_get_device_attrs(ctx, fi);
    if (ret)
        goto err;

    if (init_ep_type == FI_EP_RDM) {
        fi->tx_attr->mode &= ~FI_LOCAL_MR;
        fi->rx_attr->mode &= ~FI_LOCAL_MR;
        /* TODO implement tagged inject, all the stuff is already in place */
        fi->tx_attr->inject_size = 0;
    }

    switch (ctx->device->transport_type) {
    case IBV_TRANSPORT_IB:
        if (ibv_query_gid(ctx, 1, 0, &gid)) {
            VERBS_INFO_ERRNO(FI_LOG_FABRIC, "ibv_query_gid", errno);
            ret = -errno;
            goto err;
        }

        name_len = strlen(VERBS_IB_PREFIX) + INET6_ADDRSTRLEN;

        if (!(fi->fabric_attr->name = calloc(1, name_len + 1))) {
            ret = -FI_ENOMEM;
            goto err;
        }

        snprintf(fi->fabric_attr->name, name_len, VERBS_IB_PREFIX "%lx",
                 gid.global.subnet_prefix);

        fi->ep_attr->protocol = FI_PROTO_RDMA_CM_IB_RC;
        break;
    case IBV_TRANSPORT_IWARP:
        fi->fabric_attr->name = strdup(VERBS_IWARP_FABRIC);
        if (!fi->fabric_attr->name) {
            ret = -FI_ENOMEM;
            goto err;
        }

        fi->ep_attr->protocol = FI_PROTO_IWARP;
        fi->tx_attr->op_flags = VERBS_TX_OP_FLAGS_IWARP;
        break;
    default:
        FI_INFO(&fi_ibv_prov, FI_LOG_CORE, "Unknown transport type\n");
        ret = -FI_ENODATA;
        goto err;
    }

    if (!(fi->domain_attr->name = strdup(ctx->device->name))) {
        ret = -FI_ENOMEM;
        goto err;
    }

    *info = fi;
    return 0;
 err:
    fi_freeinfo(fi);
    return ret;
}

int fi_ibv_init_info(int init_ep_type)
{
    struct ibv_context **ctx_list;
    struct fi_info *fi = NULL, *tail = NULL;
    int ret = 0, i, num_devices;

    if (verbs_info)
        return 0;

    pthread_mutex_lock(&verbs_info_lock);
    if (verbs_info)
        goto unlock;

    if (!fi_ibv_have_device()) {
        VERBS_INFO(FI_LOG_FABRIC, "No RDMA devices found\n");
        ret = -FI_ENODATA;
        goto unlock;
    }

    ctx_list = rdma_get_devices(&num_devices);
    if (!num_devices) {
        VERBS_INFO_ERRNO(FI_LOG_FABRIC, "rdma_get_devices", errno);
        ret = -errno;
        goto unlock;
    }

    for (i = 0; i < num_devices; i++) {
        ret = fi_ibv_get_info_ctx(ctx_list[i], &fi, init_ep_type);
        if (!ret) {
            if (!verbs_info) {
                verbs_info = fi;
            } else {
                tail->next = fi;
            }
            tail = fi;
        }
    }

    ret = verbs_info ? 0 : ret;
    rdma_free_devices(ctx_list);
 unlock:
    pthread_mutex_unlock(&verbs_info_lock);
    return ret;
}

int fi_ibv_create_ep(const char *node, const char *service,
                     uint64_t flags, const struct fi_info *hints,
                     struct rdma_addrinfo **rai, struct rdma_cm_id **id)
{
    struct rdma_addrinfo rai_hints, *_rai;
    struct rdma_addrinfo **rai_current;
    int ret;

    ret = fi_ibv_fi_to_rai(hints, flags, &rai_hints);
    if (ret)
        goto out;

    if (!node && !rai_hints.ai_dst_addr) {
        if (!rai_hints.ai_src_addr) {
            node = local_node;
        }
        rai_hints.ai_flags |= RAI_PASSIVE;
    }

    ret = rdma_getaddrinfo((char *)node, (char *)service, &rai_hints, &_rai);
    if (ret) {
        VERBS_INFO_ERRNO(FI_LOG_FABRIC, "rdma_getaddrinfo", errno);
        ret = -errno;
        goto out;
    }
    /*
     * If caller requested rai, remove ib_rai entries added by IBACM to
     * prevent wrong ib_connect_hdr from being sent in connect request.
     */
    if (rai && hints && (hints->addr_format != FI_SOCKADDR_IB)) {
        for (rai_current = &_rai; *rai_current;) {
            struct rdma_addrinfo *rai_next;
            if ((*rai_current)->ai_family == AF_IB) {
                rai_next = (*rai_current)->ai_next;
                (*rai_current)->ai_next = NULL;
                rdma_freeaddrinfo(*rai_current);
                *rai_current = rai_next;
                continue;
            }
            rai_current = &(*rai_current)->ai_next;
        }
    }

    ret = rdma_create_ep(id, _rai, NULL, NULL);
    if (ret) {
        VERBS_INFO_ERRNO(FI_LOG_FABRIC, "rdma_create_ep", errno);
        goto err;
    }

    if (rai) {
        *rai = _rai;
        goto out;
    }
 err:
    rdma_freeaddrinfo(_rai);
 out:
    if (rai_hints.ai_src_addr)
        free(rai_hints.ai_src_addr);
    if (rai_hints.ai_dst_addr)
        free(rai_hints.ai_dst_addr);
    return ret;
}

static
int fi_ibv_get_matching_info(struct fi_info *check_info,
                             struct fi_info *hints, struct rdma_addrinfo *rai,
                             struct fi_info **info)
{

    int ret;
    struct fi_info *fi, *tail;

    *info = tail = NULL;

    for (; check_info; check_info = check_info->next) {
        if (hints) {
            ret = fi_ibv_check_hints(hints, check_info);
            if (ret)
                continue;
        }

        if (!(fi = fi_dupinfo(check_info))) {
            ret = -FI_ENOMEM;
            goto err1;
        }

        ret = fi_ibv_rai_to_fi(rai, fi);
        if (ret)
            goto err2;

        fi_ibv_update_info(hints, fi);

        if (!*info)
            *info = fi;
        else
            tail->next = fi;
        tail = fi;
    }

    if (!*info)
        return -FI_ENODATA;

    return 0;
 err2:
    fi_freeinfo(fi);
 err1:
    fi_freeinfo(*info);
    return ret;
}

int fi_ibv_getinfo(uint32_t version, const char *node, const char *service,
                   uint64_t flags, struct fi_info *hints,
                   struct fi_info **info)
{
    struct rdma_cm_id *id;
    struct rdma_addrinfo *rai;
    struct fi_info *check_info;
    int ret;
    int init_ep_type = FI_EP_UNSPEC;

    if (hints && hints->ep_attr) {
        if (hints->ep_attr->type == FI_EP_UNSPEC) {
            FI_WARN(&fi_ibv_prov, FI_LOG_CORE,
                    "hints->ep_attr->type is set to FI_EP_UNSPEC."
                    "Default ep_type will be FI_EP_MSG\n");
            init_ep_type = FI_EP_MSG;
        } else {
            init_ep_type = hints->ep_attr->type;
        }
    }

    ret = fi_ibv_init_info(init_ep_type);
    if (ret)
        goto err1;

    ret = fi_ibv_create_ep(node, service, flags, hints, &rai, &id);
    if (ret)
        goto err1;

    check_info = id->verbs ?
        fi_ibv_search_verbs_info(NULL, ibv_get_device_name(id->verbs->device))
        : verbs_info;

    if (!check_info) {
        VERBS_DBG(FI_LOG_FABRIC, "Unable to find check_info\n");
        ret = -FI_ENODATA;
        goto err2;
    }
    ret = fi_ibv_get_matching_info(check_info, hints, rai, info);
 err2:
    rdma_destroy_ep(id);
    rdma_freeaddrinfo(rai);
 err1:
    if (!ret || ret == -FI_ENOMEM)
        return ret;
    else
        return -FI_ENODATA;
}
