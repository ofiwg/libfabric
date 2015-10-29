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

#include <infiniband/verbs.h>

#include <rdma/fabric.h>
#include <rdma/fi_log.h>

#include <prov/verbs/src/fi_verbs.h>
#include <prov/verbs/src/verbs_queuing.h>

int fi_ibv_sockaddr_len(struct sockaddr *addr)
{
    if (!addr)
        return 0;

    switch (addr->sa_family) {
    case AF_INET:
        return sizeof(struct sockaddr_in);
    case AF_INET6:
        return sizeof(struct sockaddr_in6);
    case AF_IB:
        return sizeof(struct sockaddr_ib);
    default:
        return 0;
    }
}

extern struct fi_info *verbs_info;

struct fi_info *fi_ibv_search_verbs_info(const char *fabric_name,
                                         const char *domain_name)
{
    struct fi_info *info;

    for (info = verbs_info; info; info = info->next) {
        if ((!domain_name || !strcmp(info->domain_attr->name, domain_name)) &&
            (!fabric_name || !strcmp(info->fabric_attr->name, fabric_name))) {
            return info;
        }
    }

    return NULL;
}

int fi_ibv_open_device_by_name(struct fi_ibv_domain *domain, const char *name)
{
    struct ibv_context **dev_list;
    int i, ret = -FI_ENODEV;

    if (!name)
        return -FI_EINVAL;

    dev_list = rdma_get_devices(NULL);
    if (!dev_list)
        return -errno;

    for (i = 0; dev_list[i]; i++) {
        if (!strcmp(name, ibv_get_device_name(dev_list[i]->device))) {
            domain->verbs = dev_list[i];
            ret = 0;
            break;
        }
    }
    rdma_free_devices(dev_list);
    return ret;
}

int fi_ibv_copy_addr(void *dst_addr, size_t * dst_addrlen, void *src_addr)
{
    size_t src_addrlen = fi_ibv_sockaddr_len(src_addr);

    if (*dst_addrlen == 0) {
        *dst_addrlen = src_addrlen;
        return -FI_ETOOSMALL;
    }

    if (*dst_addrlen < src_addrlen) {
        fi_ibv_memcpy(dst_addr, src_addr, *dst_addrlen);
    } else {
        fi_ibv_memcpy(dst_addr, src_addr, src_addrlen);
    }
    *dst_addrlen = src_addrlen;
    return 0;
}

void fi_ibv_update_info(const struct fi_info *hints, struct fi_info *info)
{
    if (hints) {
        if (hints->ep_attr) {
            if (hints->ep_attr->tx_ctx_cnt)
                info->ep_attr->tx_ctx_cnt = hints->ep_attr->tx_ctx_cnt;
            if (hints->ep_attr->rx_ctx_cnt)
                info->ep_attr->rx_ctx_cnt = hints->ep_attr->rx_ctx_cnt;
        }

        if (hints->tx_attr)
            info->tx_attr->op_flags = hints->tx_attr->op_flags;

        if (hints->rx_attr)
            info->rx_attr->op_flags = hints->rx_attr->op_flags;

        if (hints->handle)
            info->handle = hints->handle;

        if (hints->domain_attr) {
            info->domain_attr->threading = hints->domain_attr->threading;
        }
    } else {
        info->tx_attr->op_flags = 0;
        info->rx_attr->op_flags = 0;
    }
}

int fi_ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr,
                     struct ibv_send_wr **bad_wr)
{
    int ret;
    ret = ibv_post_send(qp, wr, bad_wr);
    switch (ret) {
    case ENOMEM:
        return -FI_EAGAIN;
    case -1:
        /* Deal with non-compliant libibverbs drivers which set errno
         * instead of directly returning the error value */
        return (errno == ENOMEM) ? -FI_EAGAIN : -errno;
    default:
        return -ret;
    }
}
