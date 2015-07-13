/*
 * Copyright (c) 2014, Cisco Systems, Inc. All rights reserved.
 *
 * LICENSE_BEGIN
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * LICENSE_END
 *
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <dirent.h>
#include <pthread.h>
#include <errno.h>
#include <sys/stat.h>

#include <infiniband/driver.h>

#include "usnic_direct.h"
#include "usd.h"
#include "usd_ib_sysfs.h"
#include "usd_ib_cmd.h"
#include "usd_socket.h"
#include "usd_device.h"

static pthread_once_t usd_init_once = PTHREAD_ONCE_INIT;

static struct usd_ib_dev *usd_ib_dev_list;
static int usd_init_error;

TAILQ_HEAD(,usd_device) usd_device_list =
    TAILQ_HEAD_INITIALIZER(usd_device_list);

/*
 * Perform one-time initialization
 */
static void
do_usd_init(void)
{
    usd_init_error = usd_ib_get_devlist(&usd_ib_dev_list);
}

/*
 * Init routine
 */
static int
usd_init(void)
{
    /* Do initialization one time */
    pthread_once(&usd_init_once, do_usd_init);
    return usd_init_error;
}

/*
 * Return list of currently available devices
 */
int
usd_get_device_list(
    struct usd_device_entry *entries,
    int *num_entries)
{
    int n;
    struct usd_ib_dev *idp;
    int ret;

    n = 0;

    ret = usd_init();
    if (ret != 0) {
        goto out;
    }

    idp = usd_ib_dev_list;
    while (idp != NULL && n < *num_entries) {
        strncpy(entries[n].ude_devname, idp->id_usnic_name,
                sizeof(entries[n].ude_devname) - 1);
        ++n;
        idp = idp->id_next;
    }

out:
    *num_entries = n;
    return ret;
}

/*
 * Allocate a context from the driver
 */
static int
usd_open_ibctx(struct usd_context *uctx)
{
    int ret;

    ret = usd_ib_cmd_get_context(uctx);
    return ret;
}

/*
 * Rummage around and collect all the info about this device we can find
 */
static int
usd_discover_device_attrs(
    struct usd_device *dev,
    const char *dev_name)
{
    struct usd_device_attrs *dap;
    int ret;

    /* find interface name */
    ret = usd_get_iface(dev);
    if (ret != 0)
        return ret;

    ret = usd_get_mac(dev, dev->ud_attrs.uda_mac_addr);
    if (ret != 0)
        return ret;

    ret = usd_get_usnic_config(dev);
    if (ret != 0)
        return ret;

    ret = usd_get_firmware(dev);
    if (ret != 0)
        return ret;

    /* ipaddr, netmask, mtu */
    ret = usd_get_dev_if_info(dev);
    if (ret != 0)
        return ret;

    /* get what attributes we can from querying IB */
    ret = usd_ib_query_dev(dev);
    if (ret != 0)
        return ret;

    /* constants that should come from driver */
    dap = &dev->ud_attrs;
    dap->uda_max_cqe = (1 << 16) - 1;;
    dap->uda_max_send_credits = (1 << 12) - 1;
    dap->uda_max_recv_credits = (1 << 12) - 1;
    strncpy(dap->uda_devname, dev_name, sizeof(dap->uda_devname) - 1);

    return 0;
}

static void
usd_dev_free(struct usd_device *dev)
{
    if (dev->ud_arp_sockfd != -1)
        close(dev->ud_arp_sockfd);

    if (dev->ud_ctx != NULL &&
            (dev->ud_flags & USD_DEVF_CLOSE_CTX)) {
        usd_close_context(dev->ud_ctx);
    }
    free(dev);
}

/*
 * Allocate a usd_device without allocating a PD
 */
static int
usd_dev_alloc_init(struct usd_context *context, const char *dev_name, int cmd_fd,
                    int check_ready, struct usd_device **dev_o)
{
    struct usd_device *dev = NULL;
    int ret;

    dev = calloc(sizeof(*dev), 1);
    if (dev == NULL) {
        ret = -errno;
        goto out;
    }

    dev->ud_flags = 0;
    if (context == NULL) {
        ret = usd_open_context(dev_name, cmd_fd, &dev->ud_ctx);
        if (ret != 0) {
            goto out;
        }
        dev->ud_flags |= USD_DEVF_CLOSE_CTX;
    } else {
        dev->ud_ctx = context;
    }

    dev->ud_arp_sockfd = -1;

    TAILQ_INIT(&dev->ud_pending_reqs);
    TAILQ_INIT(&dev->ud_completed_reqs);

    if (context == NULL)
        ret = usd_discover_device_attrs(dev, dev_name);
    else
        ret = usd_discover_device_attrs(dev, context->ucx_ib_dev->id_usnic_name);
    if (ret != 0)
        goto out;

    dev->ud_attrs.uda_event_fd = dev->ud_ctx->event_fd;
    dev->ud_attrs.uda_num_comp_vectors = dev->ud_ctx->num_comp_vectors;

    if (check_ready) {
        ret = usd_device_ready(dev);
        if (ret != 0) {
            goto out;
        }
    }

    *dev_o = dev;
    return 0;

out:
    if (dev != NULL)
        usd_dev_free(dev);
    return ret;
}

int
usd_close_context(struct usd_context *ctx)
{
    /* XXX - verify all other resources closed out */
    if (ctx->ucx_flags & USD_CTXF_CLOSE_CMD_FD)
        close(ctx->ucx_ib_dev_fd);
    if (ctx->ucmd_ib_dev_fd != -1)
        close(ctx->ucmd_ib_dev_fd);

    free(ctx);

    return 0;
}

int
usd_open_context(const char *dev_name, int cmd_fd,
                struct usd_context **ctx_o)
{
    struct usd_context *ctx = NULL;
    struct usd_ib_dev *idp;
    int ret;

    if (dev_name == NULL)
        return -EINVAL;

    ret = usd_init();
    if (ret != 0) {
        return ret;
    }

    /* Look for matching device */
    idp = usd_ib_dev_list;
    while (idp != NULL) {
        if (dev_name == NULL || strcmp(idp->id_usnic_name, dev_name) == 0) {
            break;
        }
        idp = idp->id_next;
    }

    /* not found, leave now */
    if (idp == NULL) {
        ret = -ENXIO;
        goto out;
    }

    /*
     * Found matching device, open an instance
     */
    ctx = calloc(sizeof(*ctx), 1);
    if (ctx == NULL) {
        ret = -errno;
        goto out;
    }
    ctx->ucx_ib_dev_fd = -1;
    ctx->ucmd_ib_dev_fd = -1;
    ctx->ucx_flags = 0;

    /* Save pointer to IB device */
    ctx->ucx_ib_dev = idp;

    /* Open the fd we will be using for IB commands */
    if (cmd_fd == -1) {
        ctx->ucx_ib_dev_fd = open(idp->id_dev_path, O_RDWR);
        if (ctx->ucx_ib_dev_fd == -1) {
            ret = -ENODEV;
            goto out;
        }
        ctx->ucx_flags |= USD_CTXF_CLOSE_CMD_FD;
    } else {
        ctx->ucx_ib_dev_fd = cmd_fd;
    }

    /*
     * Open another fd to send encapsulated user commands through
     * CMD_GET_CONTEXT call. The reason to open an additional fd is
     * that ib core does not allow multiple get_context call on one
     * file descriptor.
     */
    ctx->ucmd_ib_dev_fd = open(idp->id_dev_path, O_RDWR | O_CLOEXEC);
    if (ctx->ucmd_ib_dev_fd == -1) {
        ret = -ENODEV;
        goto out;
    }

    /* allocate a context from driver */
    ret = usd_open_ibctx(ctx);
    if (ret != 0) {
        goto out;
    }

    *ctx_o = ctx;
    return 0;

out:
    if (ctx != NULL)
        usd_close_context(ctx);
    return ret;
}

/*
 * Close a raw USNIC device
 */
int
usd_close(
    struct usd_device *dev)
{
    TAILQ_REMOVE(&usd_device_list, dev, ud_link);
    usd_dev_free(dev);

    return 0;
}

/*
 * Open a raw USNIC device
 */
int
usd_open(
    const char *dev_name,
    struct usd_device **dev_o)
{
    return usd_open_with_fd(dev_name, -1, 1, 1, dev_o);
}

/*
 * Open a raw USNIC device
 */
int
usd_open_for_attrs(
    const char *dev_name,
    struct usd_device **dev_o)
{
    return usd_open_with_fd(dev_name, -1, 0, 0, dev_o);
}

/*
 * previous generic usd open function, used by libusnic_verbs_d
 */
int
usd_open_with_fd(
    const char *dev_name,
    int cmd_fd,
    int check_ready,
    int alloc_pd,
    struct usd_device **dev_o)
{
    struct usd_device *dev = NULL;
    int ret;

    ret = usd_dev_alloc_init(NULL, dev_name, cmd_fd, check_ready, &dev);
    if (ret != 0) {
        goto out;
    }

    if (alloc_pd) {
        ret = usd_ib_cmd_alloc_pd(dev, &dev->ud_pd_handle);
        if (ret != 0) {
            goto out;
        }
    }

    TAILQ_INSERT_TAIL(&usd_device_list, dev, ud_link);
    *dev_o = dev;
    return 0;

out:
    if (dev != NULL)
        usd_dev_free(dev);
    return ret;

}

/*
 * Most generic usd device open function
 */
int
usd_open_with_ctx(struct usd_context *context, int alloc_pd, int check_ready,
                    struct usd_device **dev_o)
{
    struct usd_device *dev = NULL;
    int ret;

    ret = usd_dev_alloc_init(context, NULL, -1, check_ready, &dev);
    if (ret != 0) {
        goto out;
    }

    if (alloc_pd) {
        ret = usd_ib_cmd_alloc_pd(dev, &dev->ud_pd_handle);
        if (ret != 0) {
            goto out;
        }
    }

    TAILQ_INSERT_TAIL(&usd_device_list, dev, ud_link);
    *dev_o = dev;
    return 0;

out:
    if (dev != NULL)
        usd_dev_free(dev);
    return ret;
}

/*
 * Return attributes of a device
 */
int
usd_get_device_attrs(
    struct usd_device *dev,
    struct usd_device_attrs *dattrs)
{
    int ret;

    /* ipaddr, netmask, mtu */
    ret = usd_get_dev_if_info(dev);
    if (ret != 0)
        return ret;

    /* get what attributes we can from querying IB */
    ret = usd_ib_query_dev(dev);
    if (ret != 0)
        return ret;

    *dattrs = dev->ud_attrs;
    return 0;
}

/*
 * Check that device is ready to have queues created
 */
int
usd_device_ready(
    struct usd_device *dev)
{
    if (dev->ud_attrs.uda_ipaddr_be == 0) {
        return -EADDRNOTAVAIL;
    }
    if (dev->ud_attrs.uda_link_state != USD_LINK_UP) {
        return -ENETDOWN;
    }

    return 0;
}
