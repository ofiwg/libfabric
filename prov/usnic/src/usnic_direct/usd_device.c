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
#include <sys/mman.h>

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

#define USD_IOVA_VMA_SIZE 0x10000000000ul

/*
 * vm area for IOVA address space, gotten via mmap() when a shared
 * PD is allocated. A process needs only mmap once, as the same address
 * space can be reused for a different shared PD.
 */
static struct usd_iova_vma_t {
    void                                *iova_start;
    LIST_HEAD(udev_head, usd_device)    dev_list;
    pthread_mutex_t                     mutex;
} iova_vma = { 0, LIST_HEAD_INITIALIZER(dev_list), PTHREAD_MUTEX_INITIALIZER };

/*
 * Allocate address space for io virtual addresses to be
 * programmed to iommu and HW
 */
int
usd_map_iova_space(struct usd_device *dev, void **iova_start_o)
{
    void *iova_start;
    off64_t offset;
    struct usd_device *dev_iter;
    int err;

    err = pthread_mutex_lock(&iova_vma.mutex);
    if (err != 0) {
        return -err;
    }

    /* Duplicate alloc_shpd are called on the same usd_dev, shouldn't happen */
    LIST_FOREACH(dev_iter, &iova_vma.dev_list, iova_link) {
        if (dev_iter == dev) {
            pthread_mutex_unlock(&iova_vma.mutex);
            return -EEXIST;
        }
    }

    /* If device list is empty, then alloc_shpd is not called yet */
    if (LIST_EMPTY(&iova_vma.dev_list)) {
        /*
         * The assoicated VF is not returned back until first create_qp returns
         * so cannot pass a VF ID down.
         */
        offset = USNIC_ENCODE_PGOFF(0, USNIC_MMAP_IOVA, 0);
        iova_start = mmap64(NULL, USD_IOVA_VMA_SIZE, PROT_READ + PROT_WRITE,
                            MAP_SHARED, dev->ud_ib_dev_fd, offset);
        if (iova_start == MAP_FAILED) {
            usd_err("Failed to map shared PD translation address space\n");
            pthread_mutex_unlock(&iova_vma.mutex);
            return -errno;
        }
        iova_vma.iova_start = iova_start;
    }

    /* Add the dev into devlist if alloc_shpd is called on it */
    LIST_INSERT_HEAD(&iova_vma.dev_list, dev, iova_link);
    *iova_start_o = iova_vma.iova_start;
    pthread_mutex_unlock(&iova_vma.mutex);

    return 0;
}

static int
usd_unmap_iova_space(struct usd_device *dev)
{
    struct usd_device *dev_iter;
    int found;
    int err;

    err = pthread_mutex_lock(&iova_vma.mutex);
    if (err != 0) {
        return -err;
    }

    found = 0;
    LIST_FOREACH(dev_iter, &iova_vma.dev_list, iova_link) {
        if (dev_iter == dev) {
            found = 1;
            break;
        }
    }

    if (found == 0) {
        pthread_mutex_unlock(&iova_vma.mutex);
        return 0;
    }

    LIST_REMOVE(dev, iova_link);
    if (LIST_EMPTY(&iova_vma.dev_list))
        munmap(iova_vma.iova_start, USD_IOVA_VMA_SIZE);

    pthread_mutex_unlock(&iova_vma.mutex);

    return 0;
}

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
usd_get_context(
    struct usd_device *dev)
{
    int ret;

    ret = usd_ib_cmd_get_context(dev);
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

/*
 * Close a raw USNIC device
 */
int
usd_close(
    struct usd_device *dev)
{
    usd_unmap_iova_space(dev);

    TAILQ_REMOVE(&usd_device_list, dev, ud_link);

    /* XXX - verify all other resources closed out */
    if (dev->ud_flags & USD_DEVF_CLOSE_CMD_FD)
        close(dev->ud_ib_dev_fd);
    if (dev->ud_arp_sockfd != -1)
        close(dev->ud_arp_sockfd);

    free(dev);

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
    return usd_open_with_fd(dev_name, -1, 1, dev_o);
}

/*
 * Open a raw USNIC device
 */
int
usd_open_for_attrs(
    const char *dev_name,
    struct usd_device **dev_o)
{
    return usd_open_with_fd(dev_name, -1, 0, dev_o);
}

/*
 * Open a raw USNIC device
 */
int
usd_open_with_fd(
    const char *dev_name,
    int cmd_fd,
    int check_ready,
    struct usd_device **dev_o)
{
    struct usd_ib_dev *idp;
    struct usd_device *dev = NULL;
    int ret;

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
    dev = calloc(sizeof(*dev), 1);
    if (dev == NULL) {
        ret = -errno;
        goto out;
    }
    dev->ud_ib_dev_fd = -1;
    dev->ud_arp_sockfd = -1;
    dev->ud_flags = 0;
    TAILQ_INIT(&dev->ud_pending_reqs);
    TAILQ_INIT(&dev->ud_completed_reqs);

    /* Save pointer to IB device */
    dev->ud_ib_dev = idp;

    /* Open the fd we will be using for IB commands */
    if (cmd_fd == -1) {
        dev->ud_ib_dev_fd = open(idp->id_dev_path, O_RDWR);
        if (dev->ud_ib_dev_fd == -1) {
            ret = -ENODEV;
            goto out;
        }
        dev->ud_flags |= USD_DEVF_CLOSE_CMD_FD;
    } else {
        dev->ud_ib_dev_fd = cmd_fd;
    }

    /* allocate a context from driver */
    ret = usd_get_context(dev);
    if (ret != 0) {
        goto out;
    }
    ret = usd_ib_cmd_alloc_pd(dev, &dev->ud_pd_handle);
    if (ret != 0) {
        goto out;
    }

    ret = usd_discover_device_attrs(dev, dev_name);
    if (ret != 0)
        goto out;

    if (check_ready) {
        ret = usd_device_ready(dev);
        if (ret != 0) {
            goto out;
        }
    }

    TAILQ_INSERT_TAIL(&usd_device_list, dev, ud_link);
    *dev_o = dev;

    return 0;

  out:
    if (dev != NULL) {
        if (dev->ud_flags & USD_DEVF_CLOSE_CMD_FD
            && dev->ud_ib_dev_fd != -1)
            close(dev->ud_ib_dev_fd);
        free(dev);
    }
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

#if USNIC_HAVE_SHPD
/* Create a share protection domain from default pd */
int usd_alloc_shpd(struct usd_device *dev, uint64_t share_key,
                    uint32_t *shpd_handle)
{
    void *iova_start = NULL;
    int err;

    /*
     * Don't allow creating additional shpd if the default PD
     * is assoicated with a shpd
     */
    if (dev->is_pd_shared != 0)
        return EEXIST;

    err = usd_map_iova_space(dev, &iova_start);
    if (err != 0)
        return -err;

    err = usd_ib_cmd_alloc_shpd(dev, dev->ud_pd_handle, share_key,iova_start,
                                USD_IOVA_VMA_SIZE, shpd_handle);
    if (err != 0) {
        usd_unmap_iova_space(dev);
        return -err;
    }
    dev->is_pd_shared = 1;

    return 0;
}

int usd_open_with_shpd(const char *dev_name, int cmd_fd,
                        uint32_t shpd_handle, uint64_t share_key,
                        struct usd_device **dev_o)
{
    /* copy most of code in usd_open_with_fd for now */
    struct usd_ib_dev *idp;
    struct usd_device *dev = NULL;
    int ret;

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
    dev = calloc(sizeof(*dev), 1);
    if (dev == NULL) {
        ret = -errno;
        goto out;
    }
    dev->ud_ib_dev_fd = -1;
    dev->ud_arp_sockfd = -1;
    dev->ud_flags = 0;
    TAILQ_INIT(&dev->ud_pending_reqs);
    TAILQ_INIT(&dev->ud_completed_reqs);

    /* Save pointer to IB device */
    dev->ud_ib_dev = idp;

    /* Open the fd we will be using for IB commands */
    if (cmd_fd == -1) {
        dev->ud_ib_dev_fd = open(idp->id_dev_path, O_RDWR);
        if (dev->ud_ib_dev_fd == -1) {
            ret = -ENODEV;
            goto out;
        }
        dev->ud_flags |= USD_DEVF_CLOSE_CMD_FD;
    } else {
        dev->ud_ib_dev_fd = cmd_fd;
    }

    /* allocate a context from driver */
    ret = usd_get_context(dev);
    if (ret != 0) {
        goto out;
    }
    /* Assign the default PD to the pd returned from kernel share_pd call */
    ret = usd_ib_cmd_share_pd(dev, shpd_handle, share_key, &dev->ud_pd_handle);
    if (ret != 0) {
        goto out;
    }
    dev->is_pd_shared = 1;

    ret = usd_discover_device_attrs(dev, dev_name);
    if (ret != 0)
        goto out;

    ret = usd_device_ready(dev);
    if (ret != 0) {
        goto out;
    }

    TAILQ_INSERT_TAIL(&usd_device_list, dev, ud_link);
    *dev_o = dev;

    return 0;

  out:
    if (dev != NULL) {
        if (dev->ud_flags & USD_DEVF_CLOSE_CMD_FD
            && dev->ud_ib_dev_fd != -1)
            close(dev->ud_ib_dev_fd);
        free(dev);
    }
    return ret;
}

#else
int usd_alloc_shpd(struct usd_device *dev __attribute__((__unused__)),
                    uint64_t share_key __attribute__((__unused__)),
                    uint32_t *shpd_handle __attribute__((__unused__))) {
    return -1;
}

int usd_open_with_shpd(const char *dev_name __attribute__((__unused__)),
                        int cmd_fd __attribute__((__unused__)),
                        uint32_t shpd_handle __attribute__((__unused__)),
                        uint64_t share_key __attribute__((__unused__)),
                        struct usd_device **dev_o __attribute__((__unused__)))
{
    return -1;
}
#endif
