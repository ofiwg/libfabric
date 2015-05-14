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
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>

#include "usnic_direct.h"
#include "usd.h"
#include "usd_ib_cmd.h"

/*
 * Issue driver command to register memory region
 */
int
usd_reg_mr(
    struct usd_device *dev,
    void *vaddr,
    size_t length,
    struct usd_mr **mr_o)
{
    struct usd_mr *mr;
    int ret;

    mr = calloc(sizeof(*mr), 1);
    if (mr == NULL) {
        return -errno;
    }

    ret = usd_ib_cmd_reg_mr(dev, vaddr, length, mr);

    if (ret == 0) {
        mr->umr_dev = dev;
        mr->umr_vaddr = vaddr;
        mr->umr_length = length;
        *mr_o = mr;
    } else {
        free(mr);
    }

    return ret;
}

/*
 * Issue driver command to de-register memory region
 */
int
usd_dereg_mr(
    struct usd_mr *mr)
{
    int ret;

    ret = usd_ib_cmd_dereg_mr(mr->umr_dev, mr);
    if (ret == 0)
        free(mr);

    return ret;
}

/*
 * Used to allocate memory and an mr to go with it all in one go.  Used
 * to provide memory to the vnic_* functions that call pci_alloc_consistant
 * We want to return a nicely aligned chunk of memory preceded by struct usd_mr.
 * We don't know the alignment of the memory we get back, so allocate a big
 * enough chunk to hold the following:
 *   struct usd_mr
 *   N pad bytes
 *   true length and pointer to usd_mr
 *   page aligned buffer for user
 */
static int
_usd_alloc_mr(
    size_t size,
    void **vaddr_o,
    void **base_addr_o,
    size_t *true_size_o,
    size_t *madv_size_o)
{
    void *vaddr;
    void *base_addr;
    size_t true_size;
    size_t metadata_size;
    size_t madv_size;
    int ret;

    metadata_size = sizeof(struct usd_mr) + 3 * sizeof(uintptr_t);
    madv_size = ALIGN(size, sysconf(_SC_PAGESIZE));
    true_size = madv_size + metadata_size + sysconf(_SC_PAGESIZE) - 1;
    base_addr = mmap(NULL, true_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base_addr == NULL || base_addr == MAP_FAILED) {
        usd_err("Failed to mmap region of size %lu\n", true_size);
        return -errno;
    }
    vaddr =
        (void *) ALIGN((uintptr_t) base_addr + metadata_size,
                       sysconf(_SC_PAGESIZE));
    ((uintptr_t *) vaddr)[-1] = (uintptr_t) base_addr;
    ((uintptr_t *) vaddr)[-2] = true_size;
    ((uintptr_t *) vaddr)[-3] = madv_size;

    /*
     * Disable copy-on-write for memories internally used by USD.
     * For application buffers, disabling copy-on-write should be provided by
     * usd wrapper such as libfabric or verbs plugin if fork is supported.
     * The memory to be registered starts from page-aligned address, and ends
     * at page boundary, so it's impossible for a page to be updated
     * with multiple madvise calls when each call reference different VAs on
     * the same page. This allows to avoid the need to reference count
     * the pages that get updated with mutiple madvise calls. For details,
     * see libibverbs ibv_dont_forkrange implementations.
     */
    ret = madvise(vaddr, madv_size, MADV_DONTFORK);
    if (ret != 0) {
        usd_err("Failed to disable child's access to memory %p size %lu\n",
                vaddr, size);
        munmap(base_addr, true_size);
        return errno;
    }

    *vaddr_o = vaddr;
    *base_addr_o = base_addr;
    *true_size_o = true_size;
    *madv_size_o = madv_size;

    return 0;
}

static void
_usd_free_mr(void *vaddr, void *base_addr, size_t true_size,
                size_t madv_size)
{
    madvise(vaddr, madv_size, MADV_DOFORK);
    munmap(base_addr, true_size);
}

int
usd_alloc_mr(
    struct usd_device *dev,
    size_t size,
    void **vaddr_o)
{
    void *vaddr = NULL;
    void *base_addr = NULL;
    struct usd_mr *mr;
    size_t true_size = 0;
    size_t madv_size = 0;
    int ret;

    ret = _usd_alloc_mr(size, &vaddr, &base_addr, &true_size, &madv_size);
    if (ret != 0)
        return ret;

    mr = base_addr;
    ret = usd_ib_cmd_reg_mr(dev, vaddr, size, mr);
    if (ret != 0) {
        usd_err("Failed to register memory region %p, size %lu\n",
                vaddr, size);
        _usd_free_mr(vaddr, base_addr, true_size, madv_size);
        return ret;
    }
    mr->umr_dev = dev;

    *vaddr_o = vaddr;
    return 0;

    return ret;
}

/*
 * Allocate memory and register mr that uses addresses from pre-allocated
 * iova address space for programming iommu and HW.
 * it returns the requested buffer virutal adress and the iova to be programmed
 * to HW.
 */
int usd_alloc_iova_mr(
    struct usd_device *dev,
    size_t size,
    uint32_t vfid,
    uint32_t mr_type,
    uint32_t queue_index,
    void **vaddr_o,
    void **iova_o)
{
    void *vaddr = NULL;
    void *base_addr = NULL;
    struct usd_mr *mr;
    size_t true_size = 0;
    size_t madv_size = 0;
    int ret;

    ret = _usd_alloc_mr(size, &vaddr, &base_addr, &true_size, &madv_size);
    if (ret != 0)
        return ret;

    mr = base_addr;
    ret = usd_ib_cmd_reg_mr_v1(dev, vaddr, size, USNIC_REGMR_HWADDR_IOVA,
                                vfid, mr_type, queue_index, mr);
    if (ret == 0) {
        mr->umr_dev = dev;
    } else {
        usd_err("Failed to register iova memory region %p size %lu "
                "vfid %u mr_type %u queue_index %u\n",
                vaddr, size, vfid, mr_type, queue_index);
        _usd_free_mr(vaddr, base_addr, true_size, madv_size);
        return ret;
    }

    *vaddr_o = vaddr;
    *iova_o = (void*)mr->umr_iova;

    return 0;
}

/*
 * See usd_alloc_mr() for explanation of:
 *  mr = (struct usd_mr *)((uintptr_t *)vaddr)[-1];
 */
int
usd_free_mr(
    void *vaddr)
{
    struct usd_mr *mr;
    size_t true_size;
    size_t madv_size;
    int ret;

    mr = (struct usd_mr *) ((uintptr_t *) vaddr)[-1];
    true_size = ((uintptr_t *) vaddr)[-2];
    madv_size = ((uintptr_t *) vaddr)[-3];

    ret = usd_ib_cmd_dereg_mr(mr->umr_dev, mr);
    if (ret == 0) {
        _usd_free_mr(vaddr, mr, true_size, madv_size);
    }

    return ret;
}

/*
 * Utility function for vnic_* routines
 */
char *
pci_name(
    struct pci_dev *pdev)
{
    struct usd_device *dev;

    dev = (struct usd_device *) pdev;

    return dev->ud_ib_dev->id_usnic_name;
}
