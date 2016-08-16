/*
 * Copyright (c) 2016 Intel Corporation, Inc.  All rights reserved.
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
#include <config.h>
#include <stdlib.h>
#include <fi_enosys.h>
#include <fi_util.h>
#include <assert.h>
#include <rbtree.h>

/* deep copy: make seperate copy of mr_attr and use context for prov_mr */
static struct fi_mr_attr * create_mr_attr_copy(
                        const struct fi_mr_attr *in_attr, void * prov_mr)
{
    struct fi_mr_attr * item;
    struct iovec *mr_iov;
    int i = 0;

    if (!prov_mr || !in_attr)
        return NULL;

    item = malloc(sizeof(struct fi_mr_attr));
    if (!item)
        return NULL;

    *item = *in_attr;
    item->context = prov_mr;
    mr_iov = malloc(sizeof(struct iovec) * in_attr->iov_count);
    if (!mr_iov) {
        free(item);
        return NULL;
    }

    for(i = 0; i < in_attr->iov_count; i++)
        mr_iov[i] = in_attr->mr_iov[i];

    item->mr_iov = mr_iov;

    return item;
}

static uint64_t get_mr_key(struct ofi_util_mr *mr_h)
{
    assert(mr_h->b_key != UINT64_MAX);
    return mr_h->b_key++;
}

static int verify_addr(struct ofi_util_mr * in_mr, struct fi_mr_attr * item, uint64_t in_access,
                                 uint64_t in_addr, ssize_t in_len)
{
    int i = 0;
    uint64_t start = (uint64_t)item->mr_iov[i].iov_base;
    uint64_t end = start + item->mr_iov[i].iov_len;

    if (!in_addr) {
        FI_DBG(in_mr->prov, FI_LOG_MR, "verify_addr: input address to is zero\n");
        return -FI_EINVAL;
    }

    if ((in_access & item->access) != in_access) {
        FI_DBG(in_mr->prov, FI_LOG_MR, "verify_addr: requested access is not valid\n");
        return -FI_EACCES;
    }

      for (i = 0; i < item->iov_count; i++) {
        if (start <= in_addr && end >= (in_addr + in_len))
            return 0;
    }

    return -FI_EACCES;
}

int ofi_mr_insert(struct ofi_util_mr * in_mr_h, const struct fi_mr_attr *in_attr,
                                uint64_t * out_key, void * in_prov_mr)
{
    struct fi_mr_attr * item;

    if (!in_attr || in_attr->iov_count <= 0 || !in_prov_mr) {
        return -FI_EINVAL;
    }

    item = create_mr_attr_copy(in_attr, in_prov_mr);
    if (!item)
        return -FI_ENOMEM;

    /* Scalable MR handling: use requested key and offset */
    if (in_mr_h->mr_type == FI_MR_SCALABLE) {
        item->offset = (uintptr_t) in_attr->mr_iov[0].iov_base + in_attr->offset;
        /* verify key doesn't already exist */
        if (rbtFind(in_mr_h->map_handle, &item->requested_key)) {
                free((void *)item->mr_iov);
                free(item);
                return -FI_EINVAL;
        }
    } else {
        item->requested_key = get_mr_key(in_mr_h);
        item->offset = (uintptr_t) in_attr->mr_iov[0].iov_base;
    }

    rbtInsert(in_mr_h->map_handle, &item->requested_key, item);
    *out_key = item->requested_key;

    return 0;
}

void * ofi_mr_retrieve(struct ofi_util_mr * in_mr_h,  uint64_t in_key)
{
    void * itr;
    struct fi_mr_attr * item;
    void * key = &in_key;

    itr = rbtFind(in_mr_h->map_handle, key);

    if (!itr)
        return NULL;

    rbtKeyValue(in_mr_h->map_handle, itr, (void **)&key,
                                (void **) &item);
    return item->context;
}


/* io_addr is address of buff (&buf) */
int ofi_mr_retrieve_and_verify(struct ofi_util_mr * in_mr_h, ssize_t in_len,
                                uintptr_t *io_addr, uint64_t in_key,
                                uint64_t in_access, void **out_prov_mr)
{
    int ret = 0;
    void * itr;
    struct fi_mr_attr * item;
    void * key = &in_key;

    itr = rbtFind(in_mr_h->map_handle, key);

    if (!itr)
        return -FI_EINVAL;

    rbtKeyValue(in_mr_h->map_handle, itr, &key, (void **) &item);

    /*return providers MR struct */
    if (!item || !io_addr)
        return -FI_EINVAL;

    if (out_prov_mr)
        (*out_prov_mr) = item->context;

    /*offset for scalable */
    if (in_mr_h->mr_type == FI_MR_SCALABLE)
        *io_addr = (*io_addr) + item->offset;

    ret = verify_addr(in_mr_h, item, in_access, *io_addr, in_len);

    return ret;
}

int ofi_mr_erase(struct ofi_util_mr * in_mr_h, uint64_t in_key)
{
    void * itr;
    struct fi_mr_attr * item;

    if (!in_mr_h)
        return -FI_EINVAL;

    itr = rbtFind(in_mr_h->map_handle, &in_key);

    if (!itr)
        return -FI_ENOKEY;

    /*release memory */
    rbtKeyValue(in_mr_h->map_handle, itr, (void **)&in_key,
                                (void **) &item);

    assert(item);

    free((void *)item->mr_iov);
    free(item);

    rbtErase(in_mr_h->map_handle, itr);

    return 0;
}

/*assumes uint64_t keys */
static int compare_mr_keys(void *key1, void *key2)
{
    uint64_t k1 = *((uint64_t *) key1);
    uint64_t k2 = *((uint64_t *) key2);
    return (k1 < k2) ?  -1 : (k1 > k2);
}


int ofi_mr_init(const struct fi_provider *in_prov, enum fi_mr_mode mode,
                struct ofi_util_mr ** out_new_mr)
{
    struct ofi_util_mr * new_mr = malloc(sizeof(struct ofi_util_mr));
    if (!new_mr)
        return -FI_ENOMEM;

    assert((mode == FI_MR_SCALABLE) || (mode == FI_MR_BASIC));

    new_mr->mr_type = mode;

    new_mr->map_handle = rbtNew(compare_mr_keys);
    if (!new_mr->map_handle) {
        free(new_mr);
        return -FI_ENOMEM;
    }

    new_mr->b_key = 0;

    new_mr->prov = in_prov;

    (*out_new_mr) = new_mr;

    return 0;
}


void ofi_mr_close(struct ofi_util_mr *in_mr_h)
{
    if (!in_mr_h) {
        FI_WARN(&core_prov, FI_LOG_MR, "util mr_close: received NULL input\n");
        return;
    }

    rbtDelete(in_mr_h->map_handle);
    free(in_mr_h);
}
