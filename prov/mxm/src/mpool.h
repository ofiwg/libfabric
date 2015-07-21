/*
 * Copyright (c) 2015 Intel Corporation. All rights reserved.
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
#ifndef __MPOOL_H__
#define __MPOOL_H__
#include <stdint.h>
#include <stdlib.h>

struct mpool {
        int nkeys;
        int num;
        char pool[1];
};

#define MPOOL_GET_KEY(_pool, _key) ((uint32_t*)_pool->pool)[_key]
#define MPOOL_GET_OBJ_STORAGE_PTR(_pool) ((void*)(_pool->pool + _pool->nkeys*sizeof(uint32_t)))
#define MPOOL_GET_OBJ(_pool, _pos, _type_sz)                            \
        ((char*)MPOOL_GET_OBJ_STORAGE_PTR((_pool)) + _pos*_type_sz)

static inline int
mpool_init(struct mpool **mpool, size_t el_size, int num) {
        int nkeys;
        size_t pool_size;
        int i;
        num += (num % 32) ? (32 - (num % 32)) : 0;
        nkeys = num / 32;
        pool_size = sizeof(struct mpool) - 1
                + num*el_size + nkeys*sizeof(uint32_t);
        *mpool = (struct mpool*)malloc(pool_size);

        (*mpool)->nkeys = nkeys;
        (*mpool)->num   = num;
        for (i=0; i<nkeys; i++) {
                MPOOL_GET_KEY((*mpool), i) = -1;
        }
        return 0;
}

#define MPOOL_ALLOC(_mpool, _type, _ptr) \
        do{                                                             \
                int __i;                                                \
                int __pos = -1;                                         \
                for (__i=0; __i<(_mpool)->nkeys; __i++) {               \
                        int __bit = __builtin_ffs(MPOOL_GET_KEY(_mpool, \
                                                                __i));  \
                        if (__bit) {                                    \
                                __pos = 32*__i+__bit - 1;               \
                                MPOOL_GET_KEY(_mpool, __i) &=           \
                                        ~(1 << (__bit-1));              \
                                break;                                  \
                        }                                               \
                }                                                       \
                if (__pos != -1) {                                      \
                        (_ptr) = (_type *)                              \
                                MPOOL_GET_OBJ((_mpool),                 \
                                              __pos, sizeof(_type));    \
                }else{                                                  \
                        (_ptr) = (_type *)malloc(sizeof(_type));        \
                }                                                       \
        }while(0);

#define MPOOL_RETURN(_mpool, _type, _ptr)                               \
        do {                                                            \
                char *__start =(char*)MPOOL_GET_OBJ_STORAGE_PTR(_mpool); \
                if ((char*)_ptr >= __start && (char*)                   \
                    _ptr < __start + _mpool->num*sizeof(_type)) {       \
                        int __pos = ((char*)_ptr - (char*)              \
                                     MPOOL_GET_OBJ_STORAGE_PTR(_mpool))/sizeof(_type); \
                        int __key = __pos / 32;                         \
                        int __bit = __pos % 32;                         \
                        MPOOL_GET_KEY(_mpool, __key) |= (1 << __bit);   \
                }else{                                                  \
                        free(_ptr);                                     \
                }                                                       \
        }while(0)
#endif
