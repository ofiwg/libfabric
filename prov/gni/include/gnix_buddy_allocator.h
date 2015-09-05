/*
 * Copyright (c) 2015 Los Alamos National Security, LLC. All rights reserved.
 * Copyright (c) 2015 Cray Inc.  All rights reserved.
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

#ifndef _GNIX_BUDDY_ALLOCATOR_H_
#define _GNIX_BUDDY_ALLOCATOR_H_

#include "fi_list.h"
#include "gnix_bitmap.h"
#include "gnix_util.h"
#include <math.h>
#include <stdlib.h>

/* evaluates to zero if X is not a power of two, otherwise evaluates to X - 1 */
#define IS_NOT_POW_TWO(X) (((X) & (~(X) + 1)) ^ (X))

/* Find the block size (in bytes) required for allocating LEN bytes */
#define BLOCK_SIZE(LEN, MIN) ((LEN) <= (MIN) ? (MIN) :\
			      (IS_NOT_POW_TWO(LEN)) ? (((LEN) << 1) & ~(LEN)) :\
			      (LEN))

/* Find the bitmap index for block X */
#define BITMAP_INDEX(X, BASE, MIN, LEN) (size_t) ((size_t) ((X) - (BASE)) /\
					 (MIN) + 2 * log2((LEN) / (MIN)))

/*
 * The following macro doesn't work when the base address starts at zero.
 * #define BUDDY(X, SIZE_X, BASE) (void *) ((size_t) (X) ^ (SIZE_X))
 */

/* Find the address of X's buddy block:
 * If the "index" of block X is even then the buddy must be to the right of X,
 * otherwise the buddy is to the left of X.
 */
#define BUDDY(X, LEN, BASE) (void *) ((((size_t) (BASE) - (size_t) (X)) /\
				       (LEN)) % 2 ? (size_t) (X) - (LEN) :\
				      (size_t) (X) + (LEN))

/* Calculate the offset of a free block, OFFSET = MIN * 2^MULT. */
#define OFFSET(MIN, MULT) ((MIN) * (1 << (MULT)))

/* Find the index into the free list with block size LEN. */
#define LIST_INDEX(LEN, MIN) (size_t) (log2((LEN) / (double) (MIN)))

/**
 * Structure representing a buddy allocator.
 *
 * @var base		The base address of the buffer being managed.
 * @var len		The length of the buffer the buddy allocator is managing.
 * @var min		The smallest chunk of memory that can be allocated.
 * @var max		The largest chunk of memory that can be allocated.
 *
 * @var nlists		The number of free lists.
 * @var lists		The array of free lists ordered from smallest block size.
 * at index 0 to largest block size at index nlists - 1.
 *
 * @var bitmap		Each bit is 1 if the block is allocated or split,
 * otherwise the bit is 0.
 */
typedef struct gnix_buddy_alloc_handle {
	void *base;
	size_t len;
	size_t min;
	size_t max;

	size_t nlists;
	struct dlist_entry *lists;

	gnix_bitmap_t bitmap;
} *handle_t;

/**
 * Creates a buddy allocator
 *
 * @param[in] base		Base address of buffer to be managed by
 * allocator.
 *
 * @param[in] len		Size of the buffer to be managed by allocator
 * (must be a multiple of max).
 *
 * @param[in] max		Maximum amount of memory that can be allocated
 * by a single call to _gnix_buddy_alloc (power 2).
 *
 * @param[in/out] alloc_handle	Handle to be used for when allocating/freeing
 * memory managed by the buddy allocator.
 *
 * @return FI_SUCCESS		Upon successfully creating an allocator.
 *
 * @return -FI_EINVAL		Upon an invalid parameter.
 *
 * @return -FI_ENOMEM		Upon failure to allocate memory to create the
 * buddy allocator.
 */
int _gnix_buddy_allocator_create(void *base, size_t len, size_t max,
				 handle_t *alloc_handle);

/**
 * Releases all resources associated with a buddy allocator handle.
 *
 * @param[in] alloc_handle	Buddy alloc handle to destroy.
 *
 * @return FI_SUCCESS	 	Upon successfully destroying an allocator.
 *
 * @return -FI_EINVAL 		Upon an invalid parameter.
 */
int _gnix_buddy_allocator_destroy(handle_t alloc_handle);

/**
 * Allocate a buffer from the buddy allocator
 *
 * @param[in] alloc_handle 	Previously allocated GNI buddy_alloc_handle to
 * use as allocator.
 *
 * @param[in/out] ptr		Pointer to an address where the address of the
 * allocated buffer will be returned.
 *
 * @param[in] len		Size of buffer to allocate in bytes.
 *
 * @return FI_SUCCESS		Upon successfully allocating a buffer.
 *
 * @return -FI_ENOMEM 		Upon not being able to allocate a buffer of the
 * requested size.
 *
 * @return -FI_EINVAL 		Upon an invalid parameters.
 */
int _gnix_buddy_alloc(handle_t alloc_handle, void **ptr, size_t len);

/**
 * Free a previously allocated buffer
 *
 * @param[in] alloc_handle 	Previously allocated GNI buddy_alloc_handle to
 * use as allocator.
 *
 * @param[in/out] ptr		Pointer to an address where the address of the
 * allocated buffer will be returned.
 *
 * @param[in] len		Size of buffer to allocate in bytes.
 *
 * @return FI_SUCCESS		Upon successfully allocating a buffer.
 *
 * @return -FI_EINVAL 		Upon an invalid parameters.
 */
int _gnix_buddy_free(handle_t alloc_handle, void *ptr, size_t len);
#endif /* _GNIX_BUDDY_ALLOCATOR_H_ */
