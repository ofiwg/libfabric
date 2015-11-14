/*
 * Copyright (c) 2015 Cray Inc.  All rights reserved.
 * Copyright (c) 2015 Los Alamos National Security, LLC. All rights reserved.
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

#ifndef _GNIX_FREELIST_H_
#define _GNIX_FREELIST_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <fi.h>
#include <fi_list.h>

/* Number of elements to seed the freelist with */
#define GNIX_SFL_INIT_SIZE 100
/* Initial refill size */
#define GNIX_SFL_INIT_REFILL_SIZE 10
/* Refill growth factor */
#define GNIX_SFL_GROWTH_FACTOR 2

/** Free list based on singly linked slist
 *
 * @var freelist           The free list itself
 * @var chunks             Memory chunks (must be saved for freeing)
 * @var refill_size        Number of elements for the next refill
 * @var growth_factor      Factor for increasing refill size
 * @var max_refill_size;   Max refill size
 * @var elem_size          Size of element (in bytes)
 * @var offset             Offset of slist_entry field (in bytes)
 */
struct gnix_s_freelist {
	struct slist freelist;
	struct slist chunks;
	int refill_size;
	int growth_factor;
	int max_refill_size;
	int elem_size;
	int offset;
	int ts;
	fastlock_t lock;
};

/** Initializes a gnix_s_freelist
 *
 * @param elem_size         Size of element
 * @param offset            Offset of slist_entry field
 * @param init_size         Initial freelist size
 * @param refill_size       Number of elements for next refill
 * @param growth_factor     Factor for increasing refill size
 * @param max_refill_size   Max refill size
 * @param fl                gnix_s_freelist
 * @return                  FI_SUCCESS on success, -FI_ENOMEM on failure
 */
int _gnix_sfl_init(int elem_size, int offset, int init_size,
		   int refill_size, int growth_factor,
		   int max_refill_size, struct gnix_s_freelist *fl);

/** Initializes a thread safe gnix_s_freelist
 *
 * @param elem_size         Size of element
 * @param offset            Offset of slist_entry field
 * @param init_size         Initial freelist size
 * @param refill_size       Number of elements for next refill
 * @param growth_factor     Factor for increasing refill size
 * @param max_refill_size   Max refill size
 * @param fl                gnix_s_freelist
 * @return                  FI_SUCCESS on success, -FI_ENOMEM on failure
 */
int _gnix_sfl_init_ts(int elem_size, int offset, int init_size,
		      int refill_size, int growth_factor,
		      int max_refill_size, struct gnix_s_freelist *fl);

/** Clean up a gnix_s_freelist, including deleting memory chunks
 *
 * @param fl    Freelist
 */
void _gnix_sfl_destroy(struct gnix_s_freelist *fl);

/** Return an item from the freelist
 *
 * @param e     item
 * @param fl    gnix_s_freelist
 * @return      FI_SUCCESS on success, -FI_ENOMEM or -FI_EAGAIN on failure
 */
int _gnix_sfe_alloc(struct slist_entry **e, struct gnix_s_freelist *fl);

/** Return an item to the free list
 *
 * @param e     item
 * @param fl    gnix_s_freelist
 */
void _gnix_sfe_free(struct slist_entry *e, struct gnix_s_freelist *fl);

/** Is freelist empty (primarily used for testing
 *
 * @param fl    gnix_s_freelist
 * @return      True if list is currently empty, false otherwise
 */
static inline int _gnix_sfl_empty(struct gnix_s_freelist *fl)
{
	return slist_empty(&fl->freelist);
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* _GNIX_FREELIST_H_ */
