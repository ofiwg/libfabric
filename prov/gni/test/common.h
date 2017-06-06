/*
 * Copyright (c) 2015-2017 Cray Inc. All rights reserved.
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

#ifndef PROV_GNI_TEST_COMMON_H_
#define PROV_GNI_TEST_COMMON_H_

#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <sys/time.h>
#include <criterion/criterion.h>
#include <criterion/logging.h>
#include "gnix_rdma_headers.h"
#include "gnix.h"
#include "fi_util.h"

#define BLUE "\x1b[34m"
#define COLOR_RESET "\x1b[0m"

#define CACHE_RO 0
#define CACHE_RW 1

#define GET_DOMAIN_RO_CACHE(domain) \
    ({ domain->mr_cache_info[domain->auth_key->ptag].mr_cache_ro; })
#define GET_DOMAIN_RW_CACHE(domain) \
    ({ domain->mr_cache_info[domain->auth_key->ptag].mr_cache_rw; })

/* defined in rdm_atomic.c */
extern int supported_compare_atomic_ops[FI_ATOMIC_OP_LAST][FI_DATATYPE_LAST];
extern int supported_fetch_atomic_ops[FI_ATOMIC_OP_LAST][FI_DATATYPE_LAST];

void calculate_time_difference(struct timeval *start, struct timeval *end,
		int *secs_out, int *usec_out);
int dump_cq_error(struct fid_cq *cq, void *context, uint64_t flags);

static inline struct gnix_fid_ep *get_gnix_ep(struct fid_ep *fid_ep)
{
	return container_of(fid_ep, struct gnix_fid_ep, ep_fid);
}

#define GNIX_DEFAULT_MR_MODE OFI_MR_BASIC_MAP
#endif /* PROV_GNI_TEST_COMMON_H_ */
