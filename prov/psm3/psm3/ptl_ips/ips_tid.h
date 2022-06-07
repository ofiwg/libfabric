/*

  This file is provided under a dual BSD/GPLv2 license.  When using or
  redistributing this file, you may do so under either license.

  GPL LICENSE SUMMARY

  Copyright(c) 2015 Intel Corporation.

  This program is free software; you can redistribute it and/or modify
  it under the terms of version 2 of the GNU General Public License as
  published by the Free Software Foundation.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  Contact Information:
  Intel Corporation, www.intel.com

  BSD LICENSE

  Copyright(c) 2015 Intel Corporation.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

/* Copyright (c) 2003-2014 Intel Corporation. All rights reserved. */

/* included header files  */

#ifndef _IPS_TID_H
#define _IPS_TID_H

#ifdef PSM_OPA
#include "psm_user.h"
#include "ips_tidcache.h"

struct ips_tid;

typedef void (*ips_tid_avail_cb_fn_t) (struct ips_tid *, void *context);

struct ips_tid_ctrl {
	pthread_spinlock_t tid_ctrl_lock;
	uint32_t tid_num_max;
	uint32_t tid_num_avail;
} __attribute__ ((aligned(64)));

struct ips_tid {
	const psmi_context_t *context;
	struct ips_protoexp *protoexp;

	void *tid_avail_context;
	struct ips_tid_ctrl *tid_ctrl;

	ips_tid_avail_cb_fn_t tid_avail_cb;
	uint64_t tid_num_total;
	uint32_t tid_num_inuse;
	uint32_t tid_cachesize;	/* items can be cached */
	cl_qmap_t tid_cachemap; /* RB tree implementation */
	/*
	 * tids storage.
	 * This is used in tid registration caching case for
	 * tid invalidation, acquire, replace and release,
	 * entries should be the assigned tid number.
	 */
	uint32_t *tid_array;
};

psm2_error_t ips_tid_init(const psmi_context_t *context,
		struct ips_protoexp *protoexp,
		ips_tid_avail_cb_fn_t cb, void *cb_context);
psm2_error_t ips_tid_fini(struct ips_tid *tidc);

/* Acquiring tids.
 * Buffer base has to be aligned on page boundary
 * Buffer length has to be multiple pages
 */
psm2_error_t ips_tidcache_acquire(struct ips_tid *tidc,
		const void *buf,  /* input buffer, aligned to page boundary */
		uint32_t *length, /* buffer length, aligned to page size */
		uint32_t *tid_array, /* output tidarray, */
		uint32_t *tidcnt,    /* output of tid count */
		uint32_t *pageoff  /* output of offset in first tid */
#ifdef PSM_CUDA
		, uint8_t is_cuda_ptr
#endif
		);

psm2_error_t ips_tidcache_release(struct ips_tid *tidc,
		uint32_t *tid_array, /* input tidarray, */
		uint32_t tidcnt);    /* input of tid count */

psm2_error_t ips_tidcache_cleanup(struct ips_tid *tidc);
psm2_error_t ips_tidcache_invalidation(struct ips_tid *tidc);

psm2_error_t ips_tid_acquire(struct ips_tid *tidc,
		const void *buf,  /* input buffer, aligned to page boundary */
		uint32_t *length, /* buffer length, aligned to page size */
		uint32_t *tid_array, /* output tidarray, */
		uint32_t *tidcnt
#ifdef PSM_CUDA
		, uint8_t is_cuda_ptr
#endif
		);   /* output of tid count */

psm2_error_t ips_tid_release(struct ips_tid *tidc,
		uint32_t *tid_array, /* input tidarray, */
		uint32_t tidcnt);    /* input of tid count */

PSMI_INLINE(int ips_tid_num_available(struct ips_tid *tidc))
{
	if (tidc->tid_ctrl->tid_num_avail == 0) {
		if (tidc->tid_ctrl->tid_num_max == tidc->tid_num_inuse)
			return -1;
		else
			return 0;
	}

	return tidc->tid_ctrl->tid_num_avail;
}

/* Note that the caller is responsible for making sure that NIDLE is non-zero
   before calling ips_tidcache_evict.  If NIDLE is 0 at the time of call,
   ips_tidcache_evict is unstable.
 */
uint64_t ips_tidcache_evict(struct ips_tid *tidc, uint64_t length);

#endif // PSM_OPA
#endif /* _IPS_TID_H */
