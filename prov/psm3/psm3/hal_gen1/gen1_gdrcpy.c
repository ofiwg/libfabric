#ifdef PSM_OPA
/*

  This file is provided under a dual BSD/GPLv2 license.  When using or
  redistributing this file, you may do so under either license.

  GPL LICENSE SUMMARY

  Copyright(c) 2018 Intel Corporation.

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

  Copyright(c) 2018 Intel Corporation.

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
#ifdef PSM_CUDA
#include "psm_user.h"
#include "psm2_hal.h"
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include "ips_proto.h"
#include "ptl_ips/ips_tid.h"
#include "ptl_ips/ips_expected_proto.h"
#include "gen1_user.h"
#include "ptl_ips.h"
#include "gen1_hal.h"

static int gdr_fd;

// Note: ep->epaddr->proto is always NULL.  ep->epaddr only has epid
// so we must navigate the ep->ptl_ips to get ips_proto
static inline
struct ips_proto *psm3_gen1_get_proto(psm2_ep_t ep)
{
	struct ips_proto *proto = &((struct ptl_ips*)(ep->ptl_ips.ptl))->proto;
	psmi_assert(ep == proto->ep);
	return proto;
}

uint64_t
psm3_gen1_gdr_cache_evict() {
	int ret;
	struct hfi1_gdr_cache_evict_params params;
	params.evict_params_in.version = HFI1_GDR_VERSION;
	params.evict_params_in.pages_to_evict = 4;

	ret = ioctl(gdr_fd, HFI1_IOCTL_GDR_GPU_CACHE_EVICT, &params);
	if (ret) {
		/* Fatal error */
		psm3_handle_error(PSMI_EP_NORETURN, PSM2_INTERNAL_ERR,
						  "PIN/MMAP ioctl failed ret %d errno %d\n",
						  ret, errno);
		return ret;
	}

	return params.evict_params_out.pages_evicted;
}


static uint64_t
psm3_gen1_sdma_gpu_cache_evict(int fd) {
	int ret;
	struct hfi1_sdma_gpu_cache_evict_params params;
	params.evict_params_in.version = HFI1_GDR_VERSION;
	params.evict_params_in.pages_to_evict = 2;

	ret = ioctl(fd, HFI1_IOCTL_SDMA_CACHE_EVICT, &params);
	if (ret) {
		/* Fatal error */
		psm3_handle_error(PSMI_EP_NORETURN, PSM2_INTERNAL_ERR,
						  "SDMA Cache Evict failed ret %d errno %d\n",
						  ret, errno);
		return ret;
	}

	return params.evict_params_out.pages_evicted;
}

/* handle_out_of_bar_space is called when the driver tries
 * to self evict in the GDR cache and finds no entries.
 * This could be due to the fact that all the pages pinned
 * in the BAR1 region are cached in the SDMA and TID cache.
 * We try to evict from both the caches for 30 seconds after
 * which we bail out. If successful we retry to PIN/MMAP once
 * again
 */
static uint64_t
handle_out_of_bar_space(struct ips_proto *proto)
{
	time_t lastEvictTime = 0;
	uint64_t lengthEvicted;
	time_t now;
 retry:
	now = time(NULL);

	if (!lastEvictTime)
		lastEvictTime = now;

	if (proto->protoexp && proto->protoexp->tidc.tid_cachemap.payload.nidle) {
		lengthEvicted =
			ips_tidcache_evict(&proto->protoexp->tidc, -1);

		if (lengthEvicted) {
			lastEvictTime = 0;
			return lengthEvicted; /* signals a retry of the writev command. */
		}
	}

	lengthEvicted = psm3_gen1_sdma_gpu_cache_evict(psm3_gen1_get_fd(proto->ep->context.psm_hw_ctxt));
	if (lengthEvicted) {
		lastEvictTime = 0;
		return lengthEvicted;
	}
	static const double thirtySeconds = 30.0;
	if (difftime(now, lastEvictTime) >
		thirtySeconds) {
		return 0;
	} else {
		goto retry;
	}
}

// flags=0 for send, 1 for recv
void *
psm3_gen1_gdr_convert_gpu_to_host_addr(unsigned long buf,
							 size_t size, int flags,
							 psm2_ep_t ep)
{
	struct hfi1_gdr_query_params query_params;
	int ret;
	void *host_addr_buf;

	uintptr_t pageaddr = buf & GPU_PAGE_MASK;
	/* As size is guarenteed to be in the range of 0-8kB
	 * there is a guarentee that buf+size-1 does not overflow
	 * 64 bits.
	 */
	uint32_t pagelen = (uint32_t) (PSMI_GPU_PAGESIZE +
					   ((buf + size - 1) & GPU_PAGE_MASK) -
					   pageaddr);

	psmi_assert(NULL != psm3_gen1_get_proto(ep));
	_HFI_VDBG("buf=%p size=%zu pageaddr=%p pagelen=%u flags=0x%x ep=%p\n",
		(void *)buf, size, (void *)pageaddr, pagelen, flags, ep);
	query_params.query_params_in.version = HFI1_GDR_VERSION;
	query_params.query_params_in.gpu_buf_addr = pageaddr;
	query_params.query_params_in.gpu_buf_size = pagelen;
 retry:

	ret = ioctl(gdr_fd, HFI1_IOCTL_GDR_GPU_PIN_MMAP, &query_params);

	if (ret) {
		if (errno == ENOMEM || errno == EINVAL) {
			if (!handle_out_of_bar_space(psm3_gen1_get_proto(ep))) {
				/* Fatal error */
				psm3_handle_error(PSMI_EP_NORETURN, PSM2_INTERNAL_ERR,
						  "Unable to PIN GPU pages(Out of BAR1 space) (errno: %d)\n", errno);
				return NULL;
			} else {
				goto retry;
			}
		} else {
			/* Fatal error */
			psm3_handle_error(PSMI_EP_NORETURN, PSM2_INTERNAL_ERR,
							  "PIN/MMAP ioctl failed ret %d errno %d\n",
							  ret, errno);
			return NULL;
		}
	}
	host_addr_buf = (void *)query_params.query_params_out.host_buf_addr;
	return host_addr_buf + (buf & GPU_PAGE_OFFSET_MASK);
}

void psm3_hfp_gen1_gdr_open()
{
	gdr_fd = open(GDR_DEVICE_PATH, O_RDWR);
	if (-1 == gdr_fd ) {
		/* Non-Fatal error. If device cannot be found we assume
		 * that the driver does not support GDR Copy and we fallback
		 * to sending all GPU messages using rndv protocol
		 */
		_HFI_INFO(" Warning: The HFI1 driver installed does not support GPUDirect RDMA"
				  " fast copy. Turning off GDR fast copy in PSM \n");
		is_gdr_copy_enabled = gdr_copy_limit_send =
					gdr_copy_limit_recv = 0;
		return;
	}
	return;
}

void psm3_gen1_gdr_close()
{
	close(gdr_fd);
}

#endif /* PSM_CUDA */
#endif /* PSM_OPA */
