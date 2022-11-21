/*

  This file is provided under a dual BSD/GPLv2 license.  When using or
  redistributing this file, you may do so under either license.

  GPL LICENSE SUMMARY

  Copyright(c) 2022 Intel Corporation.

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

  Copyright(c) 2022 Intel Corporation.

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

#ifdef PSM_ONEAPI

#include "psm_user.h"
#include "psm_am_internal.h"
#include "am_oneapi_memhandle.h"
#include <fcntl.h>
#include <unistd.h>
#if HAVE_DRM
#include <sys/ioctl.h>
#include <drm/i915_drm.h>
#endif
#if HAVE_LIBDRM
#include <sys/ioctl.h>
#include <libdrm/i915_drm.h>
#endif


/*
 * The key used to search the cache is the senders buf address pointer.
 * Upon a succesful hit in the cache, additional validation is required
 * as multiple senders could potentially send the same buf address value.
 */
ze_device_handle_t*
am_ze_memhandle_acquire(struct ptl_am *ptl, uintptr_t sbuf, ze_ipc_mem_handle_t *handle,
				uint32_t length, int *ipc_fd, psm2_epaddr_t epaddr)
{
	void *ze_ipc_dev_ptr = NULL;
	psm2_epid_t epid = epaddr->epid;
#if HAVE_DRM || HAVE_LIBDRM
	ze_ipc_mem_handle_t ze_handle;
	am_epaddr_t *am_epaddr = (am_epaddr_t*)epaddr;
	int fd;
	struct drm_prime_handle open_fd = {0, 0, 0};
#endif
	_HFI_VDBG("sbuf=%lu,handle=%p,length=%u,epid=%s\n",
			sbuf, handle, length, psm3_epid_fmt_internal(epid, 0));

#if HAVE_DRM || HAVE_LIBDRM
	fd = am_epaddr->peer_fds[0];
	open_fd.flags = DRM_CLOEXEC | DRM_RDWR;
	open_fd.handle = *(int *)handle;

	if (ioctl(fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &open_fd) < 0) {
		_HFI_ERROR("ioctl failed for DRM_IOCTL_PRIME_HANDLE_TO_FD: %s\n", strerror(errno));
		psm3_handle_error(ptl->ep, PSM2_INTERNAL_ERR,
			"ioctl "
			"failed for DRM_IOCTL_PRIME_HANDLE_TO_FD errno=%d",
			errno);
		return NULL;
	}
	memset(&ze_handle, 0, sizeof(ze_handle));
	memcpy(&ze_handle, &open_fd.fd, sizeof(open_fd.fd));
	*ipc_fd = open_fd.fd;
	PSMI_ONEAPI_ZE_CALL(zeMemOpenIpcHandle, ze_context, cur_ze_dev->dev, *((ze_ipc_mem_handle_t *)&ze_handle),
			 0, (void **)&ze_ipc_dev_ptr);
#else // if no drm, set up to return NULL as oneapi ipc handles don't work without drm
	ze_ipc_dev_ptr = NULL;
#endif // HAVE_DRM || HAVE_LIBDRM
	return ze_ipc_dev_ptr;

}

void
am_ze_memhandle_release(ze_device_handle_t *ze_ipc_dev_ptr)
{
	PSMI_ONEAPI_ZE_CALL(zeMemCloseIpcHandle, ze_context, ze_ipc_dev_ptr);
	return;
}

#endif
