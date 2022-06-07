/*

  This file is provided under a dual BSD/GPLv2 license.  When using or
  redistributing this file, you may do so under either license.

  GPL LICENSE SUMMARY

  Copyright(c) 2021 Intel Corporation.

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

  Copyright(c) 2021 Intel Corporation.

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

const char* psmi_oneapi_ze_result_to_string(const ze_result_t result) {
#define ZE_RESULT_CASE(RES) case ZE_RESULT_##RES: return STRINGIFY(RES)

	switch (result) {
	ZE_RESULT_CASE(SUCCESS);
	ZE_RESULT_CASE(NOT_READY);
	ZE_RESULT_CASE(ERROR_UNINITIALIZED);
	ZE_RESULT_CASE(ERROR_DEVICE_LOST);
	ZE_RESULT_CASE(ERROR_INVALID_ARGUMENT);
	ZE_RESULT_CASE(ERROR_OUT_OF_HOST_MEMORY);
	ZE_RESULT_CASE(ERROR_OUT_OF_DEVICE_MEMORY);
	ZE_RESULT_CASE(ERROR_MODULE_BUILD_FAILURE);
	ZE_RESULT_CASE(ERROR_INSUFFICIENT_PERMISSIONS);
	ZE_RESULT_CASE(ERROR_NOT_AVAILABLE);
	ZE_RESULT_CASE(ERROR_UNSUPPORTED_VERSION);
	ZE_RESULT_CASE(ERROR_UNSUPPORTED_FEATURE);
	ZE_RESULT_CASE(ERROR_INVALID_NULL_HANDLE);
	ZE_RESULT_CASE(ERROR_HANDLE_OBJECT_IN_USE);
	ZE_RESULT_CASE(ERROR_INVALID_NULL_POINTER);
	ZE_RESULT_CASE(ERROR_INVALID_SIZE);
	ZE_RESULT_CASE(ERROR_UNSUPPORTED_SIZE);
	ZE_RESULT_CASE(ERROR_UNSUPPORTED_ALIGNMENT);
	ZE_RESULT_CASE(ERROR_INVALID_SYNCHRONIZATION_OBJECT);
	ZE_RESULT_CASE(ERROR_INVALID_ENUMERATION);
	ZE_RESULT_CASE(ERROR_UNSUPPORTED_ENUMERATION);
	ZE_RESULT_CASE(ERROR_UNSUPPORTED_IMAGE_FORMAT);
	ZE_RESULT_CASE(ERROR_INVALID_NATIVE_BINARY);
	ZE_RESULT_CASE(ERROR_INVALID_GLOBAL_NAME);
	ZE_RESULT_CASE(ERROR_INVALID_KERNEL_NAME);
	ZE_RESULT_CASE(ERROR_INVALID_FUNCTION_NAME);
	ZE_RESULT_CASE(ERROR_INVALID_GROUP_SIZE_DIMENSION);
	ZE_RESULT_CASE(ERROR_INVALID_GLOBAL_WIDTH_DIMENSION);
	ZE_RESULT_CASE(ERROR_INVALID_KERNEL_ARGUMENT_INDEX);
	ZE_RESULT_CASE(ERROR_INVALID_KERNEL_ARGUMENT_SIZE);
	ZE_RESULT_CASE(ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE);
	ZE_RESULT_CASE(ERROR_INVALID_COMMAND_LIST_TYPE);
	ZE_RESULT_CASE(ERROR_OVERLAPPING_REGIONS);
	ZE_RESULT_CASE(ERROR_UNKNOWN);
	default:
		return "Unknown error";
	}

#undef ZE_RESULT_CASE
}

void psmi_oneapi_ze_memcpy(void *dstptr, const void *srcptr, size_t size)
{
	PSMI_ONEAPI_ZE_CALL(zeCommandListReset, ze_cl);
	PSMI_ONEAPI_ZE_CALL(zeCommandListAppendMemoryCopy, ze_cl, dstptr, srcptr, size, NULL, 0, NULL);
	PSMI_ONEAPI_ZE_CALL(zeCommandListClose, ze_cl);
	PSMI_ONEAPI_ZE_CALL(zeCommandQueueExecuteCommandLists, ze_cq, 1, &ze_cl, NULL);
	PSMI_ONEAPI_ZE_CALL(zeCommandQueueSynchronize, ze_cq, UINT32_MAX);
}

#endif // PSM_ONEAPI
