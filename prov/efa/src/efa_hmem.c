/*
 * Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
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

#include "efa.h"

/**
 * @brief determine the support status of cuda memory pointer
 *
 * @param	cuda_status[out]	cuda memory support status
 * @return	0 on success
 * 		negative libfabric error code on failure
 */
static int efa_hmem_support_status_update_cuda(struct efa_hmem_support_status *cuda_status)
{
#if HAVE_CUDA
	cudaError_t cuda_ret;
	void *ptr = NULL;
	struct ibv_mr *ibv_mr;
	int ibv_access = IBV_ACCESS_LOCAL_WRITE;
	size_t len = ofi_get_page_size() * 2;
	int ret;

	if (!ofi_hmem_is_initialized(FI_HMEM_CUDA)) {
		EFA_INFO(FI_LOG_DOMAIN,
		         "FI_HMEM_CUDA is not initialized\n");
		return 0;
	}

	if (efa_device_support_rdma_read())
		ibv_access |= IBV_ACCESS_REMOTE_READ;

	cuda_status->initialized = true;

	cuda_ret = ofi_cudaMalloc(&ptr, len);
	if (cuda_ret != cudaSuccess) {
		EFA_WARN(FI_LOG_DOMAIN,
			 "Failed to allocate CUDA buffer: %s\n",
			 ofi_cudaGetErrorString(cuda_ret));
		return -FI_ENOMEM;
	}

	ibv_mr = ibv_reg_mr(g_device_list[0].ibv_pd, ptr, len, ibv_access);
	if (!ibv_mr) {
		EFA_WARN(FI_LOG_DOMAIN,
			 "Failed to register CUDA buffer with the EFA device, FI_HMEM transfers that require peer to peer support will fail.\n");
		ofi_cudaFree(ptr);
		return 0;
	}

	ret = ibv_dereg_mr(ibv_mr);
	ofi_cudaFree(ptr);
	if (ret) {
		EFA_WARN(FI_LOG_DOMAIN,
			 "Failed to deregister CUDA buffer: %s\n",
			 fi_strerror(-ret));
		return ret;
	}

	cuda_status->p2p_supported = true;
#endif
	return 0;
}

/**
 * @brief determine the support status of neuron memory pointer
 *
 * @param	neuron_status[out]	neuron memory support status
 * @return	0 on success
 * 		negative libfabric error code on failure
 */
static int efa_hmem_support_status_update_neuron(struct efa_hmem_support_status *neuron_status)
{
#if HAVE_NEURON
	struct ibv_mr *ibv_mr;
	int ibv_access = IBV_ACCESS_LOCAL_WRITE;
	void *handle;
	void *ptr = NULL;
	size_t len = ofi_get_page_size() * 2;
	int ret;

	if (!ofi_hmem_is_initialized(FI_HMEM_NEURON)) {
		EFA_INFO(FI_LOG_DOMAIN,
		         "FI_HMEM_NEURON is not initialized\n");
		return 0;
	}

	if (domain->device->device_caps & EFADV_DEVICE_ATTR_CAPS_RDMA_READ) {
		ibv_access |= IBV_ACCESS_REMOTE_READ;
	} else {
		EFA_WARN(FI_LOG_DOMAIN,
			 "No EFA RDMA read support, transfers using AWS Neuron will fail.\n");
		return 0;
	}

	neuron_status->initialized = true;

	ptr = neuron_alloc(&handle, len);
	if (!ptr)
		return -FI_ENOMEM;

	ibv_mr = ibv_reg_mr(domain->ibv_pd, ptr, len, ibv_access);
	if (!ibv_mr) {
		EFA_WARN(FI_LOG_DOMAIN,
			 "Failed to register Neuron buffer with the EFA device, FI_HMEM transfers that require peer to peer support will fail.\n");
		neuron_free(&handle);
		return 0;
	}

	ret = ibv_dereg_mr(ibv_mr);
	neuron_free(&handle);
	if (ret) {
		EFA_WARN(FI_LOG_DOMAIN,
			 "Failed to deregister Neuron buffer: %s\n",
			 fi_strerror(-ret));
		return ret;
	}

	neuron_status->p2p_supported = true;
#endif
	return 0;
}

/**
 * @brief Determine the support status of all HMEM devices
 * The support status is used later when
 * determining how to initiate an HMEM transfer.
 *
 * @param 	all_status[out]		an array of struct efa_hmem_support_status,
 * 					whose size is OFI_HMEM_MAX
 * @return	0 on success
 * 		negative libfabric error code on an unexpected error
 */
int efa_hmem_support_status_update_all(struct efa_hmem_support_status *all_status)
{
	int ret, err;

	if(g_device_cnt <= 0) {
		return -FI_ENODEV;
	}

	memset(all_status, 0, OFI_HMEM_MAX * sizeof(struct efa_hmem_support_status));

	ret = 0;

	err = efa_hmem_support_status_update_cuda(&all_status[FI_HMEM_CUDA]);
	if (err) {
		ret = err;
		EFA_WARN(FI_LOG_DOMAIN, "check cuda support status failed! err: %d\n",
			 err);
	}

	err = efa_hmem_support_status_update_neuron(&all_status[FI_HMEM_NEURON]);
	if (err) {
		ret = err;
		EFA_WARN(FI_LOG_DOMAIN, "check neuron support status failed! err: %d\n",
			 err);
	}

	return ret;
}
