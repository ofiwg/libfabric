/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2021 Hewlett Packard Enterprise Development LP
 */

#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "hsa.h"
#include "hsa_ext_image.h"
#include "hsa_ext_amd.h"

/*
Build on fauci-login
export DBS=~/devbootstrap
hipcc --amdgpu-target=gfx908 -I$DBS/libcxi/install/include/libcxi \
	-I$DBS/cassini-headers/install/include \
	-I$DBS/cxi-driver/include hip_cntr_test.cpp \
	-I/opt/rocm/include/hsa -L$DBS/libcxi/install/lib \
	-L/opt/rocm/lib64/ -L/opt/rocm/lib -lcxi -lhsa-runtime64 \
	-o hip_cntr_test

On faucin1 (or n2)
~/devbootstrap/libcxi/utils/hip_cntr_test
*/

#define __user

_Pragma("clang diagnostic push")
_Pragma("clang diagnostic ignored \"-Wc99-extensions\"")
#include "libcxi.h"
_Pragma("clang diagnostic pop")

/* from gtlt example */
static int found_mempool      = 0;
static int found_host_mempool = 0;
static int found_gpu_mempool  = 0;
static     hsa_amd_memory_pool_t gtlt_hsa_host_mempool;
static     hsa_amd_memory_pool_t gtlt_hsa_gpu_mempool;

static int gtl_num_gpu_devices_on_node;
uint64_t   table_size = 0;

hsa_agent_t gtlt_hsa_gpu_agents[32];
hsa_agent_t gtlt_hsa_agents[32];
int         gtlt_hsa_num_agents;

struct cxil_dev *dev;
struct cxil_lni *lni;
struct cxi_cp *cp;
struct cxil_domain *dom;
struct cxi_ct *ct;
struct cxi_cq *trig_cmdq;
struct c_ct_writeback *wb;
struct c_ct_writeback *dev_wb;

size_t mmio_len;
void volatile *mmio_addr;
void *mapped_addr;

static hsa_status_t mempool_cb(hsa_amd_memory_pool_t mempool, void *data)
{
	hsa_status_t           hsa_err = HSA_STATUS_SUCCESS;
	hsa_amd_segment_t      segment;
	int                    can_allocate = 0;
	/* int                    all_access   = 0; */
	hsa_amd_memory_pool_t *mempool_out  = (hsa_amd_memory_pool_t *)data;

	if (found_mempool) {
		return HSA_STATUS_SUCCESS;
	}

	hsa_err = hsa_amd_memory_pool_get_info(mempool,
					       HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
					       &segment);
	if (hsa_err != HSA_STATUS_SUCCESS) {
		return hsa_err;
	}

	if (segment != HSA_AMD_SEGMENT_GLOBAL) {
		return HSA_STATUS_SUCCESS;
	}

	hsa_err = hsa_amd_memory_pool_get_info(mempool,
					       HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
					       &can_allocate);
	if (hsa_err != HSA_STATUS_SUCCESS) {
		return hsa_err;
	}

	if (!can_allocate) {
		return HSA_STATUS_SUCCESS;
	}

	found_mempool = 1;
	*mempool_out  = mempool;

	return hsa_err;
}

static hsa_status_t agent_cb(hsa_agent_t agent, void *data)
{
	hsa_status_t      hsa_err = HSA_STATUS_SUCCESS;
	hsa_device_type_t device_type;

	hsa_err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
	if (hsa_err != HSA_STATUS_SUCCESS) {
		return hsa_err;
	}

	// printf("device_type:%d\n", device_type);
	if (device_type == HSA_DEVICE_TYPE_GPU) {
		gtlt_hsa_gpu_agents[gtl_num_gpu_devices_on_node++] = agent;

		if (!found_gpu_mempool) {
			hsa_err = hsa_amd_agent_iterate_memory_pools(agent,
								     mempool_cb,
								     &gtlt_hsa_gpu_mempool);
			if (hsa_err != HSA_STATUS_SUCCESS) {
				printf("hsa_amd_agent_iterate_memory_pools ret:%d\n", hsa_err);
				return hsa_err;
			}
			found_gpu_mempool = found_mempool;
			found_mempool     = 0;
		}
	} else {
		if (!found_host_mempool) {
			hsa_err = hsa_amd_agent_iterate_memory_pools(agent,
								     mempool_cb,
								     &gtlt_hsa_host_mempool);
			if (hsa_err != HSA_STATUS_SUCCESS) {
				printf("hsa_amd_agent_iterate_memory_pools ret:%d\n", hsa_err);
				return hsa_err;
			}
			found_host_mempool = found_mempool;
			found_mempool      = 0;
		}
	}

	gtlt_hsa_agents[gtlt_hsa_num_agents++] = agent;

	return HSA_STATUS_SUCCESS;
}

void gtlt_hsa_init(void)
{
	int          ret     = 0;
	hsa_status_t hsa_err = HSA_STATUS_SUCCESS;
	char        *env     = NULL;

	ret = hsa_init();
	assert(!ret);

	gtlt_hsa_num_agents         = 0;
	gtl_num_gpu_devices_on_node = 0;

	ret = hsa_iterate_agents(agent_cb, NULL);
	assert(!ret);

	printf("gtlt_hsa_num_agents:%d\n", gtlt_hsa_num_agents);

	assert(found_host_mempool);
}

void cntr_alloc(void)
{
	int ret;
	uint32_t vni = 1;
	uint32_t pid = C_PID_ANY;

	struct cxi_cq_alloc_opts cq_opts = {};

	ret = cxil_open_device(0, &dev);
	assert(!ret);

	ret = cxil_alloc_lni(dev, &lni, CXI_DEFAULT_SVC_ID);
	assert(!ret);

	ret = cxil_alloc_cp(lni, vni, CXI_TC_BEST_EFFORT,
			    CXI_TC_TYPE_DEFAULT, &cp);
	assert(!ret);

	ret = cxil_alloc_domain(lni, vni, pid, &dom);
	assert(!ret);

	cq_opts.count = 64;
	cq_opts.flags = CXI_CQ_IS_TX | CXI_CQ_TX_WITH_TRIG_CMDS;
	cq_opts.policy = CXI_CQ_UPDATE_ALWAYS;
	ret = cxil_alloc_cmdq(lni, NULL, &cq_opts, &trig_cmdq);
	assert(!ret);

	wb = (struct c_ct_writeback *) aligned_alloc(8, sizeof(*wb));
	memset(wb, 0, sizeof(*wb));

	ret = cxil_alloc_ct(lni, wb, &ct);
	assert(!ret);

	get_mmio_addr(ct, &mmio_addr, &mmio_len);
}

void cntr_free(void)
{
	int ret;

	ret = cxil_destroy_ct(ct);
	assert(!ret);

	free(wb);

	ret = cxil_destroy_cmdq(trig_cmdq);
	assert(!ret);

	ret = cxil_destroy_domain(dom);
	assert(!ret);

	ret = cxil_destroy_cp(cp);
	assert(!ret);

	ret = cxil_destroy_lni(lni);
	assert(!ret);

	cxil_close_device(dev);
}

void expect_ct_values(struct cxi_ct *ct, uint64_t success, uint8_t failure)
{
	time_t timeout;
	struct c_ct_writeback *wb = ct->wb;

	// Wait for valid CT writeback
	timeout = time(NULL) + 5;
	while (wb->ct_writeback == 0) {
		if (time(NULL) > timeout)
			break;

		sched_yield();
	}

	assert(time(NULL) <= timeout);
	printf("ct_success:%ld\n", wb->ct_success);
	printf("ct_failure:%d\n", wb->ct_failure);

	assert(wb->ct_success == success);
	assert(wb->ct_failure == failure);

	// Reset the writeback bit
	wb->ct_writeback = 0;
}

#define CNTR_SUCCESS_MAX ((1ULL << 48) - 1)

void expect_wb_value(uint64_t *dev_wb, uint64_t value)
{
	time_t timeout;

	// Wait for valid CT writeback
	timeout = time(NULL) + 5;
	while ((*dev_wb & CNTR_SUCCESS_MAX) != value) {
		assert(time(NULL) <= timeout);
		sched_yield();
	}

	printf("dev_wb:%lx\n", *dev_wb);

	assert((*dev_wb & CNTR_SUCCESS_MAX) == value);
}

// read the writeback buffer from a kernel
__device__ void dev_expect_wb_value(uint64_t *dev_wb, uint64_t value)
{
	int i;
	int to = 10000000;

	printf("value:%lx\n", value);

	for (i = 0; i < to; i++)
	{
		if (*dev_wb == value)
			break;
	}

	printf("i: %d dev_wb:%lx\n", i, *dev_wb);

	assert(*dev_wb == value);
}

void get_mmio_addr(struct cxi_ct *ct, void volatile **addr, size_t *len)
{
	*addr = ct->doorbell;
	*len = sizeof(ct->doorbell);
}

__device__ __host__ void cntr_add(void volatile *cntr_mmio, int value)
{
	// printf("cntr_mmio:%p value:%d\n", cntr_mmio, value);
	*((uint64_t volatile *)cntr_mmio) = value  & CT_SUCCESS_MASK;
}

__device__ void cntr_adderr(void volatile *cntr_mmio, int value)
{
	*((uint64_t *)cntr_mmio + 8) = value;
}

void cntr_set(void volatile *cntr_mmio)
{
	*((uint64_t *)cntr_mmio + 16) = 0;
}

void cntr_seterr(void volatile *cntr_mmio)
{
	*((uint64_t *)cntr_mmio + 24) = 0;
}

__device__ __host__ uint64_t gen_cntr_success(uint64_t value)
{
	return (1ULL << 63) | value;
};

// Issue a CTGet to a counter
void cntr_read(void)
{
	int ret;
	struct c_ct_cmd cmd = {
		.ct = (uint16_t)ct->ctn,
	};

	ret = cxi_cq_emit_ct(trig_cmdq, C_CMD_CT_GET, &cmd);
	assert(!ret);

	cxi_cq_ring(trig_cmdq);
}

/* Cause an event to write the writeback buffer when threshold is reached */
void cntr_trigger(int threshold)
{
	int ret;
	struct c_ct_cmd cmd = {
		.ct = (uint16_t)ct->ctn,
		.trig_ct = (uint16_t)ct->ctn,
		.ct_success = 1,
		.set_ct_success = 1,
		.threshold = (uint64_t)threshold,
	};

	ret = cxi_cq_emit_ct(trig_cmdq, C_CMD_CT_TRIG_EVENT, &cmd);
	assert(!ret);

	cxi_cq_ring(trig_cmdq);
}

/* kernel runs on GPU */
__global__ void kernel1(void *mmio_a, bool do_err, int value, int n)
{
	int i;

	printf("blockDim:%d blockIdx:%d threadIdx:%d\n", (int)blockDim.x,
		   (int)blockIdx.x, (int)threadIdx.x);

	/* add error */
	if (do_err)
		cntr_adderr(mmio_a, value);
	else {
		for (i = 0; i < n; i++)
			cntr_add(mmio_a, value);
	}
}

/* Write to doorbell and wait for writeback to reach value
 * Expects that cntr_trigger() was called previously.
 */
__global__ void kernel2(void *mmio_a, void *gpu_wb, int value)
{
	cntr_add(mmio_a, value);
	dev_expect_wb_value((uint64_t*)gpu_wb, gen_cntr_success(value));
}

int main( int argc, char *argv[] )
{
	int i;
	int ret;
	int n = 8;
	int value = 2;
	int blockSize = 1;
	int gridSize = 1;
	void *gpu_wb;

	gtlt_hsa_init();

	cntr_alloc();

	ret = hsa_amd_memory_lock_to_pool((void *)mmio_addr, mmio_len, NULL, 0,
					  gtlt_hsa_host_mempool, 0,
					  &mapped_addr);
	assert(!ret);

	hipLaunchKernelGGL(kernel1, dim3(gridSize), dim3(blockSize), 0, 0,
			   mapped_addr, true, 2, 1);
	hipDeviceSynchronize( );
	printf("Finished executing Kernel\n");

	// check for error writeback
	expect_ct_values(ct, 0, 2);

	// clear error
	cntr_seterr(mmio_addr);
	expect_ct_values(ct, 0, 0);

	blockSize = 2;
	// cntr_trigger(value * n * blockSize);
	cntr_trigger(value * n * 1);

	hipLaunchKernelGGL(kernel1, dim3(gridSize), dim3(blockSize), 0, 0,
			   mapped_addr, false, value, n);
	hipDeviceSynchronize( );

	/* Writing from multiple kernels does not write the doorbell
	 * multiple times. This needs to be investigated.
	 */
	// expect_ct_values(ct, value * n * blockSize, 0);
	expect_ct_values(ct, value * n * 1, 0);

	// clear counter value
	cntr_set(mmio_addr);
	expect_ct_values(ct, 0, 0);

	for (i = 0; i < n; i++)
		cntr_add(mmio_addr, value);

	cntr_read();
	expect_ct_values(ct, value * n, 0);

	// clear counter value
	cntr_set(mmio_addr);
	expect_ct_values(ct, 0, 0);

	ret = hipMalloc(&gpu_wb, sizeof(*dev_wb));
	assert(ret == hipSuccess);

	ret = cxil_ct_wb_update(ct, (struct c_ct_writeback *) gpu_wb);
	assert(!ret);

	value = 10;
	cntr_trigger(value);

	hipLaunchKernelGGL(kernel2, dim3(gridSize), dim3(blockSize), 0, 0,
			   mapped_addr, gpu_wb, value);
	hipDeviceSynchronize( );

	hipFree(gpu_wb);
	cntr_free();

	return 0;
}
