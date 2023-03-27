/*

  This file is provided under a dual BSD/GPLv2 license.  When using or
  redistributing this file, you may do so under either license.

  GPL LICENSE SUMMARY

  Copyright(c) 2019 Intel Corporation.

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

  Copyright(c) 2019 Intel Corporation.

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

/* Copyright (c) 2109-2022 Intel Corporation. All rights reserved. */

#ifdef PSM_DSA
/* routines to take advantage of SPR Xeon Data Streaming Accelerator */
#include <stdio.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include "utils_dsa.h"

#include "utils_debug.h"
#include "utils_user.h"
#include "psm_user.h"

#include <linux/idxd.h>
#include <unistd.h>

// just need 1 page for access to DSA WorkQueue
//#define DSA_MMAP_LEN 0x1000
#define DSA_MMAP_LEN PSMI_PAGESIZE

static int dsa_available;
static uint32_t dsa_ratio;  // 1/ratio of the copy will use CPU, 0=none, 1=all
#define DSA_THRESH 8000
static uint32_t dsa_thresh; // copies > thresh will use DSA
				// note this is copy size, which will be 1 to AMLONG_MTU

// SPR has a max of 4 DSA devices per socket, each with 4 engines.
// Work queue groups can allow more than 1 WQ per engine and/or more
// then 1 engine per WQ.  A Dedicated WQ (DWQ) can only be used by 1 process.
// The memory needs based on MAX_PROC and MAX_QUEUES are modest so we set
// these high to allow experiments.
// However, as the total number of processes, queues and threads per process
// grow to exceed total number of engines available, some head of line
// blocking will occur and may reduce the value of using DSA.

// Max processes or CPU sockets (numa 0-N) per node
#define DSA_MAX_PROC 256
// Max WQ per process
// We only use more than 1 WQ per process when there is more than 1 thread
// per process (such as OneCCL workers or Intel MPI Multi-EP threading).
// But expected counts for such are modest (2-4 for Intel MPI, 8-16 for OneCCL)
#define DSA_MAX_QUEUES 32
// information parsed from PSM3_DSA_WQS
static char *dsa_wq_filename[DSA_MAX_PROC][DSA_MAX_QUEUES];
static uint32_t dsa_num_wqs[DSA_MAX_PROC];
static uint32_t dsa_num_proc;

// information relevant to our PROC
struct dsa_wq {
	const char *wq_filename;	// points into dsa_wq_filename
	void *wq_reg;	// mmap memory
	uint32_t use_count;	// how many threads assigned to this WQ
};
static struct dsa_wq dsa_wqs[DSA_MAX_QUEUES];
static uint32_t dsa_my_num_wqs;
static uint32_t dsa_my_dsa_str_len; // sum(strlen(wq_filename)+1)
static psmi_spinlock_t dsa_wq_lock; // protects dsa_wq.use_count


// Each thread is assigned a DSA WQ on 1st memcpy
static __thread void *dsa_wq_reg = NULL;
// we keep completion record in thread local storage instead of stack
// this way if a DSA completion times out and arrives late it still has a
// valid place to go
static __thread struct dsa_completion_record dsa_comp[2];

// DSA timeout in nanoseconds.  Since our largest copy will be AMLONG_MTU
// 1 second should me much more than enough.  If DSA takes longer something
// is wrong and we will stop using it for the rest of the job
#define DSA_TIMEOUT (10000000000ULL)

#if 0
/* enqcmd is applicable to shared (multi-process) DSA workqueues */
static inline unsigned char enqcmd(struct dsa_hw_desc *desc,
			volatile void *reg)
{
	unsigned char retry;

	asm volatile(".byte 0xf2, 0x0f, 0x38, 0xf8, 0x02\t\n"
			"setz %0\t\n"
			: "=r"(retry) : "a" (reg), "d" (desc));
	return retry;
}
#endif

/* movdir64b is applicable to dedicated (single process) DSA workqueues */
static inline void movdir64b(struct dsa_hw_desc *desc, volatile void *reg)
{
	asm volatile(".byte 0x66, 0x0f, 0x38, 0xf8, 0x02\t\n"
		: : "a" (reg), "d" (desc));
}


#if 0
static __always_inline
void dsa_desc_submit(void *wq_portal, int dedicated,
		struct dsa_hw_desc *hw)
{
	// make sure completion status zeroing fully written before post to HW
	//_mm_sfence();
	{ asm volatile("sfence":::"memory"); }

	/* use MOVDIR64B for DWQ */
	if (dedicated)
		movdir64b(hw, wq_portal);
	else /* use ENQCMDS for SWQ */
		while (enqcmd(hw, wq_portal))
			;
}
#else
static __always_inline
void dsa_desc_submit(void *wq_portal, struct dsa_hw_desc *hw)
{
	// make sure completion status zeroing fully written before post to HW
	//_mm_sfence();
	{ asm volatile("sfence":::"memory"); }

	/* use MOVDIR64B for DWQ */
	movdir64b(hw, wq_portal);
}
#endif

/* use DSA to copy a block of memory */
/* !rx-> copy from app to shm (sender), rx-> copy from shm to app (receiver) */
void psm3_dsa_memcpy(void *dest, const void *src, uint32_t n, int rx,
		struct dsa_stats *stats)
{
	struct dsa_hw_desc desc = {};
	struct dsa_completion_record *comp;
	void *dsa_dest;
	const void *dsa_src;
	uint32_t dsa_n;
	uint32_t cpu_n;
	uint64_t start_cycles, end_cycles;
	uint64_t loops;

#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
	if (n && PSMI_IS_GPU_ENABLED && (PSMI_IS_GPU_MEM(dest) || PSMI_IS_GPU_MEM((void *) src))) {
		_HFI_VDBG("GPU copy from %p to %p for %u\n", src, dest, n);
		PSM3_GPU_MEMCPY(dest, src, n);
		return;
	}
#endif
	if (n <= dsa_thresh) {
		_HFI_VDBG("CPU copy from %p to %p for %u\n", src, dest, n);
		memcpy(dest, src, n);
		return;
	}

	// TBD - add some statistics for DSA vs CPU copy use
	// to maximize performance we do part of the copy with CPU while we
	// wait for DSA to copy the rest
	if (dsa_ratio) {
		cpu_n = n/dsa_ratio;
		// TBD - should we compute so DSA gets a full multiple of pages and CPU
		// does the rest?  Should we start DSA on a page boundary?
		// round down to page boundary
		//cpu_n = ROUNDDOWNP2(cpu_n, PSMI_PAGESIZE);

		// round to a multiple of 8 bytes at least
		cpu_n = ROUNDDOWNP2(cpu_n, 8);
	} else {
		cpu_n = 0;
	}
	dsa_n = n - cpu_n;
	dsa_src = (void*)((uintptr_t)src + cpu_n);
	dsa_dest = (void*)((uintptr_t)dest + cpu_n);
	psmi_assert(dsa_n);
	_HFI_VDBG("DSA copy from %p to %p for %u (%u via CPU, %u via DSA)\n", src, dest, n, cpu_n, dsa_n);

	// comp ptr must be 32 byte aligned
	comp = (struct dsa_completion_record *)(((uintptr_t)&dsa_comp[0] + 0x1f) & ~0x1f);
	comp->status = 0;
	desc.opcode = DSA_OPCODE_MEMMOVE;
	/* set CRAV (comp address valid) and RCR (request comp) so get completion */
	desc.flags = IDXD_OP_FLAG_CRAV;
	desc.flags |= IDXD_OP_FLAG_RCR;
	/* the CC flag controls whether the dest writes will target memory (0) or
	 * target CPU cache.  When copying to shm (!rx) receiver will just read
	 * memory via DSA.  But when copying to app buffer (rx), app is likely to
	 * access data soon so updating CPU cache is probably better
	 */
	if (rx)
		desc.flags |= IDXD_OP_FLAG_CC;
	// BOF will block engine on page faults and have OS fix it up.  While
	// simpler for this function, this can leave an entire engine stalled and
	// cause head of line blocking for other queued operations which is worse
	// for overall server.  Best to take the pain here as page faults should
	// be rare during steady state of most apps
	// desc.flags |= IDXD_OP_FLAG_BOF;
	desc.xfer_size = dsa_n;
	desc.src_addr = (uintptr_t)dsa_src;
	desc.dst_addr = (uintptr_t)dsa_dest;
	desc.completion_addr = (uintptr_t)comp;

	//dsa_desc_submit(dsa_wq_reg, atoi(argv[2]), &desc);
	dsa_desc_submit(dsa_wq_reg, &desc);

	if (cpu_n) {
		// while DSA does it's thing, we copy rest via CPU
		memcpy(dest, src, cpu_n);
	}

	stats->dsa_copy++;
	stats->dsa_copy_bytes += dsa_n;

	// wait for DSA to finish
	start_cycles = get_cycles();
	end_cycles = start_cycles + nanosecs_to_cycles(DSA_TIMEOUT);
	psmi_assert(DSA_COMP_NONE == 0);
	loops = 0;
	while (comp->status == 0) {
		// before declaring timeout, check status one more time, just in
		// case our process got preempted for a long time
		if (get_cycles() > end_cycles && comp->status == 0) {
			_HFI_INFO("Disabling DSA: DSA Hardware Timeout\n");
			dsa_available = 0;
			memcpy(dsa_dest, dsa_src, dsa_n);
			stats->dsa_error++;
			return;
		}
		loops++;
	}
	if (!loops)
		stats->dsa_no_wait++;
	else
		stats->dsa_wait_ns += cycles_to_nanosecs(get_cycles() - start_cycles);

	if (comp->status != DSA_COMP_SUCCESS) {
		// only page faults are expected, other errors we stop using DSA.
		// In all cases we recover with a CPU memcpy
		if ((comp->status & DSA_COMP_STATUS_MASK) != DSA_COMP_PAGE_FAULT_NOBOF) {
			_HFI_INFO("Disabling DSA: DSA desc failed: unexpected status %u\n", comp->status);
			dsa_available = 0;
			stats->dsa_error++;
		} else  {
			if (comp->status & DSA_COMP_STATUS_WRITE)
				stats->dsa_page_fault_wr++;
			else
				stats->dsa_page_fault_rd++;
			_HFI_VDBG("DSA desc failed: page fault status %u\n", comp->status);
		}
		memcpy(dsa_dest, dsa_src, dsa_n);
		return;
	}
	return;
}

static void dsa_free_wqs(void)
{
	int proc;
	int i;

	for (i=0; i<dsa_my_num_wqs; i++) {
		if (dsa_wqs[i].wq_reg)
			(void)munmap(dsa_wqs[i].wq_reg, DSA_MMAP_LEN);
		dsa_wqs[i].wq_reg = NULL;
		// points into dsa_wq_filename[], don't free
		dsa_wqs[i].wq_filename = NULL;
	}
	// free what we parsed
	for (proc=0; proc < dsa_num_proc; proc++) {
		for (i=0; i<dsa_num_wqs[proc]; i++) {
			psmi_free(dsa_wq_filename[proc][i]);
			dsa_wq_filename[proc][i] = NULL;
		}
	}
}

/* initialize DSA - call once per process */
/* Some invalid inputs and DSA initialization errors are treated as fatal errors
 * since if DSA gets initialized on some nodes, but not on others, the
 * inconsistency in shm FIFO sizes causes an obsure fatal error later in
 * PSM3 intialization. So make the the error more obvious and fail sooner.
 */
int psm3_dsa_init(void)
{
	union psmi_envvar_val env_dsa_wq;
	union psmi_envvar_val env_dsa_ratio;
	union psmi_envvar_val env_dsa_thresh;
	union psmi_envvar_val env_dsa_multi;
	int proc;
	int i;
	char dsa_filename[PATH_MAX];
	int fd;

	psmi_spin_init(&dsa_wq_lock);

	// TBD - PSM3 parameter to enable DSA, maybe use PSM3_KASSIST_MODE with
	// a new value "dsa" vs existing cma-get, cma-put, none
	// right now psm3_shm_create calls psm3_get_kassist_mode, but that call
	// is per endpoint, we want to call just once per process.
	// we could parse PSM3_KASSIST_MODE once in a helper function then use the
	// value here and in psm3_get_kassist_mode much like we do for PSM3_RDMA
	// for now, default PSM3_DSA_WQS to "" and only use DSA if it is specified.
	// sysadmin must setup_dsa.sh -ddsa0 -w1 -md and repeat for dsa8, 16, etc

	if (! psm3_getenv("PSM3_DSA_WQS",
			"List of DSA WQ devices to use, one list per local process:\n"
			"     wq0,wq2:wq4,wq6:,...\n"
			"Each wq should be a unique dedicated workqueue DSA device,\n"
			"     such as /dev/dsa/wq0.0\n"
			"Colon separates the lists for different processes\n"
			"     default is '' in which case DSA is not used\n",
			PSMI_ENVVAR_LEVEL_USER, PSMI_ENVVAR_TYPE_STR,
                        (union psmi_envvar_val)"", &env_dsa_wq)) {
		char *temp = psmi_strdup(PSMI_EP_NONE, env_dsa_wq.e_str);
		char *s;
		char *delim;
		int new_proc = 0;
		proc = 0;

		if (! temp) {
			_HFI_ERROR("Can't alloocate temp string");
			return -1;
		}
		s = temp;
		psmi_assert(*s);
		do {
			new_proc = 0;
			if (! *s)	// trailing ',' or ':' on 2nd or later loop
				break;
			if (proc >= DSA_MAX_PROC) {
				_HFI_ERROR("PSM3_DSA_WQS exceeds %u per node process limit: '%s'",
							DSA_MAX_PROC, env_dsa_wq.e_str);
				psmi_free(temp);
				goto fail;
			}
			delim = strpbrk(s, ",:");
			if (delim)  {
				new_proc = (*delim == ':');
				*delim = '\0';
			}
			if (dsa_num_wqs[proc] > DSA_MAX_QUEUES) {
				_HFI_ERROR("PSM3_DSA_WQS list for process %u exceeds %u per process wqs limit: '%s'",
						proc, DSA_MAX_QUEUES, env_dsa_wq.e_str);
				psmi_free(temp);
				goto fail;
			}
			dsa_wq_filename[proc][dsa_num_wqs[proc]] = psmi_strdup(PSMI_EP_NONE, s);
			dsa_num_wqs[proc]++;
			if (new_proc)
				proc++;
			s = delim+1;
		} while (delim);
		psmi_free(temp);
		// new_proc means trailing :, ignore it
		// otherwise, last we processed counts
		if (!new_proc && proc < DSA_MAX_PROC && dsa_num_wqs[proc])
			proc++;
		dsa_num_proc = proc;
	}
	if (! dsa_num_proc) {
		_HFI_PRDBG("DSA disabled: PSM3_DSA_WQS empty\n");
		return 0;
	}

	psm3_getenv("PSM3_DSA_RATIO",
					"Portion of intra-node copy to do via CPU vs DSA\n"
					"0 - none, 1 - all, 2 - 1/2, 3 - 1/3, etc, default 5 (1/5)\n",
					PSMI_ENVVAR_LEVEL_HIDDEN, PSMI_ENVVAR_TYPE_UINT,
					(union psmi_envvar_val)5, &env_dsa_ratio);
	if (env_dsa_ratio.e_uint == 1) {
		_HFI_PRDBG("DSA disabled: PSM3_DSA_RATIO is 1\n");
		return 0;
	}
	dsa_ratio = env_dsa_ratio.e_uint;

	psm3_getenv("PSM3_DSA_THRESH",
					"DSA is used for shm data copies greater than the threshold\n",
					PSMI_ENVVAR_LEVEL_HIDDEN, PSMI_ENVVAR_TYPE_UINT,
					(union psmi_envvar_val)DSA_THRESH, &env_dsa_thresh);
	dsa_thresh = env_dsa_thresh.e_uint;
	
	// For Intel MPI CPUs are assigned to ranks in NUMA order so
	// for <= 1 process per socket, NUMA and local_rank are often equivalent
	// For Open MPI CPUs are assigned in sequential order so local_rank
	// typically must be used.
	psm3_getenv("PSM3_DSA_MULTI",
					"Is PSM3_DSA_WQS indexed by local rank or by NUMA?  This must be 1 or 2 if more than 1 process per CPU NUMA domain\n"
					"0 - NUMA, 1 - local_rank, 2=auto\n",
					PSMI_ENVVAR_LEVEL_HIDDEN, PSMI_ENVVAR_TYPE_UINT,
					(union psmi_envvar_val)1, &env_dsa_multi);
	if (env_dsa_multi.e_uint == 2) {
		// if there are fewer processes than CPU sockets and we have at
		// least 1 DSA WQ listed per CPU socket, we can use NUMA as the
		// index (assumes processes pinned one per NUMA), otherwise we must
		// use local rank as the index.
		// max_cpu_numa is the largest NUMA ID, hence +1 to compare to counts
		int num_cpu_numa = psm3_get_max_cpu_numa()+1;
		env_dsa_multi.e_uint = (psm3_get_mylocalrank_count() <= num_cpu_numa
								&& num_cpu_numa <= dsa_num_proc)
								? 0 : 1;
		_HFI_DBG("Autoselected PSM3_DSA_MULTI=%u (local ranks=%d num_cpu_numa=%d dsa_num_proc=%d)\n",
						env_dsa_multi.e_uint, psm3_get_mylocalrank_count(),
						num_cpu_numa, dsa_num_proc);
	}
	if (env_dsa_multi.e_uint) {
#if 0
		// ideally we would not need PSM3_DSA_MULTI flag and would
		// always have PSM3_DSA_WQS list WQs by NUMA domain and then
		// we could divy up the WQs among the processes within a given NUMA
		// However, while we can tell how many local processes there are
		// we can't tell our process's relative position among the
		// processes in our NUMA domain, so we can't determine which slice
		// of the WQS for our NUMA to assign to our process
		if (psm3_get_mylocalrank_count() > dsa_num_proc) {
			// compute how many local ranks per NUMA domain
			// round up pernuma to worse case if ranks not a mult of num_proc
			int pernuma = (psm3_get_mylocalrank_count()+dsa_num_proc-1) / dsa_num_proc;
			if (pernuma > dsa_num_wqs[our numa]) {
				_HFI_ERROR("PSM3_DSA_WQS only has %u WQs for NUMA %u, need at least %d",
					dsa_num_wqs[our numa], our numa,  pernuma);
				goto fail;
			}
			start = psm3_get_mylocalrank() * pernuma;
			endp1 = start + pernuma;
		}
#endif
		// we treat PSM3_DSA_WQS as a per process list not per numa
		// user must be careful to pin processes in same NUMA order
		// as the WQS list, but even if get NUMA wrong may not be too
		// bad since at least 1/2 the DSA copies cross NUMA anyway
		if (psm3_get_mylocalrank_count() > dsa_num_proc) {
			_HFI_ERROR("PSM3_DSA_WQS only has %u process lists, need at least %d",
					dsa_num_proc, psm3_get_mylocalrank_count());
			goto fail;
		}
		proc = psm3_get_mylocalrank();
	} else {
		// we assume only 1 process per socket, so our numa picks the DSA WQ
		// look at our NUMA (0, 1, ...) and pick the corresponding DSA WQ
		proc = psm3_get_current_proc_location();
		if (proc < 0) {
			_HFI_ERROR("Unable to get NUMA location of current process");
			goto fail;
		}
		if (proc >= dsa_num_proc) {
			_HFI_ERROR("PSM3_DSA_WQS only has %u process lists, need at least %d",
					dsa_num_proc, proc+1);
			goto fail;
		}
	}
	_HFI_PRDBG("DSA: Local Rank %d of %d Using WQs index %d of %u process lists in PSM3_DSA_WQS\n",
		psm3_get_mylocalrank(), psm3_get_mylocalrank_count(),
		proc, dsa_num_proc);

	// check all the WQs for our socket and open them
	dsa_my_num_wqs = dsa_num_wqs[proc];
	dsa_my_dsa_str_len=0;
	for (i=0; i<dsa_my_num_wqs; i++) {
		// key off having rw access to the DSA WQ to decide if DSA is available
		dsa_wqs[i].wq_filename = dsa_wq_filename[proc][i];
		if (! realpath(dsa_wqs[i].wq_filename, dsa_filename)) {
			_HFI_ERROR("Failed to resolve DSA WQ path %s\n", dsa_wqs[i].wq_filename);
			goto fail;
		}
		fd = open(dsa_filename, O_RDWR);
		if (fd < 0) {
			_HFI_ERROR("Unable to open DSA WQ (%s): %s\n", dsa_filename, strerror(errno));
			goto fail;
		}

		dsa_wqs[i].wq_reg = mmap(NULL, DSA_MMAP_LEN, PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);
		if (dsa_wqs[i].wq_reg == MAP_FAILED) {
			_HFI_ERROR("Unable to mmap DSA WQ (%s): %s\n", dsa_filename, strerror(errno));
			close(fd);
			goto fail;
		}

		close(fd);
		// name + a coma or space
		dsa_my_dsa_str_len += strlen(dsa_wqs[i].wq_filename)+1;
	}
	_HFI_PRDBG("DSA Available\n");
	dsa_available = 1;
	return 0;

fail:
	dsa_free_wqs();
	return -1;
}

/* output DSA information for identify */
void psm3_dsa_identify(void)
{
	if (! dsa_available)
		return;
	if (! psm3_parse_identify())
		return;

	// output the list of DSA WQs assigned to this process
	int i, len = 0, buf_len = dsa_my_dsa_str_len+1;
	char *buf = psmi_malloc(NULL, UNDEFINED, buf_len);
	if (! buf)	// keep KW happy
		return;
	for (i=0; len < buf_len-1 && i<dsa_my_num_wqs; i++) {
		len += snprintf(buf+len, buf_len-len, "%s%s", i?",":" ",
				dsa_wqs[i].wq_filename);
	}
	printf("%s %s DSA:%s\n", psm3_get_mylabel(), psm3_ident_tag, buf);
	psmi_free(buf);
}

static inline void psm3_dsa_pick_wq(void)
{
	int i, sel = 0;
	uint32_t min_use_count = UINT32_MAX;
	// pick the WQ for the current thread
	if (dsa_wq_reg)
		return;	// typical case, already picked one

	// rcvthread, pick last and don't count it
	if (pthread_equal(psm3_rcv_threadid, pthread_self())) {
		sel = dsa_my_num_wqs-1;
		_HFI_PRDBG("rcvthread picked wq %u: %s\n", sel, dsa_wqs[sel].wq_filename);
		goto found;
	}

	// pick 1st with lowest use count
	psmi_spin_lock(&dsa_wq_lock);
	for (i = 0; i < dsa_my_num_wqs; i++) {
		if (dsa_wqs[i].use_count < min_use_count) {
			sel = i;
			min_use_count = dsa_wqs[i].use_count;
		}
	}
	psmi_assert(sel < dsa_my_num_wqs);
	dsa_wqs[sel].use_count++;
	psmi_spin_unlock(&dsa_wq_lock);

	_HFI_PRDBG("picked wq %u: %s\n", sel, dsa_wqs[sel].wq_filename);
found:
	dsa_wq_reg = dsa_wqs[sel].wq_reg;
}


/* after calling psm3_dsa_init was DSA available and successfully initialized */
int psm3_dsa_available(void)
{
	return dsa_available;
}

int psm3_use_dsa(uint32_t msglen)
{
	if (! dsa_available || msglen <= dsa_thresh)
		return 0;
	psm3_dsa_pick_wq();
	return 1;
}

/* cleanup DSA - call once per process */
void psm3_dsa_fini(void)
{
	dsa_free_wqs();
	dsa_available = 0;
	psmi_spin_destroy(&dsa_wq_lock);
}

#if 0
// sample simple use
int main(int argc, char *argv[])
{
	char src[4096];
	char dst[4096];

	memset(src, 0xaa, BLEN);
	if (dsa_init())
		exit(1);
	if (! dsa_available())
		exit(1);
	dsa_memcpy(src, dst, BLEN);
	printf("desc successful\n");
	if (memcmp(src, dst, BLEN))
		printf("memmove failed\n");
	else
		printf("memmove successful\n");
	dsa_fini();
}
#endif
#endif /* PSM_DSA */
