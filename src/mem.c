/*
 * Copyright 2014-2018, Intel Corporation
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the
 *       distribution.
 *
 *     * Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <inttypes.h>

#include <ofi_osd.h>
#include <ofi.h>
#include <rdma/fi_errno.h>
#include <ofi_mem.h>
#include <rdma/fabric.h>


#define CACHE_SIZE	64

uint64_t OFI_RMA_PMEM;
void (*ofi_pmem_commit)(const void *addr, size_t len);


static void pmem_commit_clwb(const void *addr, size_t len)
{
	uintptr_t uptr;

	for (uptr = (uintptr_t) addr & ~(CACHE_SIZE - 1);
	     uptr < (uintptr_t) addr + len; uptr += CACHE_SIZE) {
		ofi_clwb(uptr);
	}
	ofi_sfence();
}

static void pmem_commit_clflushopt(const void *addr, size_t len)
{
	uintptr_t uptr;

	for (uptr = (uintptr_t) addr & ~(CACHE_SIZE - 1);
	     uptr < (uintptr_t) addr + len; uptr += CACHE_SIZE) {
		ofi_clflushopt(uptr);
	}
	ofi_sfence();
}

static void pmem_commit_clflush(const void *addr, size_t len)
{
	uintptr_t uptr;

	for (uptr = (uintptr_t) addr & ~(CACHE_SIZE - 1);
	     uptr < (uintptr_t) addr + len; uptr += CACHE_SIZE) {
		ofi_clflush(uptr);
	}
}

void ofi_pmem_init(void)
{
	if (ofi_cpu_supports(0x7, OFI_CLWB_REG, OFI_CLWB_BIT)) {
		ofi_pmem_commit = pmem_commit_clwb;
	} else if (ofi_cpu_supports(0x7, OFI_CLFLUSHOPT_REG,
				    OFI_CLFLUSHOPT_BIT)) {
		ofi_pmem_commit = pmem_commit_clflushopt;
	} else if (ofi_cpu_supports(0x1, OFI_CLFLUSH_REG, OFI_CLFLUSH_BIT)) {
		ofi_pmem_commit = pmem_commit_clflush;
	}

	if (ofi_pmem_commit)
		OFI_RMA_PMEM = FI_RMA_PMEM;
}
