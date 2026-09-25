/* SPDX-License-Identifier: GPL-2.0 OR BSD-2-Clause */
/*
 * Copyright 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef _EFA_IO_DEFS_H_
#define _EFA_IO_DEFS_H_

/* Descriptor layouts and field masks, shared with device-side XPU kernels. */
#include <rdma/efa_io_defs.h>

/* Barriers for DMA.

   These barriers are expliclty only for use with user DMA operations. If you
   are looking for barriers to use with cache-coherent multi-threaded
   consitency then look in stdatomic.h. If you need both kinds of synchronicity
   for the same address then use an atomic operation followed by one
   of these barriers.

   When reasoning about these barriers there are two objects:
     - CPU attached address space (the CPU memory could be a range of things:
       cached/uncached/non-temporal CPU DRAM, uncached MMIO space in another
       device, pMEM). Generally speaking the ordering is only relative
       to the local CPU's view of the system. Eg if the local CPU
       is not guaranteed to see a write from another CPU then it is also
       OK for the DMA device to also not see the write after the barrier.
     - A DMA initiator on a bus. For instance a PCI-E device issuing
       MemRd/MemWr TLPs.

   The ordering guarantee is always stated between those two streams. Eg what
   happens if a MemRd TLP is sent in via PCI-E relative to a CPU WRITE to the
   same memory location.

   The providers have a very regular and predictable use of these barriers,
   to make things very clear each narrow use is given a name and the proper
   name should be used in the provider as a form of documentation.
*/

/* Ensure that the device's view of memory matches the CPU's view of memory.
   This should be placed before any MMIO store that could trigger the device
   to begin doing DMA, such as a device doorbell ring.

   eg
    *dma_buf = 1;
    udma_to_device_barrier();
    mmio_write(DO_DMA_REG, dma_buf);
   Must ensure that the device sees the '1'.

   This is required to fence writes created by the libibverbs user. Those
   writes could be to any CPU mapped memory object with any cachability mode.

   NOTE: x86 has historically used a weaker semantic for this barrier, and
   only fenced normal stores to normal memory. libibverbs users using other
   memory types or non-temporal stores are required to use SFENCE in their own
   code prior to calling verbs to start a DMA.
*/
#if defined(__x86_64__)
#define udma_to_device_barrier() asm volatile("" ::: "memory")
#elif defined(__aarch64__)
#define udma_to_device_barrier() asm volatile("dmb oshst" ::: "memory")
#else
#error No architecture specific memory barrier defines found!
#endif

/* Ensure that all ordered stores from the device are observable from the
   CPU. This only makes sense after something that observes an ordered store
   from the device - eg by reading a MMIO register or seeing that CPU memory is
   updated.

   This guarantees that all reads that follow the barrier see the ordered
   stores that preceded the observation.

   For instance, this would be used after testing a valid bit in a memory
   that is a DMA target, to ensure that the following reads see the
   data written before the MemWr TLP that set the valid bit.
*/
#if defined(__x86_64__)
#define udma_from_device_barrier() asm volatile("lfence" ::: "memory")
#elif defined(__aarch64__)
#define udma_from_device_barrier() asm volatile("dmb oshld" ::: "memory")
#else
#error No architecture specific memory barrier defines found!
#endif

/* Order writes to CPU memory so that a DMA device cannot view writes after
   the barrier without also seeing all writes before the barrier. This does
   not guarantee any writes are visible to DMA.

   This would be used in cases where a DMA buffer might have a valid bit and
   data, this barrier is placed after writing the data but before writing the
   valid bit to ensure the DMA device cannot observe a set valid bit with
   unwritten data.

   Compared to udma_to_device_barrier() this barrier is not required to fence
   anything but normal stores to normal malloc memory. Usage should be:

   write_wqe
      udma_to_device_barrier();    // Get user memory ready for DMA
      wqe->addr = ...;
      wqe->flags = ...;
      udma_ordering_write_barrier();  // Guarantee WQE written in order
      wqe->valid = 1;
*/
#define udma_ordering_write_barrier() udma_to_device_barrier()

/* Promptly flush writes to MMIO Write Cominbing memory.
   This should be used after a write to WC memory. This is both a barrier
   and a hint to the CPU to flush any buffers to reduce latency to TLP
   generation.

   This is not required to have any effect on CPU memory.

   If done while holding a lock then the ordering of MMIO writes across CPUs
   must be guaranteed to follow the natural ordering implied by the lock.

   This must also act as a barrier that prevents write combining, eg
     *wc_mem = 1;
     mmio_flush_writes();
     *wc_mem = 2;
   Must always produce two MemWr TLPs, '1' and '2'. Without the barrier
   the CPU is allowed to produce a single TLP '2'.

   Note that there is no order guarantee for writes to WC memory without
   barriers.

   This is intended to be used in conjunction with WC memory to generate large
   PCI-E MemWr TLPs from the CPU.
*/

#if defined(__x86_64__)
#define mmio_flush_writes() asm volatile("sfence" ::: "memory")
#elif defined(__aarch64__)
#define mmio_flush_writes() asm volatile("dsb st" ::: "memory");
#else
#error No architecture specific memory barrier defines found!
#endif

/* Prevent WC writes from being re-ordered relative to other MMIO
   writes. This should be used before a write to WC memory.

   This must act as a barrier to prevent write re-ordering from different
   memory types:
     *mmio_mem = 1;
     mmio_flush_writes();
     *wc_mem = 2;
   Must always produce a TLP '1' followed by '2'.

   This barrier implies udma_to_device_barrier()

   This is intended to be used in conjunction with WC memory to generate large
   PCI-E MemWr TLPs from the CPU.
*/
#define mmio_wc_start() mmio_flush_writes()

#define __bf_shf(x) (__builtin_ffsll(x) - 1)

#define FIELD_GET(_mask, _reg) \
	({ (typeof(_mask)) (((_reg) & (_mask)) >> __bf_shf(_mask)); })
#define FIELD_PREP(_mask, _val) \
	({ ((typeof(_mask)) (_val) << __bf_shf(_mask)) & (_mask); })

#define EFA_GET(ptr, mask) FIELD_GET(mask##_MASK, *(ptr))

#define EFA_SET(ptr, mask, value)                       \
	({                                              \
		typeof(ptr) _ptr = ptr;                 \
		*_ptr = (*_ptr & ~(mask##_MASK)) |      \
			FIELD_PREP(mask##_MASK, value); \
	})

#endif /* _EFA_IO_DEFS_H_ */
