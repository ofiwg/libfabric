#ifndef _FI_PROV_BGQ_SPI_H_
#define _FI_PROV_BGQ_SPI_H_

/*
 * Certain BGQ SPI files expect the '__LINUX__' macro to be defined to '0'
 * or '1' instead of simply checking if the macro is defined or not.
 * Specifically, the following file needs hacking, although there are probably
 * others.
 *
 *   spi/include/mu/Addressing.h
 */
#ifndef __LINUX__
#define __LINUX__ 0
#endif

/*
 * The bgq system software, specifically 'hwi/include/common/compiler_support.h',
 * will define the __INLINE__ macro if it is not already defined to the following:
 *
 *   #define __INLINE__ extern inline __attribute__((always_inline))
 *
 * This is the non-portable "gnu 89" style which easily results in undefined
 * symbols or multiple defined symbols when used by software coded to a more
 * recent C standard.
 *
 * As a workaround the __INLINE__ macro will be defined to the more appropriate
 * 'static inline' style only for the bgq system software includes and then
 * undefined at the end of this file. This seems to fix the problem without
 * requiring any changes to the installed bgq system software files.
 */
#ifdef __INLINE__
#error __INLINE__ already defined!
#else
#define __INLINE__ static inline
#endif

#include <spi/include/kernel/location.h>
#include <spi/include/kernel/memory.h>
#include <spi/include/kernel/MU.h>
#include <spi/include/l1p/flush.h>

/*
 * Avoid the pervasive "function declared static but never defined" warning for
 * unused kernel spi functions, defined in 'spi/include/kernel/MU.h' by
 * providing an implementation.
 */
int32_t Kernel_GetNDExpectedTokens(uint32_t a, uint32_t b, uint32_t *c) { assert(0); return -1; }
int32_t Kernel_GetNDExpectedTokensDCR(uint32_t a, uint64_t *b) { assert(0); return -1; }

/* Work around buggy SPI code when -DNDEBUG is specified */
#ifdef NDEBUG
#undef NDEBUG
#include <spi/include/l2/atomic.h>
#define NDEBUG
#else
#include <spi/include/l2/atomic.h>
#endif

#include <spi/include/l2/lock.h>
#include <spi/include/mu/Addressing.h>
#include <spi/include/mu/Addressing_inlines.h>
#include <spi/include/mu/Descriptor_inlines.h>
#include <spi/include/mu/GIBarrier.h>
#include <spi/include/mu/InjFifo.h>
#include <spi/include/mu/RecFifo.h>

#undef __INLINE__

#ifndef MIN
#define MIN(a,b) (b^((a^b)&-(a<b)))
#endif
#ifndef MIN3
#define MIN3(a,b,c) (MIN(MIN(a,b),c))
#endif
#ifndef MIN4
#define MIN4(a,b,c,d) (MIN(MIN(a,b),MIN(c,d)))
#endif


struct fi_bgq_spi_injfifo {
	MUSPI_InjFifo_t	*muspi_injfifo;
	uint64_t	*sw_freeSpace;
	MUHWI_InjFifo_t	*hw_injfifo;	/* See: MUSPI_getHwFreeSpace */
	uint64_t	*sw_tailva;	/* See: MUSPI_getTailVa */

	size_t		immediate_payload_sizeof;
	uintptr_t	immediate_payload_base_vaddr;
	uint64_t	immediate_payload_base_paddr;

	uint64_t	node_scoped_fifo_id;
	void		*memory;
	void		*immediate_payload_memory;

	uintptr_t	va_start;
} __attribute((aligned(L2_CACHE_LINE_SIZE)));


int fi_bgq_spi_injfifo_init (struct fi_bgq_spi_injfifo *f,
		MUSPI_InjFifoSubGroup_t *injfifo_subgroup,
		unsigned num_fifos_to_allocate,
		const size_t injfifo_size,
		const unsigned immediate_payload_sizeof,
		const unsigned is_remote_get,
		const unsigned is_top_down);


void fi_bgq_spi_injfifo_clone (struct fi_bgq_spi_injfifo *dst, struct fi_bgq_spi_injfifo *src);

int fi_bgq_spi_injfifo_fini (struct fi_bgq_spi_injfifo *f);



static inline
MUHWI_Descriptor_t * fi_bgq_spi_injfifo_tail_wait (struct fi_bgq_spi_injfifo *f) {

	if (0 == *(f->sw_freeSpace)) {	/* unlikely */
		do {
			/* mmio read from hardware to update shadow state */
			*(f->sw_freeSpace) = f->hw_injfifo->freeSpace;
		} while (0 == *(f->sw_freeSpace));
	}

	return (MUHWI_Descriptor_t *) *f->sw_tailva;	/* updated via MUSPI_InjFifoAdvanceDesc */
}


static inline
void * fi_bgq_spi_injfifo_immediate_payload (struct fi_bgq_spi_injfifo *f,
		MUHWI_Descriptor_t *desc, uint64_t *paddr) {

	assert(f);
	assert(f->immediate_payload_base_vaddr != 0);
	assert(f->immediate_payload_sizeof != 0);
	assert(f->va_start != 0);

	const uint64_t offset =
		(((uintptr_t)desc - f->va_start) >> BGQ_MU_DESCRIPTOR_SIZE_IN_POWER_OF_2) *
		f->immediate_payload_sizeof;

	*paddr = f->immediate_payload_base_paddr + offset;

	return (void*)(f->immediate_payload_base_vaddr + offset);
}

#endif /* _FI_PROV_BGQ_SPI_H_ */
