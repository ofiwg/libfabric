/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_THREAD_ANNOTATIONS_H
#define EFA_THREAD_ANNOTATIONS_H

#include <ofi_lock.h>

/*
 * Clang Thread Safety Analysis for the EFA provider.
 *
 * Active only in the analysis build (OFI_THREAD_SAFETY_ANALYSIS defined and
 * clang with capability support); expands to nothing otherwise.
 */

#if defined(OFI_THREAD_SAFETY_ANALYSIS) && defined(__clang__) && \
	defined(__has_attribute)
#  if __has_attribute(capability)
#    define OFI_TSA_ANNOTATION(x)	__attribute__((x))
#  endif
#endif

#ifndef OFI_TSA_ANNOTATION
#  define OFI_TSA_ANNOTATION(x)
#endif

#define OFI_TSA_CAPABILITY(name)	OFI_TSA_ANNOTATION(capability(name))
#define OFI_TSA_GUARDED_BY(x)		OFI_TSA_ANNOTATION(guarded_by(x))
#define OFI_TSA_PT_GUARDED_BY(x)	OFI_TSA_ANNOTATION(pt_guarded_by(x))
#define OFI_TSA_REQUIRES(...)		OFI_TSA_ANNOTATION(requires_capability(__VA_ARGS__))
#define OFI_TSA_REQUIRES_SHARED(...)	OFI_TSA_ANNOTATION(requires_shared_capability(__VA_ARGS__))
#define OFI_TSA_EXCLUDES(...)		OFI_TSA_ANNOTATION(locks_excluded(__VA_ARGS__))
#define OFI_TSA_ACQUIRE(...)		OFI_TSA_ANNOTATION(acquire_capability(__VA_ARGS__))
#define OFI_TSA_RELEASE(...)		OFI_TSA_ANNOTATION(release_capability(__VA_ARGS__))
#define OFI_TSA_ASSERT_CAPABILITY(x)	OFI_TSA_ANNOTATION(assert_capability(x))
#define OFI_TSA_NO_ANALYSIS		OFI_TSA_ANNOTATION(no_thread_safety_analysis)

/*
 * Lock symbols.
 *
 * Each lock role is represented by a global dummy capability object.
 * Annotations reference the symbol; the wrappers lock the real genlock and
 * bind it to a symbol.  Declare a symbol once in a header and define it once
 * in a .c; use EFA_GENLOCK_LOCK/UNLOCK/HELD to take, drop, and assert it.
 */
#ifdef OFI_THREAD_SAFETY_ANALYSIS

struct OFI_TSA_CAPABILITY("mutex") ofi_tsa_lock_symbol { char dummy; };

#define OFI_TSA_LOCK_SYMBOL_DECLARE(name) \
	extern struct ofi_tsa_lock_symbol name
#define OFI_TSA_LOCK_SYMBOL_DEFINE(name) \
	struct ofi_tsa_lock_symbol name

static inline void
efa_genlock_acquire(struct ofi_genlock *lock, struct ofi_tsa_lock_symbol *sym)
	OFI_TSA_ACQUIRE(*sym) OFI_TSA_NO_ANALYSIS
{
	(void) sym;
	ofi_genlock_lock(lock);
}

static inline void
efa_genlock_release(struct ofi_genlock *lock, struct ofi_tsa_lock_symbol *sym)
	OFI_TSA_RELEASE(*sym) OFI_TSA_NO_ANALYSIS
{
	(void) sym;
	ofi_genlock_unlock(lock);
}

static inline int
efa_genlock_held(struct ofi_genlock *lock, struct ofi_tsa_lock_symbol *sym)
	OFI_TSA_ASSERT_CAPABILITY(*sym) OFI_TSA_NO_ANALYSIS
{
	(void) sym;
	return ofi_genlock_held(lock);
}

#define EFA_GENLOCK_LOCK(lock, sym)	efa_genlock_acquire((lock), &(sym))
#define EFA_GENLOCK_UNLOCK(lock, sym)	efa_genlock_release((lock), &(sym))
#define EFA_GENLOCK_HELD(lock, sym)	efa_genlock_held((lock), &(sym))

#else /* !OFI_THREAD_SAFETY_ANALYSIS */

#define OFI_TSA_LOCK_SYMBOL_DECLARE(name)	struct ofi_tsa_lock_symbol_unused_##name
#define OFI_TSA_LOCK_SYMBOL_DEFINE(name)	struct ofi_tsa_lock_symbol_unused_##name

#define EFA_GENLOCK_LOCK(lock, sym)	ofi_genlock_lock(lock)
#define EFA_GENLOCK_UNLOCK(lock, sym)	ofi_genlock_unlock(lock)
#define EFA_GENLOCK_HELD(lock, sym)	ofi_genlock_held(lock)

#endif /* OFI_THREAD_SAFETY_ANALYSIS */


#endif /* EFA_THREAD_ANNOTATIONS_H */
