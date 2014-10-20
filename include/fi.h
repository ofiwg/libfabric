/*
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
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

#ifndef _FI_H_
#define _FI_H_

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <byteswap.h>
#include <endian.h>
#include <semaphore.h>
#include <rdma/fabric.h>
#include <rdma/fi_prov.h>
#include <rdma/fi_atomic.h>


#ifdef __cplusplus
extern "C" {
#endif

#define PFX "libfabric: "

#ifdef INCLUDE_VALGRIND
#   include <valgrind/memcheck.h>
#   ifndef VALGRIND_MAKE_MEM_DEFINED
#      warning "Valgrind requested, but VALGRIND_MAKE_MEM_DEFINED undefined"
#   endif
#endif

#ifndef VALGRIND_MAKE_MEM_DEFINED
#   define VALGRIND_MAKE_MEM_DEFINED(addr, len)
#endif

#if __BYTE_ORDER == __LITTLE_ENDIAN
static inline uint64_t htonll(uint64_t x) { return bswap_64(x); }
static inline uint64_t ntohll(uint64_t x) { return bswap_64(x); }
#else
static inline uint64_t htonll(uint64_t x) { return x; }
static inline uint64_t ntohll(uint64_t x) { return x; }
#endif

#define max(a, b) ((a) > (b) ? a : b)
#define min(a, b) ((a) < (b) ? a : b)

static inline int flsll(long long int i)
{
	return i ? 65 - ffsll(htonll(i)) : 0;
}
static inline uint64_t roundup_power_of_two(uint64_t n)
{
	return 1ULL << flsll(n - 1);
}

#define FI_TAG_GENERIC	0xAAAAAAAAAAAAAAAAULL

#if DEFINE_ATOMICS
#define fastlock_t pthread_mutex_t
#define fastlock_init(lock) pthread_mutex_init(lock, NULL)
#define fastlock_destroy(lock) pthread_mutex_destroy(lock)
#define fastlock_acquire(lock) pthread_mutex_lock(lock)
#define fastlock_release(lock) pthread_mutex_unlock(lock)

typedef struct { pthread_mutex_t mut; int val; } atomic_t;
static inline int atomic_inc(atomic_t *atomic)
{
	int v;

	pthread_mutex_lock(&atomic->mut);
	v = ++(atomic->val);
	pthread_mutex_unlock(&atomic->mut);
	return v;
}
static inline int atomic_dec(atomic_t *atomic)
{
	int v;

	pthread_mutex_lock(&atomic->mut);
	v = --(atomic->val);
	pthread_mutex_unlock(&atomic->mut);
	return v;
}
static inline void atomic_init(atomic_t *atomic)
{
	pthread_mutex_init(&atomic->mut, NULL);
	atomic->val = 0;
}
#else
typedef struct {
	sem_t sem;
	volatile int cnt;
} fastlock_t;
static inline void fastlock_init(fastlock_t *lock)
{
	sem_init(&lock->sem, 0, 0);
	lock->cnt = 0;
}
static inline void fastlock_destroy(fastlock_t *lock)
{
	sem_destroy(&lock->sem);
}
static inline void fastlock_acquire(fastlock_t *lock)
{
	if (__sync_add_and_fetch(&lock->cnt, 1) > 1)
		sem_wait(&lock->sem);
}
static inline void fastlock_release(fastlock_t *lock)
{
	if (__sync_sub_and_fetch(&lock->cnt, 1) > 0)
		sem_post(&lock->sem);
}

typedef struct { volatile int val; } atomic_t;
#define atomic_inc(v) (__sync_add_and_fetch(&(v)->val, 1))
#define atomic_dec(v) (__sync_sub_and_fetch(&(v)->val, 1))
#define atomic_init(v) ((v)->val = 0)
#endif /* DEFINE_ATOMICS */

#define atomic_get(v) ((v)->val)
#define atomic_set(v, s) ((v)->val = s)

/* non exported symbols */
int fi_init(void);

int fi_read_file(const char *dir, const char *file, char *buf, size_t size);
int fi_poll_fd(int fd, int timeout);
int fi_wait_cond(pthread_cond_t *cond, pthread_mutex_t *mut, int timeout);

struct fi_info *fi_allocinfo_internal(void);
void fi_freeinfo_internal(struct fi_info *info);

int fi_sockaddr_len(struct sockaddr *addr);
size_t fi_datatype_size(enum fi_datatype datatype);
uint64_t fi_tag_bits(uint64_t mem_tag_format);
uint64_t fi_tag_format(uint64_t tag_bits);

int fi_version_register(uint32_t version, struct fi_provider *provider);

#define RDMA_CONF_DIR  SYSCONFDIR "/" RDMADIR
#define FI_CONF_DIR RDMA_CONF_DIR "/fabric"

#define DEFAULT_ABI "FABRIC_1.0"

/* symbol -> external symbol mappings */
#ifdef HAVE_SYMVER_SUPPORT

#  define symver(name, api, ver) \
        asm(".symver " #name "," #api "@" #ver)
#  define default_symver(name, api) \
        asm(".symver " #name "," #api "@@" DEFAULT_ABI)
#else
#  define symver(name, api, ver)
#  define default_symver(name, api) \
        extern __typeof(name) api __attribute__((alias(#name)))

#endif /* HAVE_SYMVER_SUPPORT */

/* symbol -> external symbol mappings */
#ifdef HAVE_SYMVER_SUPPORT

/* FABRIC_1.0: default symbol set must match linker script */
#define FABRIC_10(SYM, ESYM) asm(".symver " #SYM","#ESYM"@@FABRIC_1.0");

#else
#define FABRIC_10(SYM, ESYM)
#endif

#ifdef __cplusplus
}
#endif

#endif /* _FI_H_ */
