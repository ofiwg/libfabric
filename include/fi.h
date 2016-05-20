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

#include "config.h"

#include <assert.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>

#include <fi_abi.h>
#include <fi_file.h>
#include <fi_lock.h>
#include <fi_atom.h>
#include <fi_mem.h>

#include <rdma/fabric.h>
#include <rdma/fi_prov.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_log.h>

#include <fi_osd.h>


#ifdef __cplusplus
extern "C" {
#endif

/*
 * OS X doesn't have __BYTE_ORDER, Linux usually has BYTE_ORDER but not under
 * all features.h flags
 */
#if !defined(BYTE_ORDER)
#  if defined(__BYTE_ORDER) && \
      defined(__LITTLE_ENDIAN) && \
      defined(__BIG_ENDIAN)
#    define BYTE_ORDER __BYTE_ORDER
#    define LITTLE_ENDIAN __LITTLE_ENDIAN
#    define BIG_ENDIAN __BIG_ENDIAN
#  else
#    error "cannot determine endianness!"
#  endif
#endif

#if BYTE_ORDER == LITTLE_ENDIAN
#ifndef htonll
static inline uint64_t htonll(uint64_t x) { return bswap_64(x); }
#endif
#ifndef ntohll
static inline uint64_t ntohll(uint64_t x) { return bswap_64(x); }
#endif
#else
#ifndef htonll
static inline uint64_t htonll(uint64_t x) { return x; }
#endif
#ifndef ntohll
static inline uint64_t ntohll(uint64_t x) { return x; }
#endif
#endif

#define sizeof_field(type, field) sizeof(((type *)0)->field)

#ifndef MIN
#define MIN(a, b) \
	({ typeof (a) _a = (a); \
		typeof (b) _b = (b); \
		_a < _b ? _a : _b; })
#endif

#ifndef MAX
#define MAX(a, b) \
	({ typeof (a) _a = (a); \
		typeof (b) _b = (b); \
		_a > _b ? _a : _b; })
#endif

/* Restrict to size of struct fi_context */
struct fi_prov_context {
	int disable_logging;
};

struct fi_filter {
	char **names;
	int negated;
};

extern struct fi_filter prov_log_filter;

void fi_create_filter(struct fi_filter *filter, const char *env_name);
void fi_free_filter(struct fi_filter *filter);
int fi_apply_filter(struct fi_filter *filter, const char *name);

void fi_util_init(void);
void fi_util_fini(void);
void fi_log_init(void);
void fi_log_fini(void);
void fi_param_init(void);
void fi_param_fini(void);
void fi_param_undefine(const struct fi_provider *provider);


static inline uint64_t roundup_power_of_two(uint64_t n)
{
	if (!n || !(n & (n - 1)))
		return n;
	n--;
	n |= n >> 1;
	n |= n >> 2;
	n |= n >> 4;
	n |= n >> 8;
	n |= n >> 16;
	n |= n >> 32;
	n++;
	return n;
}

static inline size_t fi_get_aligned_sz(size_t size, size_t alignment)
{
	return ((size % alignment) == 0) ?
		size : ((size / alignment) + 1) * alignment;
}

#define FI_TAG_GENERIC	0xAAAAAAAAAAAAAAAAULL


size_t fi_datatype_size(enum fi_datatype datatype);
uint64_t fi_tag_bits(uint64_t mem_tag_format);
uint64_t fi_tag_format(uint64_t tag_bits);

int fi_send_allowed(uint64_t caps);
int fi_recv_allowed(uint64_t caps);
int fi_rma_initiate_allowed(uint64_t caps);
int fi_rma_target_allowed(uint64_t caps);

uint64_t fi_gettime_ms(void);


#define FI_LOG(prov, level, subsystem, ...)				\
	do {								\
		if (fi_log_enabled(prov, level, subsystem))		\
			fi_log(prov, level, subsystem,			\
				__func__, __LINE__, __VA_ARGS__);	\
	} while (0)

#define FI_WARN(prov, subsystem, ...)					\
	FI_LOG(prov, FI_LOG_WARN, subsystem, __VA_ARGS__)

#define FI_TRACE(prov, subsystem, ...)					\
	FI_LOG(prov, FI_LOG_TRACE, subsystem, __VA_ARGS__)

#define FI_INFO(prov, subsystem, ...)					\
	FI_LOG(prov, FI_LOG_INFO, subsystem, __VA_ARGS__)

#if defined(ENABLE_DEBUG) && ENABLE_DEBUG > 0
#define FI_DBG(prov, subsystem, ...)					\
	FI_LOG(prov, FI_LOG_DEBUG, subsystem, __VA_ARGS__)
#else
#define FI_DBG(prov_name, subsystem, ...)				\
	do {} while (0)
#endif

#ifdef __cplusplus
}
#endif

#endif /* _FI_H_ */
